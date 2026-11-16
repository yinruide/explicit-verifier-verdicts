"""Guard-LM candidates (II.4.1: "instruction / guard LM" half of the heterogeneous pair).

A guard LM is a next-token predictor, not a classifier, so the score is read off the
*logits* of the label tokens. Nothing is sampled and nothing is generated:

1. the prompt is built (chat template, or a literal template shipped in this file),
2. ``label_prefix`` -- the fixed text the model always emits before the label -- is appended
   to the prompt and teacher-forced, so the very next token position is the label,
3. one forward pass gives the logits at that position,
4. the logits of the label tokens are folded per label (``logsumexp`` over the spellings of
   the same label), softmaxed across labels, and P(unsafe) is the score.

Step 2 is what makes a model whose output starts with a fixed prefix (``"Safety: "``,
``"<score>"``) readable without generating anything. ``label_prefix`` and the spellings are
per-model data in :mod:`rafr.verifiers.candidates.registry`; the backend refuses to run if a
spelling does not resolve to exactly one token, so a wrong guess fails loudly instead of
scoring the wrong thing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rafr.llm.client import fill_slots
from rafr.verifiers.candidates.base import (
    BaseVerifier,
    Truncation,
    VerifierConfigError,
    logsumexp,
    softmax,
)

SAFE = "safe"
UNSAFE = "unsafe"
CONTROVERSIAL = "controversial"

#: How a third "controversial" verdict folds into the binary score (Qwen3Guard only).
CONTROVERSIAL_AS_UNSAFE = "unsafe"  # P(unsafe) + P(controversial)
CONTROVERSIAL_AS_SAFE = "safe"  # P(unsafe) alone; controversial mass stays on the safe side
CONTROVERSIAL_EXCLUDE = "exclude"  # renormalise over {safe, unsafe} only
CONTROVERSIAL_POLICIES = (CONTROVERSIAL_AS_UNSAFE, CONTROVERSIAL_AS_SAFE, CONTROVERSIAL_EXCLUDE)
CONTROVERSIAL_HINT = (
    f"{CONTROVERSIAL_AS_UNSAFE!r} (P(unsafe)+P(controversial)), "
    f"{CONTROVERSIAL_AS_SAFE!r} (P(unsafe) alone) or "
    f"{CONTROVERSIAL_EXCLUDE!r} (renormalise over safe+unsafe)"
)

CHAT = "chat"
RAW = "raw"
PROMPT_STYLES = (CHAT, RAW)


@dataclass(frozen=True)
class LabelGroup:
    """One verdict and every token spelling that counts as that verdict."""

    name: str
    surface_forms: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.surface_forms:
            raise VerifierConfigError(f"label {self.name!r} has no surface forms")


@runtime_checkable
class GuardBackend(Protocol):
    """A loaded causal LM plus its tokenizer.

    ``group_sizes`` says how many token ids each label group resolved to, in label order;
    ``raw_rows`` returns one flat row per prompt whose entries line up with that layout.
    ``truncate_text`` cuts the *text* before it is templated, so truncation never eats the
    prompt around it.
    """

    revision: str
    group_sizes: tuple[int, ...]

    def truncate_text(self, text: str) -> str: ...

    def raw_rows(self, prompts: Sequence[object]) -> list[list[float]]: ...


class GuardLMVerifier(BaseVerifier):
    """score_spec.md-conformant wrapper around a safety guard LM."""

    def __init__(
        self,
        *,
        verifier_id: str,
        repo_id: str,
        revision: str | None,
        labels: Sequence[LabelGroup],
        label_prefix: str,
        prompt_style: str,
        truncation: Truncation,
        raw_template: str | None = None,
        guideline: str | None = None,
        chat_template_kwargs: dict[str, object] | None = None,
        controversial_policy: str | None = None,
        dtype: str = "bfloat16",
        **kwargs: object,
    ) -> None:
        if prompt_style not in PROMPT_STYLES:
            raise VerifierConfigError(f"prompt_style must be one of {PROMPT_STYLES}")
        if prompt_style == RAW and not raw_template:
            raise VerifierConfigError(f"{verifier_id}: prompt_style='raw' needs a raw_template")
        super().__init__(
            verifier_id=verifier_id,
            repo_id=repo_id,
            revision=revision,
            truncation=truncation,
            **kwargs,  # type: ignore[arg-type]
        )
        self.labels = tuple(labels)
        self.label_prefix = label_prefix
        self.prompt_style = prompt_style
        self.raw_template = raw_template
        self.guideline = guideline
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.dtype = dtype

        names = [lab.name for lab in self.labels]
        if len(set(names)) != len(names):
            raise VerifierConfigError(f"{verifier_id}: duplicate label names {names}")
        for required in (SAFE, UNSAFE):
            if required not in names:
                raise VerifierConfigError(f"{verifier_id}: no {required!r} label in {names}")
        self._index = {name: i for i, name in enumerate(names)}

        self.controversial_policy = self._check_controversial(controversial_policy, names)
        if self.guideline is not None and not self.guideline.strip():
            raise VerifierConfigError(f"{verifier_id}: guideline is set but empty")

    def _check_controversial(self, policy: str | None, names: Sequence[str]) -> str | None:
        if CONTROVERSIAL not in names:
            if policy is not None:
                raise VerifierConfigError(
                    f"{self.verifier_id}: controversial_policy is set but the model has no "
                    f"{CONTROVERSIAL!r} label"
                )
            return None
        if policy is None:
            raise VerifierConfigError(
                f"{self.verifier_id}: this model emits a third verdict {CONTROVERSIAL!r} and "
                "how it maps onto the binary score is a design decision, not a default. "
                f"Set controversial_policy to one of {CONTROVERSIAL_HINT}."
            )
        if policy not in CONTROVERSIAL_POLICIES:
            raise VerifierConfigError(
                f"{self.verifier_id}: controversial_policy must be one of "
                f"{CONTROVERSIAL_POLICIES}, got {policy!r}"
            )
        return policy

    # -- card text ----------------------------------------------------------------------

    @property
    def score_semantics(self) -> str:
        spellings = "; ".join(
            f"{lab.name} = {', '.join(repr(s) for s in lab.surface_forms)}" for lab in self.labels
        )
        prefix = repr(self.label_prefix) if self.label_prefix else "none (label is the first token)"
        return (
            f"logits at the first generated position after the teacher-forced prefix {prefix}; "
            f"label spellings: {spellings}. Nothing is sampled and no token is generated: one "
            "forward pass, logits only."
        )

    @property
    def normalization(self) -> str:
        base = (
            "logsumexp over the token ids spelling each label, softmax across labels, "
            "score = P(unsafe)"
        )
        if self.controversial_policy is None:
            return base
        return f"{base}; controversial_policy={self.controversial_policy!r}"

    # -- prompts (pure) -----------------------------------------------------------------

    def build_prompt(self, text: str) -> str | list[dict[str, str]]:
        """The model input for one text, *without* the label prefix.

        ``raw`` returns a finished string; ``chat`` returns the conversation the backend
        hands to ``tokenizer.apply_chat_template(..., add_generation_prompt=True)``.
        """
        if self.prompt_style == RAW:
            if self.raw_template is None:  # pragma: no cover - refused in __init__
                raise VerifierConfigError(f"{self.verifier_id}: raw_template missing")
            values = {"text": text}
            if self.guideline is not None:
                values["guideline"] = self.guideline
            return fill_slots(self.raw_template, values)
        return [{"role": "user", "content": text}]

    def prepare(self, texts: Sequence[str]) -> list[object]:
        """Truncate each text (tokenizer-level, backend) and template it (pure, here)."""
        return [self.build_prompt(self.backend.truncate_text(t)) for t in texts]

    # -- backend ------------------------------------------------------------------------

    def build_backend(self) -> GuardBackend:
        from rafr.verifiers.candidates.torch_backend import TorchGuardBackend

        return TorchGuardBackend(
            repo_id=self.repo_id,
            revision=None if not self.is_pinned else self.revision,
            labels=self.labels,
            label_prefix=self.label_prefix,
            prompt_style=self.prompt_style,
            truncation=self.truncation,
            batch_size=self.inference_batch_size,
            # so the backend can measure the longest prompt this candidate can produce,
            # template included, rather than guessing from the bare text
            template_fn=self.build_prompt,
            chat_template_kwargs=self.chat_template_kwargs,
            dtype=self.dtype,
        )

    # -- pure normalisation -------------------------------------------------------------

    def label_probabilities(self, row: Sequence[float]) -> dict[str, float]:
        """Split a flat logit row by the backend's group layout and softmax across labels."""
        sizes = tuple(self.backend.group_sizes)
        if len(sizes) != len(self.labels):
            raise VerifierConfigError(
                f"{self.verifier_id}: backend reports {len(sizes)} label groups, "
                f"the verifier declares {len(self.labels)}"
            )
        if len(row) != sum(sizes):
            raise VerifierConfigError(
                f"{self.verifier_id}: logit row of {len(row)} does not match the group layout "
                f"{sizes}"
            )
        folded: list[float] = []
        at = 0
        for size in sizes:
            folded.append(logsumexp([float(x) for x in row[at : at + size]]))
            at += size
        probs = softmax(folded)
        return {lab.name: probs[i] for i, lab in enumerate(self.labels)}

    def normalise_row(self, row: Sequence[float]) -> float:
        probs = self.label_probabilities(row)
        p_unsafe = probs[UNSAFE]
        if self.controversial_policy is None:
            return p_unsafe
        p_controversial = probs[CONTROVERSIAL]
        if self.controversial_policy == CONTROVERSIAL_AS_UNSAFE:
            return p_unsafe + p_controversial
        if self.controversial_policy == CONTROVERSIAL_AS_SAFE:
            return p_unsafe
        denom = p_unsafe + probs[SAFE]
        if denom <= 0.0:
            raise VerifierConfigError(
                f"{self.verifier_id}: safe and unsafe both carry zero mass, cannot renormalise"
            )
        return p_unsafe / denom
