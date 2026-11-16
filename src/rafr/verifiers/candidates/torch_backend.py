"""The only module in the package that touches ``torch`` / ``transformers``.

Nothing here is imported at package import time: the candidates build their backend on the
first ``score()`` call (:meth:`BaseVerifier.build_backend`), so the whole test suite runs on
a machine with no GPU dependencies installed and no weights on disk.

Determinism (score_spec.md). The same ``(model_version, text)`` must give the same
score bit-for-bit across calls and processes, so this module:

* runs ``model.eval()`` under ``torch.inference_mode()`` -- no dropout, no autograd;
* never calls ``generate`` and never samples: one forward pass, logits only;
* asks for eager attention, because the fused SDPA / flash kernels pick their reduction
  order at runtime;
* turns off TF32 and cuDNN autotuning, and asks torch for deterministic algorithms;
* pads every batch to a fixed sequence length and a fixed batch size, so a float reduction
  order cannot depend on how long the neighbouring texts were or on how full the last batch
  happened to be. Both cost throughput and neither can change a score; set
  ``RAFR_VERIFIER_DYNAMIC_PADDING=1`` to trade the guarantee for speed (the choice is
  recorded in the verifier card).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from rafr.verifiers.candidates.base import UNPINNED, Truncation, VerifierConfigError
from rafr.verifiers.candidates.guard_lm import CHAT, RAW, LabelGroup

#: cuBLAS needs this *before* CUDA is initialised for deterministic GEMM workspaces;
#: the RunPod driver (not released) exports it too, this is the belt-and-braces copy.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

DEVICE_ENV = "RAFR_VERIFIER_DEVICE"

#: Prompt lengths are rounded up to a multiple of this so the fixed shape is not recomputed
#: for every candidate by hand.
PAD_MULTIPLE = 64


def pad_batch(items: list[object], size: int) -> tuple[list[object], int]:
    """Repeat the last item until the batch is exactly ``size`` long.

    Returns ``(padded, n_real)``. The filler rows are computed and thrown away; they exist
    only so that the tensor handed to the model has the same shape for every batch, which is
    what keeps a float reduction order from depending on how full the last batch was. Costs
    at most one batch of wasted compute per ``score()`` call.
    """
    n = len(items)
    if n == 0 or n >= size:
        return items, n
    return items + [items[-1]] * (size - n), n


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only on a GPU box
        raise VerifierConfigError(
            "torch is not installed. Install the GPU extra: pip install -e '.[gpu]'"
        ) from exc
    return torch


def _transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover
        raise VerifierConfigError(
            "transformers is not installed. Install the GPU extra: pip install -e '.[gpu]'"
        ) from exc
    return transformers


def configure_determinism() -> None:
    """Global torch switches. Idempotent; called once per backend construction."""
    torch = _torch()
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # warn_only: a few ops still have no deterministic CUDA kernel. Asking for eager
    # attention removes the one that matters here; anything else that warns goes into the
    # run log rather than killing a paid GPU hour.
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device() -> str:
    explicit = os.environ.get(DEVICE_ENV, "").strip()
    if explicit:
        return explicit
    torch = _torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(name: str) -> Any:
    torch = _torch()
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:
        raise VerifierConfigError(f"unknown torch dtype {name!r}") from exc
    if not isinstance(dtype, torch.dtype):
        raise VerifierConfigError(f"{name!r} is not a torch dtype")
    return dtype


def _commit_hash(obj: Any, fallback: str | None) -> str:
    """What was actually loaded. transformers stashes the resolved commit on the config."""
    sha = getattr(getattr(obj, "config", None), "_commit_hash", None) or getattr(
        obj, "_commit_hash", None
    )
    return str(sha or fallback or UNPINNED)


def _round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def _from_pretrained(loader: Any, repo_id: str, revision: str | None, dtype: Any) -> Any:
    """``from_pretrained`` with the weight dtype, across the ``torch_dtype`` -> ``dtype``
    rename in transformers 4.56. Guessing wrong here would only show up on the rented pod."""
    common = {"revision": revision, "attn_implementation": "eager"}
    try:
        return loader.from_pretrained(repo_id, dtype=dtype, **common)
    except TypeError:
        return loader.from_pretrained(repo_id, torch_dtype=dtype, **common)


# ---------------------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------------------


class TorchEncoderBackend:
    """A sequence-classification checkpoint. Rows are the raw class logits, in class order."""

    def __init__(
        self,
        *,
        repo_id: str,
        revision: str | None,
        truncation: Truncation,
        batch_size: int,
        dtype: str = "float32",
    ) -> None:
        configure_determinism()
        tf = _transformers()
        self.repo_id = repo_id
        self.truncation = truncation
        self.batch_size = batch_size
        self.device = resolve_device()

        self.tokenizer = tf.AutoTokenizer.from_pretrained(repo_id, revision=revision)
        self.tokenizer.truncation_side = truncation.side
        self.model = _from_pretrained(
            tf.AutoModelForSequenceClassification, repo_id, revision, resolve_dtype(dtype)
        )
        self.model.eval()
        self.model.to(self.device)

        self.revision = _commit_hash(self.model, revision)
        self.id2label = {int(i): str(name) for i, name in self.model.config.id2label.items()}

    def raw_rows(self, items: Sequence[object]) -> list[list[float]]:
        torch = _torch()
        padded, n_real = (
            pad_batch(list(items), self.batch_size)
            if self.truncation.fixed_batch_shape
            else (list(items), len(items))
        )
        texts = [str(t) for t in padded]
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.truncation.max_length,
            padding="max_length" if self.truncation.pad_to_max_length else "longest",
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**batch).logits
        rows = [[float(x) for x in row] for row in logits.float().cpu().tolist()]
        return rows[:n_real]


# ---------------------------------------------------------------------------------------
# guard LM
# ---------------------------------------------------------------------------------------


class TorchGuardBackend:
    """A causal LM read as a classifier: logits of the label tokens, one forward pass.

    Left padding, because the score is read at the last position and a right-padded row
    would put a pad token there. ``position_ids`` are rebuilt from the attention mask so the
    left padding does not shift the positions of the real tokens.
    """

    def __init__(
        self,
        *,
        repo_id: str,
        revision: str | None,
        labels: Sequence[LabelGroup],
        label_prefix: str,
        prompt_style: str,
        truncation: Truncation,
        batch_size: int,
        template_fn: Any = None,
        chat_template_kwargs: dict[str, object] | None = None,
        dtype: str = "bfloat16",
    ) -> None:
        configure_determinism()
        tf = _transformers()
        self.repo_id = repo_id
        self.labels = tuple(labels)
        self.label_prefix = label_prefix
        self.prompt_style = prompt_style
        self.truncation = truncation
        self.batch_size = batch_size
        # How the verifier turns a (already truncated) text into a prompt. Needed only to
        # measure the longest prompt this candidate can produce; see pad_length().
        self.template_fn = template_fn or (lambda text: text)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.device = resolve_device()

        self.tokenizer = tf.AutoTokenizer.from_pretrained(repo_id, revision=revision)
        # Left, because the score is read at the last position: right padding would put a
        # pad token there. Prompts are never truncated by the tokenizer (see raw_rows), so
        # truncation_side is not set here -- cutting a prompt from the right would remove
        # the label position itself.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = _from_pretrained(
            tf.AutoModelForCausalLM, repo_id, revision, resolve_dtype(dtype)
        )
        self.model.eval()
        self.model.to(self.device)

        self.revision = _commit_hash(self.model, revision)
        self.label_token_ids = self._resolve_label_tokens()
        self.group_sizes = tuple(len(ids) for ids in self.label_token_ids)
        self._flat_token_ids = [i for ids in self.label_token_ids for i in ids]
        self._pad_length: int | None = None

    # -- label tokens -------------------------------------------------------------------

    def _first_token_id(self, surface: str) -> int | None:
        ids = self.tokenizer.encode(surface, add_special_tokens=False)
        return int(ids[0]) if ids else None

    def _resolve_label_tokens(self) -> list[list[int]]:
        """One list of token ids per label, taken from the FIRST token of each spelling.

        The first token is what the model actually emits at the scored position; taking it
        (rather than requiring the whole word to be one token) is what makes this work for
        tokenizers that split ``"unsafe"`` into pieces. Groups must stay disjoint, otherwise
        two verdicts would be indistinguishable at that position and the score would be
        meaningless -- so that raises instead of guessing.
        """
        resolved: list[list[int]] = []
        for group in self.labels:
            ids: list[int] = []
            for surface in group.surface_forms:
                tid = self._first_token_id(surface)
                if tid is not None and tid not in ids:
                    ids.append(tid)
            if not ids:
                raise VerifierConfigError(
                    f"{self.repo_id}: none of the spellings {group.surface_forms} of label "
                    f"{group.name!r} tokenises to anything"
                )
            resolved.append(ids)

        seen: dict[int, str] = {}
        for group, ids in zip(self.labels, resolved, strict=True):
            for tid in ids:
                if tid in seen and seen[tid] != group.name:
                    raise VerifierConfigError(
                        f"{self.repo_id}: labels {seen[tid]!r} and {group.name!r} share the "
                        f"first token id {tid} ({self.tokenizer.decode([tid])!r}); the label "
                        "prefix must be extended until they differ"
                    )
                seen[tid] = group.name
        return resolved

    # -- prompts ------------------------------------------------------------------------

    def truncate_text(self, text: str) -> str:
        """Cut the text, never the template around it. Short texts pass through untouched."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= self.truncation.max_length:
            return text
        kept = (
            ids[: self.truncation.max_length]
            if self.truncation.side == "right"
            else ids[-self.truncation.max_length :]
        )
        return str(self.tokenizer.decode(kept, skip_special_tokens=True))

    def render(self, prompt: object, *, include_prefix: bool = True) -> str:
        """Finished prompt text, ready to tokenise.

        ``include_prefix=False`` leaves off the teacher-forced label prefix; that is what
        ``greedy_head`` decodes from, so the self-check sees whether the model produces the
        prefix on its own instead of being handed it.
        """
        if self.prompt_style == RAW:
            if not isinstance(prompt, str):
                raise VerifierConfigError(f"{self.repo_id}: raw prompt style needs a string")
            body = prompt
        elif self.prompt_style == CHAT:
            body = str(
                self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
            )
        else:  # pragma: no cover - refused by GuardLMVerifier
            raise VerifierConfigError(f"unknown prompt style {self.prompt_style!r}")
        return body + self.label_prefix if include_prefix else body

    def pad_length(self) -> int:
        """Fixed prompt length: the longest prompt this candidate can produce, rounded up.

        Measured once, from a text that is already at the truncation budget and then run
        through the verifier's own templating (``template_fn``), so the policy and the
        instructions around the text are counted too. No real prompt can exceed it, and the
        padded shape therefore never depends on the batch.
        """
        if self._pad_length is None:
            filler = self.truncate_text(" ".join(["word"] * self.truncation.max_length))
            probe = self.render(self.template_fn(filler))
            n = len(self.tokenizer.encode(probe))
            self._pad_length = _round_up(n + PAD_MULTIPLE, PAD_MULTIPLE)
        return self._pad_length

    # -- forward ------------------------------------------------------------------------

    def raw_rows(self, items: Sequence[object]) -> list[list[float]]:
        torch = _torch()
        padded, n_real = (
            pad_batch(list(items), self.batch_size)
            if self.truncation.fixed_batch_shape
            else (list(items), len(items))
        )
        prompts = [self.render(p) for p in padded]
        if self.truncation.pad_to_max_length:
            limit = self.pad_length()
            longest = max(len(self.tokenizer.encode(p)) for p in prompts)
            if longest > limit:
                # Truncating here would cut the tail, i.e. the label position itself, and
                # the scores would be silently meaningless. Refuse instead.
                raise VerifierConfigError(
                    f"{self.repo_id}: a prompt is {longest} tokens, above the measured "
                    f"maximum of {limit}. The text budget or the template changed after "
                    "pad_length() was measured."
                )
            batch = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=False,
                max_length=limit,
            )
        else:
            batch = self.tokenizer(prompts, return_tensors="pt", padding="longest")
        batch = batch.to(self.device)
        attention_mask = batch["attention_mask"]
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        with torch.inference_mode():
            logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits
        last = logits[:, -1, :].float()
        selected = last[:, self._flat_token_ids]
        rows = [[float(x) for x in row] for row in selected.cpu().tolist()]
        return rows[:n_real]

    # -- self-check ---------------------------------------------------------------------

    def greedy_head(self, prompt: object, max_new_tokens: int = 16) -> str:
        """Greedy continuation of one prompt, for ``batch_score --self-check`` only.

        Decoded *without* the label prefix, so the output shows what the model emits on its
        own: that is the check that ``label_prefix`` and the label spellings in the registry
        match the weights, which no offline test can make.

        This is the only place the package generates tokens, it is never on the scoring
        path, and it is greedy (``do_sample=False``), so it cannot introduce randomness into
        a run.
        """
        torch = _torch()
        body = self.render(prompt, include_prefix=False)
        ids = self.tokenizer(body, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return str(self.tokenizer.decode(out[0][ids["input_ids"].shape[-1] :]))
