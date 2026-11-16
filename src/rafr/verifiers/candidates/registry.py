"""The V1 / V2 candidate pool (II.4.1, II.4.2; final selection is the authors', App. A.1).

Two families, as II.4.1 requires ("encoder classifier + instruction / guard LM"):

===========================  ===========================================================
``toxic-bert``               unitary/toxic-bert (Detoxify), 6-way multi-label BCE head
``roberta-toxicity``         s-nlp/roberta_toxicity_classifier, 2-way softmax head
``llama-guard-3-8b/-1b``     meta-llama/Llama-Guard-3-{8B,1B}
``qwen3guard-gen-{0.6b,4b,8b}``  Qwen/Qwen3Guard-Gen-*
``shieldgemma-{2b,9b}``      google/shieldgemma-{2b,9b}
``granite-guardian-3.3-8b``  ibm-granite/granite-guardian-3.3-8b
===========================  ===========================================================

No OpenAI and no Anthropic model may appear here: the controller is GPT-5.6 and E is Claude
Opus 5, and II.9.1 keeps the verifiers off both sources.

Every repo name, license and gate status below was checked against the Hub API on
:data:`VERIFIED_ON`. ``hub_sha_at_check`` is the repo head at that moment and is provenance
only -- the revision a run actually uses is the one ``env/download_models.sh`` pins into the
download manifest and this module reads back (see :func:`pinned_revision`).

Three knobs here are design decisions that move the scores, so they are ``None`` and every
candidate that needs one refuses to be built until the authors fill it in. They are listed in the
module constants below with the options and what each one implies.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rafr.verifiers.candidates.base import (
    Truncation,
    VerifierConfigError,
    dynamic_padding_requested,
)
from rafr.verifiers.candidates.encoder import (
    MULTILABEL_SIGMOID,
    SINGLE_LABEL_SOFTMAX,
    EncoderToxicityVerifier,
)
from rafr.verifiers.candidates.guard_lm import (
    CHAT,
    CONTROVERSIAL,
    RAW,
    SAFE,
    UNSAFE,
    GuardLMVerifier,
    LabelGroup,
)

#: Date the repo names / licenses / gate status in this file were checked against
#: ``https://huggingface.co/api/models/<repo>``.
VERIFIED_ON = "2026-09-19"

ENCODER = "encoder"
GUARD_LM = "guard_lm"

#: Text budget shared by every candidate: the encoders cannot see past 512 positions, and
#: giving the guard LMs a larger budget would make the two families read different amounts
#: of the same sentence. ParaDetox items are single sentences, so nothing is expected to be
#: cut; the policy is recorded in the verifier card regardless (App. A.2).
TEXT_MAX_TOKENS = 512


# ---------------------------------------------------------------------------------------
# TODO(decision) -- the authors. Each one changes the scores, so none of them gets a default.
# ---------------------------------------------------------------------------------------

#: Qwen3Guard emits a third verdict, "Controversial", next to Safe / Unsafe. Options:
#:   "unsafe"  -> score = P(Unsafe) + P(Controversial)   (strict; raises the toxic side)
#:   "safe"    -> score = P(Unsafe)                      (lenient; controversial mass is
#:                                                        left on the safe side)
#:   "exclude" -> score = P(Unsafe) / (P(Safe) + P(Unsafe))  (renormalise, ignore the third)
#: Only relevant if a Qwen3Guard model is selected as V1 or V2.
QWEN3GUARD_CONTROVERSIAL_POLICY: str | None = None  # TODO(decision)

#: ShieldGemma scores a text against a safety policy that the caller supplies. The policies
#: actually used are those of the step 2 probe (decisions.md D-2026-09-24, section 3), built by
#: ``rafr.verifiers.candidates.probe`` through the tokenizer's own chat template (its
#: ``guideline`` argument), not through this wrapper:
#:
#: * native -- the model card's Harassment and Hate Speech guidelines (use case 1,
#:   prompt-only), one call each, score = the larger P(Yes), both sub-scores kept
#:   (``configs/prompts/verifier_sg_harassment.txt``, ``verifier_sg_hate_speech.txt``);
#: * task -- the model card's policy format with its content word replaced by "toxic", the
#:   word E's rubric-free prompt uses for what is judged: ``"No Toxicity": The prompt shall not
#:   contain or seek generation of toxic content.`` (``configs/prompts/verifier_task_def.txt``,
#:   key ``shieldgemma_policy``; decisions.md D-2026-09-24b).
#:
#: Their prompt hashes (``probe.prompt_hash``), locked 2026-09-25 after the authors confirmed the
#: rendered prompts (stop point A; ``configs/prompts/verifier_probe_prompts.lock.json``, and in
#: every row of the probe's scores table):
#:
#: * ``sg_2b.native`` 599eaafb1f9340c30e1d107f3bd37ac9350161e60e0a725f18c786fa5f10b14c
#: * ``sg_2b.task``   ccc6f1a9e2c2aff0e52f92c079262184c7b5fcefb487a941e2023864726c1278
#: * ``sg_9b.native`` a2f45307278ae5fbe476dfb1a088b94d6c022f16bd150c9bcb616d7267ac2394
#: * ``sg_9b.task``   704c54e3f73fd4173d74aa355fad68dec447ad90e4a93410b74e675fa7836baf
#:
#: This older constant stays None;
#: the registry candidates ``shieldgemma-2b/-9b`` are not used by the probe, and would still
#: need a guideline decided for them.
SHIELDGEMMA_GUIDELINE: str | None = None  # TODO(decision): only for this older wrapper

#: Granite Guardian scores one risk criterion at a time. "harm" is the general harmful-content
#: criterion; "social_bias" is prejudice based on identity. Which one stands for "toxic" here
#: is a design decision. Only relevant if Granite Guardian is selected.
GRANITE_CRITERIA_ID: str | None = None  # TODO(decision)


# ---------------------------------------------------------------------------------------
# prompt templates that are ours rather than the tokenizer's
# ---------------------------------------------------------------------------------------

#: ShieldGemma's own prompt-side template, transcribed from the model card. The slots are
#: filled by :meth:`GuardLMVerifier.build_prompt`.
#:
#: The line breaks are the card's, not ours: the model was trained on this wording with
#: these breaks, so reflowing it into long lines would change the prompt the model sees.
#: Checked 2026-09-24: the card is public and its code snippet matches this apart from the
#: final blank line, but the chat
#: template the tokenizer ships (one line per sentence, and a ``===`` separator before the
#: question) differs from it. The step 2 probe renders with that chat template instead.
#:
#: The card also has a response-side variant ("Chatbot Response:"), which we do not use: it
#: needs a user turn, and the texts scored here are standalone sentences, so any user turn
#: would have to be invented. Noted in the verifier card.
SHIELDGEMMA_PROMPT_TEMPLATE = """You are a policy expert trying to help determine whether a user
prompt is in violation of the defined safety policies.

<start_of_turn>
Human Question: {text}
<end_of_turn>

Our safety principle is defined in the below:

* {guideline}

Does the human question violate the above principle? Your answer must start
with 'Yes' or 'No'. And then walk through step by step to be sure we answer
correctly.

"""


# ---------------------------------------------------------------------------------------
# download manifest (written by env/download_models.sh)
# ---------------------------------------------------------------------------------------

MANIFEST_ENV = "RAFR_MODEL_MANIFEST"
DEFAULT_MANIFEST = "/workspace/models/model_manifest.json"


def manifest_path() -> Path:
    return Path(os.environ.get(MANIFEST_ENV, "").strip() or DEFAULT_MANIFEST)


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """The download manifest, or ``{}`` when the machine has never downloaded anything."""
    p = Path(path) if path is not None else manifest_path()
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise VerifierConfigError(f"{p}: model manifest must be a JSON object")
    return data


def pinned_revision(repo_id: str, manifest: Mapping[str, Any] | None = None) -> str | None:
    """The commit ``env/download_models.sh`` resolved for ``repo_id``, or ``None``.

    ``None`` means nothing pinned it; the verifier still works (useful for a smoke test) but
    reports ``model_version = "<repo>@unpinned"`` and formal scoring refuses it.
    """
    data = load_manifest() if manifest is None else manifest
    entry = (data.get("models") or {}).get(repo_id)
    if not entry:
        return None
    rev = str(entry.get("revision") or "").strip()
    return rev or None


# ---------------------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSpec:
    """Everything the download script and the card generator need, plus how to build it."""

    name: str
    repo_id: str
    kind: str
    license: str
    gated: bool
    dtype: str
    hub_sha_at_check: str
    notes: str = ""
    build_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_hf_token(self) -> bool:
        """Gated repos need ``HF_TOKEN`` and a manually accepted license (Llama, Gemma)."""
        return self.gated


def _binary_labels() -> tuple[LabelGroup, LabelGroup]:
    """Llama Guard 3 answers with the bare word; both spellings are listed because whether
    the tokenizer sees a leading space depends on what precedes it in the template."""
    return (
        LabelGroup(SAFE, ("safe", " safe")),
        LabelGroup(UNSAFE, ("unsafe", " unsafe")),
    )


def _yes_no_labels() -> tuple[LabelGroup, LabelGroup]:
    """ShieldGemma and Granite Guardian answer the question, not the verdict: Yes = violating."""
    return (
        LabelGroup(SAFE, ("No", " No", "no", " no")),
        LabelGroup(UNSAFE, ("Yes", " Yes", "yes", " yes")),
    )


def _qwen_labels() -> tuple[LabelGroup, LabelGroup, LabelGroup]:
    return (
        LabelGroup(SAFE, ("Safe", " Safe")),
        LabelGroup(UNSAFE, ("Unsafe", " Unsafe")),
        LabelGroup(CONTROVERSIAL, ("Controversial", " Controversial")),
    )


CANDIDATES: dict[str, CandidateSpec] = {
    # -- encoder classifiers ------------------------------------------------------------
    "toxic-bert": CandidateSpec(
        name="toxic-bert",
        repo_id="unitary/toxic-bert",
        kind=ENCODER,
        license="apache-2.0",
        gated=False,
        dtype="float32",
        hub_sha_at_check="4d6c22e74ba2fdd26bc4f7238f50766b045a0d94",
        notes=(
            "Detoxify's original checkpoint. 6 independent BCE logits "
            "(toxic, severe_toxic, obscene, threat, insult, identity_hate); only 'toxic' is read."
        ),
        build_kwargs={"head": MULTILABEL_SIGMOID, "toxic_label": "toxic"},
    ),
    "roberta-toxicity": CandidateSpec(
        name="roberta-toxicity",
        repo_id="s-nlp/roberta_toxicity_classifier",
        kind=ENCODER,
        license="openrail++",
        gated=False,
        dtype="float32",
        hub_sha_at_check="048c25bb1e199b98802784f96325f4840f22145d",
        notes=(
            "Jigsaw-trained RoBERTa, id2label = {0: neutral, 1: toxic}, "
            "problem_type=single_label_classification. Same lab as ParaDetox, so treat the "
            "error patterns as correlated with the data source when judging heterogeneity "
            "(II.4.1)."
        ),
        build_kwargs={"head": SINGLE_LABEL_SOFTMAX, "toxic_label": "toxic"},
    ),
    # -- guard LMs ----------------------------------------------------------------------
    "llama-guard-3-8b": CandidateSpec(
        name="llama-guard-3-8b",
        repo_id="meta-llama/Llama-Guard-3-8B",
        kind=GUARD_LM,
        license="llama3.1",
        gated=True,
        dtype="bfloat16",
        hub_sha_at_check="7327bd9f6efbbe6101dc6cc4736302b3cbb6e425",
        notes=(
            "Gated (manual approval) and under the Llama 3.1 Community License: research use "
            "is allowed, redistribution and the 'Built with Llama' attribution clause are not "
            "automatic. Answers 'safe' / 'unsafe' as the first generated token."
        ),
        build_kwargs={
            "labels": _binary_labels(),
            "label_prefix": "",
            "prompt_style": CHAT,
        },
    ),
    "llama-guard-3-1b": CandidateSpec(
        name="llama-guard-3-1b",
        repo_id="meta-llama/Llama-Guard-3-1B",
        kind=GUARD_LM,
        license="llama3.2",
        gated=True,
        dtype="bfloat16",
        hub_sha_at_check="acf7aafa60f0410f8f42b1fa35e077d705892029",
        notes="Same output contract as the 8B, Llama 3.2 Community License.",
        build_kwargs={
            "labels": _binary_labels(),
            "label_prefix": "",
            "prompt_style": CHAT,
        },
    ),
    "qwen3guard-gen-0.6b": CandidateSpec(
        name="qwen3guard-gen-0.6b",
        repo_id="Qwen/Qwen3Guard-Gen-0.6B",
        kind=GUARD_LM,
        license="apache-2.0",
        gated=False,
        dtype="bfloat16",
        hub_sha_at_check="fada3b2f655b89601929198343c94cd2f64d93cc",
        notes=(
            "Emits 'Safety: <Safe|Unsafe|Controversial>' then 'Categories: ...', so the "
            "label prefix 'Safety: ' is teacher-forced. Third verdict: see "
            "QWEN3GUARD_CONTROVERSIAL_POLICY. The Stream-* variants are for incremental "
            "decoding and are not candidates here."
        ),
        build_kwargs={
            "labels": _qwen_labels(),
            "label_prefix": "Safety: ",  # TODO(verify-on-gpu): confirm against a greedy decode
            "prompt_style": CHAT,
        },
    ),
    "qwen3guard-gen-4b": CandidateSpec(
        name="qwen3guard-gen-4b",
        repo_id="Qwen/Qwen3Guard-Gen-4B",
        kind=GUARD_LM,
        license="apache-2.0",
        gated=False,
        dtype="bfloat16",
        hub_sha_at_check="6ec42827da0c1ff11e7a49dc269d2e810d27e108",
        notes="See qwen3guard-gen-0.6b.",
        build_kwargs={
            "labels": _qwen_labels(),
            "label_prefix": "Safety: ",  # TODO(verify-on-gpu)
            "prompt_style": CHAT,
        },
    ),
    "qwen3guard-gen-8b": CandidateSpec(
        name="qwen3guard-gen-8b",
        repo_id="Qwen/Qwen3Guard-Gen-8B",
        kind=GUARD_LM,
        license="apache-2.0",
        gated=False,
        dtype="bfloat16",
        hub_sha_at_check="4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb",
        notes="See qwen3guard-gen-0.6b.",
        build_kwargs={
            "labels": _qwen_labels(),
            "label_prefix": "Safety: ",  # TODO(verify-on-gpu)
            "prompt_style": CHAT,
        },
    ),
    "shieldgemma-2b": CandidateSpec(
        name="shieldgemma-2b",
        repo_id="google/shieldgemma-2b",
        kind=GUARD_LM,
        license="gemma",
        gated=True,
        dtype="bfloat16",
        hub_sha_at_check="d1dffc9c8c9237a90aab09c61383791e718ef9e8",
        notes=(
            "Gated (manual approval), Gemma Terms of Use. Scores a text against a policy the "
            "caller supplies (SHIELDGEMMA_GUIDELINE) and answers 'Yes' / 'No' to whether the "
            "policy is violated, so Yes is the toxic side. The prompt template is ours "
            "(SHIELDGEMMA_PROMPT_TEMPLATE), transcribed from the model card, not the "
            "tokenizer's chat template."
        ),
        build_kwargs={
            "labels": _yes_no_labels(),
            "label_prefix": "",
            "prompt_style": RAW,
            "raw_template": SHIELDGEMMA_PROMPT_TEMPLATE,
        },
    ),
    "shieldgemma-9b": CandidateSpec(
        name="shieldgemma-9b",
        repo_id="google/shieldgemma-9b",
        kind=GUARD_LM,
        license="gemma",
        gated=True,
        dtype="bfloat16",
        hub_sha_at_check="b8b636016df4540721a098c7aab91c97ec6ee508",
        notes="See shieldgemma-2b.",
        build_kwargs={
            "labels": _yes_no_labels(),
            "label_prefix": "",
            "prompt_style": RAW,
            "raw_template": SHIELDGEMMA_PROMPT_TEMPLATE,
        },
    ),
    "granite-guardian-3.3-8b": CandidateSpec(
        name="granite-guardian-3.3-8b",
        repo_id="ibm-granite/granite-guardian-3.3-8b",
        kind=GUARD_LM,
        license="apache-2.0",
        gated=False,
        dtype="bfloat16",
        hub_sha_at_check="b3421eda4ba6fc9f9a71121d7e62de08827469a4",
        notes=(
            "Answers one risk criterion at a time (GRANITE_CRITERIA_ID) with 'yes' / 'no'. "
            "3.3 wraps the verdict in <score></score> and has an optional thinking mode, so "
            "the chat template is called with think=False and '<score>' is teacher-forced."
        ),
        build_kwargs={
            "labels": _yes_no_labels(),
            "label_prefix": "<score>",  # TODO(verify-on-gpu): confirm against a greedy decode
            "prompt_style": CHAT,
            "chat_template_kwargs": {"think": False},
        },
    ),
}

#: Repos that must never appear: the controller and E already occupy these two sources
#: (II.9.1, App. D). Checked by a test, not by convention.
FORBIDDEN_VENDOR_PREFIXES = ("openai/", "openai-community/", "anthropic/")


def names() -> list[str]:
    return sorted(CANDIDATES)


def spec(name: str) -> CandidateSpec:
    try:
        return CANDIDATES[name]
    except KeyError:
        raise VerifierConfigError(f"unknown candidate {name!r}; known: {names()}") from None


def truncation() -> Truncation:
    """One policy for every candidate (see :data:`TEXT_MAX_TOKENS`).

    The two padding flags follow ``RAFR_VERIFIER_DYNAMIC_PADDING``, resolved here rather
    than at inference time so that the flags -- and therefore the verifier card -- describe
    what the run actually did.
    """
    fixed = not dynamic_padding_requested()
    return Truncation(
        max_length=TEXT_MAX_TOKENS,
        side="right",
        pad_to_max_length=fixed,
        fixed_batch_shape=fixed,
    )


def build(name: str, *, verifier_id: str | None = None, **overrides: Any) -> Any:
    """Construct a candidate. Weights are NOT loaded here -- the backend is lazy."""
    sp = spec(name)
    kwargs: dict[str, Any] = dict(sp.build_kwargs)
    kwargs.update(overrides)
    kwargs.setdefault("dtype", sp.dtype)

    common = {
        "verifier_id": verifier_id or sp.name,
        "repo_id": sp.repo_id,
        "revision": kwargs.pop("revision", None) or pinned_revision(sp.repo_id),
        "truncation": kwargs.pop("truncation", None) or truncation(),
    }
    if sp.kind == ENCODER:
        return EncoderToxicityVerifier(**common, **kwargs)

    if sp.repo_id.startswith("Qwen/Qwen3Guard"):
        kwargs.setdefault("controversial_policy", QWEN3GUARD_CONTROVERSIAL_POLICY)
    if sp.repo_id.startswith("google/shieldgemma") and "guideline" not in kwargs:
        kwargs["guideline"] = _require_guideline(sp.name)
    if sp.repo_id.startswith("ibm-granite/granite-guardian"):
        chat_kwargs = dict(kwargs.get("chat_template_kwargs") or {})
        chat_kwargs.setdefault("think", False)
        if "guardian_config" not in chat_kwargs:
            chat_kwargs["guardian_config"] = {"criteria_id": _require_criteria(sp.name)}
        kwargs["chat_template_kwargs"] = chat_kwargs
    return GuardLMVerifier(**common, **kwargs)


def _require_guideline(name: str) -> str:
    if not SHIELDGEMMA_GUIDELINE or not SHIELDGEMMA_GUIDELINE.strip():
        raise VerifierConfigError(
            f"{name}: SHIELDGEMMA_GUIDELINE is the safety policy the model scores against. "
            "It is prompt content and therefore the authors' to write; set it in "
            "rafr.verifiers.candidates.registry before using this candidate."
        )
    return SHIELDGEMMA_GUIDELINE


def _require_criteria(name: str) -> str:
    if not GRANITE_CRITERIA_ID or not GRANITE_CRITERIA_ID.strip():
        raise VerifierConfigError(
            f"{name}: GRANITE_CRITERIA_ID selects which risk the model scores "
            "('harm' = generally harmful content, 'social_bias' = prejudice based on "
            "identity). That choice defines what the score means, so it has no default; set "
            "it in rafr.verifiers.candidates.registry before using this candidate."
        )
    return GRANITE_CRITERIA_ID


# ---------------------------------------------------------------------------------------
# zero-argument factories for models.yaml: verifiers.v{1,2}.impl
# ---------------------------------------------------------------------------------------


def toxic_bert() -> Any:
    return build("toxic-bert")


def roberta_toxicity() -> Any:
    return build("roberta-toxicity")


def llama_guard_3_8b() -> Any:
    return build("llama-guard-3-8b")


def llama_guard_3_1b() -> Any:
    return build("llama-guard-3-1b")


def qwen3guard_gen_0_6b() -> Any:
    return build("qwen3guard-gen-0.6b")


def qwen3guard_gen_4b() -> Any:
    return build("qwen3guard-gen-4b")


def qwen3guard_gen_8b() -> Any:
    return build("qwen3guard-gen-8b")


def shieldgemma_2b() -> Any:
    return build("shieldgemma-2b")


def shieldgemma_9b() -> Any:
    return build("shieldgemma-9b")


def granite_guardian_3_3_8b() -> Any:
    return build("granite-guardian-3.3-8b")


def _cli(argv: list[str] | None = None) -> int:
    """``python -m rafr.verifiers.candidates.registry`` -- the list env/download_models.sh reads.

    ``--format repos`` prints one repo id per line; ``--format tsv`` adds name, family,
    license and gate status; ``--format json`` prints the whole table.
    """
    import argparse

    parser = argparse.ArgumentParser(description="list the V1/V2 candidate pool")
    parser.add_argument("--format", choices=("repos", "tsv", "json"), default="tsv")
    parser.add_argument("--gated", choices=("only", "exclude"), help="filter by gate status")
    parser.add_argument("--name", nargs="*", help="restrict to these candidate names")
    args = parser.parse_args(argv)

    selected = [spec(n) for n in (args.name or names())]
    if args.gated == "only":
        selected = [s for s in selected if s.gated]
    elif args.gated == "exclude":
        selected = [s for s in selected if not s.gated]

    if args.format == "repos":
        for s in selected:
            print(s.repo_id)
    elif args.format == "tsv":
        for s in selected:
            print(f"{s.name}\t{s.repo_id}\t{s.kind}\t{s.license}\t{'gated' if s.gated else 'open'}")
    else:
        print(
            json.dumps(
                [
                    {
                        "name": s.name,
                        "repo_id": s.repo_id,
                        "kind": s.kind,
                        "license": s.license,
                        "gated": s.gated,
                        "dtype": s.dtype,
                        "hub_sha_at_check": s.hub_sha_at_check,
                        "verified_on": VERIFIED_ON,
                    }
                    for s in selected
                ],
                indent=2,
            )
        )
    return 0


#: candidate name -> the ``"package.module:factory"`` string that goes into
#: ``configs/models.yaml: verifiers.v{1,2}.impl`` once the authors pick V1 and V2.
FACTORIES: dict[str, str] = {
    "toxic-bert": f"{__name__}:toxic_bert",
    "roberta-toxicity": f"{__name__}:roberta_toxicity",
    "llama-guard-3-8b": f"{__name__}:llama_guard_3_8b",
    "llama-guard-3-1b": f"{__name__}:llama_guard_3_1b",
    "qwen3guard-gen-0.6b": f"{__name__}:qwen3guard_gen_0_6b",
    "qwen3guard-gen-4b": f"{__name__}:qwen3guard_gen_4b",
    "qwen3guard-gen-8b": f"{__name__}:qwen3guard_gen_8b",
    "shieldgemma-2b": f"{__name__}:shieldgemma_2b",
    "shieldgemma-9b": f"{__name__}:shieldgemma_9b",
    "granite-guardian-3.3-8b": f"{__name__}:granite_guardian_3_3_8b",
}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
