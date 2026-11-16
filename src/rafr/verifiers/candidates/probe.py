"""Step 2 probe: the candidates, their two policy variants, and a pure two-token scorer.

What is probed and how is fixed by the authors' step-2 instructions:

=================  =========================================  ===========
id                 HF repo                                    runs on
=================  =========================================  ===========
``enc_paradetox``  s-nlp/roberta_toxicity_classifier          local CPU
``sg_2b``          google/shieldgemma-2b                      GPU
``sg_9b``          google/shieldgemma-9b                      GPU
``lg_1b``          meta-llama/Llama-Guard-3-1B                GPU
``lg_12b``         meta-llama/Llama-Guard-4-12B               GPU
``gg_8b``          ibm-granite/granite-guardian-4.1-8b        GPU
=================  =========================================  ===========

There is no smaller Granite Guardian 4.1 on the Hub (``ibm-granite`` listing, 2026-09-24:
``granite-guardian-4.1-8b`` and its GGUF quantisation only), so there is no ``gg_small``.

Every guard LM runs two variants. ``native`` is the vendor's own policy: ShieldGemma's
Harassment and Hate Speech guidelines from the model card (one call each, score = the larger
P(Yes), both sub-scores kept), Llama Guard's default category list, Granite Guardian's
pre-baked harm criterion in no-think mode. ``task`` takes the one word E's rubric-free prompt
uses to name what is judged, "toxic", and puts it into each vendor's own customisation format,
replacing only the content word (2026-09-24): a ShieldGemma
"No ..." policy, a single Llama Guard category, a Granite Guardian bring-your-own criterion.
Every prompt text lives in ``configs/prompts/verifier_*.txt`` (the three task texts together in
``verifier_task_def.txt``); this module adds no wording of its own.

Input: each state is ONE user message (prompt classification) -- no assistant turn, no x0,
no other context. Granite Guardian's judging instruction is a second user message after it,
because that is how its model card and cookbook attach a criterion.

Score: one forward pass, the logits at the verdict position, a softmax over the two verdict
tokens only (ShieldGemma Yes / No, Llama Guard unsafe / safe, Granite Guardian yes / no).
When the model always emits something fixed before the verdict (Llama Guard's ``"\\n\\n"``,
Granite Guardian's empty think block and ``<score>``), those tokens are appended to the input
and the logits are read at the next position. The prefix and the verdict token ids are taken
from a greedy decode of the direction-check sentences (see :mod:`rafr.verifiers.probe_run`)
and checked against :attr:`ProbeModel.label_prefix` before anything is scored.

Nothing here imports torch; :mod:`rafr.verifiers.candidates.torch_probe` does, lazily.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rafr.core import paths
from rafr.verifiers.candidates.base import VerifierConfigError, clamp_score, softmax

# ---------------------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------------------

ENC = "enc"
SG = "sg"
LG = "lg"
GG = "gg"
FAMILIES = (ENC, SG, LG, GG)

NATIVE = "native"
TASK = "task"
VARIANTS = (NATIVE, TASK)

SEQ_CLS = "sequence_classification"
CAUSAL_LM = "causal_lm"
IMAGE_TEXT_TO_TEXT = "image_text_to_text"

CPU = "cpu"
GPU = "gpu"

#: Text budget in tokens of each model's own tokenizer, the same for every candidate (see
#: ``registry.TEXT_MAX_TOKENS``). A text is cut before it is templated, never the template.
TEXT_MAX_TOKENS = 512

#: One inference batch size for every guard LM (decoder: left padding, fixed batch shape).
LM_BATCH_SIZE = 8
ENCODER_BATCH_SIZE = 16
#: Fixed prompt lengths are rounded up to a multiple of this, after one multiple of slack.
PAD_MULTIPLE = 64

# Direction self-check (section 4): two invented sentences, never data. The
#: neutral one is also the example input of the rendered prompts shown for review.
NEUTRAL_EXAMPLE = "The community garden on Elm Street opens at eight every Saturday morning."
ABUSIVE_EXAMPLE = "You are a worthless, pathetic idiot and everyone is sick of you."

PROMPTS_DIR_NAME = "prompts"
TASK_DEF_FILE = "verifier_task_def.txt"
SG_HARASSMENT_FILE = "verifier_sg_harassment.txt"
SG_HATE_SPEECH_FILE = "verifier_sg_hate_speech.txt"
GG_HARM_FILE = "verifier_gg_harm.txt"
GG_BLOCK_FILE = "verifier_gg_block.txt"
PROMPT_FILES = (TASK_DEF_FILE, SG_HARASSMENT_FILE, SG_HATE_SPEECH_FILE, GG_HARM_FILE, GG_BLOCK_FILE)
#: Written once the authors have confirmed the rendered prompts; scoring refuses to
#: start while it is missing or while a prompt file no longer matches it.
PROMPT_LOCK_FILE = "verifier_probe_prompts.lock.json"

#: Llama Guard's custom-category key for the task variant: the one category is the first.
LG_TASK_CATEGORY_KEY = "S1"
LG_CATEGORIES_BEGIN = "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
LG_CATEGORIES_END = "<END UNSAFE CONTENT CATEGORIES>"
GG_CRITERIA_SLOT = "{criteria}"


def prompts_dir() -> Path:
    return paths.CONFIGS / PROMPTS_DIR_NAME


# ---------------------------------------------------------------------------------------
# prompt files
# ---------------------------------------------------------------------------------------


def text_sha256(text: str) -> str:
    """sha256 of UTF-8 text with CRLF folded to LF (as ``rafr.evaluator.judge.prompt_sha256``):
    git checks prompt files out as CRLF on Windows and LF on the pod."""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def read_prompt_file(name: str, directory: Path | None = None) -> str:
    """A prompt file's text, LF line endings, without the one trailing newline an editor adds."""
    path = Path(directory or prompts_dir()) / name
    if not path.is_file():
        raise VerifierConfigError(f"prompt file {path} not found")
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if text.endswith("\n"):
        text = text[:-1]
    if not text.strip():
        raise VerifierConfigError(f"prompt file {path} is empty")
    return text


#: The keys of ``verifier_task_def.txt``, in file order.
TASK_KEYS = (
    "source_word",
    "shieldgemma_policy",
    "llama_guard_category",
    "granite_guardian_criteria",
)


def read_task_texts(directory: Path | None = None) -> dict[str, str]:
    """The task variant's wording: the source word and the three vendor-format texts.

    ``key: value`` lines; blank lines and ``#`` comments are skipped; the value is everything
    after the first ``": "`` (ShieldGemma's own policy text contains ``": "`` itself). Each
    text must contain the source word, which is what makes it the task variant.
    """
    out: dict[str, str] = {}
    for line in read_prompt_file(TASK_DEF_FILE, directory).split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(": ")
        if not sep or key not in TASK_KEYS or key in out or not value.strip():
            raise VerifierConfigError(f"{TASK_DEF_FILE}: bad line {line!r}")
        out[key] = value
    if tuple(out) != TASK_KEYS:
        raise VerifierConfigError(f"{TASK_DEF_FILE}: expected keys {TASK_KEYS}, got {tuple(out)}")
    word = out["source_word"].lower()
    for key in TASK_KEYS[1:]:
        if word not in out[key].lower():
            raise VerifierConfigError(f"{TASK_DEF_FILE}: {key} does not contain {word!r}")
    return out


def prompt_file_hashes(directory: Path | None = None) -> dict[str, str]:
    """``{file name: sha256}`` over the file contents (LF-normalised) of every probe prompt."""
    d = Path(directory or prompts_dir())
    out: dict[str, str] = {}
    for name in PROMPT_FILES:
        path = d / name
        if not path.is_file():
            raise VerifierConfigError(f"prompt file {path} not found")
        out[name] = text_sha256(path.read_text(encoding="utf-8"))
    return out


# ---------------------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeModel:
    """One set of weights. ``label_prefix`` is what the model is expected to emit before the
    verdict; the GPU preflight compares it with a greedy decode and stops on a mismatch."""

    model_id: str
    family: str
    repo_id: str
    vendor: str
    base_model: str
    params_b: float
    device: str
    dtype: str
    loader: str
    toxic_word: str | None
    safe_word: str | None
    label_prefix: str | None
    hub_sha_at_check: str
    license: str
    gated: bool

    @property
    def is_lm(self) -> bool:
        return self.family != ENC


#: Repo heads and gate status as the Hub API reported them on 2026-09-24. Provenance only: the
#: revision a run uses is the one the download step resolves and records.
MODELS: dict[str, ProbeModel] = {
    "enc_paradetox": ProbeModel(
        model_id="enc_paradetox",
        family=ENC,
        repo_id="s-nlp/roberta_toxicity_classifier",
        vendor="s-nlp (Skoltech)",
        base_model="RoBERTa-base",
        params_b=0.125,
        device=CPU,
        dtype="float32",
        loader=SEQ_CLS,
        toxic_word=None,
        safe_word=None,
        label_prefix=None,
        hub_sha_at_check="048c25bb1e199b98802784f96325f4840f22145d",
        license="openrail++",
        gated=False,
    ),
    "sg_2b": ProbeModel(
        model_id="sg_2b",
        family=SG,
        repo_id="google/shieldgemma-2b",
        vendor="Google",
        base_model="Gemma 2 2B",
        params_b=2.6,
        device=GPU,
        dtype="bfloat16",
        loader=CAUSAL_LM,
        toxic_word="Yes",
        safe_word="No",
        label_prefix="",
        hub_sha_at_check="d1dffc9c8c9237a90aab09c61383791e718ef9e8",
        license="gemma",
        gated=True,
    ),
    "sg_9b": ProbeModel(
        model_id="sg_9b",
        family=SG,
        repo_id="google/shieldgemma-9b",
        vendor="Google",
        base_model="Gemma 2 9B",
        params_b=9.2,
        device=GPU,
        dtype="bfloat16",
        loader=CAUSAL_LM,
        toxic_word="Yes",
        safe_word="No",
        label_prefix="",
        hub_sha_at_check="b8b636016df4540721a098c7aab91c97ec6ee508",
        license="gemma",
        gated=True,
    ),
    "lg_1b": ProbeModel(
        model_id="lg_1b",
        family=LG,
        repo_id="meta-llama/Llama-Guard-3-1B",
        vendor="Meta",
        base_model="Llama 3.2 1B",
        params_b=1.5,
        device=GPU,
        dtype="bfloat16",
        loader=CAUSAL_LM,
        toxic_word="unsafe",
        safe_word="safe",
        label_prefix="\n\n",
        hub_sha_at_check="acf7aafa60f0410f8f42b1fa35e077d705892029",
        license="llama3.2",
        gated=True,
    ),
    "lg_12b": ProbeModel(
        model_id="lg_12b",
        family=LG,
        repo_id="meta-llama/Llama-Guard-4-12B",
        vendor="Meta",
        base_model="Llama 4 Scout (pruned to dense)",
        params_b=12.0,
        device=GPU,
        dtype="bfloat16",
        loader=IMAGE_TEXT_TO_TEXT,
        toxic_word="unsafe",
        safe_word="safe",
        label_prefix="\n\n",
        hub_sha_at_check="87acb4b94e930c3d679e6e7ee9d57e2feab9ea71",
        license="llama4",
        gated=True,
    ),
    "gg_8b": ProbeModel(
        model_id="gg_8b",
        family=GG,
        repo_id="ibm-granite/granite-guardian-4.1-8b",
        vendor="IBM",
        base_model="Granite 4.1 8B",
        params_b=8.0,
        device=GPU,
        dtype="bfloat16",
        loader=CAUSAL_LM,
        toxic_word="yes",
        safe_word="no",
        # no-think mode: "Output empty <think>\n</think> tags followed by your score in
        # <score></score> tags" (model card). The exact tokens are the preflight's to confirm.
        label_prefix="<think>\n</think>\n<score>",
        hub_sha_at_check="ab01ccca5dcfb80246369a086a4a87a29198f5af",
        license="apache-2.0",
        gated=False,
    ),
}

#: Order the GPU runs them in: small first, so a wiring problem shows up cheaply.
GPU_ORDER = ("lg_1b", "sg_2b", "gg_8b", "sg_9b", "lg_12b")


def model(model_id: str) -> ProbeModel:
    try:
        return MODELS[model_id]
    except KeyError:
        raise VerifierConfigError(
            f"unknown probe model {model_id!r}; known: {sorted(MODELS)}"
        ) from None


# ---------------------------------------------------------------------------------------
# configurations: model x variant, each a list of forward calls per text
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeCall:
    """One forward pass per text: the chat-template keyword arguments, plus Granite
    Guardian's criterion (it goes into the judging block, not into a template argument)."""

    sub_id: str
    template_kwargs: dict[str, Any] = field(default_factory=dict)
    criteria: str | None = None


@dataclass(frozen=True)
class ProbeConfig:
    verifier_id: str
    model: ProbeModel
    variant: str | None
    calls: tuple[ProbeCall, ...]
    #: source of each piece of wording, for the report
    sources: tuple[str, ...] = ()

    @property
    def aggregates(self) -> bool:
        return len(self.calls) > 1


def verifier_id_for(model_id: str, variant: str | None) -> str:
    return model_id if variant is None else f"{model_id}.{variant}"


def configs_for(model_id: str, prompt_dir: Path | None = None) -> list[ProbeConfig]:
    """Both variants of a guard LM, or the single encoder configuration."""
    m = model(model_id)
    if m.family == ENC:
        return [ProbeConfig(verifier_id_for(model_id, None), m, None, ())]
    task = read_task_texts(prompt_dir)
    word = task["source_word"]
    out: list[ProbeConfig] = []
    for variant in VARIANTS:
        calls: tuple[ProbeCall, ...]
        if m.family == SG:
            if variant == NATIVE:
                calls = (
                    ProbeCall(
                        "harassment",
                        {"guideline": read_prompt_file(SG_HARASSMENT_FILE, prompt_dir)},
                    ),
                    ProbeCall(
                        "hate_speech",
                        {"guideline": read_prompt_file(SG_HATE_SPEECH_FILE, prompt_dir)},
                    ),
                )
                sources = ("model card, 'Guidelines to use in prompts', use case 1 (prompt-only)",)
            else:
                calls = (ProbeCall(TASK, {"guideline": task["shieldgemma_policy"]}),)
                sources = (
                    f"model-card policy format, content word replaced by {word!r} "
                    f"({TASK_DEF_FILE}: shieldgemma_policy)",
                )
        elif m.family == LG:
            if variant == NATIVE:
                calls = (ProbeCall(NATIVE, {}),)
                sources = ("chat template default categories, unchanged",)
            else:
                category = {LG_TASK_CATEGORY_KEY: task["llama_guard_category"]}
                calls = (ProbeCall(TASK, {"categories": category}),)
                sources = (
                    f"the only category, key {LG_TASK_CATEGORY_KEY!r}, in the template's "
                    f"category format with the content word {word!r} "
                    f"({TASK_DEF_FILE}: llama_guard_category)",
                )
        elif m.family == GG:
            if variant == NATIVE:
                calls = (ProbeCall(NATIVE, {}, read_prompt_file(GG_HARM_FILE, prompt_dir)),)
                sources = ("model card, 'List of Pre-baked Criteria', Harm",)
            else:
                calls = (ProbeCall(TASK, {}, task["granite_guardian_criteria"]),)
                sources = (
                    f"bring-your-own criterion with the content word {word!r} "
                    f"({TASK_DEF_FILE}: granite_guardian_criteria)",
                )
        else:  # pragma: no cover - ENC handled above
            raise VerifierConfigError(f"no variants for family {m.family!r}")
        out.append(ProbeConfig(verifier_id_for(model_id, variant), m, variant, calls, sources))
    return out


def all_configs(prompt_dir: Path | None = None) -> list[ProbeConfig]:
    out: list[ProbeConfig] = []
    for model_id in MODELS:
        out.extend(configs_for(model_id, prompt_dir))
    return out


def all_verifier_ids(prompt_dir: Path | None = None) -> list[str]:
    return [c.verifier_id for c in all_configs(prompt_dir)]


# ---------------------------------------------------------------------------------------
# messages (pure)
# ---------------------------------------------------------------------------------------


def guardian_block(criteria: str, prompt_dir: Path | None = None) -> str:
    """Granite Guardian's judging block (no-think, prompt-only schema) around one criterion."""
    template = read_prompt_file(GG_BLOCK_FILE, prompt_dir)
    if template.count(GG_CRITERIA_SLOT) != 1:
        raise VerifierConfigError(f"{GG_BLOCK_FILE} must hold exactly one {GG_CRITERIA_SLOT}")
    return template.replace(GG_CRITERIA_SLOT, criteria)


def build_messages(family: str, text: str, block: str | None = None) -> list[dict[str, Any]]:
    """The conversation handed to the model's own chat template: the state is the one user
    message. Llama Guard's template reads content as a list of typed parts; Granite Guardian
    takes its judging ``block`` as a second user message."""
    if family == SG:
        return [{"role": "user", "content": text}]
    if family == LG:
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]
    if family == GG:
        if block is None:
            raise VerifierConfigError("Granite Guardian needs its judging block")
        return [{"role": "user", "content": text}, {"role": "user", "content": block}]
    raise VerifierConfigError(f"family {family!r} has no messages (an encoder takes raw text)")


def messages_for(
    config: ProbeConfig, call: ProbeCall, text: str, prompt_dir: Path | None = None
) -> list[dict[str, Any]]:
    block = None
    if config.model.family == GG:
        if call.criteria is None:
            raise VerifierConfigError(f"{config.verifier_id}: Granite call without a criterion")
        block = guardian_block(call.criteria, prompt_dir)
    return build_messages(config.model.family, text, block)


def manual_lg_categories(native_render: str, categories: dict[str, str]) -> str:
    """Llama Guard's category block replaced by ``categories``, in the template's own
    ``"<key>: <text>\\n"`` layout -- the fallback when a chat template takes no ``categories``
    argument. Built from the model's own native rendering so nothing else moves."""
    start = native_render.find(LG_CATEGORIES_BEGIN)
    end = native_render.find(LG_CATEGORIES_END)
    if start < 0 or end < start:
        raise VerifierConfigError("Llama Guard prompt without its category block markers")
    block = "".join(f"{k}: {v}\n" for k, v in categories.items())
    return native_render[: start + len(LG_CATEGORIES_BEGIN)] + block + native_render[end:]


def categories_applied(render: str, categories: dict[str, str]) -> bool:
    """Did the chat template honour ``categories``? Each custom line must be in the category
    block and nothing else may be."""
    start = render.find(LG_CATEGORIES_BEGIN)
    end = render.find(LG_CATEGORIES_END)
    if start < 0 or end < start:
        return False
    block = render[start + len(LG_CATEGORIES_BEGIN) : end]
    return block == "".join(f"{k}: {v}\n" for k, v in categories.items())


# ---------------------------------------------------------------------------------------
# prompt hash
# ---------------------------------------------------------------------------------------


def prompt_spec(config: ProbeConfig, prompt_dir: Path | None = None) -> dict[str, Any] | None:
    """Everything that decides what a guard LM is shown, besides the text and the weights
    (those are ``model_version``). None for the encoder, which has no prompt."""
    m = config.model
    if not m.is_lm:
        return None
    calls = []
    for call in config.calls:
        entry: dict[str, Any] = {"sub_id": call.sub_id, "template_kwargs": call.template_kwargs}
        if call.criteria is not None:
            entry["guardian_block"] = guardian_block(call.criteria, prompt_dir)
        calls.append(entry)
    return {
        "repo_id": m.repo_id,
        "family": m.family,
        "variant": config.variant,
        "message_format": {SG: "user_str", LG: "user_parts", GG: "user_str+guardian_block"}[
            m.family
        ],
        "calls": calls,
        "aggregate": "max" if config.aggregates else "single",
        "label_words": {"toxic": m.toxic_word, "safe": m.safe_word},
        "label_prefix": m.label_prefix,
        "text_max_tokens": TEXT_MAX_TOKENS,
        "chat_template": "the tokenizer's own, at the pinned revision",
    }


def prompt_hash(config: ProbeConfig, prompt_dir: Path | None = None) -> str | None:
    spec = prompt_spec(config, prompt_dir)
    if spec is None:
        return None
    blob = json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------
# prompt lock
# ---------------------------------------------------------------------------------------


def lock_payload(prompt_dir: Path | None = None) -> dict[str, Any]:
    return {
        "files": prompt_file_hashes(prompt_dir),
        "prompt_hash": {
            c.verifier_id: prompt_hash(c, prompt_dir)
            for c in all_configs(prompt_dir)
            if c.model.is_lm
        },
    }


def check_prompt_lock(prompt_dir: Path | None = None) -> dict[str, Any]:
    """Refuse to score until the prompts the authors confirmed are locked and still unchanged."""
    d = Path(prompt_dir or prompts_dir())
    path = d / PROMPT_LOCK_FILE
    if not path.is_file():
        raise VerifierConfigError(
            f"{path} not found: the probe prompts have not been confirmed and locked yet "
            "(stop point A). Nothing is scored before that."
        )
    locked = json.loads(path.read_text(encoding="utf-8"))
    now = lock_payload(d)
    if locked.get("files") != now["files"] or locked.get("prompt_hash") != now["prompt_hash"]:
        raise VerifierConfigError(
            f"a probe prompt changed after it was locked ({path}); prompts may not change "
            "after stop point A"
        )
    return locked


# ---------------------------------------------------------------------------------------
# the scorer (pure; the backend owns the weights)
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """What the preflight settled: the fixed prefix the model emits before its verdict and the
    two verdict token ids, all as token ids of this model's tokenizer."""

    prefix_ids: tuple[int, ...]
    prefix_text: str
    toxic_id: int
    safe_id: int
    toxic_form: str
    safe_form: str


@runtime_checkable
class ProbeBackend(Protocol):
    """A loaded guard LM. Batching, padding and the forward pass live behind this.

    ``fixed_shape``: every batch is padded to ``batch_size`` rows and to one sequence length
    per call type (``pad_to``), so a score cannot depend on its neighbours in the batch.
    """

    revision: str
    batch_size: int
    fixed_shape: bool

    def n_text_tokens(self, text: str) -> int: ...

    def truncate_text(self, text: str, max_tokens: int) -> str: ...

    def render(self, messages: list[dict[str, Any]], template_kwargs: dict[str, Any]) -> str: ...

    def n_prompt_tokens(self, prompt: str) -> int: ...

    def verdict_logits(
        self, prompts: Sequence[str], verdict: Verdict, pad_to: int | None
    ) -> list[tuple[float, float]]: ...


@dataclass
class ScoredText:
    score: float
    sub_scores: dict[str, float] | None
    n_text_tokens: int
    truncated: bool


class GuardProbeScorer:
    """``score(texts) -> list[float]``, in [0, 1], higher = more toxic, for one config."""

    def __init__(
        self,
        config: ProbeConfig,
        backend: ProbeBackend,
        verdict: Verdict,
        *,
        prompt_dir: Path | None = None,
        lg_manual: bool = False,
    ) -> None:
        if not config.model.is_lm:
            raise VerifierConfigError(f"{config.verifier_id}: not a guard LM")
        self.config = config
        self.backend = backend
        self.verdict = verdict
        self.prompt_dir = prompt_dir
        #: Llama Guard only: the chat template ignored ``categories``, so the task variant's
        #: category block is spliced into the native rendering (manual_lg_categories).
        self.lg_manual = lg_manual
        self.prompt_hash = prompt_hash(config, prompt_dir)
        self._blocks = {
            c.sub_id: guardian_block(c.criteria, prompt_dir)
            for c in config.calls
            if c.criteria is not None
        }
        self._pad_to: dict[str, int] = {}

    @property
    def verifier_id(self) -> str:
        return self.config.verifier_id

    @property
    def model_version(self) -> str:
        return f"{self.config.model.repo_id}@{self.backend.revision}"

    def render(self, call: ProbeCall, text: str) -> str:
        """The finished prompt for one call, WITHOUT the verdict prefix."""
        messages = build_messages(self.config.model.family, text, self._blocks.get(call.sub_id))
        kwargs = dict(call.template_kwargs)
        if self.lg_manual and "categories" in kwargs:
            categories = kwargs.pop("categories")
            return manual_lg_categories(self.backend.render(messages, kwargs), categories)
        return self.backend.render(messages, kwargs)

    def pad_length(self, call: ProbeCall) -> int | None:
        """Fixed sequence length for one call type: the longest prompt it can produce (a text
        at the full budget, templated, plus the verdict prefix), rounded up with slack."""
        if not self.backend.fixed_shape:
            return None
        if call.sub_id not in self._pad_to:
            filler = self.backend.truncate_text(
                " ".join(["word"] * TEXT_MAX_TOKENS), TEXT_MAX_TOKENS
            )
            n = self.backend.n_prompt_tokens(self.render(call, filler))
            n += len(self.verdict.prefix_ids)
            self._pad_to[call.sub_id] = ((n + 2 * PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        return self._pad_to[call.sub_id]

    def score_detailed(self, texts: Sequence[str]) -> list[ScoredText]:
        items = list(texts)
        if any(not isinstance(t, str) for t in items):
            raise TypeError("score() takes a list of str")
        counts = [self.backend.n_text_tokens(t) for t in items]
        cut = [
            self.backend.truncate_text(t, TEXT_MAX_TOKENS) if n > TEXT_MAX_TOKENS else t
            for t, n in zip(items, counts, strict=True)
        ]
        per_call: dict[str, list[float]] = {}
        size = self.backend.batch_size
        for call in self.config.calls:
            probs: list[float] = []
            pad_to = self.pad_length(call)
            for i in range(0, len(cut), size):
                prompts = [self.render(call, t) for t in cut[i : i + size]]
                pairs = self.backend.verdict_logits(prompts, self.verdict, pad_to)
                if len(pairs) != len(prompts):
                    raise VerifierConfigError(
                        f"{self.verifier_id}: backend returned {len(pairs)} rows for "
                        f"{len(prompts)} prompts"
                    )
                for toxic_logit, safe_logit in pairs:
                    p = softmax([float(toxic_logit), float(safe_logit)])[0]
                    probs.append(clamp_score(p, where=self.verifier_id))
            per_call[call.sub_id] = probs
        out: list[ScoredText] = []
        for j, n in enumerate(counts):
            subs = {k: v[j] for k, v in per_call.items()}
            score = max(subs.values())
            out.append(
                ScoredText(
                    score=score,
                    sub_scores=subs if self.config.aggregates else None,
                    n_text_tokens=n,
                    truncated=n > TEXT_MAX_TOKENS,
                )
            )
        return out

    def score(self, texts: list[str]) -> list[float]:
        if isinstance(texts, str):
            raise TypeError("score() takes a list of texts, not a single string")
        return [r.score for r in self.score_detailed(texts)]


# ---------------------------------------------------------------------------------------
# preflight helpers (pure)
# ---------------------------------------------------------------------------------------


def split_prefix(
    generated: Sequence[tuple[int, str]], toxic_word: str, safe_word: str
) -> tuple[list[int], int, str] | None:
    """Find the verdict in a greedy continuation given as ``(token id, decoded piece)`` pairs.

    Returns ``(prefix ids, verdict id, verdict piece)`` for the first token whose piece,
    stripped of surrounding whitespace, is exactly one of the two verdict words (case
    sensitive), or None when no such token appears.
    """
    for k, (tid, piece) in enumerate(generated):
        if piece.strip() in (toxic_word, safe_word):
            return [t for t, _ in generated[:k]], int(tid), piece
    return None


def direction_ok(neutral_score: float, abusive_score: float) -> bool:
    """section 4: the invented abusive sentence must score above the neutral one."""
    return float(abusive_score) > float(neutral_score)
