"""Generative math verifiers (Appendix M.4 v1.2; II.10 generative scoring variant).

Two adapters behind :func:`rafr.domains.math.models.build_verifier`; both write out their
reasoning first and are scored afterwards, from the logits of two verdict tokens at a verdict
position (never from a number the model writes, never from sampling):

* whole-solution reasoning judge, implementation ``hf_generative_judge`` (form ``llm_judge``):
  after the reasoning ends, a fixed verdict prompt is fed as token ids and the score is
  P(negative) over the two verdict tokens at the next position;
* step-level generative PRM, implementation ``hf_generative_step`` (form ``prm``): each step's
  verdict position is located in the generated text by the model card's pattern; the score is
  1 - min_k P(step k correct).

Fixed-shape greedy generation (II.10): ``batch_size`` rows (a short last batch filled with
copies of its last row, whose results are dropped), every row left padded to
``max_input_tokens`` (L_in), one static KV cache of ``L_in + max_new_tokens + max_forced_tokens``
positions. Every decoding step is one forward pass over a ``[batch_size, 1]`` input against the
whole cache, so a row's computation never depends on the other rows. A row stops generating at
its stop string, at an end-of-sequence token, or after ``max_new_tokens`` (L_gen) tokens
(truncation: the reasoning is cut, scored as usual and reported, never dropped); then its forced
suffix is fed token by token in the same loop and the verdict logits are read where it ends. A
finished row keeps its place in the batch until the whole batch is done; its later outputs are
not used. Decoding is greedy (argmax; ties to the lowest token id), nothing is sampled.

Inputs are the problem and the solution only; no reference answer reaches any prompt.
"""

from __future__ import annotations

import importlib
import json
import math
import re
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from rafr.domains.math.models import (
    GENERATIVE_IMPLEMENTATIONS,
    PROMPT_DIR,
    SCORE_DIRECTION,
    MathModelError,
    ModelInputError,
    ModelOutputError,
    _check_slots,
    _library_versions,
    _probabilities,
    _require_license_review,
    _text,
    _unit_score,
    _verified_local_source,
    split_solution_steps,
)
from rafr.llm.client import canonical_json, fill_slots, sha256_hex

GEN_JUDGE_IMPLEMENTATION = "hf_generative_judge"
GEN_STEP_IMPLEMENTATION = "hf_generative_step"
assert set(GENERATIVE_IMPLEMENTATIONS) == {GEN_JUDGE_IMPLEMENTATION, GEN_STEP_IMPLEMENTATION}
GEN_ADAPTER_VERSION = "math-generative-v1"
MODEL_CLASSES = ("AutoModelForCausalLM", "AutoModelForImageTextToText")
#: How the steps without a verdict are handled (the authors decide; ``TBD`` refuses to score):
#: ``judged_only`` -- the minimum over the judged steps (no step judged: every step is forced,
#: as in ``force_remaining``); ``force_remaining`` -- each missing step gets the model's own
#: verdict prefix teacher-forced after the generated text and its verdict logits read there;
#: ``count_incorrect`` -- a missing step counts as P(correct) = 0.
MISSING_RULES = ("judged_only", "force_remaining", "count_incorrect")
DECODING = "greedy argmax over the full vocabulary (ties to the lowest id); nothing sampled"
#: The decoding attention (2026-10-11, round 2 stop point A): transformers' sdpa, except
#: that a one-token query of grouped-query attention reads its own key / value head directly
#: instead of through ``repeat_kv`` copies of the whole static cache
#: (:func:`grouped_decode_attention`).
GROUPED_DECODE_ATTENTION = "rafr_gqa_decode"


def grouped_decode_attention(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    """transformers' ``sdpa_attention_forward``, except for a one-token query (every decoding and
    forced step after the prefill) of grouped-query attention.

    With a mask, transformers' sdpa copies each key / value head ``n_rep`` times over the whole
    static cache (``repeat_kv``) in every layer and every step. Here the ``n_rep`` query heads
    that share a key / value head are laid along the query-length axis instead, ``[B, H, 1, d]
    -> [B, H_kv, n_rep, d]``, and attend that head's keys and values under the row's mask
    ``[B, 1, 1, S]`` (broadcast over the ``n_rep`` positions). Query head ``h`` still attends key /
    value head ``h // n_rep`` (``repeat_kv``'s order) with the same scale and mask: the same
    attention, without the copies. Every other call (the prefill, no mask, no grouping, dropout,
    a position bias, attention weights requested) is transformers' sdpa unchanged.
    """
    torch = importlib.import_module("torch")
    sdpa = importlib.import_module("transformers.integrations.sdpa_attention")
    batch, heads, q_len, dim = query.shape
    kv_heads = key.shape[1]
    grouped = (
        q_len == 1
        and kv_heads < heads
        and heads % kv_heads == 0
        and attention_mask is not None
        and attention_mask.dim() == 4
        and tuple(attention_mask.shape[1:3]) == (1, 1)
        and attention_mask.shape[-1] == key.shape[2]
        and not dropout
        and kwargs.get("position_bias") is None
        and not kwargs.get("output_attentions", False)
    )
    if not grouped:
        return sdpa.sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )
    n_rep = heads // kv_heads
    out = torch.nn.functional.scaled_dot_product_attention(
        query.reshape(batch, kv_heads, n_rep, dim),
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=scaling,
        is_causal=False,
    )
    return out.reshape(batch, heads, 1, value.shape[-1]).transpose(1, 2).contiguous(), None


def register_attention() -> None:
    """Make :data:`GROUPED_DECODE_ATTENTION` loadable by transformers: the attention function,
    and sdpa's own mask (a custom name without a registered mask would get no mask at all)."""
    transformers = importlib.import_module("transformers")
    masks = importlib.import_module("transformers.masking_utils")
    transformers.AttentionInterface.register(GROUPED_DECODE_ATTENTION, grouped_decode_attention)
    transformers.AttentionMaskInterface.register(GROUPED_DECODE_ATTENTION, masks.sdpa_mask)


def read_prompt(
    name: str, slots: tuple[str, ...], directory: Path | None = None
) -> tuple[str, str]:
    """A prompt file (LF, without the editor's final newline) and its SHA256.

    Refused while empty or still a ``PLACEHOLDER``; ``slots`` must each appear exactly once
    and no other slot may appear.
    """
    path = Path(PROMPT_DIR if directory is None else directory) / name
    if not path.is_file():
        raise MathModelError(f"prompt file {path} not found")
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if text.endswith("\n"):
        text = text[:-1]
    if not text.strip() or "PLACEHOLDER" in text:
        raise MathModelError(f"prompt file {path} is empty or still a PLACEHOLDER")
    _check_slots(text, slots, f"prompt {name}")
    return text, sha256_hex(text)


def _prompt_name(value: Any, name: str) -> None:
    if not isinstance(value, str) or Path(value).name != value or not value.endswith(".txt"):
        raise MathModelError(f"{name} must be a .txt file name inside prompts/")


def _sha(value: Any, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise MathModelError(f"{name} must be a SHA256 hex digest")


def _json_mapping(value: Any, name: str) -> None:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise MathModelError(f"{name} must be a mapping with string keys")
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise MathModelError(f"{name} must hold JSON values") from exc


def _int_list(value: Any, name: str, *, nonempty: bool) -> None:
    if (
        not isinstance(value, list)
        or (nonempty and not value)
        or any(type(item) is not int or item < 0 for item in value)
        or len(set(value)) != len(value)
    ):
        raise MathModelError(f"{name} must be a list of distinct nonnegative token ids")


def _check_common(config: Any) -> None:
    pending = [name for name, value in asdict(config).items() if value == "TBD"]
    if pending:
        raise MathModelError(f"still TBD (the authors decide): {', '.join(pending)}")
    _text(config.model_id, "model_id")
    for name in ("revision", "tokenizer_revision"):
        value = getattr(config, name)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
            raise MathModelError(f"{name} must be an immutable 40-character commit SHA")
    if config.model_class not in MODEL_CLASSES:
        raise MathModelError(f"model_class must be one of {MODEL_CLASSES}")
    if config.dtype not in ("float32", "float16", "bfloat16"):
        raise MathModelError("unsupported dtype; quantization is not silently enabled")
    for name in ("batch_size", "max_input_tokens", "max_new_tokens", "max_forced_tokens"):
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise MathModelError(f"{name} must be a positive integer")
    for name in ("local_files_only", "license_reviewed", "trust_remote_code"):
        if type(getattr(config, name)) is not bool:
            raise MathModelError(f"{name} must be boolean")
    if config.trust_remote_code:
        raise MathModelError("generative verifiers run native transformers code only (C.28)")
    _prompt_name(config.prompt_file, "prompt_file")
    _sha(config.prompt_sha256, "prompt_sha256")
    _json_mapping(config.chat_template_kwargs, "chat_template_kwargs")
    _json_mapping(config.tokenizer_kwargs, "tokenizer_kwargs")
    if not isinstance(config.generation_prefix, str):
        raise MathModelError("generation_prefix must be a string")
    if config.stop_string is not None:
        _text(config.stop_string, "stop_string")
    _int_list(config.eos_token_ids, "eos_token_ids", nonempty=True)
    _int_list(config.suppress_token_ids, "suppress_token_ids", nonempty=False)
    for name in ("positive_token", "negative_token"):
        _text(getattr(config, name), name)
    if config.positive_token == config.negative_token:
        raise MathModelError("positive_token and negative_token must differ")
    ids = (config.positive_token_id, config.negative_token_id)
    if (ids[0] is None) != (ids[1] is None):
        raise MathModelError("set both positive_token_id and negative_token_id, or neither")
    if ids[0] is not None and (
        any(type(value) is not int or value < 0 for value in ids) or ids[0] == ids[1]
    ):
        raise MathModelError("verdict token ids must be two distinct nonnegative integers")
    _text(config.device_map, "device_map")
    _text(config.attn_implementation, "attn_implementation")
    if not isinstance(config.license_review_reference, str):
        raise MathModelError("license_review_reference must be a string")
    if config.local_path is not None:
        _text(config.local_path, "local_path")


@dataclass(frozen=True)
class GenerativeJudgeConfig:
    """A reasoning model judging the whole solution; score = P(negative verdict token).

    ``prompt_file`` (slots ``{problem}``, ``{solution}``) is the user turn, rendered by the
    tokenizer's own chat template with its generation prompt, then ``generation_prefix``.
    ``verdict_prompt_file`` (no slots) is fed as token ids after the reasoning ends; when the
    reasoning did not end with ``stop_string`` (an end-of-sequence token, or L_gen reached),
    ``stop_string`` is fed first. The verdict tokens are read at the position after it.
    """

    model_id: str
    revision: str
    tokenizer_revision: str
    model_class: str
    prompt_file: str
    prompt_sha256: str
    verdict_prompt_file: str
    verdict_prompt_sha256: str
    stop_string: str | None
    eos_token_ids: list[int]
    positive_token: str
    negative_token: str
    dtype: str
    batch_size: int
    max_input_tokens: int
    max_new_tokens: int
    max_forced_tokens: int
    positive_token_id: int | None = None
    negative_token_id: int | None = None
    generation_prefix: str = ""
    suppress_token_ids: list[int] = field(default_factory=list)
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)
    attn_implementation: str = "sdpa"
    device_map: str = "auto"
    local_files_only: bool = True
    license_reviewed: bool = False
    license_review_reference: str = ""
    trust_remote_code: bool = False
    local_path: str | None = None

    def __post_init__(self) -> None:
        _check_common(self)
        _prompt_name(self.verdict_prompt_file, "verdict_prompt_file")
        _sha(self.verdict_prompt_sha256, "verdict_prompt_sha256")
        if self.stop_string is None:
            raise MathModelError("a reasoning judge needs the stop string that ends its reasoning")

    def logical_configuration(self) -> dict[str, Any]:
        """Configuration identity excluding a machine-local checkpoint directory."""
        result = asdict(self)
        result.pop("local_path", None)
        return result


@dataclass(frozen=True)
class GenerativeStepConfig:
    """A generative PRM that judges each step; score = 1 - min_k P(step k correct).

    The solution is cut by the frozen step rule (``blank_line_runs_lf_v1``) and written as
    ``step_format`` (slots ``{k}``, ``{step}``) joined by ``step_joiner`` into the
    ``{solution}`` slot of ``prompt_file``. A generated token that is one of the two verdict
    tokens is a step verdict when the generated text right before it ends with
    ``verdict_pattern``. With one group, the group is the step number the model wrote
    (DeepCritic: "The correctness of Step k is: \\boxed{1}"); with no group, the j-th verdict is
    step j's, as the model's authors read it (ThinkPRM: ``extract_step_labels``).
    ``forced_verdict_prefix`` (slot ``{k}``) and ``verdict_close`` serve ``missing_verdicts`` =
    ``force_remaining``.
    """

    model_id: str
    revision: str
    tokenizer_revision: str
    model_class: str
    prompt_file: str
    prompt_sha256: str
    step_format: str
    step_joiner: str
    verdict_pattern: str
    forced_verdict_prefix: str
    verdict_close: dict[str, str]
    missing_verdicts: str
    stop_string: str | None
    eos_token_ids: list[int]
    positive_token: str
    negative_token: str
    dtype: str
    batch_size: int
    max_input_tokens: int
    max_new_tokens: int
    max_forced_tokens: int
    positive_token_id: int | None = None
    negative_token_id: int | None = None
    generation_prefix: str = ""
    suppress_token_ids: list[int] = field(default_factory=list)
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)
    aggregation: Literal["min"] = "min"
    attn_implementation: str = "sdpa"
    device_map: str = "auto"
    local_files_only: bool = True
    license_reviewed: bool = False
    license_review_reference: str = ""
    trust_remote_code: bool = False
    local_path: str | None = None

    def __post_init__(self) -> None:
        _check_common(self)
        _check_slots(self.step_format, ("k", "step"), "step_format")
        if not isinstance(self.step_joiner, str) or not self.step_joiner:
            raise MathModelError("step_joiner must be a nonempty string")
        _check_slots(self.forced_verdict_prefix, ("k",), "forced_verdict_prefix")
        try:
            pattern = re.compile(self.verdict_pattern)
        except (TypeError, re.error) as exc:
            raise MathModelError("verdict_pattern must be a regular expression") from exc
        if pattern.groups > 1:
            raise MathModelError(
                "verdict_pattern has at most one group (the step number); none means in order"
            )
        if (
            not isinstance(self.verdict_close, dict)
            or set(self.verdict_close) != {"positive", "negative"}
            or any(not isinstance(value, str) for value in self.verdict_close.values())
        ):
            raise MathModelError("verdict_close maps positive and negative to closing text")
        if self.missing_verdicts not in MISSING_RULES:
            raise MathModelError(
                f"missing_verdicts must be one of {MISSING_RULES} (the authors decide; TBD refuses)"
            )
        if self.aggregation != "min":
            raise MathModelError("M.4: the step scores are aggregated by their minimum")

    def logical_configuration(self) -> dict[str, Any]:
        """Configuration identity excluding a machine-local checkpoint directory."""
        result = asdict(self)
        result.pop("local_path", None)
        return result


class _Row:
    """One row of a batch through generation and its forced suffix (a small state machine).

    ``step`` is called once per decoding step with the row's greedy token and its two verdict
    logits at that step, and returns the token to feed (``None`` once the row is done). A
    readout requested by a fed token is taken from the logits of the following step.
    """

    def __init__(self, owner: _GenerativeVerifier, n_steps: int) -> None:
        self.owner = owner
        self.n_steps = n_steps
        self.state = "gen"
        self.generated: list[int] = []
        self.forced: list[int] = []
        self.reason: str | None = None
        self.queue: deque = deque()
        self.pending: dict[str, Any] | None = None
        self.readouts: list[dict[str, Any]] = []

    def step(self, greedy: int, pair: list[float]) -> int | None:
        if self.pending is not None:
            self.readouts.append(
                {**self.pending, "logits": pair, "greedy": greedy, "source": "forced"}
            )
            self.pending = None
        if self.state == "gen":
            return self._generate(greedy, pair)
        if self.state == "force":
            return self._force(pair)
        return None

    def _generate(self, token: int, pair: list[float]) -> int | None:
        owner = self.owner
        if token in owner._eos_ids:  # noqa: SLF001
            self._end("eos")
            return self._force(pair)
        before = sum(r["source"] == "generated" for r in self.readouts)
        found = owner._generated_verdict(self.generated, token, before)  # noqa: SLF001
        if found is not None:
            self.readouts.append(
                {
                    **found,
                    "logits": pair,
                    "greedy": token,
                    "source": "generated",
                    "position": len(self.generated),
                }
            )
        self.generated.append(token)
        if owner._stopped(self.generated):  # noqa: SLF001
            self._end("stop")
        elif len(self.generated) >= owner.config.max_new_tokens:
            self._end("length")
        return token

    def _end(self, reason: str) -> None:
        self.reason = reason
        self.state = "force"
        self.queue = deque(self.owner._forced_plan(self))  # noqa: SLF001

    def _force(self, pair: list[float]) -> int | None:
        while self.queue:
            kind, value, label = self.queue.popleft()
            if kind == "choose":  # the greedy verdict among the two, after a forced readout
                positive, negative = self.owner._verdict_ids  # noqa: SLF001
                chosen = positive if pair[0] >= pair[1] else negative
                close = self.owner._close_ids["positive" if chosen == positive else "negative"]  # noqa: SLF001
                self.queue.extendleft(("feed", token, None) for token in reversed(close))
                self.forced.append(chosen)
                return chosen
            self.forced.append(value)
            if label is not None:
                self.pending = label
            return value
        self.state = "done"
        return None


class _GenerativeVerifier:
    """Loading, prompts, fixed-shape greedy generation and records shared by both forms."""

    form = ""
    implementation = ""
    adapter_version = GEN_ADAPTER_VERSION
    config_class: type = object

    def __init__(
        self,
        config: Any,
        *,
        prompt_dir: Path | None = None,
        tokenizer: Any = None,
        model: Any = None,
        torch_module: Any = None,
    ) -> None:
        if not isinstance(config, self.config_class):
            raise MathModelError(f"{type(self).__name__} requires a {self.config_class.__name__}")
        supplied = [tokenizer is not None, model is not None, torch_module is not None]
        if any(supplied) and not all(supplied):
            raise MathModelError("inject tokenizer, model and torch_module together")
        self.config = config
        self._tokenizer, self._model, self._torch = tokenizer, model, torch_module
        self._injected = all(supplied)
        self._prepared = False
        self._closed = False
        self._prompt_dir = prompt_dir
        self._prompt_text, self.prompt_sha256 = read_prompt(
            config.prompt_file, ("problem", "solution"), prompt_dir
        )
        if self.prompt_sha256 != config.prompt_sha256:
            raise MathModelError(
                f"prompts/{config.prompt_file} differs from the configured prompt_sha256 "
                f"(file {self.prompt_sha256}); review the prompt and pin its hash"
            )
        self._pad_id: int | None = None
        self._special_texts: tuple[str, ...] = ()
        self._versions: dict[str, str | None] = {}
        self._verdict_ids: tuple[int, int] = (-1, -1)
        self._verdict_pieces: tuple[str, str] = ("", "")
        self._eos_ids: frozenset[int] = frozenset()
        self._stop_ids: list[int] = []
        self._prefix_ids: list[int] = []

    @property
    def config_hash(self) -> str:
        return sha256_hex(canonical_json(self.config.logical_configuration()))

    @property
    def cache_length(self) -> int:
        c = self.config
        return c.max_input_tokens + c.max_new_tokens + c.max_forced_tokens

    # -- loading ------------------------------------------------------------------------

    def load(self) -> None:
        """Explicit lazy load from the pinned revision or a manifest-checked local copy."""
        if self._closed:
            raise MathModelError("verifier is closed; construct a new adapter to reload")
        _require_license_review(self.config)
        if self._prepared:
            return
        if self.config.attn_implementation == GROUPED_DECODE_ATTENTION:
            register_attention()
        if not self._injected:
            source = (
                self.config.model_id
                if self.config.local_path is None
                else _verified_local_source(self.config)
            )
            self._torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            importlib.import_module(
                "rafr.verifiers.candidates.torch_backend"
            ).configure_determinism()
            common = {"local_files_only": self.config.local_files_only, "trust_remote_code": False}
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                source,
                revision=self.config.tokenizer_revision,
                **self.config.tokenizer_kwargs,
                **common,
            )
            self._model = getattr(transformers, self.config.model_class).from_pretrained(
                source,
                revision=self.config.revision,
                dtype=getattr(self._torch, self.config.dtype),
                device_map=self.config.device_map,
                attn_implementation=self.config.attn_implementation,
                **common,
            )
        self._model.eval()
        model_config = getattr(self._model, "config", None)
        observed_revision = getattr(model_config, "_commit_hash", None)
        if observed_revision is not None and observed_revision != self.config.revision:
            raise MathModelError("loaded model reports a different checkpoint revision")
        text_config = (
            model_config.get_text_config()
            if hasattr(model_config, "get_text_config")
            else model_config
        )
        maximum = getattr(text_config, "max_position_embeddings", None)
        if type(maximum) is not int or maximum <= 0:
            raise MathModelError("loaded model must declare a positive max_position_embeddings")
        if self.cache_length > maximum:
            raise MathModelError(
                f"L_in + L_gen + forced tokens = {self.cache_length} exceeds the model's "
                f"max_position_embeddings {maximum}"
            )
        actual = str(getattr(self._model, "dtype", "")).removeprefix("torch.")
        if actual != self.config.dtype:
            raise MathModelError(
                f"loaded weights are {actual or 'of unknown dtype'}, configured {self.config.dtype}"
            )
        specials = getattr(self._tokenizer, "all_special_tokens", None) or ()
        self._special_texts = tuple(sorted({str(token) for token in specials if str(token)}))
        pad = getattr(self._tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self._tokenizer, "eos_token_id", None)
        if type(pad) is not int:
            raise MathModelError("the tokenizer has neither a pad nor an eos token id")
        self._pad_id = pad  # masked out; never attended by a real token
        self._prepare_tokens()
        self._versions = _library_versions()
        self._prepared = True

    def _single(self, text: str, configured: int | None) -> tuple[int, str]:
        if configured is None:
            encoded = self._encode(text, add_special_tokens=False)
            if len(encoded) != 1:
                raise MathModelError(
                    f"verdict {text!r} is {len(encoded)} tokens for this tokenizer; configure "
                    "a single-token spelling or its token id"
                )
            configured = encoded[0]
        piece = self._decode([configured])
        if piece.strip() != text.strip():
            raise MathModelError(
                f"verdict token id {configured} decodes to {piece!r}, not {text!r}"
            )
        return configured, piece

    def _prepare_tokens(self) -> None:
        c = self.config
        positive = self._single(c.positive_token, c.positive_token_id)
        negative = self._single(c.negative_token, c.negative_token_id)
        if positive[0] == negative[0]:
            raise MathModelError("the two verdict tokens resolve to the same token id")
        self._verdict_ids = (positive[0], negative[0])
        self._verdict_pieces = (positive[1], negative[1])
        self._eos_ids = frozenset(c.eos_token_ids)
        if set(self._verdict_ids) & self._eos_ids:
            raise MathModelError("a verdict token is also an end-of-sequence token")
        self._stop_ids = (
            [] if c.stop_string is None else self._encode(c.stop_string, add_special_tokens=False)
        )
        if c.stop_string is not None and not self._stop_ids:
            raise MathModelError("stop_string encodes to no tokens")
        self._prefix_ids = (
            self._encode(c.generation_prefix, add_special_tokens=False)
            if c.generation_prefix
            else []
        )

    # -- text ---------------------------------------------------------------------------

    def _encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=add_special_tokens, truncation=False)
        if not isinstance(ids, list) or any(type(token) is not int for token in ids):
            raise MathModelError("tokenizer.encode must return a flat integer token list")
        return ids

    def _decode(self, ids: list[int]) -> str:
        return str(
            self._tokenizer.decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        )

    def _chat_template_hash(self) -> str:
        return sha256_hex(canonical_json(getattr(self._tokenizer, "chat_template", None)))

    def _user_text(self, problem: str, solution: str) -> str:
        raise NotImplementedError

    def _serialize(self, problem: str, solution: str) -> str:
        user = self._user_text(problem, solution)
        rendered = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            **self.config.chat_template_kwargs,
        )
        return _text(rendered, "serialized chat")

    def _n_steps(self, solution: str) -> int:
        return 0

    def _forced_budget(self, n_steps: int) -> int:
        raise NotImplementedError

    def _row(self, problem: str, solution: str) -> tuple[str, list[int], int]:
        for value in (problem, solution):
            if any(token in value for token in self._special_texts):
                raise ModelInputError(
                    "special-token text in problem or solution would be read as a control token"
                )
        n_steps = self._n_steps(solution)
        serialized = self._serialize(problem, solution)
        ids = self._encode(serialized, add_special_tokens=False) + self._prefix_ids
        if not ids or len(ids) > self.config.max_input_tokens:
            raise ModelInputError(
                f"input has {len(ids)} tokens; L_in is {self.config.max_input_tokens}; "
                "truncation is forbidden"
            )
        if self._forced_budget(n_steps) > self.config.max_forced_tokens:
            raise ModelInputError(
                f"the forced suffix may need {self._forced_budget(n_steps)} tokens; "
                f"max_forced_tokens is {self.config.max_forced_tokens}"
            )
        return serialized, ids, n_steps

    # -- generation hooks ---------------------------------------------------------------

    def _stopped(self, generated: list[int]) -> bool:
        """The stop string has just been completed by the newest token."""
        stop = self.config.stop_string
        if stop is None:
            return False
        window = len(self._stop_ids) + 4
        return stop in self._decode(generated[-window:]) and stop not in self._decode(
            generated[-window:-1]
        )

    def _generated_verdict(
        self, generated: list[int], token: int, before: int
    ) -> dict[str, Any] | None:
        return None

    def _forced_plan(self, row: _Row) -> list[tuple[str, Any, Any]]:
        raise NotImplementedError

    # -- fixed-shape generation ---------------------------------------------------------

    def _input_device(self) -> Any:
        device = getattr(self._model, "device", None)
        if hasattr(self._model, "get_input_embeddings"):
            device = self._model.get_input_embeddings().weight.device
        if device is None or str(device) == "meta":
            raise MathModelError("model input device is missing or still meta")
        return device

    def _new_cache(self) -> Any:
        transformers = importlib.import_module("transformers")
        return transformers.StaticCache(config=self._model.config, max_cache_len=self.cache_length)

    def _call(self, **tensors: Any) -> Any:
        try:
            return self._model(**tensors, use_cache=True, logits_to_keep=1)
        except TypeError as exc:
            raise MathModelError(
                "the model's forward must accept a static cache and logits_to_keep"
            ) from exc

    def _greedy(self, output: Any) -> tuple[list[int], list[list[float]]]:
        torch = self._torch
        logits = getattr(output, "logits", None)
        if logits is None:
            raise ModelOutputError("model output has no logits")
        shape = tuple(logits.shape)
        if len(shape) != 3 or shape[0] != self.config.batch_size or shape[1] < 1:
            raise ModelOutputError("logits must have shape [batch_size, positions, vocab]")
        if shape[2] <= max(self._verdict_ids):
            raise ModelOutputError("logits do not cover the verdict token ids")
        last = logits[:, -1, :].float()
        pair = last[:, list(self._verdict_ids)]
        if self.config.suppress_token_ids:
            last = last.clone()
            last[:, self.config.suppress_token_ids] = -math.inf
        greedy = torch.argmax(last, dim=-1)
        return [int(v) for v in greedy.tolist()], [[float(a), float(b)] for a, b in pair.tolist()]

    def _run_batch(self, rows: list[tuple[list[int], int]]) -> list[_Row]:
        """Generate one fixed-shape batch; returns the finished state of each real row."""
        torch = self._torch
        size, length = self.config.batch_size, self.config.max_input_tokens
        total = self.cache_length
        if not 0 < len(rows) <= size:
            raise MathModelError("a batch holds between 1 and batch_size rows")
        filled = rows + [rows[-1]] * (size - len(rows))
        device = self._input_device()
        pads = [length - len(ids) for ids, _ in filled]
        input_ids = [[self._pad_id] * n + ids for n, (ids, _) in zip(pads, filled, strict=True)]
        mask = torch.zeros((size, total), dtype=torch.long, device=device)
        for i, n in enumerate(pads):
            mask[i, n:length] = 1
        positions = [
            [0] * n + list(range(len(ids))) for n, (ids, _) in zip(pads, filled, strict=True)
        ]
        next_position = [len(ids) for ids, _ in filled]
        runs = [_Row(self, n_steps) for _, n_steps in filled]
        cache = self._new_cache()
        with torch.inference_mode():
            output = self._call(
                input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
                attention_mask=mask,
                position_ids=torch.tensor(positions, dtype=torch.long, device=device),
                past_key_values=cache,
                cache_position=torch.arange(length, dtype=torch.long, device=device),
            )
            index = length
            while True:
                greedy, pairs = self._greedy(output)
                feed = [run.step(g, p) for run, g, p in zip(runs, greedy, pairs, strict=True)]
                if all(token is None for token in feed):
                    break
                if index >= total:
                    raise ModelOutputError("generation outgrew the static cache")
                mask[:, index] = 1
                output = self._call(
                    input_ids=torch.tensor(
                        [[self._pad_id if token is None else token] for token in feed],
                        dtype=torch.long,
                        device=device,
                    ),
                    attention_mask=mask,
                    position_ids=torch.tensor(
                        [[p] for p in next_position], dtype=torch.long, device=device
                    ),
                    past_key_values=cache,
                    cache_position=torch.tensor([index], dtype=torch.long, device=device),
                )
                next_position = [p + 1 for p in next_position]
                index += 1
        del cache
        return runs[: len(rows)]

    # -- scoring ------------------------------------------------------------------------

    @staticmethod
    def _pairs(items: Any) -> list[tuple[str, str]]:
        if isinstance(items, (str, bytes, Mapping)) or not isinstance(items, Sequence):
            raise ModelInputError("score_batch takes a sequence of (problem, solution) pairs")
        pairs = []
        for item in items:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ModelInputError("each item must be exactly a (problem, solution) pair")
            pairs.append((_text(item[0], "problem"), _text(item[1], "solution")))
        return pairs

    def score_batch(self, items: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
        """Score (problem, solution) pairs; one result per pair, in input order.

        Every input is built and length-checked before the first forward pass.
        """
        pairs = self._pairs(items)
        if not pairs:
            return []
        self.load()
        prepared = [(p, s, *self._row(p, s)) for p, s in pairs]
        results = []
        for start in range(0, len(prepared), self.config.batch_size):
            chunk = prepared[start : start + self.config.batch_size]
            started = time.monotonic()
            runs = self._run_batch([(ids, n_steps) for *_, ids, n_steps in chunk])
            elapsed = time.monotonic() - started
            for (problem, solution, serialized, ids, _), run in zip(chunk, runs, strict=True):
                results.append(
                    {
                        **self._score_fields(run),
                        **self._generation_fields(run),
                        **self._record(problem, solution, serialized, ids, elapsed),
                    }
                )
        return results

    def score(self, problem: str, solution: str) -> dict[str, Any]:
        """The pipeline's single-state call; runs one full fixed-shape batch."""
        return self.score_batch([(problem, solution)])[0]

    def _score_fields(self, run: _Row) -> dict[str, Any]:
        raise NotImplementedError

    def _generation_fields(self, run: _Row) -> dict[str, Any]:
        return {
            "generated_token_ids": list(run.generated),
            "generated_text": self._decode(run.generated),
            "generated_tokens": len(run.generated),
            "stop_reason": run.reason,
            "truncated": run.reason == "length",
            "forced_token_ids": list(run.forced),
            "readouts": [
                {key: value for key, value in readout.items()} for readout in run.readouts
            ],
        }

    def _record(
        self, problem: str, solution: str, serialized: str, ids: list[int], elapsed: float
    ) -> dict[str, Any]:
        c = self.config
        return {
            "score_direction": SCORE_DIRECTION,
            "input_tokens": len(ids),
            "fixed_shape": {
                "batch_size": c.batch_size,
                "L_in": c.max_input_tokens,
                "L_gen": c.max_new_tokens,
                "L_forced": c.max_forced_tokens,
                "cache_length": self.cache_length,
            },
            "padding_side": "left",
            "pad_token_id": self._pad_id,
            "model_id": c.model_id,
            "revision": c.revision,
            "tokenizer_revision": c.tokenizer_revision,
            "configured_dtype": c.dtype,
            "actual_dtype": str(getattr(self._model, "dtype", "unknown")),
            "device": str(self._input_device()),
            "attn_implementation": c.attn_implementation,
            "implementation": self.implementation,
            "adapter_version": self.adapter_version,
            "config_hash": self.config_hash,
            "local_path": c.local_path,
            "input_hash": sha256_hex(canonical_json({"problem": problem, "solution": solution})),
            "serialized_input_hash": sha256_hex(serialized),
            "serialized_input": serialized,
            "token_ids_hash": sha256_hex(canonical_json(ids)),
            "chat_template_hash": self._chat_template_hash(),
            "library_versions": self._versions,
            "batch_elapsed_seconds": elapsed,
            "execution_mode": "injected_offline" if self._injected else "local_model",
            "license_review_reference": c.license_review_reference,
        }

    # -- card ---------------------------------------------------------------------------

    def _card_specific(self) -> dict[str, Any]:
        raise NotImplementedError

    def verifier_card(self) -> dict[str, Any]:
        """What was loaded and how it generates and scores (M.4, II.10 generative variant)."""
        self.load()
        c = self.config
        return {
            "form": self.form,
            "implementation": self.implementation,
            "adapter_version": self.adapter_version,
            "model_id": c.model_id,
            "revision": c.revision,
            "tokenizer_revision": c.tokenizer_revision,
            "model_class": c.model_class,
            "configuration": c.logical_configuration(),
            "config_hash": self.config_hash,
            "actual_dtype": str(getattr(self._model, "dtype", "unknown")),
            "device": str(self._input_device()),
            "input": {
                "prompt_file": f"prompts/{c.prompt_file}",
                "prompt_sha256": self.prompt_sha256,
                "prompt_text": self._prompt_text,
                "chat_template_kwargs": c.chat_template_kwargs,
                "chat_template_hash": self._chat_template_hash(),
                "tokenizer_kwargs": c.tokenizer_kwargs,
                "generation_prefix": c.generation_prefix,
                "generation_prefix_token_ids": list(self._prefix_ids),
            },
            "generation": {
                "decoding": DECODING,
                "suppress_token_ids": list(c.suppress_token_ids),
                "stop_string": c.stop_string,
                "stop_string_token_ids": list(self._stop_ids),
                "eos_token_ids": sorted(self._eos_ids),
                "L_in": c.max_input_tokens,
                "L_gen": c.max_new_tokens,
                "L_forced": c.max_forced_tokens,
                "truncation": "reasoning cut at L_gen, scored as usual and reported; not dropped",
            },
            "batching": {
                "batch_size": c.batch_size,
                "fixed_shape": True,
                "static_cache_length": self.cache_length,
                "padding_side": "left",
                "pad_token_id": self._pad_id,
                "last_batch": "filled with copies of its last row; their results are dropped",
                "finished_rows": "keep their place until the batch is done; outputs unused",
            },
            "reference_answer_input": False,
            "verdict_tokens": {
                "positive": [self._verdict_ids[0], self._verdict_pieces[0]],
                "negative": [self._verdict_ids[1], self._verdict_pieces[1]],
            },
            "score": {
                "direction": SCORE_DIRECTION,
                "arithmetic": "python_float64_pairwise softmax over the two verdict logits",
                **self._card_specific(),
            },
            "library_versions": self._versions,
            "execution_mode": "injected_offline" if self._injected else "local_model",
            "license_review_reference": c.license_review_reference,
        }

    def close(self) -> None:
        self._model = self._tokenizer = self._torch = None
        self._prepared = False
        self._closed = True


class GenerativeJudgeVerifier(_GenerativeVerifier):
    """A reasoning judge of the whole solution: P(negative) after a fixed verdict prompt."""

    form = "llm_judge"
    implementation = GEN_JUDGE_IMPLEMENTATION
    config_class = GenerativeJudgeConfig

    def __init__(self, config: GenerativeJudgeConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._verdict_text, self.verdict_prompt_sha256 = read_prompt(
            config.verdict_prompt_file, (), self._prompt_dir
        )
        if self.verdict_prompt_sha256 != config.verdict_prompt_sha256:
            raise MathModelError(
                f"prompts/{config.verdict_prompt_file} differs from the configured "
                f"verdict_prompt_sha256 (file {self.verdict_prompt_sha256})"
            )
        self._verdict_prompt_ids: list[int] = []

    def _prepare_tokens(self) -> None:
        super()._prepare_tokens()
        self._verdict_prompt_ids = self._encode(self._verdict_text, add_special_tokens=False)
        if not self._verdict_prompt_ids:
            raise MathModelError("the verdict prompt encodes to no tokens")

    def _user_text(self, problem: str, solution: str) -> str:
        return fill_slots(self._prompt_text, {"problem": problem, "solution": solution})

    def _forced_budget(self, n_steps: int) -> int:
        return len(self._stop_ids) + len(self._verdict_prompt_ids)

    def _forced_plan(self, row: _Row) -> list[tuple[str, Any, Any]]:
        ids = ([] if row.reason == "stop" else list(self._stop_ids)) + self._verdict_prompt_ids
        plan = [("feed", token, None) for token in ids]
        plan[-1] = ("feed", ids[-1], {"kind": "verdict"})
        return plan

    def _score_fields(self, run: _Row) -> dict[str, Any]:
        verdicts = [r for r in run.readouts if r.get("kind") == "verdict"]
        if len(verdicts) != 1:
            raise ModelOutputError("expected exactly one verdict readout per input")
        positive, negative = verdicts[0]["logits"]
        p_negative = _probabilities([[positive, negative]])[0]
        return {
            "error_score": _unit_score(p_negative),
            "p_negative": p_negative,
            "verdict_logits_positive_negative": [positive, negative],
            "verdict_token_ids_positive_negative": list(self._verdict_ids),
            "score_definition": "p_negative_softmax_over_the_two_verdict_logits_after_reasoning",
            "softmax_arithmetic": "python_float64_pairwise",
            "prompt_sha256": self.prompt_sha256,
            "verdict_prompt_sha256": self.verdict_prompt_sha256,
        }

    def _card_specific(self) -> dict[str, Any]:
        return {
            "definition": "P(negative): softmax over the two verdict token logits at the position "
            "after the verdict prompt; the verdict prompt follows the reasoning (stop string "
            "first when the reasoning did not end with it)",
            "verdict_prompt_file": f"prompts/{self.config.verdict_prompt_file}",
            "verdict_prompt_sha256": self.verdict_prompt_sha256,
            "verdict_prompt_text": self._verdict_text,
            "verdict_prompt_token_ids": list(self._verdict_prompt_ids),
        }


class GenerativeStepVerifier(_GenerativeVerifier):
    """A generative PRM: 1 - min over the steps of P(step correct) at its verdict position."""

    form = "prm"
    implementation = GEN_STEP_IMPLEMENTATION
    config_class = GenerativeStepConfig
    _LOOKBACK = 48  # generated tokens decoded to match verdict_pattern

    def __init__(self, config: GenerativeStepConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._pattern = re.compile(f"(?:{config.verdict_pattern})$")
        self._close_ids: dict[str, list[int]] = {}
        self._prefix_cache: dict[int, list[int]] = {}

    def _prepare_tokens(self) -> None:
        super()._prepare_tokens()
        self._close_ids = {
            side: (self._encode(text, add_special_tokens=False) if text else [])
            for side, text in self.config.verdict_close.items()
        }

    def _steps(self, solution: str) -> list[str]:
        return split_solution_steps(solution)

    def _n_steps(self, solution: str) -> int:
        return len(self._steps(solution))

    def _user_text(self, problem: str, solution: str) -> str:
        written = self.config.step_joiner.join(
            fill_slots(self.config.step_format, {"k": str(k), "step": step})
            for k, step in enumerate(self._steps(solution), start=1)
        )
        return fill_slots(self._prompt_text, {"problem": problem, "solution": written})

    def _forced_prefix(self, k: int) -> list[int]:
        if k not in self._prefix_cache:
            text = fill_slots(self.config.forced_verdict_prefix, {"k": str(k)})
            self._prefix_cache[k] = self._encode(text, add_special_tokens=False)
        return self._prefix_cache[k]

    def _forced_budget(self, n_steps: int) -> int:
        if self.config.missing_verdicts == "count_incorrect":
            return 0
        close = max(len(ids) for ids in self._close_ids.values())
        return sum(len(self._forced_prefix(k)) + 1 + close for k in range(1, n_steps + 1))

    def _generated_verdict(
        self, generated: list[int], token: int, before: int
    ) -> dict[str, Any] | None:
        """A step verdict: its step number as written, or (no group) the next in order."""
        if token not in self._verdict_ids:
            return None
        match = self._pattern.search(self._decode(generated[-self._LOOKBACK :]))
        if match is None:
            return None
        step = int(match.group(1)) if self._pattern.groups else before + 1
        return {"kind": "step", "step": step, "token": token}

    def _judged(self, run: _Row) -> dict[int, dict[str, Any]]:
        """The last generated verdict of each step in 1..n."""
        judged: dict[int, dict[str, Any]] = {}
        for readout in run.readouts:
            if readout["source"] == "generated" and 1 <= readout["step"] <= run.n_steps:
                judged[readout["step"]] = readout
        return judged

    def _missing(self, run: _Row) -> list[int]:
        """Steps without a generated verdict, except those after the first step the model
        judged incorrect (both step models stop there by design)."""
        judged = self._judged(run)
        negatives = [k for k, r in judged.items() if r["token"] == self._verdict_ids[1]]
        last = min(negatives) if negatives else run.n_steps
        return [k for k in range(1, last + 1) if k not in judged]

    def _forced_plan(self, row: _Row) -> list[tuple[str, Any, Any]]:
        rule = self.config.missing_verdicts
        missing = self._missing(row)
        if rule == "count_incorrect" or not missing:
            return []
        if rule == "judged_only" and self._judged(row):
            return []
        plan: list[tuple[str, Any, Any]] = []
        for k in missing:
            ids = self._forced_prefix(k)
            plan += [("feed", token, None) for token in ids[:-1]]
            plan.append(("feed", ids[-1], {"kind": "step", "step": k}))
            plan.append(("choose", None, None))
        return plan

    def _score_fields(self, run: _Row) -> dict[str, Any]:
        judged = self._judged(run)
        missing = self._missing(run)
        forced = {
            r["step"]: r
            for r in run.readouts
            if r["source"] == "forced" and r.get("kind") == "step"
        }
        steps: list[float | None] = [None] * run.n_steps
        sources: list[str | None] = [None] * run.n_steps
        for k in range(1, run.n_steps + 1):
            readout = judged.get(k) or forced.get(k)
            if readout is not None:
                positive, negative = readout["logits"]
                steps[k - 1] = _probabilities([[negative, positive]])[0]  # P(positive)
                sources[k - 1] = readout["source"]
            elif k in missing and self.config.missing_verdicts == "count_incorrect":
                steps[k - 1] = 0.0
                sources[k - 1] = "missing_counted_incorrect"
        present = [value for value in steps if value is not None]
        if not present:
            raise ModelOutputError("no step verdict was generated or forced")
        minimum = min(present)
        return {
            "error_score": _unit_score(1.0 - minimum),
            "step_scores": steps,
            "step_score_sources": sources,
            "n_steps": run.n_steps,
            "n_steps_judged_generated": len(judged),
            "missing_steps": missing,
            "missing_verdicts_rule": self.config.missing_verdicts,
            "min_step_score": minimum,
            "verdict_token_ids_positive_negative": list(self._verdict_ids),
            "score_definition": "one_minus_min_over_steps_of_p_positive_at_verdict_positions",
            "softmax_arithmetic": "python_float64_pairwise",
            "prompt_sha256": self.prompt_sha256,
        }

    def _card_specific(self) -> dict[str, Any]:
        c = self.config
        return {
            "definition": "1 - min_k P(step k correct); P = softmax over the two verdict token "
            "logits at step k's verdict position",
            "step_rule": "blank_line_runs_lf_v1",
            "step_format": c.step_format,
            "step_joiner": c.step_joiner,
            "verdict_position": f"a generated verdict token after text matching "
            f"{c.verdict_pattern!r} (the last {self._LOOKBACK} generated tokens decoded); "
            + (
                "its step is the number the pattern's group captured"
                if self._pattern.groups
                else "the j-th such verdict is step j's (in order)"
            )
            + "; the last verdict of a step counts; a step number outside 1..n is ignored",
            "missing_verdicts": c.missing_verdicts,
            "missing_definition": "steps without a generated verdict, up to the first step "
            "judged incorrect (steps after it are skipped by design)",
            "forced_verdict_prefix": c.forced_verdict_prefix,
            "verdict_close": c.verdict_close,
            "aggregation": c.aggregation,
        }


#: The generative adapters per implementation; build_verifier reads it.
GENERATIVE_ADAPTERS = {
    GEN_JUDGE_IMPLEMENTATION: GenerativeJudgeVerifier,
    GEN_STEP_IMPLEMENTATION: GenerativeStepVerifier,
}

__all__ = [
    "GEN_JUDGE_IMPLEMENTATION",
    "GEN_STEP_IMPLEMENTATION",
    "GENERATIVE_ADAPTERS",
    "GenerativeJudgeConfig",
    "GenerativeJudgeVerifier",
    "GenerativeStepConfig",
    "GenerativeStepVerifier",
    "MISSING_RULES",
    "read_prompt",
]
