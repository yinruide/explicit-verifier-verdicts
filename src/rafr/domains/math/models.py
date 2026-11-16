"""Math learned-verifier boundaries (paper Sec. 5, App. D).

Three forms, one entry point (:func:`build_verifier`), one direction: every adapter
returns an ``error_score`` in [0, 1], higher = more likely wrong.

* PRM: author-specific adapters for the two published PRMs.
* LLM judge: any Hugging Face instruct model asked, through ``prompts/llm_judge.txt``,
  whether the final answer is correct; score = P(No) after a softmax over the Yes and
  No token logits at the verdict position.
* ORM: any Hugging Face sequence-regression / reward model over problem + full
  solution; score = 1 - sigmoid(scalar logit).

The judge and ORM adapters assume nothing about the model: model ID, input format,
verdict tokens, dtype and batch size all come from the reviewed configuration
(the authors pick the candidates in the step 2 probe). Their inference follows the detox
guard-LM scorer (``rafr.verifiers.candidates.torch_probe``): one forward pass, no
sampling, left padding with rebuilt position ids, and every batch padded to a fixed
shape -- ``batch_size`` rows (the last batch filled with copies of its last row, whose
scores are dropped) by ``max_input_tokens`` columns -- so a score never depends on its
neighbours. Inputs are never truncated. No adapter receives a reference answer: the
only inputs are the problem and the solution text.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from rafr.llm.client import canonical_json, fill_slots, sha256_hex

ADAPTER_VERSION = "math-models-v2"
STEP_RULE = "blank_line_runs_lf_v1"
# paper Sec. 5: the PRM step segmentation, frozen at selection and written into the verifier_card.
STEP_RULE_TEXT = (
    "CRLF and CR line endings become LF and the solution is stripped of outer whitespace; it "
    "is cut at every run of one or more blank lines (a line holding only spaces or tabs is "
    "blank); each piece, stripped, is one step, in order. Nothing is rewritten, merged or cut "
    "further (not by sentence, formula or numbered list). A problem or solution that contains "
    "the scoring marker itself is refused, as is an input over the token limit (no truncation)."
)
QWEN_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
MODEL_REPOSITORIES = {
    "qwen_prm": "Qwen/Qwen2.5-Math-PRM-7B",
    "math_shepherd": "peiyi9979/math-shepherd-mistral-7b-prm",
}
PUBLISHED_REVISIONS = {
    "qwen_prm": "0610740060112df12585d00a1c5f4624d2f59051",
    "math_shepherd": "45dc0a3c9ec699b645085c098ed38dc99fba4617",
}
#: The public attributes transformers 4.x's base config set in ``__init__`` (generation
#: defaults and private bookkeeping aside), with their defaults. transformers 5 dropped several
#: of them, and repository code written for 4.x reads them: Qwen2.5-Math-PRM-7B's
#: ``modeling_qwen2_rm.py`` (``transformers_version`` 4.37.0) reads ``config.pad_token_id``
#: while building the model. A model that runs repository code gets the default of every one
#: its loaded config lacks; nothing it has is changed, and what was filled is in the
#: verifier_card.
LEGACY_CONFIG_DEFAULTS_SOURCE = "transformers v4.37.0 configuration_utils.PretrainedConfig.__init__"
LEGACY_CONFIG_DEFAULTS: dict[str, Any] = {
    "return_dict": True,
    "output_hidden_states": False,
    "output_attentions": False,
    "torchscript": False,
    "torch_dtype": None,
    "use_bfloat16": False,
    "tf_legacy_loss": False,
    "pruned_heads": {},
    "tie_word_embeddings": True,
    "chunk_size_feed_forward": 0,
    "is_encoder_decoder": False,
    "is_decoder": False,
    "cross_attention_hidden_size": None,
    "add_cross_attention": False,
    "tie_encoder_decoder": False,
    "architectures": None,
    "finetuning_task": None,
    "id2label": None,
    "label2id": None,
    "num_labels": 2,
    "tokenizer_class": None,
    "prefix": None,
    "bos_token_id": None,
    "pad_token_id": None,
    "eos_token_id": None,
    "sep_token_id": None,
    "decoder_start_token_id": None,
    "task_specific_params": None,
    "problem_type": None,
}


def fill_legacy_config_defaults(config: Any) -> dict[str, Any]:
    """Give ``config`` each :data:`LEGACY_CONFIG_DEFAULTS` attribute it lacks; return those."""
    filled = {}
    for name, value in LEGACY_CONFIG_DEFAULTS.items():
        if not hasattr(config, name):
            setattr(config, name, copy.deepcopy(value))
            filled[name] = copy.deepcopy(value)
    return filled


#: The transformers 4.x rotary-embedding pattern in repository code: buffers built in
#: ``__init__`` from these attributes and never saved (``persistent=False``).
LEGACY_ROTARY_ATTRIBUTES = (
    "inv_freq",
    "_set_cos_sin_cache",
    "dim",
    "base",
    "max_position_embeddings",
)


def rebuild_legacy_rotary_buffers(model: Any, dtype: Any) -> list[str]:
    """Rebuild the unsaved rotary buffers of repository code written for transformers 4.x.

    transformers 5 builds the model on the meta device and fills only saved tensors, so such
    buffers come out as zeros (Qwen2.5-Math-PRM-7B: ``inv_freq`` 0 in all 28 layers, no
    position information). Each module of that pattern is built again by its own class on the
    CPU, as 4.x built it, with the cos / sin cache in the model's ``dtype`` (4.x set the default
    dtype to it while building the model); its buffers then replace the module's on the
    module's device. Returns the rebuilt module names.
    """
    named_modules = getattr(model, "named_modules", None)
    if named_modules is None:
        return []
    rebuilt = []
    for name, module in named_modules():
        if not all(hasattr(module, attribute) for attribute in LEGACY_ROTARY_ATTRIBUTES):
            continue
        device = module.inv_freq.device
        fresh = type(module)(
            module.dim, max_position_embeddings=module.max_position_embeddings, base=module.base
        )
        fresh._set_cos_sin_cache(
            seq_len=module.max_position_embeddings, device=fresh.inv_freq.device, dtype=dtype
        )
        for buffer_name, value in fresh.named_buffers(recurse=False):
            module.register_buffer(buffer_name, value.to(device), persistent=False)
        module.max_seq_len_cached = fresh.max_seq_len_cached
        rebuilt.append(name)
    return rebuilt


class MathModelError(RuntimeError):
    """A model boundary is invalid; never impute a score."""


class ModelAccessNotReviewedError(MathModelError):
    pass


class ModelInputError(MathModelError):
    pass


class ModelOutputError(MathModelError):
    pass


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelInputError(f"{name} must be a nonempty string")
    return value


def _require_license_review(config: Any) -> None:
    if not config.license_reviewed or not config.license_review_reference.strip():
        raise ModelAccessNotReviewedError(
            "checkpoint license/use review is required; document the decision before loading"
        )


def _verified_local_source(config: Any) -> str:
    """A local checkpoint directory whose manifest pins every runtime file by SHA256."""
    root = Path(config.local_path).expanduser().resolve()
    manifest = root / "rafr_model_manifest.json"
    if not manifest.is_file():
        raise MathModelError("a local model directory requires rafr_model_manifest.json")
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MathModelError("local model manifest is unreadable or invalid JSON") from exc
    if not isinstance(record, dict):
        raise MathModelError("local model manifest must be a JSON object")
    expected = {
        "model_id": config.model_id,
        "revision": config.revision,
        "tokenizer_revision": config.tokenizer_revision,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise MathModelError("local model manifest does not match configured source revisions")
    files = record.get("files")
    if not isinstance(files, dict) or not files:
        raise MathModelError("local model manifest requires a nonempty file/hash mapping")
    for name, digest in files.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise MathModelError("local model manifest requires relative names and SHA256 hashes")
        if Path(name).is_absolute():
            raise MathModelError("local model manifest file names must be relative")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise MathModelError("local model manifest contains an absent or escaping path")
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(block)
        if hasher.hexdigest() != digest:
            raise MathModelError("local model file hash differs from the manifest")
    # Every file capable of defining loaded code, tokenization, chat template or weights
    # must be covered (transformers 5 writes the chat template as chat_template.jinja). The
    # Hub's own bookkeeping of a local_dir download (.cache/huggingface/, e.g. the revision's
    # file list trees/<revision>.json) is never loaded and is outside the manifest.
    uncovered = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != manifest.name
        and path.relative_to(root).parts[:2] != (".cache", "huggingface")
        and path.suffix in {".json", ".py", ".model", ".safetensors", ".bin", ".txt", ".jinja"}
        and path.relative_to(root).as_posix() not in files
    ]
    if uncovered:
        raise MathModelError("local model manifest does not cover all runtime model files")
    return str(root)


def _library_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("torch", "transformers", "tokenizers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


@dataclass(frozen=True)
class PRMConfig:
    kind: Literal["qwen_prm", "math_shepherd"]
    model_id: str
    revision: str
    tokenizer_revision: str
    aggregation: Literal["min", "product", "mean", "last"] = "min"
    max_tokens: int = 4096
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    device_map: str = "auto"
    local_files_only: bool = True
    license_reviewed: bool = False
    license_review_reference: str = ""
    trust_remote_code: bool = False
    local_path: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in MODEL_REPOSITORIES or self.model_id != MODEL_REPOSITORIES[self.kind]:
            raise MathModelError("adapter kind and official model_id must match")
        for name in ("revision", "tokenizer_revision"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(
                r"[0-9a-f]{40}", getattr(self, name)
            ):
                raise MathModelError(f"{name} must be an immutable 40-character commit SHA")
        if self.aggregation not in ("min", "product", "mean", "last"):
            raise MathModelError("unsupported PRM aggregation")
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 4096:
            raise MathModelError("these published PRM adapters support max_tokens from 1 to 4096")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise MathModelError("unsupported dtype; quantization is not silently enabled")
        for name in ("local_files_only", "license_reviewed", "trust_remote_code"):
            if type(getattr(self, name)) is not bool:
                raise MathModelError(f"{name} must be boolean")
        _text(self.device_map, "device_map")
        if not isinstance(self.license_review_reference, str):
            raise MathModelError("license_review_reference must be a string")
        if self.local_path is not None:
            _text(self.local_path, "local_path")

    def configuration(self) -> dict[str, Any]:
        """Serializable requested configuration; it does not assert the model was loaded."""
        return {**asdict(self), "adapter_version": ADAPTER_VERSION, "step_rule": STEP_RULE}

    def logical_configuration(self) -> dict[str, Any]:
        """Configuration identity excluding a machine-local checkpoint directory."""
        result = asdict(self)
        result.pop("local_path", None)
        return result


def split_solution_steps(solution: str) -> list[str]:
    text = _text(solution, "solution").replace("\r\n", "\n").replace("\r", "\n").strip()
    steps = [step.strip() for step in re.split(r"\n[ \t]*\n(?:[ \t]*\n)*", text)]
    if not steps or any(not step for step in steps):
        raise ModelInputError("empty solution step")
    return steps


def aggregate_steps(scores: list[float], aggregation: str) -> float:
    if not scores or any(
        isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1
        for value in scores
    ):
        raise ModelOutputError("step scores must be nonempty, finite, and within [0, 1]")
    if aggregation == "min":
        return min(scores)
    if aggregation == "product":
        return math.prod(scores)
    if aggregation == "mean":
        return math.fsum(scores) / len(scores)
    if aggregation == "last":
        return scores[-1]
    raise MathModelError("unsupported aggregation")


def _probabilities(pairs: list[list[float]]) -> list[float]:
    result = []
    for pair in pairs:
        if len(pair) != 2 or any(not math.isfinite(float(value)) for value in pair):
            raise ModelOutputError("each selected pair requires two finite logits")
        negative, positive = (float(value) for value in pair)
        maximum = max(negative, positive)
        a, b = math.exp(negative - maximum), math.exp(positive - maximum)
        result.append(b / (a + b))
    return result


class PRMVerifier:
    """Author-specific real scoring adapter, with injected tensor boundaries for tests."""

    def __init__(
        self,
        config: PRMConfig,
        *,
        tokenizer: Any = None,
        model: Any = None,
        torch_module: Any = None,
    ) -> None:
        supplied = [tokenizer is not None, model is not None, torch_module is not None]
        if any(supplied) and not all(supplied):
            raise MathModelError("inject tokenizer, model and torch_module together")
        self.config = config
        self._tokenizer, self._model, self._torch = tokenizer, model, torch_module
        self._injected = all(supplied)
        self._prepared = False
        self._closed = False
        self._marker_id: int | None = None
        self._label_ids: tuple[int, int] | None = None
        self._max_tokens = config.max_tokens
        self._legacy_defaults: dict[str, Any] = {}
        self._legacy_rebuilt: list[str] = []

    def _license_gate(self) -> None:
        _require_license_review(self.config)

    def _source(self) -> str:
        if self.config.local_path is None:
            return self.config.model_id
        return _verified_local_source(self.config)

    def load(self) -> None:
        """Explicit lazy load. Defaults to pinned local HF cache; never selects a new model."""
        if self._closed:
            raise MathModelError("verifier is closed; construct a new adapter to reload")
        self._license_gate()
        if self._prepared:
            return
        if not self._injected:
            if self.config.kind == "qwen_prm" and not self.config.trust_remote_code:
                raise ModelAccessNotReviewedError(
                    "Qwen PRM custom code must be explicitly reviewed"
                )
            source = self._source()
            self._torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            common = {
                "local_files_only": self.config.local_files_only,
                "trust_remote_code": self.config.trust_remote_code,
            }
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                source,
                revision=self.config.tokenizer_revision,
                code_revision=self.config.tokenizer_revision,
                **common,
            )
            factory = (
                transformers.AutoModel
                if self.config.kind == "qwen_prm"
                else transformers.AutoModelForCausalLM
            )
            loaded = {}
            if self.config.trust_remote_code:  # repository code written for transformers 4.x
                model_config = transformers.AutoConfig.from_pretrained(
                    source,
                    revision=self.config.revision,
                    code_revision=self.config.revision,
                    **common,
                )
                self._legacy_defaults = fill_legacy_config_defaults(model_config)
                loaded["config"] = model_config
            self._model = factory.from_pretrained(
                source,
                revision=self.config.revision,
                code_revision=self.config.revision,
                torch_dtype=getattr(self._torch, self.config.dtype),
                device_map=self.config.device_map,
                **loaded,
                **common,
            )
            if self.config.trust_remote_code:
                self._legacy_rebuilt = rebuild_legacy_rotary_buffers(
                    self._model, getattr(self._torch, self.config.dtype)
                )
        self._model.eval()
        model_config = getattr(self._model, "config", None)
        observed_revision = getattr(model_config, "_commit_hash", None)
        if observed_revision is not None and observed_revision != self.config.revision:
            raise MathModelError("loaded model reports a different checkpoint revision")
        maximum = getattr(model_config, "max_position_embeddings", None)
        if type(maximum) is not int or maximum <= 0:
            raise MathModelError("loaded model must declare a positive max_position_embeddings")
        self._max_tokens = min(self.config.max_tokens, maximum)
        marker = "<extra_0>" if self.config.kind == "qwen_prm" else "ки"
        self._marker_id = self._single_content_token(marker)
        if self.config.kind == "math_shepherd":
            pair = self._encode("+ -", add_special_tokens=False)
            if len(pair) != 2 or pair[0] == pair[1]:
                raise MathModelError("Math-Shepherd '+ -' must tokenize to two distinct labels")
            if [self._decode([item]).strip() for item in pair] != ["+", "-"]:
                raise MathModelError("Math-Shepherd label tokens do not decode to '+' and '-'")
            if pair != [self._single_content_token("+"), self._single_content_token("-")]:
                raise MathModelError("Math-Shepherd label tokenization changes in label context")
            self._label_ids = (pair[1], pair[0])  # negative, positive
        self._prepared = True

    def _encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=add_special_tokens, truncation=False)
        if not isinstance(ids, list) or any(type(token) is not int for token in ids):
            raise MathModelError("tokenizer.encode must return a flat integer token list")
        return ids

    def _decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def _single_content_token(self, value: str) -> int:
        ids = self._encode(value, add_special_tokens=False)
        # SentencePiece may prepend a whitespace-only token. Unlike blindly taking [-1],
        # verify the scoring token itself represents the WHOLE marker/label.
        if not ids or self._decode([ids[-1]]).strip() != value:
            raise MathModelError(f"the fixed tokenizer has no whole scoring token for {value!r}")
        if self._decode(ids[:-1]).strip():
            raise MathModelError(f"the marker/label {value!r} spans multiple content tokens")
        if self.config.kind == "qwen_prm" and value == "<extra_0>" and len(ids) != 1:
            raise MathModelError("Qwen's <extra_0> must be exactly one token")
        return ids[-1]

    def _serialize(self, problem: str, steps: list[str]) -> str:
        if self.config.kind == "qwen_prm":
            messages = [
                {"role": "system", "content": QWEN_SYSTEM},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
            ]
            value = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return _text(value, "serialized chat")
        body = "\n".join(f"Step {index}: {step} ки" for index, step in enumerate(steps, 1))
        return f"{problem} {body}"

    def score(self, problem: str, solution: str) -> dict[str, Any]:
        _text(problem, "problem")
        _text(solution, "solution")
        # Reject literal markers even in the problem: their positions would contaminate
        # the mask and make a fake step look as though it had been scored.
        marker = "<extra_0>" if self.config.kind == "qwen_prm" else "ки"
        if marker in problem or marker in solution:
            raise ModelInputError("literal scoring marker in problem or solution")
        steps = split_solution_steps(solution)
        self.load()
        started = time.monotonic()
        serialized = self._serialize(problem, steps)
        ids = self._encode(serialized, add_special_tokens=True)
        if not ids or len(ids) > self._max_tokens:
            raise ModelInputError(
                f"input has {len(ids)} tokens; limit is {self._max_tokens}; truncation is forbidden"
            )
        positions = [index for index, token in enumerate(ids) if token == self._marker_id]
        if len(positions) != len(steps):
            raise ModelInputError("observed marker count differs from the number of solution steps")
        device = getattr(self._model, "device", None)
        if hasattr(self._model, "get_input_embeddings"):
            device = self._model.get_input_embeddings().weight.device
        if device is None or str(device) == "meta":
            raise MathModelError("model input device is missing or still meta")
        inputs = self._torch.tensor([ids], dtype=self._torch.long, device=device)
        with self._torch.inference_mode():
            output = self._model(
                input_ids=inputs,
                attention_mask=self._torch.ones_like(inputs),
                use_cache=False,
            )
        logits = getattr(output, "logits", None)
        if logits is None:
            try:
                logits = output[0]
            except (TypeError, IndexError, KeyError) as exc:
                raise ModelOutputError("PRM output has no logits") from exc
        shape = tuple(logits.shape)
        if len(shape) != 3 or shape[:2] != (1, len(ids)):
            raise ModelOutputError("PRM logits shape must match [1, input_tokens, labels]")
        if self.config.kind == "qwen_prm":
            if shape[2] != 2:
                raise ModelOutputError("Qwen PRM must expose its two-class scoring head")
            selected = logits[0, positions, :]
        else:
            assert self._label_ids is not None
            if shape[2] <= max(self._label_ids):
                raise ModelOutputError("Math-Shepherd logits do not cover the checked label IDs")
            selected = logits[0, positions, :][:, list(self._label_ids)]
        if hasattr(selected, "detach"):
            selected = selected.detach().float().cpu()
        pairs = selected.tolist()
        if len(pairs) != len(steps):
            raise ModelOutputError("selected score count differs from the number of steps")
        step_scores = _probabilities(pairs)
        aggregate = aggregate_steps(step_scores, self.config.aggregation)
        template = getattr(self._tokenizer, "chat_template", None)
        versions = _library_versions()
        return {
            "error_score": 1.0 - aggregate,
            "step_scores": step_scores,
            "aggregate_score": aggregate,
            "aggregation": self.config.aggregation,
            "score_definition": "positive_class_pair_softmax_then_aggregate",
            "score_direction": "higher_error_score_means_more_problematic",
            "transform": "one_minus_aggregate_not_probability_calibration",
            "softmax_arithmetic": "python_float64_pairwise",
            "selected_step_logits_negative_positive": pairs,
            "step_rule": STEP_RULE,
            "steps": steps,
            "n_steps": len(steps),
            "scoring_token_id": self._marker_id,
            "label_token_ids_negative_positive": self._label_ids,
            "scoring_token_positions": positions,
            "input_tokens": len(ids),
            "effective_max_tokens": self._max_tokens,
            "truncated": False,
            "model_id": self.config.model_id,
            "revision": self.config.revision,
            "tokenizer_revision": self.config.tokenizer_revision,
            "configured_dtype": self.config.dtype,
            "actual_dtype": str(getattr(self._model, "dtype", "unknown")),
            "device": str(device),
            "adapter_version": ADAPTER_VERSION,
            "config_hash": sha256_hex(canonical_json(self.config.logical_configuration())),
            "local_path": self.config.local_path,
            "input_hash": sha256_hex(canonical_json({"problem": problem, "solution": solution})),
            "serialized_input_hash": sha256_hex(serialized),
            "serialized_input": serialized,
            "token_ids_hash": sha256_hex(canonical_json(ids)),
            "chat_template_hash": sha256_hex(canonical_json(template)),
            "library_versions": versions,
            "elapsed_seconds": time.monotonic() - started,
            "execution_mode": "injected_offline" if self._injected else "local_model",
            "license_review_reference": self.config.license_review_reference,
        }

    def verifier_card(self) -> dict[str, Any]:
        """What was loaded and how it scores. paper Sec. 5: the step segmentation rule and the
        aggregation are frozen at selection and written into the verifier_card."""
        self.load()
        qwen = self.config.kind == "qwen_prm"
        return {
            "form": "prm",
            "implementation": self.config.kind,
            "adapter_version": ADAPTER_VERSION,
            "model_id": self.config.model_id,
            "revision": self.config.revision,
            "tokenizer_revision": self.config.tokenizer_revision,
            "configuration": self.config.logical_configuration(),
            "config_hash": sha256_hex(canonical_json(self.config.logical_configuration())),
            "actual_dtype": str(getattr(self._model, "dtype", "unknown")),
            "steps": {
                "rule": STEP_RULE,
                "text": STEP_RULE_TEXT,
                "marker": "<extra_0>" if qwen else "ки",
                "marker_token_id": self._marker_id,
                "serialization": (
                    "chat template of the repository: system = the model card's prompt, user = "
                    "the problem, assistant = the steps joined by <extra_0> plus one after the "
                    "last step"
                    if qwen
                    else "'<problem> Step 1: <step 1> ки\\nStep 2: <step 2> ки...' (model card)"
                ),
                "system_prompt": QWEN_SYSTEM if qwen else None,
            },
            "aggregation": self.config.aggregation,
            "score": {
                "direction": SCORE_DIRECTION,
                "definition": "1 - aggregate (configured above) of the per-step scores",
                "step_score": (
                    "softmax over the two-class head at each marker; positive class"
                    if qwen
                    else "softmax over the '+' and '-' token logits at each marker; '+'"
                ),
                "label_token_ids_negative_positive": self._label_ids,
                "arithmetic": "python_float64_pairwise",
            },
            "max_tokens": self._max_tokens,
            "batching": "one state per forward pass, no padding",
            "truncation": "never; an input over max_tokens is refused",
            "reference_answer_input": False,
            "chat_template_hash": sha256_hex(
                canonical_json(getattr(self._tokenizer, "chat_template", None))
            ),
            "library_versions": _library_versions(),
            "transformers_compatibility": {
                "applies_to": "models that run repository code (trust_remote_code)",
                "legacy_config_defaults_filled": self._legacy_defaults,
                "defaults_source": LEGACY_CONFIG_DEFAULTS_SOURCE,
                "rotary_buffers_rebuilt": {
                    "modules": len(self._legacy_rebuilt),
                    "names": self._legacy_rebuilt,
                    "how": "the module's own class on the CPU (inv_freq float32, cos / sin "
                    "cache in the model dtype), as transformers 4.x built them",
                },
            },
            "execution_mode": "injected_offline" if self._injected else "local_model",
            "license_review_reference": self.config.license_review_reference,
        }

    def close(self) -> None:
        self._model = self._tokenizer = self._torch = None
        self._prepared = False
        self._closed = True


# ---------------------------------------------------------------------------------------
# Hugging Face backends (paper Sec. 5). Nothing below names a model: every model-specific choice
# is a configuration value.
# ---------------------------------------------------------------------------------------

PROMPT_DIR = Path(__file__).with_name("prompts")
JUDGE_PROMPT_FILE = "llm_judge.txt"
JUDGE_IMPLEMENTATION = "hf_yes_no_judge"
ORM_IMPLEMENTATION = "hf_sequence_regression"
JUDGE_ADAPTER_VERSION = "math-llm-judge-v1"
ORM_ADAPTER_VERSION = "math-orm-v1"
INPUT_FORMATS = ("chat_template", "raw")
DTYPES = ("float32", "float16", "bfloat16")
SCORE_DIRECTION = "higher_error_score_means_more_problematic"
_SLOT = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _check_slots(template: str, slots: tuple[str, ...], name: str) -> None:
    """Exactly these slots, once each. Any other slot (a reference answer, say) is refused."""
    found = _SLOT.findall(template)
    if sorted(found) != sorted(slots):
        raise MathModelError(
            f"{name} must contain the slots {list(slots)} once each, found {found}"
        )


def read_judge_prompt(
    name: str = JUDGE_PROMPT_FILE, directory: Path | None = None
) -> tuple[str, str]:
    """The judge prompt (LF line endings, without the editor's final newline) and its SHA256.

    The hash is over the LF text, so a CRLF Windows checkout and the pod agree on it.
    """
    path = Path(PROMPT_DIR if directory is None else directory) / name
    if not path.is_file():
        raise MathModelError(f"judge prompt file {path} not found")
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if text.endswith("\n"):
        text = text[:-1]
    if not text.strip() or "PLACEHOLDER" in text:
        raise MathModelError(f"judge prompt file {path} is empty or still a PLACEHOLDER")
    _check_slots(text, ("problem", "solution"), f"judge prompt {name}")
    return text, sha256_hex(text)


def _check_hf_config(config: Any) -> None:
    """Checks shared by the Hugging Face backend configurations."""
    _text(config.model_id, "model_id")
    for name in ("revision", "tokenizer_revision"):
        value = getattr(config, name)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
            raise MathModelError(f"{name} must be an immutable 40-character commit SHA")
    if config.input_format not in INPUT_FORMATS:
        raise MathModelError(f"input_format must be one of {INPUT_FORMATS}")
    if config.dtype not in DTYPES:
        raise MathModelError("unsupported dtype; quantization is not silently enabled")
    for name in ("batch_size", "max_input_tokens"):
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise MathModelError(f"{name} must be a positive integer")
    for name in ("add_special_tokens", "local_files_only", "license_reviewed", "trust_remote_code"):
        if type(getattr(config, name)) is not bool:
            raise MathModelError(f"{name} must be boolean")
    kwargs = config.chat_template_kwargs
    if not isinstance(kwargs, dict) or any(not isinstance(key, str) for key in kwargs):
        raise MathModelError("chat_template_kwargs must be a mapping with string keys")
    try:
        json.dumps(kwargs)
    except (TypeError, ValueError) as exc:
        raise MathModelError("chat_template_kwargs must hold JSON values") from exc
    if config.input_format == "raw":
        if kwargs:
            raise MathModelError("chat_template_kwargs apply only to input_format chat_template")
        if not isinstance(config.raw_template, str) or not config.raw_template.strip():
            raise MathModelError("input_format raw requires a raw_template")
    elif config.raw_template is not None:
        raise MathModelError("raw_template applies only to input_format raw")
    _text(config.device_map, "device_map")
    _text(config.attn_implementation, "attn_implementation")
    if not isinstance(config.license_review_reference, str):
        raise MathModelError("license_review_reference must be a string")
    if config.local_path is not None:
        _text(config.local_path, "local_path")


@dataclass(frozen=True)
class LLMJudgeConfig:
    """An instruct model asked whether the final answer is correct; score = P(No).

    ``input_format`` ``chat_template`` renders the prompt as one user turn with the
    tokenizer's own template and its generation prompt; ``raw`` puts it into
    ``raw_template`` (one ``{prompt}`` slot) with the model's turn markers written out.
    ``answer_prefix`` is text the model emits before its verdict; it is teacher-forced as
    token IDs, so the next position is the verdict. ``yes_token`` / ``no_token`` must each
    be one token; ``yes_token_id`` / ``no_token_id`` pin the IDs when the plain spelling
    encodes to a different variant than the one the model emits. ``prompt_sha256`` pins
    ``prompts/<prompt_file>``.
    """

    model_id: str
    revision: str
    tokenizer_revision: str
    input_format: Literal["chat_template", "raw"]
    add_special_tokens: bool
    yes_token: str
    no_token: str
    dtype: Literal["float32", "float16", "bfloat16"]
    batch_size: int
    max_input_tokens: int
    prompt_sha256: str
    prompt_file: str = JUDGE_PROMPT_FILE
    raw_template: str | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    answer_prefix: str = ""
    yes_token_id: int | None = None
    no_token_id: int | None = None
    device_map: str = "auto"
    attn_implementation: str = "eager"
    local_files_only: bool = True
    license_reviewed: bool = False
    license_review_reference: str = ""
    trust_remote_code: bool = False
    local_path: str | None = None

    def __post_init__(self) -> None:
        _check_hf_config(self)
        if self.input_format == "raw":
            _check_slots(self.raw_template, ("prompt",), "judge raw_template")
        for name in ("yes_token", "no_token"):
            _text(getattr(self, name), name)
        if self.yes_token.strip() == self.no_token.strip():
            raise MathModelError("yes_token and no_token must differ")
        ids = (self.yes_token_id, self.no_token_id)
        if (ids[0] is None) != (ids[1] is None):
            raise MathModelError("set both yes_token_id and no_token_id, or neither")
        if ids[0] is not None and (
            any(type(value) is not int or value < 0 for value in ids) or ids[0] == ids[1]
        ):
            raise MathModelError("verdict token ids must be two distinct nonnegative integers")
        if not isinstance(self.answer_prefix, str):
            raise MathModelError("answer_prefix must be a string")
        if (
            not isinstance(self.prompt_file, str)
            or Path(self.prompt_file).name != self.prompt_file
            or not self.prompt_file.endswith(".txt")
        ):
            raise MathModelError("prompt_file must be a .txt file name inside prompts/")
        if not isinstance(self.prompt_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.prompt_sha256
        ):
            raise MathModelError("prompt_sha256 must be a SHA256 hex digest")

    def logical_configuration(self) -> dict[str, Any]:
        """Configuration identity excluding a machine-local checkpoint directory."""
        result = asdict(self)
        result.pop("local_path", None)
        return result


@dataclass(frozen=True)
class ORMConfig:
    """A sequence-regression / reward model over problem + full solution.

    ``input_format`` ``chat_template``: an optional ``system_prompt`` turn, the problem as
    the user turn and the solution as the assistant turn, rendered by the tokenizer's own
    template without a generation prompt. ``raw``: ``raw_template`` with one ``{problem}``
    and one ``{solution}`` slot and every separator written out.
    """

    model_id: str
    revision: str
    tokenizer_revision: str
    input_format: Literal["chat_template", "raw"]
    add_special_tokens: bool
    dtype: Literal["float32", "float16", "bfloat16"]
    batch_size: int
    max_input_tokens: int
    system_prompt: str | None = None
    raw_template: str | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    device_map: str = "auto"
    attn_implementation: str = "eager"
    local_files_only: bool = True
    license_reviewed: bool = False
    license_review_reference: str = ""
    trust_remote_code: bool = False
    local_path: str | None = None

    def __post_init__(self) -> None:
        _check_hf_config(self)
        if self.input_format == "raw":
            _check_slots(self.raw_template, ("problem", "solution"), "ORM raw_template")
            if self.system_prompt is not None:
                raise MathModelError("system_prompt applies only to input_format chat_template")
        elif self.system_prompt is not None:
            _text(self.system_prompt, "system_prompt")

    def logical_configuration(self) -> dict[str, Any]:
        """Configuration identity excluding a machine-local checkpoint directory."""
        result = asdict(self)
        result.pop("local_path", None)
        return result


def _output_logits(output: Any) -> Any:
    logits = getattr(output, "logits", None)
    if logits is None:
        try:
            logits = output[0]
        except (TypeError, IndexError, KeyError) as exc:
            raise ModelOutputError("model output has no logits") from exc
    return logits


def _host_list(values: Any) -> list:
    if hasattr(values, "detach"):
        values = values.detach().float().cpu()
    return values.tolist()


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


class _HFVerifier:
    """Loading, tokenization and fixed-shape batching shared by the Hugging Face backends."""

    form = ""
    implementation = ""
    adapter_version = ""
    model_class = ""

    def __init__(self, config: Any, *, tokenizer: Any, model: Any, torch_module: Any) -> None:
        supplied = [tokenizer is not None, model is not None, torch_module is not None]
        if any(supplied) and not all(supplied):
            raise MathModelError("inject tokenizer, model and torch_module together")
        self.config = config
        self._tokenizer, self._model, self._torch = tokenizer, model, torch_module
        self._injected = all(supplied)
        self._prepared = False
        self._closed = False
        self._pad_id: int | None = None
        self._special_texts: tuple[str, ...] = ()
        self._versions: dict[str, str | None] = {}

    @property
    def config_hash(self) -> str:
        return sha256_hex(canonical_json(self.config.logical_configuration()))

    # -- loading ------------------------------------------------------------------------

    def load(self) -> None:
        """Explicit lazy load from the pinned revision or a manifest-checked local copy."""
        if self._closed:
            raise MathModelError("verifier is closed; construct a new adapter to reload")
        _require_license_review(self.config)
        if self._prepared:
            return
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
            common = {
                "local_files_only": self.config.local_files_only,
                "trust_remote_code": self.config.trust_remote_code,
            }
            tokenizer_kwargs = {"revision": self.config.tokenizer_revision, **common}
            model_kwargs = {
                "revision": self.config.revision,
                "dtype": getattr(self._torch, self.config.dtype),
                "device_map": self.config.device_map,
                "attn_implementation": self.config.attn_implementation,
                **common,
            }
            if self.config.trust_remote_code:
                tokenizer_kwargs["code_revision"] = self.config.tokenizer_revision
                model_kwargs["code_revision"] = self.config.revision
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(source, **tokenizer_kwargs)
            self._model = getattr(transformers, self.model_class).from_pretrained(
                source, **model_kwargs
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
        if self.config.max_input_tokens > maximum:
            raise MathModelError(
                f"max_input_tokens {self.config.max_input_tokens} exceeds the model's "
                f"max_position_embeddings {maximum}"
            )
        actual = str(getattr(self._model, "dtype", "")).removeprefix("torch.")
        if actual != self.config.dtype:
            raise MathModelError(
                f"loaded weights are {actual or 'of unknown dtype'}, configured {self.config.dtype}"
            )
        specials = getattr(self._tokenizer, "all_special_tokens", None) or ()
        self._special_texts = tuple(sorted({str(token) for token in specials if str(token)}))
        self._pad_id = self._resolve_pad_id(text_config)
        self._prepare_tokens()
        self._versions = _library_versions()
        self._prepared = True

    def _resolve_pad_id(self, text_config: Any) -> int:
        raise NotImplementedError

    def _prepare_tokens(self) -> None:
        """Model-specific token checks after loading; nothing by default."""

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

    def _render_chat(self, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
        value = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **self.config.chat_template_kwargs,
        )
        return _text(value, "serialized chat")

    def _chat_template_hash(self) -> str:
        return sha256_hex(canonical_json(getattr(self._tokenizer, "chat_template", None)))

    def _serialize(self, problem: str, solution: str) -> str:
        raise NotImplementedError

    def _suffix_ids(self) -> list[int]:
        return []

    def _check_row(self, ids: list[int]) -> None:
        """Model-specific checks of one finished input row; nothing by default."""

    def _row(self, problem: str, solution: str) -> tuple[str, list[int]]:
        for value in (problem, solution):
            if any(token in value for token in self._special_texts):
                raise ModelInputError(
                    "special-token text in problem or solution would be read as a control token"
                )
        serialized = self._serialize(problem, solution)
        ids = (
            self._encode(serialized, add_special_tokens=self.config.add_special_tokens)
            + self._suffix_ids()
        )
        if not ids or len(ids) > self.config.max_input_tokens:
            raise ModelInputError(
                f"input has {len(ids)} tokens; limit is {self.config.max_input_tokens}; "
                "truncation is forbidden"
            )
        self._check_row(ids)
        return serialized, ids

    # -- forward ------------------------------------------------------------------------

    def _input_device(self) -> Any:
        device = getattr(self._model, "device", None)
        if hasattr(self._model, "get_input_embeddings"):
            device = self._model.get_input_embeddings().weight.device
        if device is None or str(device) == "meta":
            raise MathModelError("model input device is missing or still meta")
        return device

    def _forward(self, rows: list[list[int]]) -> Any:
        """One fixed-shape batch: ``batch_size`` rows of ``max_input_tokens``, left padded.

        A short last batch is filled with copies of its last row; the caller drops their
        scores. Position IDs restart at 0 on the first real token of every row, which is
        what ``(attention_mask.cumsum(-1) - 1).clamp(min=0)`` gives in the detox backend.
        """
        size, length = self.config.batch_size, self.config.max_input_tokens
        if not 0 < len(rows) <= size:
            raise MathModelError("a forward batch holds between 1 and batch_size rows")
        filled = rows + [rows[-1]] * (size - len(rows))
        pads = [length - len(row) for row in filled]
        values = {
            "input_ids": [[self._pad_id] * n + row for n, row in zip(pads, filled, strict=True)],
            "attention_mask": [
                [0] * n + [1] * len(row) for n, row in zip(pads, filled, strict=True)
            ],
            "position_ids": [
                [0] * n + list(range(len(row))) for n, row in zip(pads, filled, strict=True)
            ],
        }
        device = self._input_device()
        tensors = {
            name: self._torch.tensor(value, dtype=self._torch.long, device=device)
            for name, value in values.items()
        }
        with self._torch.inference_mode():
            return self._call(tensors)

    def _call(self, tensors: dict[str, Any]) -> Any:
        raise NotImplementedError

    def _select(self, output: Any, n_real: int) -> list[Any]:
        raise NotImplementedError

    def _score_fields(self, raw: Any) -> dict[str, Any]:
        raise NotImplementedError

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

        Every input is built and length-checked before the first forward pass, so an
        over-long input stops the call without scoring anything.
        """
        pairs = self._pairs(items)
        if not pairs:
            return []
        self.load()
        prepared = [
            (problem, solution, *self._row(problem, solution)) for problem, solution in pairs
        ]
        results = []
        for start in range(0, len(prepared), self.config.batch_size):
            chunk = prepared[start : start + self.config.batch_size]
            started = time.monotonic()
            raw = self._select(self._forward([ids for *_, ids in chunk]), len(chunk))
            if len(raw) != len(chunk):
                raise ModelOutputError("model returned a different number of scores than inputs")
            elapsed = time.monotonic() - started
            for (problem, solution, serialized, ids), value in zip(chunk, raw, strict=True):
                results.append(
                    {
                        **self._score_fields(value),
                        **self._record(problem, solution, serialized, ids, elapsed),
                    }
                )
        return results

    def score(self, problem: str, solution: str) -> dict[str, Any]:
        """The pipeline's single-state call; runs one full fixed-shape batch."""
        return self.score_batch([(problem, solution)])[0]

    def _record(
        self, problem: str, solution: str, serialized: str, ids: list[int], elapsed: float
    ) -> dict[str, Any]:
        return {
            "score_direction": SCORE_DIRECTION,
            "input_tokens": len(ids),
            "max_input_tokens": self.config.max_input_tokens,
            "truncated": False,
            "fixed_batch_shape": [self.config.batch_size, self.config.max_input_tokens],
            "padding_side": "left",
            "pad_token_id": self._pad_id,
            "model_id": self.config.model_id,
            "revision": self.config.revision,
            "tokenizer_revision": self.config.tokenizer_revision,
            "configured_dtype": self.config.dtype,
            "actual_dtype": str(getattr(self._model, "dtype", "unknown")),
            "device": str(self._input_device()),
            "attn_implementation": self.config.attn_implementation,
            "implementation": self.implementation,
            "adapter_version": self.adapter_version,
            "config_hash": self.config_hash,
            "local_path": self.config.local_path,
            "input_hash": sha256_hex(canonical_json({"problem": problem, "solution": solution})),
            "serialized_input_hash": sha256_hex(serialized),
            "serialized_input": serialized,
            "token_ids_hash": sha256_hex(canonical_json(ids)),
            "chat_template_hash": self._chat_template_hash(),
            "library_versions": self._versions,
            "batch_elapsed_seconds": elapsed,
            "execution_mode": "injected_offline" if self._injected else "local_model",
            "license_review_reference": self.config.license_review_reference,
        }

    # -- card ---------------------------------------------------------------------------

    def _card_input(self) -> dict[str, Any]:
        raise NotImplementedError

    def _card_score(self) -> dict[str, Any]:
        raise NotImplementedError

    def verifier_card(self) -> dict[str, Any]:
        """What was loaded and how it scores (paper Sec. 5: written into the verifier_card)."""
        self.load()
        return {
            "form": self.form,
            "implementation": self.implementation,
            "adapter_version": self.adapter_version,
            "model_id": self.config.model_id,
            "revision": self.config.revision,
            "tokenizer_revision": self.config.tokenizer_revision,
            "configuration": self.config.logical_configuration(),
            "config_hash": self.config_hash,
            "actual_dtype": str(getattr(self._model, "dtype", "unknown")),
            "device": str(self._input_device()),
            "input": {
                "format": self.config.input_format,
                "add_special_tokens": self.config.add_special_tokens,
                "chat_template_kwargs": self.config.chat_template_kwargs,
                "raw_template": self.config.raw_template,
                "chat_template_hash": self._chat_template_hash(),
                **self._card_input(),
            },
            "batching": {
                "batch_size": self.config.batch_size,
                "max_input_tokens": self.config.max_input_tokens,
                "fixed_shape": True,
                "padding_side": "left",
                "pad_token_id": self._pad_id,
                "last_batch": "filled with copies of its last row; their scores are dropped",
            },
            "truncation": "never; an input over max_input_tokens is refused",
            "reference_answer_input": False,
            "score": {"direction": SCORE_DIRECTION, **self._card_score()},
            "library_versions": self._versions,
            "execution_mode": "injected_offline" if self._injected else "local_model",
            "license_review_reference": self.config.license_review_reference,
        }

    def close(self) -> None:
        self._model = self._tokenizer = self._torch = None
        self._prepared = False
        self._closed = True


class LLMJudgeVerifier(_HFVerifier):
    """P(No) at the verdict position of an instruct model (paper Sec. 5 LLM judge)."""

    form = "llm_judge"
    implementation = JUDGE_IMPLEMENTATION
    adapter_version = JUDGE_ADAPTER_VERSION
    config_class = LLMJudgeConfig
    model_class = "AutoModelForCausalLM"

    def __init__(
        self,
        config: LLMJudgeConfig,
        *,
        prompt_dir: Path | None = None,
        tokenizer: Any = None,
        model: Any = None,
        torch_module: Any = None,
    ) -> None:
        if not isinstance(config, LLMJudgeConfig):
            raise MathModelError("LLMJudgeVerifier requires an LLMJudgeConfig")
        super().__init__(config, tokenizer=tokenizer, model=model, torch_module=torch_module)
        self._prompt_text, self.prompt_sha256 = read_judge_prompt(config.prompt_file, prompt_dir)
        if self.prompt_sha256 != config.prompt_sha256:
            raise MathModelError(
                f"prompts/{config.prompt_file} differs from the configured prompt_sha256 "
                f"(file {self.prompt_sha256}); review the prompt and pin its hash"
            )
        self._verdict_ids: tuple[int, int] = (-1, -1)
        self._verdict_pieces: tuple[str, str] = ("", "")
        self._prefix_ids: list[int] = []
        self._keep_last_logits = True

    def _resolve_pad_id(self, text_config: Any) -> int:
        # The attention mask is built from row lengths, so the pad value never reaches a
        # real token; any ID the tokenizer owns will do, as in the detox backend.
        pad = getattr(self._tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self._tokenizer, "eos_token_id", None)
        if type(pad) is not int:
            raise MathModelError("the tokenizer has neither a pad nor an eos token id")
        return pad

    def _prepare_tokens(self) -> None:
        ids, pieces = [], []
        for form, configured in (
            (self.config.yes_token, self.config.yes_token_id),
            (self.config.no_token, self.config.no_token_id),
        ):
            if configured is None:
                encoded = self._encode(form, add_special_tokens=False)
                if len(encoded) != 1:
                    raise MathModelError(
                        f"verdict {form!r} is {len(encoded)} tokens for this tokenizer; "
                        "configure a single-token spelling or its token id"
                    )
                configured = encoded[0]
            piece = self._decode([configured])
            if piece.strip() != form.strip():
                raise MathModelError(
                    f"verdict token id {configured} decodes to {piece!r}, not {form!r}"
                )
            ids.append(configured)
            pieces.append(piece)
        if ids[0] == ids[1]:
            raise MathModelError("Yes and No resolve to the same token id")
        self._verdict_ids = (ids[0], ids[1])
        self._verdict_pieces = (pieces[0], pieces[1])
        self._prefix_ids = (
            self._encode(self.config.answer_prefix, add_special_tokens=False)
            if self.config.answer_prefix
            else []
        )
        if self.config.answer_prefix and not self._prefix_ids:
            raise MathModelError("answer_prefix encodes to no tokens")

    def _serialize(self, problem: str, solution: str) -> str:
        prompt = fill_slots(self._prompt_text, {"problem": problem, "solution": solution})
        if self.config.input_format == "raw":
            return fill_slots(self.config.raw_template, {"prompt": prompt})
        return self._render_chat([{"role": "user", "content": prompt}], add_generation_prompt=True)

    def _suffix_ids(self) -> list[int]:
        return list(self._prefix_ids)

    def _call(self, tensors: dict[str, Any]) -> Any:
        if self._keep_last_logits:
            try:
                return self._model(**tensors, use_cache=False, logits_to_keep=1)
            except TypeError:
                self._keep_last_logits = False  # this model's forward has no logits_to_keep
        return self._model(**tensors, use_cache=False)

    def _select(self, output: Any, n_real: int) -> list[Any]:
        logits = _output_logits(output)
        shape = tuple(logits.shape)
        if len(shape) != 3 or shape[0] != self.config.batch_size or shape[1] < 1:
            raise ModelOutputError("judge logits must have shape [batch_size, positions, vocab]")
        if shape[2] <= max(self._verdict_ids):
            raise ModelOutputError("judge logits do not cover the verdict token ids")
        pairs = _host_list(logits[:, -1, :][:, list(self._verdict_ids)])[:n_real]
        if any(len(pair) != 2 for pair in pairs):
            raise ModelOutputError("expected one Yes and one No logit per input")
        return pairs

    def _score_fields(self, raw: Any) -> dict[str, Any]:
        yes, no = (float(value) for value in raw)
        p_no = _probabilities([[yes, no]])[0]
        return {
            "error_score": _unit_score(p_no),
            "p_no": p_no,
            "verdict_logits_yes_no": [yes, no],
            "verdict_token_ids_yes_no": list(self._verdict_ids),
            "answer_prefix_token_ids": list(self._prefix_ids),
            "score_definition": "p_no_softmax_over_yes_no_token_logits",
            "softmax_arithmetic": "python_float64_pairwise",
            "prompt_file": self.config.prompt_file,
            "prompt_sha256": self.prompt_sha256,
        }

    def _card_input(self) -> dict[str, Any]:
        return {
            "prompt_file": f"prompts/{self.config.prompt_file}",
            "prompt_sha256": self.prompt_sha256,
            "prompt_text": self._prompt_text,
            "answer_prefix": self.config.answer_prefix,
            "answer_prefix_token_ids": list(self._prefix_ids),
        }

    def _card_score(self) -> dict[str, Any]:
        return {
            "definition": "P(No): softmax over the Yes and No token logits at the position "
            "after the prompt and answer_prefix; one forward pass, nothing generated",
            "verdict_tokens": {
                "yes": [self._verdict_ids[0], self._verdict_pieces[0]],
                "no": [self._verdict_ids[1], self._verdict_pieces[1]],
            },
            "arithmetic": "python_float64_pairwise",
        }


class ORMVerifier(_HFVerifier):
    """1 - sigmoid(scalar logit) of a sequence-regression / reward model (paper Sec. 5 ORM)."""

    form = "orm"
    implementation = ORM_IMPLEMENTATION
    adapter_version = ORM_ADAPTER_VERSION
    config_class = ORMConfig
    model_class = "AutoModelForSequenceClassification"

    def __init__(
        self,
        config: ORMConfig,
        *,
        tokenizer: Any = None,
        model: Any = None,
        torch_module: Any = None,
    ) -> None:
        if not isinstance(config, ORMConfig):
            raise MathModelError("ORMVerifier requires an ORMConfig")
        super().__init__(config, tokenizer=tokenizer, model=model, torch_module=torch_module)

    def _resolve_pad_id(self, text_config: Any) -> int:
        # The model's own head pools the last token that is not pad_token_id; the batch is
        # padded with exactly that ID, so the pooled token is each row's last real token.
        pad = getattr(text_config, "pad_token_id", None)
        if type(pad) is not int:
            raise MathModelError(
                "the reward model must declare pad_token_id: its last-token pooling reads it"
            )
        return pad

    def _serialize(self, problem: str, solution: str) -> str:
        if self.config.input_format == "raw":
            return fill_slots(self.config.raw_template, {"problem": problem, "solution": solution})
        messages = (
            []
            if self.config.system_prompt is None
            else [{"role": "system", "content": self.config.system_prompt}]
        )
        messages += [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ]
        return self._render_chat(messages, add_generation_prompt=False)

    def _check_row(self, ids: list[int]) -> None:
        if ids[-1] == self._pad_id:
            raise ModelInputError(
                "the input ends with pad_token_id; the model's last-token pooling would skip it"
            )

    def _call(self, tensors: dict[str, Any]) -> Any:
        return self._model(**tensors, use_cache=False)

    def _select(self, output: Any, n_real: int) -> list[Any]:
        logits = _output_logits(output)
        if tuple(logits.shape) != (self.config.batch_size, 1):
            raise ModelOutputError("an ORM must return one scalar logit per input: [batch_size, 1]")
        return [row[0] for row in _host_list(logits)][:n_real]

    def _score_fields(self, raw: Any) -> dict[str, Any]:
        logit = float(raw)
        if not math.isfinite(logit):
            raise ModelOutputError("ORM logit must be finite")
        return {
            "error_score": _unit_score(_sigmoid(-logit)),
            "logit": logit,
            "sigmoid_logit": _sigmoid(logit),
            "score_definition": "one_minus_sigmoid_of_scalar_logit",
            "transform": "sigmoid(-logit), equal to 1 - sigmoid(logit); not a calibration",
        }

    def _card_input(self) -> dict[str, Any]:
        return {"system_prompt": self.config.system_prompt}

    def _card_score(self) -> dict[str, Any]:
        return {
            "definition": "1 - sigmoid(logit) of the model's scalar head, pooled by the model "
            "at the last token that is not pad_token_id",
            "arithmetic": "python_float64",
        }


#: The installed Hugging Face adapter per form; build_verifier and VerifierSpec read it.
HF_ADAPTERS = {"llm_judge": LLMJudgeVerifier, "orm": ORMVerifier}
HF_IMPLEMENTATIONS = {form: adapter.implementation for form, adapter in HF_ADAPTERS.items()}
# paper Sec. 5: the generative adapters (``generative.py``), by implementation, with the form each
#: one is: a step-level generative PRM is ``prm``, a whole-solution reasoning judge ``llm_judge``.
GENERATIVE_IMPLEMENTATIONS = {"hf_generative_step": "prm", "hf_generative_judge": "llm_judge"}
# paper Sec. 5: the sources no math verifier may come from (the controller and G_fixed are OpenAI;
#: E is Anthropic). Matched case-insensitively against every identity and lineage entry.
EXCLUDED_SOURCES = ("openai", "anthropic")
LINEAGE_KEYS = ("vendors", "bases", "reasoning_data")


def _excluded(value: str) -> bool:
    folded = value.casefold()
    return any(source in folded for source in EXCLUDED_SOURCES) or folded.startswith(
        ("gpt-", "o1", "o3", "o4", "claude")
    )


def _check_lineage(lineage: Any) -> None:
    """``vendors``: every organisation that trained a model in the weight lineage (the base's
    pretraining, any distillation or fine-tune in between, the final fine-tune); ``bases``: the
    pretrained base family; ``reasoning_data``: the models documented as having generated the
    reasoning training data (the OpenAI rule of paper Sec. 5 reads it; it is not a pair
    criterion)."""
    if not isinstance(lineage, dict) or set(lineage) != set(LINEAGE_KEYS):
        raise MathModelError(f"lineage must map exactly {LINEAGE_KEYS} to lists of names")
    for key in LINEAGE_KEYS:
        values = lineage[key]
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise MathModelError(f"lineage {key} must be a list of nonempty names")
        if key != "reasoning_data" and not values:
            raise MathModelError(f"lineage {key} must name at least one source")
        if any(_excluded(value) for value in values):
            raise MathModelError(
                f"lineage {key} names an OpenAI or Anthropic source (M.4: not eligible)"
            )


@dataclass(frozen=True)
class VerifierSpec:
    """Reviewed source identity, separate from its implementation and host provider.

    vendor identifies the model's training source; an OpenAI-compatible HTTP server
    does not by itself make the model OpenAI-sourced. base_model is the canonical
    underlying backbone identity, not a checkpoint alias. ``lineage`` writes the
    sources as sets, for models trained from several (a fine-tune of a distillation of
    another vendor's base): see :func:`_check_lineage`.
    """

    form: Literal["prm", "orm", "llm_judge"]
    vendor: str
    base_model: str
    model_id: str
    revision: str
    implementation: str
    settings: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, list[str]] | None = None

    def __post_init__(self) -> None:
        if self.form not in ("prm", "orm", "llm_judge"):
            raise MathModelError("verifier form must be prm, orm, or llm_judge")
        for name in ("vendor", "base_model", "model_id", "revision", "implementation"):
            value = _text(getattr(self, name), name)
            if value.strip().upper() == "TBD":
                raise MathModelError(f"verifier {name} is still TBD")
        if not isinstance(self.settings, dict):
            raise MathModelError("verifier settings must be a mapping")
        identities = (self.vendor, self.base_model, self.model_id)
        if any("openai" in value.casefold() for value in identities) or any(
            value.casefold().startswith("gpt-") for value in identities[1:]
        ):
            raise MathModelError("math verifiers must not share OpenAI controller source")
        if any("anthropic" in value.casefold() for value in identities):
            raise MathModelError("math verifiers must not come from Anthropic (M.4 v1.2)")
        if self.lineage is not None:
            _check_lineage(self.lineage)
        owner = (
            {kind: "prm" for kind in MODEL_REPOSITORIES}
            | {implementation: form for form, implementation in HF_IMPLEMENTATIONS.items()}
            | GENERATIVE_IMPLEMENTATIONS
        )
        if owner.get(self.implementation, self.form) != self.form:
            raise MathModelError(
                f"the {owner[self.implementation]} implementation {self.implementation!r} "
                f"cannot be relabelled {self.form}"
            )

    def sources(self, key: str) -> frozenset[str]:
        """The case-folded vendor or base set: the lineage's, or the single reviewed name."""
        if key not in ("vendors", "bases"):
            raise MathModelError("sources are vendors or bases")
        if self.lineage is not None:
            return frozenset(value.strip().casefold() for value in self.lineage[key])
        single = self.vendor if key == "vendors" else self.base_model
        return frozenset({single.strip().casefold()})


def pair_ineligibility(first: VerifierSpec, second: VerifierSpec) -> list[str]:
    """Why two reviewed verifiers cannot be a pair under paper Sec. 5 (empty: they can).

    Vendors and bases must be disjoint as sets; forms may be the same. The OpenAI and
    Anthropic exclusions are checked on each spec when it is built.
    """
    reasons = []
    for key in ("vendors", "bases"):
        shared = sorted(first.sources(key) & second.sources(key))
        if shared:
            reasons.append(f"shared {key[:-1]}: {', '.join(shared)}")
    return reasons


def validate_verifier_pair(specs: Sequence[VerifierSpec | Mapping[str, Any]]) -> tuple:
    """paper Sec. 5: disjoint training sources and underlying backbones (forms may coincide)."""
    if len(specs) != 2:
        raise MathModelError("exactly two reviewed verifier specifications required")
    first, second = tuple(
        item if isinstance(item, VerifierSpec) else VerifierSpec(**dict(item)) for item in specs
    )
    reasons = pair_ineligibility(first, second)
    if reasons:
        raise MathModelError("the verifier pair is not eligible: " + "; ".join(reasons))
    return first, second


class MathVerifier(Protocol):
    def score(self, problem: str, solution: str) -> dict[str, Any]: ...


def _unit_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelOutputError("backend must provide a numeric continuous score")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ModelOutputError("backend score must be finite and within [0, 1]")
    return value


def _spec_settings(spec: VerifierSpec, *pinned: tuple[str, str]) -> dict[str, Any]:
    settings = dict(spec.settings)
    for name, expected in (*pinned, ("model_id", spec.model_id), ("revision", spec.revision)):
        if name in settings and settings[name] != expected:
            raise MathModelError(f"{spec.form} settings conflict with verifier {name}")
        settings[name] = expected
    return settings


def build_verifier(spec: VerifierSpec | Mapping[str, Any]) -> MathVerifier:
    """The one entry point for every form; builds the adapter, loads nothing.

    Every model choice comes from the reviewed spec and its settings; a missing or
    unknown setting fails here, before any weights or requests.
    """
    spec = spec if isinstance(spec, VerifierSpec) else VerifierSpec(**dict(spec))
    if spec.implementation in GENERATIVE_IMPLEMENTATIONS:
        from rafr.domains.math.generative import GENERATIVE_ADAPTERS

        adapter = GENERATIVE_ADAPTERS[spec.implementation]
        try:
            config = adapter.config_class(**_spec_settings(spec))
        except TypeError as exc:
            raise MathModelError(f"{spec.form} settings are incomplete or unknown: {exc}") from exc
        return adapter(config)
    if spec.form == "prm":
        if spec.implementation not in MODEL_REPOSITORIES:
            raise NotImplementedError("selected PRM implementation has not been installed")
        return PRMVerifier(PRMConfig(**_spec_settings(spec, ("kind", spec.implementation))))
    adapter = HF_ADAPTERS.get(spec.form)
    if adapter is None or spec.implementation != adapter.implementation:
        raise NotImplementedError(
            f"selected {spec.form} implementation has not been installed "
            f"(available: {HF_IMPLEMENTATIONS.get(spec.form)!r})"
        )
    try:
        config = adapter.config_class(**_spec_settings(spec))
    except TypeError as exc:
        raise MathModelError(f"{spec.form} settings are incomplete or unknown: {exc}") from exc
    return adapter(config)


__all__ = [
    "PRMConfig",
    "PRMVerifier",
    "MathModelError",
    "ModelInputError",
    "ModelOutputError",
    "ModelAccessNotReviewedError",
    "split_solution_steps",
    "aggregate_steps",
    "VerifierSpec",
    "MathVerifier",
    "LLMJudgeConfig",
    "LLMJudgeVerifier",
    "ORMConfig",
    "ORMVerifier",
    "JUDGE_IMPLEMENTATION",
    "ORM_IMPLEMENTATION",
    "read_judge_prompt",
    "validate_verifier_pair",
    "pair_ineligibility",
    "GENERATIVE_IMPLEMENTATIONS",
    "build_verifier",
]
