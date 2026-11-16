"""Torch backend for the step 2 probe's guard LMs (:mod:`rafr.verifiers.candidates.probe`).

The second module in the package that touches ``torch`` / ``transformers`` (the first is
``torch_backend.py``, which serves the older registry candidates); both import them lazily, so
the test suite runs without either installed.

Inference follows decisions.md section 4: ``model.eval()``, ``torch.no_grad()``, no sampling,
one forward pass per prompt with the logits of the last position only. Decoder batches are
left-padded, ``position_ids`` are rebuilt from the attention mask, and with ``fixed_shape``
every batch has ``batch_size`` rows (the last one is filled with copies and the copies are
dropped) and one sequence length per call type, so a score cannot depend on its neighbours.
The determinism switches are ``torch_backend.configure_determinism``'s.

Prompts are rendered by the model's own chat template (``tokenize=False``) and then tokenised
with ``add_special_tokens=False``: the templates already write their BOS token, and adding a
second one would change what the model sees. The verdict prefix is appended as token ids --
the exact ids the model produced in the preflight's greedy decode -- not as text, so a
tokeniser merge at the seam cannot alter it.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from typing import Any

from rafr.verifiers.candidates.base import UNPINNED, VerifierConfigError
from rafr.verifiers.candidates.probe import (
    CAUSAL_LM,
    IMAGE_TEXT_TO_TEXT,
    ProbeModel,
    Verdict,
)
from rafr.verifiers.candidates.torch_backend import (
    _commit_hash,
    _torch,
    _transformers,
    configure_determinism,
    resolve_device,
    resolve_dtype,
)


def _load_lm(
    tf: Any, m: ProbeModel, revision: str | None, dtype: Any, device: str
) -> tuple[Any, str]:
    """``(model, attention implementation)``: eager when the architecture has it (its reduction
    order is fixed), SDPA otherwise -- the one used is recorded with the run.

    Llama Guard 4 is a Llama 4 multimodal checkpoint and loads through its conditional-
    generation class; it is only ever fed text here.
    """
    try:
        return _load_lm_with(tf, m, revision, dtype, device, "eager"), "eager"
    except (ValueError, NotImplementedError) as exc:
        if "attn" not in str(exc).lower() and "attention" not in str(exc).lower():
            raise
        return _load_lm_with(tf, m, revision, dtype, device, "sdpa"), "sdpa"


def _load_lm_with(
    tf: Any, m: ProbeModel, revision: str | None, dtype: Any, device: str, attn: str
) -> Any:
    if m.loader == CAUSAL_LM:
        loaders = [tf.AutoModelForCausalLM]
    elif m.loader == IMAGE_TEXT_TO_TEXT:
        loaders = [
            getattr(tf, name)
            for name in ("Llama4ForConditionalGeneration", "AutoModelForImageTextToText")
            if hasattr(tf, name)
        ]
    else:
        raise VerifierConfigError(f"{m.model_id}: loader {m.loader!r} is not a guard LM")
    if not loaders:
        raise VerifierConfigError(f"{m.model_id}: this transformers has no loader for it")
    common: dict[str, Any] = {
        "revision": revision,
        "attn_implementation": attn,
        "device_map": device,
    }
    try:
        return loaders[0].from_pretrained(m.repo_id, dtype=dtype, **common)
    except TypeError:  # transformers before the torch_dtype -> dtype rename
        return loaders[0].from_pretrained(m.repo_id, torch_dtype=dtype, **common)


class TorchProbeBackend:
    """One loaded guard LM, read as a two-token classifier. Implements ``ProbeBackend``."""

    def __init__(
        self,
        *,
        model: ProbeModel,
        revision: str | None,
        batch_size: int,
        fixed_shape: bool = True,
        load_weights: bool = True,
    ) -> None:
        if not model.is_lm:
            raise VerifierConfigError(f"{model.model_id} is not a guard LM")
        tf = _transformers()
        self.spec = model
        self.batch_size = int(batch_size)
        self.fixed_shape = bool(fixed_shape)
        self.revision = revision or UNPINNED
        self.model: Any = None
        self.attn_implementation: str | None = None
        self._keep_last = True  # logits_to_keep=1; switched off if this model rejects it

        self.tokenizer = tf.AutoTokenizer.from_pretrained(model.repo_id, revision=revision)
        self.template_source = "tokenizer"
        self._templater: Any = self.tokenizer
        if not getattr(self.tokenizer, "chat_template", None):
            # Llama Guard 4 ships its template for the processor (chat_template.jinja)
            self._templater = tf.AutoProcessor.from_pretrained(model.repo_id, revision=revision)
            self.template_source = "processor"
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise VerifierConfigError(f"{model.repo_id}: tokenizer has neither pad nor eos")
        self.pad_id = int(pad)
        if not load_weights:
            self.device = "cpu"
            return

        configure_determinism()
        self.device = resolve_device()
        self.model, self.attn_implementation = _load_lm(
            tf, model, revision, resolve_dtype(model.dtype), self.device
        )
        self.model.eval()
        self.revision = _commit_hash(self.model, revision or UNPINNED)

    @classmethod
    def templates_only(cls, *, model: ProbeModel, revision: str | None) -> TorchProbeBackend:
        """Tokenizer and chat template without the weights: enough to render prompts."""
        return cls(model=model, revision=revision, batch_size=1, load_weights=False)

    # -- text ---------------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        return [int(i) for i in self.tokenizer.encode(text, add_special_tokens=False)]

    def n_text_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        ids = self.encode(text)
        if len(ids) <= max_tokens:
            return text
        return str(self.tokenizer.decode(ids[:max_tokens], skip_special_tokens=True))

    def render(self, messages: list[dict[str, Any]], template_kwargs: dict[str, Any]) -> str:
        out = self._templater.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        if isinstance(out, list):  # a processor may hand back a batch of one
            (out,) = out
        return str(out)

    def n_prompt_tokens(self, prompt: str) -> int:
        return len(self.encode(prompt))

    def token_piece(self, token_id: int) -> str:
        return str(self.tokenizer.decode([int(token_id)]))

    def single_token_id(self, form: str) -> int | None:
        """The id of ``form`` if the tokenizer encodes it as exactly one token, else None."""
        ids = self.encode(form)
        return ids[0] if len(ids) == 1 else None

    # -- forward ------------------------------------------------------------------------

    def _batch(self, rows: list[list[int]], pad_to: int | None) -> tuple[Any, Any, Any]:
        torch = _torch()
        longest = max(len(r) for r in rows)
        length = longest if pad_to is None else int(pad_to)
        if longest > length:
            raise VerifierConfigError(
                f"{self.spec.repo_id}: a prompt is {longest} tokens, above the fixed length "
                f"{length} measured for its call type"
            )
        ids = [[self.pad_id] * (length - len(r)) + r for r in rows]
        mask = [[0] * (length - len(r)) + [1] * len(r) for r in rows]
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(mask, dtype=torch.long, device=self.device)
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        return input_ids, attention_mask, position_ids

    def _last_logits(self, input_ids: Any, attention_mask: Any, position_ids: Any) -> Any:
        torch = _torch()
        if self.model is None:
            raise VerifierConfigError(f"{self.spec.repo_id}: weights not loaded")
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        with torch.no_grad():
            if self._keep_last:
                try:
                    return self.model(**kwargs, logits_to_keep=1).logits[:, -1, :].float()
                except TypeError:
                    self._keep_last = False
            return self.model(**kwargs).logits[:, -1, :].float()

    def verdict_logits(
        self, prompts: Sequence[str], verdict: Verdict, pad_to: int | None
    ) -> list[tuple[float, float]]:
        rows = [self.encode(p) + list(verdict.prefix_ids) for p in prompts]
        n_real = len(rows)
        if self.fixed_shape and n_real < self.batch_size:
            rows = rows + [rows[-1]] * (self.batch_size - n_real)
        logits = self._last_logits(*self._batch(rows, pad_to if self.fixed_shape else None))
        pair = logits[:, [verdict.toxic_id, verdict.safe_id]].cpu().tolist()
        return [(float(a), float(b)) for a, b in pair[:n_real]]

    def greedy(self, prompt: str, max_new_tokens: int = 24) -> list[tuple[int, str]]:
        """Greedy continuation (``do_sample=False``) of one prompt, as ``(id, piece)`` pairs.

        Preflight only: it shows what the model emits before its verdict, which is how the
        verdict prefix and token ids are read off the weights. Never on the scoring path.
        """
        torch = _torch()
        if self.model is None:
            raise VerifierConfigError(f"{self.spec.repo_id}: weights not loaded")
        ids = self.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.pad_id,
            )
        new = [int(t) for t in out[0][len(ids) :].tolist()]
        return [(t, self.token_piece(t)) for t in new]

    # -- lifecycle ----------------------------------------------------------------------

    def close(self) -> None:
        """Free the GPU before the next model is loaded."""
        self.model = None
        gc.collect()
        if self.device != "cpu":
            torch = _torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
