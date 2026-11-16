"""E, the independent evaluator (II.9.1).

* ONE E (model, version, prompt file, decoding) for all four uses -- x0 prefilter, y* of every
  state, the Exp 1 outcome, the Exp 2 round judgments (v0.9). There is one config block,
  ``models.evaluator``, and one builder, :func:`build_config`; no step can configure its own E;
* the prompt is the authors' rubric-free ``src/rafr/evaluator/prompts/annotator_arm_b.txt``
  (C.14). Its sha256 is pinned in
  ``models.evaluator.prompt_sha256``; :func:`build_config` recomputes it from the file and
  refuses to build E on a mismatch, and a prompt with a ``{scale}`` slot is refused outright;
* every :class:`EvalRecord` carries the prompt-file hash and the configured version, and
  :func:`check_same_e` refuses records produced by a different E (E changed -> the whole chain
  is rebuilt, II.9.1; nothing is reused across an E change);
* **called exactly as the retest called it** (2026-09-23): plain prompt, no system
  prompt, one user message, no structured output; the answer text is parsed here by
  :func:`parse_response`, a copy of the retest's parser (kappa = .945 and the arm A / B
  comparison were measured under this calling convention). :func:`request_body` returns the
  exact SDK arguments, so a dry run can print one next to the retest's;
* binary output (the parser reads ``label`` and the optional one-sentence ``reason``);
* **sees text only**: the function signature takes a string and nothing else — no arm, no
  verifier score, no observation. ``tests/test_boundaries.py`` checks that this module never
  imports the formatter, the channel or the controller.

Prompt caching (2026-09-20): the prompt is sent as two content blocks of ONE user
message, the stable prefix marked ``cache_control: ephemeral`` and the tail that starts with the
text. Their concatenation is byte-for-byte :func:`render_prompt`, so this changes billing, not
what E is asked. With the rubric-free prompt the prefix is a few dozen tokens, below the
model's minimum cacheable prefix, so the provider ignores the marker and every call is billed
uncached (2026-09-23: run it like that; the prompt is not changed to make caching
work).

Failure handling (II.9.1): a refusal (provider signal) or an answer the parser cannot read is
retried up to
``PARSE_MAX_RETRIES`` more times with the attempt index bumped (fresh cache key); if every
attempt fails, :class:`EvaluatorError` is raised so that a run stops loudly instead of
silently coding the text. The step00 prefilter catches it (the x0 is dropped and the reason
recorded), and so does step05's state judgment (the state and its x0 are dropped before the
split); Exp 2's round judgments let it abort the step.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from rafr.contracts.schemas import (
    PARSE_MAX_RETRIES,
    TBD,
    EvalRecord,
    SafetyLabel,
    text_hash,
)
from rafr.core import paths
from rafr.llm.client import LLMClient, LLMRequest, canonical_json, fill_slots, sha256_hex
from rafr.logging.writer import JsonlWriter

PURPOSE = "evaluator"
SCALE_SLOT = "{scale}"  # only to refuse it: E is rubric-free since v0.9 (C.14)
TEXT_SLOT = "{text}"

PROMPT_CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}
"""Marker on the stable part of the judge prompt: 5-minute TTL, refreshed by every read."""

CACHE_MIN_PREFIX_TOKENS = 512
"""claude-opus-5's minimum cacheable prefix. A shorter prefix is not an error — the marker is
simply ignored and ``cache_creation_input_tokens`` stays 0."""

CHARS_PER_TOKEN_UPPER = 4
"""Generous chars-per-token for English, used only to say up front that a prefix is too short
to be cached: fewer than 512 * 4 characters cannot reach 512 tokens."""

E_IDENTITY_KEYS: tuple[str, ...] = ("model", "version", "decoding", "prompt_sha256")
"""What makes two E the same E (II.9.1). The provider name is implied by the model."""

LABELS: tuple[str, ...] = ("SAFE", "UNSAFE")

# The retest's parser (annotation/annotator_selection/prompting.py, PARSER_VERSION 2), copied
# rather than imported: the formal pipeline does not import the retired annotation package.
# tests/test_evaluator.py checks that the two give the same answer on the same inputs.
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_response(text: str | None) -> tuple[str | None, str | None]:
    """``(label, reason)`` from E's answer text; ``(None, None)`` when there is no verdict.

    The first JSON object carrying a string ``label`` that is SAFE or UNSAFE after strip +
    upper-case wins, fenced or surrounded by other text; ``reason`` is kept when it is a string
    and never required. Anything else -- a bare ``SAFE`` with no JSON around it included -- is
    a parse failure, retried like a refusal (II.9.1).
    """
    if not text:
        return None, None
    stripped = _FENCE.sub("", text.strip())
    for cand in [stripped, *_OBJECT.findall(stripped)]:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("label"), str):
            lab = obj["label"].strip().upper()
            if lab in LABELS:
                reason = obj.get("reason")
                return lab, (reason.strip() if isinstance(reason, str) else None)
    return None, None


class EvaluatorError(RuntimeError):
    """E gave no verdict after every attempt; ``failures`` holds each attempt's reason."""

    def __init__(self, message: str, failures: list[str] | None = None) -> None:
        super().__init__(message)
        self.failures = list(failures or [])


class EvaluatorConfigError(ValueError):
    pass


class EChangedError(EvaluatorConfigError):
    """E output on disk was produced by a different E than the one configured now."""


def prompt_sha256(text: str) -> str:
    """sha256 of the prompt text as UTF-8 with CRLF folded to LF.

    Line endings are folded because git checks the file out as CRLF on Windows and LF on the
    Linux GPU box (``.gitattributes: text=auto``) while ``read_text`` hands both over as LF; the
    hash therefore equals the sha256 of the bytes git stores and is the same on every machine.
    """
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


@dataclass
class EvaluatorConfig:
    provider: str
    api: str | None
    model: str
    version: str | None
    decoding: dict[str, Any]
    prompt_template: str
    prompt_sha256: str = ""  # of the prompt file; computed from the template when not given
    network: dict[str, Any] = field(default_factory=dict)
    prompt_cache: bool = True  # mark the stable prefix of the prompt as cacheable

    def __post_init__(self) -> None:
        if not self.prompt_sha256:
            self.prompt_sha256 = prompt_sha256(self.prompt_template)

    def identity(self) -> dict[str, Any]:
        """The E identity (:data:`E_IDENTITY_KEYS`). Every artifact E writes records it, and
        every step that reads E output back compares it with this (:func:`check_same_e`)."""
        return {
            "model": self.model,
            "version": self.version,
            "decoding": self.decoding,
            "prompt_sha256": self.prompt_sha256,
        }

    def identity_hash(self) -> str:
        return sha256_hex(canonical_json(self.identity()))

    def config_hash_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            **self.identity(),
            # how the prompt is sent, not what it says: one block, or a cacheable prefix
            # block plus the text (2026-09-20). Recorded because II.9.1 pins E's calling
            # convention; not part of the identity because E reads the same bytes either way.
            "prompt_cache": PROMPT_CACHE_CONTROL if self.prompt_cache else None,
        }

    def prefix_too_short_to_cache(self) -> bool:
        """True when the stable prefix cannot reach :data:`CACHE_MIN_PREFIX_TOKENS` even at a
        generous chars-per-token ratio, i.e. caching is known not to apply."""
        head = self.prompt_template.partition(TEXT_SLOT)[0]
        return len(head) < CACHE_MIN_PREFIX_TOKENS * CHARS_PER_TOKEN_UPPER


# E's decoding is taken from the config verbatim (2026-09-20). This model generation
# rejects temperature outright, so `configs/models.yaml` sets `temperature: null` (= not sent,
# recorded as null); the provider refuses a non-null temperature rather than dropping it
# silently (decision 14). `effort: null` (not sent) and `max_output_tokens: 4096` are the
# retest's values (2026-09-23; II.9.1 v0.9).


def build_config(
    models_cfg: Mapping[str, Any],
    *,
    allow_tbd: bool = False,
    prompt_text: str | None = None,
) -> EvaluatorConfig:
    """The one E, from ``models.evaluator``.

    Reading the prompt from ``models.evaluator.prompt_file`` (the normal path) also checks its
    sha256 against ``models.evaluator.prompt_sha256``: a mismatch -- the file was edited, or
    the pin is stale -- refuses to build E at all, dry run included, because every E output is
    keyed on that hash. A ``{scale}`` slot or a configured ``scale_file`` is refused too (v0.9,
    C.14: E has no rubric). ``prompt_text`` injects a template directly (tests only); its hash
    is then computed, not checked.
    """
    e = models_cfg["evaluator"]
    decoding = dict(e.get("decoding") or {})
    if not allow_tbd:
        for k in ("model", "version", "prompt_sha256"):
            if e.get(k) in (None, TBD, ""):
                raise EvaluatorConfigError(f"models.evaluator.{k} is TBD")
        for k, v in decoding.items():
            if v == TBD:
                raise EvaluatorConfigError(f"models.evaluator.decoding.{k} is TBD")
    if e.get("scale_file"):
        raise EvaluatorConfigError(
            "models.evaluator.scale_file is set, but E is rubric-free since protocol v0.9 "
            "(C.14); the retired scale lives in archive/round1/prompts/ and is never read"
        )
    if e.get("structured_output"):
        raise EvaluatorConfigError(
            "models.evaluator.structured_output is true, but E is called exactly as the retest "
            "called it: plain prompt, own parser, no structured output (2026-09-23)"
        )
    digest = ""
    if prompt_text is None:
        prompt_text = paths.resolve(e["prompt_file"]).read_text(encoding="utf-8")
        digest = prompt_sha256(prompt_text)
        pinned = e.get("prompt_sha256")
        if pinned not in (None, TBD, "") and pinned != digest:
            raise EvaluatorConfigError(
                f"E prompt hash mismatch: {e['prompt_file']} hashes to {digest}, but "
                f"models.evaluator.prompt_sha256 pins {pinned}. Refusing to run: every E "
                "output is keyed on this hash (II.9.1). If the prompt really changed, the "
                "whole chain is rebuilt on the new E (archive the old one first)."
            )
    if SCALE_SLOT in prompt_text:
        raise EvaluatorConfigError(
            f"E prompt contains {SCALE_SLOT}: E is rubric-free since protocol v0.9 (C.14)"
        )
    version = e.get("version")
    return EvaluatorConfig(
        provider=e["provider"],
        api=e.get("api"),
        model=str(e["model"]),
        version=None if version in (None, TBD) else str(version),
        decoding=decoding,
        prompt_template=prompt_text,
        prompt_sha256=digest,
        network=dict(e.get("network") or {}),
    )


def record_identity(rec: EvalRecord | Mapping[str, Any]) -> dict[str, Any]:
    """The E identity an EvalRecord (model object or dict) names."""
    if isinstance(rec, Mapping):
        d: Mapping[str, Any] = rec
    else:
        d = rec.model_dump(mode="json")
    return {
        "model": d.get("model"),
        "version": d.get("e_version"),
        "decoding": d.get("decoding"),
        "prompt_sha256": d.get("e_prompt_sha256"),
    }


def check_same_e(recorded: Mapping[str, Any], cfg: EvaluatorConfig, *, where: str) -> None:
    """Refuse E output that a different E produced (II.9.1, v0.9).

    ``recorded`` is an identity as :meth:`EvaluatorConfig.identity` or :func:`record_identity`
    gives it. Model, version, decoding and prompt hash must all match.
    """
    mine = cfg.identity()
    diff = {k: (recorded.get(k), mine[k]) for k in E_IDENTITY_KEYS if recorded.get(k) != mine[k]}
    if diff:
        lines = "\n".join(
            f"    {k}: on disk {a!r} != configured {b!r}" for k, (a, b) in diff.items()
        )
        raise EChangedError(
            f"E changed since {where} was written:\n{lines}\n"
            "Prefilter, y* and outcome must come from ONE E (II.9.1). Nothing is reused across "
            "an E change: archive the old artifacts and rebuild the chain from step 0."
        )


def render_prompt(cfg: EvaluatorConfig, text: str) -> str:
    """Fill ``{text}`` in one pass: a text that itself contains ``{text}`` is judged
    verbatim, never expanded."""
    return fill_slots(cfg.prompt_template, {"text": text})


def render_parts(cfg: EvaluatorConfig, text: str) -> tuple[str, str]:
    """Split the rendered prompt at the text under judgment: ``(stable prefix, tail)``.

    ``prefix + tail == render_prompt(cfg, text)`` exactly, for every text (checked by test).
    The prefix is identical on every call and is what gets the cache marker; the tail starts
    with the sentence being judged and therefore changes every time.
    """
    head_tmpl, slot, tail_tmpl = cfg.prompt_template.partition(TEXT_SLOT)
    if not slot:  # no {text} slot: nothing varies, so there is nothing to split off
        return cfg.prompt_template, ""
    return head_tmpl, text + fill_slots(tail_tmpl, {"text": text})


def build_messages(cfg: EvaluatorConfig, text: str) -> list[dict[str, Any]]:
    """The one user message of a judge call, as one text block or as prefix + tail blocks."""
    prefix, tail = render_parts(cfg, text)
    if not cfg.prompt_cache or not prefix or not tail:
        return [{"role": "user", "content": render_prompt(cfg, text)}]
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prefix, "cache_control": dict(PROMPT_CACHE_CONTROL)},
                {"type": "text", "text": tail},
            ],
        }
    ]


def build_request(cfg: EvaluatorConfig, text: str, attempt: int = 0) -> LLMRequest:
    return LLMRequest(
        purpose=PURPOSE,
        provider=cfg.provider,
        api=cfg.api,
        model=cfg.model,
        version=cfg.version,
        system=None,
        messages=build_messages(cfg, text),
        schema=None,  # plain prompt, parsed by parse_response -- as in the retest
        decoding=cfg.decoding,
        seed=None,  # Anthropic Messages API has no seed (configs/models.yaml)
        attempt=attempt,
        network=cfg.network,
    )


def request_body(cfg: EvaluatorConfig, text: str, attempt: int = 0) -> dict[str, Any]:
    """The exact keyword arguments the Anthropic SDK receives for judging ``text`` (no key).

    For the dry run's printout next to the retest's request (2026-09-23).
    """
    from rafr.llm.providers.anthropic import request_kwargs

    return request_kwargs(build_request(cfg, text, attempt))


def judge_text(client: LLMClient, cfg: EvaluatorConfig, text: str) -> EvalRecord:
    """Judge one text. Takes the text and nothing else (II.9.1)."""
    return judge_with_failures(client, cfg, text)[0]


def judge_with_failures(
    client: LLMClient, cfg: EvaluatorConfig, text: str
) -> tuple[EvalRecord, list[str]]:
    """:func:`judge_text` plus the reason of every failed attempt before the verdict
    (``refusal`` / ``unparseable`` / ``empty``) -- the parse retries a step reports.

    A transient provider failure (429, network, 5xx) is not an attempt: the client retries it
    below this level, and if it gives up, its :class:`TransientRetriesExhaustedError` passes
    through untouched -- it is neither a parse retry nor an E failure (II.8.2, 2026-09-23).
    """
    failures: list[str] = []
    for attempt in range(PARSE_MAX_RETRIES + 1):
        req = build_request(cfg, text, attempt)
        resp = client.complete(req)
        if resp.refused:
            failures.append("refusal")
            continue
        label, reason = parse_response(resp.text)
        if label is None:
            failures.append(
                resp.error or ("empty" if not (resp.text or "").strip() else "unparseable")
            )
            continue
        rec = EvalRecord(
            text_id=text_hash(text),
            text=text,
            label=SafetyLabel(label),
            score=None,
            reason=reason,
            model=cfg.model,
            model_version=resp.model_version,
            prompt_hash=resp.prompt_hash,
            e_prompt_sha256=cfg.prompt_sha256,
            e_version=cfg.version,
            decoding=cfg.decoding,
            cached=resp.cached,
        )
        return rec, failures
    raise EvaluatorError(
        f"E failed on text {text_hash(text)} after {PARSE_MAX_RETRIES + 1} attempts: "
        f"{failures[-1] if failures else 'no attempt'}",
        failures,
    )


def judge_many(
    client: LLMClient,
    cfg: EvaluatorConfig,
    texts: Iterable[str],
    *,
    log: JsonlWriter | None = None,
) -> list[EvalRecord]:
    out: list[EvalRecord] = []
    for t in texts:
        rec = judge_text(client, cfg, t)
        if log is not None:
            log.write(rec)
        out.append(rec)
    return out
