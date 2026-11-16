"""G_fixed: the frozen rewrite policy that builds the state pool (paper Sec. 4.2).

Rules enforced here (each has a test):

* same backbone / version / detox instruction as the controller (config inherits from
  ``models.controller``); reads NO verifier feedback and no A1/B — the prompt has a single
  ``{x0}`` slot and the request carries nothing else. Its decoding block must match the
  controller's too, apart from ``temperature`` and ``seed`` (2026-09-20): on a
  reasoning model a drifting effort tier would make "same backbone, same version" hollow,
  and in Exp 2 the seam sits inside one trajectory -- x1 comes from here, x2.. from the
  controller;
* ``temperature > 0`` (paper Sec. 4.2) -- a configured value must be > 0; ``null`` means the
  parameter is not sent at all, which satisfies the rule through the provider's own default
  of 1 and is recorded as such (:func:`temperature_record`, 2026-09-20);
* one state per x0;
* invalid outputs (refusal / empty / identical to x0 / non-English) trigger regeneration with
  the next seed, up to ``generation.g_fixed.max_attempts``; then the x0 is dropped;
* every attempt is logged as a :class:`GenerationAttempt`.

Decision 4(c) (2026-09-16): Exp 2's x1 is produced by the very same function.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rafr.contracts.schemas import TBD, GenerationAttempt, GFixedOutput, InvalidReason, State
from rafr.core import paths
from rafr.core.seeding import attempt_seed
from rafr.generation.validity import judge_validity, load_refusal_phrases
from rafr.llm.client import LLMClient, LLMRequest, canonical_json, fill_slots, sha256_hex
from rafr.logging.writer import JsonlWriter

PURPOSE = "g_fixed"


class GFixedConfigError(ValueError):
    pass


@dataclass
class GFixedConfig:
    provider: str
    api: str | None
    model: str
    version: str | None
    decoding: dict[str, Any]
    base_seed: int | None
    prompt_template: str
    max_attempts: int
    refusal_phrases: list[str] = field(default_factory=list)
    non_english_threshold: float | None = None
    network: dict[str, Any] = field(default_factory=dict)


def _num_or_none(v: Any) -> float | None:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


def build_config(
    models_cfg: Mapping[str, Any],
    generation_cfg: Mapping[str, Any],
    *,
    allow_tbd: bool = False,
    prompt_text: str | None = None,
    base_seed: int | None = None,
) -> GFixedConfig:
    """Assemble the G_fixed configuration from ``configs/models.yaml`` + ``generation.yaml``.

    With ``allow_tbd=False`` any TBD in the consumed values raises; a configured
    ``temperature`` must be a number > 0, while ``null`` (not sent) is accepted -- see
    :func:`temperature_record`. With ``allow_tbd=True`` (dry run only) TBD values pass through
    untouched so that the fake provider can run end-to-end.
    """
    g = models_cfg["g_fixed"]
    backbone = models_cfg[g.get("inherit_backbone_from", "controller")]
    gen = generation_cfg["g_fixed"]
    decoding = dict(g.get("decoding") or {})

    temp = decoding.get("temperature")
    if temp == TBD:
        if not allow_tbd:
            raise GFixedConfigError("g_fixed.decoding.temperature is TBD")
    elif temp is not None:
        t = _num_or_none(temp)
        if t is None or t <= 0:
            raise GFixedConfigError(f"G_fixed temperature must be > 0 (II.2.3), got {temp!r}")
    # temp is None: the parameter is not sent, paper Sec. 4.2 holds via the provider default. The
    # key stays in `decoding` so that every logged attempt records the null explicitly.
    decoding.setdefault("temperature", None)

    max_attempts = gen.get("max_attempts")
    if max_attempts == TBD or max_attempts is None:
        if not allow_tbd:
            raise GFixedConfigError("generation.g_fixed.max_attempts is TBD")
        max_attempts = 3  # dry-run placeholder; never used in a formal step
    decoding.pop("seed", None)  # not configurable: the caller passes the component seed
    seed = base_seed
    if seed is None:
        if not allow_tbd:
            raise GFixedConfigError(
                "g_fixed base_seed is missing; pass component_seed(master, 'g_fixed')"
            )
        seed = None
    thr = gen.get("validity", {}).get("non_english", {}).get("threshold")
    thr_f = _num_or_none(thr)
    if thr_f is None and not allow_tbd:
        raise GFixedConfigError("generation.g_fixed.validity.non_english.threshold is TBD")

    if prompt_text is None:
        prompt_text = paths.resolve(g["prompt_file"]).read_text(encoding="utf-8")
    phrases_file = gen.get("validity", {}).get("refusal_phrases_file")
    phrases = load_refusal_phrases(paths.resolve(phrases_file)) if phrases_file else []

    return GFixedConfig(
        provider=backbone["provider"],
        api=backbone.get("api"),
        model=backbone["model"],
        version=backbone.get("version"),
        decoding={k: v for k, v in decoding.items() if k != "seed"},
        base_seed=int(seed) if seed is not None else None,
        prompt_template=prompt_text,
        max_attempts=int(max_attempts),
        refusal_phrases=phrases,
        non_english_threshold=thr_f,
        network=dict(backbone.get("network") or {}),
    )


SEED_DERIVATION = "attempt_seed(component_seed(master, 'g_fixed'), x0_id, attempt=k)"
"""How each call's seed is derived (code, not config); part of the config hash below."""


def config_hash_payload(cfg: GFixedConfig) -> dict[str, Any]:
    """Everything that decides what G_fixed generates for a given x0 (paper Sec. 4.2).

    Backbone, version, decoding, the component seed and how per-call seeds derive from it, the
    prompt (sha256, CRLF folded to LF as for E), the regeneration cap and both validity rules
    (refusal prefixes, non-English threshold). The pipeline rebuilds the states with exactly the
    round-1 G_fixed, and this hash is how that is checked (step04 prints and records it).
    """
    prompt = cfg.prompt_template.replace("\r\n", "\n")
    return {
        "provider": cfg.provider,
        "api": cfg.api,
        "model": cfg.model,
        "version": cfg.version,
        "decoding": cfg.decoding,
        "base_seed": cfg.base_seed,
        "seed_derivation": SEED_DERIVATION,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "max_attempts": cfg.max_attempts,
        "refusal_prefixes": list(cfg.refusal_phrases),
        "non_english_threshold": cfg.non_english_threshold,
    }


def config_hash(cfg: GFixedConfig) -> str:
    return sha256_hex(canonical_json(config_hash_payload(cfg)))


def temperature_record(cfg: GFixedConfig) -> dict[str, Any]:
    """What was actually sent for temperature, for the run snapshot (paper Sec. 4.2, 4.6).

    The design asks for ``temperature > 0`` so that regeneration with a new seed can produce a
    different output. The backbone no longer accepts the parameter (2026-09-20: only
    the provider default 1 is allowed), so the rule is satisfied by not sending it -- but a
    bare ``null`` in the logs reads like a forgotten value. This records the difference
    explicitly; ``pipelines/step04_build_states`` writes it into the run snapshot.
    """
    temp = cfg.decoding.get("temperature")
    sent = temp is not None
    return {
        "parameter": "temperature",
        "sent": sent,
        "value": temp,
        "rule": "II.2.3 requires temperature > 0",
        "satisfied_by": (
            f"configured value {temp!r}"
            if sent
            else "the provider default (1), because the model rejects the parameter"
        ),
    }


def build_request(cfg: GFixedConfig, x0_id: str, x0: str, attempt: int) -> LLMRequest:
    """The G_fixed request: prompt with only ``{x0}`` filled; no scores, no observation."""
    seed = (
        attempt_seed(cfg.base_seed, x0_id, attempt=attempt) if cfg.base_seed is not None else None
    )
    return LLMRequest(
        purpose=PURPOSE,
        provider=cfg.provider,
        api=cfg.api,
        model=cfg.model,
        version=cfg.version,
        system=None,
        messages=[{"role": "user", "content": fill_slots(cfg.prompt_template, {"x0": x0})}],
        schema=GFixedOutput,
        decoding=cfg.decoding,
        seed=seed,
        attempt=attempt,
        network=cfg.network,
    )


def state_id_for(x0_id: str) -> str:
    return f"{x0_id}-g1"


def generate_state(
    client: LLMClient,
    cfg: GFixedConfig,
    x0_id: str,
    x0: str,
    *,
    log: JsonlWriter | None = None,
    split: str | None = None,
) -> tuple[State | None, list[GenerationAttempt]]:
    """Generate the single state for ``x0``; returns ``(state or None, attempts)``."""
    attempts: list[GenerationAttempt] = []
    for attempt in range(cfg.max_attempts):
        req = build_request(cfg, x0_id, x0, attempt)
        resp = client.complete(req)
        output = (resp.parsed or {}).get("revised_text") if resp.parsed else None
        reason: InvalidReason | None = judge_validity(
            output,
            x0,
            provider_refused=resp.refused,
            phrases=cfg.refusal_phrases,
            non_english_threshold=cfg.non_english_threshold,
        )
        rec = GenerationAttempt(
            x0_id=x0_id,
            attempt=attempt,
            seed=req.seed if req.seed is not None else -1,
            raw_output=resp.text,
            provider_refused=resp.refused,
            valid=reason is None,
            invalid_reason=reason,
            model=cfg.model,
            model_version=resp.model_version,
            prompt_hash=resp.prompt_hash,
            decoding=cfg.decoding,
            cached=resp.cached,
        )
        attempts.append(rec)
        if log is not None:
            log.write(rec)
        if reason is None:
            assert output is not None
            state = State(
                state_id=state_id_for(x0_id),
                x0_id=x0_id,
                x0=x0,
                text=output,
                t=1,
                source="g_fixed",
                split=split,  # type: ignore[arg-type]
                gen_seed=req.seed,
                gen_attempt=attempt,
            )
            return state, attempts
    return None, attempts


@dataclass
class PoolResult:
    states: list[State]
    dropped: list[dict[str, Any]]  # {"x0_id", "reasons": [...]}
    n_attempts: int


def generate_pool(
    client: LLMClient,
    cfg: GFixedConfig,
    x0s: Iterable[tuple[str, str]],
    *,
    log: JsonlWriter | None = None,
    split: str | None = None,
    existing: Mapping[str, State] | None = None,
) -> PoolResult:
    """Generate one state per x0. ``existing`` states are reused (resume support)."""
    states: list[State] = []
    dropped: list[dict[str, Any]] = []
    n_attempts = 0
    for x0_id, x0 in x0s:
        if existing and x0_id in existing:
            states.append(existing[x0_id])
            continue
        state, attempts = generate_state(client, cfg, x0_id, x0, log=log, split=split)
        n_attempts += len(attempts)
        if state is None:
            dropped.append(
                {
                    "x0_id": x0_id,
                    "reasons": [a.invalid_reason.value for a in attempts if a.invalid_reason],
                }
            )
        else:
            states.append(state)
    return PoolResult(states=states, dropped=dropped, n_attempts=n_attempts)


def read_prompt(path: str | Path) -> str:
    return paths.resolve(path).read_text(encoding="utf-8")
