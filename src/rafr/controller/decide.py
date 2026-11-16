"""One accept / rewrite decision (paper Sec. 3, 4.6).

The controller receives ``x0``, ``x_t`` and an already-formatted observation STRING. It never
sees an ``Observation`` object, scores, labels, arm or lambda: those are attached to the log
by the caller. There is no arm branch here (paper App. E; enforced by tests/test_boundaries.py).

Parse-failure handling (fixed before the runs, paper Sec. 4.6):

* structured output (JSON schema) on every call;
* at most ``PARSE_MAX_RETRIES`` retries; a retry re-sends the ORIGINAL request with the fixed
  retry sentence (same for every arm) appended at the end of the user turn — the previous
  raw output is NOT carried over (decision, 2026-09-16). The attempt index enters
  the seed and the cache key, so every retry is an independent fresh sample;
* still failing -> ``parse_fail``; every attempt is logged.

Seed derivation (design decision): ``per_state`` seeds are derived from
``(base_seed, experiment, state_id, t)``, where ``base_seed`` is
``component_seed(master, "controller")`` — the experiment name (``exp1`` / ``exp2``) makes the
two experiments' samples independent and keeps their cache keys apart.

The system prompt has one substitution slot, ``{k}``: the decision budget the model is told
about. It is filled by the EXPERIMENT RUNNER through :func:`with_round_budget` (the authors,
2026-09-18): Exp 1 passes 1 (one decision per state, paper Sec. 4.6) and Exp 2 passes
``K_MAX_ROUNDS`` (paper Sec. 4.4). Telling the single-shot Exp 1 controller that it has four
decisions would bias it towards ``rewrite``, and the accept rate is the primary outcome.
:func:`decide` refuses a config whose slot is still unfilled. The user turn (rewrite template) has
``{x0}``, ``{x_t}`` and ``{observation}``, filled in one pass (:func:`rafr.llm.client.fill_slots`)
so that a text containing a slot name is copied verbatim.

The part of the rewrite template that only makes sense when x0 is shown -- the sentence
explaining why the original is there and the "Original sentence" block -- is fenced by two
marker lines, ``{x0_section_start}`` / ``{x0_section_end}``. With ``show_x0: true`` the
markers are dropped and the section stays; with ``show_x0: false`` the whole section goes
and the neighbouring paragraphs close up over it, so the model never sees an empty block
(review, 2026-09-18). A template that has ``{x0}`` outside the fence cannot be run with
``show_x0: false``: :func:`build_config` refuses rather than blanking the slot.

Network / rate-limit retries live in the provider SDK and are not counted here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from rafr.contracts.schemas import (
    PARSE_MAX_RETRIES,
    TBD,
    Arm,
    ControllerAttempt,
    ControllerOutput,
    Decision,
    Lambda,
)
from rafr.controller.parse import ParsedAttempt, parse_attempt
from rafr.core import paths
from rafr.core.seeding import derive_seed
from rafr.generation.validity import load_refusal_phrases
from rafr.llm.client import LLMClient, LLMRequest, fill_slots

PURPOSE = "controller"
SEED_SCHEMES = ("fixed", "per_state", "per_cell")
SYSTEM_K_SLOT = "{k}"
X0_SECTION_START = "{x0_section_start}"
X0_SECTION_END = "{x0_section_end}"


class ControllerConfigError(ValueError):
    pass


@dataclass
class ControllerConfig:
    provider: str
    api: str | None
    model: str
    version: str | None
    decoding: dict[str, Any]
    base_seed: int | None
    seed_scheme: str
    system_prompt: str
    rewrite_template: str
    retry_prompt: str
    show_x0: bool = True
    refusal_phrases: list[str] = field(default_factory=list)
    network: dict[str, Any] = field(default_factory=dict)
    round_budget: int | None = None  # set by with_round_budget; None = {k} not yet filled


def build_config(
    models_cfg: Mapping[str, Any],
    exp_cfg: Mapping[str, Any],
    *,
    base_seed: int | None = None,
    allow_tbd: bool = False,
    system_prompt: str | None = None,
    rewrite_template: str | None = None,
    retry_prompt: str | None = None,
    refusal_phrases_file: str | None = None,
) -> ControllerConfig:
    c = models_cfg["controller"]
    decoding = dict(c.get("decoding") or {})
    decoding.pop("seed", None)  # not configurable: the caller passes the component seed
    seed = base_seed
    scheme = c.get("seed_scheme", TBD)
    if not allow_tbd:
        for k in ("model", "version"):
            if c.get(k) in (None, TBD):
                raise ControllerConfigError(f"models.controller.{k} is TBD")
        for k, v in decoding.items():
            if v == TBD:
                raise ControllerConfigError(f"models.controller.decoding.{k} is TBD")
        if seed in (None, TBD):
            raise ControllerConfigError(
                "controller base_seed is missing; pass component_seed(master, 'controller')"
            )
        if scheme not in SEED_SCHEMES:
            raise ControllerConfigError(
                f"models.controller.seed_scheme must be one of {SEED_SCHEMES}"
            )
    if scheme not in SEED_SCHEMES:
        scheme = "fixed"  # dry-run fallback only
    if system_prompt is None:
        system_prompt = paths.resolve(c["system_prompt_file"]).read_text(encoding="utf-8")
    if rewrite_template is None:
        rewrite_template = paths.resolve(c["rewrite_prompt_file"]).read_text(encoding="utf-8")
    if retry_prompt is None:
        retry_prompt = paths.resolve(c["parse_retry_prompt_file"]).read_text(encoding="utf-8")
    phrases = (
        load_refusal_phrases(paths.resolve(refusal_phrases_file)) if refusal_phrases_file else []
    )
    show_x0 = bool(exp_cfg.get("controller", {}).get("show_x0", True))
    return ControllerConfig(
        provider=c["provider"],
        api=c.get("api"),
        model=str(c["model"]),
        version=c.get("version"),
        decoding=decoding,
        base_seed=int(seed) if isinstance(seed, int) else None,
        seed_scheme=scheme,
        system_prompt=system_prompt,
        rewrite_template=resolve_x0_section(rewrite_template, show_x0=show_x0),
        retry_prompt=retry_prompt.strip(),
        show_x0=show_x0,
        refusal_phrases=phrases,
        network=dict(c.get("network") or {}),
    )


def resolve_x0_section(template: str, *, show_x0: bool) -> str:
    """Apply ``show_x0`` to the rewrite template once, at config time.

    * ``show_x0=True``: the two marker lines are removed, everything between them stays.
    * ``show_x0=False``: the marker lines and everything between them are removed; blank
      lines around the gap collapse to a single blank line, so no empty block remains.

    Markers must come as one balanced pair on lines of their own. A template that keeps
    ``{x0}`` outside the fenced section cannot hide x0 and is refused.
    """
    lines = template.split("\n")
    starts = [i for i, ln in enumerate(lines) if ln.strip() == X0_SECTION_START]
    ends = [i for i, ln in enumerate(lines) if ln.strip() == X0_SECTION_END]
    if len(starts) != len(ends) or len(starts) > 1 or (starts and starts[0] > ends[0]):
        raise ControllerConfigError(
            f"rewrite template must contain at most one {X0_SECTION_START} / "
            f"{X0_SECTION_END} pair, each on its own line, start before end"
        )
    if starts:
        i, j = starts[0], ends[0]
        if show_x0:
            lines = lines[:i] + lines[i + 1 : j] + lines[j + 1 :]
        else:
            before, after = lines[:i], lines[j + 1 :]
            while before and not before[-1].strip():
                before.pop()
            while after and not after[0].strip():
                after.pop(0)
            lines = before + ([""] if before and after else []) + after
    out = "\n".join(lines)
    if not show_x0 and "{x0}" in out:
        raise ControllerConfigError(
            "show_x0 is false but the rewrite template has {x0} outside the "
            f"{X0_SECTION_START} / {X0_SECTION_END} section; refusing to render a blank slot"
        )
    return out


@dataclass(frozen=True)
class DecisionLabels:
    """Pass-through identifiers for the log and the seed. The controller does not read them.

    ``experiment`` is the seed namespace (``"exp1"`` / ``"exp2"``): it is fixed by the runner
    module, not by the run's log label, so a dry run of Exp 1 and Exp 2 also draws
    independently. ``state_id`` is the seed's state key: Exp 1 passes the D_Exp1 state id,
    Exp 2 (and its smoke) the x0 id -- never an id that carries the arm (paper Sec. 4.6).
    """

    state_id: str
    arm: Arm
    lam: Lambda
    t: int
    experiment: str


def call_seed(cfg: ControllerConfig, labels: DecisionLabels) -> int | None:
    """Seed of attempt 0 for this decision (design decision: experiment enters the key)."""
    if cfg.base_seed is None:
        return None
    if cfg.seed_scheme == "fixed":
        return cfg.base_seed
    if cfg.seed_scheme == "per_state":
        return derive_seed(
            cfg.base_seed, "controller", labels.experiment, labels.state_id, labels.t
        )
    return derive_seed(
        cfg.base_seed,
        "controller",
        labels.experiment,
        labels.state_id,
        labels.arm.value,
        labels.lam.value,
        labels.t,
    )


def attempt_call_seed(seed: int | None, attempt: int) -> int | None:
    """Seed of the ``attempt``-th parse retry: attempt 0 keeps the call seed, every retry is
    an independent derivation (the attempt index is also part of the cache key)."""
    if seed is None or attempt == 0:
        return seed
    return derive_seed(seed, "attempt", attempt)


SYSTEM_K_NOUN = " decisions"
"""The word after the ``{k}`` slot in system.txt ("at most {k} decisions in total")."""


def render_system_prompt(template: str, k: int) -> str:
    """Fill the ``{k}`` slot with the caller's decision budget. With K = 1 the noun after the
    slot is singular ("at most 1 decision in total", 2026-09-25); every
    other character is kept, and the prompt file itself is unchanged."""
    k = int(k)
    if k == 1:
        template = template.replace(SYSTEM_K_SLOT + SYSTEM_K_NOUN, SYSTEM_K_SLOT + " decision")
    return template.replace(SYSTEM_K_SLOT, str(k))


def with_round_budget(cfg: ControllerConfig, k: int) -> ControllerConfig:
    """The config an experiment actually runs with: ``{k}`` filled with that experiment's
    decision budget (Exp 1: 1; Exp 2: ``K_MAX_ROUNDS``). Pure; the input is untouched."""
    if k < 1:
        raise ControllerConfigError(f"round budget must be >= 1, got {k}")
    return replace(cfg, system_prompt=render_system_prompt(cfg.system_prompt, k), round_budget=k)


def render_user_turn(
    cfg: ControllerConfig, x0: str, x_t: str, observation: str, *, attempt: int = 0
) -> str:
    """Fill the three user-turn slots in one pass. The template was already resolved for
    ``show_x0`` by :func:`build_config`, so ``{x0}`` is either present and filled or absent."""
    out = fill_slots(cfg.rewrite_template, {"x0": x0, "x_t": x_t, "observation": observation})
    if attempt > 0:
        # parse retry: the original request with the fixed sentence appended at the end
        out = out.rstrip() + "\n\n" + cfg.retry_prompt
    return out


def build_request(
    cfg: ControllerConfig,
    messages: list[dict[str, str]],
    seed: int | None,
    attempt: int,
) -> LLMRequest:
    return LLMRequest(
        purpose=PURPOSE,
        provider=cfg.provider,
        api=cfg.api,
        model=cfg.model,
        version=cfg.version,
        system=cfg.system_prompt,
        messages=messages,
        schema=ControllerOutput,
        decoding=cfg.decoding,
        seed=seed,
        attempt=attempt,
        network=cfg.network,
    )


def decide(
    client: LLMClient,
    cfg: ControllerConfig,
    *,
    x0: str,
    x_t: str,
    observation: str,
    labels: DecisionLabels,
) -> Decision:
    """Make one decision. ``observation`` is the formatter's output string, nothing else."""
    if SYSTEM_K_SLOT in cfg.system_prompt:
        raise ControllerConfigError(
            f"system prompt still contains {SYSTEM_K_SLOT}: the experiment runner must call "
            "with_round_budget(cfg, k) before deciding"
        )
    base = call_seed(cfg, labels)
    attempts: list[ControllerAttempt] = []
    parsed: ParsedAttempt | None = None
    model_version: str | None = None
    for attempt in range(PARSE_MAX_RETRIES + 1):
        # every attempt is a fresh single-turn request: original prompt (+ fixed retry
        # sentence when attempt > 0), its own seed, its own cache key
        seed = attempt_call_seed(base, attempt)
        messages = [
            {
                "role": "user",
                "content": render_user_turn(cfg, x0, x_t, observation, attempt=attempt),
            }
        ]
        req = build_request(cfg, messages, seed, attempt)
        resp = client.complete(req)
        model_version = resp.model_version
        parsed = parse_attempt(resp, x_t, cfg.refusal_phrases)
        attempts.append(
            ControllerAttempt(
                attempt=attempt,
                raw_output=resp.text,
                provider_refused=resp.refused,
                parsed=parsed.parsed,
                parse_fail_reason=parsed.fail_reason,
                prompt_hash=resp.prompt_hash,
                seed=seed,
                cached=resp.cached,
            )
        )
        if parsed.ok:
            break
    assert parsed is not None
    n_retries = len(attempts) - 1
    if parsed.ok:
        return Decision(
            state_id=labels.state_id,
            arm=labels.arm,
            lam=labels.lam,
            t=labels.t,
            action=parsed.action,
            revised_text=parsed.revised_text,
            parse_fail=False,
            n_parse_retries=n_retries,
            refusal_like=parsed.refusal_like,
            empty_rewrite=parsed.empty_rewrite,
            attempts=attempts,
            model=cfg.model,
            model_version=model_version,
            decoding=cfg.decoding,
        )
    return Decision(
        state_id=labels.state_id,
        arm=labels.arm,
        lam=labels.lam,
        t=labels.t,
        action=None,
        revised_text=None,
        parse_fail=True,
        n_parse_retries=n_retries,
        refusal_like=False,
        empty_rewrite=False,
        attempts=attempts,
        model=cfg.model,
        model_version=model_version,
        decoding=cfg.decoding,
    )
