"""Data contracts shared by every module (Part II).

This module holds:

* the constants LOCKED by the design (never tunables);
* the pydantic models for State, Score, Observation, Decision, EvalRecord, LogRecord;
* the structured-output schemas that the controller and the evaluator must emit.

Nothing here depends on any other ``rafr`` package, so every package can import it
without creating a cycle and without crossing the import boundaries of paper App. E, Sec. 4.3.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------------------
# Constants locked by the design. These are NOT configuration.
# --------------------------------------------------------------------------------------

K_MAX_ROUNDS: int = 4
"""paper Sec. 4.4 — maximum number of controller decisions per trajectory in Exp 2."""

PARSE_MAX_RETRIES: int = 2
"""paper Sec. 4.6 — parse-failure retries (3 attempts in total). Network retries are separate."""

PARSE_FAIL_ALERT_THRESHOLD: float = 0.02
"""paper Sec. 4.6 — any reported parse_fail cell above this rate is flagged loudly. The reporting
grid differs by experiment: Exp 1 is arm x lambda, Exp 2 is arm x round."""

BOOTSTRAP_B: int = 10000
"""paper Sec. 4.6 — replicates for every 95% percentile bootstrap CI in the analysis and in
the gate report. The resampling UNIT is per analysis (x0 in Exp 1, trajectory stratified by
arm in Exp 2, state in the gate); the seed is configuration and is written to disk."""

BOOTSTRAP_ALPHA: float = 0.05
"""paper Sec. 4.6 — 95% percentile interval. CIs express uncertainty only: the pilot makes
no significance claims and the gate verdict uses point estimates (paper Sec. 4.3)."""

GATE_MAX_ATTEMPTS: int = 2
"""paper Sec. 4.3 — the manipulation-validity gate may be executed at most twice."""

TAU_CRITERION: str = "f1_max_plateau_midpoint"
"""paper Sec. 5 — tau_i maximizes F1 on D_fit; on a plateau, take the midpoint. F2 is
reported as an auxiliary metric and takes no part in the choice."""

SCORE_MIN: float = 0.0
SCORE_MAX: float = 1.0
"""Decision 5(a), 2026-09-16 — verifier scores live in [0, 1], higher = more toxic."""

TBD = "TBD"
"""Sentinel for undecided config values (configs/*.yaml)."""

PLACEHOLDER = "PLACEHOLDER"
"""Sentinel that must appear in every prompt / template file the authors have not written yet."""


# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------


class Arm(StrEnum):
    """Representation arms (paper Sec. 3). Only the formatter may branch on this."""

    A1 = "A1"
    B = "B"
    PERMUTED_B = "permuted_B"
    # Post-pilot extension (2026-10-31), never part of the primary runs: the
    # controller sees no verifier output; B without the two distances; B plus pi-hat; B plus
    # another state's pi-hat. Members only: no existing value, prompt or seed changes (the seed
    # carries no arm), and decide.Decision / LogRecord validate arm against this enum.
    NO_VERIFIER = "no_verifier"
    VERDICT_ONLY = "verdict_only"
    B_PIHAT = "B_pihat"
    B_PIHAT_MISMATCHED = "B_pihat_mismatched"
    ORACLE_FEEDBACK = "oracle_feedback"  # Math correction ceiling; the authors approved 2026-11-01.


class Lambda(StrEnum):
    """Evidence-degradation severity levels (paper Sec. 4.3). ``NONE`` = identity channel."""

    NONE = "none"
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class Action(StrEnum):
    ACCEPT = "accept"
    REWRITE = "rewrite"


class SafetyLabel(StrEnum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"


class MathLabel(StrEnum):
    """Math-Verify outcomes (paper App. D.1, E / paper App. D.1).

    UNRESOLVED is an invalid initial answer, but a terminal task failure in Exp 2.
    It is never silently converted into the binary INCORRECT initial-state label.
    """

    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    UNRESOLVED = "UNRESOLVED"


class Split(StrEnum):
    """paper Sec. 4.2 — all splits are group-level by x0 and pairwise disjoint."""

    D_FIT = "d_fit"
    D_GATE = "d_gate"
    D_GATE_RESERVE = "d_gate_reserve"
    D_EXP1 = "d_exp1"
    UNLABELED_POOL = "unlabeled_pool"
    # Exp 2 draws its x0 from D_Exp1 (decision, group 6d, 2026-09-16); the former `fresh`
    # source and its `exp2` split were removed.


class ConflictDirection(StrEnum):
    """paper Sec. 3 — categorical (y1, y2) conflict; ``NA`` when the verifiers agree (never
    blank)."""

    V1_UNSAFE_V2_SAFE = "v1_unsafe_v2_safe"  # (y1, y2) = (1, 0)
    V1_SAFE_V2_UNSAFE = "v1_safe_v2_unsafe"  # (y1, y2) = (0, 1)
    NA = "N/A"


class InvalidReason(StrEnum):
    """paper Sec. 4.2 — why a G_fixed output was judged invalid."""

    REFUSAL = "refusal"
    EMPTY = "empty"
    IDENTICAL_TO_X0 = "identical_to_x0"
    NON_ENGLISH = "non_english"
    # Mathematics only; the detox validity rules and defaults are unchanged (paper App. D.1).
    TRUNCATED = "truncated"
    STRUCTURED_OUTPUT = "structured_output"
    UNRESOLVED = "unresolved"
    E_ERROR = "e_error"


class ParseFailReason(StrEnum):
    """paper Sec. 4.6 — the three parse-failure categories fixed before the runs, plus one guard.

    ``MALFORMED_JSON`` is not in paper Sec. 4.6: with schema-enforced structured output it should
    never occur (only truncation at max_output_tokens can produce it). It is kept as its own
    category so it is never silently folded into ``BAD_ACTION``; it counts as a parse
    failure and is reported separately. Flagged to the authors.
    """

    REFUSAL = "refusal"
    BAD_ACTION = "action_not_in_accept_rewrite"
    EMPTY_REVISED_TEXT = "rewrite_with_empty_revised_text"
    MALFORMED_JSON = "malformed_json"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def text_hash(text: str) -> str:
    """Stable identifier for a piece of text (sha256, hex, 16 chars)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


# --------------------------------------------------------------------------------------
# Structured outputs emitted by LLMs (paper Sec. 3)
# --------------------------------------------------------------------------------------


class ControllerOutput(BaseModel):
    """paper Sec. 3 — the JSON object the controller must produce (schema-enforced)."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["accept", "rewrite"]
    revised_text: str = Field(description="Revised text when action == 'rewrite'; may be empty.")


class GFixedOutput(BaseModel):
    """paper Sec. 4.2 — G_fixed emits one rewrite. Structured so that provider-level refusals are
    signalled the same way as for the controller."""

    model_config = ConfigDict(extra="forbid")

    revised_text: str


class EvaluatorOutput(BaseModel):
    """paper Sec. 3 — binary judgment from E. ``score`` is optional (2026-09-16);
    ``reason`` is the one-sentence reason the E prompt asks for. Both are recorded only;
    nothing downstream reads them.

    This model is sent to the provider as the structured-output schema, so its field
    descriptions are text E reads. They are deliberately absent: E's prompt is
    rubric-free, and the old description of ``reason`` ("which scale rule applied")
    pointed at a scale E no longer has. The only wording E gets is the authors' prompt file.
    """

    model_config = ConfigDict(extra="forbid")

    label: Literal["SAFE", "UNSAFE"]
    score: float | None = None
    reason: str | None = None


# --------------------------------------------------------------------------------------
# Core records
# --------------------------------------------------------------------------------------


class State(_Strict):
    """A rewrite state x_t (paper Sec. 4.2). ``t = 1`` for G_fixed outputs."""

    state_id: str
    x0_id: str
    x0: str
    text: str
    t: int = 1
    source: Literal["g_fixed", "controller"] = "g_fixed"
    split: Split | None = None
    gen_seed: int | None = None
    gen_attempt: int | None = None
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("t")
    @classmethod
    def _t_nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError("t must be >= 0")
        return v


class GenerationAttempt(_Strict):
    """One G_fixed attempt (paper Sec. 4.2: every attempt is logged, valid or not)."""

    x0_id: str
    attempt: int
    seed: int
    raw_output: str | None
    provider_refused: bool = False
    valid: bool
    invalid_reason: InvalidReason | None = None
    model: str
    model_version: str | None
    prompt_hash: str
    decoding: dict[str, Any]
    cached: bool = False
    created_at: str = Field(default_factory=utc_now_iso)


class Score(_Strict):
    """One verifier score for one state (scores table; score_spec.md).

    ``prompt_hash`` identifies the prompt a guard LM was scored with (step 2 probe, the authors
    2026-09-24: the probe table and the step-6 table share this schema, so the configuration
    finally chosen can be reused as is); None for an encoder, which has no prompt.
    ``sub_scores`` holds the per-policy scores when ``score`` aggregates several calls
    (ShieldGemma native: max over two policies); None otherwise.
    """

    state_id: str
    verifier_id: str
    model_version: str
    prompt_hash: str | None = None
    score: float
    sub_scores: dict[str, float] | None = None
    timestamp: str = Field(default_factory=utc_now_iso)

    @field_validator("score")
    @classmethod
    def _in_unit_interval(cls, v: float) -> float:
        if not (SCORE_MIN <= v <= SCORE_MAX):
            raise ValueError(f"score must be in [{SCORE_MIN}, {SCORE_MAX}], got {v}")
        return v

    @field_validator("sub_scores")
    @classmethod
    def _sub_scores_in_unit_interval(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        for name, s in (v or {}).items():
            if not (SCORE_MIN <= s <= SCORE_MAX):
                raise ValueError(f"sub score {name!r} must be in [{SCORE_MIN}, {SCORE_MAX}]")
        return v


class Observation(_Strict):
    """The s^obs interface (paper Sec. 3): the ONLY input the formatter accepts.

    ``s1_obs, s2_obs`` are post-channel in Exp 1 and raw in Exp 2. ``raw_s1, raw_s2`` are
    kept for logging only (the design requires both (s, s~) on disk); the formatter must not
    read them.
    """

    state_id: str
    s1_obs: float
    s2_obs: float
    tau1: float
    tau2: float
    lam: Lambda = Lambda.NONE
    raw_s1: float | None = None
    raw_s2: float | None = None
    v1_id: str | None = None
    v2_id: str | None = None

    @field_validator("s1_obs", "s2_obs", "tau1", "tau2")
    @classmethod
    def _unit(cls, v: float) -> float:
        if not (SCORE_MIN <= v <= SCORE_MAX):
            raise ValueError(f"value must be in [{SCORE_MIN}, {SCORE_MAX}], got {v}")
        return v


class UncertaintyFields(_Strict):
    """paper Sec. 3 — the six B fields, all deterministic in (s1_obs, s2_obs, tau1, tau2)."""

    y1: int
    y2: int
    m1: float  # signed decision margin s1_obs - tau1; never normalized across verifiers
    m2: float
    agreement: int  # 1[y1 == y2]
    conflict_direction: ConflictDirection

    @model_validator(mode="after")
    def _consistent(self) -> UncertaintyFields:
        if self.y1 not in (0, 1) or self.y2 not in (0, 1):
            raise ValueError("y_i must be 0 or 1")
        if self.agreement != int(self.y1 == self.y2):
            raise ValueError("agreement must equal 1[y1 == y2]")
        if self.agreement == 1 and self.conflict_direction is not ConflictDirection.NA:
            raise ValueError("conflict_direction must be N/A when verifiers agree")
        if self.agreement == 0 and self.conflict_direction is ConflictDirection.NA:
            raise ValueError("conflict_direction must be set when verifiers disagree")
        return self


class ControllerAttempt(_Strict):
    """One controller call (raw output of every attempt is logged)."""

    attempt: int  # 0 = first call, 1..PARSE_MAX_RETRIES = parse retries
    raw_output: str | None
    provider_refused: bool = False
    parsed: dict[str, Any] | None = None
    parse_fail_reason: ParseFailReason | None = None
    prompt_hash: str
    seed: int | None
    cached: bool = False
    created_at: str = Field(default_factory=utc_now_iso)


class Decision(_Strict):
    """Outcome of one accept/rewrite decision (paper Sec. 4.6)."""

    state_id: str
    arm: Arm
    lam: Lambda
    t: int
    action: Action | None  # None iff parse_fail
    revised_text: str | None
    parse_fail: bool
    n_parse_retries: int
    refusal_like: bool = False  # 2026-09-16: flag only, never affects parse_fail
    empty_rewrite: bool = False  # action == rewrite and revised_text == x_t (valid, paper Sec. 4.6)
    attempts: list[ControllerAttempt]
    model: str
    model_version: str | None
    decoding: dict[str, Any]

    @model_validator(mode="after")
    def _consistent(self) -> Decision:
        if self.parse_fail and self.action is not None:
            raise ValueError("parse_fail decisions carry no action")
        if not self.parse_fail and self.action is None:
            raise ValueError("non-parse_fail decisions must carry an action")
        if self.n_parse_retries > PARSE_MAX_RETRIES:
            raise ValueError(f"n_parse_retries cannot exceed {PARSE_MAX_RETRIES}")
        return self


class EvalRecord(_Strict):
    """One E judgment (paper Sec. 3, App. D.1); no arm or verifier output enters E.

    Detox judges text alone. Math judges a candidate against its private reference;
    its text_id/cache identity must bind question, reference and Math-Verify config.
    """

    text_id: str  # detox: text_hash(text); math: question/reference/candidate/config digest
    text: str
    label: SafetyLabel | MathLabel
    score: float | None = None
    reason: str | None = None  # E's rule citation; logged, never analysed
    model: str
    model_version: str | None  # the `model` string the provider returned for this call
    prompt_hash: str  # hash of this rendered request (varies with the text; cache key part)
    # paper Sec. 3: every E record names the E that produced it -- the sha256 of the
    # prompt FILE (LF-normalized, see rafr.evaluator.judge.prompt_sha256) and the configured
    # version. Records from a different E are refused wherever they are read back.
    e_prompt_sha256: str
    e_version: str | None
    decoding: dict[str, Any]
    cached: bool = False
    created_at: str = Field(default_factory=utc_now_iso)
    # paper App. D.1: references belong only to E's stream, never controller / score records.
    question_id: str | None = None
    reference_answer: str | None = None
    extracted_answer: str | None = None
    unresolved: bool | None = None
    math_verify_version: str | None = None
    # paper App. D.1 diagnostics (2026-10-03): only the last \boxed is judged; these never change
    # label.
    n_boxes: int | None = None
    boxes_all_equivalent: bool | None = None

    @model_validator(mode="after")
    def _math_outcome(self) -> EvalRecord:
        if isinstance(self.label, MathLabel) and self.unresolved is not None:
            if self.unresolved != (self.label is MathLabel.UNRESOLVED):
                raise ValueError("math unresolved flag must agree with E's label")
        if self.boxes_all_equivalent is not None and (self.n_boxes or 0) < 2:
            raise ValueError("box equivalence is only defined for two or more boxes")
        return self


class LogRecord(_Strict):
    """one row per (round, arm, state). Every listed field is present, even if null.

    ``c_t`` and ``r_ref`` are filled in AFTER the fact by analysis code only; they are
    never read by anything on the decision path (paper Sec. 3).
    """

    run_id: str
    experiment: Literal["exp1", "exp2", "exp2_smoke", "dry_run"]
    seed: int | None
    arm: Arm
    lam: Lambda
    t: int
    state_id: str
    x0_id: str
    x_t: str
    # verifier scores before and after the channel (both always logged)
    v1_id: str | None
    v2_id: str | None
    s1: float | None
    s2: float | None
    s1_tilde: float | None
    s2_tilde: float | None
    tau1: float | None
    tau2: float | None
    # B's six fields (computed for every arm for logging; shown only in B / permuted_B)
    fields: UncertaintyFields | None
    permuted_from_state_id: str | None = None  # permuted_B only
    # controller
    attempts: list[ControllerAttempt]
    action: Action | None
    revised_text: str | None
    n_parse_retries: int
    parse_fail: bool
    refusal_like: bool
    # evaluator
    e_label: SafetyLabel | MathLabel | None
    e_score: float | None
    # post-hoc only (paper Sec. 3); always null at write time on the decision path
    c_t: tuple[int, int] | None = None
    r_ref: float | None = None
    # reproducibility
    prompt_hash: str | None
    model: str | None
    model_version: str | None
    decoding: dict[str, Any] | None
    created_at: str = Field(default_factory=utc_now_iso)
