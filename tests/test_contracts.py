"""Contracts: locked constants, schema invariants (paper Sec. 3, 4.6, score spec)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rafr.contracts import schemas as S


def test_locked_constants_match_v06():
    # " 新锁定" + coding instructions section 5
    assert S.K_MAX_ROUNDS == 4
    assert S.PARSE_MAX_RETRIES == 2
    assert S.PARSE_FAIL_ALERT_THRESHOLD == 0.02
    assert S.GATE_MAX_ATTEMPTS == 2
    assert S.TAU_CRITERION == "f1_max_plateau_midpoint"  # (paper Sec. 5): F1, not F2
    assert (S.SCORE_MIN, S.SCORE_MAX) == (0.0, 1.0)


def test_score_must_be_in_unit_interval():
    S.Score(state_id="s", verifier_id="v", model_version="m", score=0.0)
    S.Score(state_id="s", verifier_id="v", model_version="m", score=1.0)
    with pytest.raises(ValidationError):
        S.Score(state_id="s", verifier_id="v", model_version="m", score=1.5)
    with pytest.raises(ValidationError):
        S.Score(state_id="s", verifier_id="v", model_version="m", score=-0.1)


def test_conflict_direction_na_when_agree_and_never_blank():
    S.UncertaintyFields(
        y1=1, y2=1, m1=0.2, m2=0.1, agreement=1, conflict_direction=S.ConflictDirection.NA
    )
    with pytest.raises(ValidationError):
        S.UncertaintyFields(
            y1=1,
            y2=1,
            m1=0.2,
            m2=0.1,
            agreement=1,
            conflict_direction=S.ConflictDirection.V1_UNSAFE_V2_SAFE,
        )
    with pytest.raises(ValidationError):
        S.UncertaintyFields(
            y1=1, y2=0, m1=0.2, m2=-0.1, agreement=0, conflict_direction=S.ConflictDirection.NA
        )
    with pytest.raises(ValidationError):  # agreement must equal 1[y1 == y2]
        S.UncertaintyFields(
            y1=1, y2=0, m1=0.2, m2=-0.1, agreement=1, conflict_direction=S.ConflictDirection.NA
        )
    with pytest.raises(ValidationError):  # blank is not allowed
        S.UncertaintyFields(y1=1, y2=1, m1=0.2, m2=0.1, agreement=1, conflict_direction="")


def _attempt(**kw):
    base = dict(attempt=0, raw_output="{}", prompt_hash="h", seed=1)
    base.update(kw)
    return S.ControllerAttempt(**base)


def test_decision_parse_fail_consistency():
    ok = S.Decision(
        state_id="s",
        arm=S.Arm.A1,
        lam=S.Lambda.LOW,
        t=1,
        action=S.Action.ACCEPT,
        revised_text=None,
        parse_fail=False,
        n_parse_retries=0,
        attempts=[_attempt()],
        model="m",
        model_version=None,
        decoding={},
    )
    assert ok.action is S.Action.ACCEPT
    with pytest.raises(ValidationError):  # parse_fail carries no action
        S.Decision(
            state_id="s",
            arm=S.Arm.A1,
            lam=S.Lambda.LOW,
            t=1,
            action=S.Action.ACCEPT,
            revised_text=None,
            parse_fail=True,
            n_parse_retries=2,
            attempts=[_attempt()],
            model="m",
            model_version=None,
            decoding={},
        )
    with pytest.raises(ValidationError):  # more retries than allowed
        S.Decision(
            state_id="s",
            arm=S.Arm.A1,
            lam=S.Lambda.LOW,
            t=1,
            action=None,
            revised_text=None,
            parse_fail=True,
            n_parse_retries=3,
            attempts=[_attempt()],
            model="m",
            model_version=None,
            decoding={},
        )


def test_controller_output_schema_is_closed():
    S.ControllerOutput(action="accept", revised_text="")
    with pytest.raises(ValidationError):
        S.ControllerOutput(action="skip", revised_text="")
    with pytest.raises(ValidationError):
        S.ControllerOutput(action="accept", revised_text="", extra=1)


def test_evaluator_output_binary_with_optional_score():
    assert S.EvaluatorOutput(label="SAFE").score is None
    assert S.EvaluatorOutput(label="UNSAFE", score=0.7).score == 0.7
    with pytest.raises(ValidationError):
        S.EvaluatorOutput(label="MAYBE")


def test_log_record_has_every_ii10_field():
    required = {
        "run_id", "seed", "arm", "lam", "t", "state_id", "x_t",
        "s1", "s2", "s1_tilde", "s2_tilde", "tau1", "tau2", "fields",
        "attempts", "action", "n_parse_retries", "parse_fail",
        "e_label", "e_score", "c_t", "r_ref", "prompt_hash", "model", "model_version", "decoding",
    }  # fmt: skip
    assert required <= set(S.LogRecord.model_fields)
