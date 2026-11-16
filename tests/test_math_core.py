"""Appendix M compatibility must preserve Core access gates and detox outcomes."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from rafr.contracts.schemas import (
    Action,
    Arm,
    Decision,
    EvalRecord,
    Lambda,
    LogRecord,
    MathLabel,
    Observation,
    ParseFailReason,
    SafetyLabel,
    Split,
)
from rafr.controller.parse import parse_attempt
from rafr.data import labels
from rafr.data.splits import save_splits
from rafr.experiments.common import make_log_record
from rafr.formatter.fields import compute_fields
from rafr.llm.client import LLMResponse
from rafr.llm.ledger import Usage
from rafr.logging.writer import JsonlWriter


def evaluation(label=MathLabel.INCORRECT, **changes):
    kwargs = dict(
        text_id="question-reference-candidate-backend-digest",
        text=r"\boxed{2}",
        label=label,
        model="math-verify",
        model_version="0.9.0",
        prompt_hash="question-reference-candidate-backend-digest",
        e_prompt_sha256="frozen-extraction-configuration-digest",
        e_version="0.9.0",
        decoding={},
        question_id="q",
        reference_answer="1",
        extracted_answer="2",
        unresolved=label is MathLabel.UNRESOLVED,
        math_verify_version="0.9.0",
    )
    kwargs.update(changes)
    return EvalRecord(**kwargs)


def write_labels(tmp_path, label=MathLabel.INCORRECT, *, record_label=None):
    split_dir, labels_dir = tmp_path / "splits", tmp_path / "labels"
    splits = {sp: [f"{sp.value}-q"] for sp in Split}
    save_splits(splits, split_dir)
    with JsonlWriter(labels.y_star_path(labels_dir)) as writer:
        for sp, ids in splits.items():
            writer.write(
                labels.StateJudgment(
                    state_id=f"{sp.value}-s",
                    x0_id=ids[0],
                    label=label,
                    record=evaluation(label if record_label is None else record_label),
                    e_identity={"math_verify_version": "0.9.0"},
                )
            )
    return splits, dict(labels_dir=labels_dir, split_dir=split_dir)


def test_math_labels_round_trip_and_core_purpose_gate(tmp_path):
    splits, paths = write_labels(tmp_path)
    rows = labels.read_labels(Split.D_FIT, labels.LabelPurpose.TAU_FIT, **paths)
    assert rows[0].label is MathLabel.INCORRECT
    assert rows[0].record.reference_answer == "1"
    with pytest.raises(labels.LabelAccessError):
        labels.read_labels(Split.D_GATE, labels.LabelPurpose.TAU_FIT, **paths)
    with pytest.raises(labels.LabelAccessError):
        labels.read_labels(Split.D_GATE_RESERVE, labels.LabelPurpose.GATE, **paths)
    report = labels.split_report(splits, domain="math", **paths)
    assert report["d_fit"] == {"n": 1, "n_incorrect": 1, "incorrect_rate": 1}
    assert report["d_gate"] == {"n": 1}
    assert report["d_gate_reserve"] == {"n": 1}


def test_initial_unresolved_cannot_enter_any_split_consumption(tmp_path):
    _, paths = write_labels(tmp_path, MathLabel.UNRESOLVED)
    with pytest.raises(labels.LabelDataError):
        labels.read_labels(Split.D_EXP1, labels.LabelPurpose.EXP1_OUTCOME, **paths)


def test_y_star_cannot_disagree_with_e_record(tmp_path):
    _, paths = write_labels(tmp_path, record_label=MathLabel.CORRECT)
    with pytest.raises(labels.LabelDataError, match="disagrees"):
        labels.read_labels(Split.D_FIT, labels.LabelPurpose.TAU_FIT, **paths)


def test_math_e_metadata_does_not_enter_core_decision_log():
    obs = Observation(state_id="s", s1_obs=0.8, s2_obs=0.2, tau1=0.5, tau2=0.5)
    decision = Decision(
        state_id="s",
        arm=Arm.A1,
        lam=Lambda.NONE,
        t=1,
        action=Action.ACCEPT,
        revised_text=None,
        parse_fail=False,
        n_parse_retries=0,
        attempts=[],
        model="fixed-controller",
        model_version="v",
        decoding={},
    )
    row = make_log_record(
        run_id="r",
        experiment="exp2",
        seed=1,
        arm=Arm.A1,
        lam=Lambda.NONE,
        t=1,
        state_id="s",
        x0_id="q",
        x_t="candidate",
        obs=obs,
        fields=compute_fields(obs),
        decision=decision,
        e_label=MathLabel.INCORRECT,
    )
    assert row.e_label is MathLabel.INCORRECT
    encoded = row.model_dump(mode="json")
    assert "reference_answer" not in encoded and "extracted_answer" not in encoded
    with pytest.raises(ValidationError):
        LogRecord.model_validate({**encoded, "reference_answer": "SECRET_GOLD"})
    restored = EvalRecord.model_validate_json(evaluation().model_dump_json())
    assert restored.label is MathLabel.INCORRECT
    assert restored.reference_answer == "1"
    with pytest.raises(ValidationError, match="unresolved"):
        evaluation(MathLabel.UNRESOLVED, unresolved=False)


def test_detox_label_remains_its_original_enum(tmp_path):
    _, paths = write_labels(tmp_path, SafetyLabel.UNSAFE)
    row = labels.read_labels(Split.D_FIT, labels.LabelPurpose.TAU_FIT, **paths)[0]
    assert row.label is SafetyLabel.UNSAFE
    with pytest.raises(labels.LabelDataError, match="math"):
        labels.split_report({Split.D_FIT: ["d_fit-q"]}, domain="math", **paths)


def _controller_response(text, *, error=None, refused=False):
    return LLMResponse(
        text=text,
        parsed=None,
        refused=refused,
        stop_reason="stop",
        model_version="v",
        usage=Usage(),
        prompt_hash="h",
        cache_key="k",
        error=error,
    )


@pytest.mark.parametrize(
    "text,error,refused,expected",
    [
        ("", None, False, ParseFailReason.MALFORMED_JSON),
        (None, "provider refusal", True, ParseFailReason.REFUSAL),
    ],
)
def test_empty_and_refusal_use_existing_four_parse_categories(text, error, refused, expected):
    parsed = parse_attempt(
        _controller_response(text, error=error, refused=refused), "candidate", []
    )
    assert parsed.fail_reason is expected and not parsed.ok
    assert len(ParseFailReason) == 4
    assert json.loads(json.dumps(expected)) == expected.value


def test_parse_layer_does_not_read_the_provider_error_field():
    """M.8 (2026-10-03): network errors raise in the client; parse judges content only."""
    response = _controller_response('{"action":"accept","revised_text":""}', error="provider")
    parsed = parse_attempt(response, "candidate", [])
    assert parsed.ok and parsed.action is Action.ACCEPT and parsed.fail_reason is None
