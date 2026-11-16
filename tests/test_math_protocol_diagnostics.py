"""Hand controls for diagnostics omitted by the original math report."""

import copy
import json

import pytest
from analysis import drift, error_boundary, transport
from pipelines.math_diagnostics import (
    channel_descriptive_diagnostics,
    channel_diagnostics,
    protocol_outcome_diagnostics,
)

from rafr.channel.degrade import LADDER, DegradeChannel, LevelParams
from rafr.channel.margin_fit import fit_margin_model
from rafr.contracts.schemas import text_hash
from rafr.validity import gate


def test_expected_whole_split_floor_accounts_for_unequal_class_prevalence(monkeypatch):
    from pipelines import step08_tune_channel as core_step8

    original = core_step8.reachability
    calls = []

    def traced(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(core_step8, "reachability", traced)
    channel = DegradeChannel(
        {level: LevelParams(p, p) for level, p in zip(LADDER, (0, 0.5, 1), strict=True)},
        {v: fit_margin_model([0.2, 0.6], 0.5, v) for v in ("v1", "v2")},
        0.5,
        0.5,
        19,
    )
    result = channel_diagnostics(
        channel,
        [[0.9, 0.9], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]],
        [1, 0, 0, 0],
        ["a", "b", "c", "d"],
        0.1,
    )
    assert len(calls) == 1
    whole = result["per_verifier"]["v1"]["whole_split_reachability"]
    assert whole["n_states"] == 4 and whole["incorrect_share"] == 0.25
    assert whole["accuracy"] == pytest.approx(
        {"baseline": 1, "all_resampled_floor": 0.625, "span": 0.375}
    )
    assert whole["balanced_accuracy"] == {"baseline": 1, "all_resampled_floor": 0.5, "span": 0.5}


def test_channel_tables_keep_whole_split_accuracy_and_paired_error_subset():
    labels = [1, 1, 0, 0]
    observed = dict(
        zip(
            LADDER,
            [
                ([0.9, 0.9, 0.1, 0.1], [0.9, 0.1, 0.9, 0.1]),
                ([0.9, 0.1, 0.1, 0.1], [0.1, 0.9, 0.1, 0.1]),
                ([0.1, 0.1, 0.9, 0.1], [0.1, 0.1, 0.1, 0.9]),
            ],
            strict=True,
        )
    )
    result = channel_descriptive_diagnostics(labels, observed, [0.5, 0.5])
    assert result["descriptive"] == gate.descriptive_accuracy(labels, observed, 0.5, 0.5)
    assert result["descriptive"]["v1"]["mid"] == {
        "n": 4,
        "accuracy": 0.75,
        "balanced_accuracy": 0.75,
    }
    assert result["p_at_least_one_wrong"] == dict(zip(LADDER, [0.5, 1, 1], strict=True))
    paired = result["paired_correctness_correlation"]["v2"]["low-mid"]
    assert paired == {"phi": -1, "n": 2, "n11": 0, "n10": 1, "n01": 1, "n00": 0}
    assert result["paired_correctness_correlation"]["v1"]["low-mid"]["phi"] is None
    # The added tables have no effect on the Core gate's verdict.
    vectors = gate.detection_vectors(labels, observed, 0.5, 0.5)
    base = gate.ladder_check(gate.reliability_by_level(vectors), 0.1)
    augmented = gate.ladder_check(
        gate.reliability_by_level(vectors),
        0.1,
        result["p_at_least_one_wrong"],
        descriptive=result["descriptive"],
    )
    assert (base.passed, base.failed_conditions) == (augmented.passed, augmented.failed_conditions)


def inputs():
    observed = {"q1": [0.9, 0.9, 0.1], "q2": [0.9, 0.1, 0.1]}
    exp1 = [
        {
            "question_id": qid,
            "initial_status": "incorrect",
            "arm": arm,
            "level": level,
            "action": "accept" if arm == "A1" else "rewrite",
            "parse_fail": False,
            "input_scores": [0.9, 0.9],
            "observed_scores": [observed[qid][i]] * 2,
            "thresholds": [0.5, 0.5],
        }
        for qid in ("q1", "q2")
        for arm in ("A1", "B", "permuted_B")
        for i, level in enumerate(("low", "mid", "high"))
    ]

    def row(t, status, score, action="accept", parse_fail=False):
        return {
            "t": t,
            "input_solution": r"The same answer text: \boxed{2}.",
            "input_grade": {"status": status},
            "input_scores": [score, score],
            "thresholds": [0.5, 0.5],
            "action": action,
            "parse_fail": parse_fail,
        }

    trajectories = [
        {
            "question_id": "q1",
            "arm": "A1",
            "rounds": [row(1, "incorrect", 0.9, "rewrite"), row(2, "correct", 0.9)],
        },
        {
            "question_id": "q2",
            "arm": "B",
            "rounds": [
                row(1, "incorrect", 0.9, "rewrite"),
                row(2, "unresolved", 0.1),
                row(3, "backend_error", 0.9, None, True),
            ],
        },
    ]
    gate_r = {v: {"low": 0.9, "mid": 0.6, "high": 0.2} for v in ("v1", "v2")}
    return exp1, trajectories, gate_r


def test_outcome_tables_report_drift_without_dropping_conditions_and_later_rounds():
    exp1, trajectories, gate_r = inputs()
    before = copy.deepcopy((exp1, trajectories, gate_r))
    result = protocol_outcome_diagnostics(exp1, trajectories, gate_r, n_boot=100, seed=3)
    assert (exp1, trajectories, gate_r) == before
    ladder = result["exp1_drift"]["ladder"]
    assert ladder["n_states"] == 2
    assert [ladder["r"]["v1"][level]["estimate"] for level in ("low", "mid", "high")] == [
        1,
        0.5,
        0,
    ]
    drift_row = next(
        row
        for row in result["exp1_drift"]["vs_gate"]
        if row["verifier"] == "v1" and row["lambda"] == "high"
    )
    assert drift_row["R_exp1"] == 0 and drift_row["R_gate"] == 0.2
    assert drift_row["exp1_minus_gate"] == -0.2
    assert len(result["exp1_drift"]["vs_gate"]) == 6
    tr = result["transport"]
    assert tr["n_exp2_rows_t1"] == 2 and tr["n_exp2_rows_t_ge_2"] == 3
    assert tr["s1"]["exp1_states"]["n"] == 2  # deduplicates nine cells per question
    assert tr["s1"]["exp2_t_ge_2"]["mean"] == pytest.approx(1.9 / 3)
    assert tr["task_failure"]["exp2_t_ge_2"]["n"] == 2
    assert tr["task_failure"]["exp2_t_ge_2"]["mean"] == 0.5
    assert tr["grade_coverage"]["exp2_t_ge_2"] == {
        "n_decisions": 3,
        "input_grade_status_counts": {"backend_error": 1, "correct": 1, "unresolved": 1},
        "n_input_unresolved": 1,
        "n_missing_E": 1,
        "n_parse_fail": 1,
    }
    assert tr["by_round_exp2"][3]["task_failure"] == {"n": 0}
    assert "unsafe" not in tr
    json.dumps(result, allow_nan=False)


def test_error_boundary_uses_own_question_labels_and_keeps_e_errors_missing():
    result = protocol_outcome_diagnostics(*inputs(), n_boot=10)
    boundary = result["error_boundary_alignment"]
    assert boundary["exp1"]["cells"]["high"]["A1"]["decision_error_rate_on_misaligned"] == 1
    assert boundary["exp1"]["cells"]["high"]["B"]["decision_error_rate_on_misaligned"] == 0
    # Identical solution strings belong to different problems/labels at t=2.
    assert boundary["exp2"]["cells"]["2"]["A1"]["decision_error_rate_on_misaligned"] == 0
    assert boundary["exp2"]["cells"]["2"]["B"]["decision_error_rate_on_misaligned"] == 1
    assert boundary["exp2"]["n_rows_without_judgment"] == 1


def test_outcome_adapter_delegates_each_metric_to_core(monkeypatch):
    calls = {}
    for module, name in (
        (drift, "exp1_ladder"),
        (drift, "compare_to_gate"),
        (transport, "compare"),
        (error_boundary, "exp1"),
        (error_boundary, "exp2"),
    ):
        original = getattr(module, name)
        key = f"{module.__name__}.{name}"

        def traced(*args, _original=original, _key=key, **kwargs):
            calls[_key] = (args, kwargs)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, traced)
    protocol_outcome_diagnostics(*inputs(), n_boot=10, seed=23)
    assert set(calls) == {
        "analysis.drift.exp1_ladder",
        "analysis.drift.compare_to_gate",
        "analysis.transport.compare",
        "analysis.error_boundary.exp1",
        "analysis.error_boundary.exp2",
    }
    assert calls["analysis.drift.exp1_ladder"][1] == {"n_boot": 10, "seed": 23}
    rows, truth = calls["analysis.error_boundary.exp2"][0]
    second = [row for row in rows if row["t"] == 2]
    assert [truth[text_hash(row["x_t"])] for row in second] == ["SAFE", "UNSAFE"]
