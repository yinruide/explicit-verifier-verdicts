"""Hand-calculated controls for diagnostics, including ties and missing evidence."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pipelines.math_diagnostics import (
    MathDiagnosticError,
    channel_diagnostics,
    cue_diagnostics,
    exp1_diagnostics,
    exp2_reliability_diagnostics,
    verifier_diagnostics,
)

from rafr.channel.degrade import LADDER, DegradeChannel, LevelParams
from rafr.channel.margin_fit import fit_margin_model
from rafr.domains.math.storage import digest

SCORES = [[0.9, 0.2], [0.6, 0.8], [0.2, 0.7], [0.1, 0.1]]
LABELS = [1, 1, 0, 0]


def test_hand_computed_classifier_calibration_and_complementarity():
    result = verifier_diagnostics(SCORES, LABELS, [0.5, 0.5])
    a, b = result["per_verifier"]["v1"], result["per_verifier"]["v2"]
    assert a["confusion"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}
    assert (a["tpr"], a["fpr"], a["f1"], a["f2"], a["auroc"], a["pr_auc"]) == (1, 0, 1, 1, 1, 1)
    assert a["brier"] == pytest.approx(0.055)
    assert a["ece"] == pytest.approx(0.2)
    assert a["unique_scores"] == 4 and a["score_coverage"] == 1
    assert b["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert (b["tpr"], b["fpr"], b["f1"], b["f2"]) == (0.5, 0.5, 0.5, 0.5)
    assert b["pr_auc"] == pytest.approx(5 / 6)  # precisions 1 and 2/3 at two recall jumps
    assert b["auroc"] == 0.75  # three of four positive-negative score comparisons win
    assert b["brier"] == pytest.approx(0.295)
    joint = result["complementarity"]
    assert joint["disagreement_rate"] == joint["v1_only_correct_rate"] == 0.5
    assert joint["joint_miss_rate_on_errors"] == joint["both_wrong_rate"] == 0
    assert joint["error_indicator_phi"] is None  # perfect A has constant-zero errors
    assert result["score_is_calibrated_probability"] is False


def test_ap_reuses_core_metrics_without_a_parallel_math_definition():
    from rafr.validity import metrics

    one = verifier_diagnostics([[0.5, 0.5], [0.5, 0.5]], [1, 0], [0.5, 0.5])
    two = verifier_diagnostics([[0.5, 0.5], [0.5, 0.5]], [0, 1], [0.5, 0.5])
    for result, labels in ((one, [1, 0]), (two, [0, 1])):
        assert result["pr_auc_definition"].startswith("rafr.validity.metrics.pr_auc")
        for verifier in result["per_verifier"].values():
            assert verifier["pr_auc"] == metrics.pr_auc([0.5, 0.5], labels)
            assert verifier["auroc"] == 0.5
            assert verifier["tied_scores"] is True
            assert verifier["variance"] == 0 and verifier["unique_scores"] == 1
            assert verifier["f2"] == pytest.approx(5 / 6)


def test_cue_tables_use_core_on_incorrect_and_all_populations(monkeypatch):
    from rafr.validity import cue_diagnostics as core

    original = core.compute
    calls = []

    def traced(inputs, **kwargs):
        calls.append((inputs, kwargs))
        return original(inputs, **kwargs)

    monkeypatch.setattr(core, "compute", traced)
    result = cue_diagnostics(SCORES, LABELS, [0.5, 0.5], high_margin_quantile=0.75)
    assert len(calls) == 2
    assert calls[0][0].y_true == [1, 1]
    assert calls[1][0].y_true == LABELS
    for call in calls:
        assert call[1] == {"high_margin_quantile": 0.75}
    primary = result["primary_incorrect"]["cues"]
    assert primary["n"] == 2
    assert primary["disagreement_rate"] == 0.5
    assert set(primary) >= {
        "auroc_margin_to_correct",
        "error_rate_given_agreement",
        "error_rate_given_disagreement",
        "high_margin_wrong_share",
        "joint_margin_hist",
    }
    assert result["secondary_all"]["cues"]["n"] == 4


def test_missing_scores_keep_coverage_and_undefined_denominators_are_not_zero():
    result = verifier_diagnostics([[None, 0.1], [0.2, None]], [0, 0], [0.5, 0.5])
    assert result["n_input"] == 2 and result["n_pair_complete"] == 0
    assert result["complementarity"]["status"] == "NOT_COMPUTED"
    for verifier in result["per_verifier"].values():
        assert verifier["n_scored"] == verifier["n_missing"] == 1
        assert verifier["score_coverage"] == 0.5
        assert verifier["tpr"] is None
        assert verifier["auroc"] is None and verifier["pr_auc"] is None
        assert verifier["fpr"] == 0
    empty = verifier_diagnostics([], [], [0.5, 0.5])
    assert empty["status"] == empty["per_verifier"]["v1"]["status"] == "NOT_COMPUTED"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True, "0.2"])
def test_invalid_scores_are_not_silently_imputed_or_discarded(bad):
    with pytest.raises(MathDiagnosticError):
        verifier_diagnostics([[bad, 0.3]], [1], [0.5, 0.5])


def channel():
    q = {
        "v1": fit_margin_model([0.2, 0.8], 0.5, "v1"),
        "v2": fit_margin_model([0.1], 0.5, "v2"),
    }
    return DegradeChannel(
        {level: LevelParams(p, p) for level, p in zip(LADDER, (0, 0.5, 1), strict=True)},
        q,
        0.5,
        0.5,
        3,
    )


def test_expected_channel_ladder_uses_resampling_floor_without_tuning_or_gate_claim():
    c = channel()
    before = deepcopy(c.to_dict())
    result = channel_diagnostics(c, SCORES, LABELS, ["a", "b", "c", "d"], 0.2)
    a, b = result["per_verifier"]["v1"], result["per_verifier"]["v2"]
    assert a["base_tpr"] == 1 and a["resampling_floor"] == pytest.approx(0.5)
    assert a["expected_ladder"] == pytest.approx({"low": 1, "mid": 0.75, "high": 0.5})
    assert a["expected_adjacent_gaps"] == pytest.approx({"low-mid": 0.25, "mid-high": 0.25})
    assert b["base_tpr"] == 0.5 and b["resampling_floor"] == 0
    assert b["expected_ladder"] == {"low": 0.5, "mid": 0.25, "high": 0}
    assert a["actual_D_fit_ladder"]["low"] == 1
    assert b["actual_D_fit_ladder"]["high"] == 0  # q has no score at/above threshold
    assert a["expected_target_shortfall"] == b["expected_target_shortfall"] == 0
    assert result["is_gate_verdict"] is False and result["parameter_tuning_performed"] is False
    assert c.to_dict() == before


def test_channel_missing_truth_and_impossible_reachability_remain_explicit():
    c = channel()
    result = channel_diagnostics(c, SCORES, [0, 0, 0, 0], ["a", "b", "c", "d"], 0.2)
    assert result["status"] == "NOT_COMPUTED"
    result = channel_diagnostics(c, SCORES, LABELS, ["a", "b", "c", "d"], 0.4)
    assert result["per_verifier"]["v1"]["expected_target_shortfall"] == pytest.approx(0.15)
    assert result["is_gate_verdict"] is False
    with pytest.raises(MathDiagnosticError, match="unique"):
        channel_diagnostics(c, SCORES, LABELS, ["a", "a", "c", "d"], 0.2)


def exp1_rows():
    rows = []
    evidence = {
        "q1": {"low": [0.8, 0.2], "mid": [0.2, 0.2], "high": [0.2, 0.2]},
        "q2": {"low": [0.8, 0.8], "mid": [0.2, 0.8], "high": [0.8, 0.2]},
    }
    for qid in ("q1", "q2"):
        donor = "q2" if qid == "q1" else "q1"
        for level in ("low", "mid", "high"):
            for arm in ("A1", "B", "permuted_B"):
                payload = [qid, arm, evidence[qid][level]]
                if arm == "permuted_B":
                    payload.append(evidence[donor][level])
                rows.append(
                    {
                        "question_id": qid,
                        "arm": arm,
                        "level": level,
                        "initial_status": "incorrect",
                        "parse_fail": False,
                        "action": "accept" if (qid == "q1") == (arm != "B") else "rewrite",
                        "observed_scores": evidence[qid][level],
                        "thresholds": [0.5, 0.5],
                        "channel_assignment": {
                            "perturbed_v1": level == "high" or (level == "mid" and qid == "q1"),
                            "perturbed_v2": level == "high" or (level == "mid" and qid == "q2"),
                        },
                        "attempts": [{"prompt_hash": digest(payload)}],
                        "donor_fields_agreement": "deliberately irrelevant diagnostic decoy",
                    }
                )
    return rows


def test_exp1_actual_hash_equality_and_h2_own_state_strata_hand_controls():
    result = exp1_diagnostics(exp1_rows())
    assert result["status"] == "DESCRIPTIVE_ONLY"
    prompts = result["actual_first_prompt_hash_equality"]
    scores = result["observed_score_float64_bitwise_equality"]
    assert prompts["A1"]["mid-high"]["equal_fraction"] == 0.5
    assert prompts["permuted_B"]["mid-high"]["equal_fraction"] == 0
    assert scores["permuted_B"]["mid-high"]["equal_fraction"] == 0.5
    h2 = result["H2"]
    low = h2["by_channel_mode"]["low"]["none"]
    assert low["n_questions"] == 2 and low["delta_FA_A1_minus_B"] == 0
    assert low["per_arm"]["A1"]["FA"] == low["per_arm"]["B"]["FA"] == 0.5
    agreement = h2["by_observed_agreement"]["low"]
    assert agreement["disagree"]["delta_FA_A1_minus_B"] == 1
    assert agreement["agree"]["delta_FA_A1_minus_B"] == -1
    assert h2["n_channel_question_levels"] == h2["n_agreement_question_levels"] == 6


def test_exp1_missing_actual_request_fields_are_not_inferred_from_scores():
    rows = exp1_rows()
    for row in rows:
        del row["attempts"]
    report = exp1_diagnostics(rows)
    assert report["status"] == "PARTIAL"
    assert report["actual_first_prompt_hash_equality"]["A1"]["low-mid"]["status"] == "NOT_COMPUTED"
    assert report["H2"]["status"] == "DESCRIPTIVE_ONLY"
    assert (
        exp1_diagnostics([{"question_id": "q", "arm": "A1", "level": "low"}])["status"]
        == "NOT_COMPUTED"
    )


def test_h2_rejects_arm_specific_assignment_and_keeps_parse_fail_in_denominator():
    rows = exp1_rows()
    target = next(
        row
        for row in rows
        if row["question_id"] == "q1" and row["arm"] == "A1" and row["level"] == "low"
    )
    target.update(parse_fail=True, action=None)
    low = exp1_diagnostics(rows)["H2"]["by_channel_mode"]["low"]["none"]["per_arm"]["A1"]
    assert low == {"n": 2, "n_accept": 0, "n_parse_fail": 1, "FA": 0}
    target["channel_assignment"]["perturbed_v1"] = True
    with pytest.raises(MathDiagnosticError, match="arm-independent"):
        exp1_diagnostics(rows)


def test_exp2_current_reliability_preserves_unknown_labels_and_zero_positive_undefined_tpr():
    trajectories = [
        {
            "question_id": "a",
            "arm": "A1",
            "rounds": [
                {
                    "t": 1,
                    "input_grade": {"status": "incorrect"},
                    "input_scores": [0.9, 0.1],
                    "thresholds": [0.5, 0.5],
                },
                {
                    "t": 2,
                    "input_grade": {"status": "correct"},
                    "input_scores": [0.1, 0.1],
                    "thresholds": [0.5, 0.5],
                },
            ],
        },
        {
            "question_id": "b",
            "arm": "A1",
            "rounds": [
                {
                    "t": 1,
                    "input_grade": {"status": "unresolved"},
                    "input_scores": [0.9, 0.9],
                    "thresholds": [0.5, 0.5],
                },
            ],
        },
    ]
    result = exp2_reliability_diagnostics(trajectories)
    first = result["by_arm_round"]["A1"]["1"]
    assert first["n_active"] == 2 and first["n_unknown_input_grade"] == 1
    assert first["input_grade_coverage"] == 0.5 and first["status"] == "PARTIAL"
    assert first["per_verifier"]["v1"]["tpr"] == 1
    assert first["per_verifier"]["v2"]["tpr"] == 0
    second = result["by_arm_round"]["A1"]["2"]
    assert second["per_verifier"]["v1"]["tpr"] is None
    assert second["per_verifier"]["v1"]["fpr"] == 0
    assert result["by_arm_round"]["A1"]["3"]["status"] == "NOT_COMPUTED"
    assert result["by_arm_round"]["B"]["1"]["n_active"] == 0
    assert "downstream" in result["interpretation"]


def test_full_single_and_pair_diagnostic_uses_core_fifteen_bins(monkeypatch):
    import numpy as np

    from rafr.verifiers import diagnostics as core

    calls = []
    original = core.single

    def traced(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "single", traced)
    scores = [[0.11, 0.21], [0.68, 0.28], [0.76, 0.13], [0.89, 0.94]]
    report = verifier_diagnostics(scores, LABELS, [0.5, 0.5])
    assert len(calls) == 2
    assert report["ece_bins"] == 15
    expected = original(np.asarray(scores)[:, 0], np.asarray(LABELS), 0.5, 0.5)
    actual = report["per_verifier"]["v1"]
    assert actual["ece"] == expected["ece"]
    assert actual["accuracy"] == expected["accuracy"]
    assert actual["fnr"] == expected["fnr_unsafe"]
    pair = core.pair(
        np.asarray(scores)[:, 0], 0.5, np.asarray(scores)[:, 1], 0.5, np.asarray(LABELS)
    )
    assert report["complementarity"]["error_indicator_kappa"] == pair["kappa_errors"]


def test_h2_preregistered_stratum_is_any_perturbation_not_four_modes():
    report = exp1_diagnostics(exp1_rows())["H2"]
    binary = report["preregistered"]["any_verifier_perturbed"]["strata"]
    # Both mid-level states are perturbed, one by V1 and the other by V2.
    assert binary["True"]["cells"]["A1|mid"]["n"] == 2
    assert binary["True"]["delta_fa_by_lambda"]["mid"] == 0
    assert binary["False"]["cells"]["A1|low"]["n"] == 2


def test_actual_reliability_is_accuracy_not_only_tpr_or_f1():
    rows = [
        {
            "question_id": "wrong",
            "arm": "A1",
            "rounds": [
                {
                    "t": 1,
                    "input_grade": {"status": "incorrect"},
                    "input_scores": [0.9, 0.1],
                    "thresholds": [0.5, 0.5],
                }
            ],
        },
        {
            "question_id": "right",
            "arm": "A1",
            "rounds": [
                {
                    "t": 1,
                    "input_grade": {"status": "correct"},
                    "input_scores": [0.9, 0.1],
                    "thresholds": [0.5, 0.5],
                }
            ],
        },
    ]
    report = exp2_reliability_diagnostics(rows)
    assert report["R_t"]["A1"]["1"] == {"v1": 0.5, "v2": 0.5, "n": 2}


def test_task_reliability_retains_unresolved_but_not_e_backend_failure():
    rows = [
        {
            "question_id": status,
            "arm": "A1",
            "rounds": [
                {
                    "t": 2,
                    "input_grade": {"status": status},
                    "input_scores": [0.9, 0.1],
                    "thresholds": [0.5, 0.5],
                }
            ],
        }
        for status in ("unresolved", "backend_error")
    ]
    report = exp2_reliability_diagnostics(rows)
    assert report["R_t"]["A1"]["2"] == {"v1": 1, "v2": 0, "n": 1}
    assert report["c_t_records"][0]["c_t"] == [1, 0]
    assert report["c_t_records"][1]["c_t"] is None
