"""Hand-calculated math contrasts; uncertainty never fabricates missing truth labels."""

from __future__ import annotations

import copy
import json

import pytest
from pipelines.math_analysis import (
    EXP1_ARMS,
    LEVELS,
    MathAnalysisError,
    summarize_exp1,
    summarize_exp2,
    summarize_smoke,
)


def exp1_rows():
    # q1: interaction +1; q2: interaction -2. Mean -0.5 over TWO paired units.
    accepts = {
        "q1": {"A1": (0, 1, 1), "B": (0, 0, 0), "permuted_B": (0, 0, 1)},
        "q2": {"A1": (1, 0, 0), "B": (0, 0, 1), "permuted_B": (0, 0, 0)},
    }
    return [
        {
            "question_id": qid,
            "initial_status": "incorrect",
            "arm": arm,
            "level": level,
            "action": "accept" if accepts[qid][arm][index] else "rewrite",
            "parse_fail": False,
            "raw_scores": [0.8, 0.9],
            "observed_scores": [0.1, 0.9],
            "thresholds": [0.1, 0.5],
            "candidate_hash": f"candidate-{qid}",
        }
        for qid in accepts
        for arm in EXP1_ARMS
        for index, level in enumerate(LEVELS)
    ]


def trajectory(qid, arm, statuses, actions, parse_fails=None):
    parse_fails = parse_fails or [False] * len(actions)
    assert len(statuses) == len(actions) + 1
    return {
        "question_id": qid,
        "initial_status": "incorrect",
        "arm": arm,
        "terminal_status": statuses[-1],
        "termination_reason": "accepted" if actions[-1] == "accept" else "decision_limit",
        "is_correct": statuses[-1] == "correct"
        if statuses[-1] in ("correct", "incorrect")
        else None,
        "rounds": [
            {
                "t": index + 1,
                "action": action,
                "parse_fail": parse_fails[index],
                "input_grade": {"status": statuses[index]},
                "output_grade": {"status": statuses[index + 1]},
                "input_scores": [0.1, 0.9],
                "thresholds": [0.1, 0.5],
            }
            for index, action in enumerate(actions)
        ],
    }


def exp2_trajectories():
    # A1: an accepted wrong answer and a correct fourth rewrite; TFR=1/2.
    # B: an accepted correct answer and a wrong fourth rewrite; TFR=1/2.
    return [
        trajectory("a1", "A1", ["incorrect"] * 2, ["accept"]),
        trajectory("a2", "A1", ["incorrect"] * 4 + ["correct"], ["rewrite"] * 4),
        trajectory("b1", "B", ["incorrect", "correct", "correct"], ["rewrite", "accept"]),
        trajectory("b2", "B", ["incorrect"] * 5, ["rewrite"] * 4),
    ]


def test_exp1_hand_calculated_pairing_and_percentile_ci():
    result = summarize_exp1(exp1_rows(), n_boot=4000, seed=7)
    assert result["n_questions"] == 2 and result["n_rows"] == 18
    primary = result["primary"]
    assert primary["by_arm"]["A1"]["low"]["FA"] == 0.5
    assert primary["by_arm"]["B"]["high"]["FA"] == 0.5
    assert primary["delta_FA_A1_minus_B"]["low"]["estimate"] == 0.5
    assert primary["delta_FA_A1_minus_B"]["mid"]["estimate"] == 0.5
    assert primary["delta_FA_A1_minus_B"]["high"]["estimate"] == 0
    assert primary["interaction_high_minus_low"] == {"n": 2, "estimate": -0.5, "ci95": [-2, 1]}
    assert result["bootstrap"]["unit"] == "question_id"
    # Threshold equality is flagged: 0.1 >= 0.1 gives cell 11, never 01.
    cells = primary["by_arm"]["A1"]["high"]["evidence_sensitivity"]["cells"]
    assert cells["11"] == {"n": 2, "accept_rate": 0.5}
    json.dumps(result, allow_nan=False)


def test_exp1_parse_failure_sensitivity_keeps_pair_denominator():
    rows = exp1_rows()
    target = next(
        row
        for row in rows
        if (row["question_id"], row["arm"], row["level"]) == ("q1", "A1", "high")
    )
    target.update(action=None, parse_fail=True)
    result = summarize_exp1(rows, n_boot=50, seed=1)
    assert result["primary"]["interaction_high_minus_low"]["estimate"] == -1
    sensitivity = result["sensitivities"]
    assert sensitivity["parse_fail_as_accept"]["interaction_high_minus_low"]["estimate"] == -0.5
    assert sensitivity["complete_case"]["interaction_high_minus_low"]["estimate"] == -2
    assert sensitivity["complete_case"]["n_questions"] == 1
    assert sensitivity["complete_case"]["n_excluded_questions"] == 1
    assert result["primary"]["by_arm"]["A1"]["high"]["n"] == 2
    assert result["primary"]["by_arm"]["A1"]["high"]["parse_fail_rate"] == 0.5


def test_exp1_no_complete_cases_is_missing_not_zero():
    rows = exp1_rows()
    for row in rows:
        row.update(action=None, parse_fail=True)
    result = summarize_exp1(rows, n_boot=5, seed=0)
    assert result["primary"]["interaction_high_minus_low"]["estimate"] == 0
    complete = result["sensitivities"]["complete_case"]
    assert complete["n_questions"] == 0
    assert complete["interaction_high_minus_low"]["estimate"] is None
    assert complete["interaction_high_minus_low"]["ci95"] is None
    assert result["primary_robust_across_accountings"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "change",
    ["missing_cell", "duplicate", "missing_label", "correct_label", "mixed_scores", "mixed_state"],
)
def test_exp1_refuses_incomplete_or_unpaired_inputs(change):
    rows = exp1_rows()
    if change == "missing_cell":
        rows.pop()
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "missing_label":
        rows[0].pop("initial_status")
    elif change == "correct_label":
        rows[0]["initial_status"] = "correct"
    elif change == "mixed_scores":
        rows[0]["observed_scores"] = [0.9, 0.9]
    else:
        rows[0]["candidate_hash"] = "a different solution"
    with pytest.raises(MathAnalysisError):
        summarize_exp1(rows, n_boot=5, seed=0)


def test_exp1_order_independent_and_seeded():
    rows = exp1_rows()
    assert summarize_exp1(rows, 50, 8) == summarize_exp1(list(reversed(rows)), 50, 8)


def test_exp1_production_input_scores_must_be_shared_across_all_nine_cells():
    rows = exp1_rows()
    for row in rows:
        row["input_scores"] = row.pop("raw_scores")
    rows[0]["input_scores"] = [0.2, 0.3]
    with pytest.raises(MathAnalysisError, match="inconsistent input_scores"):
        summarize_exp1(rows, n_boot=5, seed=0)


def test_exp2_hand_calculated_failure_is_terminal_not_acceptance_or_nonconvergence():
    result = summarize_exp2(exp2_trajectories(), n_boot=4000, seed=7)
    assert result["complete"] is True
    assert result["n_trajectories"] == 4
    assert result["primary"]["by_arm"]["A1"]["TFR"] == 0.5
    assert result["primary"]["by_arm"]["B"]["TFR"] == 0.5
    assert result["primary"]["contrast"]["estimate"] == 0
    assert result["primary"]["contrast"]["ci95"] == [-1, 1]
    assert result["secondary"]["A1"]["conditional_FAR"] == 1
    assert result["secondary"]["B"]["conditional_FAR"] == 0
    assert result["secondary"]["A1"]["non_convergence_rate"] == 0.5
    assert result["round_diagnostics"]["A1"]["4"]["rewrite_transitions"] == {
        "incorrect->correct": 1
    }
    assert result["bootstrap"]["unit"] == "trajectory (stratified by arm)"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("missing_status", ["timeout", "backend_error"])
def test_exp2_unknown_truth_retains_denominator_and_returns_bounds(missing_status):
    rows = exp2_trajectories()
    rows[1]["terminal_status"] = missing_status
    rows[1]["is_correct"] = None
    rows[1]["rounds"][-1]["output_grade"]["status"] = missing_status
    result = summarize_exp2(rows, n_boot=50, seed=1)
    assert result["complete"] is False and result["n_missing_terminal_labels"] == 1
    a1 = result["primary"]["by_arm"]["A1"]
    assert a1["n"] == 2 and a1["n_incorrect"] == 1 and a1["n_missing"] == 1
    assert a1["TFR"] is None and a1["TFR_bounds"] == [0.5, 1]
    contrast = result["primary"]["contrast"]
    assert contrast["estimate"] is None and contrast["ci95"] is None
    assert contrast["missing_label_bounds"] == [0, 0.5]
    json.dumps(result, allow_nan=False)


def test_exp2_parse_fail_as_accept_uses_that_rounds_input_grade():
    rows = [
        trajectory(
            "a",
            "A1",
            ["incorrect", "incorrect", "correct", "correct"],
            [None, "rewrite", "accept"],
            [True, False, False],
        ),
        trajectory("b", "B", ["incorrect", "incorrect"], ["accept"]),
    ]
    result = summarize_exp2(rows, n_boot=20, seed=0)
    assert result["primary"]["contrast"]["estimate"] == -1
    assert result["sensitivities"]["parse_fail_as_accept"]["contrast"]["estimate"] == 0
    complete = result["sensitivities"]["complete_case"]
    assert complete["by_arm"]["A1"]["n_excluded_by_sensitivity"] == 1
    assert complete["by_arm"]["A1"]["n"] == 0
    assert complete["contrast"]["estimate"] is None
    assert result["robust_across_accountings"] is None
    assert result["round_diagnostics"]["A1"]["1"]["n_parse_fail"] == 1


def test_exp2_unresolved_round_is_retained_and_sensitivity_a_counts_it_as_failure():
    rows = [
        trajectory(
            "a",
            "A1",
            ["incorrect", "unresolved", "unresolved", "unresolved", "correct"],
            ["rewrite", None, "rewrite", "rewrite"],
            [False, True, False, False],
        ),
        trajectory("b", "B", ["incorrect", "incorrect"], ["accept"]),
    ]
    result = summarize_exp2(rows, n_boot=10, seed=0)
    assert result["primary"]["complete"] is True
    assert result["complete"] is True
    assert result["sensitivities"]["parse_fail_as_accept"]["by_arm"]["A1"]["TFR"] == 1
    assert result["unresolved_rates"]["by_arm_round"]["A1"]["2"]["unresolved_rate"] == 1
    assert result["unresolved_rates"]["status"] == "ATTENTION_UNRESOLVED_RATE"


def test_exp2_terminal_unresolved_is_failure_and_sensitivity_c_excludes_only_that_trajectory():
    rows = exp2_trajectories()
    rows[1]["terminal_status"] = "unresolved"
    rows[1]["is_correct"] = None
    rows[1]["task_success"] = False
    rows[1]["rounds"][-1]["output_grade"]["status"] = "unresolved"
    result = summarize_exp2(rows, n_boot=50, seed=1)
    assert result["complete"] is True
    assert result["n_missing_terminal_labels"] == 0
    assert result["n_terminal_unresolved"] == 1
    primary = result["primary"]
    assert primary["by_arm"]["A1"]["n_incorrect"] == 1
    assert primary["by_arm"]["A1"]["n_unresolved"] == 1
    assert primary["by_arm"]["A1"]["n_task_failures"] == 2
    assert primary["by_arm"]["A1"]["TFR"] == 1
    assert primary["contrast"]["estimate"] == 0.5
    sensitivity = result["sensitivities"]["terminal_unresolved_excluded"]
    assert sensitivity["by_arm"]["A1"]["n"] == 1
    assert sensitivity["by_arm"]["A1"]["n_excluded_by_sensitivity"] == 1
    assert sensitivity["contrast"]["estimate"] == 0.5
    last = result["unresolved_rates"]["by_arm_round"]["A1"]["5"]
    assert last == {
        "n_texts": 1,
        "n_unresolved": 1,
        "unresolved_rate": 1,
        "above_two_percent": True,
    }
    assert "refusal_like" not in json.dumps(result)
    json.dumps(result, allow_nan=False)


def test_sensitivity_c_no_remaining_arm_reports_unestimable_not_zero():
    rows = [
        trajectory("a", "A1", ["incorrect"] * 4 + ["unresolved"], ["rewrite"] * 4),
        trajectory("b", "B", ["incorrect", "incorrect"], ["accept"]),
    ]
    result = summarize_exp2(rows, n_boot=10, seed=1)
    assert result["primary"]["complete"] is True
    sensitivity = result["sensitivities"]["terminal_unresolved_excluded"]
    assert sensitivity["complete"] is False
    assert sensitivity["contrast"]["estimate"] is None


def test_unresolved_threshold_is_strict_and_reports_fourth_rewrite():
    rows = []
    for arm in ("A1", "B"):
        for i in range(50):
            statuses = ["incorrect"] * 4 + ["unresolved" if i == 0 else "correct"]
            rows.append(trajectory(f"{arm}-{i}", arm, statuses, ["rewrite"] * 4))
    report = summarize_exp2(rows, n_boot=5, seed=0)
    for arm in ("A1", "B"):
        cell = report["unresolved_rates"]["by_arm_round"][arm]["5"]
        assert cell["unresolved_rate"] == 0.02
        assert cell["above_two_percent"] is False
    assert report["unresolved_rates"]["alert"] is False


def test_smoke_only_a1_exactly_30_and_no_arm_comparisons():
    rows = [trajectory(str(i), "A1", ["incorrect", "incorrect"], ["accept"]) for i in range(30)]
    report = summarize_smoke(rows)
    assert report["stage"] == "SMOKE" and report["is_research_result"] is False
    assert report["n_trajectories"] == 30 and report["task_failure_rate"] == 1
    encoded = json.dumps(report)
    assert "contrast" not in encoded and "A1_minus_B" not in encoded
    assert '"B"' not in encoded and "sensitivities" not in encoded
    rows[0]["arm"] = "B"
    with pytest.raises(MathAnalysisError):
        summarize_smoke(rows)
    with pytest.raises(MathAnalysisError, match="30"):
        summarize_smoke(rows[1:])


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_id",
        "fifth_round",
        "short_limit",
        "continued_after_accept",
        "wrong_terminal",
        "changed_on_parse_fail",
        "missing_initial",
    ],
)
def test_exp2_refuses_corrupt_or_incomplete_trajectory(change):
    rows = exp2_trajectories()
    if change == "duplicate_id":
        rows[2]["question_id"] = rows[0]["question_id"]
    elif change == "fifth_round":
        rows[1]["rounds"].append(copy.deepcopy(rows[1]["rounds"][-1]))
    elif change == "short_limit":
        rows[3]["rounds"].pop()
    elif change == "continued_after_accept":
        rows[1]["rounds"][0]["action"] = "accept"
    elif change == "wrong_terminal":
        rows[1]["terminal_status"] = "incorrect"
    elif change == "changed_on_parse_fail":
        rows[1]["rounds"][-1].update(action=None, parse_fail=True)
    else:
        rows[0].pop("initial_status")
    with pytest.raises(MathAnalysisError):
        summarize_exp2(rows, n_boot=5, seed=0)


def test_unknown_thresholds_are_not_replaced_with_half():
    rows = exp2_trajectories()
    for trajectory_row in rows:
        for round_row in trajectory_row["rounds"]:
            round_row.pop("thresholds")
    report = summarize_exp2(rows, n_boot=5, seed=0)
    evidence = report["round_diagnostics"]["A1"]["1"]["evidence_sensitivity"]
    assert evidence["status"] == "NOT_COMPUTED"
    assert evidence["n_missing_scores_or_thresholds"] == 2


def test_exp2_order_independent_and_no_input_mutation():
    rows = exp2_trajectories()
    before = copy.deepcopy(rows)
    report = summarize_exp2(rows, n_boot=50, seed=9)
    assert report == summarize_exp2(list(reversed(rows)), n_boot=50, seed=9)
    assert rows == before


@pytest.mark.parametrize("n_boot,seed", [(0, 1), (True, 1), (1, -1), (1, True)])
def test_bootstrap_arguments_are_explicit(n_boot, seed):
    with pytest.raises(MathAnalysisError):
        summarize_exp1(exp1_rows(), n_boot, seed)
    with pytest.raises(MathAnalysisError):
        summarize_exp2(exp2_trajectories(), n_boot, seed)


def test_mixed_provenance_and_formal_stub_cannot_be_promoted():
    rows = exp1_rows()
    for row in rows:
        row.update(domain="math", stage="formal", execution_mode="offline_stub")
    with pytest.raises(MathAnalysisError, match="stub/fake"):
        summarize_exp1(rows, 5, 0)
    rows = exp2_trajectories()
    rows[0]["stage"] = "formal"
    with pytest.raises(MathAnalysisError, match="partial stage"):
        summarize_exp2(rows, 5, 0)


def test_permuted_control_and_bootstrap_follow_core_preregistered_analysis():
    from analysis import exp1_did

    rows = exp1_rows()
    core_rows = [
        {**r, "state_id": r["question_id"], "lam": r["level"], "e_label": "UNSAFE"} for r in rows
    ]
    report = summarize_exp1(rows, n_boot=23, seed=11)["primary"]
    expected = exp1_did.did(core_rows, control="permuted_B", treated="B", n_boot=23, seed=11)
    actual = report["secondary_permuted_B_interaction_high_minus_low"]
    assert actual["estimate"] == expected["estimate"]
    assert actual["ci95"] == expected["ci95"]
    assert report["secondary_permuted_B_minus_B"]["high"]["estimate"] == 0
    main = exp1_did.did(core_rows, n_boot=23, seed=11)
    assert report["interaction_high_minus_low"]["ci95"] == main["ci95"]


def test_rewrite_counts_exclude_accept_and_parse_failure_decisions():
    rows = [
        trajectory(
            "a",
            "A1",
            ["incorrect"] * 5,
            [None, "rewrite", None, "rewrite"],
            [True, False, True, False],
        ),
        trajectory("b", "B", ["incorrect", "correct", "correct"], ["rewrite", "accept"]),
    ]
    result = summarize_exp2(rows, n_boot=5, seed=0)
    assert result["secondary"]["A1"]["rewrite_rounds_distribution"] == {"2": 1}
    assert result["secondary"]["B"]["rewrite_rounds_distribution"] == {"1": 1}
    assert result["secondary"]["B"]["rewrite_rounds_median"] == 1
    assert result["secondary"]["B"]["decision_rounds_distribution"] == {"2": 1}


def test_mechanism_reports_accept_accuracy_and_rewrite_improvement():
    report = summarize_exp2(exp2_trajectories(), n_boot=5, seed=0)
    a, b = report["mechanism"]["A1"], report["mechanism"]["B"]
    assert a["accept_decision_accuracy"] == 0
    assert b["accept_decision_accuracy"] == 1
    assert a["rewrite_improvement_rate"] == 0.25
    assert b["rewrite_improvement_rate"] == 0.2
    assert a["n_rewrites"] == 4 and b["n_rewrites"] == 5


def test_unresolved_task_failure_keeps_mechanism_denominator_and_question_identity():
    rows = [
        trajectory(
            "same-answer-a", "A1", ["incorrect", "unresolved", "unresolved"], ["rewrite", "accept"]
        ),
        trajectory(
            "same-answer-b", "B", ["incorrect", "correct", "correct"], ["rewrite", "accept"]
        ),
    ]
    report = summarize_exp2(rows, n_boot=5, seed=0)
    assert report["mechanism"]["A1"]["accept_decision_accuracy"] == 0
    assert report["mechanism"]["B"]["accept_decision_accuracy"] == 1
    assert report["mechanism"]["A1"]["rewrite_improvement_rate"] == 0
    assert report["mechanism"]["B"]["rewrite_improvement_rate"] == 1
    assert report["mechanism"]["A1"]["n_rows_without_judgment"] == 0


def overlength(qid, arm, statuses, actions):
    """A trajectory ended by a rewrite longer than a V's L_in (2026-10-13)."""
    row = trajectory(qid, arm, statuses, actions)
    row["termination_reason"] = "v_input_overlength"
    row["v_input_overlength"] = {"t": len(actions) + 1, "tokens": 4200, "L_in": 4096}
    return row


def test_overlength_is_its_final_text_reported_by_arm_and_dropped_like_parse_fail():
    from pipelines import math_exp2_loop
    from pipelines.math_analysis import OVERLENGTH

    assert OVERLENGTH == math_exp2_loop.OVERLENGTH == "v_input_overlength"
    rows = exp2_trajectories() + [
        overlength("a3", "A1", ["incorrect", "correct"], ["rewrite"]),
        trajectory("b3", "B", ["incorrect"] * 2, ["accept"]),
    ]
    report = summarize_exp2(rows, n_boot=200, seed=1)
    # main analysis: the overlength rewrite is x_T, judged by E (correct here)
    assert report["primary"]["by_arm"]["A1"]["n"] == 3
    assert report["primary"]["by_arm"]["A1"]["n_task_failures"] == 1
    assert report["secondary"]["A1"]["n_v_input_overlength"] == 1
    assert report["secondary"]["A1"]["v_input_overlength_rate"] == pytest.approx(1 / 3)
    assert report["secondary"]["B"]["n_v_input_overlength"] == 0
    # complete-case sensitivity (paper App. E, as a parse_fail trajectory): dropped
    complete = report["sensitivities"]["complete_case"]["by_arm"]["A1"]
    assert complete["n"] == 2 and complete["n_excluded_by_sensitivity"] == 1
    smoke = [overlength(f"s{i}", "A1", ["incorrect"] * 2, ["rewrite"]) for i in range(2)]
    smoke += [trajectory(f"t{i}", "A1", ["incorrect"] * 2, ["accept"]) for i in range(28)]
    assert summarize_smoke(smoke)["n_v_input_overlength"] == 2


@pytest.mark.parametrize(
    "change",
    [
        {"v_input_overlength": None},
        {"rounds_after_accept": True},
    ],
)
def test_overlength_needs_its_record_and_a_last_rewrite(change):
    row = overlength("a3", "A1", ["incorrect", "correct"], ["rewrite"])
    if "v_input_overlength" in change:
        row["v_input_overlength"] = None
    else:
        row["rounds"][-1]["action"] = "accept"
        row["rounds"][-1]["output_grade"] = row["rounds"][-1]["input_grade"]
        row["terminal_status"] = "incorrect"
        row["is_correct"] = False
    with pytest.raises(MathAnalysisError, match="v_input_overlength"):
        summarize_exp2(
            exp2_trajectories() + [row, trajectory("b3", "B", ["incorrect"] * 2, ["accept"])],
            n_boot=200,
            seed=1,
        )


def test_smoke_and_exp2_report_the_overlength_rate_and_which_v_ended_it():
    """2026-10-16: the smoke reports the overlength rate; GLM's loop L_in is 4,096,
    R1-Distill-Llama's 1,344, so the ended trajectories are told apart by V."""

    def ended(qid, arm, verifier_id, l_in):
        row = overlength(qid, arm, ["incorrect", "incorrect"], ["rewrite"])
        row["v_input_overlength"] = {
            "t": 2,
            "tokens": l_in + 10,
            "L_in": l_in,
            "verifier_id": verifier_id,
        }
        return row

    rows = [trajectory(str(i), "A1", ["incorrect", "incorrect"], ["accept"]) for i in range(27)]
    rows += [
        ended("r1a", "A1", "r1", 1344),
        ended("r1b", "A1", "r1", 1344),
        ended("g", "A1", "glm", 4096),
    ]
    report = summarize_smoke(rows)
    assert report["n_v_input_overlength"] == 3 and report["v_input_overlength_rate"] == 0.1
    assert report["v_input_overlength_by_verifier"] == {"glm": 1, "r1": 2}
    two_arms = exp2_trajectories() + [ended("x", "A1", "r1", 1344), ended("y", "B", "r1", 1344)]
    secondary = summarize_exp2(two_arms, n_boot=50, seed=1)["secondary"]
    assert secondary["A1"]["v_input_overlength_by_verifier"] == {"r1": 1}
    assert secondary["B"]["v_input_overlength_by_verifier"] == {"r1": 1}
