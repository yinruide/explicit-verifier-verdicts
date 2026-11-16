"""Paired/stratified math analysis under protocol v1.1 Appendix M.

These functions analyse validated run records, including offline control records.
They do not promote a stub/development run to formal status; the run manifest and
stage-completion validation remain the execution layer's responsibility.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
from analysis import evidence_sensitivity as core_evidence
from analysis import exp1_did as core_exp1
from analysis import exp2_tfr as core_exp2
from analysis._common import accepted, bootstrap_ci, mean_or_nan, same_sign

from rafr.contracts.schemas import text_hash

EXP1_ARMS = ("A1", "B", "permuted_B")
LEVELS = ("low", "mid", "high")
GRADES = frozenset({"correct", "incorrect", "unresolved", "timeout", "backend_error"})
KNOWN_GRADES = frozenset({"correct", "incorrect"})
TASK_GRADES = KNOWN_GRADES | {"unresolved"}
#: Termination of an Exp 2 trajectory whose rewrite is longer than a V's L_in (2026-10-13;
#: ``pipelines.math_exp2_loop.OVERLENGTH``).
OVERLENGTH = "v_input_overlength"


class MathAnalysisError(ValueError):
    """Incomplete or inconsistent records cannot produce a complete analysis."""


def _settings(n_boot: int, seed: int) -> dict:
    if type(n_boot) is not int or n_boot < 1:
        raise MathAnalysisError("n_boot must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise MathAnalysisError("seed must be a nonnegative integer")
    return {"method": "percentile_bootstrap", "level": 0.95, "n_boot": n_boot, "seed": seed}


def _identifier(record: dict) -> str:
    value = record.get("question_id")
    if not isinstance(value, str) or not value.strip():
        raise MathAnalysisError("nonempty question_id required")
    if record.get("initial_status") != "incorrect":
        raise MathAnalysisError("initial_status must explicitly be incorrect")
    return value


def _decision(record: dict) -> None:
    if type(record.get("parse_fail")) is not bool:
        raise MathAnalysisError("parse_fail must explicitly be boolean")
    if record["parse_fail"]:
        if record.get("action") is not None:
            raise MathAnalysisError("parse_fail requires normalized action=None")
    elif record.get("action") not in ("accept", "rewrite"):
        raise MathAnalysisError("successful decision requires accept/rewrite action")


def _accepted(record: dict, rule: str) -> int:
    return int(bool(accepted(record, rule)))


def _paired_contrast(values: list[float], n_boot: int, seed: int) -> dict:
    if not values:
        return {"n": 0, "estimate": None, "ci95": None}
    lo, hi = bootstrap_ci(values, mean_or_nan, n_boot=n_boot, seed=seed)
    return {
        "n": len(values),
        "estimate": mean_or_nan(values),
        "ci95": [lo, hi],
    }


def _provenance(records: list[dict]) -> dict:
    result = {}
    for field in ("domain", "stage", "execution_mode", "run_id"):
        present = [row[field] for row in records if field in row]
        if present and len(present) != len(records):
            raise MathAnalysisError(f"partial {field} provenance")
        if present and (
            any(not isinstance(value, str) for value in present) or len(set(present)) > 1
        ):
            raise MathAnalysisError(f"mixed {field} provenance")
        result[field] = present[0] if present else None
    if result["domain"] not in (None, "math"):
        raise MathAnalysisError("non-math records")
    if result["stage"] == "formal" and result["execution_mode"] in ("offline_stub", "stub", "fake"):
        raise MathAnalysisError("formal records cannot carry stub/fake execution provenance")
    return result


def _score_pair(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise MathAnalysisError("score/threshold value must have two numbers")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        or not 0 <= item <= 1
        for item in value
    ):
        raise MathAnalysisError("scores/thresholds must be finite and in [0,1]")
    return [float(item) for item in value]


def _evidence_cells(records: list[dict], rule: str, score_field: str) -> dict:
    core_rows = []
    missing = 0
    for row in records:
        scores, thresholds = _score_pair(row.get(score_field)), _score_pair(row.get("thresholds"))
        if scores is None or thresholds is None:
            missing += 1
            continue
        core_rows.append(
            {
                "arm": "A1",
                "t": 1,
                "action": row["action"],
                "parse_fail": row["parse_fail"],
                "s1_tilde": scores[0],
                "s2_tilde": scores[1],
                "tau1": thresholds[0],
                "tau2": thresholds[1],
            }
        )
    table = core_evidence.exp2(core_rows, accounting=rule)["cells"].get("1", {}).get("A1", {})
    return {
        "status": "NOT_COMPUTED" if missing == len(records) else "DESCRIPTIVE_ONLY",
        "n_missing_scores_or_thresholds": missing,
        "cells": {
            f"{a}{b}": {
                "n": table.get(f"({a},{b})", {}).get("n", 0),
                "accept_rate": table[f"({a},{b})"]["accept_rate"]
                if table and table[f"({a},{b})"]["n"]
                else None,
            }
            for a, b in core_evidence.PAIRS
        },
    }


def _exp1_accounting(
    index: dict[tuple[str, str, str], dict],
    question_ids: list[str],
    rule: str,
    n_boot: int,
    seed: int,
) -> dict:
    by_arm = {}
    for arm in EXP1_ARMS:
        by_arm[arm] = {}
        for level in LEVELS:
            rows = [index[qid, arm, level] for qid in question_ids]
            n = len(rows)
            accepts = sum(_accepted(row, rule) for row in rows)
            parse_fails = sum(row["parse_fail"] for row in rows)
            by_arm[arm][level] = {
                "n": n,
                "n_accept": accepts,
                "FA": accepts / n if n else None,
                "n_parse_fail": parse_fails,
                "parse_fail_rate": parse_fails / n if n else None,
                "parse_fail_above_two_percent": parse_fails / n > 0.02 if n else False,
                "evidence_sensitivity": _evidence_cells(rows, rule, "observed_scores"),
            }

    def difference(qid: str, level: str, comparator: str) -> int:
        return _accepted(index[qid, "A1", level], rule) - _accepted(
            index[qid, comparator, level], rule
        )

    deltas = {
        level: _paired_contrast([difference(qid, level, "B") for qid in question_ids], n_boot, seed)
        for level in LEVELS
    }
    core_rows = [
        {**index[qid, arm, level], "state_id": qid, "lam": level, "e_label": "UNSAFE"}
        for qid in question_ids
        for arm in EXP1_ARMS
        for level in LEVELS
    ]

    def did(control):
        if not question_ids:
            return {"n": 0, "estimate": None, "ci95": None}
        result = core_exp1.did(
            core_rows, accounting=rule, control=control, treated="B", n_boot=n_boot, seed=seed
        )
        return {"n": result["n_states"], "estimate": result["estimate"], "ci95": result["ci95"]}

    interaction = did("A1")
    permuted = {
        level: _paired_contrast(
            [
                _accepted(index[qid, "permuted_B", level], rule)
                - _accepted(index[qid, "B", level], rule)
                for qid in question_ids
            ],
            n_boot,
            seed,
        )
        for level in LEVELS
    }
    return {
        "n_questions": len(question_ids),
        "by_arm": by_arm,
        "delta_FA_A1_minus_B": deltas,
        "interaction_high_minus_low": interaction,
        "secondary_permuted_B_minus_B": permuted,
        "secondary_permuted_B_interaction_high_minus_low": did("permuted_B"),
        "secondary_levels": list(LEVELS),
    }


def summarize_exp1(rows: list[dict], n_boot: int = 10_000, seed: int = 0) -> dict:
    """Validate nine paired cells per initial-incorrect question, then estimate contrasts."""
    bootstrap = {**_settings(n_boot, seed), "unit": "question_id"}
    if not isinstance(rows, list) or not rows:
        raise MathAnalysisError("Exp1 requires nonempty records")
    provenance = _provenance(rows)
    index = {}
    for row in rows:
        qid = _identifier(row)
        _decision(row)
        arm, level = row.get("arm"), row.get("level")
        if arm not in EXP1_ARMS or level not in LEVELS:
            raise MathAnalysisError("unknown Exp1 arm/level")
        key = (qid, arm, level)
        if key in index:
            raise MathAnalysisError("duplicate Exp1 question/arm/level")
        index[key] = row
    question_ids = sorted({key[0] for key in index})
    expected = {(qid, arm, level) for qid in question_ids for arm in EXP1_ARMS for level in LEVELS}
    if set(index) != expected:
        raise MathAnalysisError("each Exp1 question must have all nine cells")
    for qid in question_ids:
        group = [index[qid, arm, level] for arm in EXP1_ARMS for level in LEVELS]
        for field in ("candidate_hash", "input_scores", "raw_scores"):
            values = [row[field] for row in group if field in row]
            if values and (len(values) != 9 or any(value != values[0] for value in values)):
                raise MathAnalysisError(f"Exp1 shared state has inconsistent {field}")
        for level in LEVELS:
            observations = [index[qid, arm, level].get("observed_scores") for arm in EXP1_ARMS]
            if any(value is not None for value in observations) and any(
                value != observations[0] for value in observations
            ):
                raise MathAnalysisError("Exp1 arms received different observed scores")
    complete_qids = [
        qid
        for qid in question_ids
        if not any(index[qid, arm, level]["parse_fail"] for arm in EXP1_ARMS for level in LEVELS)
    ]
    primary = _exp1_accounting(index, question_ids, "main", n_boot, seed)
    sensitivities = {
        "parse_fail_as_accept": _exp1_accounting(
            index, question_ids, "parse_fail_as_accept", n_boot, seed
        ),
        "complete_case": {
            **_exp1_accounting(index, complete_qids, "main", n_boot, seed),
            "n_excluded_questions": len(question_ids) - len(complete_qids),
        },
    }
    estimates = [
        block["interaction_high_minus_low"]["estimate"]
        for block in (primary, *sensitivities.values())
    ]
    return {
        "experiment": "math_exp1",
        "provenance": provenance,
        "cohort": "initial_status=incorrect; one shared state per question",
        "n_questions": len(question_ids),
        "n_rows": len(rows),
        "bootstrap": bootstrap,
        "primary": primary,
        "sensitivities": sensitivities,
        "primary_robust_across_accountings": same_sign(estimates)
        if all(value is not None for value in estimates)
        else None,
        "implementation": "analysis.exp1_did.did and analysis._common.bootstrap_ci",
        "inference_note": (
            "Pilot estimates and uncertainty only; no significance claim or effect gate."
        ),
    }


def _grade(record: dict, field: str) -> str:
    value = record.get(field)
    if not isinstance(value, dict) or value.get("status") not in GRADES:
        raise MathAnalysisError(f"{field} must contain an explicit recognized grade status")
    return value["status"]


def _validate_trajectories(trajectories: list[dict], *, smoke: bool = False) -> None:
    seen = set()
    for trajectory in trajectories:
        qid = _identifier(trajectory)
        if qid in seen:
            raise MathAnalysisError("one trajectory per question across both arms is required")
        seen.add(qid)
        if trajectory.get("arm") not in (("A1",) if smoke else ("A1", "B")):
            raise MathAnalysisError("Exp2 arm must be A1 or B")
        rounds = trajectory.get("rounds")
        if not isinstance(rounds, list) or not 1 <= len(rounds) <= 4:
            raise MathAnalysisError("Exp2 requires one to four completed decision rounds")
        for t, row in enumerate(rounds, 1):
            if type(row.get("t")) is not int or row["t"] != t:
                raise MathAnalysisError("Exp2 rounds must be consecutive from one")
            _decision(row)
            before, after = _grade(row, "input_grade"), _grade(row, "output_grade")
            if t == 1 and before != "incorrect":
                raise MathAnalysisError("Exp2 first input grade must be incorrect")
            if row["parse_fail"] or row["action"] == "accept":
                if before != after:
                    raise MathAnalysisError("accept/parse_fail cannot change the candidate grade")
            if row["action"] == "accept" and t != len(rounds):
                raise MathAnalysisError("no decision is allowed after accept")
            if t > 1 and before != rounds[t - 2]["output_grade"]["status"]:
                raise MathAnalysisError("Exp2 consecutive candidate grades disagree")
        terminal = trajectory.get("terminal_status")
        if terminal not in GRADES or terminal != rounds[-1]["output_grade"]["status"]:
            raise MathAnalysisError("terminal status must match the last output grade")
        accepted = rounds[-1]["action"] == "accept" and not rounds[-1]["parse_fail"]
        if trajectory.get("termination_reason") == OVERLENGTH:
            # 2026-10-13: the last rewrite was too long for a V; it is x_T
            if (
                accepted
                or rounds[-1]["parse_fail"]
                or rounds[-1]["action"] != "rewrite"
                or len(rounds) >= 4
                or not isinstance(trajectory.get(OVERLENGTH), dict)
            ):
                raise MathAnalysisError("v_input_overlength must follow a rewrite before round 4")
        elif trajectory.get("termination_reason") != ("accepted" if accepted else "decision_limit"):
            raise MathAnalysisError("termination_reason disagrees with final action")
        elif not accepted and len(rounds) != 4:
            raise MathAnalysisError("decision_limit must consume exactly four decisions")
        if "is_correct" in trajectory:
            expected = terminal == "correct" if terminal in KNOWN_GRADES else None
            if trajectory["is_correct"] is not expected:
                raise MathAnalysisError("terminal is_correct disagrees with status")
        if "task_success" in trajectory:
            expected = terminal == "correct" if terminal in TASK_GRADES else None
            if trajectory["task_success"] is not expected:
                raise MathAnalysisError("terminal task_success disagrees with status")
    if smoke:
        return
    if {row["arm"] for row in trajectories} != {"A1", "B"}:
        raise MathAnalysisError("Exp2 requires at least one randomized trajectory in each arm")
    arm_counts = Counter(row["arm"] for row in trajectories)
    if abs(arm_counts["A1"] - arm_counts["B"]) > 1:
        raise MathAnalysisError("Exp2 randomized arms must be balanced within one trajectory")


def _exp2_accounting(trajectories: list[dict], rule: str, n_boot: int, seed: int) -> dict:
    values: dict[str, list[str]] = {"A1": [], "B": []}
    excluded = {"A1": 0, "B": 0}
    for trajectory in trajectories:
        arm = trajectory["arm"]
        first_failure = next((row for row in trajectory["rounds"] if row["parse_fail"]), None)
        # M.8 (2026-10-13): a v_input_overlength trajectory is dropped as a parse_fail one
        overlong = trajectory["termination_reason"] == OVERLENGTH
        if rule == "complete_case" and (first_failure is not None or overlong):
            excluded[arm] += 1
            continue
        status = trajectory["terminal_status"]
        if rule == "terminal_unresolved_excluded" and status == "unresolved":
            excluded[arm] += 1
            continue
        if rule == "parse_fail_as_accept" and first_failure is not None:
            status = first_failure["input_grade"]["status"]
        values[arm].append(status)
    by_arm = {}
    for arm, statuses in values.items():
        n = len(statuses)
        counts = Counter(statuses)
        unknown = sum(status not in TASK_GRADES for status in statuses)
        failures = counts["incorrect"] + counts["unresolved"]
        by_arm[arm] = {
            "n": n,
            "n_correct": counts["correct"],
            "n_incorrect": counts["incorrect"],
            "n_unresolved": counts["unresolved"],
            "n_task_failures": failures,
            "n_missing": unknown,
            "missing_by_status": {key: counts[key] for key in sorted(GRADES - TASK_GRADES)},
            "TFR": failures / n if n and not unknown else None,
            "TFR_bounds": [failures / n, (failures + unknown) / n] if n else None,
            "n_excluded_by_sensitivity": excluded[arm],
        }
    complete = all(by_arm[arm]["n"] > 0 and by_arm[arm]["n_missing"] == 0 for arm in values)
    estimate, ci = None, None
    if complete:
        # The math-specific accounting only selects/relabels trajectories. The
        # randomized arm contrast and its bootstrap are the unmodified Core.
        result = core_exp2.contrast(
            [
                {"arm": arm, "task_success": status == "correct"}
                for arm, statuses in values.items()
                for status in statuses
            ],
            n_boot=n_boot,
            seed=seed,
        )
        estimate, ci = result["estimate"], result["ci95"]
    bounds_a, bounds_b = by_arm["A1"]["TFR_bounds"], by_arm["B"]["TFR_bounds"]
    bounds = (
        [bounds_a[0] - bounds_b[1], bounds_a[1] - bounds_b[0]] if bounds_a and bounds_b else None
    )
    return {
        "by_arm": by_arm,
        "complete": complete,
        "contrast": {
            "definition": "TFR_A1-TFR_B; positive favors B",
            "estimate": estimate,
            "ci95": ci,
            "missing_label_bounds": bounds,
        },
    }


def _round_diagnostics(trajectories: list[dict], *, arms=("A1", "B")) -> dict:
    result = {}
    for arm in arms:
        result[arm] = {}
        for t in range(1, 5):
            rows = [
                row
                for trajectory in trajectories
                if trajectory["arm"] == arm
                for row in trajectory["rounds"]
                if row["t"] == t
            ]
            known = [row for row in rows if row["input_grade"]["status"] in KNOWN_GRADES]
            transitions = Counter(
                f"{row['input_grade']['status']}->{row['output_grade']['status']}"
                for row in rows
                if not row["parse_fail"] and row["action"] == "rewrite"
            )
            n = len(rows)
            input_unresolved = sum(row["input_grade"]["status"] == "unresolved" for row in rows)
            output_unresolved = sum(row["output_grade"]["status"] == "unresolved" for row in rows)
            result[arm][str(t)] = {
                "n_decisions": n,
                "n_accept": sum(_accepted(row, "main") for row in rows),
                "n_parse_fail": sum(row["parse_fail"] for row in rows),
                "parse_fail_rate": sum(row["parse_fail"] for row in rows) / n if n else None,
                "parse_fail_above_two_percent": (
                    sum(row["parse_fail"] for row in rows) / n > 0.02 if n else False
                ),
                "n_known_input_grade": len(known),
                "n_unknown_input_grade": n - len(known),
                "n_input_unresolved": input_unresolved,
                "n_output_unresolved": output_unresolved,
                "input_unresolved_rate": input_unresolved / n if n else None,
                "output_unresolved_rate": output_unresolved / n if n else None,
                "unresolved_above_two_percent": (
                    max(input_unresolved, output_unresolved) / n > 0.02 if n else False
                ),
                "decision_accuracy_on_known_inputs": (
                    sum(
                        _accepted(row, "main") == (row["input_grade"]["status"] == "correct")
                        for row in known
                    )
                    / len(known)
                    if known
                    else None
                ),
                "rewrite_transitions": dict(sorted(transitions.items())),
                "evidence_sensitivity": _evidence_cells(rows, "main", "input_scores"),
                "interpretation": (
                    "DESCRIPTIVE_ONLY; do not stratify treatment by downstream reliability"
                ),
            }
    return result


def _unresolved_rates(trajectories: list[dict], *, arms=("A1", "B")) -> dict:
    """Rates on every recorded x_t, including x_5 after the fourth rewrite.

    Accepted/retained texts are not duplicated at a new time point merely because
    they are terminal. The t=5 cell exists only for the final budgeted rewrite.
    """
    result = {}
    alerts = []
    for arm in arms:
        groups = {t: [] for t in range(1, 6)}
        for trajectory in trajectories:
            if trajectory["arm"] != arm:
                continue
            for row in trajectory["rounds"]:
                groups[row["t"]].append(row["input_grade"]["status"])
            last = trajectory["rounds"][-1]
            if last["t"] == 4 and last["action"] == "rewrite" and not last["parse_fail"]:
                groups[5].append(trajectory["terminal_status"])
        result[arm] = {}
        for t, statuses in groups.items():
            n, unresolved = len(statuses), statuses.count("unresolved")
            rate = unresolved / n if n else None
            alert = rate is not None and rate > 0.02
            result[arm][str(t)] = {
                "n_texts": n,
                "n_unresolved": unresolved,
                "unresolved_rate": rate,
                "above_two_percent": alert,
            }
            if alert:
                alerts.append({"arm": arm, "t": t, "n": n, "unresolved_rate": rate})
    return {
        "by_arm_round": result,
        "threshold": 0.02,
        "alert": bool(alerts),
        "alert_cells": alerts,
        "status": "ATTENTION_UNRESOLVED_RATE" if alerts else "REPORTED",
        "denominator": "all recorded texts at each arm and round, including E errors",
    }


def _mechanism(trajectories: list[dict]) -> dict:
    """Adapt per-question E records to Core's label-join interface.

    Core consumes the hash only as a join key. The opaque keys below identify a
    question, round and input/output side, so identical solution text on different
    mathematical questions can never overwrite one another's E labels.
    UNRESOLVED is task failure under M.8; E errors remain missing.
    """
    import json

    rows, truth = [], {}
    for trajectory in trajectories:
        for row in trajectory["rounds"]:
            keys = {
                side: json.dumps([trajectory["question_id"], row["t"], side])
                for side in ("input", "output")
            }
            for side, key in keys.items():
                status = row[f"{side}_grade"]["status"]
                if status in TASK_GRADES:
                    truth[text_hash(key)] = "SAFE" if status == "correct" else "UNSAFE"
            rows.append(
                {
                    "arm": trajectory["arm"],
                    "action": row["action"],
                    "parse_fail": row["parse_fail"],
                    "x_t": keys["input"],
                    "revised_text": keys["output"],
                }
            )
    result = core_exp2.mechanism(rows, truth)
    for report in result.values():
        report["rewrite_transitions"] = {
            key.replace("UNSAFE", "task_failure").replace("SAFE", "task_success"): value
            for key, value in report["rewrite_transitions"].items()
        }
    return {
        arm: {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in report.items()
        }
        for arm, report in result.items()
    }


def _overlength_by_verifier(trajectories: list[dict]) -> dict[str, int]:
    """Ended trajectories by the V whose loop L_in the text exceeded (the first V that refused
    it; II.10, 2026-10-13)."""
    counts = Counter(
        (row.get(OVERLENGTH) or {}).get("verifier_id", "unknown")
        for row in trajectories
        if row["termination_reason"] == OVERLENGTH
    )
    return dict(sorted(counts.items()))


def summarize_exp2(trajectories: list[dict], n_boot: int = 10_000, seed: int = 0) -> dict:
    """M.8: unresolved is a task failure; E backend errors remain unknown."""
    bootstrap = {**_settings(n_boot, seed), "unit": "trajectory (stratified by arm)"}
    if not isinstance(trajectories, list) or not trajectories:
        raise MathAnalysisError("Exp2 requires nonempty trajectories")
    provenance = _provenance(trajectories)
    _validate_trajectories(trajectories)
    trajectories = sorted(trajectories, key=lambda row: row["question_id"])
    primary = _exp2_accounting(trajectories, "main", n_boot, seed)
    secondary = {}
    for arm in ("A1", "B"):
        members = [row for row in trajectories if row["arm"] == arm]
        accepted = [row for row in members if row["termination_reason"] == "accepted"]
        accepted_known = all(row["terminal_status"] in TASK_GRADES for row in accepted)
        counts = Counter(len(row["rounds"]) for row in members)
        core_secondary = core_exp2.secondary(
            [
                {
                    "arm": arm,
                    "accepted": row["termination_reason"] == "accepted",
                    "task_success": row["terminal_status"] == "correct",
                    "n_rewrites": sum(
                        not r["parse_fail"] and r["action"] == "rewrite" for r in row["rounds"]
                    ),
                }
                for row in members
            ]
        )[arm]
        overlong = sum(row["termination_reason"] == OVERLENGTH for row in members)
        secondary[arm] = {
            "n": len(members),
            "n_accepted": len(accepted),
            "n_v_input_overlength": overlong,
            "v_input_overlength_rate": overlong / len(members),
            "v_input_overlength_by_verifier": _overlength_by_verifier(members),
            "conditional_FAR": (
                sum(row["terminal_status"] != "correct" for row in accepted) / len(accepted)
                if accepted and accepted_known
                else None
            ),
            "non_convergence_rate": (len(members) - len(accepted)) / len(members),
            "decision_rounds_distribution": {
                str(key): value for key, value in sorted(counts.items())
            },
            "decision_rounds_median": float(np.median([len(row["rounds"]) for row in members])),
            "rewrite_rounds_median": core_secondary["rewrite_rounds_median"],
            "rewrite_rounds_distribution": {
                str(key): value
                for key, value in core_secondary["rewrite_rounds_distribution"].items()
            },
            "right_censored_at_K": 4,
        }
    unknown_round_grades = sum(
        row[field]["status"] not in TASK_GRADES
        for trajectory in trajectories
        for row in trajectory["rounds"]
        for field in ("input_grade", "output_grade")
    )
    sensitivities = {
        rule: _exp2_accounting(trajectories, rule, n_boot, seed)
        for rule in ("parse_fail_as_accept", "complete_case", "terminal_unresolved_excluded")
    }
    estimates = [
        block["contrast"]["estimate"]
        for block in (
            primary,
            sensitivities["parse_fail_as_accept"],
            sensitivities["complete_case"],
        )
    ]
    return {
        "experiment": "math_exp2",
        "provenance": provenance,
        "cohort": "initial_status=incorrect; all randomized trajectories retained",
        "n_trajectories": len(trajectories),
        "complete": primary["complete"] and unknown_round_grades == 0,
        "n_missing_terminal_labels": sum(
            row["terminal_status"] not in TASK_GRADES for row in trajectories
        ),
        "n_terminal_unresolved": sum(
            row["terminal_status"] == "unresolved" for row in trajectories
        ),
        "n_unknown_round_grade_records": unknown_round_grades,
        "bootstrap": bootstrap,
        "primary": primary,
        "sensitivities": sensitivities,
        "robust_across_accountings": same_sign(estimates)
        if all(value is not None for value in estimates)
        else None,
        "implementation": "analysis.exp2_tfr.contrast/secondary/mechanism",
        "secondary": secondary,
        "mechanism": _mechanism(trajectories),
        "mechanism_label_basis": (
            "task_success: correct=True; incorrect/unresolved=False; E errors missing"
        ),
        "round_diagnostics": _round_diagnostics(trajectories),
        "unresolved_rates": _unresolved_rates(trajectories),
        "inference_note": (
            "Pilot estimates and uncertainty only; downstream diagnostics are not causal strata."
        ),
    }


def summarize_smoke(trajectories: list[dict]) -> dict:
    """M.10: one A1 arm, no effects, contrasts, paired tests or arm comparisons."""
    if not isinstance(trajectories, list) or len(trajectories) != 30:
        raise MathAnalysisError("smoke requires exactly 30 trajectories")
    provenance = _provenance(trajectories)
    _validate_trajectories(trajectories, smoke=True)
    counts = Counter(row["terminal_status"] for row in trajectories)
    n = len(trajectories)
    missing = sum(counts[status] for status in GRADES - TASK_GRADES)
    round_errors = sum(
        row[field]["status"] not in TASK_GRADES
        for trajectory in trajectories
        for row in trajectory["rounds"]
        for field in ("input_grade", "output_grade")
    )
    return {
        "experiment": "math_smoke",
        "stage": "SMOKE",
        "is_research_result": False,
        "arm": "A1",
        "provenance": provenance,
        "n_trajectories": n,
        "complete": missing == 0 and round_errors == 0,
        "terminal_status_counts": dict(sorted(counts.items())),
        "task_failure_rate": (counts["incorrect"] + counts["unresolved"]) / n
        if missing == 0
        else None,
        "non_convergence_rate": sum(
            row["termination_reason"] == "decision_limit" for row in trajectories
        )
        / n,
        "n_v_input_overlength": sum(
            row["termination_reason"] == OVERLENGTH for row in trajectories
        ),
        # 2026-10-16: the smoke reports the overlength rate (loop L_in: GLM-Z1-9B 4,096,
        # R1-Distill-Llama 1,344), and which V's L_in each ended trajectory exceeded
        "v_input_overlength_rate": sum(
            row["termination_reason"] == OVERLENGTH for row in trajectories
        )
        / n,
        "v_input_overlength_by_verifier": _overlength_by_verifier(trajectories),
        "round_diagnostics": _round_diagnostics(trajectories, arms=("A1",))["A1"],
        "unresolved_rates": _unresolved_rates(trajectories, arms=("A1",)),
        "interpretation": (
            "Engineering smoke only; observed controller behavior does not tune the chain."
        ),
    }
