"""Descriptive math diagnostics; never a substitute for the held-out research gate.

Verifier/channel diagnostics consume D_fit labels only. PRM error-scores are not
calibrated probabilities; Brier/ECE below are numeric descriptions, not evidence
that the scores estimate final-answer error probabilities.
"""

from __future__ import annotations

import math
import struct
from collections import Counter
from dataclasses import asdict
from itertools import combinations
from numbers import Real

import numpy as np
from analysis import drift as core_drift
from analysis import error_boundary as core_error_boundary
from analysis import exp1_did as core_exp1
from analysis import reliability as core_reliability
from analysis import transport as core_transport

from rafr.channel.degrade import FIXED_P, LADDER, DegradeChannel
from rafr.contracts.schemas import text_hash
from rafr.validity import cue_diagnostics as core_cues
from rafr.validity import gate as core_gate
from rafr.validity import metrics
from rafr.verifiers import diagnostics as core_diagnostics
from rafr.verifiers.probe_analysis import ECE_BINS

ARMS = ("A1", "B", "permuted_B")
LEVELS = ("low", "mid", "high")


class MathDiagnosticError(ValueError):
    pass


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MathDiagnosticError("scores/thresholds must be real numbers or explicit missing None")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise MathDiagnosticError("scores/thresholds must be finite and within [0,1]")
    return result


def _pair(value, *, missing=False):
    if value is None and missing:
        return (None, None)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise MathDiagnosticError("a score/threshold pair must have exactly two values")
    return tuple(None if v is None and missing else _number(v) for v in value)


def _inputs(scores, labels, taus):
    if len(scores) != len(labels):
        raise MathDiagnosticError("score and label rows must be aligned")
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise MathDiagnosticError("labels must be explicit integer 0/1; 1 means incorrect")
    return [_pair(row, missing=True) for row in scores], list(labels), _pair(taus)


def _finite(value):
    return float(value) if math.isfinite(value) else None


def _ratio(a, b):
    return a / b if b else None


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def cue_diagnostics(scores, labels, taus, *, high_margin_quantile=0.75) -> dict:
    """paper Sec. 4.3: adapt math's 1=INCORRECT labels to the unmodified Core five-cue table.

    Both populations are reported before effects: INCORRECT states are primary,
    and the whole split is secondary. Missing scores remain in coverage counts.
    """
    pairs, labels, taus = _inputs(scores, labels, taus)
    if (
        isinstance(high_margin_quantile, bool)
        or not isinstance(high_margin_quantile, Real)
        or not 0 <= high_margin_quantile <= 1
    ):
        raise MathDiagnosticError("high_margin_quantile must be in [0,1]")
    output = {
        "implementation": "rafr.validity.cue_diagnostics.compute",
        "positive_class": "INCORRECT",
        "is_gate_verdict": False,
        "high_margin_quantile": high_margin_quantile,
    }
    for name, condition in (
        ("primary_incorrect", lambda y: y == 1),
        ("secondary_all", lambda y: True),
    ):
        wanted = [i for i, label in enumerate(labels) if condition(label)]
        selected = [i for i in wanted if None not in pairs[i]]
        report = {
            "status": "DESCRIPTIVE_ONLY" if selected else "NOT_COMPUTED",
            "n_input": len(wanted),
            "n_scored": len(selected),
            "n_missing_pairs": len(wanted) - len(selected),
        }
        if selected:
            report["cues"] = _finite_json(
                asdict(
                    core_cues.compute(
                        core_cues.CueInputs(
                            s1_obs=[pairs[i][0] for i in selected],
                            s2_obs=[pairs[i][1] for i in selected],
                            tau1=taus[0],
                            tau2=taus[1],
                            y_true=[labels[i] for i in selected],
                        ),
                        high_margin_quantile=float(high_margin_quantile),
                    )
                )
            )
        output[name] = report
    return output


def verifier_diagnostics(scores, labels, taus, *, population="D_fit") -> dict:
    pairs, labels, taus = _inputs(scores, labels, taus)
    n = len(labels)
    output = {
        "status": "DESCRIPTIVE_ONLY" if n else "NOT_COMPUTED",
        "population": population,
        "positive_class": "initial_answer_incorrect",
        "n_input": n,
        "n_positive": sum(labels),
        "n_negative": n - sum(labels),
        "score_is_calibrated_probability": False,
        "calibration_interpretation": (
            "Core Brier/ECE treat uncalibrated error-scores numerically as probabilities for "
            "description only; this does not establish probability calibration."
        ),
        "pr_auc_definition": "rafr.validity.metrics.pr_auc (Core average precision)",
        "ece_bins": ECE_BINS,
        "implementation": "rafr.verifiers.diagnostics.single/pair",
        "zero_division": "undefined precision/TPR/FPR are null; undefined F1/F2 use zero",
        "per_verifier": {},
    }
    for index, name in enumerate(("v1", "v2")):
        valid = [i for i, pair in enumerate(pairs) if pair[index] is not None]
        values = [pairs[i][index] for i in valid]
        truth = [labels[i] for i in valid]
        report = {
            "status": "DESCRIPTIVE_ONLY" if valid else "NOT_COMPUTED",
            "n_scored": len(valid),
            "n_missing": n - len(valid),
            "score_coverage": _ratio(len(valid), n),
            "threshold": taus[index],
            "n_positive_scored": sum(truth),
            "n_negative_scored": len(truth) - sum(truth),
            "unique_scores": len(set(values)),
            "tied_scores": len(values) != len(set(values)),
        }
        if values:
            predictions = [int(value >= taus[index]) for value in values]
            tp, fp, fn, tn = metrics.confusion(truth, predictions)
            core = _finite_json(
                core_diagnostics.single(
                    np.asarray(values), np.asarray(truth), taus[index], sum(labels) / n
                )
            )
            report.update(
                confusion={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                precision=_ratio(tp, tp + fp),
                tpr=core["tpr"],
                fpr=core["fpr"],
                fnr=core["fnr_unsafe"],
                accuracy=core["accuracy"],
                error_rate=1 - core["accuracy"],
                f1=core["f1"],
                f2=core["f2"],
                pr_auc=core["pr_auc"],
                auroc=core["auroc"],
                brier=core["brier"],
                ece=core["ece"],
                minimum=min(values),
                maximum=max(values),
                mean=float(np.mean(values)),
                variance=float(np.var(values)),
                variance_ddof=0,
            )
        else:
            report.update(
                {
                    key: None
                    for key in (
                        "confusion",
                        "precision",
                        "tpr",
                        "fpr",
                        "fnr",
                        "accuracy",
                        "error_rate",
                        "f1",
                        "f2",
                        "pr_auc",
                        "auroc",
                        "brier",
                        "ece",
                        "minimum",
                        "maximum",
                        "mean",
                        "variance",
                    )
                }
            )
            report["reason"] = "no_observed_scores"
        output["per_verifier"][name] = report
    complete = [i for i, pair in enumerate(pairs) if None not in pair]
    output["n_pair_complete"] = len(complete)
    joint = {
        "status": "DESCRIPTIVE_ONLY" if complete else "NOT_COMPUTED",
        "n": len(complete),
        "n_missing_pairs": n - len(complete),
    }
    if complete:
        y = [labels[i] for i in complete]
        a = [int(pairs[i][0] >= taus[0]) for i in complete]
        b = [int(pairs[i][1] >= taus[1]) for i in complete]
        core = _finite_json(
            core_diagnostics.pair(
                np.asarray([pairs[i][0] for i in complete]),
                taus[0],
                np.asarray([pairs[i][1] for i in complete]),
                taus[1],
                np.asarray(y),
            )
        )
        joint.update(
            n_positive=sum(y),
            n_negative=len(y) - sum(y),
            disagreement_rate=core["disagreement"],
            joint_miss_rate_on_errors=core["both_miss_unsafe"],
            joint_false_positive_rate_on_correct=_ratio(
                sum(t == 0 and pa == pb == 1 for t, pa, pb in zip(y, a, b, strict=True)),
                len(y) - sum(y),
            ),
            both_wrong_rate=core["p_both_wrong"],
            both_correct_rate=core["p_both_right"],
            v1_only_wrong_rate=core["p_v1_only"],
            v2_only_wrong_rate=core["p_v2_only"],
            v1_only_correct_rate=core["p_v2_only"],
            v2_only_correct_rate=core["p_v1_only"],
            error_indicator_phi=core["phi_errors"],
            error_indicator_kappa=core["kappa_errors"],
        )
    else:
        joint["reason"] = "no_complete_verifier_pairs"
    output["complementarity"] = joint
    output["cue_diagnostics"] = cue_diagnostics(scores, labels, taus)
    output["pr_auc_tie_note"] = (
        "Core's stable ordering of equal scores is reused; tied-score AP may depend on row order."
    )
    return output


def channel_diagnostics(channel: DegradeChannel, scores, labels, ids, delta_r) -> dict:
    from types import SimpleNamespace

    from pipelines.step08_tune_channel import reachability

    pairs, labels, taus = _inputs(scores, labels, [channel.tau1, channel.tau2])
    if (
        len(ids) != len(labels)
        or len(set(ids)) != len(ids)
        or any(not isinstance(qid, str) or not qid for qid in ids)
    ):
        raise MathDiagnosticError("channel diagnostic IDs must be unique and aligned")
    delta_r = _number(delta_r)
    if delta_r <= 0:
        raise MathDiagnosticError("delta_r must be positive")
    for level in LADDER:
        if any(channel.levels[level].p(v) != FIXED_P[level] for v in ("v1", "v2")):
            raise MathDiagnosticError("diagnostic requires the fixed p=0/0.5/1 schedule")
    result = {
        "status": "DESCRIPTIVE_ONLY",
        "population": "D_fit_initial_answer_incorrect",
        "n_input": len(labels),
        "n_positive": sum(labels),
        "delta_r": delta_r,
        "p_by_level": {level.value: FIXED_P[level] for level in LADDER},
        "is_gate_verdict": False,
        "parameter_tuning_performed": False,
        "interpretation": (
            "Expected reachability and D_fit realization only; held-out gate remains required."
        ),
        "per_verifier": {},
        "cue_by_level": {},
    }
    for level in LADDER:
        complete = [i for i, pair in enumerate(pairs) if None not in pair]
        result["cue_by_level"][level.value] = cue_diagnostics(
            [channel(ids[i], *pairs[i], level) for i in complete],
            [labels[i] for i in complete],
            taus,
        )
    if not sum(labels) or any(None in pair for pair in pairs):
        result.update(
            status="NOT_COMPUTED",
            reason=(
                "no_explicitly_incorrect_D_fit_state"
                if not sum(labels)
                else "missing_verifier_scores"
            ),
        )
        return result
    positive = [i for i, label in enumerate(labels) if label == 1]
    observed = {
        level: [channel(qid, *pair, level) for qid, pair in zip(ids, pairs, strict=True)]
        for level in LADDER
    }
    # Reuse the Core step-8 reachability calculation, including its whole-split
    # accuracy/BA baseline, fully resampled floor and span (paper Sec. 4.3).
    core_reach = reachability(
        channel.q,
        [SimpleNamespace(state_id=qid) for qid in ids],
        {qid: pair[0] for qid, pair in zip(ids, pairs, strict=True)},
        {qid: pair[1] for qid, pair in zip(ids, pairs, strict=True)},
        dict(zip(ids, labels, strict=True)),
        *taus,
        delta_r,
    )
    for index, name in enumerate(("v1", "v2")):
        base = core_reach[name]["baseline_tpr"]
        floor = core_reach[name]["all_resampled_floor"]
        expected = {
            level.value: (1 - FIXED_P[level]) * base + FIXED_P[level] * floor for level in LADDER
        }
        actual = {
            level.value: sum(observed[level][i][index] >= taus[index] for i in positive)
            / len(positive)
            for level in LADDER
        }
        gap = (base - floor) / 2
        whole = dict(core_reach[name]["descriptive_whole_split"])
        whole["incorrect_share"] = whole.pop("e_unsafe_share")
        whole["note"] = (
            "II.5.2 descriptive only; the gate uses detection rates on INCORRECT states."
        )
        result["per_verifier"][name] = {
            "base_tpr": base,
            "resampling_floor": floor,
            "q_pool_n": channel.q[name].n,
            "expected_ladder": expected,
            "expected_adjacent_gaps": {"low-mid": gap, "mid-high": gap},
            "expected_target_shortfall": max(0.0, delta_r - gap),
            "actual_D_fit_ladder": actual,
            "actual_D_fit_adjacent_gaps": {
                "low-mid": actual["low"] - actual["mid"],
                "mid-high": actual["mid"] - actual["high"],
            },
            "whole_split_reachability": _finite_json(whole),
        }
    return result


def channel_descriptive_diagnostics(labels, observed, taus) -> dict:
    """Core's companion tables, separate from the detection-rate gate verdict.

    ``observed`` contains every state of the split. Only the correlation and
    probability of at least one missed error use its INCORRECT subset, matching
    Core step 8/9; whole-split accuracy must keep CORRECT states too.
    """
    taus = _pair(taus)
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise MathDiagnosticError("channel labels must be explicit 0/1; 1 means INCORRECT")
    vectors = core_gate.detection_vectors(labels, observed, *taus)
    return _finite_json(
        {
            "descriptive": core_gate.descriptive_accuracy(labels, observed, *taus),
            "p_at_least_one_wrong": {
                level: 1.0
                - sum(
                    a * b for a, b in zip(vectors["v1"][level], vectors["v2"][level], strict=True)
                )
                / len(vectors["v1"][level])
                for level in LADDER
            },
            "paired_correctness_correlation": core_gate.paired_correctness_correlation(vectors),
        }
    )


def _bits(value):
    pair = _pair(value)
    return b"".join(struct.pack("!d", number) for number in pair)


def _prompt_hash(row):
    attempts = row.get("attempts")
    if not isinstance(attempts, list) or not attempts or not isinstance(attempts[0], dict):
        return None
    value = attempts[0].get("prompt_hash")
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        return None
    return value


def _equality(index, ids, arm, first, second, field):
    equal, compared = 0, 0
    for qid in ids:
        a, b = index.get((qid, arm, first)), index.get((qid, arm, second))
        if a is None or b is None:
            continue
        if field == "prompt":
            x, y = _prompt_hash(a), _prompt_hash(b)
        else:
            x = _bits(a["observed_scores"]) if a.get("observed_scores") is not None else None
            y = _bits(b["observed_scores"]) if b.get("observed_scores") is not None else None
        if x is None or y is None:
            continue
        equal += x == y
        compared += 1
    return {
        "status": "NOT_COMPUTED"
        if not compared
        else "PARTIAL"
        if compared < len(ids)
        else "DESCRIPTIVE_ONLY",
        "n_questions": len(ids),
        "n_compared": compared,
        "n_missing": len(ids) - compared,
        "n_equal": equal,
        "equal_fraction": _ratio(equal, compared),
    }


def _stratum_summary(groups):
    output = {}
    for level in LEVELS:
        output[level] = {}
        for key, cells in sorted(groups[level].items()):
            per_arm = {}
            for arm in ARMS:
                rows = [cell[arm] for cell in cells]
                accepts = sum(not row["parse_fail"] and row["action"] == "accept" for row in rows)
                per_arm[arm] = {
                    "n": len(rows),
                    "n_accept": accepts,
                    "n_parse_fail": sum(row["parse_fail"] for row in rows),
                    "FA": _ratio(accepts, len(rows)),
                }
            output[level][key] = {
                "n_questions": len(cells),
                "per_arm": per_arm,
                "delta_FA_A1_minus_B": per_arm["A1"]["FA"] - per_arm["B"]["FA"],
                "delta_FA_permuted_B_minus_B": per_arm["permuted_B"]["FA"] - per_arm["B"]["FA"],
            }
    return output


def _h2(index, ids):
    mode_groups = {level: {} for level in LEVELS}
    agreement_groups = {level: {} for level in LEVELS}
    missing = Counter()
    core_rows, channel_rows = [], []
    for qid in ids:
        for level in LEVELS:
            cells = {arm: index.get((qid, arm, level)) for arm in ARMS}
            if any(row is None for row in cells.values()):
                missing["incomplete_three_arm_cell"] += 1
                continue
            if any(row.get("initial_status") != "incorrect" for row in cells.values()):
                missing["initial_incorrect_status_not_established"] += 1
                continue
            if any(
                type(row.get("parse_fail")) is not bool
                or (
                    row.get("action") is not None
                    if row.get("parse_fail")
                    else row.get("action") not in {"accept", "rewrite"}
                )
                for row in cells.values()
            ):
                missing["decision_accounting_unavailable"] += 1
                continue
            observed, flags = [], []
            for row in cells.values():
                if row.get("observed_scores") is None or row.get("thresholds") is None:
                    observed.append(None)
                else:
                    scores, taus = _pair(row["observed_scores"]), _pair(row["thresholds"])
                    observed.append(
                        (
                            _bits(scores),
                            _bits(taus),
                            (scores[0] >= taus[0]) == (scores[1] >= taus[1]),
                        )
                    )
                assignment = row.get("channel_assignment", {})
                pair = (assignment.get("perturbed_v1"), assignment.get("perturbed_v2"))
                flags.append(pair if all(type(flag) is bool for flag in pair) else None)
            if None in observed:
                missing["agreement_evidence_unavailable"] += 1
            else:
                if any(value != observed[0] for value in observed):
                    raise MathDiagnosticError(
                        "H2 strata require identical own-state evidence across arms"
                    )
                agreement = "agree" if observed[0][2] else "disagree"
                agreement_groups[level].setdefault(agreement, []).append(cells)
            if None in flags:
                missing["channel_flags_unavailable"] += 1
            else:
                if any(value != flags[0] for value in flags):
                    raise MathDiagnosticError("H2 strata require arm-independent channel flags")
                a, b = flags[0]
                mode = "both" if a and b else "v1_only" if a else "v2_only" if b else "none"
                mode_groups[level].setdefault(mode, []).append(cells)
                if None not in observed:
                    channel_rows.append(
                        {"state_id": qid, "lam": level, "perturbed_v1": a, "perturbed_v2": b}
                    )
                    for row in cells.values():
                        scores, taus = _pair(row["observed_scores"]), _pair(row["thresholds"])
                        core_rows.append(
                            {
                                "state_id": qid,
                                "lam": level,
                                "e_label": "UNSAFE",
                                "arm": row["arm"],
                                "action": row["action"],
                                "parse_fail": row["parse_fail"],
                                "s1_tilde": scores[0],
                                "s2_tilde": scores[1],
                                "tau1": taus[0],
                                "tau2": taus[1],
                            }
                        )
    n_mode = sum(len(cells) for groups in mode_groups.values() for cells in groups.values())
    n_agreement = sum(
        len(cells) for groups in agreement_groups.values() for cells in groups.values()
    )
    return {
        "status": "NOT_COMPUTED"
        if not n_mode and not n_agreement
        else "PARTIAL"
        if missing
        else "DESCRIPTIVE_ONLY",
        "definition": (
            "Own state's shared pre-controller evidence; never donor fields "
            "or controller action strata."
        ),
        "parse_failure_accounting": "Retained in denominator; main FA counts no acceptance.",
        "n_expected_question_levels": len(ids) * 3,
        "n_channel_question_levels": n_mode,
        "n_agreement_question_levels": n_agreement,
        "missing_evidence_counts": dict(missing),
        "by_channel_mode": _stratum_summary(mode_groups),
        "by_observed_agreement": _stratum_summary(agreement_groups),
        "preregistered": _finite_json(core_exp1.h2_strata(core_rows, channel_rows)),
        "four_modes_note": "Additional descriptive detail; the preregistered strata are binary.",
        "is_gate_verdict": False,
    }


def exp1_diagnostics(rows) -> dict:
    index = {}
    for row in rows:
        qid, arm, level = row.get("question_id"), row.get("arm"), row.get("level")
        if not isinstance(qid, str) or not qid or arm not in ARMS or level not in LEVELS:
            return {"status": "NOT_COMPUTED", "reason": "missing_or_unknown_question_arm_level"}
        key = qid, arm, level
        if key in index:
            raise MathDiagnosticError("duplicate Exp1 question/arm/level")
        index[key] = row
    if not index:
        return {"status": "NOT_COMPUTED", "reason": "no_Exp1_records"}
    ids = sorted({key[0] for key in index})
    prompt = {
        arm: {
            f"{a}-{b}": _equality(index, ids, arm, a, b, "prompt")
            for a, b in combinations(LEVELS, 2)
        }
        for arm in ARMS
    }
    evidence = {
        arm: {
            f"{a}-{b}": _equality(index, ids, arm, a, b, "scores")
            for a, b in combinations(LEVELS, 2)
        }
        for arm in ARMS
    }
    h2 = _h2(index, ids)
    statuses = [h2["status"]] + [
        item["status"]
        for report in (prompt, evidence)
        for levels in report.values()
        for item in levels.values()
    ]
    status = (
        "NOT_COMPUTED"
        if set(statuses) == {"NOT_COMPUTED"}
        else "DESCRIPTIVE_ONLY"
        if set(statuses) == {"DESCRIPTIVE_ONLY"}
        else "PARTIAL"
    )
    return {
        "status": status,
        "n_questions": len(ids),
        "n_rows": len(rows),
        "is_gate_verdict": False,
        "actual_first_prompt_hash_equality": prompt,
        "observed_score_float64_bitwise_equality": evidence,
        "H2": h2,
        "interpretation": (
            "Diagnostics only; no significance claim and no pass/fail based on model behavior."
        ),
    }


def exp2_reliability_diagnostics(trajectories) -> dict:
    """Current verifier scores against independent current-input E, descriptively.

    Active populations depend on previous treatment decisions. These tables must
    never become new causal strata or replace the randomized arm comparison.
    Unknown E labels remain in coverage counts; they are not scored as negatives.
    """
    groups = {arm: {t: [] for t in range(1, 5)} for arm in ("A1", "B")}
    seen = set()
    core_rows, truth = [], {}
    for trajectory in trajectories:
        qid, arm = trajectory.get("question_id"), trajectory.get("arm")
        if not isinstance(qid, str) or not qid or arm not in groups:
            raise MathDiagnosticError(
                "Exp2 diagnostics require question identity and randomized arm"
            )
        if qid in seen:
            raise MathDiagnosticError("duplicate Exp2 trajectory question")
        seen.add(qid)
        rounds = trajectory.get("rounds")
        if not isinstance(rounds, list):
            raise MathDiagnosticError("Exp2 trajectory needs recorded rounds")
        round_ids = set()
        for row in rounds:
            t = row.get("t")
            if type(t) is not int or t not in range(1, 5) or t in round_ids:
                raise MathDiagnosticError("Exp2 diagnostic rounds must be unique in 1..4")
            if row.get("arm", arm) != arm:
                raise MathDiagnosticError("round arm differs from randomized trajectory")
            round_ids.add(t)
            groups[arm][t].append(row)
            scores = _pair(row.get("input_scores"), missing=True)
            thresholds = row.get("thresholds")
            grade = row.get("input_grade", {}).get("status")
            if thresholds is not None and None not in scores:
                taus = _pair(thresholds)
                # Core's text hash is only a label-join key here. Scope it to the
                # mathematical question: identical answer text can have different truth.
                key = repr((qid, t))
                if grade in {"correct", "incorrect", "unresolved"}:
                    truth[text_hash(key)] = "SAFE" if grade == "correct" else "UNSAFE"
                core_rows.append(
                    {
                        "question_id": qid,
                        "arm": arm,
                        "t": t,
                        "x_t": key,
                        "s1": scores[0],
                        "s2": scores[1],
                        "tau1": taus[0],
                        "tau2": taus[1],
                        "math_status": grade,
                    }
                )
    core_reliability.fill_c_t(core_rows, truth)
    output = {
        "status": "DESCRIPTIVE_ONLY" if trajectories else "NOT_COMPUTED",
        "population": "active Exp2 decisions, by randomized arm and decision number",
        "interpretation": (
            "Treatment-downstream description only; do not re-stratify causal comparisons "
            "using observed reliability or remaining active states."
        ),
        "n_trajectories": len(trajectories),
        "by_arm_round": {},
        "R_t": {
            arm: {str(t): cell for t, cell in rounds.items()}
            for arm, rounds in core_reliability.r_t_by_arm(core_rows).items()
        },
        "R_t_basis": (
            "Core per-verifier task-success accuracy; correct=True, incorrect/unresolved=False. "
            "Only E backend errors or missing scores are excluded; "
            "unresolved is separately counted."
        ),
        "c_t_records": [
            {key: row.get(key) for key in ("question_id", "arm", "t", "math_status", "c_t")}
            for row in core_rows
        ],
        "implementation": "analysis.reliability.fill_c_t/r_t_by_arm",
    }
    for arm, rounds in groups.items():
        output["by_arm_round"][arm] = {}
        for t, rows in rounds.items():
            counts = Counter(
                row.get("input_grade", {}).get("status", "missing")
                if isinstance(row.get("input_grade"), dict)
                else "missing"
                for row in rows
            )
            known = [
                row
                for row in rows
                if isinstance(row.get("input_grade"), dict)
                and row["input_grade"].get("status") in {"correct", "incorrect"}
            ]
            report = {
                "status": "DESCRIPTIVE_ONLY",
                "n_active": len(rows),
                "input_grade_status_counts": dict(counts),
                "n_known_input_grade": len(known),
                "n_unknown_input_grade": len(rows) - len(known),
                "input_grade_coverage": _ratio(len(known), len(rows)),
                "classification_population": "known independent input-grade labels only",
                "per_verifier": {},
            }
            if not rows:
                report.update(status="NOT_COMPUTED", reason="no_active_decisions")
            elif any(row.get("thresholds") is None for row in rows):
                report.update(status="NOT_COMPUTED", reason="missing_thresholds")
            else:
                thresholds = [_pair(row["thresholds"]) for row in rows]
                if any(pair != thresholds[0] for pair in thresholds):
                    raise MathDiagnosticError("Exp2 diagnostic requires shared fixed thresholds")
                values = verifier_diagnostics(
                    [row.get("input_scores") for row in known],
                    [int(row["input_grade"]["status"] == "incorrect") for row in known],
                    thresholds[0],
                )
                for i, name in enumerate(("v1", "v2")):
                    selected = values["per_verifier"][name]
                    report["per_verifier"][name] = {
                        key: selected[key]
                        for key in (
                            "status",
                            "n_scored",
                            "n_missing",
                            "n_positive_scored",
                            "n_negative_scored",
                            "tpr",
                            "fpr",
                            "f1",
                            "accuracy",
                            "confusion",
                        )
                    }
                    all_scores = [_pair(row.get("input_scores"), missing=True)[i] for row in rows]
                    report["per_verifier"][name]["n_active_with_score"] = sum(
                        score is not None for score in all_scores
                    )
                if len(known) < len(rows):
                    report["status"] = "PARTIAL" if known else "NOT_COMPUTED"
                    report["reason"] = "unknown_input_grades_retained_in_coverage"
            report["classification_population"] = (
                "correct/incorrect reasoning grades only; unresolved excluded from this auxiliary "
                "classifier table but retained as task failure in the primary R_t table"
            )
            output["by_arm_round"][arm][str(t)] = report
    return output


def _outcome_core_rows(exp1_rows, trajectories):
    """Domain boundary for existing Core report algorithms, with no text-key collision."""
    import json

    first = []
    for row in exp1_rows:
        if row.get("initial_status") != "incorrect":
            raise MathDiagnosticError("Exp1 diagnostic population must be initial INCORRECT")
        scores = _pair(row.get("input_scores", row.get("raw_scores")))
        observed, taus = _pair(row["observed_scores"]), _pair(row["thresholds"])
        first.append(
            {
                "state_id": row["question_id"],
                "arm": row["arm"],
                "lam": row["level"],
                "action": row["action"],
                "parse_fail": row["parse_fail"],
                "s1": scores[0],
                "s2": scores[1],
                "s1_tilde": observed[0],
                "s2_tilde": observed[1],
                "tau1": taus[0],
                "tau2": taus[1],
                "e_label": "UNSAFE",
                "math_status": "incorrect",
            }
        )
    second, truth = [], {}
    for trajectory in trajectories:
        for row in trajectory["rounds"]:
            # Core only uses x_t as a label-join key. In math, identical solutions
            # can answer different questions, so raw solution hashes are unsafe.
            key = json.dumps([trajectory["question_id"], row["t"]])
            scores, taus = _pair(row["input_scores"]), _pair(row["thresholds"])
            status = row["input_grade"]["status"]
            if status in {"correct", "incorrect", "unresolved"}:
                truth[text_hash(key)] = "SAFE" if status == "correct" else "UNSAFE"
            elif status not in {"timeout", "backend_error"}:
                raise MathDiagnosticError("unknown input E status in outcome diagnostics")
            second.append(
                {
                    "question_id": trajectory["question_id"],
                    "arm": trajectory["arm"],
                    "t": row["t"],
                    "x_t": key,
                    "action": row["action"],
                    "parse_fail": row["parse_fail"],
                    "s1": scores[0],
                    "s2": scores[1],
                    # The design's Exp2 channel is identity.
                    "s1_tilde": scores[0],
                    "s2_tilde": scores[1],
                    "tau1": taus[0],
                    "tau2": taus[1],
                    "math_status": status,
                }
            )
    return first, second, truth


def _status_coverage(rows):
    counts = Counter(row["math_status"] for row in rows)
    return {
        "n_decisions": len(rows),
        "input_grade_status_counts": dict(sorted(counts.items())),
        "n_input_unresolved": counts["unresolved"],
        "n_missing_E": counts["timeout"] + counts["backend_error"],
        "n_parse_fail": sum(row["parse_fail"] for row in rows),
    }


def protocol_outcome_diagnostics(exp1_rows, trajectories, gate_r, *, n_boot=10_000, seed=0):
    """paper App. C.1, D.1 / paper Sec. 6.2 tables via the existing Core analysis algorithms.

    This reporting adapter never deletes a condition, changes a gate verdict or
    invents a drift threshold. UNRESOLVED is task failure; evaluator errors stay
    missing. Original math statuses and JSON parse failures are counted separately
    so task-failure proportions cannot be mistaken for resolved-answer error rates.
    """
    first, second, truth = _outcome_core_rows(exp1_rows, trajectories)
    exp1_only = _exp1_outcome(first, gate_r, n_boot=n_boot, seed=seed)
    transport = core_transport.compare(first, second, truth)
    transport["task_failure"] = transport.pop("unsafe")
    for cell in transport["by_round_exp2"].values():
        cell["task_failure"] = cell.pop("unsafe")
    transport["note"] = (
        "Exp2 t=1 uses D_Exp1 itself; t>=2 is the informative comparison. The Exp1 "
        "initial answers are INCORRECT by construction. Scores and margins are raw "
        "pre-channel values. Task failure means incorrect or unresolved; E errors are "
        "excluded from the label proportion only, never treated as correct. JSON "
        "parse_fail is a distinct controller event. Distribution gaps are descriptive."
    )
    transport["grade_coverage"] = {
        "exp1_unique_states": {
            "n_states": len({row["state_id"] for row in first}),
            "input_grade_status": "incorrect",
        },
        "exp2_decision_points": _status_coverage(second),
        "exp2_t_ge_2": _status_coverage([row for row in second if row["t"] >= 2]),
        "by_arm_round": {
            arm: {
                str(t): _status_coverage(
                    [row for row in second if row["arm"] == arm and row["t"] == t]
                )
                for t in range(1, 5)
            }
            for arm in ("A1", "B")
        },
    }
    boundary = {
        "exp1": exp1_only["error_boundary_alignment"]["exp1"],
        "exp2": core_error_boundary.exp2(second, truth),
        "label_basis": (
            "correct=task success; incorrect/unresolved=task failure; E errors missing. "
            "Among decisions where at least one observed verifier disagrees with E, "
            "report controller decision-error rates. Descriptive, not causal strata."
        ),
    }
    for experiment in ("exp1", "exp2"):
        boundary[experiment]["definition"] = boundary["label_basis"]
    return _finite_json(
        {
            "exp1_drift": exp1_only["exp1_drift"],
            "transport": transport,
            "error_boundary_alignment": boundary,
            "result_shape": exp1_only["result_shape"],
        }
    )


def _exp1_outcome(first, gate_r, *, n_boot, seed) -> dict:
    ladder = core_drift.exp1_ladder(first, n_boot=n_boot, seed=seed)
    ladder["population"] = "D_Exp1 (every initial answer E-INCORRECT)"
    return {
        "exp1_drift": {
            "ladder": ladder,
            "vs_gate": core_drift.compare_to_gate(ladder, gate_r),
            "interpretation": (
                "Descriptive transport diagnostic; no automatic drift threshold or "
                "condition removal. All preregistered primary conditions remain."
            ),
        },
        "error_boundary_alignment": {"exp1": core_error_boundary.exp1(first)},
        "result_shape": core_exp1.result_shape(first),
    }


def exp1_outcome_diagnostics(exp1_rows, gate_r, *, n_boot=10_000, seed=0):
    """The Exp 1 part of :func:`protocol_outcome_diagnostics` (paper App. C.1, D.1 drift against the
    gate, paper Sec. 6.2 error boundary, result shape), for Exp 1's report before Exp 2 exists; the
    same
    algorithms and the same seed, so the numbers equal the full report's."""
    first, _, _ = _outcome_core_rows(exp1_rows, [])
    out = _exp1_outcome(first, gate_r, n_boot=n_boot, seed=seed)
    out["error_boundary_alignment"]["exp1"]["definition"] = (
        "correct=task success; incorrect/unresolved=task failure; E errors missing. "
        "Among decisions where at least one observed verifier disagrees with E, "
        "report controller decision-error rates. Descriptive, not causal strata."
    )
    return _finite_json(out)
