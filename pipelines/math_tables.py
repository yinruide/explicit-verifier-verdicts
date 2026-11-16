"""Math's primary analysis tables in detox's format: one JSON per table, ``tables.md``,
``sections.json`` and ``analysis_full.json`` (the files ``python -m analysis.prereg_tables``
writes for detox, step 13).

    python pipelines/math_tables.py [--run-dir runs/math_formal_n9000] [--out-dir DIR]

detox computes its tables from its Exp 1 / Exp 2 logs. Math's primary analyses are
computed once, by ``MathPipeline.report`` (``pipelines/math_analysis.py``,
``pipelines/math_diagnostics.py``; section 3), into the run's
``report.json``. This module only lays that report out as detox's tables: it computes no
estimate, interval or diagnostic of its own. The section map follows the
analysis plan (the Core plan, which paper Sec. 5, App. D instantiates), so a math table carries the
detox table's id where the analysis is the same; math-only tables (paper App. E, D.1) are added, and
a detox table math has no counterpart for says why (no panel, paper Sec. 5; no semantic
preservation, paper App. D.1).

Before laying out the report, the run's frozen markers are checked (Exp 1's ``COMPLETE.json``,
Exp 2's ``judge_complete.json``) and the report's counts are compared with the files they came
from; a report whose inputs changed since is refused. ``fake=True`` requires the FAKE marker of
step 10's report and writes ``FAKE_*`` files; the formal run refuses a FAKE report. Output:
``<run>/analysis/`` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.prereg_tables import PILOT_NOTE, SectionSpec, Table, _cell  # noqa: E402

from rafr.core import paths  # noqa: E402

RUN = paths.RUNS / "math_formal_n9000"
FAKE_PREFIX = "FAKE_"
ARMS1 = ("A1", "B", "permuted_B")
ARMS2 = ("A1", "B")
LEVELS = ("low", "mid", "high")
PAIRS = (("low", "mid"), ("low", "high"), ("mid", "high"))
CELLS = {"00": "(0,0)", "01": "(0,1)", "10": "(1,0)", "11": "(1,1)"}
EXP1_RULES = ("main", "parse_fail_as_accept", "complete_case")
EXP2_RULES = ("main", "parse_fail_as_accept", "complete_case", "terminal_unresolved_excluded")
TAGS = ("math-tau-frozen", "math-channel-frozen", "math-analysis-preregistered")


class MathTablesError(RuntimeError):
    pass


# ---- the map ---------------------------------------------------------------------------

MA = "pipelines.math_analysis"
MD = "pipelines.math_diagnostics"

SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec("1", "hypotheses and sign convention (I.2)", (f"{MA}.summarize_exp1",),
                ("T01_h1_direction",)),
    SectionSpec("2", "the pilot's standing (III.1.1)", (), (),
                ("no computation: every table carries the pilot note; no test, no p-value",)),
    SectionSpec(
        "3",
        "construction of the analysis objects (II.2.1-II.2.4, II.3; M.2)",
        (f"{MA}.summarize_exp1", f"{MD}.exp1_diagnostics",
         "pipelines.math_pipeline.MathPipeline.record_exp1_outcome"),
        ("T03_exp1_construction",),
        ("the construction itself (G_fixed first answers, E's y*, the splits) is steps 4-5, done "
         "and frozen (docs/math/step04_05_record.md); this checks what the Exp 1 report and its "
         "markers show of it",),
    ),
    SectionSpec(
        "4",
        "metric definitions (II.9.3; M.9)",
        (f"{MA}.summarize_exp1", f"{MA}.summarize_exp2"),
        ("T04_metrics",),
        ("semantic preservation is detox's (II.9.3); math's secondary metric in its place is the "
         "rewrite transition count (M.9, T08_rewrite_transitions)",),
    ),
    SectionSpec("5", "E and y* (II.3, II.9.1; M.9: E = Math-Verify)",
                ("pipelines.math_pipeline.MathPipeline.judge",), ("T05_e_identity",)),
    SectionSpec(
        "6",
        "Exp 1 primary analysis (III.1.2, II.8.2 / C.24)",
        (f"{MA}.summarize_exp1", f"{MD}.exp1_diagnostics"),
        ("T06_exp1_effective_n", "T06_exp1_cells", "T06_exp1_did", "T06_exp1_delta_by_lambda",
         "T06_exp1_delta_by_lambda_ci", "T06_prompt_identity"),
    ),
    SectionSpec("6.1", "point estimates and intervals (III.1.2)",
                (f"{MA}._paired_contrast", f"{MA}._exp2_accounting"), ("T061_bootstrap",)),
    SectionSpec("6.2", "H2 strata (III.1.2); r_hat_ref alongside (II.1.1)",
                (f"{MD}.exp1_diagnostics",), ("T062_h2_strata", "T062_r_ref")),
    SectionSpec("6.3", "evidence sensitivity (III.1.2)",
                (f"{MA}._evidence_cells", f"{MA}._round_diagnostics"),
                ("T063_evidence_exp1", "T063_evidence_exp2")),
    SectionSpec(
        "7",
        "Exp 2 start, loop and final output (II.8.3)",
        ("pipelines.math_pipeline.MathPipeline.install_loop",
         "pipelines.math_pipeline.MathPipeline.judge"),
        ("T07_exp2_construction",),
    ),
    SectionSpec(
        "8",
        "Exp 2 analyses (III.1.2; M.8, M.9)",
        (f"{MA}.summarize_exp2", f"{MA}._mechanism", f"{MD}.exp2_reliability_diagnostics"),
        ("T08_tfr_by_arm", "T08_tfr_contrast", "T08_secondary", "T08_rewrite_rounds",
         "T08_decision_rounds", "T08_r_t", "T08_mechanism", "T08_rewrite_transitions",
         "T08_overlength"),
        ("Rep x R_t interaction (exploratory mechanism only): the plan gives no estimator, and "
         "R_t is post-treatment (a collider); defining one now would add analysis the "
         "preregistration does not contain",),
    ),
    SectionSpec("8.1", "panel sensitivity (III.1.2, Exp 2 only)", (), (),
                ("math has no panel (M.3): E is code, so the arm-correlated bias of an LLM E that "
                 "the panel guards against does not arise; the arm-related threat in math, format "
                 "failure, is M.8's sensitivity (c) (T09_robustness)",)),
    SectionSpec("8.2", "semantic preservation, descriptive", (), (),
                ("detox only (II.9.3); math reports rewrite transition counts instead "
                 "(M.9, T08_rewrite_transitions)",)),
    SectionSpec(
        "9",
        "parse_fail accounting and sensitivities (II.8.2; M.8: unresolved, v_input_overlength)",
        (f"{MA}._exp1_accounting", f"{MA}._exp2_accounting", f"{MA}._unresolved_rates"),
        ("T09_parse_fail_exp1", "T09_parse_fail_exp2", "T09_unresolved_exp2", "T09_robustness"),
        ("refusal_like sensitivity not done (M.8: a refusal goes through the provider signal "
         "and yields no answer, covered by (c))",),
    ),
    SectionSpec("10", "drift on Exp 1 (II.6.3)", (f"{MD}.exp1_outcome_diagnostics",),
                ("T10_drift",), ("no 'clear drift' verdict: II.6.3 names no threshold",)),
    SectionSpec("11", "transport diagnostic (I.5.4, II.10)",
                (f"{MD}.protocol_outcome_diagnostics",),
                ("T11_transport", "T11_transport_by_round")),
    SectionSpec("12", "result shapes (III.2.3)", (f"{MD}.exp1_outcome_diagnostics",),
                ("T12_shape",),
                ("no flag for 'Delta ~ 0' or 'permuted-B ~ B': III.2.3 gives no tolerance",)),
    SectionSpec("12.1", "error boundary alignment (III.3)",
                (f"{MD}.protocol_outcome_diagnostics",),
                ("T121_error_boundary_exp1", "T121_error_boundary_exp2")),
    SectionSpec("13", "feasibility (III.1.3)", (), (),
                ("the power re-run with measured parameters is for after the pilot (its target N "
                 "comes from the pilot's effect size) and its scripts (feas/) are not in the "
                 "repo",)),
    SectionSpec("13.1", "the gate (II.6.1)", ("pipelines.math_pipeline.MathPipeline.gate",), (),
                ("done at step 9 (runs/math_formal_n9000/gate_1.json); not re-run here",)),
    SectionSpec("14", "freezing and reproduction (II.10)", ("pipelines.math_tables.run_context",),
                ("T14_provenance",), ("the three math git tags are the authors' (App. A.1)",)),
    SectionSpec("15", "values filled in (App. D; M.12)", (), (),
                ("no computation: the values are in protocol M.12",)),
)  # fmt: skip


# ---- helpers ---------------------------------------------------------------------------


def _sign(x: float | None) -> str:
    if x is None or x != x:
        return "undefined"
    return "positive" if x > 0 else "negative" if x < 0 else "zero"


def _ci(block: dict[str, Any]) -> list[Any]:
    ci = block.get("ci95") or [None, None]
    return [block.get("estimate"), ci[0], ci[1]]


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _n_accepted(cell: dict[str, Any]) -> int | None:
    rate, n = cell.get("accept_rate"), cell.get("n")
    return None if rate is None or rate != rate or not n else round(rate * n)


def _rounds(block: dict[str, Any]) -> list[str]:
    return sorted(block, key=int)


def _exp1_block(rep: dict[str, Any], rule: str) -> dict[str, Any]:
    return rep["exp1"]["primary"] if rule == "main" else rep["exp1"]["sensitivities"][rule]


def _exp2_block(rep: dict[str, Any], rule: str) -> dict[str, Any]:
    return rep["exp2"]["primary"] if rule == "main" else rep["exp2"]["sensitivities"][rule]


def _check(name: str, value: Any, expected: Any, ref: str) -> dict[str, Any]:
    ok = None if value is None else value == expected
    return {"check": name, "value": value, "expected": expected, "ok": ok, "ref": ref}


def _checks_table(tid: str, section: str, title: str, checks: list[dict[str, Any]]) -> Table:
    rows = [[c["check"], c["value"], c["expected"], c["ok"], c["ref"]] for c in checks]
    return Table(
        tid,
        section,
        title,
        ["check", "value", "expected", "ok", "ref"],
        rows,
        ["a value of -- means the marker is not in this run (FAKE report: no run files)"]
        if any(c["value"] is None for c in checks)
        else [],
    )


# ---- the run's markers ------------------------------------------------------------------


def run_context(run_dir: Path | None) -> dict[str, Any]:
    """What the tables read beside ``report.json``: the run's frozen markers and file hashes.
    Empty for a FAKE report (no run files)."""
    if run_dir is None:
        return {}
    run_dir = Path(run_dir)
    exp1, exp2 = run_dir / "exp1", run_dir / "exp2"
    judgments = _jsonl(exp2 / "round_judgments.jsonl")
    versions = Counter(
        row[field].get("math_verify_version")
        for row in judgments
        for field in ("input_grade", "output_grade")
    )
    snapshot = _json(run_dir / "snapshot.json") or {}
    fit = _json(run_dir / "fit.json") or {}
    return {
        "run_dir": run_dir,
        "outcome": _json(exp1 / "outcome_hash.json"),
        "exp1_complete": _json(exp1 / "COMPLETE.json"),
        "assignment": _json(exp2 / "assignment.json"),
        "loop": _json(exp2 / "loop_complete.json"),
        "judge": _json(exp2 / "judge_complete.json"),
        "trajectories": _jsonl(exp2 / "trajectories.jsonl"),
        "n_round_judgments": len(judgments),
        "e_versions": dict(versions),
        "grader": (snapshot.get("math_identity") or {}).get("grader") or {},
        "frozen_taus": [t.get("tau") for t in fit.get("tau", [])] if fit else None,
        "sha256": {
            name: _sha256(run_dir / name)
            for name in (
                "report.json",
                "exp1/decisions.jsonl",
                "exp1/channel.jsonl",
                "exp1/report_exp1.json",
                "exp2/trajectories.jsonl",
                "exp2/round_judgments.jsonl",
                "exp2/decisions.jsonl",
                "exp2/loop_complete.json",
                "exp2/assignment.json",
                "exp2_smoke/smoke_report.json",
                "gate_1.json",
                "fit.json",
            )
        },
    }


def check_run(rep: dict[str, Any], ctx: dict[str, Any]) -> None:
    """The frozen markers still match their files, and the report's counts match the files it
    came from; otherwise the report is stale or the run changed."""
    run_dir = ctx["run_dir"]
    complete = ctx["exp1_complete"] or {}
    for key, name in (("decisions_sha256", "decisions.jsonl"), ("channel_sha256", "channel.jsonl")):
        if complete.get(key) != _sha256(run_dir / "exp1" / name):
            raise MathTablesError(f"exp1/{name} changed since Exp 1's COMPLETE.json")
    judge = ctx["judge"] or {}
    for name, digest in (judge.get("files") or {}).items():
        if digest != _sha256(run_dir / "exp2" / name):
            raise MathTablesError(f"exp2/{name} changed since judge_complete.json")
    if not judge.get("files") or judge.get("SMOKE") is not False:
        raise MathTablesError("Exp 2 is not judged (judge_complete.json) or is a smoke")
    trajectories = ctx["trajectories"]
    if rep["exp2"]["n_trajectories"] != len(trajectories):
        raise MathTablesError("report.json's Exp 2 count differs from trajectories.jsonl")
    for arm in ARMS2:
        members = [t for t in trajectories if t["arm"] == arm]
        failures = sum(not t["task_success"] for t in members)
        block = rep["exp2"]["primary"]["by_arm"][arm]
        if (block["n"], block["n_task_failures"]) != (len(members), failures):
            raise MathTablesError(f"report.json's {arm} TFR counts differ from trajectories.jsonl")
    if rep["exp1"]["n_rows"] != complete.get("n_decisions"):
        raise MathTablesError("report.json's Exp 1 count differs from Exp 1's COMPLETE.json")


# ---- one builder per table ------------------------------------------------------------

Builder = Callable[[dict[str, Any], dict[str, Any]], Table]


def t01(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    rows = []
    for rule in EXP1_RULES:
        d = _exp1_block(rep, rule)["interaction_high_minus_low"]
        est, lo, hi = _ci(d)
        rows.append([rule, "A1 - B", est, lo, hi, d.get("n"), _sign(est)])
    return Table(
        "T01_h1_direction",
        "1",
        "H1: Delta_FA(high) - Delta_FA(low) = D-bar, A1 - B (positive = B better, H1's direction)",
        ["accounting", "contrast", "D_bar", "ci95_lo", "ci95_hi", "n_states", "sign"],
        rows,
        [
            "H1 is stated as Delta_FA(lambda_high) > Delta_FA(lambda_low), i.e. D-bar > 0 (I.2)",
            "same sign under all three accountings: "
            f"{rep['exp1']['primary_robust_across_accountings']}",
            "FA = P(accept | first answer E-INCORRECT) over D_Exp1 (M.9)",
            PILOT_NOTE,
        ],
    )


def t03(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    n = ctx["n_exp1"]
    primary = rep["exp1"]["primary"]
    h2 = rep["exp1_diagnostics"]["H2"]["preregistered"]
    outcome = ctx.get("outcome")
    complete = ctx.get("exp1_complete")
    cell_ns = sorted({primary["by_arm"][a][lam]["n"] for a in ARMS1 for lam in LEVELS})
    missing_evidence = sum(
        primary["by_arm"][a][lam]["evidence_sensitivity"]["n_missing_scores_or_thresholds"]
        for a in ARMS1
        for lam in LEVELS
    )
    missing_prompts = sum(
        rep["exp1_diagnostics"]["actual_first_prompt_hash_equality"][a][f"{x}-{y}"]["n_missing"]
        for a in ARMS1
        for x, y in PAIRS
    )
    checks = [
        _check("n_states", rep["exp1"]["n_questions"], n, "II.2.4, M.2.4"),
        _check(
            "states_not_e_incorrect",
            None if outcome is None else outcome["n"] - outcome["n_incorrect"],
            0,
            "II.2.4, M.2.4: D_Exp1 is E-INCORRECT only (outcome_hash.json)",
        ),
        _check(
            "outcome_hash_recorded_before_calls",
            None if outcome is None else bool(outcome.get("d_exp1_labels_sha256")),
            True,
            "II.9.1: E's outcome hashed before any controller call",
        ),
        _check("rows", rep["exp1"]["n_rows"], 9 * n, "II.1.3: 3 arms x 3 lambda per state"),
        _check("rows_per_cell", cell_ns, [n], "one decision per (state, arm, lambda)"),
        _check(
            "complete_marker_decisions",
            None if complete is None else complete.get("n_decisions"),
            9 * n,
            "exp1/COMPLETE.json",
        ),
        _check(
            "cells_whose_perturbation_stratum_differs_across_arms",
            h2["any_verifier_perturbed"]["cells_differing_across_arms"],
            0,
            "II.5.1: the channel is applied once per (state, lambda)",
        ),
        _check(
            "cells_whose_observed_disagreement_differs_across_arms",
            h2["observed_disagreement"]["cells_differing_across_arms"],
            0,
            "II.5.1, III.1.2: the arms see the same evidence",
        ),
        _check("missing_scores_or_thresholds", missing_evidence, 0, "II.10"),
        _check("missing_first_prompt_hashes", missing_prompts, 0, "C.24"),
    ]
    return _checks_table(
        "T03_exp1_construction",
        "3",
        "Exp 1: what the report and markers show of the construction",
        checks,
    )


def t04(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    rows: list[list[Any]] = []
    for arm in ARMS1:
        for lam in LEVELS:
            c = rep["exp1"]["primary"]["by_arm"][arm][lam]
            rows.append(
                ["Exp 1 primary", "decision-level false accept rate", arm, f"lambda={lam}",
                 c["FA"], c["n"]]
            )  # fmt: skip
    for arm in ARMS2:
        c = rep["exp2"]["primary"]["by_arm"][arm]
        rows.append(["Exp 2 primary", "TFR", arm, "all randomized trajectories", c["TFR"], c["n"]])
    for arm in ARMS2:
        s = rep["exp2"]["secondary"][arm]
        rows += [
            ["Exp 2 secondary", "conditional FAR", arm, "accepted trajectories",
             s["conditional_FAR"], s["n_accepted"]],
            ["Exp 2 secondary", "rewrite rounds (median; right-censored at K)", arm, "",
             s["rewrite_rounds_median"], s["n"]],
            ["Exp 2 secondary", "non-convergence rate", arm, "", s["non_convergence_rate"], s["n"]],
        ]  # fmt: skip
        tr = rep["exp2"]["mechanism"][arm]["rewrite_transitions"]
        for label, key in (
            ("incorrect -> correct", "task_failure->task_success"),
            ("correct -> incorrect", "task_success->task_failure"),
        ):
            rows.append(
                ["Exp 2 secondary", f"rewrite transitions {label} (M.9)", arm, "rewrites",
                 tr.get(key, 0), rep["exp2"]["mechanism"][arm]["n_rewrites"]]
            )  # fmt: skip
    for arm in ARMS2:
        m = rep["exp2"]["mechanism"][arm]
        rows += [
            ["mechanism", "decision accuracy", arm, "all decisions", m["decision_accuracy"],
             m["n_decisions"]],
            ["mechanism", "rewrite improvement rate", arm, "rewrites of task failures",
             m["rewrite_improvement_rate"],
             sum(v for k, v in m["rewrite_transitions"].items() if k.startswith("task_failure"))],
        ]  # fmt: skip
    rows.append(["mechanism", "R_t^(a)", "all", "see T08_r_t", None, None])
    return Table(
        "T04_metrics",
        "4",
        "the metrics of II.9.3 / M.9 at a glance (main accounting)",
        ["layer", "metric", "arm", "cell", "value", "n"],
        rows,
        [
            "TFR and conditional FAR are different quantities (C.10); Exp 1 FA and Exp 2 TFR are "
            "not comparable",
            "task failure = E-INCORRECT or unresolved (M.8); transitions are counted, not named "
            "over-correction (M.9)",
            PILOT_NOTE,
        ],
    )


def t05(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    grader = ctx.get("grader") or {}
    version = grader.get("math_verify_version")
    outcome = ctx.get("outcome")
    versions = ctx.get("e_versions")
    rows = [
        [
            "exp1_outcome (step 5's E judgments of D_Exp1)",
            grader.get("backend_identity"),
            version,
            grader.get("contract_version"),
            None if outcome is None else outcome["n"],
            None if not grader else True,
        ],
        [
            "exp2_round_judgments (input and output grades)",
            grader.get("backend_identity"),
            None if not versions else sorted(versions),
            grader.get("contract_version"),
            ctx.get("n_round_judgments") and 2 * ctx["n_round_judgments"],
            None if not versions else list(versions) == [version],
        ],
    ]
    return Table(
        "T05_e_identity",
        "5",
        "every E record names one E (II.9.1; M.9: Math-Verify, version pinned)",
        ["source", "e", "math_verify_version", "contract", "n_records", "single_e"],
        rows,
        [
            "E is code: no prompt hash; its identity is the Math-Verify version and answer "
            "contract (snapshot.json math_identity.grader)"
        ],
    )


def t06_effective_n(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    return Table(
        "T06_exp1_effective_n",
        "6",
        "N of the Exp 1 primary metric (III.1.2)",
        ["n_states", "n_did_units"],
        [[rep["exp1"]["n_questions"],
          rep["exp1"]["primary"]["interaction_high_minus_low"]["n"]]],
        ["D_Exp1 holds E-INCORRECT first answers only (M.2.4), so the false accept rate's "
         "denominator is D_Exp1 in full and N = n_states"],
    )  # fmt: skip


def t06_cells(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    rows = []
    for arm in ARMS1:
        for lam in LEVELS:
            c = rep["exp1"]["primary"]["by_arm"][arm][lam]
            rows.append([arm, lam, c["n"], c["FA"], c["n_accept"]])
    return Table(
        "T06_exp1_cells",
        "6",
        "FA(arm, lambda) = P(accept) over D_Exp1, main accounting",
        ["arm", "lambda", "n", "false_accept_rate", "n_accept"],
        rows,
    )


def t06_did(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    rows = []
    for role, contrast, key in (
        ("primary", "A1 - B", "interaction_high_minus_low"),
        ("secondary", "permuted_B - B", "secondary_permuted_B_interaction_high_minus_low"),
    ):
        for rule in EXP1_RULES:
            d = _exp1_block(rep, rule)[key]
            rows.append([role, contrast, rule, "high vs low", *_ci(d), d.get("n")])
    b = rep["exp1"]["bootstrap"]
    return Table(
        "T06_exp1_did",
        "6",
        "D-bar: (Y_A1 - Y_B at high) - (at low), per x0; permuted-B - B secondary",
        ["role", "contrast", "accounting", "levels", "D_bar", "ci95_lo", "ci95_hi", "n_states"],
        rows,
        [
            f"95% percentile bootstrap over x0, B = {b['n_boot']}, seed {b['seed']} (III.1.2)",
            PILOT_NOTE,
        ],
    )


def t06_delta(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    p = rep["exp1"]["primary"]
    rows = [
        ["A1 - B", *(p["delta_FA_A1_minus_B"][lam]["estimate"] for lam in LEVELS)],
        ["permuted_B - B", *(p["secondary_permuted_B_minus_B"][lam]["estimate"] for lam in LEVELS)],
    ]
    return Table(
        "T06_exp1_delta_by_lambda",
        "6",
        "Delta_FA(lambda) at three levels (secondary, main accounting)",
        ["contrast", "low", "mid", "high"],
        rows,
        ["lambda_mid is not in the primary D-bar (III.1.2); read for the shape of III.2.3"],
    )


def t06_delta_ci(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    p = rep["exp1"]["primary"]
    rows = []
    for contrast, key in (
        ("A1 - B", "delta_FA_A1_minus_B"),
        ("permuted_B - B", "secondary_permuted_B_minus_B"),
    ):
        for lam in LEVELS:
            d = p[key][lam]
            rows.append([contrast, lam, *_ci(d), d.get("n")])
    return Table(
        "T06_exp1_delta_by_lambda_ci",
        "6",
        "Delta_FA(lambda) with its interval per level (paired by question, main accounting)",
        ["contrast", "lambda", "estimate", "ci95_lo", "ci95_hi", "n"],
        rows,
        ["math's report carries the per-level intervals; detox's table gives the estimates only",
         PILOT_NOTE],
    )  # fmt: skip


def t06_prompt_identity(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    diag = rep["exp1_diagnostics"]
    prompts = diag["actual_first_prompt_hash_equality"]
    scores = diag["observed_score_float64_bitwise_equality"]
    rows = []
    for x, y in PAIRS:
        key = f"{x}-{y}"
        per_arm = []
        for arm in ARMS1:
            p, s = prompts[arm][key], scores[arm][key]
            per_arm.append([f"{x}|{y}", arm, p["n_compared"], p["n_equal"], p["equal_fraction"],
                            s["n_equal"], s["equal_fraction"]])  # fmt: skip
        n = sum(r[2] for r in per_arm)
        np_, ns = sum(r[3] for r in per_arm), sum(r[5] for r in per_arm)
        rows.append([f"{x}|{y}", "all", n, np_, np_ / n if n else None, ns, ns / n if n else None])
        rows += per_arm
    return Table(
        "T06_prompt_identity",
        "6",
        "C.24: per arm and pair of levels, share of (state, arm) whose first controller prompt "
        "is the same (main) and whose s_obs is bit for bit the same (auxiliary)",
        ["levels", "arm", "n", "prompt_identical", "prompt_share", "s_obs_identical",
         "s_obs_share"],
        rows,
        ["identical prompts share one controller answer (II.8.2, common random numbers)"],
    )  # fmt: skip


def t061(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    b1, b2 = rep["exp1"]["bootstrap"], rep["exp2"]["bootstrap"]
    ladder = rep["exp1_drift"]["ladder"]["bootstrap"]
    rows = [
        ["Exp 1 D-bar and Delta_FA", b1["unit"], b1["method"], b1["level"], b1["n_boot"],
         b1["seed"]],
        ["Exp 1 drift ladder", ladder["unit"], ladder["method"], ladder["level"],
         ladder["n_boot"], ladder["seed"]],
        ["Exp 2 TFR contrast", b2["unit"], b2["method"], b2["level"], b2["n_boot"], b2["seed"]],
    ]  # fmt: skip
    return Table(
        "T061_bootstrap",
        "6.1",
        "interval settings (III.1.2)",
        ["analysis", "resampling unit", "method", "level", "B", "seed"],
        rows,
        [PILOT_NOTE],
    )


def t062_h2(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    pre = rep["exp1_diagnostics"]["H2"]["preregistered"]
    rows = []
    notes = [pre["note"], f"source of stratifier (i): {pre['source']}"]
    for stratifier in ("any_verifier_perturbed", "observed_disagreement"):
        block = pre[stratifier]
        for stratum in ("True", "False"):
            s = block["strata"].get(stratum) or {"cells": {}, "delta_fa_by_lambda": {}}
            for lam in LEVELS:
                n_a1 = (s["cells"].get(f"A1|{lam}") or {}).get("n", 0)
                n_b = (s["cells"].get(f"B|{lam}") or {}).get("n", 0)
                rows.append([stratifier, stratum, lam, n_a1, n_b, s["delta_fa_by_lambda"].get(lam)])
        notes.append(
            f"{stratifier}: cells whose stratum differs across arms = "
            f"{block['cells_differing_across_arms']} (must be 0)"
        )
    notes.append("with p fixed at 0 / 0.5 / 1, stratifier (i) varies only at lambda_mid (C.27)")
    return Table(
        "T062_h2_strata",
        "6.2",
        "Delta_FA(lambda), A1 - B, within the two arm-independent strata",
        ["stratifier", "stratum", "lambda", "n_A1", "n_B", "delta_fa"],
        rows,
        notes,
    )


def t062_r_ref(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    ref = rep["exp1_diagnostics"]["h2_reference_r_ref"]
    rows = [
        [v, b["detection_rate"], b["n_unsafe"], b["accuracy"], ref["split"], ref["channel"]]
        for v, b in ref["by_verifier"].items()
    ]
    return Table(
        "T062_r_ref",
        "6.2",
        "r_hat_ref: offline reference, reported alongside H2, never a stratifier (II.1.1)",
        ["verifier", "detection_rate (E-INCORRECT)", "n_e_incorrect", "accuracy_whole_split",
         "split", "lambda"],
        rows,
        [f"source: the gate report (gate_1.json); {ref['use']}"],
    )  # fmt: skip


def t063_exp1(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    rows = []
    for lam in LEVELS:
        for arm in ARMS1:
            cells = rep["exp1"]["primary"]["by_arm"][arm][lam]["evidence_sensitivity"]["cells"]
            for key, label in CELLS.items():
                c = cells.get(key) or {"n": 0, "accept_rate": None}
                rows.append([lam, arm, label, c["n"], _n_accepted(c), c["accept_rate"]])
    return Table(
        "T063_evidence_exp1",
        "6.3",
        "accept rate by arm x lambda x observed (y1, y2), main accounting",
        ["lambda", "arm", "(y1,y2)", "n", "n_accepted", "accept_rate"],
        rows,
        ["all three arms see the same evidence for the same (state, lambda), so the four cells "
         "are arm-independent; read across cells within an arm for whether the controller heeds "
         "the checkers, across arms within a cell for the representation"],
    )  # fmt: skip


def t063_exp2(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    rows = []
    diag = rep["exp2"]["round_diagnostics"]
    for t in _rounds(diag["A1"]):
        for arm in ARMS2:
            cells = (diag[arm].get(t) or {}).get("evidence_sensitivity", {}).get("cells", {})
            for key, label in CELLS.items():
                c = cells.get(key) or {"n": 0, "accept_rate": None}
                rows.append([int(t), arm, label, c["n"], _n_accepted(c), c["accept_rate"]])
    return Table(
        "T063_evidence_exp2",
        "6.3",
        "accept rate by arm x t x observed (y1, y2), main accounting",
        ["t", "arm", "(y1,y2)", "n", "n_accepted", "accept_rate"],
        rows,
        ["descriptive: at t >= 2 the evidence depends on the arm's own trajectory "
         "(post-treatment), so no contrast is drawn from this table (III.1.2)"],
    )  # fmt: skip


def t07(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    n_arm = ctx["n_per_arm"]
    sizes = {arm: rep["exp2"]["secondary"][arm]["n"] for arm in ARMS2}
    assignment = ctx.get("assignment")
    loop = ctx.get("loop")
    checks = [
        _check("arm_sizes", sizes, dict.fromkeys(ARMS2, n_arm), "II.8.3"),
        _check(
            "assignment_counts",
            None if assignment is None else dict(sorted(Counter(assignment.values()).items())),
            dict.fromkeys(ARMS2, n_arm),
            "II.8.3: balanced permutation, assignment.json on disk",
        ),
        _check("rounds_complete", None if loop is None else loop.get("rounds"), [1, 2, 3, 4],
               "II.8.3: K = 4"),
        _check("smoke", None if loop is None else loop.get("SMOKE"), False, "M.10"),
        _check(
            "thresholds_are_frozen_tau",
            None if loop is None or ctx.get("frozen_taus") is None
            else loop.get("thresholds") == ctx["frozen_taus"],
            True,
            "step 7 (fit.json), tag math-tau-frozen",
        ),
        _check("all_trajectories_judged", rep["exp2"]["complete"], True,
               "II.8.3: E judges every round text and x_T after the loop"),
        _check("missing_terminal_labels", rep["exp2"]["n_missing_terminal_labels"], 0, "II.9.1"),
        _check("unknown_round_grade_records", rep["exp2"]["n_unknown_round_grade_records"], 0,
               "II.9.1"),
        _check("v_input_overlength_total", sum(
            rep["exp2"]["secondary"][arm]["n_v_input_overlength"] for arm in ARMS2),
            0, "M.8 (reported, not required: T08_overlength)"),
    ]  # fmt: skip
    return _checks_table(
        "T07_exp2_construction", "7", "Exp 2: start, randomization, loop and terminal text", checks
    )


def t08_tfr(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    rows = []
    for arm in ARMS2:
        b = rep["exp2"]["primary"]["by_arm"][arm]
        rows.append([arm, b["n"], b["TFR"], b["n_task_failures"], b["n_correct"],
                     b["n_incorrect"], b["n_unresolved"]])  # fmt: skip
    return Table(
        "T08_tfr_by_arm",
        "8",
        "TFR = P(task_success(x_T) = False) over all randomized trajectories (M.9)",
        ["arm", "n", "TFR", "n_task_failures", "n_correct", "n_incorrect", "n_unresolved"],
        rows,
        ["unresolved x_T is a task failure (M.8)"],
    )


def t08_contrast(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    p = rep["exp2"]["primary"]
    b = rep["exp2"]["bootstrap"]
    return Table(
        "T08_tfr_contrast",
        "8",
        "primary causal claim: TFR(A1) - TFR(B) (positive = B better)",
        ["rule", "estimate", "ci95_lo", "ci95_hi", "n_A1", "n_B"],
        [["main", *_ci(p["contrast"]), p["by_arm"]["A1"]["n"], p["by_arm"]["B"]["n"]]],
        [
            f"95% percentile bootstrap, trajectories resampled within arm, B = {b['n_boot']}, "
            f"seed {b['seed']}",
            PILOT_NOTE,
        ],
    )


def t08_secondary(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    rows = []
    for arm in ARMS2:
        s = rep["exp2"]["secondary"][arm]
        rows.append([arm, s["n"], s["n_accepted"], s["conditional_FAR"],
                     s["non_convergence_rate"], s["rewrite_rounds_median"],
                     s["decision_rounds_median"]])  # fmt: skip
    return Table(
        "T08_secondary",
        "8",
        "secondary: conditional FAR (post-treatment selection), non-convergence, rounds",
        ["arm", "n", "n_accepted", "conditional_FAR", "non_convergence_rate",
         "rewrite_rounds_median", "decision_rounds_median"],
        rows,
        ["conditional FAR = P(failure | accepted): secondary / mechanism only (II.9.3)"],
    )  # fmt: skip


def _distribution(rep: dict[str, Any], key: str) -> list[list[Any]]:
    rows = []
    for arm in ARMS2:
        dist = rep["exp2"]["secondary"][arm][key]
        rows += [[arm, int(k), v] for k, v in sorted(dist.items(), key=lambda kv: int(kv[0]))]
    return rows


def t08_rewrite_rounds(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    return Table(
        "T08_rewrite_rounds",
        "8",
        "distribution of rewrite rounds per trajectory (right-censored at K = 4)",
        ["arm", "n_rewrites", "n_trajectories"],
        _distribution(rep, "rewrite_rounds_distribution"),
    )


def t08_decision_rounds(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    return Table(
        "T08_decision_rounds",
        "8",
        "distribution of controller decisions per trajectory, T (K = 4)",
        ["arm", "n_decisions", "n_trajectories"],
        _distribution(rep, "decision_rounds_distribution"),
        ["4 decisions = accepted at t = 4 or not converged (non_convergence_rate in "
         "T08_secondary); a v_input_overlength trajectory ends early (M.8)"],
    )  # fmt: skip


def t08_r_t(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    diag = rep["exp2_reliability_diagnostics"]
    rows = []
    for arm in ARMS2:
        for t in _rounds(diag["R_t"][arm]):
            c = diag["R_t"][arm][t]
            rows.append([arm, int(t), c["n"], c["v1"], c["v2"]])
    return Table(
        "T08_r_t",
        "8",
        "R_t^(a): realized verifier reliability per round and arm (an outcome)",
        ["arm", "t", "n", "R_v1", "R_v2"],
        rows,
        [
            "post-treatment: reported as an outcome only, never a stratifier or design factor "
            "(II.1.1)",
            diag["R_t_basis"],
        ],
    )


def t08_mechanism(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    rows = []
    for arm in ARMS2:
        m = rep["exp2"]["mechanism"][arm]
        rows.append([arm, m["n_decisions"], m["decision_accuracy"], m["n_accepts"],
                     m["accept_decision_accuracy"], m["n_rewrites"],
                     m["rewrite_improvement_rate"], m["rewrite_degradation_rate"],
                     m["n_rows_without_judgment"]])  # fmt: skip
    return Table(
        "T08_mechanism",
        "8",
        "decision accuracy and rewrite improvement rate (E's round judgments)",
        ["arm", "n_decisions", "decision_accuracy", "n_accepts", "accept_decision_accuracy",
         "n_rewrites", "rewrite_improvement_rate", "rewrite_degradation_rate",
         "n_rows_without_judgment"],
        rows,
        [rep["exp2"]["mechanism_label_basis"]],
    )  # fmt: skip


def t08_transitions(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    rows = []
    for arm in ARMS2:
        tr = rep["exp2"]["mechanism"][arm]["rewrite_transitions"]
        for key in sorted(tr):
            before, after = key.split("->")
            rows.append([arm, before, after, tr[key]])
    return Table(
        "T08_rewrite_transitions",
        "8",
        "rewrite transition counts (M.9): task success before -> after each rewrite",
        ["arm", "before", "after", "n_rewrites"],
        rows,
        ["correct -> incorrect and incorrect -> correct are each reported; not named "
         "over-correction (M.9)"],
    )  # fmt: skip


def t08_overlength(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    rows = []
    for arm in ARMS2:
        s = rep["exp2"]["secondary"][arm]
        rows.append([arm, s["n"], s["n_v_input_overlength"], s["v_input_overlength_rate"],
                     s["v_input_overlength_by_verifier"]])  # fmt: skip
    return Table(
        "T08_overlength",
        "8",
        "v_input_overlength by arm (M.8, II.10: a rewrite longer than a V's loop L_in ends the "
        "trajectory)",
        ["arm", "n", "n_v_input_overlength", "v_input_overlength_rate", "by_verifier"],
        rows,
        ["main accounting keeps these trajectories; complete-case sensitivity (b) drops them "
         "(M.8)"],
    )  # fmt: skip


def t09_pf_exp1(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    rows = []
    for arm in ARMS1:
        for lam in LEVELS:
            c = rep["exp1"]["primary"]["by_arm"][arm][lam]
            rows.append([arm, lam, c["n"], c["n_parse_fail"], c["parse_fail_rate"],
                         c["parse_fail_above_two_percent"]])  # fmt: skip
    alerts = [r for r in rows if r[-1]]
    return Table(
        "T09_parse_fail_exp1",
        "9",
        "parse_fail rate by arm x lambda; > 2% flagged (II.8.2)",
        ["arm", "lambda", "n", "n_parse_fail", "parse_fail_rate", "ALERT"],
        rows,
        [f"cells above the threshold: {len(alerts)}" if alerts else "no cell above the threshold"],
    )


def t09_pf_exp2(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    rows = []
    diag = rep["exp2"]["round_diagnostics"]
    for arm in ARMS2:
        for t in _rounds(diag[arm]):
            c = diag[arm][t]
            rows.append([arm, int(t), c["n_decisions"], c["n_parse_fail"], c["parse_fail_rate"],
                         c["parse_fail_above_two_percent"]])  # fmt: skip
    alerts = [r for r in rows if r[-1]]
    return Table(
        "T09_parse_fail_exp2",
        "9",
        "parse_fail rate by arm x t; > 2% flagged (II.8.2)",
        ["arm", "t", "n", "n_parse_fail", "parse_fail_rate", "ALERT"],
        rows,
        [f"cells above the threshold: {len(alerts)}" if alerts else "no cell above the threshold"],
    )


def t09_unresolved(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    u = rep["exp2"]["unresolved_rates"]
    rows = []
    for arm in ARMS2:
        for t in _rounds(u["by_arm_round"][arm]):
            c = u["by_arm_round"][arm][t]
            rows.append([arm, int(t), c["n_texts"], c["n_unresolved"], c["unresolved_rate"],
                         c["above_two_percent"]])  # fmt: skip
    return Table(
        "T09_unresolved_exp2",
        "9",
        "unresolved rate by arm x round (M.8); any cell > 2% reported prominently",
        ["arm", "t", "n_texts", "n_unresolved", "unresolved_rate", "ALERT"],
        rows,
        [
            f"status: {u['status']}; cells above {u['threshold']:.0%}: {len(u['alert_cells'])}",
            f"denominator: {u['denominator']}",
            "t = 5 is the last rewrite x_5 of a trajectory that never accepted (II.8.3)",
        ],
    )


def t09_robustness(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    rows = []
    for rule in EXP1_RULES:
        d = _exp1_block(rep, rule)["interaction_high_minus_low"]
        est, lo, hi = _ci(d)
        rows.append(["Exp 1", "D-bar (A1 - B)", rule, est, lo, hi, d.get("n"), _sign(est)])
    for rule in EXP2_RULES:
        block = _exp2_block(rep, rule)
        est, lo, hi = _ci(block["contrast"])
        n = sum(block["by_arm"][arm]["n"] for arm in ARMS2)
        rows.append(["Exp 2", "TFR(A1) - TFR(B)", rule, est, lo, hi, n, _sign(est)])
    return Table(
        "T09_robustness",
        "9",
        "main accounting and the II.8.2 / M.8 sensitivities",
        ["experiment", "estimate", "rule", "value", "ci95_lo", "ci95_hi", "n", "sign"],
        rows,
        [
            "Exp 1 robust (main, (a), (b) same sign): "
            f"{rep['exp1']['primary_robust_across_accountings']}",
            f"Exp 2 robust (main, (a), (b) same sign): {rep['exp2']['robust_across_accountings']}",
            "same sign = all strictly positive or all strictly negative; a zero estimate is "
            "neither",
            "terminal_unresolved_excluded is M.8's sensitivity (c), not part of the robustness "
            "rule; complete_case (b) also drops v_input_overlength trajectories (M.8)",
        ],
    )


def t10(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    drift = rep["exp1_drift"]
    b = drift["ladder"]["bootstrap"]
    rows = [
        [r["verifier"], r["lambda"], r["R_exp1"], r["ci95_lo"], r["ci95_hi"], r["n"],
         r["R_gate"], r["exp1_minus_gate"]]
        for r in drift["vs_gate"]
    ]  # fmt: skip
    return Table(
        "T10_drift",
        "10",
        "R_lambda^(i) on D_Exp1 (observed judgments, all first answers E-INCORRECT) vs the "
        "gate's on D_gate",
        ["verifier", "lambda", "R_exp1", "ci95_lo", "ci95_hi", "n", "R_gate", "exp1_minus_gate"],
        rows,
        [
            f"bootstrap over x0, B = {b['n_boot']}, seed {b['seed']}",
            "reported as a transport diagnostic; no condition is removed (II.6.3)",
        ],
    )


def t11(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    tr = rep["transport"]
    rows = []
    for q in ("s1", "s2", "m1", "m2", "task_failure"):
        block = tr[q]
        for pop, ks in (
            ("exp1_states", None),
            ("exp2_decision_points", block.get("ks")),
            ("exp2_t_ge_2", block.get("ks_t_ge_2")),
        ):
            s = block[pop]
            rows.append([q, pop, s["n"], s["mean"], s["sd"], s["median"], s["q10"], s["q90"], ks])
    return Table(
        "T11_transport",
        "11",
        "scores (raw), margins and task failure: Exp 1 states vs Exp 2 decision points",
        ["quantity", "population", "n", "mean", "sd", "median", "q10", "q90", "ks_vs_exp1"],
        rows,
        [tr["note"], f"Exp 2 rows t = 1: {tr['n_exp2_rows_t1']}, t >= 2: "
         f"{tr['n_exp2_rows_t_ge_2']}"],
    )  # fmt: skip


def t11_by_round(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    by = rep["transport"]["by_round_exp2"]
    rows = []
    for t in _rounds(by):
        for q in ("s1", "s2", "task_failure"):
            s = by[t].get(q)
            if s:
                rows.append([int(t), q, s["n"], s["mean"], s["median"]])
    return Table(
        "T11_transport_by_round",
        "11",
        "Exp 2 decision points by round",
        ["t", "quantity", "n", "mean", "median"],
        rows,
    )


def t12(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    shape = rep["result_shape"]
    rows = []
    for contrast in ("A1 - B", "permuted_B - B"):
        s = shape[contrast]
        d = s["delta_fa_by_lambda"]
        rows.append([contrast, d["low"], d["mid"], d["high"], s["rising_in_lambda"],
                     s["negative_at_low"]])  # fmt: skip
    return Table(
        "T12_shape",
        "12",
        "III.2.3 shapes read off the point estimates",
        ["contrast", "delta_low", "delta_mid", "delta_high", "rising_in_lambda", "negative_at_low"],
        rows,
        [shape["note"], "null result A/B: decision accuracy by arm (T08_mechanism) and error "
         "boundary alignment (T121_*)"],
    )  # fmt: skip


def _error_boundary(rep: dict[str, Any], part: str, keys: Sequence[str], arms) -> list[list]:
    cells = rep["error_boundary_alignment"][part]["cells"]
    rows = []
    for key in keys:
        for arm in arms:
            c = (cells.get(key) or {}).get(arm)
            if c is None:
                continue
            rows.append(
                [int(key) if part == "exp2" else key, arm, c["n"], c["n_misaligned"],
                 c["misaligned_share"], c["decision_error_rate_on_misaligned"],
                 c["decision_error_rate_overall"]]
            )  # fmt: skip
    return rows


EB_COLUMNS = ["n", "n_misaligned", "misaligned_share", "error_rate_on_misaligned",
              "error_rate_overall"]  # fmt: skip


def t121_exp1(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    eb = rep["error_boundary_alignment"]["exp1"]
    return Table(
        "T121_error_boundary_exp1",
        "12.1",
        "error boundary alignment by arm x lambda (III.3, main accounting)",
        ["lambda", "arm", *EB_COLUMNS],
        _error_boundary(rep, "exp1", LEVELS, ARMS1),
        [eb["definition"]],
    )


def t121_exp2(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    eb = rep["error_boundary_alignment"]["exp2"]
    return Table(
        "T121_error_boundary_exp2",
        "12.1",
        "error boundary alignment by arm x t (III.3, main accounting)",
        ["t", "arm", *EB_COLUMNS],
        _error_boundary(rep, "exp2", sorted(eb["cells"], key=int), ARMS2),
        [eb["definition"]],
    )


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def t14(rep: dict[str, Any], ctx: dict[str, Any]) -> Table:
    rows: list[list[Any]] = [["FAKE", bool(rep.get("FAKE"))]]
    rows.append(["report protocol", rep.get("protocol")])
    rows.append(["stage / execution_mode", f"{rep.get('stage')} / {rep.get('execution_mode')}"])
    run_dir = ctx.get("run_dir")
    if run_dir is not None:
        rows.append(["run dir", paths.shown(run_dir)])
        for name, digest in ctx["sha256"].items():
            rows.append([f"{name} sha256", digest])
        outcome = ctx.get("outcome") or {}
        rows.append(["D_Exp1 labels sha256 (outcome_hash.json)",
                     outcome.get("d_exp1_labels_sha256")])  # fmt: skip
    for tag in TAGS:
        rows.append([f"tag {tag}", _git("rev-list", "-n1", tag)])
    head = _git("rev-parse", "HEAD")
    rows.append(["git HEAD at tabulation", head])
    rows.append(
        [
            "git dirty at tabulation",
            None if head is None else _git("status", "--porcelain") is not None,
        ]
    )
    rows.append(["tabulated at", ctx.get("tabulated_at")])
    return Table(
        "T14_provenance",
        "14",
        "what these tables were laid out from",
        ["item", "value"],
        rows,
        ["every number is report.json's (MathPipeline.report); this module computes none"],
    )


BUILDERS: dict[str, Builder] = {
    "T01_h1_direction": t01,
    "T03_exp1_construction": t03,
    "T04_metrics": t04,
    "T05_e_identity": t05,
    "T06_exp1_effective_n": t06_effective_n,
    "T06_exp1_cells": t06_cells,
    "T06_exp1_did": t06_did,
    "T06_exp1_delta_by_lambda": t06_delta,
    "T06_exp1_delta_by_lambda_ci": t06_delta_ci,
    "T06_prompt_identity": t06_prompt_identity,
    "T061_bootstrap": t061,
    "T062_h2_strata": t062_h2,
    "T062_r_ref": t062_r_ref,
    "T063_evidence_exp1": t063_exp1,
    "T063_evidence_exp2": t063_exp2,
    "T07_exp2_construction": t07,
    "T08_tfr_by_arm": t08_tfr,
    "T08_tfr_contrast": t08_contrast,
    "T08_secondary": t08_secondary,
    "T08_rewrite_rounds": t08_rewrite_rounds,
    "T08_decision_rounds": t08_decision_rounds,
    "T08_r_t": t08_r_t,
    "T08_mechanism": t08_mechanism,
    "T08_rewrite_transitions": t08_transitions,
    "T08_overlength": t08_overlength,
    "T09_parse_fail_exp1": t09_pf_exp1,
    "T09_parse_fail_exp2": t09_pf_exp2,
    "T09_unresolved_exp2": t09_unresolved,
    "T09_robustness": t09_robustness,
    "T10_drift": t10,
    "T11_transport": t11,
    "T11_transport_by_round": t11_by_round,
    "T12_shape": t12,
    "T121_error_boundary_exp1": t121_exp1,
    "T121_error_boundary_exp2": t121_exp2,
    "T14_provenance": t14,
}


def build_tables(report: dict[str, Any], ctx: dict[str, Any]) -> list[Table]:
    """Every table of :data:`SECTIONS`, in section order."""
    listed = [t for s in SECTIONS for t in s.tables]
    missing = sorted(set(listed) ^ set(BUILDERS))
    if missing:
        raise MathTablesError(f"SECTIONS and BUILDERS disagree on {missing}")
    return [BUILDERS[t](report, ctx) for t in listed]


# ---- output (detox's layout) -----------------------------------------------------------


def render_markdown(tables: Sequence[Table], *, fake: bool) -> str:
    out: list[str] = []
    title = (
        "FAKE -- math preregistered tables on FAKE data (step 10)"
        if fake
        else "Preregistered tables (math)"
    )
    out += [f"# {title}", ""]
    if fake:
        out += [
            "**FAKE.** Every number below comes from math step 10's FAKE report (arm labels "
            "shuffled). None of it is a result.",
            "",
        ]
    out += [
        PILOT_NOTE,
        "",
        "Every number is `report.json`'s (`MathPipeline.report`, protocol v1.2 Appendix M); "
        "this layout computes none. Table ids follow detox's (`analysis/prereg_tables.py`); "
        "math-only tables: T06_exp1_delta_by_lambda_ci, T08_decision_rounds, "
        "T08_rewrite_transitions, T08_overlength, T09_unresolved_exp2.",
        "",
    ]
    by_section: dict[str, list[Table]] = {}
    for t in tables:
        by_section.setdefault(t.section, []).append(t)
    for spec in SECTIONS:
        out += [f"## Section {spec.section}: {spec.title}", ""]
        if spec.functions:
            out += ["Functions: " + ", ".join(f"`{f}`" for f in spec.functions), ""]
        for reason in spec.not_implemented:
            out += [f"Not computed here: {reason}.", ""]
        for t in by_section.get(spec.section, []):
            out += [f"### {'FAKE ' if fake else ''}{t.id}: {t.title}", ""]
            out.append("| " + " | ".join(t.columns) + " |")
            out.append("|" + "---|" * len(t.columns))
            for row in t.rows:
                out.append("| " + " | ".join(_cell(x) for x in row) + " |")
            out.append("")
            for n in t.notes:
                out.append(f"* {n}")
            if t.notes:
                out.append("")
    return "\n".join(out).rstrip() + "\n"


def write_tables(
    tables: Sequence[Table], report: dict[str, Any], out_dir: Path, *, fake: bool
) -> list[Path]:
    """One JSON per table, the Markdown of all of them, the full report and the section map,
    as detox's ``write_tables`` writes them (``FAKE_*`` for a FAKE report)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = FAKE_PREFIX if fake else ""
    written = []

    def dump(name: str, obj: Any) -> None:
        p = out_dir / f"{prefix}{name}"
        p.write_text(
            json.dumps(obj, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
            newline="\n",
        )
        written.append(p)

    for t in tables:
        dump(f"{t.id}.json", {"FAKE": fake, **asdict(t)})
    dump("analysis_full.json", {**report, "FAKE": fake})
    dump("sections.json", {"FAKE": fake, "sections": [asdict(s) for s in SECTIONS]})
    md = out_dir / f"{prefix}tables.md"
    md.write_text(render_markdown(tables, fake=fake), encoding="utf-8", newline="\n")
    written.append(md)
    return written


def tabulate(
    report: dict[str, Any],
    out_dir: Path,
    *,
    run_dir: Path | None,
    fake: bool,
    n_exp1: int = 300,
    n_per_arm: int = 150,
) -> list[Path]:
    """Check the report against its run (formal) or its FAKE marker, then write the tables."""
    if fake and report.get("FAKE") is not True:
        raise MathTablesError("fake=True needs a report marked FAKE (math step 10)")
    if not fake and report.get("FAKE"):
        raise MathTablesError("the formal tables refuse a FAKE report")
    ctx = run_context(None if fake else run_dir)
    if not fake:
        check_run(report, ctx)
    ctx.update(
        n_exp1=n_exp1,
        n_per_arm=n_per_arm,
        tabulated_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
    )
    return write_tables(build_tables(report, ctx), report, out_dir, fake=fake)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", type=Path, default=RUN)
    ap.add_argument("--out-dir", type=Path, default=None, help="default <run-dir>/analysis")
    args = ap.parse_args(argv)
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    for p in tabulate(report, args.out_dir or run_dir / "analysis", run_dir=run_dir, fake=False):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
