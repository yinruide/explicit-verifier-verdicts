"""Lay out the paper's tables from the analysis outputs that ``make tables`` regenerates.

This module computes no estimate, interval or diagnostic of its own, with two exceptions that are
stated in the output: the base-rate combination (``tab:baserate``: p * TFR + (1 - p) * corruption,
the formula in the table's caption) and the decomposition of the paired Experiment 2 contrast by
primary assignment (``tab:detox-halves``: terminal failure counts of each arm on each half of the
primary randomization). Every other cell is copied from one of:

* ``<out>/detox_primary/`` and ``<out>/math_primary/``: the primary tables
  (``python -m analysis.prereg_tables``; ``pipelines/math_tables.py``);
* ``<out>/ext_analysis_{detox,math}_*/report.json``: the extension contrasts;
* ``<out>/ext_decision_quality_{detox,math}_*/report.json``: operating points, rules, verdict
  following (post hoc);
* ``<out>/ext_ceiling_summary_math_*/report.json``: the correction ceiling (post hoc);
* ``<out>/paper_appendix_ab/report.json``: cue diagnostics and minimum detectable effects;
* the data directory: the gate reports (``data/diagnostics/step09_report.json``,
  ``runs/math_formal_n9000/gate_1.json``), the calibration fits and their diagnostics
  (``frozen/calibration_*.json`` and the calibration runs), the first-attempt label counts.

    python analysis/paper_tables.py --out out

Writes ``<out>/paper/tables.json`` (every cell, full precision) and ``<out>/paper/tables.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from collections.abc import Sequence
from typing import Any

from rafr.core import paths
from rafr.logging.writer import read_jsonl

LEVELS = ("low", "mid", "high")
ARMS_OP = ("no_verifier", "A1", "verdict_only", "B", "B_pihat", "B_pihat_mismatched", "permuted_B")
ARM_NAMES = {
    "no_verifier": "No-Verifier",
    "A1": "A1",
    "verdict_only": "Verdict-only",
    "B": "B",
    "B_pihat": "B+pi",
    "B_pihat_mismatched": "B+pi (mismatch)",
    "permuted_B": "B_perm",
    "oracle_feedback": "Oracle",
}
RULES = ("V1_only", "V2_only", "OR", "AND")
RULE_NAMES = {
    "V1_only": "Rule: V1 only",
    "V2_only": "Rule: V2 only",
    "OR": "Rule: OR",
    "AND": "Rule: AND",
}
# Experiment 1 contrasts in the order of the paper's contrast tables: (X, Y)
PAIRS = (
    ("no_verifier", "A1"),
    ("A1", "verdict_only"),
    ("verdict_only", "B"),
    ("B", "B_pihat"),
    ("B_pihat_mismatched", "B_pihat"),
    ("A1", "B"),
    ("permuted_B", "B"),
)
# Experiment 2 contrasts (X - Y) in the order of the paper's Experiment 2 tables
PAIRS2 = (
    ("no_verifier", "A1"),
    ("A1", "verdict_only"),
    ("verdict_only", "B"),
    ("B", "B_pihat"),
    ("A1", "B"),
)
ARMS2 = ("no_verifier", "A1", "verdict_only", "B", "B_pihat")
CALIBRATION_RUNS = {
    "detox": "runs/ext_calibration_detox_20261031T084115Z/diagnostics.json",
    "math": "runs/math_ext_calibration_candidate_20261105T104155Z/diagnostics.json",
}
CALIBRATION_FROZEN = {
    "detox": "frozen/calibration_detox.json",
    "math": "frozen/calibration_math.json",
}
GATE = {
    "detox": "data/diagnostics/step09_report.json",
    "math": "runs/math_formal_n9000/gate_1.json",
}
DETOX_LABELS = "data/labels/state_judgments.jsonl"
MATH_INITIAL = "runs/math_formal_n9000/initial_complete.json"
DETOX_PRIMARY_EXP2 = "runs/exp2_20260928T033025Z/trajectories.jsonl"
DETOX_FILL_EXP2 = "runs/exp2ext_e_incorrect_fill_20261031T152908Z/trajectories.jsonl"


class PaperTablesError(RuntimeError):
    pass


# ---- reading -------------------------------------------------------------------------------


def _json(p: Path | str) -> Any:
    p = Path(p)
    if not p.is_file():
        raise PaperTablesError(f"missing input {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _latest(out: Path, prefix: str) -> Path:
    found = sorted(d for d in out.glob(f"{prefix}_*") if d.is_dir() and "FAKE" not in d.name)
    if not found:
        raise PaperTablesError(f"no {prefix}_* directory under {out}; run `make tables` first")
    return found[-1] / "report.json"


def _table(d: Path, tid: str) -> dict[str, Any]:
    t = _json(d / f"{tid}.json")
    return {"columns": t["columns"], "rows": t["rows"]}


def _rows(t: dict[str, Any], **match: Any) -> list[dict[str, Any]]:
    cols = t["columns"]
    out = []
    for r in t["rows"]:
        row = dict(zip(cols, r, strict=False))
        if all(str(row.get(k)) == str(v) for k, v in match.items()):
            out.append(row)
    return out


def _one(t: dict[str, Any], **match: Any) -> dict[str, Any]:
    rows = _rows(t, **match)
    if len(rows) != 1:
        raise PaperTablesError(f"expected one row for {match}, found {len(rows)}")
    return rows[0]


def _ci(est: float, ci: Sequence[float] | None) -> list[float | None]:
    return [est, *(ci if ci else (None, None))]


def _neg(cell: list[float | None]) -> list[float | None]:
    """X - Y from a Y - X estimate and interval."""
    e, lo, hi = cell
    return [-e, -hi if hi is not None else None, -lo if lo is not None else None]


# ---- tables --------------------------------------------------------------------------------


class Builder:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.tables: dict[str, dict[str, Any]] = {}
        self.dp = {"detox": out / "detox_primary", "math": out / "math_primary"}
        self.ext = {d: _json(_latest(out, f"ext_analysis_{d}")) for d in ("detox", "math")}
        self.dq = {d: _json(_latest(out, f"ext_decision_quality_{d}")) for d in ("detox", "math")}
        self.ceiling = _json(_latest(out, "ext_ceiling_summary_math"))
        self.app = _json(out / "paper_appendix_ab" / "report.json")

    def add(
        self, label: str, title: str, columns: list[str], rows: list[list[Any]], note: str = ""
    ) -> None:
        self.tables[label] = {"title": title, "columns": columns, "rows": rows, "note": note}

    # -- main text --------------------------------------------------------------------------

    def primary(self) -> None:
        rows = []
        for dom in ("detox", "math"):
            cells = _table(self.dp[dom], "T06_exp1_cells")
            delta = _table(self.dp[dom], "T06_exp1_delta_by_lambda")
            for arm in ("A1", "B"):
                fa = [_one(cells, arm=arm, **{"lambda": lv})["false_accept_rate"] for lv in LEVELS]
                rows.append([dom, f"FA, {arm}", *fa])
            d = _one(delta, contrast="A1 - B")
            rows.append([dom, "Delta_FA", *(d[lv] for lv in LEVELS)])
        self.add(
            "tab:primary/top",
            "Planned primary estimates: FA and Delta_FA (A1 - B)",
            ["domain", "quantity", "low", "mid", "high"],
            rows,
        )
        rows = []
        for label, tid, key, est in (
            ("Exp. 1: D-bar", "T06_exp1_did", {"role": "primary", "accounting": "main"}, "D_bar"),
            ("Exp. 2: Delta TFR", "T08_tfr_contrast", {"rule": "main"}, "estimate"),
        ):
            row = [label]
            for dom in ("detox", "math"):
                r = _one(_table(self.dp[dom], tid), **key)
                row.append([r[est], r["ci95_lo"], r["ci95_hi"]])
            rows.append(row)
        self.add(
            "tab:primary/bottom",
            "D-bar (Experiment 1) and TFR(A1) - TFR(B) (Experiment 2)",
            ["quantity", "detox", "math"],
            rows,
        )

    def baserate(self) -> None:
        labels = [r["label"] for r in read_jsonl(paths.resolve(DETOX_LABELS))]
        p_detox = labels.count("UNSAFE") / len(labels)
        mi = _json(paths.resolve(MATH_INITIAL))
        p_math = mi["n_incorrect"] / mi["n_valid"]
        p = {"detox": p_detox, "math": p_math}
        rows = [["Accept all", 100 * p_detox, 100 * p_math]]
        for arm in ARMS2:
            row = [ARM_NAMES[arm]]
            for dom in ("detox", "math"):
                e2 = self.ext[dom]["exp2"]
                tfr = e2["e_incorrect"]["tfr"]["arms"][arm]["rate"]
                cor = e2["e_correct"]["corruption_rate"]["arms"][arm]["rate"]
                row.append(100 * (p[dom] * tfr + (1 - p[dom]) * cor))
            rows.append(row)
        oracle = self.ceiling["main"]["tfr"]["arms"]["oracle_feedback"]["rate"]
        rows.append(["Oracle", None, 100 * p_math * oracle])
        self.add(
            "tab:baserate",
            "Final failure rate (%) at each domain's first-attempt failure rate",
            ["arm", "detox", "math"],
            rows,
            note=f"DERIVED here: p * TFR + (1 - p) * corruption; p = {p_detox:.4f} (detox, "
            f"{labels.count('UNSAFE')}/{len(labels)}), {p_math:.4f} (math, "
            f"{mi['n_incorrect']}/{mi['n_valid']}); oracle assumes no corruption.",
        )

    # -- appendix A / B ---------------------------------------------------------------------

    def cue(self) -> None:
        for dom in ("detox", "math"):
            rows = []
            for split in ("D_fit", "D_gate"):
                for pop, name in (("primary", "Failing"), ("secondary", "All")):
                    for lv in LEVELS:
                        hits = [
                            r
                            for r in self.app["cue"][dom]
                            if r["split"] == split and r["population"] == pop and r["lambda"] == lv
                        ]
                        if len(hits) != 1:
                            raise PaperTablesError(
                                f"cue {dom} {split} {pop} {lv}: {len(hits)} rows"
                            )
                        r = hits[0]
                        rows.append(
                            [
                                split,
                                name,
                                lv,
                                r["n"],
                                r["auroc_v1"],
                                r["auroc_v2"],
                                r["disagreement"],
                                r["err_agree"],
                                r["err_disagree_v1"],
                                r["err_disagree_v2"],
                                r["hmw_v1"],
                                r["hmw_v2"],
                            ]
                        )
            self.add(
                f"tab:cue-{dom}",
                f"Cue diagnostics ({dom})",
                [
                    "split",
                    "set",
                    "lambda",
                    "n",
                    "AUROC V1",
                    "AUROC V2",
                    "Disagree",
                    "Err|agree",
                    "Err|dis V1",
                    "Err|dis V2",
                    "HMW V1",
                    "HMW V2",
                ],
                rows,
            )

    def mde(self) -> None:
        pre = self.app["prereg"]
        k = self.app["constants"]["K_two_sided"]
        sigma = pre["implied_sigma_D_row_011_mean"]
        rows = [["Planned", "A1 - B", None, sigma, None, k * sigma / 300**0.5]]
        for dom in ("detox", "math"):
            for key, name in (
                ("dbar_primary", "A1 - B (primary)"),
                ("dbar_secondary", "B_perm - B"),
            ):
                r = self.app["exp1"][dom][key]
                rows.append(
                    [
                        dom,
                        name,
                        [r["estimate"], *r["ci95_reported"]],
                        r["sd_d"],
                        r["share_nonzero"],
                        r["mde80_two_sided"],
                    ]
                )
        self.add(
            "tab:mde-did",
            "Sensitivity of the Experiment 1 interaction",
            ["domain", "contrast", "D-bar [95%]", "SD(D_j)", "share D_j != 0", "MDE"],
            rows,
            note="Planned row: implied SD from the feasibility table (mean of the 0.11 row); "
            "MDE = K * SD / sqrt(300).",
        )
        rows = []
        for x, y in PAIRS2:
            row = [f"{ARM_NAMES[x]} - {ARM_NAMES[y]}"]
            for dom in ("detox", "math"):
                hit = [
                    r
                    for r in self.app["exp2"][dom]["e_incorrect"]["rows"]
                    if r["X"] == x and r["Y"] == y
                ]
                if len(hit) != 1:
                    raise PaperTablesError(f"exp2 MDE {dom} {x} - {y}")
                r = hit[0]
                row += [[r["estimate"], *r["ci95_reported"]], r["mde80_two_sided"]]
            rows.append(row)
        row = ["A1 - B (planned, 150/150)"]
        for dom in ("detox", "math"):
            r = self.app["exp2"][dom]["pilot_unpaired"]
            row += [[r["estimate"], *r["ci95_reported"]], r["mde80_two_sided"]]
        rows.append(row)
        self.add(
            "tab:mde-exp2",
            "Sensitivity of the Experiment 2 contrasts in TFR",
            ["X - Y", "detox Delta TFR", "detox MDE", "math Delta TFR", "math MDE"],
            rows,
        )

    # -- domain appendices -------------------------------------------------------------------

    def gate(self, dom: str) -> None:
        g = _json(paths.resolve(GATE[dom]))
        drift = _table(self.dp[dom], "T10_drift")
        rows = []
        for v in ("v1", "v2"):
            rows.append(
                [
                    f"D_gate failing (n = {g['n_e_unsafe'] if dom == 'detox' else g['n_states']})",
                    v.upper(),
                    *(g["r"][v][lv] for lv in LEVELS),
                ]
            )
        for v in ("v1", "v2"):
            cells = []
            for lv in LEVELS:
                r = _one(drift, verifier=v, **{"lambda": lv})
                cells.append([r["R_exp1"], r["ci95_lo"], r["ci95_hi"]])
            rows.append(["D_exp (n = 300)", v.upper(), *cells])
        self.add(
            f"tab:{dom}-gate",
            f"Detection rates ({dom})",
            ["set", "verifier", "low", "mid", "high"],
            rows,
            note=f"gaps: {json.dumps(g['gaps'])}",
        )

    def calibration(self, dom: str) -> None:
        diag = _json(paths.resolve(CALIBRATION_RUNS[dom]))
        frozen = _json(paths.resolve(CALIBRATION_FROZEN[dom]))
        rows = []
        for v in ("v1", "v2"):
            fv = frozen["verifiers"][v]
            dv = diag["verifiers"][v]
            g, pi = dv["g_vs_failure"], dv["pihat_vs_verdict_wrong"]
            rows.append(
                [
                    v.upper(),
                    fv["a"],
                    fv["b_fit"],
                    fv["b"],
                    _ci(g["ece"], g["ece_ci"]),
                    g["brier"],
                    _ci(pi["ece"], pi["ece_ci"]),
                ]
            )
        pc = frozen["prior_correction"]
        self.add(
            f"tab:{dom}-calib",
            f"Calibration of pi-hat ({dom})",
            ["verifier", "a_i", "b_fit", "b_i", "ECE of g_i", "Brier", "ECE of pi-hat_i"],
            rows,
            note=f"prior correction: n={pc['n']}, n_positive={pc['n_positive']}, "
            f"p_pool={pc['p_pool']:.4f}, p_R={pc['p_R']:.4f}, delta={pc['delta']:.4f}",
        )

    def primary_full(self, dom: str) -> None:
        d = self.dp[dom]
        cells = _table(d, "T06_exp1_cells")
        delta = _table(d, "T06_exp1_delta_by_lambda")
        did = _table(d, "T06_exp1_did")
        rows = []
        for arm in ("A1", "B", "permuted_B"):
            rows.append(
                [
                    ARM_NAMES[arm],
                    *(_one(cells, arm=arm, **{"lambda": lv})["false_accept_rate"] for lv in LEVELS),
                ]
            )
        for c, name in (("A1 - B", "Delta_FA: A1 - B"), ("permuted_B - B", "B_perm - B")):
            r = _one(delta, contrast=c)
            rows.append([name, *(r[lv] for lv in LEVELS)])
        for role, name in (("primary", "D-bar (A1 - B)"), ("secondary", "D-bar (B_perm - B)")):
            r = _one(did, role=role, accounting="main")
            rows.append([name, [r["D_bar"], r["ci95_lo"], r["ci95_hi"]], None, None])
        self.add(
            f"tab:{dom}-primary/exp1",
            f"Primary results ({dom}), Experiment 1",
            ["arm", "low", "mid", "high"],
            rows,
        )
        tfr = _table(d, "T08_tfr_by_arm")
        sec = _table(d, "T08_secondary")
        mech = _table(d, "T08_mechanism")
        con = _table(d, "T08_tfr_contrast")
        rows = [["TFR (E)", *(_one(tfr, arm=a)["TFR"] for a in ("A1", "B"))]]
        if dom == "detox":
            p = _one(_table(d, "T081_panel"), member="grok")
            rows.append(["TFR (Grok)", p["TFR_A1"], p["TFR_B"]])
        rows += [
            [
                "Failure rate among accepted",
                *(_one(sec, arm=a)["conditional_FAR"] for a in ("A1", "B")),
            ],
            ["Non-convergence", *(_one(sec, arm=a)["non_convergence_rate"] for a in ("A1", "B"))],
            ["Decision accuracy", *(_one(mech, arm=a)["decision_accuracy"] for a in ("A1", "B"))],
            [
                "Rewrites that fix a failure",
                *(_one(mech, arm=a)["rewrite_improvement_rate"] for a in ("A1", "B")),
            ],
        ]
        r = _one(con, rule="main")
        rows.append(["TFR(A1) - TFR(B) (E)", [r["estimate"], r["ci95_lo"], r["ci95_hi"]], None])
        if dom == "detox":
            rows.append(
                ["TFR(A1) - TFR(B) (Grok)", [p["estimate"], p["ci95_lo"], p["ci95_hi"]], None]
            )
        else:
            r = _one(
                _table(d, "T09_robustness"), experiment="Exp 2", rule="terminal_unresolved_excluded"
            )
            rows.append(
                ["Excluding unresolved x_T", [r["value"], r["ci95_lo"], r["ci95_hi"]], None]
            )
        self.add(
            f"tab:{dom}-primary/exp2",
            f"Primary results ({dom}), Experiment 2",
            ["outcome", "A1", "B"],
            rows,
        )

    def h2(self) -> None:
        t = _table(self.dp["detox"], "T062_h2_strata")
        rows = []
        for strat, name in (
            ("any_verifier_perturbed", "At least one score replaced"),
            ("observed_disagreement", "Observed verdicts disagree"),
        ):
            for stratum, sname in (("True", "Yes"), ("False", "No")):
                row = [name, sname]
                for lv in LEVELS:
                    r = _one(t, stratifier=strat, stratum=stratum, **{"lambda": lv})
                    row += [r["n_A1"], r["delta_fa"]]
                rows.append(row)
        self.add(
            "tab:detox-h2",
            "H2 strata in detoxification (Delta_FA, A1 - B)",
            ["stratifier", "stratum", "n low", "dFA low", "n mid", "dFA mid", "n high", "dFA high"],
            rows,
        )

    def operating_points(self, dom: str) -> None:
        op = self.dq[dom]["operating_points"]
        rows = []
        for lv in LEVELS:
            pts = op[lv]["points"]
            for arm in ARMS_OP:
                a = pts["arms"][arm]
                rows.append(
                    [
                        lv,
                        ARM_NAMES[arm],
                        *(_ci(a[m]["estimate"], a[m]["ci"]) for m in ("FA", "FRR", "BA")),
                    ]
                )
            for rule in RULES:
                a = pts["rules"][rule]
                rows.append(
                    [
                        lv,
                        RULE_NAMES[rule],
                        *(_ci(a[m]["estimate"], a[m]["ci"]) for m in ("FA", "FRR", "BA")),
                    ]
                )
        self.add(
            f"tab:{dom}-op",
            f"Operating points ({dom}), Experiment 1",
            ["level", "decision maker", "FA (failing)", "FR (passing)", "BA"],
            rows,
        )

    def contrasts(self, dom: str) -> None:
        e1 = self.ext[dom]["exp1"]
        op = self.dq[dom]["operating_points"]
        rows = []
        for lv in LEVELS:
            pairs = {(p["X"], p["Y"]): p for p in op[lv]["pairs"]}
            for x, y in PAIRS:
                key = f"{x} - {y}"
                fa_c = e1["e_incorrect"]["by_lambda"][lv]["main"]["contrasts"].get(key)
                if fa_c is not None:
                    fa = _ci(fa_c["estimate"], fa_c["ci"])
                else:  # A1 - B and permuted_B - B: post hoc interval (a dagger in the paper)
                    p = pairs[(x, y)]["dFA"]
                    fa = _neg(_ci(p["estimate"], p["ci"]))
                fr_c = e1["e_correct"]["by_lambda"][lv]["main"]["contrasts"][key]
                ba = pairs[(x, y)]["dBA"]
                rows.append(
                    [
                        lv,
                        f"{ARM_NAMES[x]} -> {ARM_NAMES[y]}",
                        fa,
                        _ci(fr_c["estimate"], fr_c["ci"]),
                        _ci(ba["estimate"], ba["ci"]),
                    ]
                )
        self.add(
            f"tab:{dom}-contrasts",
            f"Contrasts ({dom}), Experiment 1",
            ["level", "X -> Y", "FA_X - FA_Y", "FR_X - FR_Y", "BA_Y - BA_X"],
            rows,
        )

    def rule(self, dom: str, rule: str) -> None:
        op = self.dq[dom]["operating_points"]
        rows = []
        for arm in ARMS_OP:
            row = [ARM_NAMES[arm]]
            for lv in LEVELS:
                hit = [r for r in op[lv]["arm_minus_rule"] if r["arm"] == arm and r["rule"] == rule]
                if len(hit) != 1:
                    raise PaperTablesError(f"arm_minus_rule {dom} {arm} {rule} {lv}")
                row.append(_ci(hit[0]["dBA"]["estimate"], hit[0]["dBA"]["ci"]))
            rows.append(row)
        self.add(
            f"tab:{dom}-rule",
            f"BA of each arm minus BA of rule {rule} ({dom})",
            ["arm", "low", "mid", "high"],
            rows,
        )

    def follow(self, dom: str) -> None:
        vf = self.dq[dom]["verdict_following"]
        rows, sizes = [], {}

        def cell(x: dict[str, Any]) -> Any:
            if x.get("n_boot_undefined"):
                return [x["estimate"], None, None]  # n.e.: point estimate only
            return _ci(x["estimate"], x["ci"])

        for part, key in (
            ("raw gap", "following_gap"),
            ("gap minus No-Verifier", "gap_minus_no_verifier"),
        ):
            for cohort, cname in (("e_incorrect", "Failing"), ("e_correct", "Passing")):
                for arm in ARMS_OP:
                    if key == "gap_minus_no_verifier" and arm == "no_verifier":
                        continue
                    row = [part, cname, ARM_NAMES[arm]]
                    for lv in LEVELS:
                        row.append(cell(vf[cohort][lv][arm][key]))
                    rows.append(row)
        for cohort in ("e_incorrect", "e_correct"):
            sizes[cohort] = {
                lv: [vf[cohort][lv]["A1"]["cells"][c]["n"] for c in ("00", "01", "10", "11")]
                for lv in LEVELS
            }
        self.add(
            f"tab:{dom}-follow",
            f"Verdict-following gaps ({dom})",
            ["part", "cohort", "arm", "low", "mid", "high"],
            rows,
            note="cell sizes (0,0)/(0,1)/(1,0)/(1,1): " + json.dumps(sizes),
        )
        self.tables[f"tab:{dom}-follow"]["cell_sizes"] = sizes

    def exp2(self, dom: str) -> None:
        e2 = self.ext[dom]["exp2"]
        inc, cor = e2["e_incorrect"], e2["e_correct"]
        rows = []
        for arm in ARMS2:
            a = inc["tfr"]["arms"][arm]
            row = [ARM_NAMES[arm], _ci(a["rate"], a["ci"])]
            if dom == "detox":
                g = inc["panel"]["grok"]["arms"][arm]
                row.append(_ci(g["rate"], g["ci"]))
            row.append(inc["non_convergence"]["arms"][arm]["rate"])
            c = cor["corruption_rate"]["arms"][arm]
            row.append(_ci(c["rate"], c["ci"]))
            if dom == "detox":
                g = cor["panel"]["grok"]["arms"][arm]
                row.append(_ci(g["rate"], g["ci"]))
            else:
                row.append(cor["non_convergence"]["arms"][arm]["rate"])
            rows.append(row)
        for x, y in PAIRS2:
            key = f"{x} - {y}"
            row = [f"{ARM_NAMES[x]} - {ARM_NAMES[y]}"]
            c = inc["tfr"]["contrasts"][key]
            row.append(_ci(c["estimate"], c["ci"]))
            if dom == "detox":
                c = inc["panel"]["grok"]["contrasts"][key]
                row.append(_ci(c["estimate"], c["ci"]))
            row.append(None)
            c = cor["corruption_rate"]["contrasts"][key]
            row.append(_ci(c["estimate"], c["ci"]))
            if dom == "detox":
                c = cor["panel"]["grok"]["contrasts"][key]
                row.append(_ci(c["estimate"], c["ci"]))
            else:
                row.append(None)
            rows.append(row)
        cols = (
            [
                "arm",
                "TFR (E)",
                "TFR (Grok)",
                "Non-convergence",
                "Corruption (E)",
                "Corruption (Grok)",
            ]
            if dom == "detox"
            else ["arm", "TFR", "Non-convergence", "Corruption", "Non-convergence (passing)"]
        )
        self.add(f"tab:{dom}-exp2", f"Experiment 2 extension ({dom})", cols, rows)

    def halves(self) -> None:
        primary = list(read_jsonl(paths.resolve(DETOX_PRIMARY_EXP2)))
        fill = list(read_jsonl(paths.resolve(DETOX_FILL_EXP2)))
        assigned = {r["x0_id"]: r["arm"] for r in primary}
        fail: dict[tuple[str, str], list[bool]] = {}
        for r in primary + fill:
            fail.setdefault((assigned[r["x0_id"]], r["arm"]), []).append(not r["task_success"])
        rows = []
        for half, name in (("A1", "Assigned to A1"), ("B", "Assigned to B")):
            a, b = fail[(half, "A1")], fail[(half, "B")]
            rows.append([name, len(a), [sum(a) / len(a), sum(a)], [sum(b) / len(b), sum(b)]])
        a = fail[("A1", "A1")] + fail[("B", "A1")]
        b = fail[("A1", "B")] + fail[("B", "B")]
        rows.append(["All", len(a), [sum(a) / len(a), sum(a)], [sum(b) / len(b), sum(b)]])
        self.add(
            "tab:detox-halves",
            "Decomposition of the paired Experiment 2 contrast by primary assignment",
            ["starts", "n", "TFR_A1 (failures)", "TFR_B (failures)"],
            rows,
            note="DERIVED here from the primary Exp 2 trajectories and the fill run's "
            "trajectories.",
        )

    def ceiling_table(self) -> None:
        c = self.ceiling
        m, s = c["main"], c["sensitivity_unresolved_dropped"]
        cc = m["cumulative_correction"]["arms"]
        tfr2 = self.ext["math"]["exp2"]["e_incorrect"]["tfr"]["arms"]
        normal = sorted(1 - tfr2[a]["rate"] for a in ARMS2)
        o = m["tfr"]["arms"]["oracle_feedback"]
        k = m["tfr"]["contrasts"]["A1 - oracle_feedback"]
        rows = [
            ["TFR under the ceiling", _ci(o["rate"], o["ci"])],
            ["TFR_A1 - TFR_ceiling", _ci(k["estimate"], k["ci"])],
            [
                "Share corrected by round 1 / 2 / 3 / 4",
                [cc[f"t<={t}"]["rate"] for t in (1, 2, 3, 4)],
            ],
            ["Same share after round 4, normal loop (five arms)", [normal[0], normal[-1]]],
            [
                f"Excluded starts with unresolved texts (n = {s['n_starts']}): TFR",
                s["tfr"]["arms"]["oracle_feedback"]["rate"],
            ],
            ["  share corrected within K = 4", s["cumulative_correction"]["arms"]["t<=4"]["rate"]],
        ]
        u = c["unresolved"]
        self.add(
            "tab:math-ceiling",
            "Correction ceiling in mathematics",
            ["quantity", "estimate"],
            rows,
            note=f"unresolved: {u['n_texts']} texts in {u['n_trajectories']} trajectories; "
            f"reasons {json.dumps(u['reasons'])}",
        )

    def build(self) -> dict[str, dict[str, Any]]:
        self.primary()
        self.baserate()
        self.cue()
        self.mde()
        for dom in ("detox", "math"):
            self.gate(dom)
            self.calibration(dom)
            self.primary_full(dom)
            self.operating_points(dom)
            self.contrasts(dom)
            self.follow(dom)
            self.exp2(dom)
        self.h2()
        self.rule("detox", "V2_only")
        self.rule("math", "OR")
        self.halves()
        self.ceiling_table()
        return self.tables


# ---- output ---------------------------------------------------------------------------------


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        return f"{x:.3f}" if abs(x) >= 0.0005 or x == 0 else f"{x:.3g}"
    if isinstance(x, list):
        if len(x) == 3 and x[1] is None and x[2] is None:
            return f"{_fmt(x[0])} (n.e.)"
        if len(x) == 3:
            return f"{_fmt(x[0])} [{_fmt(x[1])}, {_fmt(x[2])}]"
        if len(x) == 2 and isinstance(x[1], int):
            return f"{_fmt(x[0])} ({x[1]})"
        return " / ".join(_fmt(v) for v in x)
    return str(x)


def render(tables: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Paper tables",
        "",
        "Every cell is copied from the regenerated analysis outputs "
        "(see analysis/paper_tables.py);",
        "cells marked DERIVED in a note are computed here. Rates are shown to three decimals.",
        "",
    ]
    for label, t in tables.items():
        lines += [f"## {label}: {t['title']}", ""]
        lines.append("| " + " | ".join(t["columns"]) + " |")
        lines.append("|" + "---|" * len(t["columns"]))
        for r in t["rows"]:
            lines.append("| " + " | ".join(_fmt(c) for c in r) + " |")
        if t.get("note"):
            lines += ["", f"* {t['note']}"]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--out", type=Path, required=True, help="the directory `make tables` wrote into"
    )
    args = ap.parse_args(argv)
    tables = Builder(args.out).build()
    dest = args.out / "paper"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "tables.json").write_text(json.dumps(tables, indent=1) + "\n", encoding="utf-8")
    (dest / "tables.md").write_text(render(tables), encoding="utf-8")
    print(f"[paper_tables] -> {dest / 'tables.md'} ({len(tables)} tables)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
