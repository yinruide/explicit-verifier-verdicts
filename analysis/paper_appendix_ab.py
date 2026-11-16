"""Paper appendices A and B: cue diagnostics and minimum detectable effects (2026-11-08).

Zero cost: no API call, no pod, no label read. Every number is read from runs already on disk,
and every recomputed estimate is cross-checked against the report it duplicates. A failed check
stops the script before anything is written.

A. The five cue diagnostics of paper App. A (``rafr.validity.cue_diagnostics``) for detox and math,
   copied from the files steps 8 and 9 wrote. D_fit is the lambda = 0 baseline and D_gate is the
   first (passed) gate attempt. Each is reported on the primary set (the E-UNSAFE / E-INCORRECT
   subset) and on the secondary set (the whole split). The denominators of every column follow
   from the stored rates (:func:`cue_denominators`). AUROC(|m_i| -> c^(i)) has the verifier's own
   correctness c^(i) as its class, not y*. On the failure-only subset c^(i) is detection (m_i >= 0)
   against a miss, so both classes are present and the AUROC is defined.
B. MDE = (z_.975 + z_.80) * SD(d) / sqrt(n): 80% power, two-sided alpha 0.05, d = the paired
   per-unit differences, SD with ddof 1 (normal approximation):
   - D-bar = (Y^A1 - Y^B)_high - (Y^A1 - Y^B)_low per x0 (and permuted-B - B), both domains;
   - Exp 1 per lambda: FA contrasts (e_incorrect cohort), FRR contrasts (e_correct cohort) and the
     BA contrasts dBA = BA(Y) - BA(X) = (dFA + dFRR) / 2, with SE = sqrt(SE_FA^2 + SE_FRR^2) / 2
     because the two cohorts are independent;
   - Exp 2: the extension's paired TFR contrasts over the 300 starts, and the pilot's unpaired
     150 / 150 A1 vs B;
   - the feasibility table computed before the runs (paper App. B), back-solved for its implied
   SD(D_i). Its
     simulation script (``feas/``) was never committed, so the simulation itself cannot be rerun;
   - the gate's pass-rate checks against the observed gaps.

Every error contrast is X - Y = less information - more information (``configs/ext.yaml``,
the direction of D-bar = A1 - B): a positive value means the added information reduced errors.

Inputs are read with ``analysis.ext_decision_quality.load_cohorts`` (Exp 1) and
``analysis.ext_analysis.gather`` (Exp 2), as the extension tables were, so the accounting is the
same: a parse_fail counts as not accepted.

    python analysis/paper_appendix_ab.py --out runs/paper_appendix_ab_20261108
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import NormalDist
from typing import Any

import numpy as np

from analysis import ext_analysis as W
from analysis import ext_contrasts as X
from analysis import ext_decision_quality as DQ
from rafr.calibration.apply import file_sha256
from rafr.core import paths
from rafr.logging.writer import read_jsonl

OUT_PREFIX = "paper_appendix_ab"
STATUS = (
    "Paper appendix numbers (2026-11-08). Zero cost; existing runs only; no label read. "
    "The MDE depends on the observed variance only, not on the observed effect (not observed "
    "power)."
)
_ND = NormalDist()
Z975, Z95, Z80 = _ND.inv_cdf(0.975), _ND.inv_cdf(0.95), _ND.inv_cdf(0.80)
K_TWO_SIDED = Z975 + Z80  # alpha 0.05 two-sided, power 0.80
K_ONE_SIDED = Z95 + Z80  # alpha 0.05 one-sided (the feasibility table's alpha), power 0.80
DOMAINS = ("detox", "math")
LAMBDAS = ("low", "mid", "high")
CHECK_TOL = 1e-12
INT_TOL = 1e-6  # a stored rate times n must be this close to an integer

RUNS: dict[str, dict[str, Any]] = {
    "detox": {
        "exp1_e_incorrect": ["runs/exp1ext_e_incorrect_20261031T112435Z"],
        "exp1_e_correct": ["runs/exp1ext_e_correct_20261031T112435Z"],
        "exp2_e_incorrect": [
            "runs/exp2ext_e_incorrect_nv_20261031T112435Z",
            "runs/exp2ext_e_incorrect_bpihat_20261031T152908Z",
            "runs/exp2ext_e_incorrect_verdict_20261031T152908Z",
            "runs/exp2ext_e_incorrect_fill_20261031T152908Z",
        ],
        "exp2_e_correct": [
            "runs/exp2ext_e_correct_nv_20261031T112435Z",
            "runs/exp2ext_e_correct_verifier_20261031T152908Z",
        ],
        "ext_report": "runs/ext_analysis_20261101T004507Z/report.json",
        "dq_report": "runs/ext_decision_quality_detox_20261101T074059Z/report.json",
        "did": "runs/analysis_20260928T124804Z/T06_exp1_did.json",
        "by_lambda": "runs/analysis_20260928T124804Z/T06_exp1_delta_by_lambda.json",
        "by_lambda_ci": None,  # the detox tables give the per-lambda Delta_FA without a CI
        "pilot_tfr": "runs/analysis_20260928T124804Z/T08_tfr_contrast.json",
    },
    "math": {
        "exp1_e_incorrect": ["runs/math_exp1ext_e_incorrect_01"],
        "exp1_e_correct": ["runs/math_exp1ext_e_correct_01"],
        "exp2_e_incorrect": [
            "runs/math_exp2ext_nv_i_01",
            "runs/math_exp2ext_e_incorrect_bpihat_01",
            "runs/math_exp2ext_e_incorrect_verdict_01",
            "runs/math_exp2ext_e_incorrect_fill_01",
        ],
        "exp2_e_correct": ["runs/math_exp2ext_nv_c_01", "runs/math_exp2ext_e_correct_verifier_01"],
        "ext_report": "runs/ext_analysis_math_20261106T203556Z/report.json",
        "dq_report": "runs/ext_decision_quality_math_20261106T110046Z/report.json",
        "did": "runs/math_formal_n9000/analysis/T06_exp1_did.json",
        "by_lambda": "runs/math_formal_n9000/analysis/T06_exp1_delta_by_lambda.json",
        "by_lambda_ci": "runs/math_formal_n9000/analysis/T06_exp1_delta_by_lambda_ci.json",
        "pilot_tfr": "runs/math_formal_n9000/analysis/T08_tfr_contrast.json",
    },
}

# (file, key pattern) of the cue diagnostics; {pop} and {lam} are filled per row
CUE_FILES: dict[str, dict[str, tuple[str, str]]] = {
    "detox": {
        "D_fit": ("data/diagnostics/step08_report.json", "cue_diagnostics.{pop}.{lam}"),
        "D_gate": ("data/diagnostics/step09_report.json", "cue_diagnostics.{pop}.{lam}"),
    },
    "math": {
        "D_fit": ("runs/math_formal_n9000/channel_check.json", "cue_by_level.{lam}.{pop}.cues"),
        "D_gate": ("runs/math_formal_n9000/gate_1.json", "cue_diagnostics.{lam}.{pop}"),
    },
}
CUE_POPULATIONS: dict[str, dict[str, dict[str, str]]] = {
    "detox": {
        "D_fit": {"primary": "unsafe", "secondary": "all"},
        "D_gate": {"primary": "unsafe", "secondary": "all"},
    },
    "math": {
        "D_fit": {"primary": "primary_incorrect", "secondary": "secondary_all"},
        "D_gate": {"primary": "incorrect", "secondary": "all"},
    },
}
PRIMARY_SET = {"detox": "E-UNSAFE", "math": "E-INCORRECT"}

# Exp 1 pairs (X, Y): the extension's contrasts plus A1 - B and permuted-B - B
PAIRS: tuple[tuple[str, str], ...] = (
    ("no_verifier", "A1"),
    ("A1", "verdict_only"),
    ("verdict_only", "B"),
    ("B", "B_pihat"),
    ("B_pihat_mismatched", "B_pihat"),
    ("A1", "B"),
    ("permuted_B", "B"),
)

# paper App. B / section 13: power by effect and N (one-sided 0.05)
PREREG_TABLE: dict[float, dict[int, float]] = {
    0.11: {100: 0.33, 200: 0.53, 300: 0.68, 400: 0.79, 600: 0.91},
    0.06: {100: 0.15, 200: 0.21, 300: 0.29, 400: 0.33, 600: 0.45},
    0.03: {100: 0.09, 200: 0.12, 300: 0.13, 400: 0.15, 600: 0.19},
}
PREREG_SOURCES = ("docs/protocol/RAFR_Protocol_v1_3.md", "prereg/analysis_plan.md")
GATE_POWER_CHECK = "runs/step08_power_check_20260927T031158Z/summary.json"
GATE_OBSERVED = {
    "detox": ("data/diagnostics/step09_report.json", "n_e_unsafe"),
    "math": ("runs/math_formal_n9000/gate_1.json", "n_states"),
}
MATH_GATE_SIMULATION_SOURCE = "docs/math/step06_handoff.md"


class AppendixError(RuntimeError):
    pass


@dataclass
class Recorder:
    """Source files (with sha256) and cross-checks of one run."""

    root: Path
    sources: dict[str, str] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)

    def path(self, p: str | Path) -> Path:
        p = Path(p)
        if p.is_absolute():
            return p
        return paths.resolve(p) if self.root == paths.ROOT else self.root / p

    def rel(self, p: str | Path) -> str:
        p = self.path(p)
        if self.root == paths.ROOT:
            return paths.shown(p)
        return p.resolve().relative_to(self.root.resolve()).as_posix()

    def src(self, p: str | Path) -> str:
        r = self.rel(p)
        if r not in self.sources:
            f = self.path(r)
            if not f.is_file():
                raise AppendixError(f"missing input {r}")
            self.sources[r] = file_sha256(f)
        return r

    def json(self, p: str | Path) -> Any:
        r = self.src(p)
        return json.loads(self.path(r).read_text(encoding="utf-8"))

    def check(self, what: str, mine: float, theirs: float) -> None:
        a, b = float(mine), float(theirs)
        self.checks.append({"what": what, "mine": a, "theirs": b, "ok": abs(a - b) <= CHECK_TOL})

    def check_true(self, what: str, ok: bool) -> None:
        self.checks.append({"what": what, "ok": bool(ok)})


# ---- statistics --------------------------------------------------------------------------------


def paired_stats(d: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Mean, SD (ddof 1), SE and the MDEs of paired differences (nan entries dropped)."""
    v = np.asarray(d, dtype=float)
    v = v[~np.isnan(v)]
    n = int(v.size)
    if n < 2:
        raise AppendixError(f"{n} paired differences: an SD needs at least 2")
    sd = float(v.std(ddof=1))
    se = sd / math.sqrt(n)
    return {
        "n": n,
        "estimate": float(v.mean()),
        "sd_d": sd,
        "se": se,
        "mde80_two_sided": K_TWO_SIDED * se,
        "mde80_one_sided": K_ONE_SIDED * se,
        "share_nonzero": float(np.mean(v != 0)),
    }


def ba_contrast(fa: Mapping[str, float], frr: Mapping[str, float]) -> dict[str, float]:
    """dBA = BA(Y) - BA(X) from the X - Y error contrasts of two independent cohorts."""
    se = 0.5 * math.sqrt(fa["se"] ** 2 + frr["se"] ** 2)
    return {
        "estimate": 0.5 * (fa["estimate"] + frr["estimate"]),
        "se": se,
        "mde80_two_sided": K_TWO_SIDED * se,
        "mde80_one_sided": K_ONE_SIDED * se,
    }


def unpaired_stats(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Two independent samples (the pilot's Exp 2): difference of means and its MDE."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    sx, sy = float(x.std(ddof=1)), float(y.std(ddof=1))
    se = math.sqrt(sx**2 / x.size + sy**2 / y.size)
    return {
        "n_x": int(x.size),
        "n_y": int(y.size),
        "rate_x": float(x.mean()),
        "rate_y": float(y.mean()),
        "estimate": float(x.mean() - y.mean()),
        "sd_x": sx,
        "sd_y": sy,
        "se": se,
        "mde80_two_sided": K_TWO_SIDED * se,
        "mde80_one_sided": K_ONE_SIDED * se,
    }


def implied_sigma(delta: float, n: int, power: float) -> float:
    """SD(D_i) at which a one-sided 0.05 z-test of mean delta over n units has ``power``."""
    return delta * math.sqrt(n) / (Z95 + _ND.inv_cdf(power))


def power_one_sided(delta: float, n: int, sigma: float) -> float:
    return _ND.cdf(delta * math.sqrt(n) / sigma - Z95)


def nominal_high_count(n: int, quantile: float) -> int:
    """How many of n distinct values lie at or above their ``quantile`` (numpy's linear rule);
    ties at the cut make the real count larger."""
    v = np.arange(n, dtype=float)
    return int(np.sum(v >= np.quantile(v, quantile)))


def _count(rate: float, n: int, what: str) -> int:
    k = rate * n
    if abs(k - round(k)) > INT_TOL:
        raise AppendixError(f"{what}: {rate} x {n} = {k} is not a count")
    return int(round(k))


def cue_denominators(c: Mapping[str, Any], n: int) -> dict[str, Any]:
    """Each cue column's denominator, from the stored rates of one (split, set, lambda):
    AUROC per verifier on all n (its classes: judged correctly / wrongly), disagreement on n,
    err|agree on the agreeing judgments, err|disagree on the disagreeing ones, high-margin-but-
    wrong on the judgments with |m_i| at or above the quantile (not stored: the no-tie count)."""
    n_dis = _count(c["disagreement_rate"], n, "disagreement")
    out: dict[str, Any] = {"n": n, "n_disagree": n_dis, "n_agree": n - n_dis}
    for v in ("v1", "v2"):
        k = _count(c["reliability"][v], n, f"reliability {v}")
        out[f"n_correct_{v}"] = k
        out[f"n_wrong_{v}"] = n - k
    out["n_high_nominal"] = nominal_high_count(n, float(c["high_margin_quantile"]))
    return out


# ---- A. cue diagnostics ------------------------------------------------------------------------


def _dig(obj: Any, dotted: str) -> Any:
    for k in dotted.split("."):
        obj = obj[k]
    return obj


def cue_rows(rec: Recorder) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for dom, splits in CUE_FILES.items():
        for split, (f, pattern) in splits.items():
            data = rec.json(f)
            for role, pop in CUE_POPULATIONS[dom][split].items():
                for lam in ("none", *LAMBDAS):
                    key = pattern.format(pop=pop, lam=lam)
                    try:
                        c = _dig(data, key)
                    except KeyError:
                        continue  # math's D_fit has no separate 'none' (low is p = 0)
                    n = c.get("n")
                    if n is None:  # math D_fit: n on the parent entry
                        n = _dig(data, key.rsplit(".", 1)[0])["n_scored"]
                    n = int(n)
                    counts = np.asarray(c["joint_margin_hist"]["counts"])
                    rec.check(f"{dom} {split} {role} {lam}: histogram total = n", counts.sum(), n)
                    rec.check(
                        f"{dom} {split} {role} {lam}: err|agree V1 = V2",
                        c["error_rate_given_agreement"]["v1"],
                        c["error_rate_given_agreement"]["v2"],
                    )
                    out.setdefault(dom, []).append(
                        {
                            "split": split,
                            "population": role,
                            "population_key": pop,
                            "lambda": lam,
                            "n": n,
                            "auroc_v1": c["auroc_margin_to_correct"]["v1"],
                            "auroc_v2": c["auroc_margin_to_correct"]["v2"],
                            "disagreement": c["disagreement_rate"],
                            "err_agree": c["error_rate_given_agreement"]["v1"],
                            "err_disagree_v1": c["error_rate_given_disagreement"]["v1"],
                            "err_disagree_v2": c["error_rate_given_disagreement"]["v2"],
                            "hmw_v1": c["high_margin_wrong_share"]["v1"],
                            "hmw_v2": c["high_margin_wrong_share"]["v2"],
                            "high_margin_quantile": c["high_margin_quantile"],
                            "reliability_v1": c["reliability"]["v1"],
                            "reliability_v2": c["reliability"]["v2"],
                            "denominators": cue_denominators(c, n),
                            "joint_hist": c["joint_margin_hist"],
                            "source": f,
                            "key": key,
                        }
                    )
    g = rec.json("data/diagnostics/gate_diagnostics.json")["cue_diagnostics"]
    s9 = rec.json("data/diagnostics/step09_report.json")["cue_diagnostics"]
    rec.check_true("detox gate_diagnostics.json cue == step09_report.json cue", g == s9)
    for dom, rows in out.items():
        idx = {(r["split"], r["population"], r["lambda"]): r for r in rows}
        for (split, role, lam), r in idx.items():
            if lam == "none":
                lo = idx[(split, role, "low")]
                same = all(r[k] == lo[k] for k in r if k not in ("lambda", "key"))
                rec.check_true(f"{dom} {split} {role}: lambda none == low", same)
    return out


# ---- B. MDE ------------------------------------------------------------------------------------


def _row(table: Mapping[str, Any], **match: str) -> list[Any]:
    cols = table["columns"]
    hits = [r for r in table["rows"] if all(r[cols.index(k)] == v for k, v in match.items())]
    if len(hits) != 1:
        raise AppendixError(f"{table.get('id')}: {len(hits)} rows match {match}")
    return hits[0]


def _cell(table: Mapping[str, Any], row: Sequence[Any], col: str) -> Any:
    return row[table["columns"].index(col)]


def exp1_block(rec: Recorder, dom: str) -> dict[str, Any]:
    spec = RUNS[dom]
    d = DQ.domain_inputs(dom)
    rec.src(d["ext_config"])
    cohorts, inputs, _ = DQ.load_cohorts(
        dom,
        [paths.resolve(p) for p in spec["exp1_e_incorrect"]],
        [paths.resolve(p) for p in spec["exp1_e_correct"]],
        original_decisions=d["original_decisions"],
        original_sha256=d["original_sha256"],
        reused_arms=d["reused_arms"],
        prefix=d["prefix"],
    )
    for entries in inputs.values():
        for e in entries:
            rec.src(e["file"])
    inc, cor = cohorts[X.E_INCORRECT], cohorts[X.E_CORRECT]
    rep = rec.json(spec["ext_report"])
    dq = rec.json(spec["dq_report"])
    out: dict[str, Any] = {"n_e_incorrect": len(inc.states), "n_e_correct": len(cor.states)}

    did = rec.json(spec["did"])
    acc = inc.accept
    for role, x in (("primary", "A1"), ("secondary", "permuted_B")):
        s = paired_stats((acc[x]["high"] - acc["B"]["high"]) - (acc[x]["low"] - acc["B"]["low"]))
        row = _row(did, contrast=f"{x} - B", accounting="main")
        rec.check(f"{dom} D-bar {x} - B vs {spec['did']}", s["estimate"], _cell(did, row, "D_bar"))
        lo, hi = _cell(did, row, "ci95_lo"), _cell(did, row, "ci95_hi")
        s |= {"contrast": f"{x} - B", "ci95_reported": [lo, hi], "ci_source": spec["did"]}
        s["se_from_ci"] = (hi - lo) / (2 * Z975)
        out[f"dbar_{role}"] = s

    rows = []
    for lam in inc.lams:
        for x, y in PAIRS:
            name = f"{x} - {y}"
            fa = paired_stats(inc.accept[x][lam] - inc.accept[y][lam])
            frr = paired_stats((1 - cor.accept[x][lam]) - (1 - cor.accept[y][lam]))
            for metric, s, cohort in (("FA", fa, X.E_INCORRECT), ("FRR", frr, X.E_CORRECT)):
                c = rep["exp1"][cohort]["by_lambda"][lam]["main"]["contrasts"].get(name)
                if c is not None:
                    rec.check(f"{dom} {metric} {lam} {name} vs ext report", s["estimate"],
                              c["estimate"])  # fmt: skip
                    s |= {"ci95_reported": c["ci"], "ci_source": spec["ext_report"]}
                elif metric == "FA":
                    # A1 - B and permuted-B - B on e_incorrect: the primary run's tables
                    t = rec.json(spec["by_lambda"])
                    rec.check(f"{dom} FA {lam} {name} vs {spec['by_lambda']}", fa["estimate"],
                              _cell(t, _row(t, contrast=name), lam))  # fmt: skip
                    s["ci95_reported"] = None
                    if spec["by_lambda_ci"]:
                        t = rec.json(spec["by_lambda_ci"])
                        r = _row(t, contrast=name, **{"lambda": lam})
                        s |= {
                            "ci95_reported": [_cell(t, r, "ci95_lo"), _cell(t, r, "ci95_hi")],
                            "ci_source": spec["by_lambda_ci"],
                        }
                else:
                    raise AppendixError(f"{dom} FRR {lam} {name}: not in {spec['ext_report']}")
            ba = ba_contrast(fa, frr)
            p = next(p for p in dq["operating_points"][lam]["pairs"] if (p["X"], p["Y"]) == (x, y))
            rec.check(f"{dom} dBA {lam} {name} vs dq report", ba["estimate"], p["dBA"]["estimate"])
            ba |= {"ci95_reported": p["dBA"]["ci"], "ci_source": spec["dq_report"]}
            rows.append({"lambda": lam, "X": x, "Y": y, "FA": fa, "FRR": frr, "BA": ba})
    out["rows"] = rows
    return out


def exp2_block(rec: Recorder, dom: str) -> dict[str, Any]:
    spec = RUNS[dom]
    ext = W.load_domain_ext(dom)
    plan, _, exp2, _ = W.gather(
        dom,
        ext,
        exp2_e_incorrect=[paths.resolve(p) for p in spec["exp2_e_incorrect"]],
        exp2_e_correct=[paths.resolve(p) for p in spec["exp2_e_correct"]],
    )
    for p in spec["exp2_e_incorrect"] + spec["exp2_e_correct"]:
        rec.src(Path(p) / "trajectories.jsonl")
    original = Path(ext["exp2"]["e_incorrect"]["original_run"]) / "trajectories.jsonl"
    rec.src(original)
    rep = rec.json(spec["ext_report"])
    out: dict[str, Any] = {}
    for cohort in X.COHORTS:
        key = X.EXP2_METRIC[cohort][0]
        failed: dict[str, dict[str, float]] = {}
        for f in exp2[cohort]:
            failed.setdefault(f["arm"], {})[f["x0_id"]] = float(not f["task_success"])
        rows = []
        for x, y in plan["exp2"][cohort]:
            if x not in failed or y not in failed:
                rows.append({"X": x, "Y": y, "status": "not run"})
                continue
            units = sorted(set(failed[x]) & set(failed[y]))
            s = paired_stats([failed[x][u] - failed[y][u] for u in units])
            c = rep["exp2"][cohort][key]["contrasts"][f"{x} - {y}"]
            rec.check(f"{dom} exp2 {cohort} {x} - {y} vs ext report", s["estimate"], c["estimate"])
            rec.check(f"{dom} exp2 {cohort} {x} - {y} n vs ext report", s["n"], c["n_pairs"])
            s |= {
                "ci95_reported": c["ci"],
                "rate_x": float(np.mean([failed[x][u] for u in units])),
                "rate_y": float(np.mean([failed[y][u] for u in units])),
            }
            rows.append({"X": x, "Y": y, "status": "ok", **s})
        out[cohort] = {"metric": key, "rows": rows}

    finals = W.original_finals(read_jsonl(paths.resolve(original)), dom)
    by = {a: [float(not f["task_success"]) for f in finals if f["arm"] == a] for a in ("A1", "B")}
    s = unpaired_stats(by["A1"], by["B"])
    t = rec.json(spec["pilot_tfr"])
    row = _row(t, rule="main")
    rec.check(f"{dom} pilot TFR A1 - B vs {spec['pilot_tfr']}", s["estimate"],
              _cell(t, row, "estimate"))  # fmt: skip
    s |= {
        "ci95_reported": [_cell(t, row, "ci95_lo"), _cell(t, row, "ci95_hi")],
        "ci_source": spec["pilot_tfr"],
        "source": rec.rel(original),
    }
    out["pilot_unpaired"] = s
    return out


def _doc_source(rec: Recorder, f: str) -> str:
    """Record a design document's sha256 when it is present. The design documents are not part
    of the public release; the values this script takes from them are written out above."""
    if not paths.resolve(f).is_file():
        return f"{f} (not in the public release)"
    return rec.src(f)


def prereg_block(rec: Recorder) -> dict[str, Any]:
    for f in PREREG_SOURCES:
        _doc_source(rec, f)
    implied = {
        str(dl): {str(n): implied_sigma(dl, n, pw) for n, pw in row.items()}
        for dl, row in PREREG_TABLE.items()
    }
    return {
        "sources": list(PREREG_SOURCES),
        "alpha": "one-sided 0.05 (4000 simulations, random intercept sigma 1.0, compound symmetry)",
        "script": "feas/ (named in III.1.3) is not in the repository and was never committed",
        "table": {str(k): {str(n): v for n, v in r.items()} for k, r in PREREG_TABLE.items()},
        "implied_sigma_D": implied,
        "implied_sigma_D_row_011_mean": float(np.mean(list(implied["0.11"].values()))),
    }


def gate_block(rec: Recorder) -> dict[str, Any]:
    pc = rec.json(GATE_POWER_CHECK)
    out: dict[str, Any] = {
        "detox_step08_power_check": {
            "source": GATE_POWER_CHECK,
            "n_seeds": pc["n_seeds"],
            "n_e_unsafe": pc["n_e_unsafe"],
            "joint_pass_share": {k: v["share"] for k, v in pc["joint_pass_share"].items()},
            "gap_sd": {v: {g: x["sd"] for g, x in gaps.items()} for v, gaps in pc["gaps"].items()},
        },
        "math_step02_simulation_source": _doc_source(rec, MATH_GATE_SIMULATION_SOURCE),
    }
    for dom, (f, n_key) in GATE_OBSERVED.items():
        g = rec.json(f)
        out[f"{dom}_observed"] = {
            "source": f,
            "n": g[n_key],
            "passed": g["passed"],
            "gaps": g["gaps"],
            "gap_ci": g["ci"]["gaps"],
            "gap_se_from_ci": {
                v: {k: (c[1] - c[0]) / (2 * Z975) for k, c in gaps.items()}
                for v, gaps in g["ci"]["gaps"].items()
            },
        }
    return out


def analyze(root: Path | None = None) -> dict[str, Any]:
    rec = Recorder(Path(root or paths.ROOT))
    report: dict[str, Any] = {
        "status": STATUS,
        "formula": (
            "MDE = (z_.975 + z_.80) * SD(d) / sqrt(n); BA: SE = sqrt(SE_FA^2 + SE_FRR^2) / 2"
        ),
        "constants": {
            "z975": Z975,
            "z95": Z95,
            "z80": Z80,
            "K_two_sided": K_TWO_SIDED,
            "K_one_sided": K_ONE_SIDED,
        },
        "cue": cue_rows(rec),
        "exp1": {d: exp1_block(rec, d) for d in DOMAINS},
        "exp2": {d: exp2_block(rec, d) for d in DOMAINS},
        "prereg": prereg_block(rec),
        "gate": gate_block(rec),
    }
    report["checks"] = rec.checks
    report["sources"] = dict(sorted(rec.sources.items()))
    bad = [c for c in rec.checks if not c["ok"]]
    if bad:
        raise AppendixError(f"{len(bad)} cross-checks failed: {bad[:5]}")
    return report


# ---- rendering ---------------------------------------------------------------------------------

ARM = {
    "no_verifier": "NV",
    "A1": "A₁",
    "verdict_only": "VO",
    "B": "B",
    "B_pihat": "B+π̂",
    "B_pihat_mismatched": "B+π̂(错配)",
    "permuted_B": "perm-B",
}


def _f(x: Any, nd: int = 3, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return s.replace("-", "−")


def _ci(c: Sequence[float] | None) -> str:
    return "—" if not c else f"[{_f(c[0], 3, True)}, {_f(c[1], 3, True)}]"


def _md(x: Any) -> str:
    return str(x).replace("|", "\\|")  # a literal | (as in |m_i|) would split the cell


def _table(head: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(map(_md, head)) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return out + ["| " + " | ".join(map(_md, r)) + " |" for r in rows]


def _pair(x: str, y: str) -> str:
    return f"{ARM[x]} − {ARM[y]}"


def _set(role: str) -> str:
    return "主" if role == "primary" else "辅"


CUE_INTRO = (
    "## 附录 A：cue diagnostics（II.6.2 五项）",
    "",
    "五项按 protocol II.6.2 与 `src/rafr/validity/cue_diagnostics.py`：",
    "",
    "1. AUROC(|m_i| → c^(i))：|m_i| 区分 verifier i 判对与判错。"
    "**类别是 verifier 自己的正确性 c^(i) = 1[y_i = y*]，不是 y\\***。",
    "2. disagreement 率 P(y₁ ≠ y₂)。",
    "3. agreement / disagreement 条件下的 verifier 错误率。"
    "两者一致时同对同错，V₁、V₂ 的值相同（逐行核对过），只列一列。",
    "4. high-margin-but-wrong 比例：|m_i| 处于该集合内 |m_i| 的 0.75 分位"
    "及以上的判定中，判错的比例（逐 λ、逐 verifier 现取切点）。",
    "5. (m₁, m₂) 联合分布：10 × 10 二维直方图。原文件只存分箱计数，"
    "没有标量；计数与边界复制在 `report.json` 的 `cue.<领域>[*].joint_hist`，"
    "原位置见每块下方的「键」。",
    "",
    "### 计算集合",
    "",
    "每一行的五项都在**同一个集合**上算：主集合是该切分的 E-UNSAFE（detox）"
    "/ E-INCORRECT（math）子集，辅集合是该切分全体（II.6.2，v1.0）。"
    "没有「AUROC 在全体、其余在失败子集」这种混用。",
    "",
    "失败子集上 y* 只有一类，但 AUROC 的类别是 c^(i)：在失败子集上，"
    "c^(i) = 1 等价于 verifier 报出（m_i ≥ 0），c^(i) = 0 是漏判（m_i < 0），"
    "两类都有，所以 AUROC 有定义。这时它衡量的是报出的 |m_i| 是否大于漏判的。"
    "只有某个 verifier 在某档全部报出或全部漏判时才算不出（记为 —），"
    "本表没有这种格子。全体集合上，c^(i) = 1 同时包含 SAFE 判对（m_i < 0）"
    "与 UNSAFE 判对（m_i ≥ 0）。",
    "",
    "各列的分母（n 为该行集合的规模；D_fit 为 λ = 0 基线，D_gate 为第一次 gate）：",
    "",
)
CUE_COLUMNS = (
    (
        "AUROC V_i",
        "该行集合全部 n 条",
        "n；类别大小为判对 n·R_i、判错 n·(1 − R_i)，R_i 是该集合上 V_i 的"
        "正确率（失败子集上即检出率）",
    ),
    ("disagree", "该行集合全部 n 条", "n"),
    ("err│agree", "其中 y₁ = y₂ 的", "n_agree = n·(1 − disagree)"),
    ("err│dis V_i", "其中 y₁ ≠ y₂ 的", "n_dis = n·disagree"),
    (
        "HMW V_i",
        "其中 |m_i| ≥ 该集合 |m_i| 的 0.75 分位的",
        "n_high：原文件未存；分数无并列时为下表的名义值，有并列时更多",
    ),
    ("联合分布", "该行集合全部 n 条", "n（直方图计数之和 = n，逐行核对过）"),
)
CUE_HEAD = (
    "切分", "集合", "λ", "n", "AUROC V₁", "AUROC V₂", "disagree", "err│agree",
    "err│dis V₁", "err│dis V₂", "HMW V₁", "HMW V₂",
)  # fmt: skip
DENOM_HEAD = (
    "切分", "集合", "λ", "n", "AUROC V₁ 判对 / 判错", "AUROC V₂ 判对 / 判错",
    "disagree", "err│agree", "err│dis（V₁、V₂ 同）", "HMW 名义 n_high（V₁、V₂ 同）",
)  # fmt: skip


def _render_cue(r: Mapping[str, Any]) -> list[str]:
    lines = list(CUE_INTRO)
    lines += _table(("列", "计算集合", "分母"), CUE_COLUMNS)
    seen = [
        f"{dom} {x['split']} {_set(x['population'])}"
        for dom in DOMAINS
        for x in r["cue"][dom]
        if x["lambda"] == "none"
    ]
    lines += [
        "",
        "λ_low 的 p = 0，即未扰动。detox 的两个文件与 math 的 gate_1 另存一档 `none`，"
        f"与 `low` 逐项相同（核对：{'、'.join(seen)}），表中不重复列出。",
        "",
    ]
    for i, dom in enumerate(DOMAINS, start=1):
        rows = [x for x in r["cue"][dom] if x["lambda"] != "none"]
        lines += [f"### A.{i} {dom}（主集合 = {PRIMARY_SET[dom]} 子集）", ""]
        body = [
            [
                x["split"], _set(x["population"]), x["lambda"], x["n"],
                _f(x["auroc_v1"]), _f(x["auroc_v2"]), _f(x["disagreement"]),
                _f(x["err_agree"]), _f(x["err_disagree_v1"]), _f(x["err_disagree_v2"]),
                _f(x["hmw_v1"]), _f(x["hmw_v2"]),
            ]
            for x in rows
        ]  # fmt: skip
        lines += _table(CUE_HEAD, body)
        lines += ["", "各列分母：", ""]
        body = []
        for x in rows:
            d = x["denominators"]
            body.append([
                x["split"], _set(x["population"]), x["lambda"], d["n"],
                f"{d['n_correct_v1']} / {d['n_wrong_v1']}",
                f"{d['n_correct_v2']} / {d['n_wrong_v2']}",
                d["n"], d["n_agree"], d["n_disagree"], d["n_high_nominal"],
            ])  # fmt: skip
        lines += _table(DENOM_HEAD, body)
        lines += [""]
        for split in ("D_fit", "D_gate"):
            x = next(x for x in rows if x["split"] == split)
            lines += [
                f"- {split}：`{x['source']}`，键 `{x['key']}`（换 λ / 集合同一路径；"
                "直方图在其下的 `joint_margin_hist`）"
            ]
        lines += [""]
    return lines


def render(r: Mapping[str, Any]) -> str:
    k1, k2 = r["constants"]["K_one_sided"], r["constants"]["K_two_sided"]
    n_bad = sum(not c["ok"] for c in r["checks"])
    det, mth = r["exp1"]["detox"]["dbar_primary"], r["exp1"]["math"]["dbar_primary"]
    lines = [
        "# 论文附录 A / B 取数（零花费，不调 API，不读标签）",
        "",
        "生成：`python analysis/paper_appendix_ab.py --out <目录>`；全部数字在同目录 "
        "`report.json`。读入沿用 `analysis/ext_decision_quality.load_cohorts` 与 "
        "`analysis/ext_analysis.gather`，与已出的表同一口径（parse_fail 记为未接受）。"
        f"交叉核对 {len(r['checks'])} 项（重算的点估计、配对 n 与已出的表逐位比较，"
        f"另有直方图计数与 n、none 与 low 的一致），失败 {n_bad} 项。",
        "",
    ]
    lines += _render_cue(r)
    lines += [
        "## 附录 B：最小可检测效应（80% power，双侧 α = 0.05）",
        "",
        f"MDE = (z₀.₉₇₅ + z₀.₈₀) · SD(d) / √n = {k2:.4f} · SE，d 为逐单位配对差，"
        f"SD 取观测值（ddof = 1）。单侧 α = 0.05 的系数为 {k1:.4f}，只用于与预注册表对照。"
        "正态近似；按 t(299) 计算 MDE 约大 0.3%。",
        "",
        "符号：误差类指标的对比一律 X − Y = 信息少 − 信息多，正值表示多给的信息减少了"
        "错误（`configs/ext.yaml` 的约定，与 D̄ = A₁ − B 同向）。"
        "ΔBA = BA(Y) − BA(X) = (ΔFA + ΔFRR)/2，正值同样表示信息多的一侧更好。"
        "SE 与已出表的 bootstrap 95% CI 半宽 / 1.96 吻合（例：D̄ 的 SE detox "
        f"{det['se']:.4f} 对 {det['se_from_ci']:.4f}，math {mth['se']:.4f} 对 "
        f"{mth['se_from_ci']:.4f}）。",
        "",
        "MDE 只依赖方差，不依赖观测到的效应，所以它不是 observed power。它回答的是："
        "N = 300、按观测到的方差，80% 功效能检出多大的差。",
        "",
        "### B.1 D̄（Exp 1 主交互，(Y^A₁ − Y^B)_high − (Y^A₁ − Y^B)_low，按 x₀）",
        "",
    ]
    body = []
    for dom in DOMAINS:
        for k, lab in (("dbar_primary", "A₁ − B（主）"), ("dbar_secondary", "perm-B − B（次）")):
            d = r["exp1"][dom][k]
            body.append([
                dom, lab, d["n"], _f(d["estimate"], 4, True), _ci(d["ci95_reported"]),
                _f(d["sd_d"], 4), _f(d["se"], 4), _f(d["share_nonzero"]),
                _f(d["mde80_two_sided"], 4), _f(d["mde80_one_sided"], 4),
            ])  # fmt: skip
    head = (
        "领域", "对比", "n", "D̄", "95% CI（已报）", "SD(D_i)", "SE", "D_i ≠ 0 的比例",
        "MDE 双侧", "MDE 单侧",
    )  # fmt: skip
    lines += _table(head, body)
    lines += [
        "",
        "来源：detox `runs/exp1_20260928T000021Z/decisions.jsonl`，CI 取自 "
        "`runs/analysis_20260928T124804Z/T06_exp1_did.json`；math "
        "`runs/math_formal_n9000/exp1/decisions.jsonl`，CI 取自 "
        "`runs/math_formal_n9000/analysis/T06_exp1_did.json`。",
        "",
    ]
    lines += _render_prereg(r)
    lines += _render_exp1(r)
    lines += _render_exp2(r)
    lines += ["## 来源文件（sha256）", ""]
    lines += _table(("文件", "sha256"), [[f"`{k}`", f"`{v}`"] for k, v in r["sources"].items()])
    return "\n".join([*lines, ""])


PREREG_INTRO = (
    "### B.2 预注册的 power 计算与观测值对照",
    "",
    "**(a) Exp 1 可行性计算**（protocol III.1.3 / `prereg/analysis_plan.md` 第 13 节）："
    "4000 次模拟，随机截距 σ = 1.0，**单侧** α = 0.05，配对差中差，compound symmetry；"
    "正文写「待实测后重跑」。正文说脚本存于 repo 的 `feas/`，但这个目录不在 repo 里，"
    "git 历史里也从未提交过，所以预注册的功效模拟无法复现。下面按正态近似从表反推"
    "它隐含的 SD(D_i)：σ_D = Δ·√N / (z₀.₉₅ + Φ⁻¹(power))。",
    "",
)
PREREG_AFTER = (
    "正文当时已写「模拟采用 compound symmetry，真实相关性更高时估计偏保守」："
    "实测只有约四分之一的 x₀ 的 D_i ≠ 0，同一 x₀ 在两臂、两档下的四个决策大多相同，"
    "相关性高于模拟设定，差值方差相应更小。（λ_high 与 λ_low 的 prompt 逐字节相同的"
    "比例是 0 / 900，所以 C.24 的决策复用与这里无关。）功效一列按单侧 0.05、"
    "正态近似计算，与预注册表同口径。"
)
GATE_PREREG_DETOX_STEP2 = (
    "第 2 步模拟（C.19 / C.21）：δ_R = 0.10 下单次联合通过率约 0.82，0.15 下约 0.33"
)
GATE_PREREG_DETOX_SD = "正文：D_gate 的 E-UNSAFE 子集约 107 条时，单个 gap 点估计的抽样 SD 约 0.05"
GATE_PREREG_MATH = (
    "第 2 步第二轮模拟：δ_R = 0.10 下单次联合通过率 0.778，0.15 下 0.065"
    "（剔除重叠后 0.929 / 0.331）"
)
EXT_NO_PREREG = (
    "**(c) 扩展实验**（No Verifier / Verdict-only / B+π̂ / 错配）没有预注册的 power 计算，"
    "全部结果标为探索性。第 63 组复盘时事后算过 detox Exp 1 各对比的 MDE"
    "（`runs/ext_diag_20261101/diag_exp1.json`：E-incorrect 0.054–0.089、"
    "E-correct 0.038–0.056），与 B.3 的 detox FA / FRR 列一致，"
    "差别只在第 63 组的 SD 用 ddof = 0。"
)


def _render_prereg(r: Mapping[str, Any]) -> list[str]:
    pr, g = r["prereg"], r["gate"]
    k1, k2 = r["constants"]["K_one_sided"], r["constants"]["K_two_sided"]
    sig = pr["implied_sigma_D_row_011_mean"]
    n = 300
    lines = list(PREREG_INTRO)
    ns = ("100", "200", "300", "400", "600")
    body = []
    for dl, row in pr["table"].items():
        imp = pr["implied_sigma_D"][dl]
        body.append([dl, *[_f(row[k], 2) for k in ns], " / ".join(_f(imp[k], 2) for k in ns)])
    lines += _table(("Δ_high − Δ_low", *[f"N={k}" for k in ns], "隐含 σ_D（各 N）"), body)
    lines += [
        "",
        f"Δ = 0.11 一行最稳定，隐含 σ_D ≈ {sig:.3f}。0.06 与 0.03 两行在 0.90–1.01 之间，"
        "功效接近 α 时反推不稳。对照：",
        "",
    ]
    pre_1, pre_2 = k1 * sig / math.sqrt(n), k2 * sig / math.sqrt(n)
    body = [["预注册表（隐含）", _f(sig), _f(pre_1), _f(pre_2), "0.68（表）", "0.29（表）",
             "0.13（表）"]]  # fmt: skip
    for dom in DOMAINS:
        d = r["exp1"][dom]["dbar_primary"]
        s = d["sd_d"]
        body.append([
            f"观测 {dom}", _f(s), _f(d["mde80_one_sided"]), _f(d["mde80_two_sided"]),
            *[_f(power_one_sided(dl, n, s), 2) for dl in (0.11, 0.06, 0.03)],
        ])  # fmt: skip
    head = (
        "", "σ_D", "MDE（N=300，单侧 0.05）", "MDE（N=300，双侧 0.05）",
        "Δ=0.11 的功效", "Δ=0.06 的功效", "Δ=0.03 的功效",
    )  # fmt: skip
    lines += _table(head, body)
    ratio = [r["exp1"][d]["dbar_primary"]["sd_d"] / sig for d in DOMAINS]
    obs_1 = [r["exp1"][d]["dbar_primary"]["mde80_one_sided"] for d in DOMAINS]
    obs_2 = [r["exp1"][d]["dbar_primary"]["mde80_two_sided"] for d in DOMAINS]
    shrink = (
        f"观测到的 SD(D_i) 是预注册隐含值的 {min(ratio):.2f}–{max(ratio):.2f} 倍，"
        f"N = 300 的 MDE 相应从 {pre_1:.3f}（单侧）/ {pre_2:.3f}（双侧）降到 "
        f"{min(obs_1):.3f}–{max(obs_1):.3f} / {min(obs_2):.3f}–{max(obs_2):.3f}。"
    )
    lines += [
        "",
        shrink + PREREG_AFTER,
        "",
        "**(b) gate 的 ladder check**（II.6.1：按点估计判定，4 个相邻 gap 都要 ≥ δ_R = 0.10）：",
        "",
    ]
    pc = g["detox_step08_power_check"]
    sds = [s for v in pc["gap_sd"].values() for s in v.values()]
    js = pc["joint_pass_share"]
    proto = "`docs/protocol/RAFR_Protocol_v1_3.md` III.1.3"
    step8 = (
        f"第 8 步核对（D_fit E-UNSAFE {pc['n_e_unsafe']} 条，{pc['n_seeds']} 个 seed）："
        f"联合通过率 δ_R 0.05 / 0.10 / 0.15 = {js['0.05']} / {js['0.1']} / {js['0.15']}；"
        f"gap 的模拟 SD {_f(min(sds))}–{_f(max(sds))}"
    )
    rows = [
        ["detox", GATE_PREREG_DETOX_STEP2, proto],
        ["detox", step8, f"`{pc['source']}`"],
        ["detox", GATE_PREREG_DETOX_SD, proto],
        ["math", GATE_PREREG_MATH, f"`{g['math_step02_simulation_source']}`"],
    ]
    lines += _table(("领域", "预注册 / 事前计算", "来源"), rows)
    lines += [
        "",
        "观测（第一次 gate，两个领域都通过；SE 由已报 bootstrap 95% CI 的宽度 / 3.92 换算）：",
        "",
    ]
    for dom in DOMAINS:
        o = g[f"{dom}_observed"]
        txt = "；".join(
            f"{v.upper()} {k} {_f(o['gaps'][v][k])}（SE≈{_f(o['gap_se_from_ci'][v][k])}）"
            for v in ("v1", "v2")
            for k in ("low-mid", "mid-high")
        )
        lines += [f"- {dom}（D_gate {PRIMARY_SET[dom]} {o['n']} 条，`{o['source']}`）：{txt}"]
    se = {
        dom: [s for v in g[f"{dom}_observed"]["gap_se_from_ci"].values() for s in v.values()]
        for dom in DOMAINS
    }
    lines += [
        "",
        f"预注册的「SD 约 0.05」与 detox 实测（{min(se['detox']):.3f}–{max(se['detox']):.3f}，"
        f"n = {g['detox_observed']['n']}）一致；math 的 n 是 {g['math_observed']['n']}，"
        f"SE {min(se['math']):.3f}–{max(se['math']):.3f}。",
        "",
        EXT_NO_PREREG,
        "",
    ]
    return lines


EXP1_INTRO = (
    "### B.3 Exp 1 各 λ 档：FA、FRR 与 BA 对比",
    "",
    "FA 在 E-incorrect cohort（n = 300；A₁ / B / perm-B 是原预注册 run 的决策），"
    "FRR 在 E-correct cohort（n = 300），两个 cohort 互相独立，"
    "ΔBA 的 SE = √(SE_FA² + SE_FRR²)/2。「CI」是已出表的 95% percentile bootstrap，"
    "供对照；detox 的 A₁ − B、perm-B − B 在 FA 上没有逐 λ 的 CI 表"
    "（预注册表只给点估计）。",
    "",
)
EXP1_SOURCES = (
    "来源：detox `runs/exp1_20260928T000021Z/decisions.jsonl`（A₁、B、perm-B）、"
    "`runs/exp1ext_e_incorrect_20261031T112435Z/decisions.jsonl`、"
    "`runs/exp1ext_e_correct_20261031T112435Z/decisions.jsonl`；math "
    "`runs/math_formal_n9000/exp1/decisions.jsonl`、"
    "`runs/math_exp1ext_e_incorrect_01/decisions.jsonl`、"
    "`runs/math_exp1ext_e_correct_01/decisions.jsonl`。点估计与 CI 对照：FA / FRR 为 "
    "`runs/ext_analysis_20261101T004507Z/report.json`、"
    "`runs/ext_analysis_math_20261106T203556Z/report.json`；ΔBA 为 "
    "`runs/ext_decision_quality_detox_20261101T074059Z/report.json`、"
    "`runs/ext_decision_quality_math_20261106T110046Z/report.json`；"
    "原 run 的 A₁ − B、perm-B − B 为 `T06_exp1_delta_by_lambda*.json`。"
)
EXP1_HEAD = (
    "λ", "X − Y", "ΔFA", "SD", "MDE", "ΔFRR", "SD", "MDE", "ΔBA", "SE", "MDE", "ΔBA 的 CI",
)  # fmt: skip


def _render_exp1(r: Mapping[str, Any]) -> list[str]:
    lines = list(EXP1_INTRO)
    for dom in DOMAINS:
        e = r["exp1"][dom]
        body = []
        for row in e["rows"]:
            fa, fr, ba = row["FA"], row["FRR"], row["BA"]
            body.append([
                row["lambda"], _pair(row["X"], row["Y"]),
                _f(fa["estimate"], 3, True), _f(fa["sd_d"]), _f(fa["mde80_two_sided"]),
                _f(fr["estimate"], 3, True), _f(fr["sd_d"]), _f(fr["mde80_two_sided"]),
                _f(ba["estimate"], 3, True), _f(ba["se"], 4), _f(ba["mde80_two_sided"]),
                _ci(ba["ci95_reported"]),
            ])  # fmt: skip
        lines += [f"**{dom}**", ""]
        lines += _table(EXP1_HEAD, body)
        spans = {}
        for key in ("FA", "FRR", "BA"):
            v = [x[key]["mde80_two_sided"] for x in e["rows"]]
            spans[key] = f"{_f(min(v))}–{_f(max(v))}"
        lines += [
            "",
            f"MDE 范围：FA {spans['FA']}，FRR {spans['FRR']}，BA {spans['BA']}。",
            "",
        ]
    return [*lines, EXP1_SOURCES, ""]


EXP2_INTRO = (
    "### B.4 Exp 2：TFR 配对差",
    "",
    "扩展实验的全配对：E-incorrect 的 300 个起点（D_Exp1）上每个 arm 都跑了，"
    "A₁ / B 由原 run 加 P2 补齐拼成，差值按起点配对。错配 arm 只在 Exp 1 跑，记为 not run。",
    "",
)
EXP2_SOURCES = (
    "来源：detox `runs/exp2ext_e_incorrect_{nv_20261031T112435Z,bpihat_20261031T152908Z,"
    "verdict_20261031T152908Z,fill_20261031T152908Z}/trajectories.jsonl` 与原 run "
    "`runs/exp2_20260928T033025Z/trajectories.jsonl`；math "
    "`runs/math_exp2ext_{nv_i_01,e_incorrect_bpihat_01,e_incorrect_verdict_01,"
    "e_incorrect_fill_01}/trajectories.jsonl` 与原 run "
    "`runs/math_formal_n9000/exp2/trajectories.jsonl`。点估计、配对 n 与 CI 对照 "
    "`runs/ext_analysis_20261101T004507Z/report.json`、"
    "`runs/ext_analysis_math_20261106T203556Z/report.json`；原 pilot 对照 "
    "`runs/analysis_20260928T124804Z/T08_tfr_contrast.json`、"
    "`runs/math_formal_n9000/analysis/T08_tfr_contrast.json`。"
)
CORRUPTION_NOTE = (
    "附（不是 TFR，备查）：E-correct cohort 的改错率（corruption rate）配对差。"
    "基础率约 0–4%，正态近似的 MDE 在这里不可靠，只供参考。"
)
EXP2_HEAD = (
    "X − Y", "n", "TFR(X)", "TFR(Y)", "ΔTFR", "95% CI（已报）", "SD(d)", "d ≠ 0 的比例", "MDE",
)  # fmt: skip
CORRUPTION_HEAD = (
    "领域", "X − Y", "n", "改错率(X)", "改错率(Y)", "Δ", "95% CI（已报）", "SD(d)", "MDE",
)  # fmt: skip


def _render_exp2(r: Mapping[str, Any]) -> list[str]:
    k2 = r["constants"]["K_two_sided"]
    lines = list(EXP2_INTRO)
    for dom in DOMAINS:
        x = r["exp2"][dom]
        body = []
        for row in x[X.E_INCORRECT]["rows"]:
            if row["status"] != "ok":
                body.append([_pair(row["X"], row["Y"]), "not run", *["—"] * 7])
                continue
            body.append([
                _pair(row["X"], row["Y"]), row["n"], _f(row["rate_x"]), _f(row["rate_y"]),
                _f(row["estimate"], 3, True), _ci(row["ci95_reported"]), _f(row["sd_d"]),
                _f(row["share_nonzero"]), _f(row["mde80_two_sided"]),
            ])  # fmt: skip
        p = x["pilot_unpaired"]
        body.append([
            "A₁ − B（原 pilot，非配对）", f"{p['n_x']} / {p['n_y']}", _f(p["rate_x"]),
            _f(p["rate_y"]), _f(p["estimate"], 3, True), _ci(p["ci95_reported"]),
            f"{_f(p['sd_x'])} / {_f(p['sd_y'])}", "—", _f(p["mde80_two_sided"]),
        ])  # fmt: skip
        lines += [f"**{dom}**", ""]
        lines += _table(EXP2_HEAD, body)
        lines += [""]
    lines += [
        f"原 pilot 一行按两独立样本算：MDE = {k2:.4f}·√(s_A²/n_A + s_B²/n_B)，"
        "CI 取自 `T08_tfr_contrast.json`。",
        "",
        EXP2_SOURCES,
        "",
        CORRUPTION_NOTE,
        "",
    ]
    body = []
    for dom in DOMAINS:
        for row in r["exp2"][dom][X.E_CORRECT]["rows"]:
            if row["status"] == "ok":
                body.append([
                    dom, _pair(row["X"], row["Y"]), row["n"], _f(row["rate_x"]),
                    _f(row["rate_y"]), _f(row["estimate"], 3, True), _ci(row["ci95_reported"]),
                    _f(row["sd_d"]), _f(row["mde80_two_sided"]),
                ])  # fmt: skip
    lines += _table(CORRUPTION_HEAD, body)
    return [*lines, ""]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out",
        type=Path,
        help="output directory (default: runs/paper_appendix_ab_<UTC timestamp>)",
    )
    args = ap.parse_args(argv)
    report = analyze()
    if args.out is None:
        out = paths.run_dir(f"{OUT_PREFIX}_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}")
    else:
        out = paths.resolve(args.out)
        out.mkdir(parents=True, exist_ok=True)
    files = {
        "report.json": json.dumps(report, indent=1, ensure_ascii=False) + "\n",
        "tables.md": render(report),
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"[paper_appendix_ab] -> {(out / name).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
