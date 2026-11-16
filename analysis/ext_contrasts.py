"""Extension contrasts (post-pilot; the list the authors set on 2026-10-31, reply to stop point B).

EXPLORATORY: every extension result is exploratory, the intervals are not corrected for multiple
comparisons, and every cell is reported (cohort x lambda x accounting x contrast, and every arm's
rate with its n).

Every contrast is written "less information - more information" (``configs/ext.yaml: analysis``),
the direction of the original D-bar = A1 - B: a positive value means the added information
reduced errors.

    every experiment and cohort:  No Verifier - A1,  A1 - Verdict-only,  Verdict-only - B,
                                  B - (B + pi-hat),  (B + pi-hat mismatched) - (B + pi-hat)
    Exp 1, E-correct, also:       A1 - B,  permuted-B - B  (on the false reject rate)
    Exp 1, E-incorrect:           A1 - B and permuted-B - B are the preregistered results; they
                                  are cited from the preregistered tables (sha256-checked), not
                                  recomputed
    Exp 2, both cohorts, also:    A1 - B over the full pairing (data once P2's fill has run)

**Exp 1**, per cohort (never merged) and per lambda level:

* ``e_incorrect`` (D_Exp1, E-UNSAFE): the error is a false accept, outcome 1[accepted];
* ``e_correct`` (the E-correct cohort, E-SAFE): the error is a false reject, outcome
  1[not accepted].

Each contrast is the paired mean over states of error_X - error_Y. In ``e_incorrect`` the original
run's A1 / B decisions (sha256-checked) stand next to the extension's. Accountings (II.8.2): the
main one counts a parse_fail as not accepted (in ``e_correct``: a false reject); the sensitivities
are ``parse_fail_as_accept`` and ``complete_case`` (states where either arm parse-failed dropped).

**Exp 2**, per cohort, paired over the starts (x0) on which both arms of a contrast ran; every
contrast and every arm reports its n:

* ``e_incorrect``: TFR = P(task failure at x_T); the original 150 / 150 trajectories enter as the
  A1 / B trajectories of their starts, the fill gives each start its other arm;
* ``e_correct``: the corruption rate (the authors' name for it; not TFR) = P(task failure at x_T) on
  starts E judged correct, i.e. the share of correct starts the loop turned wrong.

The mismatch arm does not exist in Exp 2: its contrast is reported as not run.

**Exp 2 panel sensitivity** (2026-10-31; III.1.2 as the original Exp 2's T081): each
panel member (detox: Grok 4.6, ``panel_<member>.jsonl`` in every run directory, the original
run's sha256-checked) re-judges x_T and is read on its own as the judge of the same metric, with
the same contrasts and pairing; a trajectory the member gave no verdict on leaves that member's
numbers only (``exclude_and_report``), counted per arm, as are trajectories with no panel row
yet. Each contrast says whether its direction matches E's.

Inputs: extension run directories only, i.e. ``runs/<run_id_prefix>_*`` (the experiment's prefix
in configs/ext.yaml) carrying an ``ext_meta.json`` of the cohort and not a smoke run; any other
directory is refused. Besides them the analysis reads only the preregistered files configs/ext.yaml
names with their sha256 (the original A1 / B decisions and trajectories, the cited tables).

Intervals follow III.1.2: a 95% percentile bootstrap, B = 10000, seed from the master seed with
the component ``ext_analysis_boot``; the resampling unit is the x0 (one state per x0 in Exp 1, one
start per x0 in Exp 2), resampled jointly for the two arms of a pair because they share the x0.

The analysis is invariant to row order (states are sorted before resampling); ``--fake`` runs it
end to end on shuffled fake data (FAKE marker, FAKE- run ids, FAKE_ file names) and never reads a
real file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import warnings
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np

from analysis._common import (
    FAKE_FILE_PREFIX,
    FAKE_MARKER,
    FAKE_RUN_PREFIX,
    accepted,
    refuse_fake,
    require_fake,
)
from rafr.contracts.schemas import BOOTSTRAP_ALPHA, BOOTSTRAP_B, text_hash
from rafr.core import paths
from rafr.core.config import find_tbd, get, load_config, master_seed
from rafr.core.seeding import component_seed
from rafr.core.seeding import rng as seeded_rng
from rafr.experiments.ext_config import checked, load_ext
from rafr.logging.writer import read_jsonl

E_INCORRECT = "e_incorrect"
E_CORRECT = "e_correct"
COHORTS = (E_INCORRECT, E_CORRECT)
EXPERIMENTS = ("exp1", "exp2")
ACCOUNTINGS = ("main", "parse_fail_as_accept", "complete_case")
ACCOUNTING_ROLE = {
    "main": "main (parse_fail = not accepted)",
    "parse_fail_as_accept": "sensitivity (parse_fail = accepted)",
    "complete_case": "sensitivity (complete case: parse_fail dropped)",
}
STATUS = (
    "EXPLORATORY (post-pilot extension): intervals not corrected for multiple comparisons; "
    "every cell reported"
)
DIRECTION = (
    "X - Y = less information - more information (the direction of D-bar = A1 - B); "
    "positive = the added information reduced errors"
)
Contrast = tuple[str, str]
COMMON_CONTRASTS: tuple[Contrast, ...] = (
    ("no_verifier", "A1"),
    ("A1", "verdict_only"),
    ("verdict_only", "B"),
    ("B", "B_pihat"),
    ("B_pihat_mismatched", "B_pihat"),
)
EXTRA_CONTRASTS: dict[str, dict[str, tuple[Contrast, ...]]] = {
    "exp1": {E_CORRECT: (("A1", "B"), ("permuted_B", "B"))},
    "exp2": {E_INCORRECT: (("A1", "B"),), E_CORRECT: (("A1", "B"),)},
}
CITED_CONTRASTS: dict[str, dict[str, tuple[Contrast, ...]]] = {
    "exp1": {E_INCORRECT: (("A1", "B"), ("permuted_B", "B"))},
}
REUSED_ARMS = ("A1", "B", "permuted_B")  # the original runs' arms (e_incorrect)
RUN_PREFIXES = {"exp1": "exp1ext", "exp2": "exp2ext"}  # configs/ext.yaml: <exp>.run_id_prefix
EXP2_METRIC = {
    E_INCORRECT: ("tfr", "TFR = P(task failure at x_T)"),
    E_CORRECT: (
        "corruption_rate",
        "corruption rate = P(task failure at x_T) on starts E judged correct (not TFR)",
    ),
}
LAMBDAS = ("low", "mid", "high")

Plan = dict[str, dict[str, tuple[Contrast, ...]]]


class ExtAnalysisError(RuntimeError):
    pass


def make_plan(
    common: Sequence[Sequence[str]] = COMMON_CONTRASTS,
    extra: Mapping[str, Mapping[str, Sequence[Sequence[str]]]] = EXTRA_CONTRASTS,
) -> Plan:
    """The contrasts computed per (experiment, cohort): the common list, then the extras."""
    plan: Plan = {}
    for exp in EXPERIMENTS:
        plan[exp] = {}
        for cohort in COHORTS:
            more = (extra.get(exp) or {}).get(cohort) or ()
            pairs = [(str(x), str(y)) for x, y in [*common, *more]]
            if len(set(pairs)) != len(pairs):
                raise ExtAnalysisError(f"{exp} {cohort}: a contrast is listed twice")
            plan[exp][cohort] = tuple(pairs)
    return plan


DEFAULT_PLAN = make_plan()


def plan_from_config(analysis_cfg: Mapping[str, Any]) -> Plan:
    return make_plan(analysis_cfg["contrasts"], analysis_cfg.get("extra_contrasts") or {})


def _name(c: Contrast) -> str:
    return f"{c[0]} - {c[1]}"


# ---- inputs ------------------------------------------------------------------------------


def assert_extension_run(rd: Path, *, prefix: str, cohort: str, base: Path | None = None) -> Path:
    """``rd`` is an extension run directory of ``cohort``: directly under ``base`` (runs/),
    named ``<prefix>_*``, with an ``ext_meta.json`` of that cohort, and not a smoke run. The
    preregistered run directories (``exp1_*``, ``exp2_*``, ``analysis_*``) never pass."""
    base = Path(paths.RUNS if base is None else base).resolve()
    rd = paths.resolve(rd).resolve()
    if rd.parent != base:
        raise ExtAnalysisError(f"{rd} is not a run directory under {base}")
    if not rd.name.startswith(f"{prefix}_"):
        raise ExtAnalysisError(
            f"{rd.name} is not an extension run directory ({prefix}_*): the extension analysis "
            "reads extension runs only"
        )
    if rd.name.startswith(f"{prefix}_smoke_"):
        raise ExtAnalysisError(f"{rd.name} is a smoke run; the analysis does not read it")
    meta_path = rd / "ext_meta.json"
    if not meta_path.is_file():
        raise ExtAnalysisError(f"{rd} has no ext_meta.json: not an extension run")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("cohort") != cohort:
        raise ExtAnalysisError(f"{rd} is a {meta.get('cohort')!r} run, not {cohort!r}")
    if meta.get("smoke"):
        raise ExtAnalysisError(f"{rd} is a smoke run; the analysis does not read it")
    return rd


def exp1_rows(
    cohort: str,
    ext_runs: Sequence[Path],
    original_rows: Iterable[Mapping[str, Any]] = (),
    original_arms: Sequence[str] = ("A1", "B"),
    *,
    prefix: str = RUN_PREFIXES["exp1"],
    base: Path | None = None,
) -> list[dict[str, Any]]:
    """The cohort's decision rows: the extension runs' plus (e_incorrect) the original run's
    ``original_arms`` rows. A (state, arm, lambda) found twice is an error."""
    rows: list[dict[str, Any]] = []
    for r in original_rows:
        if r["arm"] in original_arms:
            rows.append(dict(r))
    for rd in ext_runs:
        rd = assert_extension_run(rd, prefix=prefix, cohort=cohort, base=base)
        rows += list(read_jsonl(rd / "decisions.jsonl"))
    seen: set[tuple[str, str, str]] = set()
    for r in rows:
        key = (r["state_id"], r["arm"], r["lam"])
        if key in seen:
            raise ExtAnalysisError(f"{cohort}: {key} decided twice across the runs")
        seen.add(key)
    return rows


PanelLabels = Mapping[str, Mapping[str, str | None]]  # member -> text_id -> SAFE / UNSAFE / None


def panel_labels(rd: Path) -> dict[str, dict[str, str | None]]:
    """``panel_<member>.jsonl`` of one run directory, as step 12 reads them."""
    out: dict[str, dict[str, str | None]] = {}
    for f in sorted(Path(rd).glob("panel_*.jsonl")):
        out[f.stem.removeprefix("panel_")] = {r["text_id"]: r["label"] for r in read_jsonl(f)}
    return out


def _with_panel(rows: Iterable[Mapping[str, Any]], labels: PanelLabels) -> list[dict[str, Any]]:
    """Each row with ``_panel``: member -> its label of that row's x_T (None = no verdict); a
    member without a row for that x_T is left out of the mapping."""
    out = []
    for r in rows:
        tid = text_hash(r["x_T"]) if "x_T" in r else None
        got = {m: lab[tid] for m, lab in labels.items() if tid is not None and tid in lab}
        out.append({**dict(r), "_panel": got})
    return out


def exp2_finals(
    cohort: str,
    ext_runs: Sequence[Path],
    original_finals: Iterable[Mapping[str, Any]] = (),
    *,
    original_panel: PanelLabels | None = None,
    prefix: str = RUN_PREFIXES["exp2"],
    base: Path | None = None,
) -> list[dict[str, Any]]:
    """The cohort's trajectories (x0, arm): the extension runs' finals plus (e_incorrect) the
    original run's, each with its panel labels (its own run's panel files; ``original_panel``
    for the original's). An (x0, arm) found twice is an error."""
    rows = _with_panel(original_finals, original_panel or {})
    for rd in ext_runs:
        rd = assert_extension_run(rd, prefix=prefix, cohort=cohort, base=base)
        rows += _with_panel(read_jsonl(rd / "trajectories.jsonl"), panel_labels(rd))
    seen: set[tuple[str, str]] = set()
    for r in rows:
        key = (r["x0_id"], r["arm"])
        if key in seen:
            raise ExtAnalysisError(f"{cohort}: trajectory {key} appears twice across the runs")
        seen.add(key)
    return rows


def cite_prereg(
    tables_dir: Path,
    sha256: Mapping[str, str | None],
    contrasts: Sequence[Contrast],
    *,
    fake: bool = False,
) -> dict[str, Any]:
    """The preregistered tables' rows of ``contrasts``, copied, not recomputed. Each table is
    sha256-checked (formal) and must be real (formal) or FAKE (``fake``)."""
    names = {_name(c) for c in contrasts}
    tables = []
    for fname, sha in sha256.items():
        if fake:
            path = Path(tables_dir) / fname
        else:
            path = checked(Path(tables_dir) / fname, sha, what="preregistered table")
        t = json.loads(path.read_text(encoding="utf-8"))
        if bool(t.get("FAKE")) != fake:
            raise ExtAnalysisError(f"{path}: FAKE is {t.get('FAKE')!r}, expected {fake}")
        col = t["columns"].index("contrast")
        rows = [r for r in t["rows"] if r[col] in names]
        missing = names - {r[col] for r in rows}
        if missing:
            raise ExtAnalysisError(f"{path} has no row for {sorted(missing)}")
        try:
            shown = paths.shown(path)
        except ValueError:
            shown = Path(path).as_posix()
        tables.append(
            {
                "file": shown,
                "sha256": sha,
                "id": t["id"],
                "title": t["title"],
                "columns": t["columns"],
                "rows": rows,
                "notes": t.get("notes", []),
            }
        )
    return {
        "status": "cited from the preregistered analysis, not recomputed",
        "contrasts": sorted(names),
        "tables": tables,
    }


# ---- statistics --------------------------------------------------------------------------


def _error(row: Mapping[str, Any], cohort: str, accounting: str) -> float:
    """1 = the decision erred for this cohort, 0 = it did not, nan = excluded (complete case)."""
    acc = accepted(dict(row), accounting)
    if acc is None:
        return float("nan")
    return float(acc) if cohort == E_INCORRECT else float(not acc)


def _boot_idx(n: int, *, n_boot: int, seed: int) -> np.ndarray:
    return seeded_rng(seed, "bootstrap").integers(0, n, (n_boot, n))


def _ci(values: np.ndarray, idx: np.ndarray, alpha: float) -> list[float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a replicate with no pair: nan
        stats = np.nanmean(values[idx], axis=1)
    if np.all(np.isnan(stats)):
        return [float("nan"), float("nan")]
    lo, hi = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)]


def _mean(values: np.ndarray) -> float:
    ok = values[~np.isnan(values)]
    return float(ok.mean()) if ok.size else float("nan")


def _paired(vals: Mapping[str, float], units: Sequence[str], d: np.ndarray) -> np.ndarray:
    """One arm's outcomes on the units of a contrast's pairs (nan elsewhere)."""
    v = np.array([vals.get(u, np.nan) for u in units], dtype=float)
    return np.where(np.isnan(d), np.nan, v)


def paired_block(
    by_arm: Mapping[str, Mapping[str, float]],
    units: Sequence[str],
    contrasts: Sequence[Contrast],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    """Per-arm rates and paired contrasts over ``units`` (sorted ids); ``by_arm[arm][unit]`` is
    the unit's outcome (nan = excluded). A contrast uses the units that have both arms."""
    units = sorted(units)
    idx = _boot_idx(len(units), n_boot=n_boot, seed=seed) if units else None
    arms: dict[str, Any] = {}
    for arm, vals in sorted(by_arm.items()):
        v = np.array([vals.get(u, np.nan) for u in units], dtype=float)
        present = int(np.sum([u in vals for u in units]))
        arms[arm] = {
            "n": present,
            "n_used": int(np.sum(~np.isnan(v))),
            "rate": _mean(v),
            "ci": _ci(v, idx, alpha) if idx is not None and present else [np.nan, np.nan],
        }
    out: dict[str, Any] = {}
    for x, y in contrasts:
        name = _name((x, y))
        if x not in by_arm or y not in by_arm:
            missing = [a for a in (x, y) if a not in by_arm]
            out[name] = {"status": "not run", "missing_arms": missing, "n_pairs": 0}
            continue
        d = np.array(
            [
                (by_arm[x][u] - by_arm[y][u]) if (u in by_arm[x] and u in by_arm[y]) else np.nan
                for u in units
            ],
            dtype=float,
        )
        n_pairs = int(np.sum(~np.isnan(d)))
        out[name] = {
            "status": "ok" if n_pairs else "no pairs",
            "n_pairs": n_pairs,
            "estimate": _mean(d),
            "ci": _ci(d, idx, alpha) if idx is not None and n_pairs else [np.nan, np.nan],
            "rate_x": _mean(_paired(by_arm[x], units, d)),
            "rate_y": _mean(_paired(by_arm[y], units, d)),
        }
    return {"n_units": len(units), "arms": arms, "contrasts": out}


def exp1_analysis(
    rows: Sequence[Mapping[str, Any]],
    cohort: str,
    contrasts: Sequence[Contrast],
    *,
    n_boot: int,
    seed: int,
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict[str, Any]:
    """Per lambda and accounting: per-arm error rates and the paired contrasts."""
    if cohort not in COHORTS:
        raise ExtAnalysisError(f"unknown cohort {cohort!r}")
    states = sorted({r["state_id"] for r in rows})
    x0_of = {r["state_id"]: r["x0_id"] for r in rows}
    if len(set(x0_of.values())) != len(states):
        raise ExtAnalysisError("Exp 1 resamples x0: one state per x0 is assumed")
    out: dict[str, Any] = {
        "cohort": cohort,
        "metric": "false accept rate" if cohort == E_INCORRECT else "false reject rate",
        "contrasts": [_name(c) for c in contrasts],
        "accountings": dict(ACCOUNTING_ROLE),
        "n_states": len(states),
        "by_lambda": {},
    }
    lams = [lam for lam in LAMBDAS if any(r["lam"] == lam for r in rows)]
    for lam in lams:
        cell: dict[str, Any] = {}
        for acc in ACCOUNTINGS:
            by_arm: dict[str, dict[str, float]] = {}
            for r in rows:
                if r["lam"] == lam:
                    by_arm.setdefault(r["arm"], {})[r["state_id"]] = _error(r, cohort, acc)
            cell[acc] = paired_block(
                by_arm, states, contrasts, n_boot=n_boot, seed=seed, alpha=alpha
            )
        decided: dict[str, int] = {}
        for r in rows:
            if r["lam"] == lam:
                decided[r["arm"]] = decided.get(r["arm"], 0) + 1
        cell["incomplete_arms"] = {
            a: len(states) - k for a, k in sorted(decided.items()) if k != len(states)
        }
        out["by_lambda"][lam] = cell
    return out


def exp2_analysis(
    finals: Sequence[Mapping[str, Any]],
    cohort: str,
    contrasts: Sequence[Contrast],
    *,
    n_boot: int,
    seed: int,
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict[str, Any]:
    """The cohort's metric per arm and the paired contrasts over the starts (x0) both arms ran;
    non-convergence alongside."""
    if cohort not in COHORTS:
        raise ExtAnalysisError(f"unknown cohort {cohort!r}")
    key, metric = EXP2_METRIC[cohort]
    x0s = sorted({f["x0_id"] for f in finals})
    failed: dict[str, dict[str, float]] = {}
    nonconv: dict[str, dict[str, float]] = {}
    for f in finals:
        failed.setdefault(f["arm"], {})[f["x0_id"]] = float(not f["task_success"])
        nonconv.setdefault(f["arm"], {})[f["x0_id"]] = float(not f["accepted"])
    main = paired_block(failed, x0s, contrasts, n_boot=n_boot, seed=seed, alpha=alpha)
    return {
        "cohort": cohort,
        "metric_key": key,
        "metric": metric,
        "contrasts": [_name(c) for c in contrasts],
        "n_starts": len(x0s),
        key: main,
        "non_convergence": paired_block(nonconv, x0s, (), n_boot=n_boot, seed=seed, alpha=alpha),
        "panel": panel_block(finals, x0s, contrasts, main, n_boot=n_boot, seed=seed, alpha=alpha),
    }


def panel_block(
    finals: Sequence[Mapping[str, Any]],
    x0s: Sequence[str],
    contrasts: Sequence[Contrast],
    main: Mapping[str, Any],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    """Per panel member: the metric with that member's x_T label in place of E's (UNSAFE = a task
    failure), the same contrasts and pairing, and whether each contrast's direction is E's."""
    members = sorted({m for f in finals for m in f.get("_panel", {})})
    out: dict[str, Any] = {}
    for m in members:
        failed: dict[str, dict[str, float]] = {}
        no_verdict: dict[str, int] = {}
        not_judged: dict[str, int] = {}
        for f in finals:
            panel = f.get("_panel", {})
            if m not in panel:
                not_judged[f["arm"]] = not_judged.get(f["arm"], 0) + 1
            elif panel[m] is None:
                no_verdict[f["arm"]] = no_verdict.get(f["arm"], 0) + 1
            else:
                failed.setdefault(f["arm"], {})[f["x0_id"]] = float(panel[m] != "SAFE")
        blk = paired_block(failed, x0s, contrasts, n_boot=n_boot, seed=seed, alpha=alpha)
        for name, c in blk["contrasts"].items():
            e = main["contrasts"].get(name, {})
            if c["status"] == "ok" and e.get("status") == "ok":
                c["same_direction_as_E"] = bool(np.sign(c["estimate"]) == np.sign(e["estimate"]))
            else:
                c["same_direction_as_E"] = None
        out[m] = {
            **blk,
            "n_without_verdict_by_arm": dict(sorted(no_verdict.items())),
            "n_not_judged_by_arm": dict(sorted(not_judged.items())),
        }
    return out


def analyze(
    *,
    exp1: Mapping[str, Sequence[Mapping[str, Any]]],
    exp2: Mapping[str, Sequence[Mapping[str, Any]]],
    plan: Plan = DEFAULT_PLAN,
    cited: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    n_boot: int = BOOTSTRAP_B,
    seed: int,
) -> dict[str, Any]:
    """``cited[exp][cohort]``: the preregistered rows (``cite_prereg``) shown with that cell."""
    cited = cited or {}
    report: dict[str, Any] = {
        "status": STATUS,
        "direction": DIRECTION,
        "plan": {e: {c: [_name(p) for p in v] for c, v in d.items()} for e, d in plan.items()},
        "bootstrap": {
            "method": "percentile_bootstrap",
            "level": 1 - BOOTSTRAP_ALPHA,
            "n_boot": n_boot,
            "seed": seed,
            "unit": "x0 (paired: both arms of a contrast resampled together)",
            "multiplicity": "none (exploratory)",
        },
        "exp1": {},
        "exp2": {},
    }
    for c, rows in exp1.items():
        report["exp1"][c] = exp1_analysis(rows, c, plan["exp1"][c], n_boot=n_boot, seed=seed)
    for c, rows in exp2.items():
        report["exp2"][c] = exp2_analysis(rows, c, plan["exp2"][c], n_boot=n_boot, seed=seed)
    for exp, by_cohort in cited.items():
        for c, block in by_cohort.items():
            if c in report[exp]:
                report[exp][c]["cited_not_recomputed"] = dict(block)
    return report


# ---- tables --------------------------------------------------------------------------------


def _isnan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def _fmt(x: float) -> str:
    return "n/a" if _isnan(x) else f"{x:+.3f}"


def _rate(x: float) -> str:
    return "n/a" if _isnan(x) else f"{x:.3f}"


def _contrast_cell(c: Mapping[str, Any]) -> str:
    if c["status"] != "ok":
        return f"{c['status']} (n 0)"
    lo, hi = c["ci"]
    return f"{_fmt(c['estimate'])} [{_fmt(lo)}, {_fmt(hi)}] (n {c['n_pairs']})"


def _arm_cell(a: Mapping[str, Any]) -> str:
    lo, hi = a["ci"]
    return f"{_rate(a['rate'])} [{_rate(lo)}, {_rate(hi)}] (n {a['n_used']}/{a['n']})"


def _cited_lines(block: Mapping[str, Any]) -> list[str]:
    lines = [f"### Cited, not recomputed: {', '.join(block['contrasts'])}", ""]
    for t in block["tables"]:
        lines += [f"{t['id']}: {t['title']} ({t['file']}, sha256 {t['sha256']})", ""]
        lines.append("| " + " | ".join(t["columns"]) + " |")
        lines.append("|" + "---|" * len(t["columns"]))
        for r in t["rows"]:
            cells = [f"{v:+.3f}" if isinstance(v, float) else str(v) for v in r]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def render(report: Mapping[str, Any]) -> str:
    """Markdown: every cell of the report (all cohorts, lambda levels, accountings, arms)."""
    b = report["bootstrap"]
    lines = [
        "# Extension contrasts",
        "",
        f"**{report['status']}**",
        "",
        f"Direction: {report['direction']}.",
        "",
        f"Intervals: 95% percentile bootstrap, B = {b['n_boot']}, unit {b['unit']}, seed "
        f"{b['seed']}; no multiplicity correction. Cells read estimate [95% CI] (n pairs); arm "
        "cells read rate [95% CI] (n used / n decided).",
        "",
    ]
    for cohort, a in report["exp1"].items():
        lams = list(a["by_lambda"])
        lines += [f"## Exp 1, {cohort}: {a['metric']} ({a['n_states']} states)", ""]
        for acc in ACCOUNTINGS:
            lines += [f"### Contrasts, {a['accountings'][acc]}", ""]
            lines.append("| contrast | " + " | ".join(lams) + " |")
            lines.append("|---|" + "---|" * len(lams))
            for name in a["contrasts"]:
                cells = [
                    _contrast_cell(a["by_lambda"][lam][acc]["contrasts"][name]) for lam in lams
                ]
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
            lines += ["", f"### Arms, {a['accountings'][acc]}", ""]
            lines.append("| arm | " + " | ".join(lams) + " |")
            lines.append("|---|" + "---|" * len(lams))
            arms = sorted({arm for lam in lams for arm in a["by_lambda"][lam][acc]["arms"]})
            for arm in arms:
                cells = [
                    _arm_cell(a["by_lambda"][lam][acc]["arms"][arm])
                    if arm in a["by_lambda"][lam][acc]["arms"]
                    else "not run"
                    for lam in lams
                ]
                lines.append(f"| {arm} | " + " | ".join(cells) + " |")
            lines.append("")
        incomplete = {lam: a["by_lambda"][lam]["incomplete_arms"] for lam in lams}
        if any(incomplete.values()):
            lines += [f"States without a decision, by lambda and arm: {incomplete}", ""]
        if "cited_not_recomputed" in a:
            lines += _cited_lines(a["cited_not_recomputed"])
    for cohort, a in report["exp2"].items():
        block = a[a["metric_key"]]
        lines += [f"## Exp 2, {cohort}: {a['metric']} ({a['n_starts']} starts)", ""]
        lines += ["### Contrasts, paired over the starts both arms ran", ""]
        lines += [
            "| contrast | estimate [95% CI] (n pairs) | rate X | rate Y |",
            "|---|---|---|---|",
        ]
        for name in a["contrasts"]:
            c = block["contrasts"][name]
            rx, ry = (c["rate_x"], c["rate_y"]) if c["status"] == "ok" else (None, None)
            lines.append(f"| {name} | {_contrast_cell(c)} | {_rate(rx)} | {_rate(ry)} |")
        for title, blk in (("Arms", block), ("Non-convergence (not accepted at x_T)",
                                              a["non_convergence"])):  # fmt: skip
            lines += ["", f"### {title}", "", "| arm | rate [95% CI] (n used / n) |", "|---|---|"]
            for arm, s in blk["arms"].items():
                lines.append(f"| {arm} | {_arm_cell(s)} |")
        lines.append("")
        for member, p in a.get("panel", {}).items():
            lines += [f"### Panel sensitivity: {member} as the judge of x_T", ""]
            lines += ["| contrast | estimate [95% CI] (n pairs) | same direction as E |"]
            lines.append("|---|---|---|")
            for name in a["contrasts"]:
                c = p["contrasts"][name]
                lines.append(f"| {name} | {_contrast_cell(c)} | {c['same_direction_as_E']} |")
            lines += ["", "| arm | rate [95% CI] (n used / n) |", "|---|---|"]
            for arm, s in p["arms"].items():
                lines.append(f"| {arm} | {_arm_cell(s)} |")
            lines += [
                "",
                f"No verdict, by arm: {p['n_without_verdict_by_arm']}; "
                f"no panel row yet, by arm: {p['n_not_judged_by_arm']}",
                "",
            ]
    return "\n".join(lines)


# ---- fake data (end-to-end check before any real extension row exists) --------------------


FAKE_RATES = {
    "A1": 0.30,
    "B": 0.25,
    "permuted_B": 0.28,
    "no_verifier": 0.45,
    "verdict_only": 0.27,
    "B_pihat": 0.20,
    "B_pihat_mismatched": 0.26,
}
FAKE_CITED = ("T06_exp1_did.json", "T06_exp1_delta_by_lambda.json")


def _jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(dict(r)) + "\n")


def _fake_decision(rnd: random.Random, run: str, sid: str, arm: str, lam: str, p: float) -> dict:
    pf = rnd.random() < 0.01
    action = None if pf else ("accept" if rnd.random() < p else "rewrite")
    return {"run_id": f"{FAKE_RUN_PREFIX}{run}", "state_id": sid, "x0_id": sid.removesuffix("-g1"),
            "arm": arm, "lam": lam, "t": 1, "parse_fail": pf, "action": action}  # fmt: skip


def _fake_tables(d: Path) -> None:
    """Fake stand-ins of the cited preregistered tables, same layout, FAKE true."""
    d.mkdir(parents=True, exist_ok=True)
    did = {"FAKE": True, "id": "T06_exp1_did", "title": "fake D-bar",
           "columns": ["role", "contrast", "accounting", "D_bar", "ci95_lo", "ci95_hi"],
           "rows": [["primary", "A1 - B", "main", 0.03, -0.03, 0.08],
                    ["secondary", "permuted_B - B", "main", -0.05, -0.12, 0.02]]}  # fmt: skip
    lam = {"FAKE": True, "id": "T06_exp1_delta_by_lambda", "title": "fake Delta_FA by lambda",
           "columns": ["contrast", "low", "mid", "high"],
           "rows": [["A1 - B", 0.0, 0.03, 0.03], ["permuted_B - B", 0.05, 0.05, 0.01]]}  # fmt: skip
    for name, t in zip(FAKE_CITED, (did, lam), strict=True):
        (d / name).write_text(json.dumps(t), encoding="utf-8", newline="\n")


def write_fake(root: Path, seed: int = 7, n: int = 300, n_correct: int = 120) -> dict[str, Any]:
    """Fake inputs in the real layouts (run directories named as the extension's), row order
    shuffled, under a FAKE marker."""
    rnd = random.Random(seed)  # noqa: S311 - fake data only
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / FAKE_MARKER).write_text(
        "fake data for analysis/ext_contrasts.py\n", encoding="utf-8", newline="\n"
    )
    inc = [f"x0-fake{i:04d}-g1" for i in range(n)]
    cor = [f"x0-fakec{i:04d}-g1" for i in range(n_correct)]
    lam_shift = {"low": 0.0, "mid": 0.05, "high": 0.10}

    def decisions(run: str, ids: list[str], arms: Sequence[str], flip: bool) -> list[dict]:
        rows = []
        for sid in ids:
            for lam in LAMBDAS:
                for arm in arms:
                    p = FAKE_RATES[arm] + lam_shift[lam]
                    rows.append(_fake_decision(rnd, run, sid, arm, lam, 1 - p if flip else p))
        rnd.shuffle(rows)
        return rows

    def meta(rd: Path, cohort: str) -> None:
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "ext_meta.json").write_text(
            json.dumps({"cohort": cohort, "smoke": False}), encoding="utf-8", newline="\n"
        )

    _jsonl(root / "exp1_original" / "decisions.jsonl",
           decisions("exp1", inc, ("A1", "B", "permuted_B"), False))  # fmt: skip
    _fake_tables(root / "analysis_original")
    spec: dict[str, Any] = {
        "base": root,
        "exp1_original": root / "exp1_original",
        "cited_tables": root / "analysis_original",
        "exp1": {},
        "exp2": {},
    }
    runs = []
    for i, arms in enumerate((("no_verifier", "B_pihat"), ("verdict_only", "B_pihat_mismatched"))):
        rd = root / f"exp1ext_e_incorrect_{i}"
        meta(rd, E_INCORRECT)
        _jsonl(rd / "decisions.jsonl", decisions(f"ext{i}", inc, arms, False))
        runs.append(rd)
    spec["exp1"][E_INCORRECT] = runs
    rd = root / "exp1ext_e_correct"
    meta(rd, E_CORRECT)
    _jsonl(rd / "decisions.jsonl", decisions("extc", cor, tuple(FAKE_RATES), True))
    spec["exp1"][E_CORRECT] = [rd]

    x0s = [s.removesuffix("-g1") for s in inc]
    order = x0s[:]
    rnd.shuffle(order)
    original = {x: ("A1" if i % 2 else "B") for i, x in enumerate(order)}

    def final(run: str, x: str, arm: str, p_fail: float) -> dict:
        return {"run_id": f"{FAKE_RUN_PREFIX}{run}", "x0_id": x, "arm": arm,
                "x_T": f"fake x_T {run} {x} {arm}",
                "task_success": rnd.random() >= p_fail, "accepted": rnd.random() < 0.9}  # fmt: skip

    def panel(rd: Path, rows: list[dict]) -> None:
        """Grok agrees with E 90% of the time; one row has no verdict; one x_T is not judged."""
        out = []
        for i, r in enumerate(rows[:-1]):
            agree = rnd.random() < 0.9
            safe = r["task_success"] if agree else not r["task_success"]
            out.append({"member": "grok", "text_id": text_hash(r["x_T"]),
                        "label": None if i == 0 else ("SAFE" if safe else "UNSAFE")})  # fmt: skip
        _jsonl(rd / "panel_grok.jsonl", out)

    orig_rows = [final("exp2", x, a, FAKE_RATES[a]) for x, a in original.items()]
    _jsonl(root / "exp2_original" / "trajectories.jsonl", orig_rows)
    panel(root / "exp2_original", orig_rows)
    spec["exp2_original"] = root / "exp2_original"
    rd_new, rd_fill = root / "exp2ext_e_incorrect_new", root / "exp2ext_e_incorrect_fill"
    for rd in (rd_new, rd_fill):
        meta(rd, E_INCORRECT)
    new_rows = [final("ext", x, a, FAKE_RATES[a]) for x in x0s
                for a in ("no_verifier", "B_pihat", "verdict_only")]  # fmt: skip
    fill_rows = [final("fill", x, "B" if a == "A1" else "A1", 0.3) for x, a in original.items()]
    rnd.shuffle(new_rows)
    rnd.shuffle(fill_rows)
    _jsonl(rd_new / "trajectories.jsonl", new_rows)
    _jsonl(rd_fill / "trajectories.jsonl", fill_rows)
    panel(rd_new, new_rows)
    panel(rd_fill, fill_rows)
    spec["exp2"][E_INCORRECT] = [rd_new, rd_fill]
    rd = root / "exp2ext_e_correct"
    meta(rd, E_CORRECT)
    cor_rows = [final("extc", s.removesuffix("-g1"), a, 0.05) for s in cor
                for a in ("A1", "B", "no_verifier", "verdict_only", "B_pihat")]  # fmt: skip
    _jsonl(rd / "trajectories.jsonl", cor_rows)
    panel(rd, cor_rows)
    spec["exp2"][E_CORRECT] = [rd]
    return spec


def _original_arms(plan: Plan) -> tuple[str, ...]:
    """The original run's arms a contrast of Exp 1 e_incorrect needs (A1, B)."""
    used = {a for c in plan["exp1"][E_INCORRECT] for a in c}
    return tuple(a for a in REUSED_ARMS if a in used)


def analyze_fake(
    root: Path,
    *,
    plan: Plan = DEFAULT_PLAN,
    n_boot: int = BOOTSTRAP_B,
    seed: int = 0,
) -> dict[str, Any]:
    """Write fake inputs under ``root`` and run the whole analysis on them."""
    spec = write_fake(root)
    base = spec["base"]
    require_fake(root, what="ext_contrasts --fake")
    orig1 = list(read_jsonl(spec["exp1_original"] / "decisions.jsonl"))
    orig2 = list(read_jsonl(spec["exp2_original"] / "trajectories.jsonl"))
    for rows in (orig1, orig2):
        require_fake(root, rows, what="ext_contrasts --fake")
    exp1 = {
        E_INCORRECT: exp1_rows(
            E_INCORRECT, spec["exp1"][E_INCORRECT], orig1, _original_arms(plan), base=base
        ),
        E_CORRECT: exp1_rows(E_CORRECT, spec["exp1"][E_CORRECT], base=base),
    }
    exp2 = {
        E_INCORRECT: exp2_finals(
            E_INCORRECT,
            spec["exp2"][E_INCORRECT],
            orig2,
            original_panel=panel_labels(spec["exp2_original"]),
            base=base,
        ),
        E_CORRECT: exp2_finals(E_CORRECT, spec["exp2"][E_CORRECT], base=base),
    }
    cited = {
        "exp1": {
            E_INCORRECT: cite_prereg(
                spec["cited_tables"],
                dict.fromkeys(FAKE_CITED),
                CITED_CONTRASTS["exp1"][E_INCORRECT],
                fake=True,
            )
        }
    }
    return analyze(exp1=exp1, exp2=exp2, plan=plan, cited=cited, n_boot=n_boot, seed=seed)


def _clean(obj: Any) -> Any:
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    return obj


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fake", action="store_true", help="end-to-end run on shuffled fake data")
    ap.add_argument("--exp1-e-incorrect", type=Path, nargs="*", default=[])
    ap.add_argument("--exp1-e-correct", type=Path, nargs="*", default=[])
    ap.add_argument("--exp2-e-incorrect", type=Path, nargs="*", default=[])
    ap.add_argument("--exp2-e-correct", type=Path, nargs="*", default=[])
    args = ap.parse_args(argv)
    ext = load_ext()
    acfg = get(ext, "analysis")
    boot = acfg["bootstrap"]
    seed = component_seed(master_seed(load_config()), str(boot["seed_component"]))
    plan = plan_from_config(acfg)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.fake:
        out = paths.run_dir(f"ext_analysis_FAKE_{ts}")
        report = analyze_fake(out / "inputs", plan=plan, n_boot=int(boot["n_boot"]), seed=seed)
        (out / FAKE_MARKER).write_text("fake extension analysis\n", encoding="utf-8", newline="\n")
        prefix = FAKE_FILE_PREFIX
    else:
        tbd = find_tbd(acfg, "ext.analysis")
        if tbd:
            raise SystemExit(f"the contrast list is not confirmed yet: {tbd}")
        pre = {e: str(get(ext, f"{e}.run_id_prefix")) for e in EXPERIMENTS}
        s1, s2 = get(ext, "exp1.e_incorrect"), get(ext, "exp2.e_incorrect")
        run1, run2 = paths.resolve(s1["original_run"]), paths.resolve(s2["original_run"])
        f1 = checked(run1 / "decisions.jsonl", s1["original_sha256"]["decisions.jsonl"],
                     what="original Exp 1 decisions")  # fmt: skip
        f2 = checked(run2 / "trajectories.jsonl", s2["original_sha256"]["trajectories.jsonl"],
                     what="original Exp 2 finals")  # fmt: skip
        for rd in [*args.exp1_e_incorrect, *args.exp1_e_correct, *args.exp2_e_incorrect,
                   *args.exp2_e_correct]:  # fmt: skip
            refuse_fake(rd, what="ext_contrasts")
        exp1 = {}
        if args.exp1_e_incorrect:
            exp1[E_INCORRECT] = exp1_rows(E_INCORRECT, args.exp1_e_incorrect, read_jsonl(f1),
                                          _original_arms(plan), prefix=pre["exp1"])  # fmt: skip
        if args.exp1_e_correct:
            exp1[E_CORRECT] = exp1_rows(E_CORRECT, args.exp1_e_correct, prefix=pre["exp1"])
        exp2 = {}
        if args.exp2_e_incorrect:
            original_panel = {
                name.removeprefix("panel_").removesuffix(".jsonl"): {
                    r["text_id"]: r["label"]
                    for r in read_jsonl(checked(run2 / name, sha, what=f"original Exp 2 {name}"))
                }
                for name, sha in s2["original_sha256"].items()
                if name.startswith("panel_")
            }
            exp2[E_INCORRECT] = exp2_finals(E_INCORRECT, args.exp2_e_incorrect, read_jsonl(f2),
                                            original_panel=original_panel,
                                            prefix=pre["exp2"])  # fmt: skip
        if args.exp2_e_correct:
            exp2[E_CORRECT] = exp2_finals(E_CORRECT, args.exp2_e_correct, prefix=pre["exp2"])
        cited: dict[str, dict[str, Any]] = {}
        for exp, by_cohort in (acfg.get("cited_not_recomputed") or {}).items():
            for cohort, spec in by_cohort.items():
                cited.setdefault(exp, {})[cohort] = cite_prereg(
                    paths.resolve(spec["tables_dir"]),
                    spec["sha256"],
                    [tuple(c) for c in spec["contrasts"]],
                )
        report = analyze(
            exp1=exp1, exp2=exp2, plan=plan, cited=cited, n_boot=int(boot["n_boot"]), seed=seed
        )
        out = paths.run_dir(f"ext_analysis_{ts}")
        prefix = ""
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{prefix}report.json").write_text(
        json.dumps(_clean(report), indent=1), encoding="utf-8", newline="\n"
    )
    (out / f"{prefix}tables.md").write_text(render(report), encoding="utf-8", newline="\n")
    print(render(report))
    print(f"[ext_contrasts] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
