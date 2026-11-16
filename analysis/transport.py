"""Transport diagnostic (II.10, I.5.4, II.6.3): compare the Exp 1 state pool with the Exp 2
decision points on toxicity (raw scores), margins and UNSAFE share. Reported after the fact;
a KS statistic summarizes each distributional gap (no test claim in the pilot).

Exp 2's x0 source is D_Exp1 (decision, group 6d): the t = 1 decision points ARE the Exp 1
states, so the all-rounds comparison is diluted by identical rows. Only the rounds t >= 2
carry information; ``compare`` reports them separately and says so in ``note``. Since v0.9 the
Exp 1 state pool is E-UNSAFE by construction (C.16); the note says that the UNSAFE share
therefore differs by design (II.10)."""

from __future__ import annotations

from typing import Any

import numpy as np

from rafr.contracts.schemas import text_hash


def _summary(xs: list[float]) -> dict[str, float]:
    a = np.asarray([x for x in xs if x is not None], dtype=float)
    if a.size == 0:
        return {"n": 0}
    q = np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=0)),
        "q10": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "q90": float(q[4]),
    }


def ks_statistic(a: list[float], b: list[float]) -> float:
    x = np.sort(np.asarray([v for v in a if v is not None], dtype=float))
    y = np.sort(np.asarray([v for v in b if v is not None], dtype=float))
    if x.size == 0 or y.size == 0:
        return float("nan")
    grid = np.concatenate([x, y])
    fx = np.searchsorted(x, grid, side="right") / x.size
    fy = np.searchsorted(y, grid, side="right") / y.size
    return float(np.max(np.abs(fx - fy)))


def _label(r: dict[str, Any], truth: dict[str, str] | None) -> str | None:
    """E's label of the row's x_t: the row's own (Exp 1), else the round judgments (Exp 2)."""
    if r.get("e_label") is not None:
        return r["e_label"]
    return None if truth is None else truth.get(text_hash(r["x_t"]))


def _collect(rows: list[dict[str, Any]], truth: dict[str, str] | None = None) -> dict[str, list]:
    labels = [_label(r, truth) for r in rows]
    return {
        "s1": [r["s1"] for r in rows],
        "s2": [r["s2"] for r in rows],
        "m1": [r["s1"] - r["tau1"] for r in rows if r["s1"] is not None and r["tau1"] is not None],
        "m2": [r["s2"] - r["tau2"] for r in rows if r["s2"] is not None and r["tau2"] is not None],
        "unsafe": [float(y == "UNSAFE") for y in labels if y is not None],
    }


def compare(
    exp1_rows: list[dict[str, Any]],
    exp2_rows: list[dict[str, Any]],
    truth: dict[str, str] | None = None,
) -> dict[str, Any]:
    """``exp1_rows``: one row per state (any arm/lambda; raw s is the same); ``exp2_rows``:
    every decision point of Exp 2 (all rounds). Exp 2 rows carry no E label, so the UNSAFE share
    of its decision points needs ``truth`` (``text_hash(x_t) -> label`` from
    ``round_judgments.jsonl``, II.8.3); without it that share has n = 0."""
    # de-duplicate Exp 1 to one row per state
    seen, e1 = set(), []
    for r in exp1_rows:
        if r["state_id"] not in seen:
            seen.add(r["state_id"])
            e1.append(r)
    a, b = _collect(e1), _collect(exp2_rows, truth)
    later = [r for r in exp2_rows if int(r["t"]) >= 2]
    c = _collect(later, truth)
    out: dict[str, Any] = {
        "note": (
            "Exp 2 x0 = D_Exp1: t = 1 decision points are the Exp 1 states themselves, so only "
            "t >= 2 is informative; read exp2_t_ge_2 / ks_t_ge_2 and by_round_exp2, not the "
            "all-rounds column. v0.9 (II.10, C.16): the Exp 1 state pool is E-UNSAFE by "
            "construction, the Exp 2 decision points at t >= 2 are not -- the UNSAFE share "
            "differs by design, not by transport failure"
        ),
        "n_exp2_rows_t1": len(exp2_rows) - len(later),
        "n_exp2_rows_t_ge_2": len(later),
    }
    for k in ("s1", "s2", "m1", "m2", "unsafe"):
        out[k] = {
            "exp1_states": _summary(a[k]),
            "exp2_decision_points": _summary(b[k]),
            "ks": ks_statistic(a[k], b[k]),
            "exp2_t_ge_2": _summary(c[k]),
            "ks_t_ge_2": ks_statistic(a[k], c[k]),
        }
    out["by_round_exp2"] = {}
    for t in sorted({int(r["t"]) for r in exp2_rows}):
        rt = [r for r in exp2_rows if int(r["t"]) == t]
        c = _collect(rt, truth)
        out["by_round_exp2"][t] = {k: _summary(c[k]) for k in ("s1", "s2", "unsafe")}
    return out
