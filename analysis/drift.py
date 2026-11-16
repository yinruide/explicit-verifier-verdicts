"""Exp 1 drift (analysis plan section 10; protocol II.6.3): the reliability ladder the channel
actually produced on the Exp 1 states, next to the one the gate measured on D_gate.

R_lambda^(i)(D_Exp1) is the gate's quantity (II.6.1, C.20) computed on D_Exp1: the share of the
states, all E-UNSAFE by construction (II.2.4), whose OBSERVED judgment ``1[s_i~ >= tau_i]``
flags them at level lambda. Every arm sees the same s~ for a (state, lambda) (II.5.1), so the
rows are reduced to one per (state, lambda) first. The 95% percentile bootstrap resamples the
analysis unit, the x0 (one state per x0; III.1.2), B = 10000, seed recorded.

Reported, not judged: II.6.3 names no threshold for a "clear" drift, so this module computes
the two ladders and their difference and draws no conclusion. Nothing here removes a condition
(II.6.3) or excuses a result; the drift itself is a transport diagnostic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from analysis._common import bootstrap_ci, bootstrap_spec, mean_or_nan
from analysis.exp1_did import LAMBDAS, check_d_exp1_all_unsafe
from rafr.contracts.schemas import BOOTSTRAP_B

VERIFIERS = ("v1", "v2")


def per_state(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, tuple[int, int]]]:
    """state_id -> {lambda: (y1_obs, y2_obs)}; refuses rows whose arms disagree on the evidence."""
    check_d_exp1_all_unsafe(list(rows))  # type: ignore[arg-type]
    out: dict[str, dict[str, tuple[int, int]]] = {}
    for r in rows:
        y = (int(r["s1_tilde"] >= r["tau1"]), int(r["s2_tilde"] >= r["tau2"]))
        cell = out.setdefault(r["state_id"], {})
        if cell.setdefault(r["lam"], y) != y:
            raise ValueError(
                f"{r['state_id']} at lambda={r['lam']}: arms saw different evidence (II.5.1)"
            )
    return out


def exp1_ladder(
    rows: Sequence[Mapping[str, Any]], *, seed: int = 0, n_boot: int = BOOTSTRAP_B
) -> dict[str, Any]:
    """R_lambda^(i) on D_Exp1 with a bootstrap CI over x0 (one state per x0)."""
    states = per_state(rows)
    units = list(states.values())
    out: dict[str, Any] = {
        "population": "D_Exp1 (every state E-UNSAFE)",
        "n_states": len(units),
        "bootstrap": bootstrap_spec("x0", seed=seed, n_boot=n_boot),
        "r": {},
    }
    for vi, v in enumerate(VERIFIERS):
        out["r"][v] = {}
        for lam in LAMBDAS:
            vals = [float(u[lam][vi]) for u in units if lam in u]

            def stat(sample: Sequence[float]) -> float:
                return mean_or_nan(list(sample))

            lo, hi = bootstrap_ci(vals, stat, n_boot=n_boot, seed=seed)
            out["r"][v][lam] = {"estimate": mean_or_nan(vals), "ci95": [lo, hi], "n": len(vals)}
    return out


def compare_to_gate(
    ladder: Mapping[str, Any], gate_r: Mapping[str, Mapping[str, float]] | None
) -> list[dict[str, Any]]:
    """One row per (verifier, lambda): R on D_Exp1, R on D_gate (step 9), their difference."""
    rows = []
    for v in VERIFIERS:
        for lam in LAMBDAS:
            e1 = ladder["r"][v][lam]
            g = None if gate_r is None else (gate_r.get(v) or {}).get(lam)
            rows.append(
                {
                    "verifier": v,
                    "lambda": lam,
                    "R_exp1": e1["estimate"],
                    "ci95_lo": e1["ci95"][0],
                    "ci95_hi": e1["ci95"][1],
                    "n": e1["n"],
                    "R_gate": g,
                    "exp1_minus_gate": None if g is None else e1["estimate"] - g,
                }
            )
    return rows
