"""Exp 1 primary analysis (paper Sec. 4.6) + the two paper Sec. 4.6 sensitivity analyses.

Per state i (analysis unit = x0, one state per x0), with Y = the accept indicator (every
D_Exp1 state is E-UNSAFE, so an accept is a false accept):

    D_i = (Y^A1_high,i - Y^B_high,i) - (Y^A1_low,i - Y^B_low,i)

**Levels (paper Sec. 4.6).** D_i uses ONLY the two extreme levels, lambda_high and lambda_low.
lambda_mid never enters the primary difference-in-differences; the three-level
Delta_FA(lambda) curve is reported as a SECONDARY result, for the monotonicity shape.


**Interval (paper Sec. 4.6).** D-bar with a 95% percentile bootstrap CI, resampling unit = x0,
B = 10000, seed recorded in the output (``analysis/_common.py``). The pilot makes no
significance claims: the interval expresses uncertainty only.

**N (paper Sec. 4.6, 3).** D_Exp1 holds E-UNSAFE states only, so the denominator of
the false accept rate is D_Exp1 in full: N = |D_Exp1| = 300 and the N = 300 column of paper App. B
applies directly. A row whose ``e_label`` is not UNSAFE means D_Exp1 is not what paper Sec. 4.2
builds; :func:`check_d_exp1_all_unsafe` raises on it instead of quietly shrinking the denominator.

**H2 (paper Sec. 4.6).** :func:`h2_strata` reports Delta_FA(lambda) inside two arm-independent
strata: (i) whether at least one verifier was perturbed at this (state, lambda), (ii) whether
the two verifiers disagree in the OBSERVED evidence. All three arms see the same evidence for
the same (state, lambda), so neither stratifier is post-treatment. This is a mechanism
diagnostic, not a causal test. r_hat_ref is reported alongside by the runner and is
deliberately not available here: it is never a stratifier (paper Sec. 3).

Sign convention (paper Sec. 3): all deltas are "A1 minus B", positive = B better. Cells:
FA(arm, lambda) = P(accept) over every D_Exp1 state. The unnecessary-rewrite rate is gone
(paper Sec. 4.3): its denominator, the SAFE states, is empty by construction.
Accountings: main (parse_fail = not accepted), parse_fail_as_accept, complete_case (drop the
whole state if any cell failed to parse, preserving the pairing). Robust iff the three point
estimates share a sign. Secondary contrast: permuted_B vs B (paper Sec. 3).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis._common import (
    ACCOUNTINGS,
    accepted,
    bootstrap_ci,
    bootstrap_spec,
    mean_or_nan,
    same_sign,
)
from rafr.contracts.schemas import BOOTSTRAP_B

ARMS = ("A1", "B", "permuted_B")
LAMBDAS = ("low", "mid", "high")

PRIMARY_LAMBDA_HIGH = "high"
PRIMARY_LAMBDA_LOW = "low"
"""paper Sec. 4.6: the primary contrast is the two extreme levels only."""

SECONDARY_LAMBDA = "mid"
"""paper Sec. 4.6: lambda_mid is reported through the three-level Delta_FA curve only."""


class Exp1DataError(ValueError):
    """The Exp 1 rows contradict paper Sec. 4.2: D_Exp1 holds E-UNSAFE states only ."""


def check_d_exp1_all_unsafe(rows: list[dict[str, Any]]) -> None:
    """Raise :class:`Exp1DataError` unless every row carries ``e_label == "UNSAFE"``.

    The false accept rate's denominator is D_Exp1 in full (paper Sec. 3). A SAFE or missing
    label cannot be dropped from it silently: it means the split or the outcome file is wrong.
    """
    bad = sorted({r["state_id"] for r in rows if r.get("e_label") != "UNSAFE"})
    if bad:
        raise Exp1DataError(
            f"{len(bad)} Exp 1 state(s) are not E-UNSAFE (e.g. {bad[:3]}); D_Exp1 holds E-UNSAFE "
            "states only (II.2.4, C.16), so the false accept rate's denominator is D_Exp1 in "
            "full and a SAFE state means the split or the outcome file is wrong"
        )


def _table(rows: list[dict[str, Any]], accounting: str) -> dict[str, dict[tuple[str, str], bool]]:
    """state_id -> {(arm, lam): accepted}; states with an excluded cell are dropped."""
    check_d_exp1_all_unsafe(rows)
    per: dict[str, dict[tuple[str, str], bool | None]] = defaultdict(dict)
    for r in rows:
        per[r["state_id"]][(r["arm"], r["lam"])] = accepted(r, accounting)
    out: dict[str, dict[tuple[str, str], bool]] = {}
    for sid, cells in per.items():
        if any(v is None for v in cells.values()):
            continue
        out[sid] = {k: bool(v) for k, v in cells.items()}
    return out


def cell_rates(rows: list[dict[str, Any]], accounting: str = "main") -> dict[str, dict[str, Any]]:
    """FA(arm, lambda) = P(accept) over the D_Exp1 states in the cell (all E-UNSAFE)."""
    tab = _table(rows, accounting)
    cells: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for lam in LAMBDAS:
            acc = [tab[s][(arm, lam)] for s in tab if (arm, lam) in tab[s]]
            if not acc:
                continue
            cells[f"{arm}|{lam}"] = {
                "n": len(acc),
                "false_accept_rate": mean_or_nan([float(a) for a in acc]),
            }
    return cells


def effective_n(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """paper Sec. 4.6: the primary metric's N is D_Exp1 in full.

    ``n_states`` is |D_Exp1| as it appears in the log; ``n_did_units`` is how many of those
    states contribute a D_i under the main accounting (a state drops out only if one of the four
    primary cells is missing from the log).
    """
    check_d_exp1_all_unsafe(rows)
    return {
        "n_states": len({r["state_id"] for r in rows}),
        "n_did_units": len(_did_units(rows, "main")),
        "note": (
            "III.1.2 (v0.9): D_Exp1 holds E-UNSAFE states only, so the false accept rate's "
            "denominator is D_Exp1 in full and N = n_states; the N = 300 column of the III.1.3 "
            "feasibility table applies directly."
        ),
    }


def _did_units(
    rows,
    accounting,
    control="A1",
    treated="B",
    lam_hi=PRIMARY_LAMBDA_HIGH,
    lam_lo=PRIMARY_LAMBDA_LOW,
):
    tab = _table(rows, accounting)
    units = []
    for cells in tab.values():
        keys = [(control, lam_hi), (treated, lam_hi), (control, lam_lo), (treated, lam_lo)]
        if any(k not in cells for k in keys):
            continue
        d = (float(cells[keys[0]]) - float(cells[keys[1]])) - (
            float(cells[keys[2]]) - float(cells[keys[3]])
        )
        units.append(d)
    return units


def did(
    rows: list[dict[str, Any]],
    *,
    accounting: str = "main",
    control: str = "A1",
    treated: str = "B",
    seed: int = 0,
    n_boot: int = BOOTSTRAP_B,
) -> dict[str, Any]:
    units = _did_units(rows, accounting, control, treated)
    est = mean_or_nan(units)
    lo, hi = bootstrap_ci(units, mean_or_nan, n_boot=n_boot, seed=seed)
    return {
        "contrast": f"{control} - {treated}",
        # paper Sec. 4.6: the two extremes only
        "lambdas": [PRIMARY_LAMBDA_HIGH, PRIMARY_LAMBDA_LOW],
        "accounting": accounting,
        "n_states": len(units),
        "estimate": est,
        "ci95": [lo, hi],
        "bootstrap": bootstrap_spec("x0", seed=seed, n_boot=n_boot),
    }


def delta_by_lambda(
    rows: list[dict[str, Any]], accounting: str = "main", control="A1", treated="B"
) -> dict[str, float]:
    """Delta_FA(lambda) at all three levels — the SECONDARY result of paper Sec. 4.6 ."""
    cells = cell_rates(rows, accounting)
    out = {}
    for lam in LAMBDAS:
        a, b = cells.get(f"{control}|{lam}"), cells.get(f"{treated}|{lam}")
        if a and b:
            out[lam] = a["false_accept_rate"] - b["false_accept_rate"]
    return out


def result_shape(rows: list[dict[str, Any]], accounting: str = "main") -> dict[str, Any]:
    """The reading of the three-level curves, from the point estimates only.

    For A1 - B and for permuted_B - B: Delta_FA at low / mid / high and the two shapes
    names that need no tolerance -- rising in lambda (low < mid < high) and negative at
    lambda_low. "Delta ~ 0 everywhere" and "permuted-B ~ B" would need a tolerance the design does
    not give, so they get the values and no flag. Descriptive; the pilot tests nothing.
    """
    out: dict[str, Any] = {
        "accounting": accounting,
        "note": (
            "point estimates only; 'Delta ~ 0' and 'permuted-B ~ B' carry no flag: III.2.3 gives "
            "no tolerance for them"
        ),
    }
    for control in ("A1", "permuted_B"):
        d = delta_by_lambda(rows, accounting, control=control, treated="B")
        vals = [d.get(lam) for lam in LAMBDAS]
        complete = all(v is not None for v in vals)
        out[f"{control} - B"] = {
            "delta_fa_by_lambda": d,
            "rising_in_lambda": complete and vals[0] < vals[1] < vals[2],
            "negative_at_low": d.get(PRIMARY_LAMBDA_LOW) is not None and d["low"] < 0,
        }
    return out


# H2 strata (paper Sec. 4.6) ---------------------------------------------------------


def perturbation_index(channel_rows: list[dict[str, Any]]) -> dict[tuple[str, str], bool]:
    """``channel.jsonl`` -> {(state_id, lam): at least one verifier perturbed} ."""
    return {
        (r["state_id"], r["lam"]): bool(r.get("perturbed_v1")) or bool(r.get("perturbed_v2"))
        for r in channel_rows
    }


def any_verifier_perturbed(
    row: dict[str, Any], index: dict[tuple[str, str], bool] | None = None
) -> bool:
    """H2 stratifier (i). Uses the logged perturbation assignment when available; otherwise
    falls back to comparing the raw and post-channel scores on the row itself."""
    if index is not None:
        hit = index.get((row["state_id"], row["lam"]))
        if hit is not None:
            return hit
    return row["s1"] != row["s1_tilde"] or row["s2"] != row["s2_tilde"]


def observed_disagreement(row: dict[str, Any]) -> bool:
    """H2 stratifier (ii): do the two verifiers disagree in the OBSERVED evidence?

    y_i = 1[s_i^obs >= tau_i] on the post-channel scores — exactly what the formatter shows.
    """
    return (row["s1_tilde"] >= row["tau1"]) != (row["s2_tilde"] >= row["tau2"])


def _stratum_consistency(rows, key) -> int:
    """How many (state, lambda) cells have a stratum value that differs across arms.

    Must be 0: the three arms see identical evidence for the same (state, lambda) (paper Sec. 4.3),
    which is what makes these stratifiers arm-independent rather than post-treatment.
    """
    seen: dict[tuple[str, str], set[bool]] = defaultdict(set)
    for r in rows:
        seen[(r["state_id"], r["lam"])].add(key(r))
    return sum(1 for v in seen.values() if len(v) > 1)


def h2_strata(
    rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]] | None = None,
    *,
    accounting: str = "main",
    control: str = "A1",
    treated: str = "B",
) -> dict[str, Any]:
    """Delta_FA(lambda) inside the two arm-independent strata of paper Sec. 4.6.

    Mechanism diagnostic, not a causal test. Both stratifiers are functions of the (state,
    lambda) evidence alone, which every arm shares, so conditioning on them does not condition
    on anything downstream of treatment. r_hat_ref is NOT a stratifier and is not an input
    here; the runner reports it next to this block (paper Sec. 3).
    """
    index = perturbation_index(channel_rows) if channel_rows is not None else None
    stratifiers = {
        "any_verifier_perturbed": (
            lambda r: any_verifier_perturbed(r, index),
            "(i) at least one verifier was perturbed at this (state, lambda)",
        ),
        "observed_disagreement": (
            observed_disagreement,
            "(ii) the two verifiers disagree in the observed evidence",
        ),
    }
    out: dict[str, Any] = {
        "source": "channel.jsonl" if index is not None else "s vs s_tilde on the decision rows",
        "accounting": accounting,
        "contrast": f"{control} - {treated}",
        "note": (
            "mechanism diagnostic (III.1.2), not a causal test; both stratifiers are "
            "arm-independent, so this is not post-treatment conditioning. r_hat_ref is "
            "reported alongside and never used to stratify (II.1.1)."
        ),
    }
    for name, (key, desc) in stratifiers.items():
        block: dict[str, Any] = {
            "definition": desc,
            "cells_differing_across_arms": _stratum_consistency(rows, key),
            "strata": {},
        }
        for value in (True, False):
            subset = [r for r in rows if key(r) is value]
            cells = cell_rates(subset, accounting) if subset else {}
            block["strata"][str(value)] = {
                "n_rows": len(subset),
                "n_states": len({r["state_id"] for r in subset}),
                "delta_fa_by_lambda": delta_by_lambda(subset, accounting, control, treated)
                if subset
                else {},
                "cells": cells,
            }
        out[name] = block
    return out


def analyze(
    rows: list[dict[str, Any]],
    *,
    seed: int = 0,
    n_boot: int = BOOTSTRAP_B,
    channel_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    primary = {acc: did(rows, accounting=acc, seed=seed, n_boot=n_boot) for acc in ACCOUNTINGS}
    secondary = {
        acc: did(rows, accounting=acc, control="permuted_B", treated="B", seed=seed, n_boot=n_boot)
        for acc in ACCOUNTINGS
    }
    return {
        "sign_convention": "A1 minus B; positive = B better (I.2)",
        "bootstrap": bootstrap_spec("x0", seed=seed, n_boot=n_boot),
        "effective_n": effective_n(rows),
        "cells_main": cell_rates(rows, "main"),
        "primary_did": primary,
        "primary_lambdas": [PRIMARY_LAMBDA_HIGH, PRIMARY_LAMBDA_LOW],
        "primary_robust_across_accountings": same_sign(
            [primary[a]["estimate"] for a in ACCOUNTINGS]
        ),
        "secondary_delta_fa_by_lambda_main": {
            "values": delta_by_lambda(rows, "main"),
            "note": (
                "III.1.2 (v0.8): all three levels, SECONDARY result; lambda_mid is not part "
                "of the primary DiD. Read for the monotonic shape of III.2.3."
            ),
        },
        "secondary_permutedB_did": secondary,
        "secondary_permutedB_delta_fa_by_lambda_main": delta_by_lambda(
            rows, "main", control="permuted_B", treated="B"
        ),
        "result_shape": result_shape(rows),
        "h2_strata": h2_strata(rows, channel_rows),
        "pilot_note": "directional evidence only; no significance claims (III.1.1)",
    }
