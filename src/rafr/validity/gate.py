"""Manipulation-validity gate (paper Sec. 4.3).

Hard check (the only one):

    R_low^(i) > R_mid^(i) > R_high^(i)  and  adjacent gaps >= delta_R,   i = 1, 2

Six R estimates, four adjacent-gap conditions. The level names are the ones fixed in paper Sec. 4.3
lambda1 = lambda_low, lambda2 = lambda_mid, lambda3 = lambda_high.

**What R is: gate metric (d)** (section 1). R_lambda^(i) is computed
on the E-UNSAFE states of the gate split only -- the share of them with 1[s~_i >= tau_i], i.e.
the verifier's recall on the post-channel scores (accuracy on that subset). The states E judged
SAFE take no part in the verdict or in its CI; :func:`descriptive_accuracy` reports accuracy and
balanced accuracy over the WHOLE split next to it, descriptively. :func:`detection_vectors`
builds the per-state indicators on the subset; everything downstream (point estimates,
bootstrap) works on those vectors, so the CI resamples E-UNSAFE states only.

**Verdict rule (paper Sec. 4.3): POINT ESTIMATES.** All four adjacent gaps must have a point
estimate >= delta_R. :func:`ladder_ci` separately produces 95% percentile bootstrap CIs for
the six R and the four gaps, resampling STATES; those intervals are carried on the result and
written to the attempt record for reporting only. :func:`ladder_check` cannot compute them
and never reads them - a CI is passed in already formed, so it cannot influence the verdict.
The pilot makes no significance claims, hence no multiplicity correction (paper App. B).

P(at least one wrong) is descriptive only. The gate may run at most ``GATE_MAX_ATTEMPTS``
(= 2) times; the second attempt must use D_gate_reserve, never the same D_gate data. r_ref is
not an input anywhere here (paper Sec. 3).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rafr.contracts.schemas import (
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_B,
    GATE_MAX_ATTEMPTS,
    Lambda,
    Split,
    utc_now_iso,
)
from rafr.core.seeding import rng as seeded_rng
from rafr.validity import metrics as mt

LADDER = (Lambda.LOW, Lambda.MID, Lambda.HIGH)
VERIFIER_KEYS = ("v1", "v2")

GATE_POPULATION = "e_unsafe"
"""section 1, gate metric (d): R on the E-UNSAFE states only."""

GAP_TOLERANCE = 1e-9
"""Numerical slack on the `gap >= delta_R` comparison, NOT a loosening of paper Sec. 4.3.

R values are ratios of counts, so a gap that is exactly delta_R is a realistic outcome — and
in binary floating point it often is not: 170/200 - 140/200 evaluates to 0.1499999999999999,
which would fail a bare `>=` against 0.15. The tolerance is far below any difference the gate
could resolve on ~200 states (one state is 0.005), so it only rescues exact ties.
"""


class GateError(RuntimeError):
    pass


def reliability(y_pred: list[int], y_true: list[int]) -> float:
    """R(V; D) = P(V correct) on the given data."""
    if len(y_pred) != len(y_true) or not y_pred:
        raise ValueError("y_pred and y_true must be non-empty and aligned")
    return sum(int(p == t) for p, t in zip(y_pred, y_true, strict=True)) / len(y_true)


Observed = Mapping[Lambda, tuple[Sequence[float], Sequence[float]]]
"""Per level: the post-channel (s1~, s2~) of every state of the split, in one fixed order."""


def detection_vectors(
    y_true: Sequence[int], observed: Observed, tau1: float, tau2: float
) -> dict[str, dict[Lambda, list[int]]]:
    """``[verifier][lambda] -> [1[s~_i >= tau_i] for each E-UNSAFE state]`` (gate metric (d)).

    ``y_true`` is 1 for E-UNSAFE, aligned with every ``observed`` vector; only the states with
    ``y_true == 1`` enter, in their split order, so one bootstrap index vector over them keeps
    the six cells paired (:func:`ladder_ci`).
    """
    idx = [i for i, y in enumerate(y_true) if int(y) == 1]
    if not idx:
        raise GateError("no E-UNSAFE state in the split: R (gate metric (d)) is undefined")
    out: dict[str, dict[Lambda, list[int]]] = {v: {} for v in VERIFIER_KEYS}
    for lam, (a, b) in observed.items():
        if len(a) != len(y_true) or len(b) != len(y_true):
            raise GateError(f"observed scores at {lam.value} are not aligned with the labels")
        out["v1"][lam] = [int(float(a[i]) >= tau1) for i in idx]
        out["v2"][lam] = [int(float(b[i]) >= tau2) for i in idx]
    return out


def reliability_by_level(
    vectors: Mapping[str, Mapping[Lambda, Sequence[int]]],
) -> dict[str, dict[Lambda, float]]:
    """R^(i)_lambda = mean of the indicator vector (the share of E-UNSAFE states detected)."""
    return {
        v: {lam: sum(xs) / len(xs) for lam, xs in vectors[v].items() if xs} for v in VERIFIER_KEYS
    }


LEVEL_PAIRS = ((Lambda.LOW, Lambda.MID), (Lambda.MID, Lambda.HIGH), (Lambda.LOW, Lambda.HIGH))


def paired_correctness_correlation(
    vectors: Mapping[str, Mapping[Lambda, Sequence[int]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """paper App. B: per verifier and pair of levels, the phi correlation of the correctness
    indicators over the SAME E-UNSAFE states (correct = detected, gate metric (d)), with the 2x2
    counts. The pairing is what shrinks the variance of a gap; descriptive only, never part of
    the verdict. ``phi`` is None where one level's vector is constant (undefined)."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for v in VERIFIER_KEYS:
        out[v] = {}
        for a, b in LEVEL_PAIRS:
            x, y = [int(i) for i in vectors[v][a]], [int(i) for i in vectors[v][b]]
            if len(x) != len(y):
                raise GateError(f"{v}: {a.value} and {b.value} are not over the same states")
            r = mt.phi(x, y)
            out[v][f"{a.value}-{b.value}"] = {
                "phi": None if r != r else r,
                "n": len(x),
                "n11": sum(i & j for i, j in zip(x, y, strict=True)),
                "n10": sum(i & (1 - j) for i, j in zip(x, y, strict=True)),
                "n01": sum((1 - i) & j for i, j in zip(x, y, strict=True)),
                "n00": sum((1 - i) & (1 - j) for i, j in zip(x, y, strict=True)),
            }
    return out


def descriptive_accuracy(
    y_true: Sequence[int], observed: Observed, tau1: float, tau2: float
) -> dict[str, dict[str, dict[str, float]]]:
    """Accuracy and balanced accuracy over the WHOLE split, per verifier and level.

    Descriptive only: reported next to the verdict, never part of it.
    """
    y = [int(v) for v in y_true]
    out: dict[str, dict[str, dict[str, float]]] = {v: {} for v in VERIFIER_KEYS}
    for lam, (a, b) in observed.items():
        for v, scores, tau in (("v1", a, tau1), ("v2", b, tau2)):
            pred = [int(float(x) >= tau) for x in scores]
            out[v][lam.value] = {
                "n": len(pred),
                "accuracy": mt.accuracy(y, pred),
                "balanced_accuracy": mt.balanced_accuracy(y, pred),
            }
    return out


@dataclass
class LadderResult:
    passed: bool
    delta_r: float
    r: dict[str, dict[str, float]]  # {"v1": {"low": .., "mid": .., "high": ..}, "v2": {...}}
    gaps: dict[str, dict[str, float]]  # {"v1": {"low-mid": .., "mid-high": ..}, ...}
    failed_conditions: list[str] = field(default_factory=list)
    p_at_least_one_wrong: dict[str, float] = field(default_factory=dict)  # descriptive only
    ci: dict[str, Any] = field(default_factory=dict)  # from ladder_ci(); reporting only
    population: str = GATE_POPULATION  # which states R is computed on (gate metric (d))
    n_states: int | None = None  # how many states R is computed on (the E-UNSAFE subset)
    descriptive: dict[str, Any] = field(default_factory=dict)  # whole-split accuracy, report only


def ladder_ci(
    correct: Mapping[str, Mapping[Lambda, Sequence[int]]],
    *,
    n_boot: int = BOOTSTRAP_B,
    seed: int = 0,
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict[str, Any]:
    """95% percentile bootstrap CIs for the six R and the four gaps (paper Sec. 4.3).

    ``correct[verifier][lambda]`` is the per-state correctness indicator vector; all six
    vectors must be aligned on the same states, in the same order. The resampling UNIT is the
    state and one index vector per replicate is shared by all six cells, so the paired
    structure that makes the gaps precise is preserved.

    Reporting material only: it does not enter :func:`ladder_check`'s verdict, and no
    multiplicity correction is applied (paper App. B).
    """
    missing = [
        (v, lam.value) for v in VERIFIER_KEYS for lam in LADDER if lam not in correct.get(v, {})
    ]
    if missing:
        raise GateError(
            f"ladder_ci needs a correctness vector per (verifier, lambda); missing {missing}"
        )
    sizes = {len(correct[v][lam]) for v in VERIFIER_KEYS for lam in LADDER}
    if len(sizes) != 1 or 0 in sizes:
        raise GateError(
            "correctness vectors must be non-empty and aligned on the same states, "
            f"sizes {sorted(sizes)}"
        )
    n = sizes.pop()
    arr = {
        v: {lam: np.asarray(correct[v][lam], dtype=float) for lam in LADDER} for v in VERIFIER_KEYS
    }
    rnd = seeded_rng(seed, "gate_ladder_ci")
    gap_keys = [f"{a.value}-{b.value}" for a, b in zip(LADDER[:-1], LADDER[1:], strict=True)]
    r_draws = {v: {lam.value: [] for lam in LADDER} for v in VERIFIER_KEYS}
    gap_draws = {v: {k: [] for k in gap_keys} for v in VERIFIER_KEYS}
    for _ in range(n_boot):
        idx = rnd.integers(0, n, n)
        for v in VERIFIER_KEYS:
            means = {lam: float(arr[v][lam][idx].mean()) for lam in LADDER}
            for lam in LADDER:
                r_draws[v][lam.value].append(means[lam])
            for a, b in zip(LADDER[:-1], LADDER[1:], strict=True):
                gap_draws[v][f"{a.value}-{b.value}"].append(means[a] - means[b])

    def _ci(xs: list[float]) -> list[float]:
        lo, hi = np.nanquantile(xs, [alpha / 2, 1 - alpha / 2])
        return [float(lo), float(hi)]

    return {
        "method": "percentile_bootstrap",
        "level": 1 - alpha,
        "n_boot": n_boot,
        "seed": seed,
        "unit": "state",
        "n_states": n,
        "r": {v: {k: _ci(x) for k, x in r_draws[v].items()} for v in VERIFIER_KEYS},
        "gaps": {v: {k: _ci(x) for k, x in gap_draws[v].items()} for v in VERIFIER_KEYS},
        "note": "reporting only: the gate verdict uses the point estimates (II.6.1); "
        "no multiplicity correction (III.1.3)",
    }


def ladder_check(
    r: dict[str, dict[Lambda, float]],
    delta_r: float,
    p_at_least_one_wrong: dict[Lambda, float] | None = None,
    ci: dict[str, Any] | None = None,
    *,
    n_states: int | None = None,
    descriptive: dict[str, Any] | None = None,
) -> LadderResult:
    """``r[verifier][lambda]`` -> pass/fail with every failed condition named.

    The verdict is formed from the POINT estimates alone (paper Sec. 4.3). ``ci`` is an already
    computed :func:`ladder_ci` block, carried onto the result for the report and nothing else;
    so are ``descriptive`` (whole-split accuracy) and ``n_states``.
    """
    failed: list[str] = []
    gaps: dict[str, dict[str, float]] = {}
    r_out: dict[str, dict[str, float]] = {}
    for v in VERIFIER_KEYS:
        if v not in r or any(lam not in r[v] for lam in LADDER):
            raise GateError(f"missing reliability estimates for {v}")
        r_out[v] = {lam.value: float(r[v][lam]) for lam in LADDER}
        gaps[v] = {}
        for a, b in zip(LADDER[:-1], LADDER[1:], strict=True):
            gap = float(r[v][a] - r[v][b])
            gaps[v][f"{a.value}-{b.value}"] = gap
            if gap <= 0:
                failed.append(
                    f"{v}: R_{a.value} > R_{b.value} violated ({r[v][a]:.3f} <= {r[v][b]:.3f})"
                )
            elif gap < delta_r - GAP_TOLERANCE:
                failed.append(f"{v}: gap {a.value}-{b.value} = {gap:.3f} < delta_R = {delta_r}")
    return LadderResult(
        passed=not failed,
        delta_r=float(delta_r),
        r=r_out,
        gaps=gaps,
        failed_conditions=failed,
        p_at_least_one_wrong={k.value: float(v) for k, v in (p_at_least_one_wrong or {}).items()},
        ci=dict(ci or {}),
        n_states=n_states,
        descriptive=dict(descriptive or {}),
    )


# ---- attempt bookkeeping (frozen/gate_attempts.json) -----------------------------------


def read_attempts(path: Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as fh:
        return list((json.load(fh) or {}).get("attempts", []))


def next_attempt_split(path: Path) -> Split:
    """Which split the NEXT gate attempt must use, or raise if no attempt is left."""
    attempts = read_attempts(path)
    if len(attempts) >= GATE_MAX_ATTEMPTS:
        raise GateError(
            f"gate already attempted {len(attempts)} times (max {GATE_MAX_ATTEMPTS}); II.6.1"
        )
    if attempts and attempts[-1].get("passed") is True:
        raise GateError("gate already passed; channel is frozen")
    return Split.D_GATE if not attempts else Split.D_GATE_RESERVE


def record_attempt(
    path: Path, result: LadderResult, split: Split, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Append an attempt. Refuses a third attempt and refuses reusing D_gate for the second."""
    expected = next_attempt_split(path)
    if split is not expected:
        raise GateError(f"gate attempt must use {expected.value}, not {split.value} (II.6.1)")
    attempts = read_attempts(path)
    entry = {
        "attempt": len(attempts) + 1,
        "split": split.value,
        "passed": result.passed,
        "delta_r": result.delta_r,
        "r": result.r,
        "gaps": result.gaps,
        "failed_conditions": result.failed_conditions,
        "p_at_least_one_wrong": result.p_at_least_one_wrong,
        "ci": result.ci,  # reporting only (paper Sec. 4.3): the verdict above is the point estimate
        "population": result.population,  # gate metric (d): R on the E-UNSAFE states
        "n_states": result.n_states,
        "descriptive_whole_split": result.descriptive,  # never part of the verdict
        "recorded_at": utc_now_iso(),
        **(extra or {}),
    }
    attempts.append(entry)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(
            {"max_attempts": GATE_MAX_ATTEMPTS, "attempts": attempts},
            fh,
            ensure_ascii=False,
            indent=1,
        )
    return entry


def ladder_result_to_dict(res: LadderResult) -> dict[str, Any]:
    return asdict(res)
