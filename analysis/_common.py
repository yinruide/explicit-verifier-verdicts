"""Shared helpers for the analysis scripts: accounting rules (II.8.2) and bootstrap CIs.

Interval algorithm (III.1.2, locked in v0.8): every CI in the analysis is a 95% PERCENTILE
bootstrap with B = ``BOOTSTRAP_B`` = 10000 replicates and a seed that is recorded and written
to disk. What differs between analyses is only the resampling unit:

* Exp 1  -> the x0 (one state per x0), via :func:`bootstrap_ci`;
* Exp 2  -> trajectories resampled WITHIN arm (stratified), via :func:`stratified_bootstrap_ci`;
* gate   -> the state, in ``rafr.validity.gate.ladder_ci``.

The pilot makes no significance claims (III.1.1): intervals express uncertainty, they never
decide anything — not the gate verdict (II.6.1) and not a claim's direction.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rafr.contracts.schemas import BOOTSTRAP_ALPHA, BOOTSTRAP_B
from rafr.core.seeding import rng as seeded_rng

Accounting = str  # "main" | "parse_fail_as_accept" | "complete_case"
ACCOUNTINGS: tuple[Accounting, ...] = ("main", "parse_fail_as_accept", "complete_case")

# ---- fake data (0.5 step 10) ------------------------------------------------------------
# Step 10 runs every analysis on fake logs before any real Exp 1 / Exp 2 row exists. The fake
# logs have the real row schemas, so nothing in a row could tell them apart -- except these
# marks, which the fake generator always writes and every reader checks:

FAKE_MARKER = "FAKE"
"""A file of this name in a run directory (or in any directory above it) marks it fake."""

FAKE_RUN_PREFIX = "FAKE-"
"""Every fake log row's ``run_id`` starts with this."""

FAKE_FILE_PREFIX = "FAKE_"
"""Every table step 10 writes is named ``FAKE_<table>``."""


class FakeDataError(RuntimeError):
    """A formal reader met fake data, or the fake dry run met data that is not fake."""


def fake_marker_dir(path: Path) -> Path | None:
    """The directory holding the FAKE marker that covers ``path``, if any."""
    p = Path(path).resolve()
    for d in (p, *p.parents):
        if (d / FAKE_MARKER).is_file():
            return d
    return None


def rows_marked_fake(rows: Iterable[Mapping[str, Any]]) -> bool:
    return any(str(r.get("run_id") or "").startswith(FAKE_RUN_PREFIX) for r in rows)


def refuse_fake(path: Path, rows: Iterable[Mapping[str, Any]] = (), *, what: str) -> None:
    """Formal readers (step11 / step12 / the analysis in formal mode) never read fake data."""
    marker = fake_marker_dir(path)
    if marker is not None or rows_marked_fake(rows) or FAKE_FILE_PREFIX in Path(path).name:
        raise FakeDataError(
            f"{what}: {path} is step 10's fake data (marker "
            f"{marker / FAKE_MARKER if marker else 'FAKE- run_id / FAKE_ file name'}); the formal "
            "pipeline does not read it"
        )


def require_fake(path: Path, rows: Iterable[Mapping[str, Any]] = (), *, what: str) -> None:
    """Step 10 reads nothing but fake data: no real label, score or controller output."""
    rows = list(rows)
    if fake_marker_dir(path) is None:
        raise FakeDataError(f"{what}: {path} carries no FAKE marker; step 10 reads fake data only")
    real = [
        r.get("run_id")
        for r in rows
        if "run_id" in r and not str(r.get("run_id") or "").startswith(FAKE_RUN_PREFIX)
    ]
    if real:
        raise FakeDataError(
            f"{what}: {len(real)} row(s) in {path} are not fake (run_id {real[0]!r}); step 10 "
            "reads fake data only"
        )


def accepted(row: dict[str, Any], accounting: Accounting = "main") -> bool | None:
    """Accept indicator under an accounting rule; ``None`` = excluded (complete-case)."""
    if row["parse_fail"]:
        if accounting == "main":
            return False  # II.8.2: parse_fail counts as NOT accepted
        if accounting == "parse_fail_as_accept":
            return True
        return None
    return row["action"] == "accept"


def bootstrap_ci(
    units: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
    *,
    n_boot: int = BOOTSTRAP_B,
    seed: int = 0,
    alpha: float = BOOTSTRAP_ALPHA,
) -> tuple[float, float]:
    """Percentile bootstrap over analysis units (x0 / state); deterministic in ``seed``."""
    units = list(units)
    if not units:
        return float("nan"), float("nan")
    r = seeded_rng(seed, "bootstrap")
    n = len(units)
    stats = []
    for _ in range(n_boot):
        idx = r.integers(0, n, n)
        stats.append(statistic([units[i] for i in idx]))
    lo, hi = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def stratified_bootstrap_ci(
    strata: Mapping[str, Sequence[Any]],
    statistic: Callable[[Mapping[str, Sequence[Any]]], float],
    *,
    n_boot: int = BOOTSTRAP_B,
    seed: int = 0,
    alpha: float = BOOTSTRAP_ALPHA,
) -> tuple[float, float]:
    """Percentile bootstrap that resamples WITHIN each stratum, keeping stratum sizes fixed.

    Exp 2 (III.1.2): the strata are the arms, so each replicate holds the randomized arm
    sizes constant instead of letting the pooled draw move them around.
    """
    strata = {k: list(v) for k, v in strata.items()}
    if not strata or any(not v for v in strata.values()):
        return float("nan"), float("nan")
    r = seeded_rng(seed, "bootstrap")
    stats = []
    for _ in range(n_boot):
        draw = {}
        for k, v in strata.items():
            idx = r.integers(0, len(v), len(v))
            draw[k] = [v[i] for i in idx]
        stats.append(statistic(draw))
    lo, hi = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_spec(unit: str, *, seed: int, n_boot: int = BOOTSTRAP_B) -> dict[str, Any]:
    """The block every analysis output carries so the interval is reproducible (III.1.2)."""
    return {
        "method": "percentile_bootstrap",
        "level": 1 - BOOTSTRAP_ALPHA,
        "n_boot": n_boot,
        "seed": seed,
        "unit": unit,
    }


def mean_or_nan(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def same_sign(values: Sequence[float]) -> bool:
    vs = [v for v in values if v is not None and not np.isnan(v)]
    if len(vs) < 2:
        return False
    return all(v > 0 for v in vs) or all(v < 0 for v in vs)
