"""Descriptive cue diagnostics (II.6.2) — the five items, per lambda and per verifier.

1. AUROC(|m_i| -> c^(i))
2. disagreement rate P(y1 != y2)
3. verifier error rate conditional on agreement / disagreement
4. high-margin-but-wrong share
5. joint (m1, m2) distribution (2-D histogram)

Reported, never used for a decision. The same quantities at lambda = 0 on D_fit are the
baseline. Must be computed and written BEFORE anyone looks at A1 - B; ``step09`` writes them
and ``step11`` refuses to run without that file. Which states they are computed on is the
caller's (II.6.2, v1.0: the E-UNSAFE subset primary, the whole split secondary; step08 /
step09 pass each population in turn).

"High margin": the cut is ``high_margin_quantile`` of |m_i| (``configs/channel.yaml``; the authors
2026-09-17, the 0.75 quantile; App. D v1.0 defers to the configured value) — the share of
wrong judgments among those with |m_i| at or above that quantile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rafr.validity import metrics as mt


@dataclass
class CueInputs:
    """Post-channel observations with truth, for one lambda level."""

    s1_obs: list[float]
    s2_obs: list[float]
    tau1: float
    tau2: float
    y_true: list[int]  # 1 = UNSAFE

    def margins(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.s1_obs, dtype=float) - self.tau1,
            np.asarray(self.s2_obs, dtype=float) - self.tau2,
        )

    def judgments(self) -> tuple[np.ndarray, np.ndarray]:
        m1, m2 = self.margins()
        return (m1 >= 0).astype(int), (m2 >= 0).astype(int)

    def correctness(self) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(self.y_true, dtype=int)
        y1, y2 = self.judgments()
        return (y1 == y).astype(int), (y2 == y).astype(int)


@dataclass
class CueReport:
    n: int
    auroc_margin_to_correct: dict[str, float]
    disagreement_rate: float
    error_rate_given_agreement: dict[str, float]
    error_rate_given_disagreement: dict[str, float]
    high_margin_wrong_share: dict[str, float]
    high_margin_quantile: float
    joint_margin_hist: dict[str, Any] = field(default_factory=dict)
    reliability: dict[str, float] = field(default_factory=dict)  # R^(i) on this data
    p_at_least_one_wrong: float = float("nan")


def compute(inputs: CueInputs, *, high_margin_quantile: float, n_bins: int = 10) -> CueReport:
    m1, m2 = inputs.margins()
    y1, y2 = inputs.judgments()
    c1, c2 = inputs.correctness()
    n = len(m1)
    disagree = y1 != y2
    agree = ~disagree

    def _rate(mask: np.ndarray, c: np.ndarray) -> float:
        return float(1 - c[mask].mean()) if mask.any() else float("nan")

    def _high_wrong(m: np.ndarray, c: np.ndarray) -> float:
        cut = float(np.quantile(np.abs(m), high_margin_quantile))
        mask = np.abs(m) >= cut
        return float(1 - c[mask].mean()) if mask.any() else float("nan")

    lo1, hi1 = float(np.min(m1)), float(np.max(m1))
    lo2, hi2 = float(np.min(m2)), float(np.max(m2))
    hist, e1, e2 = np.histogram2d(
        m1,
        m2,
        bins=n_bins,
        range=[[lo1, hi1 if hi1 > lo1 else lo1 + 1e-9], [lo2, hi2 if hi2 > lo2 else lo2 + 1e-9]],
    )
    return CueReport(
        n=n,
        auroc_margin_to_correct={"v1": mt.auroc(np.abs(m1), c1), "v2": mt.auroc(np.abs(m2), c2)},
        disagreement_rate=float(disagree.mean()) if n else float("nan"),
        error_rate_given_agreement={"v1": _rate(agree, c1), "v2": _rate(agree, c2)},
        error_rate_given_disagreement={"v1": _rate(disagree, c1), "v2": _rate(disagree, c2)},
        high_margin_wrong_share={"v1": _high_wrong(m1, c1), "v2": _high_wrong(m2, c2)},
        high_margin_quantile=float(high_margin_quantile),
        joint_margin_hist={
            "counts": hist.astype(int).tolist(),
            "m1_edges": [float(x) for x in e1],
            "m2_edges": [float(x) for x in e2],
        },
        reliability={"v1": float(c1.mean()), "v2": float(c2.mean())},
        p_at_least_one_wrong=float(((c1 == 0) | (c2 == 0)).mean()) if n else float("nan"),
    )
