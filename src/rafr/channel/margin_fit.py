"""Margin-matched q (II.5.2, C.2): the empirical score distribution of each verifier.

Decision (group 6a, 2026-09-16): q is *independent resampling*. A perturbed verifier's score
is redrawn from that verifier's empirical score distribution on the unlabeled pool (no labels
involved). Because |m_i| = |s_i - tau_i| is a function of the score, resampling the score
from the pool automatically aligns the corrupted |m_i| distribution with the uncorrupted one
("corrupt 后的 |m_i| 幅度分布与未 corrupt 部分对齐"); the judgment may or may not flip.

The model stores the sorted scores; sampling is inverse-CDF on the empirical values (linear
within a step). A histogram of |m| is kept as a summary for the report only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPORT_BINS: int = 20
"""Resolution of the |m| histogram in the fit report (decision, 2026-09-17).

Not configuration: sampling draws a replacement score from the empirical inverse CDF and never
touches these bins, so the number only changed a picture in the step08 report.
"""


@dataclass
class MarginModel:
    verifier_id: str
    tau: float
    scores: list[float]  # sorted empirical scores on the unlabeled pool
    n_bins: int = REPORT_BINS
    histogram: dict[str, list[float]] = field(default_factory=dict)  # of |m| = |s - tau|

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def magnitudes(self) -> list[float]:
        """Sorted |m| implied by the stored scores (reporting only)."""
        return sorted(abs(s - self.tau) for s in self.scores)

    def sample_score(self, rng: np.random.Generator) -> float:
        """Inverse-CDF draw of a SCORE from the empirical distribution (linear within a step)."""
        if not self.scores:
            raise ValueError("empty margin model")
        u = rng.random()
        pos = u * (len(self.scores) - 1)
        lo = int(np.floor(pos))
        hi = min(lo + 1, len(self.scores) - 1)
        frac = pos - lo
        return float(self.scores[lo] * (1 - frac) + self.scores[hi] * frac)

    def prob_at_least(self, threshold: float) -> float:
        """P(s >= threshold) for s drawn by :meth:`sample_score` -- exactly, not by sampling.

        The draw is ``f(u)`` with ``u ~ U[0, 1)`` and ``f`` the piecewise-linear interpolation
        of the sorted scores at ``pos = u * (n - 1)``; ``f`` is non-decreasing, so the
        probability is ``1 - u*`` with ``u*`` the first ``u`` where ``f`` reaches the threshold.
        Uses the pool's scores only, never a label: with ``threshold = tau_i`` it is the
        all-resampled floor r_i = P_{s~q_i}(s >= tau_i) of the gate's recall (step08).
        """
        s = self.scores
        if not s:
            raise ValueError("empty margin model")
        t = float(threshold)
        n = len(s)
        if n == 1:
            return 1.0 if s[0] >= t else 0.0
        if s[0] >= t:
            return 1.0
        if s[-1] < t:
            return 0.0
        k = next(i for i in range(1, n) if s[i] >= t)  # s[k - 1] < t <= s[k]
        frac = (t - s[k - 1]) / (s[k] - s[k - 1])
        return float(1.0 - ((k - 1) + frac) / (n - 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifier_id": self.verifier_id,
            "tau": self.tau,
            "n": self.n,
            "n_bins": self.n_bins,
            "kind": "independent_resampling",
            "histogram": self.histogram,
            "scores": self.scores,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MarginModel:
        if "scores" not in d:
            raise ValueError(
                "margin model file predates the independent-resampling q (group 6a decision); "
                "re-run step08"
            )
        return cls(
            verifier_id=d["verifier_id"],
            tau=float(d["tau"]),
            scores=[float(x) for x in d["scores"]],
            n_bins=int(d.get("n_bins", REPORT_BINS)),
            histogram=d.get("histogram", {}),
        )


def fit_margin_model(
    scores: list[float], tau: float, verifier_id: str, n_bins: int = REPORT_BINS
) -> MarginModel:
    """Fit q for one verifier from raw scores on the unlabeled pool (no labels)."""
    if len(scores) == 0:
        raise ValueError("no scores to fit the margin model")
    arr = np.sort(np.asarray(scores, dtype=float))
    mags = np.abs(arr - float(tau))
    counts, edges = np.histogram(mags, bins=int(n_bins), range=(0.0, max(1e-9, float(mags.max()))))
    return MarginModel(
        verifier_id=verifier_id,
        tau=float(tau),
        scores=[float(x) for x in arr],
        n_bins=int(n_bins),
        histogram={"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]},
    )


def save_models(
    models: dict[str, MarginModel], path: Path, extra: dict[str, Any] | None = None
) -> None:
    payload = {"q": {k: m.to_dict() for k, m in models.items()}, **(extra or {})}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)


def load_models(path: Path) -> dict[str, MarginModel]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {k: MarginModel.from_dict(v) for k, v in data["q"].items()}
