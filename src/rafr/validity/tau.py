"""tau_i selection on D_fit (II.4.1, v0.9): maximize F1; on a plateau take the midpoint.

v0.9 (C.17) replaced F2 with the symmetric F1, so that the threshold choice is no longer
coupled to the direction of the primary metric; F2 is still reported, as an auxiliary.

Candidate thresholds are the midpoints between consecutive distinct scores plus the two
extremes; the judgment is ``y = 1[s >= tau]``. A plateau is a maximal run of consecutive
candidates that all attain the maximum F1; tau is the midpoint of that run's threshold
interval. F2, PR-AUC, precision, recall, balanced accuracy are reported as auxiliaries and
never take part in the selection. The result is frozen to ``frozen/tau.json`` (tag
``tau-frozen``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rafr.contracts.schemas import TAU_CRITERION
from rafr.core.textio import crlf_rewrite_notice
from rafr.validity import metrics as mt

PLATEAU_TOL = 1e-12


@dataclass
class TauFit:
    verifier_id: str
    tau: float
    f1_max: float
    plateau_lo: float
    plateau_hi: float
    n_plateaus: int  # number of disjoint maximal runs; > 1 is reported for the authors
    n_candidates: int
    n: int
    n_positive: int
    criterion: str
    aux: dict[str, float]


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    u = np.unique(scores)
    if len(u) == 1:
        return np.array([u[0]])
    mids = (u[:-1] + u[1:]) / 2
    return np.concatenate(([u[0]], mids, [u[-1] + (u[-1] - u[-2]) / 2]))


def fit_tau(scores, labels, verifier_id: str) -> TauFit:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if len(s) != len(y) or len(s) == 0:
        raise ValueError("scores and labels must be non-empty and aligned")
    cands = candidate_thresholds(s)
    f1s = np.array([mt.f1(y, (s >= c).astype(int)) for c in cands])
    best = float(f1s.max())
    is_max = np.abs(f1s - best) <= PLATEAU_TOL
    # maximal runs of consecutive maximizers
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(cands):
        if is_max[i]:
            j = i
            while j + 1 < len(cands) and is_max[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    # widest run; ties -> first (reported via n_plateaus for the authors)
    widths = [cands[j] - cands[i] for i, j in runs]
    ri = int(np.argmax(widths))
    lo, hi = float(cands[runs[ri][0]]), float(cands[runs[ri][1]])
    tau = (lo + hi) / 2
    pred = (s >= tau).astype(int)
    prec, rec = mt.precision_recall(y, pred)
    aux = {
        "pr_auc": mt.pr_auc(s, y),
        "auroc": mt.auroc(s, y),
        "precision_at_tau": prec,
        "recall_at_tau": rec,
        "balanced_accuracy_at_tau": mt.balanced_accuracy(y, pred),
        "f2_at_tau": mt.f2(y, pred),
        "accuracy_at_tau": mt.accuracy(y, pred),
    }
    return TauFit(
        verifier_id=verifier_id,
        tau=float(tau),
        f1_max=best,
        plateau_lo=lo,
        plateau_hi=hi,
        n_plateaus=len(runs),
        n_candidates=int(len(cands)),
        n=int(len(s)),
        n_positive=int(y.sum()),
        criterion=TAU_CRITERION,
        aux=aux,
    )


def save_tau(fits: dict[str, TauFit], path: Path, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fits": {k: asdict(v) for k, v in fits.items()}, **(extra or {})}
    with crlf_rewrite_notice(path), path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)


def load_tau(path: Path) -> dict[str, TauFit]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {k: TauFit(**v) for k, v in data["fits"].items()}
