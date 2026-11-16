"""Small, dependency-free classification metrics used by tau fitting, the gate and the
diagnostics. Positive class = UNSAFE (label 1); scores are higher-is-more-toxic."""

from __future__ import annotations

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def confusion(y_true, y_pred) -> tuple[int, int, int, int]:
    t, p = _arr(y_true).astype(int), _arr(y_pred).astype(int)
    tp = int(((t == 1) & (p == 1)).sum())
    fp = int(((t == 0) & (p == 1)).sum())
    fn = int(((t == 1) & (p == 0)).sum())
    tn = int(((t == 0) & (p == 0)).sum())
    return tp, fp, fn, tn


def precision_recall(y_true, y_pred) -> tuple[float, float]:
    tp, fp, fn, _ = confusion(y_true, y_pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec


def f_beta(y_true, y_pred, beta: float) -> float:
    prec, rec = precision_recall(y_true, y_pred)
    if prec == 0.0 and rec == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def f2(y_true, y_pred) -> float:
    return f_beta(y_true, y_pred, 2.0)


def f1(y_true, y_pred) -> float:
    return f_beta(y_true, y_pred, 1.0)


def accuracy(y_true, y_pred) -> float:
    t, p = _arr(y_true), _arr(y_pred)
    return float((t == p).mean()) if len(t) else float("nan")


def balanced_accuracy(y_true, y_pred) -> float:
    tp, fp, fn, tn = confusion(y_true, y_pred)
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return (tpr + tnr) / 2


def auroc(scores, y_true) -> float:
    """Rank-based AUROC (Mann-Whitney), ties get half credit. NaN if one class is absent."""
    s, t = _arr(scores), _arr(y_true).astype(int)
    pos, neg = s[t == 1], s[t == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1  # average rank, 1-based
        i = j + 1
    rank_sum_pos = ranks[t == 1].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pr_auc(scores, y_true) -> float:
    """Area under the precision-recall curve (step-wise, average precision)."""
    s, t = _arr(scores), _arr(y_true).astype(int)
    if t.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    t = t[order]
    tp = np.cumsum(t)
    fp = np.cumsum(1 - t)
    prec = tp / (tp + fp)
    ap = float((prec * t).sum() / t.sum())
    return ap


def brier(probs, y_true) -> float:
    p, t = _arr(probs), _arr(y_true)
    return float(((p - t) ** 2).mean()) if len(p) else float("nan")


def ece(probs, y_true, n_bins: int = 10) -> float:
    p, t = _arr(probs), _arr(y_true)
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - t[mask].mean())
    return float(total)


def phi(a, b) -> float:
    """Phi coefficient between two binary vectors (e.g. error indicators)."""
    tp, fp, fn, tn = confusion(a, b)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den else float("nan")


def cohen_kappa(a, b) -> float:
    a, b = _arr(a).astype(int), _arr(b).astype(int)
    if len(a) == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return float((po - pe) / (1 - pe)) if pe != 1.0 else float("nan")
