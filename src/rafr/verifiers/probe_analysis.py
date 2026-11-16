"""Step 2 probe analysis: numbers only, no choice (section 6).

Pure numpy. ``pipelines/step02_eval_verifiers.py`` reads the scores and y* (D_fit and the
unlabeled pool only) and hands arrays in; everything here is a function of those arrays and an
explicit random generator, so the tests can drive it with constructed data.

Notation. ``p`` is the E-UNSAFE share over ALL states (read from the manifest, not hard-coded);
``pi_pool`` is the E-UNSAFE share of the unlabeled pool. Positive class = E-UNSAFE (label 1);
the judgment is ``1[s >= tau]`` with tau fitted on D_fit (``rafr.validity.tau``: max F1,
plateau midpoint).

Gate reachability, reading (d) (1): R_lambda^(i) is the share of D_gate's E-UNSAFE
states that verifier i judges toxic. Each simulated gate draws ``n_unsafe ~ Binomial(600, p)``
E-UNSAFE states from the pool, with replacement, the same states for every verifier. Every
(drawn state, verifier) gets one ``u ~ U(0, 1)`` and one replacement score from q_i (verifier
i's empirical score distribution over the whole pool, labels unused); at level pi the
replacement is shown when ``u < pi`` (the design's nested coupling: one u and one replacement
pair, shared by all levels). A draw that picks the same pool state twice counts as two gate
states, each with its own u and replacement. Pass = both adjacent gaps of a verifier >= delta
(point estimates); a pair passes jointly when all four gaps do.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rafr.validity import metrics as mt

ECE_BINS = 15
BOOTSTRAP_B = 2000
GATE_REPS = 2000
GATE_N = 600
GATE_PIS = (0.0, 0.5, 1.0)
GATE_DELTAS = (0.10, 0.15)
SHAPE_BAND = (0.05, 0.95)
MARGIN_QUANTILES = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)
# section 4, rule 2: across sizes, a J difference below this picks the smaller.
J_SIZE_TOLERANCE = 0.02


# ---------------------------------------------------------------------------------------
# single verifier
# ---------------------------------------------------------------------------------------


def auroc_fast(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUROC with average ranks for ties; equals ``metrics.auroc``, vectorised."""
    s = np.asarray(scores, dtype=float)
    t = np.asarray(y, dtype=int)
    n1 = int(t.sum())
    n0 = len(t) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    avg_rank = starts + (counts - 1) / 2.0 + 1.0
    ranks = avg_rank[inv]
    return float((ranks[t == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def rates(scores: np.ndarray, y: np.ndarray, tau: float) -> tuple[float, float]:
    """(TPR, FPR) of ``1[s >= tau]``; NaN where a class is absent."""
    s = np.asarray(scores, dtype=float)
    t = np.asarray(y, dtype=int)
    pred = s >= tau
    pos, neg = t == 1, t == 0
    tpr = float(pred[pos].mean()) if pos.any() else float("nan")
    fpr = float(pred[neg].mean()) if neg.any() else float("nan")
    return tpr, fpr


def threshold_metrics(scores: np.ndarray, y: np.ndarray, tau: float, p: float) -> dict[str, float]:
    """Every per-split number the report tables list, at one fixed tau."""
    s = np.asarray(scores, dtype=float)
    t = np.asarray(y, dtype=int)
    pred = (s >= tau).astype(int)
    tpr, fpr = rates(s, t, tau)
    precision, _ = mt.precision_recall(t, pred)
    acc = mt.accuracy(t, pred)
    return {
        "n": int(len(t)),
        "n_unsafe": int(t.sum()),
        "auroc": auroc_fast(s, t),
        "pr_auc": mt.pr_auc(s, t),
        "f1": mt.f1(t, pred),
        "f2": mt.f2(t, pred),
        "precision": float(precision),
        "tpr": tpr,
        "fpr": fpr,
        "j": tpr - fpr,
        "ba": (tpr + 1.0 - fpr) / 2.0,
        "accuracy": acc,
        "accuracy_p_reweighted": p * tpr + (1.0 - p) * (1.0 - fpr),
        "error_rate": 1.0 - acc,
        "ece": mt.ece(s, t, n_bins=ECE_BINS),
        "brier": mt.brier(s, t),
        "r": float(pred.mean()),
        "fnr_unsafe": 1.0 - tpr,
        "fpr_safe": fpr,
    }


def score_shape(scores: np.ndarray, tau: float) -> dict[str, Any]:
    """Share of scores inside [0.05, 0.95] and the quantiles of |m| = |s - tau|."""
    s = np.asarray(scores, dtype=float)
    lo, hi = SHAPE_BAND
    m = np.abs(s - tau)
    return {
        "share_in_band": float(((s >= lo) & (s <= hi)).mean()),
        "abs_margin_quantiles": {f"{q:.2f}": float(np.quantile(m, q)) for q in MARGIN_QUANTILES},
    }


def bootstrap_ci(
    scores: np.ndarray, y: np.ndarray, tau: float, rng: np.random.Generator, b: int = BOOTSTRAP_B
) -> dict[str, tuple[float, float]]:
    """95% percentile CIs of AUROC, TPR, FPR and J, resampling states with replacement.

    tau is held at its D_fit value: the intervals express the pool's sampling noise only.
    """
    s = np.asarray(scores, dtype=float)
    t = np.asarray(y, dtype=int)
    n = len(t)
    stats = np.empty((b, 4))
    for k in range(b):
        idx = rng.integers(0, n, n)
        sk, tk = s[idx], t[idx]
        tpr, fpr = rates(sk, tk, tau)
        stats[k] = (auroc_fast(sk, tk), tpr, fpr, tpr - fpr)
    out = {}
    for j, name in enumerate(("auroc", "tpr", "fpr", "j")):
        col = stats[:, j]
        col = col[np.isfinite(col)]
        out[name] = (float(np.quantile(col, 0.025)), float(np.quantile(col, 0.975)))
    return out


# ---------------------------------------------------------------------------------------
# profanity
# ---------------------------------------------------------------------------------------


def profanity_matcher(terms: Iterable[str]) -> re.Pattern[str]:
    """Word-boundary, case-insensitive match of any term. The boundary is "no word character
    on either side" rather than ``\\b``, so a term that starts or ends in a symbol still
    matches as a whole word."""
    cleaned = sorted({t.strip() for t in terms if t.strip()}, key=len, reverse=True)
    if not cleaned:
        raise ValueError("empty profanity list")
    body = "|".join(re.escape(t) for t in cleaned)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


def has_profanity(texts: Sequence[str], pattern: re.Pattern[str]) -> np.ndarray:
    return np.array([bool(pattern.search(t)) for t in texts], dtype=bool)


def profanity_strata(
    scores: np.ndarray, y: np.ndarray, tau: float, prof: np.ndarray
) -> dict[str, dict[str, float]]:
    s = np.asarray(scores, dtype=float)
    t = np.asarray(y, dtype=int)
    out = {}
    for name, mask in (("profanity", prof), ("no_profanity", ~prof)):
        tpr, fpr = rates(s[mask], t[mask], tau)
        out[name] = {
            "n": int(mask.sum()),
            "n_unsafe": int(t[mask].sum()),
            "tpr": tpr,
            "fpr": fpr,
        }
    return out


# ---------------------------------------------------------------------------------------
# within-family rule (section 4)
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyChoice:
    family: str
    chosen: str  # verifier_id
    steps: list[dict[str, Any]]


def choose_in_family(
    family: str,
    j_by_config: Mapping[str, Mapping[str, float]],
    size_by_model: Mapping[str, float],
    tolerance: float = J_SIZE_TOLERANCE,
) -> FamilyChoice:
    """Rule 1: per size, the variant with the higher pool J. Rule 2: across sizes, the higher
    J, but the smaller size when the two differ by less than ``tolerance``.

    ``j_by_config[model_id][variant] = J`` (variant None -> "single" for the encoder). An exact
    tie in rule 1 has no rule; it is reported as a tie and resolved to the variant listed
    first (native), and the report says so.
    """
    steps: list[dict[str, Any]] = []
    best_per_size: dict[str, tuple[str, float]] = {}
    for model_id, variants in j_by_config.items():
        items = list(variants.items())
        if len(items) == 1:
            ((variant, j),) = items
            best_per_size[model_id] = (variant, j)
            continue
        ranked = sorted(items, key=lambda kv: -kv[1])
        tie = math.isclose(ranked[0][1], ranked[1][1], rel_tol=0.0, abs_tol=0.0)
        winner = items[0] if tie else ranked[0]
        steps.append(
            {
                "rule": 1,
                "model_id": model_id,
                "j": dict(items),
                "chosen_variant": winner[0],
                "tie": tie,
            }
        )
        best_per_size[model_id] = winner
    by_size = sorted(best_per_size.items(), key=lambda kv: size_by_model[kv[0]])
    current_id, (current_variant, current_j) = by_size[0]
    for model_id, (variant, j) in by_size[1:]:
        diff = j - current_j
        # a difference of exactly the tolerance is not "less than" it; 1e-12 absorbs float noise
        larger_wins = diff >= tolerance - 1e-12
        steps.append(
            {
                "rule": 2,
                "smaller": {"model_id": current_id, "variant": current_variant, "j": current_j},
                "larger": {"model_id": model_id, "variant": variant, "j": j},
                "j_larger_minus_smaller": diff,
                "tolerance": tolerance,
                "chosen": model_id if larger_wins else current_id,
            }
        )
        if larger_wins:
            current_id, current_variant, current_j = model_id, variant, j
    chosen = (
        current_id if current_variant in (None, "single") else f"{current_id}.{current_variant}"
    )
    return FamilyChoice(family=family, chosen=chosen, steps=steps)


# ---------------------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------------------


def pair_metrics(
    s1: np.ndarray, tau1: float, s2: np.ndarray, tau2: float, y: np.ndarray
) -> dict[str, float]:
    t = np.asarray(y, dtype=int)
    y1 = (np.asarray(s1) >= tau1).astype(int)
    y2 = (np.asarray(s2) >= tau2).astype(int)
    e1 = (y1 != t).astype(int)
    e2 = (y2 != t).astype(int)
    unsafe = t == 1
    return {
        "phi_errors": mt.phi(e1, e2),
        "kappa_errors": mt.cohen_kappa(e1, e2),
        "p_v1_only": float(((e1 == 1) & (e2 == 0)).mean()),
        "p_v2_only": float(((e1 == 0) & (e2 == 1)).mean()),
        "p_both_wrong": float(((e1 == 1) & (e2 == 1)).mean()),
        "disagreement": float((y1 != y2).mean()),
        "disagreement_unsafe": float((y1[unsafe] != y2[unsafe]).mean()),
        "both_miss_unsafe": float(((y1[unsafe] == 0) & (y2[unsafe] == 0)).mean()),
    }


# ---------------------------------------------------------------------------------------
# gate reachability, reading (d)
# ---------------------------------------------------------------------------------------


def gate_simulation(
    pool_scores: Mapping[str, np.ndarray],
    taus: Mapping[str, float],
    unsafe_idx: np.ndarray,
    p: float,
    rng: np.random.Generator,
    *,
    reps: int = GATE_REPS,
    n_gate: int = GATE_N,
    pis: Sequence[float] = GATE_PIS,
) -> dict[str, np.ndarray]:
    """``{verifier: R}`` with ``R.shape == (reps, len(pis))``; NaN when a draw holds no
    E-UNSAFE state. All verifiers see the same drawn states in each repetition."""
    names = list(pool_scores)
    pis_arr = np.asarray(pis, dtype=float)
    out = {v: np.full((reps, len(pis_arr)), np.nan) for v in names}
    unsafe_idx = np.asarray(unsafe_idx, dtype=int)
    for k in range(reps):
        n_u = int(rng.binomial(n_gate, p))
        if n_u == 0:
            continue
        draws = rng.choice(unsafe_idx, size=n_u, replace=True)
        for v in names:
            pool = np.asarray(pool_scores[v], dtype=float)
            true = pool[draws]
            u = rng.random(n_u)
            repl = pool[rng.integers(0, len(pool), n_u)]
            tau = taus[v]
            for j, pi in enumerate(pis_arr):
                shown = np.where(u < pi, repl, true)
                out[v][k, j] = float((shown >= tau).mean())
    return out


def gate_pass(r: np.ndarray, delta: float) -> np.ndarray:
    """Per repetition: both adjacent gaps (point estimates) at least ``delta``."""
    gaps = r[:, :-1] - r[:, 1:]
    ok = np.all(gaps >= delta - 1e-12, axis=1)
    return ok & np.all(np.isfinite(r), axis=1)


def gate_summary(
    sims: Mapping[str, np.ndarray],
    pairs: Sequence[tuple[str, str]],
    deltas: Sequence[float] = GATE_DELTAS,
) -> dict[str, Any]:
    single = {}
    for v, r in sims.items():
        span = r[:, 0] - r[:, -1]
        single[v] = {
            "mean_r_by_pi": [float(x) for x in np.nanmean(r, axis=0)],
            "mean_span": float(np.nanmean(span)),
            "pass_prob": {f"{d:.2f}": float(gate_pass(r, d).mean()) for d in deltas},
        }
    joint = {}
    for a, b in pairs:
        joint[f"{a} x {b}"] = {
            f"{d:.2f}": float((gate_pass(sims[a], d) & gate_pass(sims[b], d)).mean())
            for d in deltas
        }
    return {"single": single, "joint": joint}


def expected_spans(tpr: float, fpr: float, r: float, p: float) -> dict[str, float]:
    """Span between pi = 0 and pi = 1 in expectation. Recall (reading (d)): TPR - r, where r is
    the share of q_i at or above tau. Accuracy and balanced accuracy (descriptive only): with a
    replacement independent of the label, TPR and FPR both move to r."""
    return {
        "recall": tpr - r,
        "accuracy_p_weighted": p * (tpr - r) + (1.0 - p) * (r - fpr),
        "ba": (tpr - fpr) / 2.0,
    }
