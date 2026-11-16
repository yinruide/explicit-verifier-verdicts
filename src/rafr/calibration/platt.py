"""Logit-scale Platt calibration and its diagnostics: arrays in, numbers out.

Settings (2026-10-31, ``configs/ext.yaml: calibration``):

* the score is clipped to [eps, 1 - eps] before the logit, eps = 1e-7; how many scores were
  clipped is counted per verifier x split and reported;
* Platt's smoothed targets -- (N+ + 1) / (N+ + 2) for a positive, 1 / (N- + 2) for a negative --
  and no regularization: the fit minimizes the cross-entropy against those targets exactly, by
  Newton's method with a backtracking line search;
* ECE over 10 equal-frequency bins: the predictions sorted (stable, so ties keep the caller's
  order) and cut into 10 runs whose sizes differ by at most one, ``numpy.array_split``'s rule;
  ECE = sum_b (n_b / n) |mean prediction_b - observed rate_b|;
* every interval is the design's: a 95% percentile bootstrap with B replicates, the state as the
  resampling unit, seeded by the caller. Equal-frequency bins are recut inside every replicate
  for ECE; the reliability table's per-bin intervals keep the full sample's bins;
* the prior correction (2026-10-31; :func:`prior_shift`): the
  intercept moves by logit(p_R) - logit(p_pool), the slope stays.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rafr.core.seeding import rng as seeded_rng

TARGETS = "platt_smoothed"


class PlattFitError(RuntimeError):
    """The fit did not converge or is not finite: no calibrator is written."""


def clip_logit(s: Sequence[float] | np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """``(logit(clip(s, eps, 1 - eps)), clipped)``; ``clipped`` marks the scores that moved."""
    if not 0.0 < eps < 0.5:
        raise ValueError(f"eps must be in (0, 0.5), got {eps!r}")
    x = np.asarray(s, dtype=np.float64)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or np.any(x < 0.0) or np.any(x > 1.0):
        raise ValueError("scores must be a 1-d array of finite values in [0, 1]")
    c = np.clip(x, eps, 1.0 - eps)
    return np.log(c) - np.log1p(-c), c != x


def sigmoid(u: np.ndarray) -> np.ndarray:
    out = np.empty_like(u, dtype=np.float64)
    pos = u >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
    e = np.exp(u[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def prior_shift(n: int, n_positive: int, n_added_positive: int) -> dict[str, Any]:
    """The prior correction of a fit whose sample lacks ``n_added_positive`` states that are all
    positives (2026-10-31): the slope a stays, the intercept becomes b + delta with

        p_pool = n_positive / n,   p_R = (n_positive + n_added_positive) / (n + n_added_positive),
        delta  = logit(p_R) - logit(p_pool).

    It holds when the missing states were a uniform random draw from the positives, which changes
    the class prior and not either class's score distribution; the caller checks that premise.
    """
    if not 0 < n_positive < n:
        raise ValueError(f"need 0 < n_positive < n, got {n_positive} of {n}")
    if n_added_positive < 0:
        raise ValueError(f"n_added_positive must be >= 0, got {n_added_positive}")
    p_pool = n_positive / n
    p_r = (n_positive + n_added_positive) / (n + n_added_positive)
    delta = (math.log(p_r) - math.log1p(-p_r)) - (math.log(p_pool) - math.log1p(-p_pool))
    return {
        "n": n,
        "n_positive": n_positive,
        "n_added_positive": n_added_positive,
        "p_pool": p_pool,
        "p_R": p_r,
        "delta": delta,
    }


def platt_targets(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """The smoothed targets of every point, and the positive / negative target values."""
    y = np.asarray(y)
    if y.ndim != 1 or not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be a 1-d array of 0 / 1")
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    t_pos = (n_pos + 1.0) / (n_pos + 2.0)
    t_neg = 1.0 / (n_neg + 2.0)
    return np.where(y == 1, t_pos, t_neg), t_pos, t_neg


def _loss(a: float, b: float, z: np.ndarray, t: np.ndarray) -> float:
    u = a * z + b
    # -[t log sigma(u) + (1 - t) log(1 - sigma(u))] = t softplus(-u) + (1 - t) softplus(u)
    return float(np.sum(t * np.logaddexp(0.0, -u) + (1.0 - t) * np.logaddexp(0.0, u)))


@dataclass(frozen=True)
class PlattFit:
    a: float
    b: float
    n: int
    n_positive: int
    n_negative: int
    target_positive: float
    target_negative: float
    loss: float
    max_abs_gradient: float
    iterations: int
    converged: bool
    n_clipped: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_platt(
    s: Sequence[float] | np.ndarray,
    y: Sequence[int] | np.ndarray,
    eps: float,
    *,
    max_iter: int = 200,
    grad_tol: float = 1e-9,
) -> PlattFit:
    """Fit ``g(s) = sigma(a logit(s) + b)`` to the smoothed targets.

    Newton steps on (a, b) with step halving until the loss does not increase; converged when
    the largest absolute gradient component is below ``grad_tol`` times the number of points.
    Starts at Platt's point a = 0, b = log((N+ + 1) / (N- + 1)). Raises :class:`PlattFitError`
    if the Hessian is singular, the iterations run out, or the result is not finite.
    """
    z, clipped = clip_logit(s, eps)
    t, t_pos, t_neg = platt_targets(np.asarray(y))
    n = z.size
    n_pos = int(np.sum(np.asarray(y) == 1))
    if n_pos == 0 or n_pos == n:
        raise PlattFitError("both classes are needed to fit the calibrator")
    a, b = 0.0, float(np.log((n_pos + 1.0) / (n - n_pos + 1.0)))
    loss = _loss(a, b, z, t)
    tol = grad_tol * n
    grad = np.zeros(2)
    for it in range(1, max_iter + 1):
        p = sigmoid(a * z + b)
        r = p - t
        grad = np.array([np.sum(r * z), np.sum(r)])
        if np.max(np.abs(grad)) < tol:
            return _result(a, b, n, n_pos, t_pos, t_neg, loss, grad, it - 1, int(clipped.sum()))
        w = p * (1.0 - p)
        h = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]])
        if not np.all(np.isfinite(h)) or np.linalg.det(h) <= 0.0:
            raise PlattFitError(f"singular Hessian at iteration {it} (a={a!r}, b={b!r})")
        step = np.linalg.solve(h, grad)
        lam = 1.0
        while True:
            na, nb = a - lam * step[0], b - lam * step[1]
            new = _loss(na, nb, z, t)
            if new <= loss or lam < 1e-12:
                break
            lam /= 2.0
        if new > loss:
            raise PlattFitError(f"line search failed at iteration {it}")
        a, b, loss = float(na), float(nb), new
    raise PlattFitError(
        f"no convergence in {max_iter} iterations (max |grad| {float(np.max(np.abs(grad)))!r})"
    )


def _result(
    a: float,
    b: float,
    n: int,
    n_pos: int,
    t_pos: float,
    t_neg: float,
    loss: float,
    grad: np.ndarray,
    iterations: int,
    n_clipped: int,
) -> PlattFit:
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(loss)):
        raise PlattFitError(f"fit is not finite (a={a!r}, b={b!r}, loss={loss!r})")
    return PlattFit(
        a=float(a),
        b=float(b),
        n=int(n),
        n_positive=int(n_pos),
        n_negative=int(n - n_pos),
        target_positive=float(t_pos),
        target_negative=float(t_neg),
        loss=float(loss),
        max_abs_gradient=float(np.max(np.abs(grad))),
        iterations=int(iterations),
        converged=True,
        n_clipped=int(n_clipped),
    )


def predict(a: float, b: float, s: Sequence[float] | np.ndarray, eps: float) -> np.ndarray:
    """``g(s)`` for every score."""
    z, _ = clip_logit(s, eps)
    return sigmoid(a * z + b)


def pihat_from_g(g: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """pi-hat = 1 - g where the verifier flags (y = 1), g where it does not."""
    g = np.asarray(g, dtype=np.float64)
    flags = np.asarray(flags)
    return np.where(flags == 1, 1.0 - g, g)


def bin_sizes(n: int, n_bins: int) -> list[int]:
    """``numpy.array_split``'s sizes: the first ``n % n_bins`` bins get one more."""
    if n < n_bins:
        raise ValueError(f"{n} points cannot fill {n_bins} equal-frequency bins")
    q, r = divmod(n, n_bins)
    return [q + 1] * r + [q] * (n_bins - r)


def _bounds(n: int, n_bins: int) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(bin_sizes(n, n_bins))])


def ece_rows(p: np.ndarray, y: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-frequency ECE of every row of ``p`` (shape (r, n)) against ``y`` (same shape)."""
    order = np.argsort(p, axis=1, kind="stable")
    ps = np.take_along_axis(p, order, axis=1)
    ys = np.take_along_axis(y.astype(np.float64), order, axis=1)
    bounds = _bounds(p.shape[1], n_bins)
    ece = np.zeros(p.shape[0])
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        ece += (hi - lo) / p.shape[1] * np.abs(ps[:, lo:hi].mean(1) - ys[:, lo:hi].mean(1))
    return ece


def ece_equal_frequency(p: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    return float(ece_rows(np.asarray(p, float)[None, :], np.asarray(y)[None, :], n_bins)[0])


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def bin_membership(p: np.ndarray, n_bins: int) -> np.ndarray:
    """The equal-frequency bin (0 .. n_bins - 1) of every point, in the caller's order."""
    order = np.argsort(np.asarray(p, float), kind="stable")
    member = np.empty(order.size, dtype=np.int64)
    bounds = _bounds(order.size, n_bins)
    for j, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        member[order[lo:hi]] = j
    return member


def diagnostics(
    p: Sequence[float] | np.ndarray,
    y: Sequence[int] | np.ndarray,
    *,
    n_bins: int,
    n_boot: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    """ECE, Brier and the reliability table of predictions ``p`` for outcomes ``y``, with their
    percentile bootstrap intervals (unit: the point)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("p and y must be 1-d arrays of the same length")
    n = p.size
    member = bin_membership(p, n_bins)
    r = seeded_rng(seed, "bootstrap")
    idx = r.integers(0, n, (n_boot, n))
    pb, yb = p[idx], y[idx]
    q = [alpha / 2, 1 - alpha / 2]
    ece_b = ece_rows(pb, yb, n_bins)
    brier_b = np.mean((pb - yb) ** 2, axis=1)
    mb = member[idx]
    bins: list[dict[str, Any]] = []
    for j in range(n_bins):
        inside = member == j
        mask = mb == j
        cnt = mask.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate_b = np.where(cnt > 0, (yb * mask).sum(1) / np.maximum(cnt, 1), np.nan)
            mean_b = np.where(cnt > 0, (pb * mask).sum(1) / np.maximum(cnt, 1), np.nan)
        lo, hi = np.nanquantile(rate_b, q)
        mlo, mhi = np.nanquantile(mean_b, q)
        bins.append(
            {
                "bin": j,
                "n": int(inside.sum()),
                "n_positive": int(y[inside].sum()),
                "p_min": float(p[inside].min()),
                "p_max": float(p[inside].max()),
                "mean_prediction": float(p[inside].mean()),
                "mean_prediction_ci": [float(mlo), float(mhi)],
                "observed_rate": float(y[inside].mean()),
                "observed_rate_ci": [float(lo), float(hi)],
            }
        )
    e_lo, e_hi = np.quantile(ece_b, q)
    b_lo, b_hi = np.quantile(brier_b, q)
    return {
        "n": int(n),
        "n_positive": int(y.sum()),
        "base_rate": float(y.mean()),
        "mean_prediction": float(p.mean()),
        "ece": ece_equal_frequency(p, y, n_bins),
        "ece_ci": [float(e_lo), float(e_hi)],
        "brier": brier(p, y),
        "brier_ci": [float(b_lo), float(b_hi)],
        "reliability": bins,
        "n_bins": n_bins,
        "binning": "equal_frequency (stable sort, numpy.array_split sizes)",
    }


__all__ = [
    "TARGETS",
    "PlattFit",
    "PlattFitError",
    "bin_membership",
    "bin_sizes",
    "brier",
    "clip_logit",
    "diagnostics",
    "ece_equal_frequency",
    "ece_rows",
    "fit_platt",
    "pihat_from_g",
    "platt_targets",
    "predict",
    "prior_shift",
    "sigmoid",
]
