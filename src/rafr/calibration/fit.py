"""Fit the pi-hat calibrators on the unlabeled pool and diagnose them on D_gate.

y* is read through the label gate only: the pool with purpose CALIBRATION_FIT, D_gate with
CALIBRATION_DIAG (2026-10-31). Before anything is fitted, the pool is checked to share
no x0 with D_fit, D_gate, D_gate_reserve or D_Exp1 (:func:`assert_pool_disjoint`) -- only split
files are opened for that, no label.

Two diagnostic sets per verifier, both on D_gate at the raw score (lambda = none):

* ``g_vs_failure``: g_i(s_i) against task failure (E's y*);
* ``pihat_vs_verdict_wrong``: pi-hat_i against 1[y_i != task failure], the event pi-hat
  estimates.

There is no quality threshold: a converged, finite fit without leakage is used and reported as it
is. What does stop the run is a clipped share above ``max_clip_share`` (1%) in any verifier x
split cell (:class:`ClipShareError`, which carries the whole count table).

Prior correction (2026-10-31). The pool is what was left after
D_Exp1 took its failures from the states the three random splits left, so the pool's failure
rate is below the population's while, D_Exp1 being a uniform draw among those failures, neither
class's score distribution changed. The intercept is therefore moved by
logit(p_R) - logit(p_pool) (:func:`rafr.calibration.platt.prior_shift`), from the pool's counts
and the design fact that every D_Exp1 state is a failure. Its two premises are checked first
(:func:`check_prior_premises`), from the state list, the split files, the split plan and the
pool's own labels -- no label of D_gate or D_Exp1 is read.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rafr.calibration import platt
from rafr.calibration.apply import FROZEN_KIND, VERIFIER_KEYS, Calibrator, ids_sha256
from rafr.contracts.schemas import MathLabel, SafetyLabel, Split
from rafr.core import paths
from rafr.data import labels as label_gate
from rafr.data.splits import (
    PROCEDURE,
    RANDOM_SPLITS,
    SPLIT_ORDER,
    SplitError,
    load_split,
    make_splits,
)

DISJOINT_FROM: tuple[Split, ...] = (
    Split.D_FIT,
    Split.D_GATE,
    Split.D_GATE_RESERVE,
    Split.D_EXP1,
)


class CalibrationLeakError(RuntimeError):
    """The fitting data shares an x0 with a split it must stay apart from."""


class PriorPremiseError(RuntimeError):
    """A premise of the prior correction does not hold: nothing is fitted or written."""


class ClipShareError(RuntimeError):
    """More than the allowed share of scores was clipped in some verifier x split cell."""

    def __init__(self, report: dict[str, Any], cells: list[str]) -> None:
        self.report = report
        self.cells = cells
        super().__init__(
            "clipped share above the limit in " + ", ".join(cells) + "; stop and report"
        )


def assert_pool_disjoint(split_dir: Path | None = None) -> dict[str, Any]:
    """The unlabeled pool shares no x0 with D_fit, D_gate, D_gate_reserve, D_Exp1. Every split
    file must be there: a missing one would make the check pass vacuously."""
    split_dir = Path(split_dir or paths.DATA_SPLITS)
    pool = set(load_split(Split.UNLABELED_POOL, split_dir))
    if not pool:
        raise CalibrationLeakError(f"the unlabeled pool is empty or missing in {split_dir}")
    sizes = {Split.UNLABELED_POOL.value: len(pool)}
    for sp in DISJOINT_FROM:
        other = set(load_split(sp, split_dir))
        if not other:
            raise CalibrationLeakError(f"{sp.value} is empty or missing in {split_dir}")
        sizes[sp.value] = len(other)
        both = sorted(pool & other)
        if both:
            raise CalibrationLeakError(
                f"the unlabeled pool shares {len(both)} x0 with {sp.value} (e.g. {both[:3]})"
            )
    return {"disjoint_by_x0": True, "sizes": sizes}


def check_prior_premises(
    *,
    all_x0: Sequence[str],
    splits: Mapping[Split, Sequence[str]],
    plan: Mapping[str, Any],
    expected_seed: int,
    pool_failure_x0: Iterable[str],
) -> dict[str, Any]:
    """The two premises of the prior correction, by x0; raises :class:`PriorPremiseError`.

    1. pool + D_Exp1 = all x0 - D_fit - D_gate - D_gate_reserve, the pool and D_Exp1 disjoint
       (``all_x0``: every x0 that entered the split, from the state list, not from the splits);
    2. D_Exp1 is the uniform random draw among the remaining failures: the split procedure
       (:func:`rafr.data.splits.make_splits`, :data:`PROCEDURE`), rerun on ``all_x0`` with the
       recorded seed -- which must be ``expected_seed`` -- and "failure" = the pool's failures
       (``pool_failure_x0``) plus every D_Exp1 x0 (the design fact), reproduces every split file
       exactly, order included. D_Exp1 is then the first |D_Exp1| failures of the remainder in
       one seeded shuffle.

    ``splits[sp]`` is the split file's x0 list in file order.
    """
    all_list = list(all_x0)
    everyone = set(all_list)
    if len(everyone) != len(all_list):
        raise PriorPremiseError("duplicate x0 in the state list")
    pool, exp1 = list(splits[Split.UNLABELED_POOL]), list(splits[Split.D_EXP1])
    random_part: set[str] = set()
    for sp in RANDOM_SPLITS:
        random_part |= set(splits[sp])
    if not random_part <= everyone:
        raise PriorPremiseError("D_fit / D_gate / D_gate_reserve hold an x0 not in the state list")
    if set(pool) & set(exp1):
        raise PriorPremiseError("the pool and D_Exp1 share an x0")
    rest = everyone - random_part
    union = set(pool) | set(exp1)
    if union != rest:
        raise PriorPremiseError(
            "pool + D_Exp1 is not all x0 - D_fit - D_gate - D_gate_reserve: "
            f"{len(union - rest)} extra, {len(rest - union)} missing"
        )
    if plan.get("procedure") != PROCEDURE:
        raise PriorPremiseError(f"split procedure {plan.get('procedure')!r}, not {PROCEDURE!r}")
    seed = int(plan["seed"])
    if seed != int(expected_seed):
        raise PriorPremiseError(f"split seed {seed} is not the expected {expected_seed}")
    failures = set(pool_failure_x0)
    if not failures <= set(pool):
        raise PriorPremiseError("a pool failure x0 is not in the pool")
    positives = failures | set(exp1)

    def is_failure(x0: str) -> bool:
        if x0 not in rest:
            raise PriorPremiseError(f"the procedure asked for the label of {x0}, outside the rest")
        return x0 in positives

    try:
        redrawn = make_splits(all_list, is_failure, plan["sizes"], seed)
    except SplitError as exc:
        raise PriorPremiseError(f"the split procedure does not rerun: {exc}") from exc
    for sp in SPLIT_ORDER:
        if list(redrawn[sp]) != list(splits[sp]):
            raise PriorPremiseError(
                f"{sp.value}: the split file is not what the procedure draws with the recorded "
                "seed, so D_Exp1 cannot be taken for a uniform draw among the remaining failures"
            )
    return {
        "complement": "pool + d_exp1 = all - d_fit - d_gate - d_gate_reserve (by x0): holds",
        "uniform_draw": (
            f"{PROCEDURE} rerun with seed {seed} reproduces all five split files exactly: d_exp1 "
            "is the first |d_exp1| failures of the remainder in one seeded shuffle"
        ),
        "labels_read": "the pool's (calibration_fit); d_exp1 = failures by design; none other",
        "n_all": len(everyone),
        "sizes": {sp.value: len(splits[sp]) for sp in SPLIT_ORDER},
        "n_pool_failures": len(failures),
        "seed": seed,
    }


def task_failures(
    split: Split,
    purpose: label_gate.LabelPurpose,
    failure: SafetyLabel | MathLabel,
    *,
    labels_dir: Path | None = None,
    split_dir: Path | None = None,
) -> dict[str, int]:
    """state_id -> 1 if E judged the state a task failure, else 0, through the label gate."""
    return {
        y.state_id: int(y.label == failure)
        for y in label_gate.read_labels(split, purpose, labels_dir=labels_dir, split_dir=split_dir)
    }


def clip_report(
    scores: Mapping[str, Mapping[str, Sequence[float]]], eps: float, max_share: float
) -> dict[str, Any]:
    """Clipped counts per split x verifier (``scores[split][verifier key]``); raises
    :class:`ClipShareError` after counting every cell if any share exceeds ``max_share``."""
    report: dict[str, Any] = {"epsilon": eps, "max_share": max_share, "cells": {}}
    over: list[str] = []
    for split, by_v in scores.items():
        for k, values in by_v.items():
            _, clipped = platt.clip_logit(np.asarray(values, float), eps)
            low = int(np.sum(np.asarray(values, float) < eps))
            n = int(clipped.size)
            share = float(clipped.sum() / n) if n else 0.0
            report["cells"][f"{split}|{k}"] = {
                "n": n,
                "clipped": int(clipped.sum()),
                "clipped_low": low,
                "clipped_high": int(clipped.sum()) - low,
                "share": share,
            }
            if share > max_share:
                over.append(f"{split}|{k} ({int(clipped.sum())}/{n})")
    if over:
        raise ClipShareError(report, over)
    return report


def fit_record(
    *,
    domain: str,
    ids: Sequence[str],
    failures: Mapping[str, int],
    scores: Mapping[str, Mapping[str, float]],
    verifier_ids: Mapping[str, str],
    eps: float,
    provenance: Mapping[str, Any],
    prior_correction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The frozen calibrator's content: both fits on ``ids`` (in that order). With
    ``prior_correction`` (:func:`rafr.calibration.platt.prior_shift`, plus what the caller
    checked) each verifier's ``b`` is Platt's ``b_fit`` + delta, its ``a`` unchanged."""
    ids = list(ids)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate state ids in the fitting data")
    y = np.array([failures[i] for i in ids], dtype=np.int64)
    if prior_correction is not None and (
        prior_correction["n"] != len(ids) or prior_correction["n_positive"] != int(y.sum())
    ):
        raise ValueError("the prior correction's counts are not the fitting data's")
    fits = {}
    for k in VERIFIER_KEYS:
        s = np.array([scores[k][i] for i in ids], dtype=np.float64)
        fit = platt.fit_platt(s, y, eps).as_dict()
        if prior_correction is not None:
            fit["b_fit"] = fit["b"]
            fit["b"] = fit["b_fit"] + float(prior_correction["delta"])
        fits[k] = {"verifier_id": verifier_ids[k], **fit}
    extra: dict[str, Any] = {}
    if prior_correction is not None:
        extra["prior_correction"] = {
            "rule": "a_i unchanged; b_i = b_fit_i + delta, delta = logit(p_R) - logit(p_pool)",
            **dict(prior_correction),
        }
    return {
        "kind": FROZEN_KIND,
        "domain": domain,
        "model": "g_i(s) = sigmoid(a_i * logit(clip(s, eps, 1 - eps)) + b_i) = P(task failure | s)",
        "pihat": "pi-hat_i = 1 - g_i(s_i) if y_i = 1 else g_i(s_i); y_i = 1[s_i >= tau_i]",
        "epsilon": eps,
        "targets": platt.TARGETS,
        "regularization": "none",
        **extra,
        "verifiers": fits,
        "fit_data": {
            "split": Split.UNLABELED_POOL.value,
            "n": len(ids),
            "n_task_failure": int(y.sum()),
            "state_ids": ids,
            "state_ids_sha256": ids_sha256(ids),
            **dict(provenance),
        },
    }


def diagnose(
    calibrator: Calibrator,
    *,
    ids: Sequence[str],
    failures: Mapping[str, int],
    scores: Mapping[str, Mapping[str, float]],
    taus: Mapping[str, float],
    n_bins: int,
    n_boot: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    """Both diagnostic sets for both verifiers on ``ids`` (D_gate, raw scores)."""
    ids = list(ids)
    y = np.array([failures[i] for i in ids], dtype=np.int64)
    out: dict[str, Any] = {
        "n": len(ids),
        "n_task_failure": int(y.sum()),
        "bootstrap": {
            "method": "percentile_bootstrap",
            "level": 1 - alpha,
            "n_boot": n_boot,
            "seed": seed,
            "unit": "state",
        },
        "verifiers": {},
    }
    for k in VERIFIER_KEYS:
        a, b = calibrator.params(k)
        s = np.array([scores[k][i] for i in ids], dtype=np.float64)
        g = platt.predict(a, b, s, calibrator.eps)
        flags = (s >= taus[k]).astype(np.int64)
        wrong = (flags != y).astype(np.int64)
        pihat = platt.pihat_from_g(g, flags)
        kw = dict(n_bins=n_bins, n_boot=n_boot, seed=seed, alpha=alpha)
        out["verifiers"][k] = {
            "verifier_id": calibrator.verifier_id(k),
            "tau": taus[k],
            "n_flagged": int(flags.sum()),
            "n_verdict_wrong": int(wrong.sum()),
            "g_vs_failure": platt.diagnostics(g, y, **kw),
            "pihat_vs_verdict_wrong": platt.diagnostics(pihat, wrong, **kw),
        }
    return out


__all__ = [
    "DISJOINT_FROM",
    "CalibrationLeakError",
    "ClipShareError",
    "PriorPremiseError",
    "assert_pool_disjoint",
    "check_prior_premises",
    "clip_report",
    "diagnose",
    "fit_record",
    "task_failures",
]
