"""Deterministic seed derivation.

Every random decision in the pipeline derives its seed from a *named* key so that
reproducibility never depends on call order. In particular the channel's randomness is
keyed by ``(state_id, lambda)`` and never by arm (paper Sec. 4.3), and Exp 2 randomization is keyed
by ``x0_id`` (paper Sec. 4.4).
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

_MASK_63 = (1 << 63) - 1


def derive_seed(base_seed: int, *keys: object) -> int:
    """Return a stable 63-bit seed from ``base_seed`` and any hashable, str-able keys.

    ``derive_seed(7, "channel", state_id, lam)`` is identical across processes and
    platforms; it does not depend on Python's per-process hash randomization.
    """
    h = hashlib.sha256()
    h.update(str(int(base_seed)).encode("utf-8"))
    for k in keys:
        h.update(b"\x1f")
        h.update(str(k).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big") & _MASK_63


COMPONENTS: tuple[str, ...] = (
    "sampling",  # step00's processing order of the x0 pool (so a --limit batch is random)
    "splits",  # round 1 splits; kept only for the retired annotation export order
    "splits_v09",  # the split procedure (paper Sec. 4.2); fixed before step 5 judges any state
    "g_fixed",
    "controller",
    "channel",
    "permuted_b",
    "exp1_run",
    "exp2_run",
    "exp2_randomization",
    "gate_ci",
    "step02_probe",  # step 2 probe: bootstrap CIs and the gate-reachability simulation
    "exp2_smoke",  # pre- smoke runs: which unlabeled-pool x0 (kept for old run dirs)
    # the step 10a Exp 2 smoke (2026-09-25; configs/exp2_smoke.yaml): its 30 E-UNSAFE
    # pool x0, its run seed and its (one-arm) randomization -- an experiment id of its own
    "exp2_smoke_x0",
    "exp2_smoke_run",
    "exp2_smoke_randomization",
    "dry_run",  # step 3 dry run: which E-UNSAFE pool x0 (engineering only)
    "pod_numeric_check",  # the 203 D_fit states re-scored on the pod before step 6 / Exp 2
    "perturb_sample",  # step 6 stability: the 150 + 150 states and their prefixes
    "perturb_paraphrase",  # step 6 stability: the per-state seed of each paraphrase call
    # post-pilot extension (2026-10-31); new names, so no existing stream moves
    "ext_pihat_mismatch",  # the pi-hat mismatch derangement, independent of permuted_b's
    "ext_calibration_boot",  # bootstrap of the calibration diagnostics on D_gate
    "ext_ecorrect_cohort",  # which E-correct D_gate states form the E-correct cohort
    "ext_analysis_boot",  # bootstrap of the extension contrasts
    "ext_smoke",  # the smoke: which states outside the formal sets (as step 10a)
    # post hoc (2026-11-01), after the detox extension's results had been seen
    "ext_decision_quality_boot",  # bootstrap of analysis/ext_decision_quality.py
)
"""Every random component of the pipeline (decision, 2026-09-17).

One master seed lives in ``configs/data.yaml: seeds.master``; each component derives its own
from it by name through :func:`component_seed`. Separately configurable per-component seeds
carried no information that this derivation does not, and were that many more values to freeze,
record and mistype. The names are fixed: changing one re-randomizes that component only.
"""


def component_seed(master: int, component: str) -> int:
    """The seed of one named pipeline component, derived from the master seed."""
    if component not in COMPONENTS:
        raise ValueError(f"unknown seed component {component!r}; expected one of {COMPONENTS}")
    return derive_seed(master, "component", component)


def rng(base_seed: int, *keys: object) -> np.random.Generator:
    """NumPy generator seeded by :func:`derive_seed`."""
    return np.random.default_rng(derive_seed(base_seed, *keys))


def py_rng(base_seed: int, *keys: object) -> random.Random:
    """Stdlib ``random.Random`` seeded by :func:`derive_seed` (for shuffles / permutations)."""
    return random.Random(derive_seed(base_seed, *keys))  # noqa: S311 - not for security


def attempt_seed(base_seed: int, *keys: object, attempt: int) -> int:
    """Seed for the ``attempt``-th regeneration of the same item (paper Sec. 4.2: "next seed")."""
    return derive_seed(base_seed, *keys, "attempt", attempt)
