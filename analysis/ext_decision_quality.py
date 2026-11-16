"""Decision quality of Exp 1's arms: verdict following, threshold rules, operating points.

POST HOC (2026-11-01): added on 2026-11-01, after the detox extension's results had been
seen. EXPLORATORY: the intervals are not corrected for multiple comparisons, and every cell is
reported. Domain-agnostic: the code knows arms, two cohorts, lambda levels and two verifier
flags, never a domain; the entry names the domain and its run directories, so math reuses
it as is. Read only: no API call, no pod.

Inputs: Exp 1's two cohorts.

* ``e_incorrect`` (E judged every state incorrect): the reused arms (detox: A1, B, permuted-B) are
  the original Exp 1 run's decisions, read after a sha256 check against the domain's extension
  config; every other arm comes from the extension run(s);
* ``e_correct`` (E judged every state correct): every arm from the extension run(s).

Every decision is put in the cell of its (state, lambda)'s own observed flags (y1, y2): y_i = 1
when verifier i's post-channel score reaches its threshold, i.e. verifier i reports an error. The
flags are read from the state's A1 row (2026-11-01). The arms shown that (state, lambda)'s
own flags -- A1, Verdict-only, B, B + pi-hat and the pi-hat mismatch arm (the same flags and
scores, another state's pi-hat) -- must log the same flags (also across the original and the
extension runs); No Verifier (shown no verifier output) and permuted-B (shown another state's
evidence: the donor in ``permuted_from_state_id``; math's original run logs the donor's flags on
its rows) are not checked and need not log flags.

1. Verdict following, per arm x lambda x cohort: the accept rate and n in each of the four
   (y1, y2) cells; the following gap = accept rate(0,0) - accept rate(1,1); the gap minus No
   Verifier's (same cohort and lambda: No Verifier sees no verdict, so its gap only reflects how
   the verifiers' flags correlate with the controller's own reading); and the share of the arm's
   decisions equal to each threshold rule's.
2. Threshold rules, deterministic in the observed flags: V1 only (reject iff y1 = 1), V2 only
   (reject iff y2 = 1), OR (reject iff either flags), AND (reject iff both flag).
3. Operating points: FA = the accept rate on ``e_incorrect``, FRR = 1 - the accept rate on
   ``e_correct``, BA = 1 - (FA + FRR) / 2, for every arm and rule x lambda. BA(Y) - BA(X) for the
   seven confirmed (X less information, Y more) pairs, with dFA = FA(Y) - FA(X), dFRR = FRR(Y) -
   FRR(X) and whether Y dominates X, twice: from the point estimates (:func:`dominance`) and from
   the 95% intervals of dFA, dFRR and dBA (:func:`dominance_ci`, 2026-11-01); BA(arm) -
   BA(rule) for every arm and rule, likewise; one FA-vs-FRR panel per lambda (PNG and PDF).

The accounting counts a parse_fail as not accepted (paper Sec. 4.6, the extension's main
accounting); parse_fail counts are reported per arm. A lambda level whose channel replaces every
score by a state-independent draw (detox: lambda_high, p = 1 for both verifiers, independent
resampling) is marked: its flags carry no information about the text.

Intervals: 95% percentile bootstrap, B = 10000. The two cohorts are resampled by x0 (one state per
x0), independently of each other; within a cohort every arm, rule and lambda uses the same draw
(paired). The seed derives from the domain's master seed (``ext_analysis.DOMAINS``; the authors
2026-11-01) with the component ``ext_decision_quality_boot``; cohort c draws with the key
("bootstrap", c). A replicate in which a statistic is undefined (an empty (y1, y2) cell) is left
out of that interval and counted.

The domain's extension config comes from the same registry (detox ``configs/ext.yaml``, math
``configs/math/ext.yaml``); ``--ext-config`` overrides the path.

``--fake`` runs the whole entry on fake data (FAKE marker, FAKE- run ids, FAKE_ file names) and
never reads a real file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from analysis import ext_analysis as W
from analysis import ext_contrasts as X
from analysis._common import (
    FAKE_FILE_PREFIX,
    FAKE_MARKER,
    FAKE_RUN_PREFIX,
    accepted,
    refuse_fake,
    require_fake,
)
from rafr.calibration.apply import file_sha256
from rafr.contracts.schemas import BOOTSTRAP_ALPHA, BOOTSTRAP_B
from rafr.core import paths
from rafr.core.config import get, load_yaml
from rafr.core.seeding import component_seed
from rafr.core.seeding import rng as seeded_rng
from rafr.experiments.ext_config import checked
from rafr.logging.writer import read_jsonl

E_INCORRECT, E_CORRECT = X.E_INCORRECT, X.E_CORRECT
COHORTS = X.COHORTS
LAMBDAS = X.LAMBDAS
STATUS = (
    "POST HOC: added 2026-11-01, after the detox extension's results had been seen. "
    "EXPLORATORY: intervals not corrected for multiple comparisons; every cell reported"
)
SEED_COMPONENT = "ext_decision_quality_boot"
ACCOUNTING = "main"
ACCOUNTING_TEXT = "parse_fail counts as not accepted (II.8.2; the extension's main accounting)"
PATTERNS = ("00", "01", "10", "11")  # y1 y2
REFERENCE_ARM = "no_verifier"
ARM_ORDER = (
    "no_verifier",
    "A1",
    "verdict_only",
    "B",
    "B_pihat",
    "B_pihat_mismatched",
    "permuted_B",
)
# the confirmed contrasts (configs/ext.yaml: analysis.contrasts + Exp 1 e_correct's extras),
# as (X less information, Y more information)
BA_PAIRS: tuple[tuple[str, str], ...] = (
    ("no_verifier", "A1"),
    ("A1", "verdict_only"),
    ("verdict_only", "B"),
    ("B", "B_pihat"),
    ("B_pihat_mismatched", "B_pihat"),
    ("A1", "B"),
    ("permuted_B", "B"),
)
RUN_FILES = ("decisions.jsonl", "ext_meta.json", "configs/channel.yaml", "configs/ext.yaml")
OUT_PREFIX = "ext_decision_quality"
FLAG_REFERENCE_ARM = "A1"  # the cell of a (state, lambda) is its A1 row's flags
# arms shown the (state, lambda)'s own flags: they must log the reference flags
OWN_FLAG_ARMS = ("A1", "verdict_only", "B", "B_pihat", "B_pihat_mismatched")
# shown no flags (No Verifier) or another state's (permuted-B): their logged flags are not read
OTHER_FLAG_ARMS = ("no_verifier", "permuted_B")
NOTES = (
    "Cells are the (state, lambda)'s own observed flags (y1, y2), y_i = 1 = verifier i reports "
    "an error, read from its A1 row; the arms shown the state's own flags (A1, Verdict-only, B, "
    "B + pi-hat, the pi-hat mismatch arm) log the same flags (checked). No Verifier and "
    "permuted-B are not checked: they are not shown the state's own flags.",
    "Within a cell the arms with verifier evidence were shown that (state, lambda)'s evidence, "
    "except permuted-B (another state's evidence, the donor in permuted_from_state_id) and the "
    "pi-hat mismatch arm (the same flags and scores, another state's pi-hat). No Verifier was "
    "shown no verifier output.",
    "No Verifier's following gap only reflects how the flags correlate with the controller's own "
    "reading; 'gap - No Verifier' subtracts it (same cohort and lambda, paired).",
    "Dominance compares the point estimates of dFA and dFRR: 'dominates' = both lower, "
    "'dominated' = both higher, 'moved' = one lower and one higher (only the operating point "
    "moved); 'weakly dominates' / 'weakly dominated' / 'equal' when a difference is exactly 0.",
    "Dominance (CI) (2026-11-01) reads the 95% intervals; 'excludes 0' = the whole "
    "interval lies on one side of 0. 'dominates' / 'dominated' = the dFA and dFRR intervals both "
    "exclude 0 on the same side (below / above); with opposite sides, 'improved with trade-off' "
    "/ 'worsened with trade-off' when the dBA interval excludes 0 above / below, 'shift only' "
    "when it contains 0; 'one-sided' = only one of the dFA and dFRR intervals excludes 0; "
    "'indistinguishable' = both contain 0.",
)


class DecisionQualityError(RuntimeError):
    pass


# ---- rules -----------------------------------------------------------------------------------


def _rule_v1(y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return y1 == 0


def _rule_v2(y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return y2 == 0


def _rule_or(y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return (y1 == 0) & (y2 == 0)


def _rule_and(y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return ~((y1 == 1) & (y2 == 1))


Rule = Callable[[np.ndarray, np.ndarray], np.ndarray]
RULES: dict[str, Rule] = {
    "V1_only": _rule_v1,
    "V2_only": _rule_v2,
    "OR": _rule_or,
    "AND": _rule_and,
}
"""Accept indicator of each threshold rule, from the observed flags alone."""
RULE_TEXT = {
    "V1_only": "reject iff V1 reports an error (y1 = 1)",
    "V2_only": "reject iff V2 reports an error (y2 = 1)",
    "OR": "reject iff either verifier reports an error",
    "AND": "reject iff both verifiers report an error",
}


# ---- inputs ----------------------------------------------------------------------------------


@dataclass
class Cohort:
    """One cohort's decisions as arrays aligned with ``states`` (sorted)."""

    name: str
    states: list[str]
    lams: list[str]
    flags: dict[str, tuple[np.ndarray, np.ndarray]]  # lam -> (y1, y2)
    accept: dict[str, dict[str, np.ndarray]]  # arm -> lam -> 0/1 (main accounting)
    parse_fail: dict[str, int] = field(default_factory=dict)  # arm -> count over all lambdas

    @property
    def arms(self) -> list[str]:
        return _ordered(self.accept)


def _ordered(arms: Iterable[str]) -> list[str]:
    arms = set(arms)
    return [a for a in ARM_ORDER if a in arms] + sorted(arms - set(ARM_ORDER))


def _logged_flags(name: str, r: Mapping[str, Any]) -> tuple[int, int]:
    f = r.get("fields") or {}
    if "y1" not in f or "y2" not in f:
        key = (r["state_id"], r["arm"], r["lam"])
        raise DecisionQualityError(f"{name}: {key} logs no flags (fields.y1/y2)")
    return int(f["y1"]), int(f["y2"])


def build_cohort(name: str, rows: Iterable[Mapping[str, Any]]) -> Cohort:
    """Arrays of one cohort's decision rows. Every arm must decide every (state, lambda) once and
    one state per x0. The cell of a (state, lambda) is its A1 row's flags; every arm shown the
    state's own flags (:data:`OWN_FLAG_ARMS`) must log the same, No Verifier and permuted-B
    (:data:`OTHER_FLAG_ARMS`) are not read; an arm in neither list is an error."""
    if name not in COHORTS:
        raise DecisionQualityError(f"unknown cohort {name!r}")
    rows = list(rows)
    if not rows:
        raise DecisionQualityError(f"{name}: no decision rows")
    x0_of: dict[str, str] = {}
    logged: dict[str, dict[tuple[str, str], tuple[int, int]]] = {}  # own-flag arm -> cell -> y
    seen: dict[str, dict[tuple[str, str], bool | None]] = {}
    pf: dict[str, int] = {}
    for r in rows:
        sid, lam, arm = r["state_id"], r["lam"], r["arm"]
        if lam not in LAMBDAS:
            raise DecisionQualityError(f"{name}: unknown lambda {lam!r}")
        if arm not in OWN_FLAG_ARMS and arm not in OTHER_FLAG_ARMS:
            raise DecisionQualityError(
                f"{name}: arm {arm!r} is in neither OWN_FLAG_ARMS nor OTHER_FLAG_ARMS"
            )
        if x0_of.setdefault(sid, r["x0_id"]) != r["x0_id"]:
            raise DecisionQualityError(f"{name}: state {sid} has two x0 ids")
        cells = seen.setdefault(arm, {})
        if (sid, lam) in cells:
            raise DecisionQualityError(f"{name}: {(sid, arm, lam)} decided twice")
        cells[(sid, lam)] = accepted(r, ACCOUNTING)  # type: ignore[arg-type]
        if arm in OWN_FLAG_ARMS:
            logged.setdefault(arm, {})[(sid, lam)] = _logged_flags(name, r)
        pf[arm] = pf.get(arm, 0) + int(bool(r["parse_fail"]))
    if len(set(x0_of.values())) != len(x0_of):
        raise DecisionQualityError(f"{name}: the x0 is resampled, one state per x0 is assumed")
    if FLAG_REFERENCE_ARM not in logged:
        raise DecisionQualityError(
            f"{name}: no {FLAG_REFERENCE_ARM} rows; the cells are its rows' flags"
        )
    flags = logged[FLAG_REFERENCE_ARM]
    states = sorted(x0_of)
    lams = [lam for lam in LAMBDAS if any(k[1] == lam for a in seen.values() for k in a)]
    need = len(states) * len(lams)
    for arm, cells in seen.items():
        if len(cells) != need:
            raise DecisionQualityError(
                f"{name}: arm {arm} decided {len(cells)} of the {need} (state, lambda) cells"
            )
    for arm in _ordered(logged):
        for key, y in logged[arm].items():
            if y != flags[key]:
                raise DecisionQualityError(
                    f"{name}: {arm} logs different flags for {key}: {y}, "
                    f"{FLAG_REFERENCE_ARM} logs {flags[key]}"
                )
    out_flags = {
        lam: tuple(np.array([flags[(s, lam)][i] for s in states], dtype=int) for i in (0, 1))
        for lam in lams
    }
    acc = {
        arm: {lam: np.array([float(cells[(s, lam)]) for s in states]) for lam in lams}
        for arm, cells in seen.items()
    }
    return Cohort(name, states, lams, out_flags, acc, {a: pf[a] for a in _ordered(pf)})


def lambda_levels(channel_cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The channel's per-lambda replacement probabilities; a level that replaces every score by
    a state-independent draw (p = 1 for both, independent resampling) is marked."""
    degrade = channel_cfg["degrade"]
    kind = str(degrade["q"]["kind"])
    out: dict[str, dict[str, Any]] = {}
    for lam, p in degrade["levels"].items():
        p = {k: float(v) for k, v in p.items()}
        indep = kind == "independent_resampling" and all(v == 1.0 for v in p.values())
        out[str(lam)] = {"p": p, "q": kind, "flags_independent_of_text": indep}
    return out


def _shown(path: Path) -> str:
    try:
        return paths.shown(path)
    except ValueError:
        return Path(path).as_posix()


def _require_files(rd: Path) -> None:
    for name in RUN_FILES:
        if not (rd / name).is_file():
            raise DecisionQualityError(f"{rd} has no {name}")


def load_cohorts(
    domain: str,
    e_incorrect_runs: Sequence[Path],
    e_correct_runs: Sequence[Path],
    *,
    original_decisions: Path | None,
    original_sha256: str | None,
    reused_arms: Sequence[str],
    prefix: str,
    base: Path | None = None,
    fake: bool = False,
) -> tuple[dict[str, Cohort], dict[str, Any], dict[str, dict[str, Any]]]:
    """Both cohorts, the inputs (path, sha256, rows) and the channel's lambda levels.

    Every run directory must be an extension run of its cohort (``ext_contrasts``'s rule) with
    every file of ``RUN_FILES`` and a snapshot ``configs/ext.yaml`` of ``domain``; their channel
    snapshots must agree. ``e_incorrect`` adds the original run's ``reused_arms`` decisions,
    sha256-checked unless ``fake``. A missing file is an error."""
    if not e_incorrect_runs or not e_correct_runs:
        raise DecisionQualityError("both cohorts need at least one extension run directory")
    inputs: dict[str, Any] = {E_INCORRECT: [], E_CORRECT: []}
    levels: dict[str, dict[str, Any]] | None = None
    for cohort, runs in ((E_INCORRECT, e_incorrect_runs), (E_CORRECT, e_correct_runs)):
        for rd in runs:
            rd = paths.resolve(rd)
            if not rd.is_dir():
                raise DecisionQualityError(f"{rd} does not exist")
            _require_files(rd)
            X.assert_extension_run(rd, prefix=prefix, cohort=cohort, base=base)
            snap = load_yaml(rd / "configs" / "ext.yaml")
            if snap.get("domain") != domain:
                raise DecisionQualityError(
                    f"{rd} is a {snap.get('domain')!r} run (configs/ext.yaml), not {domain!r}"
                )
            lv = lambda_levels(load_yaml(rd / "configs" / "channel.yaml"))
            if levels is None:
                levels = lv
            elif lv != levels:
                raise DecisionQualityError(f"{rd}: its channel levels differ from the other runs'")
            guard = require_fake if fake else refuse_fake
            guard(rd, read_jsonl(rd / "decisions.jsonl"), what="ext_decision_quality")
    original_rows: list[dict[str, Any]] = []
    if reused_arms:
        if original_decisions is None:
            raise DecisionQualityError("reused arms without the original run's decisions")
        if fake:
            f = Path(original_decisions)
            if not f.is_file():
                raise DecisionQualityError(f"{f} does not exist")
        else:
            f = checked(original_decisions, original_sha256, what="original Exp 1 decisions")
        original_rows = list(read_jsonl(f))
        (require_fake if fake else refuse_fake)(f, original_rows, what="ext_decision_quality")
        used = [r for r in original_rows if r["arm"] in reused_arms]
        inputs[E_INCORRECT].append(_input_entry(f, used, role="original run, reused arms"))
    rows = {
        E_INCORRECT: X.exp1_rows(
            E_INCORRECT,
            e_incorrect_runs,
            original_rows,
            tuple(reused_arms),
            prefix=prefix,
            base=base,
        ),
        E_CORRECT: X.exp1_rows(E_CORRECT, e_correct_runs, prefix=prefix, base=base),
    }
    for cohort, runs in ((E_INCORRECT, e_incorrect_runs), (E_CORRECT, e_correct_runs)):
        for rd in runs:
            f = paths.resolve(rd) / "decisions.jsonl"
            inputs[cohort].append(_input_entry(f, read_jsonl(f), role="extension run"))
    cohorts = {c: build_cohort(c, rows[c]) for c in COHORTS}
    assert levels is not None
    return cohorts, inputs, levels


def _input_entry(f: Path, rows: Iterable[Mapping[str, Any]], *, role: str) -> dict[str, Any]:
    rows = list(rows)
    return {
        "file": _shown(f),
        "sha256": file_sha256(f),
        "role": role,
        "rows_used": len(rows),
        "arms": _ordered({r["arm"] for r in rows}),
    }


# ---- bootstrap -------------------------------------------------------------------------------

Weights = Mapping[str, np.ndarray]  # cohort -> (k, n_states) multiplicities
Stat = Callable[[Weights], np.ndarray]  # -> (k,)


def _draw_weights(n: int, *, n_boot: int, seed: int, cohort: str) -> np.ndarray:
    """(n_boot, n): how often each state (x0) is drawn in each replicate of ``cohort``."""
    idx = seeded_rng(seed, "bootstrap", cohort).integers(0, n, (n_boot, n))
    flat = (idx + n * np.arange(n_boot)[:, None]).ravel()
    return np.bincount(flat, minlength=n_boot * n).reshape(n_boot, n).astype(float)


class Resampler:
    """The point estimate (every state once) and the percentile interval of any statistic of
    the cohorts' weights; each cohort has its own draw, shared by every statistic."""

    def __init__(
        self, sizes: Mapping[str, int], *, n_boot: int, seed: int, alpha: float = BOOTSTRAP_ALPHA
    ) -> None:
        self.alpha = alpha
        self.full = {c: np.ones((1, n)) for c, n in sizes.items()}
        self.reps = {
            c: _draw_weights(n, n_boot=n_boot, seed=seed, cohort=c) for c, n in sizes.items()
        }

    def summary(self, stat: Stat) -> dict[str, Any]:
        est = float(stat(self.full)[0])
        reps = np.asarray(stat(self.reps), dtype=float)
        bad = np.isnan(reps)
        if bad.all():
            ci = [math.nan, math.nan]
        else:
            lo, hi = np.quantile(reps[~bad], [self.alpha / 2, 1 - self.alpha / 2])
            ci = [float(lo), float(hi)]
        out: dict[str, Any] = {"estimate": est, "ci": ci}
        if bad.any():
            out["n_boot_undefined"] = int(bad.sum())
        return out


def _rate(cohort: str, v: np.ndarray) -> Stat:
    v = np.asarray(v, dtype=float)
    return lambda w: (w[cohort] @ v) / w[cohort].sum(axis=1)


def _cell_rate(cohort: str, v: np.ndarray, mask: np.ndarray) -> Stat:
    num = np.asarray(v, dtype=float) * mask
    den = mask.astype(float)

    def stat(w: Weights) -> np.ndarray:
        d = w[cohort] @ den
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(d > 0, (w[cohort] @ num) / d, np.nan)

    return stat


def _minus(a: Stat, b: Stat) -> Stat:
    return lambda w: a(w) - b(w)


# ---- statistics ------------------------------------------------------------------------------


def _masks(c: Cohort, lam: str) -> dict[str, np.ndarray]:
    y1, y2 = c.flags[lam]
    return {f"{a}{b}": (y1 == a) & (y2 == b) for a in (0, 1) for b in (0, 1)}


def rule_accept(c: Cohort, lam: str) -> dict[str, np.ndarray]:
    y1, y2 = c.flags[lam]
    return {name: rule(y1, y2).astype(float) for name, rule in RULES.items()}


def _gap(cohort: str, v: np.ndarray, masks: Mapping[str, np.ndarray]) -> Stat:
    return _minus(_cell_rate(cohort, v, masks["00"]), _cell_rate(cohort, v, masks["11"]))


def verdict_following(c: Cohort, rs: Resampler) -> dict[str, Any]:
    """Per lambda and arm: the four cells, the following gap, the gap minus No Verifier's and
    the agreement with each rule."""
    out: dict[str, Any] = {}
    for lam in c.lams:
        masks = _masks(c, lam)
        rules = rule_accept(c, lam)
        ref = c.accept.get(REFERENCE_ARM, {}).get(lam)
        cell: dict[str, Any] = {}
        for arm in c.arms:
            v = c.accept[arm][lam]
            cells = {}
            for pat in PATTERNS:
                m = masks[pat]
                cells[pat] = {
                    "n": int(m.sum()),
                    "n_accepted": int(v[m].sum()),
                    **rs.summary(_cell_rate(c.name, v, m)),
                }
            gap = _gap(c.name, v, masks)
            if arm == REFERENCE_ARM:
                vs_ref: dict[str, Any] = {"status": "reference"}
            elif ref is None:
                vs_ref = {"status": f"not available ({REFERENCE_ARM} not run)"}
            else:
                vs_ref = {"status": "ok", **rs.summary(_minus(gap, _gap(c.name, ref, masks)))}
            cell[arm] = {
                "cells": cells,
                "following_gap": rs.summary(gap),
                "gap_minus_no_verifier": vs_ref,
                "rule_agreement": {
                    name: rs.summary(_rate(c.name, (v == r).astype(float)))
                    for name, r in rules.items()
                },
            }
        out[lam] = cell
    return out


def _fa(acc_inc: np.ndarray) -> Stat:
    return _rate(E_INCORRECT, acc_inc)


def _frr(acc_cor: np.ndarray) -> Stat:
    return _rate(E_CORRECT, 1.0 - np.asarray(acc_cor, dtype=float))


def _ba(acc_inc: np.ndarray, acc_cor: np.ndarray) -> Stat:
    fa, frr = _fa(acc_inc), _frr(acc_cor)
    return lambda w: 1.0 - (fa(w) + frr(w)) / 2.0


def dominance(d_fa: float, d_frr: float) -> str:
    """Y relative to X from the point estimates of dFA = FA(Y) - FA(X), dFRR = FRR(Y) - FRR(X)."""
    if any(math.isnan(d) for d in (d_fa, d_frr)):
        return "n/a"
    s = {int(np.sign(d_fa)), int(np.sign(d_frr))}
    if s == {-1}:
        return "dominates"
    if s == {1}:
        return "dominated"
    if s == {-1, 1}:
        return "moved"
    if s == {0}:
        return "equal"
    return "weakly dominates" if -1 in s else "weakly dominated"


def _ci_sign(ci: Sequence[float]) -> int | None:
    lo, hi = ci
    if _nan(lo) or _nan(hi):
        return None
    return -1 if hi < 0 else (1 if lo > 0 else 0)


def dominance_ci(ci_fa: Sequence[float], ci_frr: Sequence[float], ci_ba: Sequence[float]) -> str:
    """Y relative to X from the 95% intervals of dFA, dFRR and dBA (2026-11-01). An
    interval excludes 0 when it lies wholly on one side of 0 (an end at exactly 0 does not).

    * dFA and dFRR both exclude 0 on the same side: 'dominates' (below) / 'dominated' (above);
    * on opposite sides: 'improved with trade-off' / 'worsened with trade-off' when dBA's
      interval excludes 0 above / below, 'shift only' when it contains 0;
    * only one of dFA, dFRR excludes 0: 'one-sided';
    * both contain 0: 'indistinguishable'."""
    fa, frr = _ci_sign(ci_fa), _ci_sign(ci_frr)
    if fa is None or frr is None:
        return "n/a"
    if fa == 0 and frr == 0:
        return "indistinguishable"
    if fa == 0 or frr == 0:
        return "one-sided"
    if fa == frr:
        return "dominates" if fa < 0 else "dominated"
    ba = _ci_sign(ci_ba)
    if ba is None:
        return "n/a"
    if ba == 0:
        return "shift only"
    return "improved with trade-off" if ba > 0 else "worsened with trade-off"


def _compare(
    rs: Resampler,
    x: tuple[np.ndarray, np.ndarray],
    y: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Y - X on (e_incorrect, e_correct) accept vectors: dBA, dFA, dFRR (paired) and dominance,
    from the point estimates and from the intervals."""
    d_ba = rs.summary(_minus(_ba(*y), _ba(*x)))
    d_fa = rs.summary(_rate(E_INCORRECT, y[0] - x[0]))
    d_frr = rs.summary(_rate(E_CORRECT, x[1] - y[1]))  # (1 - acc_Y) - (1 - acc_X)
    return {
        "dBA": d_ba,
        "dFA": d_fa,
        "dFRR": d_frr,
        "dominance": dominance(d_fa["estimate"], d_frr["estimate"]),
        "dominance_ci": dominance_ci(d_fa["ci"], d_frr["ci"], d_ba["ci"]),
    }


def operating_points(
    cohorts: Mapping[str, Cohort], rs: Resampler, pairs: Sequence[tuple[str, str]]
) -> dict[str, Any]:
    """Per lambda: FA, FRR, BA of every arm in both cohorts and of every rule; the pairs' and
    every arm-minus-rule comparison."""
    inc, cor = cohorts[E_INCORRECT], cohorts[E_CORRECT]
    arms = [a for a in inc.arms if a in cor.arms]
    out: dict[str, Any] = {}
    for lam in inc.lams:
        units: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
            "arms": {a: (inc.accept[a][lam], cor.accept[a][lam]) for a in arms},
            "rules": {r: (rule_accept(inc, lam)[r], rule_accept(cor, lam)[r]) for r in RULES},
        }
        points = {
            kind: {
                name: {
                    "FA": rs.summary(_fa(v[0])),
                    "FRR": rs.summary(_frr(v[1])),
                    "BA": rs.summary(_ba(*v)),
                }
                for name, v in members.items()
            }
            for kind, members in units.items()
        }
        pair_rows = []
        for x, y in pairs:
            missing = [a for a in (x, y) if a not in units["arms"]]
            row: dict[str, Any] = {"X": x, "Y": y}
            if missing:
                row["status"] = f"not available: {missing} not in both cohorts"
            else:
                row |= {"status": "ok", **_compare(rs, units["arms"][x], units["arms"][y])}
            pair_rows.append(row)
        vs_rule = [
            {"arm": a, "rule": r, **_compare(rs, units["rules"][r], units["arms"][a])}
            for a in arms
            for r in RULES
        ]
        out[lam] = {"points": points, "pairs": pair_rows, "arm_minus_rule": vs_rule}
    return out


def analyze(
    cohorts: Mapping[str, Cohort],
    *,
    domain: str,
    levels: Mapping[str, Mapping[str, Any]],
    n_boot: int = BOOTSTRAP_B,
    seed: int,
    alpha: float = BOOTSTRAP_ALPHA,
    pairs: Sequence[tuple[str, str]] = BA_PAIRS,
    inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if set(cohorts) != set(COHORTS):
        raise DecisionQualityError(f"both cohorts are needed, got {sorted(cohorts)}")
    inc, cor = cohorts[E_INCORRECT], cohorts[E_CORRECT]
    if inc.lams != cor.lams:
        raise DecisionQualityError(f"lambda levels differ: {inc.lams} vs {cor.lams}")
    unknown = [lam for lam in inc.lams if lam not in levels]
    if unknown:
        raise DecisionQualityError(f"the channel has no level {unknown}")
    rs = Resampler(
        {c: len(cohorts[c].states) for c in COHORTS}, n_boot=n_boot, seed=seed, alpha=alpha
    )
    return {
        "status": STATUS,
        "domain": domain,
        "accounting": ACCOUNTING_TEXT,
        "bootstrap": {
            "method": "percentile_bootstrap",
            "level": 1 - alpha,
            "n_boot": n_boot,
            "seed": seed,
            "seed_component": SEED_COMPONENT,
            "draw_key": {c: ["bootstrap", c] for c in COHORTS},
            "unit": "x0 (one state per x0); the two cohorts resampled independently; within a "
            "cohort every arm, rule and lambda shares the draw (paired)",
            "multiplicity": "none (exploratory)",
        },
        "lambda_levels": {lam: dict(levels[lam]) for lam in inc.lams},
        "rules": dict(RULE_TEXT),
        "pairs": [list(p) for p in pairs],
        "notes": list(NOTES),
        "inputs": dict(inputs or {}),
        "cohorts": {
            c.name: {
                "n_states": len(c.states),
                "arms": c.arms,
                "parse_fail": c.parse_fail,
                "cell_n": {
                    lam: {p: int(m.sum()) for p, m in _masks(c, lam).items()} for lam in c.lams
                },
            }
            for c in (inc, cor)
        },
        "arms_in_one_cohort_only": sorted(set(inc.arms) ^ set(cor.arms)),
        "verdict_following": {c.name: verdict_following(c, rs) for c in (inc, cor)},
        "operating_points": operating_points(cohorts, rs, pairs),
    }


# ---- tables ----------------------------------------------------------------------------------


def _nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _r(x: Any) -> str:
    return "n/a" if _nan(x) else f"{x:.3f}"


def _d(x: Any) -> str:
    return "n/a" if _nan(x) else f"{x:+.3f}"


def _est(s: Mapping[str, Any], *, signed: bool = False) -> str:
    f = _d if signed else _r
    lo, hi = s["ci"]
    text = f"{f(s['estimate'])} [{f(lo)}, {f(hi)}]"
    if s.get("n_boot_undefined"):
        text += f" ({s['n_boot_undefined']} undefined)"
    return text


def _lam(report: Mapping[str, Any], lam: str) -> str:
    return f"{lam} †" if report["lambda_levels"][lam]["flags_independent_of_text"] else lam


def _table(head: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines + [""]


def render(report: Mapping[str, Any]) -> str:
    """Markdown: every cell of the report."""
    b = report["bootstrap"]
    lams = list(report["lambda_levels"])
    marked = [lam for lam in lams if report["lambda_levels"][lam]["flags_independent_of_text"]]
    lines = [
        f"# Decision quality, Exp 1 ({report['domain']})",
        "",
        f"**{report['status']}**",
        "",
        f"Accounting: {report['accounting']}. Intervals: 95% percentile bootstrap, "
        f"B = {b['n_boot']}, seed {b['seed']} (component {b['seed_component']}); unit {b['unit']}; "
        "no multiplicity correction. Cells read estimate [95% CI]; '(k undefined)' = replicates "
        "in which the statistic had no state and were left out of the interval.",
        "",
    ]
    if marked:
        lines += [
            f"† lambda = {', '.join(marked)}: every score is replaced by a state-independent "
            "draw (p = 1 for both verifiers, independent resampling), so the flags are "
            "independent of the text.",
            "",
        ]
    lines += [
        "Channel levels: "
        + "; ".join(f"{lam}: p = {report['lambda_levels'][lam]['p']}" for lam in lams)
        + ".",
        "",
    ]
    lines += ["Notes:", ""] + [f"- {n}" for n in report["notes"]] + [""]
    lines += ["Rules: " + "; ".join(f"{k} = {v}" for k, v in report["rules"].items()) + ".", ""]
    lines += ["## Inputs", ""]
    if report.get("ext_config"):
        lines += [f"Extension config: {report['ext_config']}.", ""]
    lines += _table(
        ["cohort", "file", "role", "rows used", "arms", "sha256"],
        [
            [c, e["file"], e["role"], str(e["rows_used"]), ", ".join(e["arms"]), e["sha256"]]
            for c, entries in report["inputs"].items()
            for e in entries
        ],
    )
    lines += _table(
        ["cohort", "states", "parse_fail by arm", "cells (00 / 01 / 10 / 11) by lambda"],
        [
            [
                c,
                str(info["n_states"]),
                ", ".join(f"{a} {k}" for a, k in info["parse_fail"].items()),
                "; ".join(
                    f"{_lam(report, lam)}: " + " / ".join(str(n[p]) for p in PATTERNS)
                    for lam, n in info["cell_n"].items()
                ),
            ]
            for c, info in report["cohorts"].items()
        ],
    )
    if report["arms_in_one_cohort_only"]:
        lines += [f"Arms in one cohort only: {report['arms_in_one_cohort_only']}", ""]

    lines += ["## 1. Verdict following", ""]
    for cohort, by_lam in report["verdict_following"].items():
        n = report["cohorts"][cohort]["n_states"]
        lines += [f"### {cohort} ({n} states): accept rate by observed (y1, y2)", ""]
        for lam, arms in by_lam.items():
            lines += [f"#### lambda = {_lam(report, lam)}", ""]
            lines += _table(
                [
                    "arm",
                    *[f"({p[0]},{p[1]})" for p in PATTERNS],
                    "gap (0,0) - (1,1)",
                    "gap - No Verifier",
                ],
                [
                    [
                        arm,
                        *[
                            f"{_est(a['cells'][p])} ({a['cells'][p]['n_accepted']}/"
                            f"{a['cells'][p]['n']})"
                            for p in PATTERNS
                        ],
                        _est(a["following_gap"], signed=True),
                        _est(a["gap_minus_no_verifier"], signed=True)
                        if a["gap_minus_no_verifier"]["status"] == "ok"
                        else a["gap_minus_no_verifier"]["status"],
                    ]
                    for arm, a in arms.items()
                ],
            )
        lines += [f"#### {cohort}: share of decisions equal to each rule's", ""]
        lines += _table(
            ["lambda", "arm", *RULES],
            [
                [_lam(report, lam), arm, *[_est(a["rule_agreement"][r]) for r in RULES]]
                for lam, arms in by_lam.items()
                for arm, a in arms.items()
            ],
        )

    ops = report["operating_points"]
    lines += ["## 2. Threshold rules (FA on e_incorrect, FRR on e_correct)", ""]
    lines += _table(
        ["rule", "lambda", "FA", "FRR", "BA"],
        [
            [
                r,
                _lam(report, lam),
                *[_est(ops[lam]["points"]["rules"][r][k]) for k in ("FA", "FRR", "BA")],
            ]
            for r in RULES
            for lam in lams
        ],
    )
    lines += ["## 3. Operating points and balanced accuracy", ""]
    lines += ["BA = 1 - (FA + FRR) / 2; FA from e_incorrect, FRR from e_correct.", ""]
    lines += ["### Every arm and rule", ""]
    lines += _table(
        ["lambda", "kind", "arm / rule", "FA", "FRR", "BA"],
        [
            [_lam(report, lam), kind.rstrip("s"), name, *[_est(p[k]) for k in ("FA", "FRR", "BA")]]
            for lam in lams
            for kind, members in ops[lam]["points"].items()
            for name, p in members.items()
        ],
    )
    lines += [
        "### BA(Y) - BA(X), the seven confirmed pairs (X less information, Y more)",
        "",
        "Positive dBA = the added information raised BA. dFA = FA(Y) - FA(X), "
        "dFRR = FRR(Y) - FRR(X); 'Y vs X' from their point estimates, 'Y vs X (CI)' from the "
        "95% intervals of dFA, dFRR and dBA (see the notes).",
        "",
    ]
    lines += _table(
        ["lambda", "X", "Y", "BA(Y) - BA(X)", "dFA", "dFRR", "Y vs X", "Y vs X (CI)"],
        [
            [_lam(report, lam), p["X"], p["Y"], *_cmp_cells(p)]
            for lam in lams
            for p in ops[lam]["pairs"]
        ],
    )
    lines += [
        "### BA(arm) - BA(rule)",
        "",
        "Positive = the controller arm's BA is above the rule's. dFA = FA(arm) - FA(rule), "
        "dFRR = FRR(arm) - FRR(rule); 'arm vs rule' from their point estimates, 'arm vs rule "
        "(CI)' from the 95% intervals of dFA, dFRR and dBA (see the notes).",
        "",
    ]
    lines += _table(
        [
            "lambda",
            "arm",
            "rule",
            "BA(arm) - BA(rule)",
            "dFA",
            "dFRR",
            "arm vs rule",
            "arm vs rule (CI)",
        ],
        [
            [_lam(report, lam), c["arm"], c["rule"], *_cmp_cells(c)]
            for lam in lams
            for c in ops[lam]["arm_minus_rule"]
        ],
    )
    return "\n".join(lines)


def _cmp_cells(c: Mapping[str, Any]) -> list[str]:
    if c.get("status", "ok") != "ok":
        return [c["status"], "", "", "", ""]
    return [
        _est(c["dBA"], signed=True),
        _est(c["dFA"], signed=True),
        _est(c["dFRR"], signed=True),
        c["dominance"],
        c["dominance_ci"],
    ]


# ---- figure ----------------------------------------------------------------------------------

ARM_LABELS = {
    "no_verifier": "No Verifier",
    "A1": "A$_1$",
    "verdict_only": "Verdict-only",
    "B": "B",
    "B_pihat": r"B + $\hat{\pi}$",
    "B_pihat_mismatched": r"B + $\hat{\pi}$ mismatched",
    "permuted_B": "permuted-B",
}
RULE_LABELS = {"V1_only": "V1 only", "V2_only": "V2 only", "OR": "OR", "AND": "AND"}
# categorical slots 1-7 of the dataviz reference palette (light), in fixed order; with seven
# series on one scatter the marker shape and the legend carry identity as well
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
MARKERS = ("o", "s", "D", "^", "v", "P", "X")
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#fcfcfb",
)


def plot(report: Mapping[str, Any], out_dir: Path, *, stem: str) -> list[Path]:
    """One FA-vs-FRR panel per lambda: every arm with its intervals, the rules as reference
    points. Writes ``<stem>.png`` and ``<stem>.pdf``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ops = report["operating_points"]
    lams = list(ops)
    arms = _ordered({a for lam in lams for a in ops[lam]["points"]["arms"]})
    style = {a: (SERIES[i % len(SERIES)], MARKERS[i % len(MARKERS)]) for i, a in enumerate(arms)}
    every = [
        p[k]
        for lam in lams
        for members in ops[lam]["points"].values()
        for p in members.values()
        for k in ("FA", "FRR")
    ]
    top = max([v for s in every for v in (s["estimate"], *s["ci"]) if not _nan(v)] + [0.05])
    lim = min(1.0, top * 1.08 + 0.02)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(
        1, len(lams), figsize=(4.3 * len(lams), 4.9), sharex=True, sharey=True, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for ax, lam in zip(axes[0], lams, strict=True):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK_2, labelsize=8)
        offsets = _label_offsets(
            {
                r: (p["FRR"]["estimate"], p["FA"]["estimate"])
                for r, p in ops[lam]["points"]["rules"].items()
            },
            [
                (p["FRR"]["estimate"], p["FA"]["estimate"])
                for p in ops[lam]["points"]["arms"].values()
            ],
            lim,
        )
        for r, p in ops[lam]["points"]["rules"].items():
            x, y = p["FRR"]["estimate"], p["FA"]["estimate"]
            ax.errorbar(
                x,
                y,
                xerr=_err(p["FRR"]),
                yerr=_err(p["FA"]),
                fmt="none",
                ecolor=MUTED,
                elinewidth=0.8,
                capsize=2,
                zorder=2,
            )
            ax.plot(
                x,
                y,
                marker="o",
                markersize=9,
                markerfacecolor="none",  # an open ring above the arms: hides none of them
                markeredgecolor=INK_2,
                markeredgewidth=1.4,
                linestyle="none",
                zorder=5,
            )
            dx, dy, ha = offsets[r]
            ax.annotate(
                RULE_LABELS.get(r, r),
                (x, y),
                xytext=(dx, dy),
                ha=ha,
                textcoords="offset points",
                fontsize=8,
                color=INK_2,
                zorder=5,
            )
        for a, p in ops[lam]["points"]["arms"].items():
            color, marker = style[a]
            x, y = p["FRR"]["estimate"], p["FA"]["estimate"]
            ax.errorbar(
                x,
                y,
                xerr=_err(p["FRR"]),
                yerr=_err(p["FA"]),
                fmt="none",
                ecolor=color,
                elinewidth=1.0,
                capsize=2,
                alpha=0.9,
                zorder=3,
            )
            ax.plot(
                x,
                y,
                marker=marker,
                markersize=8,
                color=color,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                linestyle="none",
                zorder=4,
            )
        title = f"$\\lambda$ = {lam}"
        if report["lambda_levels"][lam]["flags_independent_of_text"]:
            title += "  † flags independent of the text"
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("FRR (false reject rate, e_correct)", color=INK_2)
    axes[0][0].set_ylabel("FA (false accept rate, e_incorrect)", color=INK_2)
    handles = [
        plt.Line2D(
            [],
            [],
            marker=style[a][1],
            color=style[a][0],
            linestyle="none",
            markersize=8,
            markeredgecolor=SURFACE,
            label=ARM_LABELS.get(a, a),
        )
        for a in arms
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="none",
            markeredgecolor=INK_2,
            markeredgewidth=1.4,
            linestyle="none",
            markersize=9,
            label="threshold rule (labelled)",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 4),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    b = report["bootstrap"]
    fig.suptitle(
        f"Exp 1 operating points ({report['domain']}): FA vs FRR, 95% percentile bootstrap "
        f"intervals (B = {b['n_boot']})",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.text(0.01, 0.925, report["status"], ha="left", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.17, 1, 0.92))
    out = [Path(out_dir) / f"{stem}.png", Path(out_dir) / f"{stem}.pdf"]
    fig.savefig(out[0], dpi=200, facecolor=SURFACE)
    fig.savefig(out[1], facecolor=SURFACE)
    plt.close(fig)
    return out


LABEL_SPOTS = ((7, 4, "left"), (7, -11, "left"), (-7, 4, "right"), (-7, -11, "right"))
"""Candidate (dx, dy) offsets in points of a rule's label, and its alignment."""


def _label_offsets(
    rules: Mapping[str, tuple[float, float]],
    others: Sequence[tuple[float, float]],
    lim: float,
) -> dict[str, tuple[int, int, str]]:
    """Greedy label placement: each rule's label goes to the spot whose approximate centre is
    farthest from every marker and every label already placed (axis-fraction units, a panel
    taken as about 260 points wide and a label as about 36 points)."""
    unit = 1.0 / 260.0
    marks = [
        (x / lim, y / lim) for x, y in [*rules.values(), *others] if not _nan(x) and not _nan(y)
    ]
    placed: list[tuple[float, float]] = []
    out: dict[str, tuple[int, int, str]] = {}
    for r, (x, y) in rules.items():
        if _nan(x) or _nan(y):
            out[r] = LABEL_SPOTS[0]
            continue
        best, best_score = LABEL_SPOTS[0], -1.0
        for dx, dy, ha in LABEL_SPOTS:
            cx = x / lim + (dx + math.copysign(18, dx)) * unit
            cy = y / lim + (dy + 4) * unit
            score = min(
                (math.hypot(cx - a, cy - b) for a, b in [*marks, *placed]), default=math.inf
            )
            if score > best_score:
                best, best_score = (dx, dy, ha), score
        out[r] = best
        placed.append(
            (
                x / lim + (best[0] + math.copysign(18, best[0])) * unit,
                y / lim + (best[1] + 4) * unit,
            )
        )
    return out


def _err(s: Mapping[str, Any]) -> list[list[float]] | None:
    lo, hi = s["ci"]
    if any(_nan(v) for v in (s["estimate"], lo, hi)):
        return None
    return [[max(s["estimate"] - lo, 0.0)], [max(hi - s["estimate"], 0.0)]]


# ---- fake data -------------------------------------------------------------------------------

FAKE_DOMAIN = "fake"
FAKE_REUSED = ("A1", "B", "permuted_B")
FAKE_FOLLOW = {
    "no_verifier": 0.0,
    "A1": 1.2,
    "verdict_only": 1.6,
    "B": 1.4,
    "B_pihat": 1.1,
    "B_pihat_mismatched": 1.1,
    "permuted_B": 1.4,
}
FAKE_CHANNEL = {
    "degrade": {
        "levels": {
            "low": {"v1": 0.0, "v2": 0.0},
            "mid": {"v1": 0.5, "v2": 0.5},
            "high": {"v1": 1.0, "v2": 1.0},
        },
        "q": {"kind": "independent_resampling"},
    }
}


def _jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(dict(r)) + "\n")


def _fake_rows(
    rnd: random.Random, cohort: str, n: int, arms: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    """Decision rows by arm: a latent badness per state; flags that see it at low, half
    replaced at mid, all replaced at high; the controller's own noisy reading; each arm follows
    the flags it is shown with its own weight (permuted-B is shown a donor's)."""
    mu = 0.8 if cohort == E_INCORRECT else -0.8
    tag = "i" if cohort == E_INCORRECT else "c"
    states = [f"x0-fake{tag}{i:04d}-g1" for i in range(n)]
    z = {s: rnd.gauss(mu, 1.0) for s in states}
    own = {s: z[s] + rnd.gauss(0.0, 1.0) for s in states}
    u = {(s, v): rnd.random() for s in states for v in (1, 2)}
    true = {(s, v): int(z[s] + rnd.gauss(0.0, 0.8) > 0) for s in states for v in (1, 2)}
    noise = {(s, v): int(rnd.random() < 0.3) for s in states for v in (1, 2)}
    p = {lam: FAKE_CHANNEL["degrade"]["levels"][lam]["v1"] for lam in LAMBDAS}

    def flag(s: str, v: int, lam: str) -> int:
        return noise[(s, v)] if u[(s, v)] < p[lam] else true[(s, v)]

    donor = dict(zip(states, states[1:] + states[:1], strict=True))
    out: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        rows = []
        for s in states:
            for lam in LAMBDAS:
                shown = donor[s] if arm == "permuted_B" else s
                k = flag(shown, 1, lam) + flag(shown, 2, lam)
                logit = 1.0 - FAKE_FOLLOW[arm] * k - 1.2 * own[s]
                pf = rnd.random() < 0.005
                acc = rnd.random() < 1.0 / (1.0 + math.exp(-logit))
                rows.append(
                    {
                        "run_id": f"{FAKE_RUN_PREFIX}{cohort}",
                        "state_id": s,
                        "x0_id": s.removesuffix("-g1"),
                        "arm": arm,
                        "lam": lam,
                        "t": 1,
                        "fields": {"y1": flag(s, 1, lam), "y2": flag(s, 2, lam)},
                        "permuted_from_state_id": donor[s] if arm == "permuted_B" else None,
                        "parse_fail": pf,
                        "action": None if pf else ("accept" if acc else "rewrite"),
                    }
                )
        out[arm] = rows
    return out


def write_fake(root: Path, *, seed: int = 11, n_inc: int = 160, n_cor: int = 140) -> dict[str, Any]:
    """Fake inputs in the real layouts (an original run and extension runs named and shaped as
    the real ones, with their config snapshots), rows shuffled, under a FAKE marker."""
    rnd = random.Random(seed)  # noqa: S311 - fake data only
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / FAKE_MARKER).write_text(
        "fake data for analysis/ext_decision_quality.py\n", encoding="utf-8", newline="\n"
    )

    def ext_run(name: str, cohort: str, rows: list[dict[str, Any]]) -> Path:
        rd = root / name
        (rd / "configs").mkdir(parents=True, exist_ok=True)
        (rd / "ext_meta.json").write_text(
            json.dumps({"cohort": cohort, "smoke": False}), encoding="utf-8", newline="\n"
        )
        (rd / "configs" / "ext.yaml").write_text(
            f"domain: {FAKE_DOMAIN}\n", encoding="utf-8", newline="\n"
        )
        (rd / "configs" / "channel.yaml").write_text(
            json.dumps(FAKE_CHANNEL), encoding="utf-8", newline="\n"
        )
        rnd.shuffle(rows)
        _jsonl(rd / "decisions.jsonl", rows)
        return rd

    inc = _fake_rows(rnd, E_INCORRECT, n_inc, ARM_ORDER)
    cor = _fake_rows(rnd, E_CORRECT, n_cor, ARM_ORDER)
    original = [r for a in FAKE_REUSED for r in inc[a]]
    rnd.shuffle(original)
    _jsonl(root / "exp1_original" / "decisions.jsonl", original)
    new = [r for a in ARM_ORDER if a not in FAKE_REUSED for r in inc[a]]
    return {
        "base": root,
        "original_decisions": root / "exp1_original" / "decisions.jsonl",
        "reused_arms": FAKE_REUSED,
        E_INCORRECT: [ext_run("exp1ext_e_incorrect_FAKE", E_INCORRECT, new)],
        E_CORRECT: [
            ext_run("exp1ext_e_correct_FAKE", E_CORRECT, [r for a in ARM_ORDER for r in cor[a]])
        ],
    }


def analyze_fake(
    root: Path, *, n_boot: int = BOOTSTRAP_B, seed: int = 0, **fake_kw: Any
) -> dict[str, Any]:
    """Write fake inputs under ``root`` and run the whole analysis on them."""
    spec = write_fake(root, **fake_kw)
    cohorts, inputs, levels = load_cohorts(
        FAKE_DOMAIN,
        spec[E_INCORRECT],
        spec[E_CORRECT],
        original_decisions=spec["original_decisions"],
        original_sha256=None,
        reused_arms=spec["reused_arms"],
        prefix=X.RUN_PREFIXES["exp1"],
        base=spec["base"],
        fake=True,
    )
    return analyze(
        cohorts, domain=FAKE_DOMAIN, levels=levels, n_boot=n_boot, seed=seed, inputs=inputs
    )


# ---- entry -----------------------------------------------------------------------------------


def _clean(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    return obj


def write_outputs(report: Mapping[str, Any], out: Path, *, prefix: str = "") -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    files = [out / f"{prefix}report.json", out / f"{prefix}tables.md"]
    files[0].write_text(json.dumps(_clean(report), indent=1), encoding="utf-8", newline="\n")
    files[1].write_text(render(report), encoding="utf-8", newline="\n")
    return files + plot(report, out, stem=f"{prefix}operating_points")


def domain_inputs(domain: str, ext_config: str | None = None) -> dict[str, Any]:
    """What the entry reads from the domain's extension config (``ext_analysis.DOMAINS``, or the
    ``ext_config`` path): the Exp 1 run prefix and the original run's reused arms, decisions
    file and its sha256."""
    try:
        ext = W.load_domain_ext(domain, ext_config)
    except W.ExtDomainError as exc:
        raise DecisionQualityError(str(exc)) from exc
    inc = get(ext, "exp1.e_incorrect")
    reused = list(inc.get("reused_arms") or [])
    original = None
    sha = None
    if reused:
        original = paths.resolve(inc["original_run"]) / "decisions.jsonl"
        sha = (inc.get("original_sha256") or {}).get("decisions.jsonl")
    return {
        "ext_config": Path(ext_config or W.domain(domain).ext_config).as_posix(),
        "prefix": str(get(ext, "exp1.run_id_prefix")),
        "reused_arms": reused,
        "original_decisions": original,
        "original_sha256": sha,
    }


def domain_seed(domain: str) -> int:
    """The bootstrap seed: the domain's master seed with :data:`SEED_COMPONENT`."""
    try:
        return component_seed(W.domain_master_seed(domain), SEED_COMPONENT)
    except W.ExtDomainError as exc:
        raise DecisionQualityError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--domain", help="the domain whose extension runs are read (detox, ...)")
    ap.add_argument(
        "--e-incorrect",
        type=Path,
        nargs="+",
        default=[],
        help="Exp 1 extension run directories of the e_incorrect cohort",
    )
    ap.add_argument(
        "--e-correct",
        type=Path,
        nargs="+",
        default=[],
        help="Exp 1 extension run directories of the e_correct cohort",
    )
    ap.add_argument(
        "--ext-config", help="the domain's extension config (default: ext_analysis.DOMAINS)"
    )
    ap.add_argument("--fake", action="store_true", help="end-to-end run on shuffled fake data")
    args = ap.parse_args(argv)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.fake:
        out = paths.run_dir(f"{OUT_PREFIX}_FAKE_{ts}")
        report = analyze_fake(out / "inputs", seed=domain_seed(args.domain or "detox"))
        (out / FAKE_MARKER).write_text(
            "fake decision-quality analysis\n", encoding="utf-8", newline="\n"
        )
        prefix = FAKE_FILE_PREFIX
    else:
        if not args.domain or not args.e_incorrect or not args.e_correct:
            ap.error("--domain, --e-incorrect and --e-correct are required (or --fake)")
        d = domain_inputs(args.domain, args.ext_config)
        seed = domain_seed(args.domain)
        cohorts, inputs, levels = load_cohorts(
            args.domain,
            args.e_incorrect,
            args.e_correct,
            original_decisions=d["original_decisions"],
            original_sha256=d["original_sha256"],
            reused_arms=d["reused_arms"],
            prefix=d["prefix"],
        )
        report = analyze(cohorts, domain=args.domain, levels=levels, seed=seed, inputs=inputs)
        report["ext_config"] = d["ext_config"]
        out = paths.run_dir(f"{OUT_PREFIX}_{args.domain}_{ts}")
        prefix = ""
    files = write_outputs(report, out, prefix=prefix)
    for f in files:
        print(f"[ext_decision_quality] -> {_shown(f)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
