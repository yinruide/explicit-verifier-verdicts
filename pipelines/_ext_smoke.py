"""The engineering smokes of the extension (2026-10-31): states outside every
formal set, as step 10a's smoke.

* E-incorrect: the unlabeled pool's E-UNSAFE states, read under DRY_RUN (as step 3 and step
  10a did), never D_Exp1;
* E-correct: D_gate's E-correct states (detox: SAFE) that the E-correct cohort did not draw,
  read under EXT_SMOKE, never the cohort.

``n`` of them are drawn without replacement with the seed component ``ext_smoke`` (one stream per
cohort, so Exp 1's and Exp 2's smoke of a cohort run the same states when ``n`` is the same).
Their y* is the smoke's outcome (Exp 1) and its t = 1 judgment (Exp 2). A smoke also runs in
the seed namespace ``<namespace>_smoke`` and with a cache of its own, in a run directory named
``<prefix>_smoke_<cohort>_<ts>`` that the extension analysis refuses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipelines import _validity_io as io
from rafr.contracts.schemas import EvalRecord, SafetyLabel, Split, State
from rafr.core import paths
from rafr.core.config import get
from rafr.core.seeding import component_seed
from rafr.data import labels as L
from rafr.data.splits import load_split
from rafr.evaluator import judge
from rafr.experiments.ext_cohort import draw_smoke, read_cohort

SMOKE_COMPONENT = "ext_smoke"


@dataclass(frozen=True)
class SmokeSource:
    split: Split
    purpose: L.LabelPurpose
    label: SafetyLabel


SOURCES: dict[str, SmokeSource] = {
    "e_incorrect": SmokeSource(Split.UNLABELED_POOL, L.LabelPurpose.DRY_RUN, SafetyLabel.UNSAFE),
    "e_correct": SmokeSource(Split.D_GATE, L.LabelPurpose.EXT_SMOKE, SafetyLabel.SAFE),
}


def smoke_candidates(
    cohort: str,
    *,
    exclude: set[str] | frozenset[str] = frozenset(),
    labels_dir: Path | None = None,
    split_dir: Path | None = None,
) -> tuple[list[str], dict[str, EvalRecord]]:
    """The cohort's smoke candidates (state ids, sorted) and their y* records."""
    src = SOURCES[cohort]
    ys = L.read_labels(src.split, src.purpose, labels_dir=labels_dir, split_dir=split_dir)
    records = {y.state_id: y.record for y in ys}
    ids = sorted(y.state_id for y in ys if y.label is src.label and y.state_id not in exclude)
    return ids, records


def smoke_states(
    ext: Mapping[str, Any], cohort: str, n: int, master: int, ecfg: judge.EvaluatorConfig
) -> tuple[list[State], dict[str, EvalRecord]]:
    """``n`` smoke states of ``cohort`` (in the split's file order) and their y* records, each
    made by the configured E. Asserts that none is in D_Exp1 or the E-correct cohort."""
    src = SOURCES[cohort]
    spec = get(ext, "cohorts.e_correct")
    _, cohort_ids = read_cohort(paths.resolve(spec["file"]), spec["file_sha256"])
    ids, records = smoke_candidates(cohort, exclude=set(cohort_ids))
    drawn = set(draw_smoke(ids, n, component_seed(master, SMOKE_COMPONENT), cohort))
    states = [s for s in io.load_states(src.split) if s.state_id in drawn]
    if len(states) != n:
        raise SystemExit(f"{len(states)} of the {n} smoke states have a state row")
    exp1_x0 = set(load_split(Split.D_EXP1, paths.DATA_SPLITS))
    formal = [s.state_id for s in states if s.x0_id in exp1_x0 or s.state_id in set(cohort_ids)]
    if formal:
        raise SystemExit(f"smoke state(s) inside a formal set: {formal[:3]}")
    for s in states:
        judge.check_same_e(
            judge.record_identity(records[s.state_id]), ecfg, where=f"{src.split.value}'s y*"
        )
    return states, {s.state_id: records[s.state_id] for s in states}


__all__ = ["SMOKE_COMPONENT", "SOURCES", "SmokeSource", "smoke_candidates", "smoke_states"]
