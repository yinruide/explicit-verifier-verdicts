"""Purpose-gated access to y* (paper Sec. 4.2, 3, paper Sec. 4.3).

**y* is E's judgment of each state** (paper Sec. 3): step05 judges every state once with the one
E and writes the records to ``data/labels/state_judgments.jsonl``; there are no human labels
any more. That file is read here and nowhere else (step05, which writes it, is the only other
module allowed to open it; ``tests/test_v09.py`` scans for that). Every read names a split and
a purpose, and :data:`ALLOWED` decides:

* D_fit: tau fitting, the step 8 check of C_lambda, the baseline diagnostics, step 2's verifier
  selection;
* D_gate: the gate;
* D_gate_reserve: nothing, until ``frozen/gate_attempts.json`` records a FAILED first gate
  attempt; then the gate only. No statistic or report reads it before that -- the split
  report gives its size and nothing else (paper Sec. 4.3);
* D_Exp1: the Exp 1 outcome (it IS D_Exp1's y*), read for the analysis; and Exp 2's t = 1
  judgments, which ARE the same judgments (paper Sec. 4.4,: t = 1 reuses step 5's E judgment);
* unlabeled pool: step 2's verifier selection, and step 3's dry run (which E-UNSAFE pool x0 it
  runs on, and their outcome); the channel reads scores, never y*;
* D_fit and the unlabeled pool: step 6's verifier diagnostics -- the full table on
  D_fit + pool only, and the 150 E-UNSAFE + 150 E-SAFE states of the wording-perturbation
  sample (2026-09-25).
* post-pilot extension (2026-10-31), after the gate and both experiments: the
  unlabeled pool for fitting the pi-hat calibrator; D_gate for its diagnostics and for the
  E-correct cohort (drawing it, and its y* as that cohort's outcome); D_gate again for the
  E-correct smoke (states outside the cohort, and their y*).

``SPLIT_REPORT`` is the one read step05 makes after the split: the E-UNSAFE count and share of
D_fit (2026-09-23). The split report gives every other split its size only, and no
code path reads D_gate's y* outside the gate -- together with the total over all states (the
segment-3 report) nothing else would let the reserve's count be worked out by subtraction.

Nothing on the decision path (channel, formatter, controller) may import this module;
``tests/test_boundaries.py`` enforces that.

Math (paper App. D.1) uses the same access matrix with CORRECT/INCORRECT and
separate run-local label/split paths. Its initial-state producer is
``pipelines.math_initial``. UNRESOLVED is recorded for audit but never exposed as
an initial y*: invalid states are removed before splitting.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from rafr.contracts.schemas import EvalRecord, MathLabel, SafetyLabel, Split
from rafr.core import paths
from rafr.data.splits import load_split

Y_STAR_FILENAME = "state_judgments.jsonl"
"""E's judgment of every state, under ``paths.DATA_LABELS``. Written by step05 only."""


class LabelPurpose(StrEnum):
    TAU_FIT = "tau_fit"  # paper Sec. 5
    CHANNEL_CHECK = "channel_check"  # paper Sec. 4.3 / step 8: check C_lambda on D_fit
    BASELINE_DIAGNOSTICS = "baseline_diagnostics"  # paper App. A lambda = 0 baseline
    VERIFIER_SELECTION = "verifier_selection"  # step 2
    VERIFIER_DIAGNOSTICS = "verifier_diagnostics"  # step 6: on D_fit + pool only
    DRY_RUN = "dry_run"  # step 3: E-UNSAFE pool x0 for the engineering run
    GATE = "gate"  # paper Sec. 4.3
    EXP1_OUTCOME = "exp1_outcome"  # paper Sec. 3: D_Exp1's y* is the Exp 1 outcome
    EXP2_T1 = "exp2_t1"  # paper Sec. 4.4: Exp 2's x1 are D_Exp1's states; t = 1 reuses y*
    SPLIT_REPORT = "split_report"  # step05's split report: D_fit's E-UNSAFE count only
    # post-pilot extension (2026-10-31): the pi-hat calibrator is fitted on the pool and
    # diagnosed on D_gate; the E-correct cohort is drawn from D_gate's E-correct states, whose
    # y* is then that cohort's outcome (Exp 1) and its t = 1 judgment (Exp 2)
    CALIBRATION_FIT = "calibration_fit"
    CALIBRATION_DIAG = "calibration_diag"
    ECORRECT_COHORT = "ecorrect_cohort"
    # the smoke of the E-correct cohort (2026-10-31): D_gate's E-correct states
    # left out of the cohort, which states they are and their y*. The E-incorrect smoke reads the
    # pool's E-UNSAFE states under DRY_RUN, as step 3 and step 10a did.
    EXT_SMOKE = "ext_smoke"


class Operation(StrEnum):
    ANNOTATE = "annotate"  # only the retired annotation/ package still asks
    SCORE = "score"
    FIT = "fit"


ALLOWED: dict[Split, frozenset[LabelPurpose]] = {
    Split.D_FIT: frozenset(
        {
            LabelPurpose.TAU_FIT,
            LabelPurpose.CHANNEL_CHECK,
            LabelPurpose.BASELINE_DIAGNOSTICS,
            LabelPurpose.VERIFIER_SELECTION,
            LabelPurpose.VERIFIER_DIAGNOSTICS,
            LabelPurpose.SPLIT_REPORT,
        }
    ),
    Split.D_GATE: frozenset(
        {
            LabelPurpose.GATE,
            LabelPurpose.CALIBRATION_DIAG,
            LabelPurpose.ECORRECT_COHORT,
            LabelPurpose.EXT_SMOKE,
        }
    ),
    # and only once the first gate attempt is recorded as failed (paper Sec. 4.3)
    Split.D_GATE_RESERVE: frozenset({LabelPurpose.GATE}),
    Split.D_EXP1: frozenset({LabelPurpose.EXP1_OUTCOME, LabelPurpose.EXP2_T1}),
    Split.UNLABELED_POOL: frozenset(
        {
            LabelPurpose.VERIFIER_SELECTION,
            LabelPurpose.VERIFIER_DIAGNOSTICS,
            LabelPurpose.DRY_RUN,
            LabelPurpose.CALIBRATION_FIT,
        }
    ),
}

SPLIT_REPORT_COUNTS: frozenset[Split] = frozenset({Split.D_FIT})
"""The one split whose E-UNSAFE count the split report gives (2026-09-23)."""


class LabelAccessError(PermissionError):
    pass


class LabelDataError(RuntimeError):
    pass


class StateJudgment(BaseModel):
    """One row of the y* file: E's judgment of one state, or the failure that dropped it.

    ``label`` is None when E still failed after its retries (paper Sec. 3); such a state and its x0
    are dropped before the split. A later run of step05's judge phase retries it, and the
    last row per state is the one that counts.
    """

    model_config = ConfigDict(extra="forbid")

    state_id: str
    x0_id: str
    label: SafetyLabel | MathLabel | None
    e_error: str | None = None
    record: EvalRecord | None = None
    # the E that made (or failed to make) this judgment: judge.EvaluatorConfig.identity()
    e_identity: dict[str, Any]


class YStar(BaseModel):
    """y* of one state, as handed out by :func:`read_labels`."""

    model_config = ConfigDict(extra="forbid")

    state_id: str
    x0_id: str
    split: Split
    label: SafetyLabel | MathLabel
    record: EvalRecord


def y_star_path(labels_dir: Path | None = None) -> Path:
    return Path(labels_dir or paths.DATA_LABELS) / Y_STAR_FILENAME


def load_judgment_rows(path: Path | None = None) -> dict[str, StateJudgment]:
    """Every state's latest row of the y* file, keyed by state_id.

    For step05 (the producer: resume, drop list, the split) and for :func:`read_labels`.
    Anything else reads y* through :func:`read_labels`, which applies the purpose gate.
    """
    p = Path(path) if path else y_star_path()
    rows: dict[str, StateJudgment] = {}
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = StateJudgment.model_validate(json.loads(line))
                rows[row.state_id] = row
    return rows


def read_gate_attempts(path: Path | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else paths.FROZEN_GATE_ATTEMPTS
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh) or {}
    return list(data.get("attempts", []))


def reserve_unlocked(gate_attempts_path: Path | None = None) -> bool:
    """True iff the gate attempts file records at least one FAILED attempt (paper Sec. 4.3)."""
    return any(a.get("passed") is False for a in read_gate_attempts(gate_attempts_path))


def check_split_usable(
    split: Split, operation: Operation, *, gate_attempts_path: Path | None = None
) -> None:
    """Raise if ``operation`` on ``split`` is forbidden right now.

    Only D_gate_reserve is time-locked: before the first gate failure it may not be scored,
    fitted on or have its y* read (paper Sec. 4.3). The label-purpose matrix is :func:`read_labels`'s.
    """
    if split is Split.D_GATE_RESERVE and not reserve_unlocked(gate_attempts_path):
        raise LabelAccessError(
            f"D_gate_reserve may not be used for '{operation.value}' before the first gate "
            "attempt has been recorded as failed (II.6.1)"
        )
    if split in (Split.D_EXP1, Split.UNLABELED_POOL) and operation is Operation.ANNOTATE:
        raise LabelAccessError(f"{split.value} is never human-annotated (II.2.4)")


def read_labels(
    split: Split,
    purpose: LabelPurpose,
    *,
    labels_dir: Path | None = None,
    split_dir: Path | None = None,
    gate_attempts_path: Path | None = None,
) -> list[YStar]:
    """y* of every state in ``split``, for ``purpose``, or :class:`LabelAccessError`.

    The purpose gate and the reserve lock are checked BEFORE any file is opened.
    """
    allowed = ALLOWED.get(split, frozenset())
    if purpose not in allowed:
        raise LabelAccessError(
            f"y* of {split.value} may not be read for '{purpose.value}' "
            f"(allowed: {sorted(p.value for p in allowed) or 'none'})"
        )
    if split is Split.D_GATE_RESERVE:
        check_split_usable(split, Operation.FIT, gate_attempts_path=gate_attempts_path)
    x0_ids = load_split(split, split_dir or paths.DATA_SPLITS)
    if not x0_ids:
        raise LabelDataError(f"no x0 in {split.value}: has step05 split the states yet?")
    by_x0: dict[str, StateJudgment] = {}
    for row in load_judgment_rows(y_star_path(labels_dir)).values():
        by_x0[row.x0_id] = row
    out: list[YStar] = []
    missing: list[str] = []
    for x0 in x0_ids:
        row = by_x0.get(x0)
        if (
            row is None
            or row.label is None
            or row.label is MathLabel.UNRESOLVED
            or row.record is None
        ):
            missing.append(x0)
            continue
        if row.record.label != row.label:
            raise LabelDataError(f"E record label disagrees with y* for {x0}")
        out.append(
            YStar(
                state_id=row.state_id,
                x0_id=row.x0_id,
                split=split,
                label=row.label,
                record=row.record,
            )
        )
    if missing:
        raise LabelDataError(
            f"{len(missing)} x0 of {split.value} have no y* (e.g. {missing[:3]}); the split "
            "and the judgment file disagree"
        )
    return out


def split_report(
    splits: Mapping[Split, list[str]],
    *,
    labels_dir: Path | None = None,
    split_dir: Path | None = None,
    domain: str = "detox",
) -> dict[str, dict[str, Any]]:
    """Per split: its size; for D_fit also the E-UNSAFE count and share (2026-09-23).

    Only D_fit's y* is opened (purpose SPLIT_REPORT). D_gate, D_gate_reserve, D_Exp1 and the
    unlabeled pool are reported from their split files alone; D_Exp1 is E-UNSAFE by
    construction, so its size is the whole story.
    """
    if domain not in {"detox", "math"}:
        raise ValueError("split report domain must be detox or math")
    positive = MathLabel.INCORRECT if domain == "math" else SafetyLabel.UNSAFE
    allowed_labels = (
        {MathLabel.CORRECT, MathLabel.INCORRECT}
        if domain == "math"
        else {SafetyLabel.SAFE, SafetyLabel.UNSAFE}
    )
    out: dict[str, dict[str, Any]] = {}
    for sp, ids in splits.items():
        if sp not in SPLIT_REPORT_COUNTS:
            out[sp.value] = {"n": len(ids)}
            continue
        ys = read_labels(
            sp, LabelPurpose.SPLIT_REPORT, labels_dir=labels_dir, split_dir=split_dir
        )
        if any(y.label not in allowed_labels for y in ys):
            raise LabelDataError(f"D_fit labels do not match the {domain} domain")
        n_unsafe = sum(1 for y in ys if y.label is positive)
        out[sp.value] = {
            "n": len(ys),
            "n_incorrect" if domain == "math" else "n_unsafe": n_unsafe,
            "incorrect_rate" if domain == "math" else "unsafe_rate": (
                n_unsafe / len(ys) if ys else None
            ),
        }
    return out
