"""Log record kinds and the file layout of a run directory.

The row schemas themselves live in :mod:`rafr.contracts.schemas` (``LogRecord``,
``GenerationAttempt``, ``EvalRecord``, ``Score``); this module only names the streams.

Run directory layout (``runs/<run_id>/``)::

    snapshot.json          git HEAD, dirty flag, config snapshot, argv, timestamps
    configs/*.yaml         verbatim copy of configs/ at launch
    decisions.jsonl        LogRecord rows (one per round x arm x state)
    generation.jsonl       GenerationAttempt rows (G_fixed; every attempt, valid or not)
    evaluations.jsonl      EvalRecord rows (E); Exp 2: every round text, judged after the loop
    round_judgments.jsonl  Exp 2 only: (x0, arm, t) -> E label of x_t, incl. the terminal text
    scores.jsonl           Score rows (raw verifier scores, pre-channel)
    channel.jsonl          (state_id, lambda, s, s~) rows (Exp 1 only)
    permutation.jsonl      permuted_B block assignment (Exp 1 only)
    ledger.jsonl           token / cost rows
    rounds/round_<t>/      Exp 2, one directory per round: trajectories (the resume point),
                           that round's decisions / scores / ledger rows, COMPLETE.json
                           (sha256 of each file; written last, so no half round exists)
"""

from __future__ import annotations

from enum import StrEnum

from rafr.contracts.schemas import EvalRecord, GenerationAttempt, LogRecord, Score


class RecordKind(StrEnum):
    DECISION = "decisions"
    GENERATION = "generation"
    EVALUATION = "evaluations"
    SCORE = "scores"
    CHANNEL = "channel"
    PERMUTATION = "permutation"
    LEDGER = "ledger"


ROW_MODELS = {
    RecordKind.DECISION: LogRecord,
    RecordKind.GENERATION: GenerationAttempt,
    RecordKind.EVALUATION: EvalRecord,
    RecordKind.SCORE: Score,
}


def stream_filename(kind: RecordKind) -> str:
    return f"{kind.value}.jsonl"
