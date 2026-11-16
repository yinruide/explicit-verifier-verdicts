"""Verifiers V1, V2 (II.4), against ``src/rafr/contracts/score_spec.md``.

============= =========================================================================
``stub.py``   deterministic hash scorer for the dry run and the tests; never formal
``candidates/`` the real candidate pool (encoder classifiers + guard LMs), lazily loaded
``batch_score.py`` ``to_score.jsonl`` -> the scores table, resumable, for the rented GPU
``cards/``    ``verifier_card.md`` generated from the download manifest
============= =========================================================================

``perturb.py`` and ``diagnostics.py`` (II.4.2) are not written yet.

The scores table format (App. A.2) is owned here: ``data/scores/<verifier_id>.jsonl`` with
one :class:`rafr.contracts.schemas.Score` row per line. D_gate_reserve, if it is ever scored
(after a failed first gate, II.6.1), goes to a table of its own,
``data/scores/reserve/<verifier_id>.jsonl``, and never into the existing one (decisions.md
D-2026-09-27: the existing tables are CRLF files with registered raw-byte hashes, and must not
change).
"""

from __future__ import annotations

from pathlib import Path

from rafr.contracts.schemas import Score
from rafr.core.textio import crlf_rewrite_notice
from rafr.logging.writer import read_jsonl

RESERVE_DIR = "reserve"


def scores_table_path(verifier_id: str, scores_dir: Path) -> Path:
    return Path(scores_dir) / f"{verifier_id}.jsonl"


def reserve_scores_table_path(verifier_id: str, scores_dir: Path) -> Path:
    """D_gate_reserve's own table (D-2026-09-27): never appended to the main table."""
    return Path(scores_dir) / RESERVE_DIR / f"{verifier_id}.jsonl"


def load_scores_table(path: Path) -> dict[str, float]:
    """state_id -> score (last row wins if a state was re-scored)."""
    out: dict[str, float] = {}
    for r in read_jsonl(path):
        row = Score.model_validate(r)
        out[row.state_id] = row.score
    return out


def write_scores_table(rows: list[Score], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with crlf_rewrite_notice(path), path.open("a", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(r.model_dump_json() + "\n")
    return path
