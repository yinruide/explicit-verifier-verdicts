"""Shared I/O for the validity steps (07 fit tau, 08 check the channel, 09 gate).

Kept in one private helper so the three scripts read states, labels and scores the same way.
Label access always goes through ``rafr.data.labels`` with an explicit purpose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rafr.contracts.schemas import TBD, SafetyLabel, Split, State
from rafr.core import paths
from rafr.core.config import get
from rafr.data import labels as L
from rafr.logging.writer import read_jsonl
from rafr.verifiers import load_scores_table, reserve_scores_table_path, scores_table_path
from rafr.verifiers.stub import is_stub


class ValidityIOError(RuntimeError):
    pass


def verifier_ids(models_cfg: dict[str, Any]) -> tuple[str, str]:
    v1 = get(models_cfg, "verifiers.v1.id")
    v2 = get(models_cfg, "verifiers.v2.id")
    for v in (v1, v2):
        if v in (None, TBD):
            raise ValidityIOError(
                "models.verifiers.v1.id / v2.id are TBD (the authors, after step 2)"
            )
        if is_stub(str(v)):
            raise ValidityIOError(f"{v} is a stub verifier; formal steps refuse stubs (II.4.4)")
    return str(v1), str(v2)


def load_states(split: Split) -> list[State]:
    p = paths.DATA_POOLS / f"states_{split.value}.jsonl"
    if not p.exists():
        raise ValidityIOError(f"no states for {split.value} at {p}; run step05 split")
    return [State.model_validate(r) for r in read_jsonl(p)]


def load_scores_for(states: list[State], verifier_id: str) -> dict[str, float]:
    table = load_scores_table(scores_table_path(verifier_id, paths.DATA_SCORES))
    reserve = reserve_scores_table_path(verifier_id, paths.DATA_SCORES)
    if reserve.is_file() and any(s.state_id not in table for s in states):
        # D_gate_reserve's scores live in their own table (decisions.md D-2026-09-27)
        table = {**table, **load_scores_table(reserve)}
    missing = [s.state_id for s in states if s.state_id not in table]
    if missing:
        raise ValidityIOError(
            f"{len(missing)} state(s) have no {verifier_id} score (e.g. {missing[:3]}); run step06"
        )
    return {s.state_id: table[s.state_id] for s in states}


def load_labels_for(states: list[State], split: Split, purpose: L.LabelPurpose) -> dict[str, int]:
    """state_id -> 1 (UNSAFE) / 0 (SAFE), through the purpose gate."""
    recs = L.read_labels(split, purpose, gate_attempts_path=paths.FROZEN_GATE_ATTEMPTS)
    by = {r.state_id: r.label for r in recs}
    missing = [s.state_id for s in states if s.state_id not in by]
    if missing:
        raise ValidityIOError(
            f"{len(missing)} state(s) in {split.value} have no label (e.g. {missing[:3]})"
        )
    return {s.state_id: int(by[s.state_id] is SafetyLabel.UNSAFE) for s in states}


def aligned(states: list[State], *maps: dict[str, Any]) -> list[list[Any]]:
    return [[m[s.state_id] for s in states] for m in maps]


def diagnostics_dir(channel_cfg: dict[str, Any]) -> Path:
    return paths.resolve(get(channel_cfg, "diagnostics.out_dir", "data/diagnostics"))
