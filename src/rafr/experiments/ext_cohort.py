"""The E-correct cohort (post-pilot extension, 2026-10-31).

D_gate states that E judged correct (detox: SAFE). If there are more than ``n`` (300), ``n`` are
drawn without replacement, seeded by the component ``ext_ecorrect_cohort``; if there are ``n`` or
fewer, all of them, never resampled. The candidates' y* is read by the caller through the label
gate (purpose ECORRECT_COHORT); this module sees state ids only. The file is written once and
then read back with its sha256.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rafr.calibration.apply import file_sha256, ids_sha256
from rafr.core.seeding import py_rng


class CohortError(RuntimeError):
    pass


def draw_cohort(candidate_ids: Sequence[str], n: int, seed: int) -> list[str]:
    """``n`` of the candidates without replacement (all of them if there are ``n`` or fewer),
    returned sorted. Deterministic in (the candidate set, ``n``, ``seed``)."""
    ids = sorted(candidate_ids)
    if len(ids) != len(set(ids)):
        raise CohortError("duplicate candidate state ids")
    if n < 1:
        raise CohortError(f"cohort size must be >= 1, got {n}")
    if len(ids) <= n:
        return ids
    return sorted(py_rng(seed, "ecorrect_cohort").sample(ids, n))


def draw_smoke(candidate_ids: Sequence[str], n: int, seed: int, stream: str) -> list[str]:
    """The smoke (2026-10-31): ``n`` of the candidates -- states outside every
    formal set, chosen by the caller -- without replacement, returned sorted. Deterministic in
    (the candidate set, ``n``, ``seed``, ``stream``)."""
    ids = sorted(candidate_ids)
    if len(ids) != len(set(ids)):
        raise CohortError("duplicate candidate state ids")
    if not 0 < n <= len(ids):
        raise CohortError(f"a smoke takes 1..{len(ids)} states, got {n}")
    return sorted(py_rng(seed, f"smoke_{stream}").sample(ids, n))


def write_cohort(path: Path, ids: Sequence[str], meta: Mapping[str, Any]) -> str:
    """Write the cohort (one JSON row per state; the first row is the header) and return its
    sha256. Refuses to overwrite: the cohort is drawn once."""
    path = Path(path)
    if path.exists():
        raise CohortError(f"{path} exists; the cohort is drawn once")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as fh:
        header = {"header": True, **dict(meta), "n": len(ids), "state_ids_sha256": ids_sha256(ids)}
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for sid in ids:
            fh.write(json.dumps({"state_id": sid}) + "\n")
    return file_sha256(path)


def read_cohort(path: Path, expected_sha256: str) -> tuple[dict[str, Any], list[str]]:
    """(header, state ids) of a cohort file whose sha256 is ``expected_sha256``."""
    path = Path(path)
    got = file_sha256(path)
    if got != expected_sha256:
        raise CohortError(f"{path} has sha256 {got}, expected {expected_sha256}")
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows or not rows[0].get("header"):
        raise CohortError(f"{path} has no header row")
    header, ids = rows[0], [r["state_id"] for r in rows[1:]]
    if header.get("state_ids_sha256") != ids_sha256(ids) or header.get("n") != len(ids):
        raise CohortError(f"{path}: the header does not match its rows")
    return header, ids


__all__ = ["CohortError", "draw_cohort", "draw_smoke", "read_cohort", "write_cohort"]
