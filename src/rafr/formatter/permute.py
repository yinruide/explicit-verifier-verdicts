"""Permuted-B (paper Sec. 3; 2026-09-16, revised 2026-09-25).

The whole six-field block of a state is replaced by the block of another state. The mapping is
a derangement (no state keeps its own block), keyed by the seed ONLY, and written to disk so
that the run is reproducible: every state has the same donor at all three lambda levels, and at
each level it shows the donor's block of THAT level (the donor's fields at the current lambda).
Keying by (lambda, seed), as before, added a fresh donor draw to every lambda contrast of
permuted-B, noise that A1 and B do not have -- in a cell whose evidence did not change they
share one controller answer, and now permuted-B does too (2026-09-25).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from rafr.core.seeding import py_rng


class PermutationError(ValueError):
    pass


def derangement(state_ids: Sequence[str], seed: int) -> dict[str, str]:
    """Return ``{state_id: donor_state_id}`` with no fixed points. Needs >= 2 states.

    Deterministic in ``(state_ids order, seed)``; uses rejection sampling with a seeded RNG
    (expected ~e tries). No lambda: one table serves every level.
    """
    ids = list(state_ids)
    if len(ids) != len(set(ids)):
        raise PermutationError("duplicate state ids")
    if len(ids) < 2:
        raise PermutationError("a derangement needs at least two states")
    rng = py_rng(seed, "permuted_B")
    perm = ids[:]
    while True:
        rng.shuffle(perm)
        if all(a != b for a, b in zip(ids, perm, strict=True)):
            break
    return dict(zip(ids, perm, strict=True))


def save_permutation(mapping: dict[str, str], seed: int, path: Path) -> None:
    """Write the one table of the run (replacing any file there)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for sid, donor in mapping.items():
            fh.write(json.dumps({"seed": seed, "state_id": sid, "donor": donor}) + "\n")
    tmp.replace(path)


def load_permutation(path: Path) -> tuple[int, dict[str, str]]:
    """``(seed, {state_id: donor})`` of a table :func:`save_permutation` wrote.

    A table written per lambda (the pre-2026-09-25 format, with a ``lam`` key) is refused rather
    than read as if it were one table: its three levels have three different donors.
    """
    out: dict[str, str] = {}
    seeds: set[int] = set()
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if "lam" in r:
                raise PermutationError(
                    f"{path} is a per-lambda table (keyed by (lambda, seed)); since D-2026-09-25b "
                    "the table is keyed by the seed only: start the run in a new directory"
                )
            seeds.add(int(r["seed"]))
            if r["state_id"] in out:
                raise PermutationError(f"{path}: state {r['state_id']} appears twice")
            out[r["state_id"]] = r["donor"]
    if len(seeds) != 1:
        raise PermutationError(f"{path}: expected one seed, found {sorted(seeds)}")
    if any(k == v for k, v in out.items()):
        raise PermutationError(f"{path}: a state is its own donor")
    return seeds.pop(), out
