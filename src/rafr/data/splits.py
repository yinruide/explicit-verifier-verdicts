"""Group-level splits by x0 (II.2.4, v0.9): D_fit, D_gate, D_gate_reserve, D_Exp1, pool.

Every split is a set of ``x0_id``; the sets are pairwise disjoint by construction and
:func:`assert_disjoint` re-checks them whenever they are loaded. Exp 2 reuses the D_Exp1 x0
(II.8.3), so it has no split of its own.

The v0.9 procedure (:data:`PROCEDURE`), run by step05 after E has judged every state:

1. D_fit, D_gate, D_gate_reserve are cut from one seeded shuffle of ALL x0, in that order. No
   label is looked at: the three are random samples of the whole state population, so tau has
   both classes to fit on (C.16);
2. D_Exp1 is drawn from the x0 that remain, keeping only those whose state E judged UNSAFE
   (C.16): the first ``d_exp1`` of them in the same shuffled order, which is a uniform random
   sample without replacement of the remaining E-UNSAFE states;
3. every remaining x0 goes to the unlabeled pool.

Fewer than ``d_exp1`` E-UNSAFE x0 left after step 1 raises :class:`InsufficientUnsafeError`;
no other split is ever shrunk to make room. The label is consulted only through the
``is_unsafe`` callable and only for x0 left after step 1 -- the tests hold it to that.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from rafr.contracts.schemas import Split
from rafr.core.config import get, master_seed
from rafr.core.seeding import component_seed, py_rng
from rafr.core.textio import crlf_rewrite_notice

SPLIT_ORDER: tuple[Split, ...] = (
    Split.D_FIT,
    Split.D_GATE,
    Split.D_GATE_RESERVE,
    Split.D_EXP1,
    Split.UNLABELED_POOL,
)

RANDOM_SPLITS: tuple[Split, ...] = (Split.D_FIT, Split.D_GATE, Split.D_GATE_RESERVE)
"""Drawn first, from every state, blind to the labels (II.2.4)."""

PROCEDURE = "random_then_e_unsafe_v09"
"""Name of the procedure implemented by :func:`make_splits`; configs/data.yaml names it too,
and step05 refuses to split when the two disagree."""

SEED_COMPONENT = "splits_v09"
"""The v0.9 split seed's component name (a new one: round 1 used ``splits``)."""

REST = "rest"


class SplitError(ValueError):
    pass


class InsufficientUnsafeError(SplitError):
    """Too few E-UNSAFE states remain for D_Exp1 after the three random splits (II.2.4)."""

    def __init__(self, have: int, need: int, n_remaining: int):
        self.have, self.need, self.n_remaining = have, need, n_remaining
        super().__init__(
            f"only {have} of the {n_remaining} states left after D_fit / D_gate / "
            f"D_gate_reserve are E-UNSAFE; D_Exp1 needs {need}. Stopping: no other split is "
            "shrunk to make room (II.2.4)."
        )


def _fixed_size(sizes: Mapping[str, Any], sp: Split) -> int:
    v = sizes.get(sp.value)
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise SplitError(f"splits.sizes.{sp.value} must be a non-negative integer, got {v!r}")
    return v


def make_splits(
    x0_ids: Iterable[str],
    is_unsafe: Callable[[str], bool],
    sizes: Mapping[str, Any],
    seed: int,
) -> dict[Split, list[str]]:
    """The v0.9 split (:data:`PROCEDURE`); see the module docstring.

    ``x0_ids`` are the x0 whose state has a y* (E failures already removed); their order does
    not matter, they are sorted before the seeded shuffle. ``is_unsafe(x0_id)`` is E's
    judgment of that x0's state; it is called only for x0 left after the three random splits.
    ``sizes`` must give integers for d_fit / d_gate / d_gate_reserve / d_exp1 and ``"rest"``
    for the unlabeled pool.
    """
    raw = list(x0_ids)
    ids = sorted(set(raw))
    if len(ids) != len(raw):
        raise SplitError("duplicate x0_id in input")
    if sizes.get(Split.UNLABELED_POOL.value) != REST:
        raise SplitError("splits.sizes.unlabeled_pool must be 'rest' (II.2.4)")
    n_random = {sp: _fixed_size(sizes, sp) for sp in RANDOM_SPLITS}
    n_exp1 = _fixed_size(sizes, Split.D_EXP1)
    if sum(n_random.values()) > len(ids):
        raise SplitError(
            f"the three random splits need {sum(n_random.values())} x0, only {len(ids)} have a y*"
        )
    py_rng(seed, "shuffle").shuffle(ids)

    out: dict[Split, list[str]] = {sp: [] for sp in SPLIT_ORDER}
    cursor = 0
    for sp in RANDOM_SPLITS:
        out[sp] = ids[cursor : cursor + n_random[sp]]
        cursor += n_random[sp]
    remaining = ids[cursor:]
    unsafe = [i for i in remaining if is_unsafe(i)]
    if len(unsafe) < n_exp1:
        raise InsufficientUnsafeError(len(unsafe), n_exp1, len(remaining))
    out[Split.D_EXP1] = unsafe[:n_exp1]
    taken = set(out[Split.D_EXP1])
    out[Split.UNLABELED_POOL] = [i for i in remaining if i not in taken]
    assert_disjoint(out)
    return out


def split_plan(configs: Mapping[str, Any]) -> dict[str, Any]:
    """What the split will be, fixed before step 5 judges a single state (II.2.4).

    Written into ``data/manifest.json`` (``splits.plan``) the first time a v0.9 step runs;
    step05 refuses to split if the configured plan no longer equals the recorded one.
    """
    data = configs.get("data") or {}
    procedure = get(data, "splits.procedure")
    component = get(data, "splits.seed_component")
    if procedure != PROCEDURE:
        raise SplitError(
            f"data.splits.procedure is {procedure!r}; this code implements {PROCEDURE!r}"
        )
    if component != SEED_COMPONENT:
        raise SplitError(
            f"data.splits.seed_component is {component!r}; the v0.9 split uses {SEED_COMPONENT!r}"
        )
    master = master_seed(configs)
    return {
        "procedure": PROCEDURE,
        "seed_component": SEED_COMPONENT,
        "master_seed": master,
        "seed": component_seed(master, SEED_COMPONENT),
        "sizes": dict(get(data, "splits.sizes") or {}),
        "order": [sp.value for sp in SPLIT_ORDER],
        "description": (
            "1) d_fit, d_gate, d_gate_reserve cut in that order from one seeded shuffle of all "
            "x0 with a y*, labels unread; 2) d_exp1 = the first d_exp1 E-UNSAFE x0 among the "
            "rest, in the same shuffled order; 3) unlabeled_pool = everything left. Too few "
            "E-UNSAFE -> stop, no split shrunk (II.2.4, C.16)"
        ),
    }


def assert_disjoint(splits: Mapping[Split, Iterable[str]]) -> None:
    seen: dict[str, Split] = {}
    for sp, ids in splits.items():
        for i in ids:
            if i in seen:
                raise SplitError(f"x0 {i} appears in both {seen[i].value} and {sp.value}")
            seen[i] = sp


def save_splits(splits: Mapping[Split, Iterable[str]], out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for sp, ids in splits.items():
        p = out_dir / f"{sp.value}.jsonl"
        with crlf_rewrite_notice(p), p.open("w", encoding="utf-8", newline="\n") as fh:
            for i in ids:
                fh.write(json.dumps({"x0_id": i}) + "\n")
        written[sp.value] = p
    return written


def load_split(split: Split, split_dir: Path) -> list[str]:
    p = Path(split_dir) / f"{split.value}.jsonl"
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as fh:
        return [json.loads(line)["x0_id"] for line in fh if line.strip()]


def load_all_splits(split_dir: Path) -> dict[Split, list[str]]:
    out = {sp: load_split(sp, split_dir) for sp in SPLIT_ORDER}
    assert_disjoint(out)
    return out


def split_of(x0_id: str, splits: Mapping[Split, Iterable[str]]) -> Split | None:
    for sp, ids in splits.items():
        if x0_id in set(ids):
            return sp
    return None
