"""Evidence sensitivity (III.1.2, v1.0): the accept rate by observed judgment pair.

Descriptive. For every arm and every lambda level (Exp 1) or round (Exp 2): the accept rate in
the four cells of the OBSERVED judgment pair

    (y1_obs, y2_obs) in {(0,0), (0,1), (1,0), (1,1)},    y_i_obs = 1[s_i^obs >= tau_i]

with each cell's size. ``y_i_obs`` is read off the CURRENT state's own post-channel scores, the
ones the formatter showed in the A1 part of its prompt (``s_tilde``, i.e. after C_lambda in
Exp 1), so the pair is the one the controller saw for this state. Three things it is NOT: not the
pre-channel raw scores; not y* (E's label plays no part in the cell); and for permuted-B not the
donor's block -- the donor's six fields are what that arm was shown as derived fields, but the
scores and thresholds in its prompt are the state's own, and the cell follows those. All three
Exp 1 arms see the same evidence for the same (state, lambda), so this stratification is
arm-independent and is not post-treatment conditioning (:func:`exp1` counts the cells where it
would not hold, which must be 0). In Exp 2 the evidence at t >= 2 depends on the trajectory, so
that table is descriptive only.

Use (III.1.2 / III.2.3): when Delta ~ 0, tell "the controller ignored the checkers' evidence"
(the accept rate hardly moves across the four cells) from "it read the evidence but the
representation did not matter" (it moves, but the same way in every arm). The step 6 smoke run
accepted at round 2 with both checkers flagging; this table makes that quantifiable. The
controller prompt is never adjusted because of it: tuning the controller for an effect would be
tuning the subject.

"Accept" follows the main accounting of II.8.2 (parse_fail = not accepted); the accounting is a
parameter so the two sensitivities can be re-run through the same function.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis._common import accepted, mean_or_nan
from analysis.error_boundary import verifier_judgments

PAIRS: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
"""The four cells of (y1_obs, y2_obs), in the order III.1.2 lists them."""


def pair_key(pair: tuple[int, int]) -> str:
    return f"({pair[0]},{pair[1]})"


def observed_pair(row: dict[str, Any]) -> tuple[int, int]:
    """(y1_obs, y2_obs) as SHOWN to the controller (from ``s_tilde``), 1 = flags."""
    return verifier_judgments(row)


def _cell(rows: list[dict[str, Any]], accounting: str) -> dict[str, Any]:
    acc: list[float] = []
    n_excluded = 0
    for r in rows:
        a = accepted(r, accounting)
        if a is None:
            n_excluded += 1
            continue
        acc.append(float(a))
    return {
        "n": len(acc),
        "n_excluded_by_accounting": n_excluded,
        "n_accepted": int(sum(acc)),
        "accept_rate": mean_or_nan(acc),
    }


def _table(rows: list[dict[str, Any]], group_key: str, accounting: str) -> dict[str, Any]:
    cells: dict[Any, dict[str, dict[tuple[int, int], list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: {p: [] for p in PAIRS})
    )
    for r in rows:
        cells[r[group_key]][r["arm"]][observed_pair(r)].append(r)
    out: dict[str, Any] = {
        "grouped_by": group_key,
        "accounting": accounting,
        "definition": (
            "accept rate by arm and by the observed judgment pair (y1_obs, y2_obs), "
            "y_i_obs = 1[s_i^obs >= tau_i] on the post-channel scores the formatter showed; "
            "n per cell (III.1.2, v1.0). Descriptive, never a stratifier of the primary analysis."
        ),
        "pairs": [pair_key(p) for p in PAIRS],
        "cells": {},
    }
    for g in sorted(cells, key=str):
        out["cells"][str(g)] = {
            arm: {pair_key(p): _cell(by_pair[p], accounting) for p in PAIRS}
            for arm, by_pair in sorted(cells[g].items())
        }
    return out


def _pairs_differing_across_arms(rows: list[dict[str, Any]]) -> int:
    """How many (state, lambda) cells show a different observed pair to different arms.

    Must be 0 in Exp 1: the channel is applied once per (state, lambda) and every arm is shown
    the same (s1~, s2~) (II.5.1), which is what makes the table arm-independent.
    """
    seen: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for r in rows:
        seen[(r["state_id"], r["lam"])].add(observed_pair(r))
    return sum(1 for v in seen.values() if len(v) > 1)


def exp1(rows: list[dict[str, Any]], *, accounting: str = "main") -> dict[str, Any]:
    """Per lambda level, per arm, the four-cell table (III.1.2, v1.0)."""
    rep = _table(rows, "lam", accounting)
    rep["cells_differing_across_arms"] = _pairs_differing_across_arms(rows)
    rep["note"] = (
        "all three arms see the same evidence for the same (state, lambda), so the four cells "
        "are arm-independent (cells_differing_across_arms must be 0); read across cells within "
        "an arm for whether the controller heeds the checkers, across arms for whether the "
        "representation changes that (III.2.3)"
    )
    return rep


def exp2(rows: list[dict[str, Any]], *, accounting: str = "main") -> dict[str, Any]:
    """Per round t, per arm, the same table (III.1.2, v1.0). Exp 2 has no channel, so the
    observed pair is the verifiers' own judgment of x_t; from t >= 2 it depends on the
    trajectory and the table is descriptive only."""
    rep = _table(rows, "t", accounting)
    rep["note"] = (
        "descriptive: at t >= 2 the evidence depends on the arm's own trajectory (post-treatment), "
        "so no contrast is drawn from this table (III.1.2)"
    )
    return rep


def render(rep: dict[str, Any]) -> str:
    lines = [f"accept rate by arm x {rep['grouped_by']} x observed (y1, y2) (III.1.2)"]
    for g, arms in rep["cells"].items():
        for arm, by_pair in arms.items():
            parts = []
            for key in rep["pairs"]:
                c = by_pair[key]
                rate = "  n/a" if c["n"] == 0 else f"{c['accept_rate']:.0%}"
                parts.append(f"{key} {rate} (n={c['n']})")
            lines.append(f"  {arm}|{g:>5}: " + "  ".join(parts))
    return "\n".join(lines)
