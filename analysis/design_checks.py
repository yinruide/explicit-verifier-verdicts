"""Checks that the logs are what the analysis plan says they are (paper Sec. 3, 4.2, 4.4,
4.6).

Nothing here estimates anything. Each function returns one row per check with the value found,
the value expected and ``ok``; a failed check is reported, never repaired. The primary analyses
do not depend on these rows passing: the Exp 1 denominator check that does stop an analysis
lives in ``exp1_did.check_d_exp1_all_unsafe``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from rafr.contracts.schemas import K_MAX_ROUNDS

EXP1_ARMS = ("A1", "B", "permuted_B")
EXP1_LAMBDAS = ("low", "mid", "high")


def _check(name: str, value: Any, expected: Any, ok: bool, ref: str) -> dict[str, Any]:
    return {"check": name, "value": value, "expected": expected, "ok": bool(ok), "ref": ref}


def exp1(rows: Sequence[Mapping[str, Any]], *, n_expected: int = 300) -> list[dict[str, Any]]:
    """Section 3 (paper Sec. 4.2) and section 6 on the Exp 1 decision rows."""
    states = {r["state_id"] for r in rows}
    x0_per_state = defaultdict(set)
    states_per_x0 = defaultdict(set)
    cells = defaultdict(set)
    shown: dict[tuple[str, str], set[tuple[Any, Any]]] = defaultdict(set)
    donors: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        x0_per_state[r["state_id"]].add(r["x0_id"])
        states_per_x0[r["x0_id"]].add(r["state_id"])
        cells[r["state_id"]].add((r["arm"], r["lam"]))
        shown[(r["state_id"], r["lam"])].add((r["s1_tilde"], r["s2_tilde"]))
        if r["arm"] == "permuted_B":
            donors[r["state_id"]].add(r.get("permuted_from_state_id") or "")
    full = {(a, lam) for a in EXP1_ARMS for lam in EXP1_LAMBDAS}
    not_unsafe = sorted({r["state_id"] for r in rows if r.get("e_label") != "UNSAFE"})
    self_donor = sum(1 for s, d in donors.items() if s in d)
    return [
        _check("n_states", len(states), n_expected, len(states) == n_expected, "II.2.4"),
        _check("states_not_e_unsafe", len(not_unsafe), 0, not not_unsafe, "II.2.4, C.16"),
        _check(
            "x0_with_more_than_one_state",
            sum(1 for v in states_per_x0.values() if len(v) > 1),
            0,
            all(len(v) == 1 for v in states_per_x0.values()),
            "section 3: analysis unit = x0, one state per x0",
        ),
        _check(
            "states_missing_a_cell_of_3x3",
            sum(1 for c in cells.values() if c != full),
            0,
            all(c == full for c in cells.values()),
            "II.1.3: 3 x 3 within-state",
        ),
        _check(
            "rows_per_state_cell",
            dict(Counter(Counter((r["state_id"], r["arm"], r["lam"]) for r in rows).values())),
            {1: len(states) * len(full)},
            all(
                v == 1 for v in Counter((r["state_id"], r["arm"], r["lam"]) for r in rows).values()
            ),
            "one decision per (state, arm, lambda)",
        ),
        _check(
            "state_lambda_cells_with_different_evidence_across_arms",
            sum(1 for v in shown.values() if len(v) > 1),
            0,
            all(len(v) == 1 for v in shown.values()),
            "II.5.1: the channel is applied once per (state, lambda)",
        ),
        _check(
            "permuted_b_donor_varies_across_lambda",
            sum(1 for d in donors.values() if len(d) > 1),
            0,
            all(len(d) == 1 for d in donors.values()),
            "II.7.3, C.22: one donor per state, keyed by the seed only",
        ),
        _check("permuted_b_own_block", self_donor, 0, self_donor == 0, "II.7.3: derangement"),
    ]


def exp2(
    finals: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    assignment: Sequence[Mapping[str, Any]],
    *,
    exp1_rows: Sequence[Mapping[str, Any]] | None = None,
    round_judgments: Sequence[Mapping[str, Any]] | None = None,
    k: int = K_MAX_ROUNDS,
    n_per_arm: int = 150,
) -> list[dict[str, Any]]:
    """Section 7 (paper Sec. 4.6, 4.4) on the Exp 2 logs."""
    arms = Counter(f["arm"] for f in finals)
    x0_arms: dict[str, set[str]] = defaultdict(set)
    for f in finals:
        x0_arms[f["x0_id"]].add(f["arm"])
    seeds = {a.get("seed") for a in assignment}
    by_x0: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_x0[r["x0_id"]].append(r)
    for v in by_x0.values():
        v.sort(key=lambda r: int(r["t"]))
    too_many = sum(1 for v in by_x0.values() if len(v) > k)
    bad_t = sum(1 for r in rows if not 1 <= int(r["t"]) <= k)
    after_accept = 0
    pf_moved = 0
    rw_not_moved = 0
    for v in by_x0.values():
        for a, b in zip(v, v[1:], strict=False):
            if not a["parse_fail"] and a["action"] == "accept":
                after_accept += 1
            if a["parse_fail"] and b["x_t"] != a["x_t"]:
                pf_moved += 1  # paper Sec. 4.6: x_{t+1} = x_t after a parse_fail
            if not a["parse_fail"] and a["action"] == "rewrite" and b["x_t"] != a["revised_text"]:
                rw_not_moved += 1
    terminal_wrong = 0
    for f in finals:
        v = by_x0.get(f["x0_id"], [])
        if f["accepted"]:
            acc = [r for r in v if int(r["t"]) == int(f["accepted_at"])]
            ok = bool(acc) and acc[0]["x_t"] == f["x_T"]
        else:
            last = v[-1] if v else None
            want = None
            if last is not None:
                want = last["x_t"] if last["parse_fail"] else last["revised_text"]
            ok = last is not None and len(v) == k and f["x_T"] == want
        terminal_wrong += int(not ok)
    out = [
        _check(
            "arm_sizes",
            dict(sorted(arms.items())),
            {"A1": n_per_arm, "B": n_per_arm},
            dict(arms) == {"A1": n_per_arm, "B": n_per_arm},
            "II.8.3",
        ),
        _check(
            "x0_in_more_than_one_arm",
            sum(1 for v in x0_arms.values() if len(v) > 1),
            0,
            all(len(v) == 1 for v in x0_arms.values()),
            "II.8.3: each x0 enters one arm",
        ),
        _check(
            "assignment_seed_recorded",
            sorted(seeds, key=str),
            "one seed on every assignment row",
            len(seeds) == 1 and None not in seeds and len(assignment) == len(finals),
            "II.8.3: balanced permutation, seed on disk",
        ),
        _check("decisions_beyond_k", too_many, 0, too_many == 0, f"II.8.3: K = {k}"),
        _check("rows_with_t_outside_1_k", bad_t, 0, bad_t == 0, f"II.8.3: t = 1..{k}"),
        _check("rows_after_an_accept", after_accept, 0, after_accept == 0, "II.8.3"),
        _check("parse_fail_followed_by_new_text", pf_moved, 0, pf_moved == 0, "II.8.2"),
        _check("rewrite_not_carried_to_next_round", rw_not_moved, 0, rw_not_moved == 0, "II.8.3"),
        _check(
            "x_T_not_by_the_terminal_rule",
            terminal_wrong,
            0,
            terminal_wrong == 0,
            "II.8.3: accepted x_t, else the 4th decision's output",
        ),
        _check(
            "rows_with_lambda_other_than_none",
            sum(1 for r in rows if r["lam"] != "none"),
            0,
            all(r["lam"] == "none" for r in rows),
            "II.1.3: no channel in Exp 2",
        ),
    ]
    if exp1_rows is not None:
        x1 = {r["x0_id"]: r["x_t"] for r in exp1_rows}
        first = {r["x0_id"]: r["x_t"] for r in rows if int(r["t"]) == 1}
        diff = sum(1 for x, text in first.items() if x1.get(x) != text)
        out.append(
            _check(
                "x1_not_the_d_exp1_state",
                diff,
                0,
                diff == 0 and set(first) == set(x1),
                "II.8.3: x1 = the x0's D_Exp1 G_fixed state",
            )
        )
    if round_judgments is not None:
        t1 = [j for j in round_judgments if int(j["t"]) == 1]
        not_reused = sum(1 for j in t1 if j.get("source") != "reused")
        out.append(
            _check(
                "t1_judgments_not_reused",
                not_reused,
                0,
                not_reused == 0,
                "II.8.3: t = 1 reuses step 5's y*",
            )
        )
        t1_safe = sum(1 for j in t1 if j.get("e_label") != "UNSAFE")
        out.append(
            _check("t1_not_e_unsafe", t1_safe, 0, t1_safe == 0, "II.8.3, C.16: x1 is E-UNSAFE")
        )
    return out


def e_identity(sources: Mapping[str, Iterable[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Section 5 (paper Sec. 3): every E record names the E that made it; one E for everything.

    ``sources`` maps a label (e.g. ``"exp1_outcome"``, ``"exp2_evaluations"``) to EvalRecord
    rows. One row per (source, prompt sha256, version) with its count, and ``single_e`` on
    every row: whether all sources together name exactly one E.
    """
    counts: Counter[tuple[str, str | None, str | None]] = Counter()
    for name, recs in sources.items():
        for r in recs:
            counts[(name, r.get("e_prompt_sha256"), r.get("e_version"))] += 1
    identities = {(p, v) for _, p, v in counts}
    return [
        {
            "source": name,
            "e_prompt_sha256": p,
            "e_version": v,
            "n_records": n,
            "single_e": len(identities) == 1,
        }
        for (name, p, v), n in sorted(counts.items(), key=lambda kv: tuple(map(str, kv[0])))
    ]
