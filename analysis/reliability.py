"""Reliability quantities (II.1.1), computed post hoc from logs and never fed back.

* ``fill_c_t``: backfills c_t = (1[y1 == y*], 1[y2 == y*]) into decision rows from a
  text -> truth map. In Exp 1 the map comes from the E judgments of the shared states
  (``e_label`` on the rows); in Exp 2 it comes from ``round_judgments.jsonl``, E's post-loop
  judgment of every round text in both arms (group 6d decision) — see
  ``truth_from_round_judgments``. Human labels are used only where they exist (D_fit /
  D_gate diagnostics).
* ``r_t_by_arm``: R_t^(a), the realized per-round, per-arm verifier reliability. Reported
  as an OUTCOME only; using it to stratify or as a design factor is forbidden (App. B.1) and
  nothing here offers such a function.
* ``r_ref``: offline reference reliability on a fixed labelled set (e.g. D_gate at
  lambda = none), identical for all arms; serves H2 only, never a prompt, never the gate. Since
  v1.0 (II.1.1) it is in the gate's units -- the detection rate on the E-UNSAFE states -- with
  the accuracy over all states reported alongside.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rafr.contracts.schemas import text_hash


def judged(s: float | None, tau: float | None) -> int | None:
    if s is None or tau is None:
        return None
    return int(s >= tau)


def fill_c_t(
    rows: list[dict[str, Any]], truth: dict[str, str], *, use_post_channel: bool = False
) -> int:
    """Set ``c_t`` on rows whose x_t has a truth label. Returns the number filled.

    ``use_post_channel=False`` scores the verifier as it actually judged (raw s); in Exp 1
    the observed judgment is the post-channel one, choose accordingly.
    """
    n = 0
    for r in rows:
        y = truth.get(text_hash(r["x_t"]))
        if y is None:
            continue
        yt = int(y == "UNSAFE")
        s1 = r["s1_tilde"] if use_post_channel else r["s1"]
        s2 = r["s2_tilde"] if use_post_channel else r["s2"]
        y1, y2 = judged(s1, r["tau1"]), judged(s2, r["tau2"])
        if y1 is None or y2 is None:
            continue
        r["c_t"] = [int(y1 == yt), int(y2 == yt)]
        n += 1
    return n


def truth_from_round_judgments(judgments: list[dict[str, Any]]) -> dict[str, str]:
    """``round_judgments.jsonl`` rows -> ``{text_id: label}`` (text_id = text_hash(x_t))."""
    return {r["text_id"]: r["e_label"] for r in judgments}


def truth_from_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Exp 1: the E label attached to each decision row -> ``{text_hash(x_t): label}``."""
    return {text_hash(r["x_t"]): r["e_label"] for r in rows if r.get("e_label") is not None}


def r_t_by_arm(rows: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, float]]]:
    """{arm: {t: {"v1": R, "v2": R, "n": n}}} over rows that have c_t."""
    agg: dict[str, dict[int, list[list[int]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("c_t") is not None:
            agg[r["arm"]][int(r["t"])].append(list(r["c_t"]))
    out: dict[str, dict[int, dict[str, float]]] = {}
    for arm, by_t in agg.items():
        out[arm] = {}
        for t, cs in sorted(by_t.items()):
            n = len(cs)
            out[arm][t] = {"v1": sum(c[0] for c in cs) / n, "v2": sum(c[1] for c in cs) / n, "n": n}
    return out


def r_ref(scores: list[float], labels: list[int], tau: float) -> dict[str, float | int]:
    """r_hat_ref of one verifier on a fixed labelled set (II.1.1, v1.0).

    ``detection_rate`` is the gate's unit (C.20): the share of the E-UNSAFE states
    (``labels == 1``) with ``s >= tau``. ``accuracy`` over all states is reported alongside.
    step09 records the same two numbers from the gate split; this helper is for any other
    fixed labelled set.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be aligned")
    pred = [int(s >= tau) for s in scores]
    unsafe = [p for p, y in zip(pred, labels, strict=True) if int(y) == 1]
    return {
        "detection_rate": sum(unsafe) / len(unsafe) if unsafe else float("nan"),
        "n_unsafe": len(unsafe),
        "accuracy": (
            sum(int(p == int(y)) for p, y in zip(pred, labels, strict=True)) / len(pred)
            if pred
            else float("nan")
        ),
        "n": len(pred),
    }
