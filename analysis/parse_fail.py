"""parse_fail reporting (II.8.2). Any cell above 2% is flagged loudly, with a breakdown by
failure reason (the three preregistered categories + the malformed_json guard).

**Reporting grid (v0.8, II.8.2).** The dimension differs by experiment: Exp 1 reports
arm x lambda, Exp 2 reports arm x round. ``by`` selects it — ``"lam"`` (the default, Exp 1) or
``"t"`` (Exp 2). The reason is unchanged: B's input is longer, so its failure rate may be
higher, and the main-analysis accounting (parse_fail = not accepted) depresses FA, which
happens to favour B.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from rafr.contracts.schemas import PARSE_FAIL_ALERT_THRESHOLD, ParseFailReason

MALFORMED_JSON = ParseFailReason.MALFORMED_JSON.value
"""II.8.2's fourth category (structured output cut off): counted and reported on its own."""

BY_LAMBDA = "lam"
"""Exp 1 grid: arm x lambda (II.8.2)."""

BY_ROUND = "t"
"""Exp 2 grid: arm x round (II.8.2, v0.8)."""


def report(
    rows: list[dict[str, Any]],
    threshold: float = PARSE_FAIL_ALERT_THRESHOLD,
    *,
    by: str = BY_LAMBDA,
) -> dict[str, Any]:
    if by not in (BY_LAMBDA, BY_ROUND):
        raise ValueError(f"by must be {BY_LAMBDA!r} (Exp 1) or {BY_ROUND!r} (Exp 2), got {by!r}")
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cells[f"{r['arm']}|{r[by]}"].append(r)
    out: dict[str, Any] = {"threshold": threshold, "by": by, "cells": {}, "alerts": []}
    for key, rs in sorted(cells.items()):
        n_fail = sum(1 for r in rs if r["parse_fail"])
        reasons = Counter()
        retries = Counter()
        final = Counter()
        for r in rs:
            retries[r["n_parse_retries"]] += 1
            for a in r.get("attempts", []):
                if a.get("parse_fail_reason"):
                    reasons[a["parse_fail_reason"]] += 1
            if r["parse_fail"] and r.get("attempts"):
                final[r["attempts"][-1].get("parse_fail_reason")] += 1
        rate = n_fail / len(rs) if rs else 0.0
        out["cells"][key] = {
            "n": len(rs),
            "n_parse_fail": n_fail,
            "parse_fail_rate": rate,
            "ALERT": rate > threshold,
            # II.8.2: the category of each parse_fail DECISION is its last attempt's; the
            # truncated structured output (malformed_json) is counted and reported on its own
            "parse_fail_by_final_reason": dict(sorted(final.items(), key=lambda kv: str(kv[0]))),
            "n_parse_fail_malformed_json": final.get(MALFORMED_JSON, 0),
            "attempt_failure_reasons": dict(reasons),
            "retries_distribution": dict(sorted(retries.items())),
        }
        if rate > threshold:
            out["alerts"].append(
                f"!!! {key}: parse_fail rate {rate:.1%} > {threshold:.0%} (n={len(rs)})"
            )
    return out


def render(rep: dict[str, Any]) -> str:
    dim = "round" if rep.get("by") == BY_ROUND else "lambda"
    lines = [f"parse_fail by arm x {dim} (II.8.2)"]
    for key, c in rep["cells"].items():
        flag = "  <-- ALERT" if c["ALERT"] else ""
        lines.append(
            f"  {key:>18}: {c['parse_fail_rate']:.2%} ({c['n_parse_fail']}/{c['n']}){flag}"
        )
    lines += rep["alerts"]
    return "\n".join(lines)
