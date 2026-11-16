"""Math's correction ceiling (oracle feedback) beside No Verifier and A1, on the same starts.

POST HOC (2026-11-06): added after the math extension tables had been seen. EXPLORATORY:
the intervals are not corrected for multiple comparisons, and every cell is reported. Read only:
no API call, no pod.

Inputs: the ceiling run (``ceiling_results.json``, checked against ``ceiling_complete.json``) and
its E judgments (``extension_e_evaluations.jsonl``); the No Verifier e_incorrect run and the P2
fill run, with the original Exp 2 finals (sha256 from ``configs/math/ext.yaml``), read through
``analysis.ext_analysis.gather`` exactly as the extension analysis reads them. A1 therefore covers
all 300 starts: the original Exp 2's A1 half plus the fill run's A1 on the other half.

1. Cumulative correction rate by round t = 1..4: the share of the starts whose first rewrite that
   E judged correct came at a decision <= t (UNRESOLVED counts as not correct, paper App. D.1).
2. TFR = P(task failure at x_T) of the ceiling (``oracle_feedback``), No Verifier and A1, and the
   paired contrasts no_verifier - oracle_feedback and A1 - oracle_feedback (less information -
   more information, the extension's direction: positive = the oracle feedback reduced failures).
3. Sensitivity: the starts whose ceiling trajectory has any UNRESOLVED E judgment are dropped
   from every arm (pairs stay paired); 1 and 2 are repeated on the rest.
4. The UNRESOLVED judgments: E's reason and the boxed answer's type (:func:`answer_type`, a fixed
   rule list in priority order), per distinct judged text.

Intervals: ``ext_contrasts.paired_block`` (95% percentile bootstrap, unit x0, paired). B and the
seed are the extension analysis's (``analysis.bootstrap`` of ``configs/math/ext.yaml``, the
component of math's master seed), so the 300 starts are resampled with the draws of the
extension's Exp 2 e_incorrect table.

    python analysis/ext_ceiling_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from analysis import ext_analysis as A
from analysis import ext_contrasts as X
from rafr.calibration.apply import file_sha256
from rafr.contracts.schemas import BOOTSTRAP_ALPHA
from rafr.core import paths
from rafr.core.config import get
from rafr.core.seeding import component_seed
from rafr.logging.writer import read_jsonl

OUT_PREFIX = "ext_ceiling_summary_math"
CEILING_RUN = "runs/math_ceiling_formal_01"
NV_RUN = "runs/math_exp2ext_nv_i_01"
FILL_RUN = "runs/math_exp2ext_e_incorrect_fill_01"
CEILING, NV, A1 = "oracle_feedback", "no_verifier", "A1"
CONTRASTS = ((NV, CEILING), (A1, CEILING))
K = 4
JUDGED = {"correct", "incorrect"}  # any other round status is an UNRESOLVED judgment

# Boxed-answer types of an UNRESOLVED judgment, first match wins.
ANSWER_TYPES: tuple[tuple[str, str], ...] = (
    ("no final box", ""),
    ("E timeout (three wall-clock timeouts)", ""),
    ("piecewise (cases)", r"\\begin\{cases\}"),
    ("contains words (\\text)", r"\\text\b|\\text\{"),
    ("set / set-builder / union", r"\\\{|\\cup|\\setminus"),
    ("unevaluated integral, sum, sup or series", r"\\i+nt|\\sum|\\sup|\\inf\b|\\cdots|\\ldots|min"),
    ("named object (operator, distribution, group, special set)",
     r"\\operatorname|\\mathcal|\\mathfrak|\\mathbb|\\rtimes|\\mathrm"),
    ("equation or relation", r"=|\\ne\b|\\le|\\ge|<|>"),
)  # fmt: skip
OTHER = "other"


class CeilingSummaryError(RuntimeError):
    pass


def _shown(p: Path) -> str:
    p = Path(p)
    try:
        return paths.shown(p)
    except ValueError:
        return p.as_posix()


def answer_type(reason: str, prediction: str | None) -> str:
    """The UNRESOLVED judgment's answer type (``ANSWER_TYPES``, first match wins)."""
    if "missing_final_box" in reason:
        return ANSWER_TYPES[0][0]
    if "timeout" in reason:
        return ANSWER_TYPES[1][0]
    for name, pattern in ANSWER_TYPES[2:]:
        if re.search(pattern, prediction or ""):
            return name
    return OTHER


def first_correct(rounds: Sequence[Mapping[str, Any]]) -> int | None:
    """The decision t at which E first judged the rewrite correct (None = never)."""
    for r in sorted(rounds, key=lambda r: r["t"]):
        if r["grade"]["status"] == "correct":
            return int(r["t"])
    return None


def unresolved_rounds(rounds: Sequence[Mapping[str, Any]]) -> int:
    return sum(r["grade"]["status"] not in JUDGED for r in rounds)


def load_ceiling(rd: Path) -> list[dict[str, Any]]:
    """The ceiling's 300 results, after checking every file its completion marker binds."""
    complete = json.loads((rd / "ceiling_complete.json").read_text(encoding="utf-8"))
    for name, sha in complete["files"].items():
        got = hashlib.sha256((rd / name).read_bytes()).hexdigest()
        if got != sha:
            raise CeilingSummaryError(f"{rd / name}: sha256 {got} != ceiling_complete {sha}")
    meta = json.loads((rd / "ext_meta.json").read_text(encoding="utf-8"))
    if meta.get("SMOKE") or meta.get("experiment") != "correction_ceiling":
        raise CeilingSummaryError(f"{rd} is not a formal correction ceiling run")
    rows = json.loads((rd / "ceiling_results.json").read_text(encoding="utf-8"))
    if len(rows) != complete["n"] or len({r["x0_id"] for r in rows}) != len(rows):
        raise CeilingSummaryError(f"{rd}: results do not match the completion marker")
    for r in rows:
        fc = first_correct(r["rounds"])
        if r["arm"] != CEILING or bool(r["task_success"]) != (fc is not None):
            raise CeilingSummaryError(f"{r['x0_id']}: task_success disagrees with its rounds")
        if len(r["rounds"]) != r["n_decisions"] or not 1 <= r["n_decisions"] <= K:
            raise CeilingSummaryError(f"{r['x0_id']}: {r['n_decisions']} decisions")
    return rows


def outcomes(
    ceiling: Sequence[Mapping[str, Any]], finals: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], set[str]]:
    """(cumulative correction by round, task failure per arm, starts with an UNRESOLVED ceiling
    judgment); every dict is ``[name][x0] -> 0/1``."""
    cum = {f"t<={t}": {} for t in range(1, K + 1)}
    failed: dict[str, dict[str, float]] = {CEILING: {}, NV: {}, A1: {}}
    flagged = set()
    for r in ceiling:
        fc = first_correct(r["rounds"])
        for t in range(1, K + 1):
            cum[f"t<={t}"][r["x0_id"]] = float(fc is not None and fc <= t)
        failed[CEILING][r["x0_id"]] = float(not r["task_success"])
        if unresolved_rounds(r["rounds"]):
            flagged.add(r["x0_id"])
    for f in finals:
        if f["arm"] in (NV, A1):
            if f["x0_id"] in failed[f["arm"]]:
                raise CeilingSummaryError(f"{f['arm']} {f['x0_id']} appears twice")
            failed[f["arm"]][f["x0_id"]] = float(not f["task_success"])
    starts = set(failed[CEILING])
    for arm in (NV, A1):
        if set(failed[arm]) != starts:
            raise CeilingSummaryError(f"{arm} does not cover the ceiling's {len(starts)} starts")
    return cum, failed, flagged


def blocks(
    cum: Mapping[str, Mapping[str, float]],
    failed: Mapping[str, Mapping[str, float]],
    units: Sequence[str],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    keep = set(units)
    sub = {k: {u: v for u, v in d.items() if u in keep} for k, d in cum.items()}
    fsub = {k: {u: v for u, v in d.items() if u in keep} for k, d in failed.items()}
    kw = {"n_boot": n_boot, "seed": seed, "alpha": BOOTSTRAP_ALPHA}
    return {
        "n_starts": len(units),
        "cumulative_correction": X.paired_block(sub, sorted(units), (), **kw),
        "tfr": X.paired_block(fsub, sorted(units), CONTRASTS, **kw),
    }


def unresolved_table(rd: Path, ceiling: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The UNRESOLVED judgments: per distinct judged text (E's evaluations), by reason and type."""
    rows = [
        e for e in read_jsonl(rd / "extension_e_evaluations.jsonl") if e["label"] == "UNRESOLVED"
    ]
    by_text = {e["text_id"]: e for e in rows}
    types: dict[str, list[dict[str, Any]]] = {}
    for e in by_text.values():
        types.setdefault(answer_type(e["reason"], e.get("extracted_answer")), []).append(e)
    order = [n for n, _ in ANSWER_TYPES] + [OTHER]
    table = []
    for name in order:
        es = types.get(name, [])
        if not es:
            continue
        distinct = list({(e.get("extracted_answer") or ""): e for e in reversed(es)}.values())[::-1]
        examples = [
            {
                "answer": (e.get("extracted_answer") or "")[:70],
                "reference": str(e["reference_answer"])[:40],
            }
            for e in distinct[:2]
        ]
        table.append(
            {
                "type": name,
                "n_texts": len(es),
                "reasons": dict(Counter(e["reason"] for e in es)),
                "examples": examples,
            }
        )
    at_final = sum(
        1 for r in ceiling if r["rounds"] and r["rounds"][-1]["grade"]["status"] not in JUDGED
    )
    return {
        "n_texts": len(by_text),
        "n_rounds": sum(unresolved_rounds(r["rounds"]) for r in ceiling),
        "n_trajectories": sum(1 for r in ceiling if unresolved_rounds(r["rounds"])),
        "n_trajectories_unresolved_at_x_T": at_final,
        "reasons": dict(Counter(e["reason"] for e in by_text.values())),
        "types": table,
    }


def analyze(
    ceiling_run: Path, nv_run: Path, fill_run: Path, *, n_boot: int | None = None
) -> dict[str, Any]:
    ext = A.load_domain_ext("math")
    seed_component = str(get(ext, "analysis.bootstrap.seed_component"))
    seed = component_seed(A.domain_master_seed("math"), seed_component)
    n_boot = int(get(ext, "analysis.bootstrap.n_boot")) if n_boot is None else n_boot
    _, _, exp2, _ = A.gather("math", ext, exp2_e_incorrect=[nv_run, fill_run])
    ceiling = load_ceiling(ceiling_run)
    cum, failed, flagged = outcomes(ceiling, exp2[A.E_INCORRECT])
    starts = sorted(failed[CEILING])
    rest = sorted(set(starts) - flagged)
    return {
        "post_hoc": "added 2026-11-06 after the math extension tables had been seen; exploratory",
        "domain": "math",
        "cohort": A.E_INCORRECT,
        "arms": {
            CEILING: _shown(ceiling_run),
            NV: _shown(nv_run),
            A1: f"original Exp 2 (A1 half) + {_shown(fill_run)}",
        },
        "contrasts": [X._name(c) for c in CONTRASTS],
        "bootstrap": {"n_boot": n_boot, "seed_component": seed_component, "seed": seed},
        "main": blocks(cum, failed, starts, n_boot=n_boot, seed=seed),
        "sensitivity_unresolved_dropped": {
            "n_dropped": len(flagged),
            "rule": "starts whose ceiling trajectory has any UNRESOLVED E judgment, dropped from "
            "every arm",
            **blocks(cum, failed, rest, n_boot=n_boot, seed=seed),
        },
        "unresolved": unresolved_table(ceiling_run, ceiling),
    }


def _section(title: str, b: Mapping[str, Any]) -> list[str]:
    lines = [f"## {title} ({b['n_starts']} starts)", ""]
    lines += ["### Cumulative correction rate (first E-correct rewrite at decision <= t)", ""]
    lines += ["| round | rate [95% CI] (n used / n) |", "|---|---|"]
    for name, a in b["cumulative_correction"]["arms"].items():
        lines.append(f"| {name} | {X._arm_cell(a)} |")
    lines += ["", "### TFR = P(task failure at x_T)", ""]
    lines += ["| arm | rate [95% CI] (n used / n) |", "|---|---|"]
    for name in (CEILING, NV, A1):
        lines.append(f"| {name} | {X._arm_cell(b['tfr']['arms'][name])} |")
    lines += [
        "",
        "| contrast | estimate [95% CI] (n pairs) | rate X | rate Y |",
        "|---|---|---|---|",
    ]
    for name, c in b["tfr"]["contrasts"].items():
        lines.append(
            f"| {name} | {X._contrast_cell(c)} | {X._rate(c['rate_x'])} | {X._rate(c['rate_y'])} |"
        )
    return [*lines, ""]


def render(report: Mapping[str, Any]) -> str:
    bs = report["bootstrap"]
    s = report["sensitivity_unresolved_dropped"]
    u = report["unresolved"]
    lines = [
        "# Correction ceiling summary (math, e_incorrect)",
        "",
        "**POST HOC (added 2026-11-06 after the math extension tables had been seen). "
        "EXPLORATORY: intervals not corrected for multiple comparisons; every cell reported**",
        "",
        f"Arms: {CEILING} = {report['arms'][CEILING]}; {NV} = {report['arms'][NV]}; "
        f"{A1} = {report['arms'][A1]}.",
        "",
        "Direction: X - Y = less information - more information; positive = the oracle feedback "
        "reduced task failures. UNRESOLVED counts as not correct (M.9).",
        "",
        f"Intervals: 95% percentile bootstrap, B = {bs['n_boot']}, unit x0 (paired), seed "
        f"{bs['seed']} (component {bs['seed_component']}, as the extension analysis); "
        "no multiplicity correction.",
        "",
        *_section("Main", report["main"]),
        *_section(
            f"Sensitivity: {s['n_dropped']} starts with an UNRESOLVED ceiling judgment dropped", s
        ),  # fmt: skip
        "## UNRESOLVED judgments of the ceiling",
        "",
        f"{u['n_texts']} distinct texts, {u['n_rounds']} rounds, {u['n_trajectories']} "
        f"trajectories ({u['n_trajectories_unresolved_at_x_T']} with an UNRESOLVED x_T). "
        f"Reasons: {', '.join(f'{k} {v}' for k, v in u['reasons'].items())}.",
        "",
        "| answer type | texts | reasons | examples (answer -> reference) |",
        "|---|---:|---|---|",
    ]
    for t in u["types"]:
        reasons = ", ".join(f"{k} {v}" for k, v in t["reasons"].items())
        ex = "; ".join(
            f"`{e['answer'] or '(none)'}` -> `{e['reference']}`".replace("|", "\\|")
            for e in t["examples"]
        )
        lines.append(f"| {t['type']} | {t['n_texts']} | {reasons} | {ex} |")
    return "\n".join([*lines, ""])


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ceiling", type=Path, default=Path(CEILING_RUN))
    ap.add_argument("--no-verifier", type=Path, default=Path(NV_RUN))
    ap.add_argument("--fill", type=Path, default=Path(FILL_RUN))
    args = ap.parse_args(argv)
    report = analyze(
        paths.resolve(args.ceiling), paths.resolve(args.no_verifier), paths.resolve(args.fill)
    )
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out = paths.run_dir(f"{OUT_PREFIX}_{ts}")
    inputs = {
        "ext_config": "configs/math/ext.yaml",
        "ext_config_sha256": file_sha256(paths.resolve("configs/math/ext.yaml")),
        "runs": {
            "ceiling": args.ceiling.as_posix(),
            "no_verifier": args.no_verifier.as_posix(),
            "fill": args.fill.as_posix(),
        },
    }
    files = {
        "report.json": json.dumps(X._clean(report), indent=1, ensure_ascii=False) + "\n",
        "tables.md": render(report),
        "inputs.json": json.dumps(inputs, indent=1) + "\n",
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"[ext_ceiling_summary] -> {paths.shown(out / name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
