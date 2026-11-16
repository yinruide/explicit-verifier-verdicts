"""Regenerate the paper's tables and figures from the released run records (no API call, no GPU).

    python analysis/reproduce.py tables  --data DATA_DIR [--out out]
    python analysis/reproduce.py figures --data DATA_DIR [--out out]
    python analysis/reproduce.py verify  --data DATA_DIR [--out out]

``DATA_DIR`` holds ``runs/``, ``data/`` and ``frozen/`` as released (``--data`` or the
``RAFR_DATA_DIR`` environment variable). Nothing is written into it: every output goes under
``--out``.

* ``tables`` runs the analysis entry points in order -- the primary tables of both domains, the
  extension contrasts, the decision-quality analysis, the correction ceiling, appendices A/B --
  and then ``analysis/paper_tables.py``, which lays out every table of the paper in
  ``<out>/paper/tables.md`` (and ``tables.json``).
* ``figures`` draws the two operating-point figures from the regenerated decision-quality
  reports into ``<out>/figures/``.
* ``verify`` compares every number of the regenerated analysis outputs with the original outputs
  shipped in the data directory and exits non-zero on any difference.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DETOX_EXP1 = "runs/exp1_20260928T000021Z"
DETOX_EXP2 = "runs/exp2_20260928T033025Z"
DETOX_GATE = "data/diagnostics/step09_report.json"
MATH_RUN = "runs/math_formal_n9000"
DETOX_EXT = {
    "exp1_e_incorrect": ["runs/exp1ext_e_incorrect_20261031T112435Z"],
    "exp1_e_correct": ["runs/exp1ext_e_correct_20261031T112435Z"],
    "exp2_e_incorrect": [
        "runs/exp2ext_e_incorrect_nv_20261031T112435Z",
        "runs/exp2ext_e_incorrect_bpihat_20261031T152908Z",
        "runs/exp2ext_e_incorrect_verdict_20261031T152908Z",
        "runs/exp2ext_e_incorrect_fill_20261031T152908Z",
    ],
    "exp2_e_correct": [
        "runs/exp2ext_e_correct_nv_20261031T112435Z",
        "runs/exp2ext_e_correct_verifier_20261031T152908Z",
    ],
}
MATH_EXT = {
    "exp1_e_incorrect": ["runs/math_exp1ext_e_incorrect_01"],
    "exp1_e_correct": ["runs/math_exp1ext_e_correct_01"],
    "exp2_e_incorrect": [
        "runs/math_exp2ext_nv_i_01",
        "runs/math_exp2ext_e_incorrect_bpihat_01",
        "runs/math_exp2ext_e_incorrect_verdict_01",
        "runs/math_exp2ext_e_incorrect_fill_01",
    ],
    "exp2_e_correct": ["runs/math_exp2ext_nv_c_01", "runs/math_exp2ext_e_correct_verifier_01"],
}
# regenerated output -> the original output shipped in the data directory
ORIGINALS = {
    "detox_primary": "runs/analysis_20260928T124804Z",
    "math_primary": "runs/math_formal_n9000/analysis",
    "ext_analysis_detox": "runs/ext_analysis_detox_20261101T054907Z/report.json",
    "ext_decision_quality_detox": "runs/ext_decision_quality_detox_20261101T074059Z/report.json",
    "ext_analysis_math": "runs/ext_analysis_math_20261106T203556Z/report.json",
    "ext_decision_quality_math": "runs/ext_decision_quality_math_20261106T110046Z/report.json",
    "ext_ceiling_summary_math": "runs/ext_ceiling_summary_math_20261106T225017Z/report.json",
    "paper_appendix_ab": "runs/paper_appendix_ab_20261108/report.json",
}
PROVENANCE_TABLES = {"T14_provenance.json"}  # git state and paths at tabulation time
GENERATED_PREFIXES = (
    "ext_analysis_detox_",
    "ext_analysis_math_",
    "ext_decision_quality_detox_",
    "ext_decision_quality_math_",
    "ext_ceiling_summary_math_",
)


def _data_dir(arg: str | None) -> Path:
    d = arg or os.environ.get("RAFR_DATA_DIR")
    if not d:
        sys.exit("no data directory: pass --data DIR or set RAFR_DATA_DIR")
    p = Path(d).resolve()
    if not (p / "runs").is_dir():
        sys.exit(f"{p} has no runs/ directory; is it the released data directory?")
    return p


def _run(cmd: list[str], data: Path, out: Path) -> None:
    env = dict(os.environ, RAFR_DATA_DIR=str(data), RAFR_OUT_DIR=str(out))
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")])
    print("+", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, check=True)  # noqa: S603


def _latest(out: Path, prefix: str) -> Path:
    found = sorted(d for d in out.glob(f"{prefix}*") if d.is_dir() and "FAKE" not in d.name)
    if not found:
        sys.exit(f"no {prefix}* under {out}: run `tables` first")
    return found[-1]


def tables(data: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for d in list(out.iterdir()):
        if d.is_dir() and (d.name.startswith(GENERATED_PREFIXES) or d.name in ORIGINALS):
            shutil.rmtree(d)
    _run(
        [
            "-m",
            "analysis.prereg_tables",
            "--exp1-run-dir",
            str(data / DETOX_EXP1),
            "--exp2-run-dir",
            str(data / DETOX_EXP2),
            "--gate-report",
            str(data / DETOX_GATE),
            "--out-dir",
            str(out / "detox_primary"),
        ],
        data,
        out,
    )
    _run(
        [
            "pipelines/math_tables.py",
            "--run-dir",
            str(data / MATH_RUN),
            "--out-dir",
            str(out / "math_primary"),
        ],
        data,
        out,
    )
    for dom, spec in (("detox", DETOX_EXT), ("math", MATH_EXT)):
        _run(
            [
                "analysis/ext_analysis.py",
                "--domain",
                dom,
                "--exp1-e-incorrect",
                *spec["exp1_e_incorrect"],
                "--exp1-e-correct",
                *spec["exp1_e_correct"],
                "--exp2-e-incorrect",
                *spec["exp2_e_incorrect"],
                "--exp2-e-correct",
                *spec["exp2_e_correct"],
            ],
            data,
            out,
        )
        _run(
            [
                "analysis/ext_decision_quality.py",
                "--domain",
                dom,
                "--e-incorrect",
                *spec["exp1_e_incorrect"],
                "--e-correct",
                *spec["exp1_e_correct"],
            ],
            data,
            out,
        )
    _run(["analysis/ext_ceiling_summary.py"], data, out)
    _run(["analysis/paper_appendix_ab.py", "--out", str(out / "paper_appendix_ab")], data, out)
    _run(["analysis/paper_tables.py", "--out", str(out)], data, out)


def figures(data: Path, out: Path) -> None:
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for dom in ("detox", "math"):
        report = _latest(out, f"ext_decision_quality_{dom}_") / "report.json"
        _run(
            [
                "analysis/paper_operating_points.py",
                "--report",
                str(report),
                "--out",
                str(out / "figures" / f"{dom}_operating_points"),
            ],
            data,
            out,
        )


def _numbers(x: Any, path: str = "") -> dict[str, float]:
    """Every number in a JSON tree, keyed by its path (strings are provenance, not compared)."""
    out: dict[str, float] = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out |= _numbers(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out |= _numbers(v, f"{path}[{i}]")
    elif isinstance(x, bool):
        out[path] = float(x)
    elif isinstance(x, int | float):
        out[path] = float(x)
    return out


def _compare(mine: Path, orig: Path, ignore: tuple[str, ...] = ()) -> list[str]:
    a = _numbers(json.loads(mine.read_text(encoding="utf-8")))
    b = _numbers(json.loads(orig.read_text(encoding="utf-8")))
    bad = []
    for k in sorted(set(a) | set(b)):
        if any(s in k for s in ignore):
            continue
        x, y = a.get(k), b.get(k)
        same = x == y or (x is not None and y is not None and x != x and y != y)  # NaN == NaN
        if not same:
            bad.append(f"{mine.name}{k}: regenerated {x!r}, original {y!r}")
    return bad


def verify(data: Path, out: Path) -> int:
    problems: list[str] = []
    n = 0
    for name in ("detox_primary", "math_primary"):
        orig_dir = data / ORIGINALS[name]
        for orig in sorted(orig_dir.glob("T*.json")):
            if orig.name in PROVENANCE_TABLES:
                continue
            mine = out / name / orig.name
            if not mine.is_file():
                problems.append(f"{name}/{orig.name}: not regenerated")
                continue
            problems += _compare(mine, orig)
            n += 1
    for name in ORIGINALS:
        if name in ("detox_primary", "math_primary"):
            continue
        mine = (
            (out / name / "report.json")
            if name == "paper_appendix_ab"
            else (_latest(out, f"{name}_") / "report.json")
        )
        # seeds and git provenance are recorded, not estimated; the appendix's source hashes cover
        # design documents that are not released
        problems += _compare(mine, data / ORIGINALS[name], ignore=("/sources/",))
        n += 1
    for p in problems:
        print("MISMATCH", p)
    print(
        f"[verify] {n} regenerated outputs compared number by number with the originals: "
        f"{'all equal' if not problems else f'{len(problems)} differences'}"
    )
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("tables", "figures", "verify"))
    ap.add_argument("--data", help="released data directory (default: $RAFR_DATA_DIR)")
    ap.add_argument("--out", default="out", help="output directory (default: out/)")
    args = ap.parse_args(argv)
    data = _data_dir(args.data)
    out = Path(args.out).resolve()
    if out == data or data in out.parents:
        sys.exit("--out must not be inside the data directory")
    if args.what == "tables":
        tables(data, out)
    elif args.what == "figures":
        figures(data, out)
    else:
        return verify(data, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
