"""Every pipeline step must run the way the Makefile and README run it: as a script.

``python pipelines/step05_annotation.py`` puts pipelines/ on sys.path, not the repo root, so
a step importing one of the repo-root packages -- analysis, annotation, pipelines -- died
with ModuleNotFoundError unless the caller had exported PYTHONPATH by hand (the authors,
2026-09-20: step05, and in fact 07, 08, 09, 10, 11 and 12 too; the pythonpath in
pyproject.toml is a pytest setting and does nothing for a direct run). Each step now puts
the repo root on sys.path itself, and this is what keeps it that way -- including for a step
whose imports are all ``rafr`` today and grow a ``from analysis import ...`` tomorrow.

``--help`` only: argparse exits during parsing, so nothing here runs a step, spends an API
call or writes anything. What is under test is the import block at the top of the file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STEPS = sorted(ROOT.glob("pipelines/step*.py"))


def test_every_step_is_discovered():
    """Guard the glob: a rename must not quietly turn the test below into a no-op."""
    assert [p.name for p in STEPS], f"no pipelines/step*.py under {ROOT}"


@pytest.mark.parametrize("step", STEPS, ids=lambda p: p.stem)
def test_step_runs_as_a_script_without_pythonpath(step: Path, tmp_path: Path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, str(step), "--help"],
        cwd=tmp_path,  # from a foreign cwd too: paths resolve off the repo root, not it
        env=env,
        capture_output=True,
        text=True,
        errors="replace",  # the help text carries tau/lambda; the console codepage may not
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"`python {step.relative_to(ROOT).as_posix()} --help` exited {proc.returncode} with "
        f"no PYTHONPATH set:\n{proc.stdout}\n{proc.stderr}"
    )
    assert proc.stdout.startswith(f"usage: {step.name}"), (
        f"{step.name} did not print its argparse help:\n{proc.stdout}\n{proc.stderr}"
    )
