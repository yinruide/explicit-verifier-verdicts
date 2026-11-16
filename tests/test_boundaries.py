"""Static invariants enforced by scanning the source tree (paper App. E, Sec. 4.3, 3).

1. No arm branch outside ``rafr.formatter`` and ``rafr.experiments``.
2. ``rafr.channel``, ``rafr.formatter``, ``rafr.controller`` never import (transitively)
   ``annotation``, ``rafr.data.labels`` or ``rafr.evaluator``.
3. ``rafr.evaluator`` never imports the formatter, the channel or the controller (E sees text
   only).
4. ``c_t`` / ``r_ref`` never appear on the decision path.
"""

from __future__ import annotations

import ast
import re
import tokenize
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "rafr"
DECISION_PATH = ("channel", "formatter", "controller")
FORBIDDEN_FOR_DECISION_PATH = ("annotation", "rafr.data.labels", "rafr.evaluator", "rafr.analysis")
FORBIDDEN_FOR_EVALUATOR = ("rafr.formatter", "rafr.channel", "rafr.controller")

ARM_BRANCH_PATTERNS = [
    re.compile(r"\bif\s+.*\barm\b.*(==|!=|\bis\b|\bin\b)"),
    re.compile(r"\bmatch\s+.*\barm\b"),
    re.compile(r"\bArm\.[A-Z0-9_]+\s*(==|!=)"),
    re.compile(r"(==|!=)\s*Arm\.[A-Z0-9_]+"),
    re.compile(r"\barm\s*(==|!=)\s*"),
    re.compile(r"\barm\.value\s*(==|!=)"),
]
ARM_BRANCH_ALLOWED_DIRS = ("formatter", "experiments")


def _py_files(root: Path):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _transitive_imports(start: set[str]) -> set[str]:
    """Follow rafr.* imports through the source tree."""
    seen: set[str] = set()
    stack = [m for m in start if m.startswith("rafr")]
    index = {_module_name(p): p for p in _py_files(SRC)}
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        p = index.get(m)
        if p is None:
            continue
        for imp in _imports(p):
            if imp.startswith("rafr") and imp not in seen:
                stack.append(imp)
    return seen


@pytest.mark.parametrize("pkg", DECISION_PATH)
def test_decision_path_never_imports_labels_annotation_or_evaluator(pkg):
    for f in _py_files(SRC / pkg):
        closure = _transitive_imports(_imports(f)) | _imports(f)
        bad = [
            m
            for m in closure
            if any(m == x or m.startswith(x + ".") for x in FORBIDDEN_FOR_DECISION_PATH)
        ]
        assert not bad, f"{f} reaches {bad}"


def test_evaluator_never_imports_decision_path():
    for f in _py_files(SRC / "evaluator"):
        closure = _transitive_imports(_imports(f)) | _imports(f)
        bad = [
            m
            for m in closure
            if any(m == x or m.startswith(x + ".") for x in FORBIDDEN_FOR_EVALUATOR)
        ]
        assert not bad, f"{f} reaches {bad}"


def test_no_arm_branch_outside_formatter_and_experiments():
    offenders = []
    for f in _py_files(SRC):
        rel = f.relative_to(SRC)
        if rel.parts and rel.parts[0] in ARM_BRANCH_ALLOWED_DIRS:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if any(p.search(code) for p in ARM_BRANCH_PATTERNS):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "arm branches outside formatter/experiments:\n" + "\n".join(offenders)


def test_arm_branch_scanner_actually_detects_a_branch(tmp_path: Path):
    sample = "def f(arm):\n    if arm == Arm.B:\n        return 1\n"
    assert any(p.search(line) for line in sample.splitlines() for p in ARM_BRANCH_PATTERNS)


def _names(path: Path) -> set[str]:
    """Identifiers used in code (comments and string literals excluded)."""
    out: set[str] = set()
    with path.open("rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type == tokenize.NAME:
                out.add(tok.string)
    return out


def test_c_t_and_r_ref_absent_from_decision_path():
    forbidden = {"c_t", "r_ref", "r_hat_ref", "y_star"}
    for pkg in DECISION_PATH:
        for f in _py_files(SRC / pkg):
            bad = _names(f) & forbidden
            assert not bad, f"{f} uses {bad}"


def test_r_ref_never_enters_the_gate_and_r_t_never_stratifies_primary_analyses():
    """paper Sec. 3: r_ref is not a gate input; R_t is never a stratification variable."""
    root = SRC.parent.parent
    gate_files = [SRC / "validity" / "gate.py", SRC / "validity" / "cue_diagnostics.py"]
    for f in gate_files:
        assert not (_names(f) & {"r_ref", "r_hat_ref"}), f"{f} references r_ref"
    primary = [
        root / "analysis" / "exp1_did.py",
        root / "analysis" / "exp2_tfr.py",
        # paper Sec. 6.2 feeds the null-result A/B split, so it is held to the same rule
        root / "analysis" / "error_boundary.py",
    ]
    for f in primary:
        names = _names(f)
        assert "reliability" not in _imports(f) and "analysis.reliability" not in _imports(f)
        assert not (names & {"r_t_by_arm", "fill_c_t", "c_t", "r_ref"}), (
            f"{f} uses post-treatment reliability"
        )


def test_x0_prefilter_never_reaches_a_verifier():
    """the x0 prefilter uses E, never V1/V2 — verifier error is the object
    of study and must not select the data. step00 is the only place the prefilter runs."""
    step00 = SRC.parents[1] / "pipelines" / "step00_prepare_data.py"
    closure = _transitive_imports(_imports(step00)) | _imports(step00)
    bad = [m for m in closure if m == "rafr.verifiers" or m.startswith("rafr.verifiers.")]
    assert not bad, f"{step00} reaches {bad}"
    names = _names(step00)
    assert not (names & {"score_pair", "InProcessScorer", "StubVerifier", "stub_pair"})
