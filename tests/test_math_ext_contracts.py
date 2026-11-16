"""2026-11-01 handoff contracts, separate from real-artifact regression evidence.

No provider is initialized. Missing original runs/bindings are explicit skips, never
replaced by synthetic data. Repository-root capture avoids mutable paths fixture leakage.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import re
import subprocess
from pathlib import Path

import pytest

from rafr.contracts.schemas import Arm, Lambda, LogRecord, Observation, Split
from rafr.controller import decide as ctrl
from rafr.core.config import NotFrozenError, load_yaml
from rafr.domains.math.generation import build_controller_config
from rafr.experiments.ext_config import assert_ext_frozen, load_ext
from rafr.formatter import math_feedback, templates
from rafr.formatter.extended import PiHat, format_extended
from rafr.formatter.math_extended import build_no_verifier_config, math_wording, original_feedback

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "5fca69dab5de98f32b186d75aed8646ad2f631e0"
# Files the math extension may not change. analysis/ext_analysis.py and
# analysis/ext_decision_quality.py are not listed: the authors' side maintains them (handoff
# 7.2) and changes them after the merge (2026-11-05: math's original Exp 2 finals mapping).
PROTECTED = (
    "configs/ext.yaml",
    "frozen",
    "analysis/ext_contrasts.py",
    "src/rafr/experiments/exp1.py",
    "src/rafr/experiments/exp2.py",
    "src/rafr/experiments/exp1_ext.py",
    "src/rafr/experiments/exp2_ext.py",
    "src/rafr/formatter",
    "src/rafr/channel",
    "src/rafr/validity",
    "src/rafr/controller",
    "src/rafr/logging",
    "src/rafr/data/labels.py",
    "src/rafr/core/seeding.py",
    "pipelines/math_pipeline.py",
    "pipelines/math_exp2_loop.py",
    "pipelines/math_tables.py",
    "src/rafr/domains/math/prompts",
)
FROZEN_FIVE = (
    "frozen/calibration_detox.json",
    "src/rafr/controller/prompts/no_verifier_system.txt",
    "src/rafr/controller/prompts/no_verifier_rewrite.txt",
    "src/rafr/formatter/prompts/detox/pihat.txt",
    "analysis/ext_contrasts.py",
)
NV_PATHS = (
    "src/rafr/formatter/prompts/math_ext/no_verifier_system.txt",
    "src/rafr/formatter/prompts/math_ext/no_verifier_rewrite.txt",
)


def _git(*args: str) -> bytes:
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)
    if p.returncode:
        pytest.skip("BLOCKED: Git baseline unavailable; protected-file verification not performed")
    return p.stdout


def _blob(name: str, ref: str = BASELINE) -> bytes:
    return _git("show", f"{ref}:{name}")


def _checkout_text(raw: bytes) -> bytes:
    # Existing Windows prompts may have CRLF; preserve every other byte (handoff 5.3/6.1).
    return raw.replace(b"\r\n", b"\n")


def _ext() -> dict:
    return load_ext(ROOT / "configs" / "math")


def _observation(s1=0.8, s2=0.2) -> Observation:
    return Observation(
        state_id="contract-public",
        lam=Lambda.LOW,
        s1_obs=s1,
        s2_obs=s2,
        tau1=0.5,
        tau2=0.5,
        raw_s1=s1,
        raw_s2=s2,
        v1_id="fixture1",
        v2_id="fixture2",
    )


def test_protected_original_code_and_artifacts_match_baseline():
    names = _git("ls-tree", "-rz", "--name-only", BASELINE, "--", *PROTECTED).decode().split("\0")
    names = [name for name in names if name]
    assert names and "pipelines/math_exp2_loop.py" in names
    changed = [
        name
        for name in names
        if not (ROOT / name).is_file() or _checkout_text((ROOT / name).read_bytes()) != _blob(name)
    ]
    assert not changed, f"protected original content changed: {changed}"


def test_arm_addition_preserves_all_original_names_values_and_order():
    old = ast.parse(_blob("src/rafr/contracts/schemas.py").decode())
    old_arm = next(n for n in old.body if isinstance(n, ast.ClassDef) and n.name == "Arm")
    pairs = [
        (n.targets[0].id, n.value.value)
        for n in old_arm.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and isinstance(n.value, ast.Constant)
    ]
    actual = [(a.name, a.value) for a in Arm]
    assert [p for p in actual if p[0] != "ORACLE_FEEDBACK"] == pairs
    assert set(actual) - set(pairs) == {("ORACLE_FEEDBACK", "oracle_feedback")}
    current = ast.parse((ROOT / "src/rafr/contracts/schemas.py").read_text(encoding="utf-8"))
    current_arm = next(n for n in current.body if isinstance(n, ast.ClassDef) and n.name == "Arm")
    current_arm.body = [
        n
        for n in current_arm.body
        if not (
            isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "ORACLE_FEEDBACK"
        )
    ]
    assert ast.dump(current, include_attributes=False) == ast.dump(old, include_attributes=False)


def test_math_references_shared_detox_blocks_and_has_no_local_copies():
    prompts = _ext()["prompts"]
    assert prompts["verdict_only_file"] == "src/rafr/formatter/prompts/detox/verdict_only.txt"
    assert prompts["pihat_file"] == "src/rafr/formatter/prompts/detox/pihat.txt"
    assert {p.name for p in (ROOT / "src/rafr/formatter/prompts/math_ext").glob("*.txt")} == {
        "no_verifier_system.txt",
        "no_verifier_rewrite.txt",
    }
    assert [prompts[k] for k in ("no_verifier_system_file", "no_verifier_rewrite_file")] == list(
        NV_PATHS
    )
    shared = (ROOT / prompts["pihat_file"]).read_text(encoding="utf-8").rstrip("\n")
    assert math_wording().pihat_template == shared.replace("{v1_name}", templates.V1_NAME).replace(
        "{v2_name}", templates.V2_NAME
    )


def test_lf_attributes_are_added_only_for_new_prompt_files():
    old = _blob(".gitattributes").decode().splitlines()
    current = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    original_rules = [line for line in old if line.strip() and not line.startswith("#")]
    rules = [line for line in current if line.strip() and not line.startswith("#")]
    assert [line for line in rules if line in original_rules] == original_rules
    assert set(rules) - set(original_rules) == {f"{name} text eol=lf" for name in NV_PATHS}
    for name in NV_PATHS:
        raw = (ROOT / name).read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n"), name


# 2026-11-05 (review of 30791a0): the math No Verifier system is math's system.txt with the
# checker parts removed and, as in detox's No Verifier system, the non-checker clause of each mixed
# sentence kept: "You see the problem and the complete current solution." (the authors' wording; the
# one added word is "and") and "Produce a complete replacement solution.". The rewrite only loses
# the checker block. (old, new) pairs, applied in order to the original file.
MATH_NO_VERIFIER_EDITS = {
    "system": (
        (
            " Two automated checkers score each revision. You see the problem, the complete"
            " current solution and what the checkers report. You decide what happens next.",
            " You see the problem and the complete current solution. You decide what happens next.",
        ),
        (
            " needs correction. Produce a complete replacement solution, which will be scored"
            " again.\n",
            " needs correction. Produce a complete replacement solution.\n",
        ),
        (
            "The checkers are imperfect. They miss errors and flag correct solutions. Their output"
            " is evidence about the solution, not a verdict on it. Weigh it against your own"
            " reasoning.\n\n",
            "",
        ),
    ),
    "rewrite": (("\n\nChecker output:\n---\n{observation}\n---", ""),),
}


@pytest.mark.parametrize("name", ["system", "rewrite"])
def test_no_verifier_is_the_original_with_only_the_recorded_edits(name):
    original = (ROOT / f"src/rafr/domains/math/prompts/{name}.txt").read_text(encoding="utf-8")
    new = (ROOT / f"src/rafr/formatter/prompts/math_ext/no_verifier_{name}.txt").read_text(
        encoding="utf-8"
    )
    expected = original
    for old, replacement in MATH_NO_VERIFIER_EDITS[name]:
        assert expected.count(old) == 1, old
        expected = expected.replace(old, replacement)
    assert new == expected
    assert "{observation}" not in new
    assert not re.search(r"checker|score|threshold|flag", new, re.IGNORECASE)


@pytest.mark.parametrize("budget", [1, 4])
def test_no_verifier_formal_request_has_no_checker_terms_and_same_controller(budget):
    # This constructs a request only. No provider/client/API is involved.
    original = build_controller_config(18042406, budget)
    config = build_no_verifier_config(18042406, budget)
    for field in (
        "provider",
        "api",
        "model",
        "version",
        "decoding",
        "base_seed",
        "seed_scheme",
        "retry_prompt",
        "show_x0",
        "round_budget",
        "refusal_phrases",
        "network",
    ):
        assert getattr(config, field) == getattr(original, field), field
    assert config.decoding["max_output_tokens"] == 16384
    assert config.decoding["reasoning_effort"] == "medium" and "temperature" not in config.decoding
    user = ctrl.render_user_turn(config, "Compute 2 + 2.", "The result is 4.", "PRIVATE_EVIDENCE")
    request = ctrl.build_request(config, [{"role": "user", "content": user}], 7, 0)
    rendered = (request.system or "") + "\n" + str(request.messages)
    assert not re.search(r"checker|score|threshold|flag", rendered, re.IGNORECASE)
    assert "PRIVATE_EVIDENCE" not in rendered and "{observation}" not in config.rewrite_template


@pytest.mark.parametrize("scores", [(0.8, 0.2), (0.5, 0.5), (0.0001, 0.99999)])
def test_b_pihat_preserves_b_exactly_and_only_pihat_allows_probability(scores):
    obs = _observation(*scores)
    wording = math_wording()
    a = original_feedback(Arm.A1, obs)
    b = original_feedback(Arm.B, obs)
    verdict = format_extended(Arm.VERDICT_ONLY, obs, wording)
    shown = format_extended(Arm.B_PIHAT, obs, wording, pihat=PiHat(0.234567, 0.765432))
    assert shown.startswith(b + wording.separator)
    extra = shown[len(b + wording.separator) :]
    assert "probability" in extra.casefold()
    assert "0.23457" in extra and "0.76543" in extra
    for block in (a, b, verdict):
        assert not any(word in block.casefold() for word in math_feedback.FORBIDDEN_WORDS)
    assert not any(
        word in extra.casefold() for word in math_feedback.FORBIDDEN_WORDS if word != "probability"
    )


def test_verdict_only_exact_deletion_and_no_distance_artifacts():
    original = templates.DERIVED_BLOCK
    removed = (
        "{v1_name} distance from its threshold: {m1}\n",
        "{v2_name} distance from its threshold: {m2}\n",
        " The distance is the score minus the threshold, signed: positive means at or above "
        "the threshold, negative means below.",
        " Distances from different checkers are not comparable to each other.",
    )
    for part in removed:
        assert original.count(part) == 1
        original = original.replace(part, "")
    shared = ROOT / _ext()["prompts"]["verdict_only_file"]
    assert shared.read_text(encoding="utf-8").removesuffix("\n") == original
    shown = format_extended(Arm.VERDICT_ONLY, _observation(), math_wording())
    assert not re.search(r"margin|distance|[+-]\d", shown, re.IGNORECASE)


def test_math_ext_config_matches_handoff_and_analysis_registry():
    from analysis.ext_analysis import DOMAINS

    ext = _ext()
    detox = load_yaml(ROOT / "configs/ext.yaml")
    assert ext["domain"] == "math" and DOMAINS["math"].ext_config == "configs/math/ext.yaml"
    assert DOMAINS["math"].panel_labels is None
    for exp in ("exp1", "exp2"):
        assert ext[exp]["run_id_prefix"] == f"math_{exp}ext"
        for cohort in ("e_incorrect", "e_correct"):
            assert ext[exp][cohort]["seed_namespace"] == f"math_{exp}" + (
                "_ecorrect" if cohort == "e_correct" else ""
            )
    for key in (
        "fit_split",
        "diagnostics_split",
        "epsilon",
        "max_clip_share",
        "targets",
        "ece_bins",
        "bootstrap",
    ):
        assert ext["calibration"][key] == detox["calibration"][key]
    for key in ("source_split", "n", "seed_component"):
        assert ext["cohorts"]["e_correct"][key] == detox["cohorts"]["e_correct"][key]
    assert ext["cohorts"]["e_correct"]["correct_label"] == "CORRECT"
    assert ext["smoke"]["seed_component"] == detox["smoke"]["seed_component"]
    assert ext["exp1"]["lambdas"] == detox["exp1"]["lambdas"]
    assert ext["exp2"]["k"] == 4
    assert ext["exp2"]["e_incorrect"]["t1_scores"]["reuse"] is True
    assert ext["exp2"]["e_correct"]["t1_scores"] == {
        "reuse": False,
        "compare_with_step06": True,
    }
    for key in ("contrasts", "extra_contrasts", "bootstrap"):
        assert ext["analysis"][key] == detox["analysis"][key]
    cited = ext["analysis"]["cited_not_recomputed"]["exp1"]["e_incorrect"]
    assert cited["tables_dir"] == "runs/math_formal_n9000/analysis"
    assert set(cited["sha256"]) == {"T06_exp1_did.json", "T06_exp1_delta_by_lambda.json"}
    assert "permutation.json" in ext["exp1"]["e_incorrect"]["original_sha256"]
    assert "assignment.json" in ext["exp2"]["e_incorrect"]["original_sha256"]
    assert "panel_" not in json.dumps(ext)
    assert ext["ceiling"]["arm"] == "oracle_feedback"
    assert ext["ceiling"]["seed_namespace"] == "math_ceiling" and ext["ceiling"]["k"] == 4


def test_unbound_artifact_config_remains_blocked_for_formal_execution():
    ext = _ext()
    if "TBD" not in json.dumps(ext):
        pytest.skip("bindings supplied; the missing-binding guard is covered by Core tests")
    with pytest.raises(NotFrozenError):
        assert_ext_frozen(ext)


def test_preregistered_math_does_not_discover_extension_runs():
    examples = [
        f"math_{exp}ext_{cohort}_20300101T000000Z"
        for exp in ("exp1", "exp2")
        for cohort in ("e_incorrect", "e_correct", "smoke_e_incorrect")
    ]
    patterns = []
    for name in ("pipelines/math_pipeline.py", "pipelines/math_tables.py"):
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"glob", "rglob"}
            ):
                assert node.args and isinstance(node.args[0], ast.Constant), (
                    "dynamic discovery requires review"
                )
                patterns.append(node.args[0].value)
    assert all(not fnmatch.fnmatch(name, pattern) for name in examples for pattern in patterns)
    tree = ast.parse((ROOT / "pipelines/math_pipeline.py").read_text(encoding="utf-8"))
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name in {"report", "report_exp1", "_exp1_rows"}:
            assert not any(
                isinstance(n, ast.Attribute) and n.attr in {"glob", "rglob", "RUNS"}
                for n in ast.walk(fn)
            )


def _real_source():
    from pipelines.math_ext_run import load_source

    binding = _ext().get("source_manifest")
    if binding in (None, "", "TBD"):
        pytest.skip(
            "BLOCKED real-artifact validation: source_manifest not supplied; "
            "no pilot replay claimed"
        )
    manifest = Path(binding)
    manifest = manifest if manifest.is_absolute() else ROOT / manifest
    assert not manifest.resolve().is_relative_to((ROOT / "archive").resolve())
    if not manifest.is_file():
        pytest.skip(f"BLOCKED real-artifact validation: source manifest absent: {manifest}")
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    root = Path(spec["source_root"])
    root = root if root.is_absolute() else manifest.parent / root
    if not root.is_dir():
        pytest.skip(f"BLOCKED real-artifact validation: original math run absent: {root}")
    assert not root.resolve().is_relative_to((ROOT / "archive").resolve())
    source = load_source(manifest)
    assert source.execution_mode == "live", "a fixture is not real-artifact regression evidence"
    return source


def test_real_original_math_log_records_validate_without_synthetic_substitution():
    source = _real_source()
    names = ["exp1/decisions.jsonl", "exp2/decisions.jsonl"]
    names += [f"exp2/rounds/round_{t}/decisions.jsonl" for t in range(1, 5)]
    count = 0
    for name in names:
        rows = source.read_bound_rows(name)
        # A complete later round can contain zero decisions after every trajectory
        # has terminated. Require its bound file, not invented extra decisions.
        if name in {"exp1/decisions.jsonl", "exp2/decisions.jsonl"}:
            assert rows, name
        for row in rows:
            LogRecord.model_validate(row)
            count += 1
    assert count >= 3000
    source.assert_unchanged()


def test_real_original_math_requests_and_seeds_replay_without_synthetic_substitution():
    source = _real_source()
    from pipelines.math_ext_regression import replay_original

    assert len(source.states(Split.D_EXP1)) == 300
    result = replay_original(source)
    assert result["exp1"]["n_original_decisions"] == 2700
    assert result["exp2"]["n_first_round_requests"] == 300
    assert sum(result["all_original_log_rows"].values()) >= 3000
