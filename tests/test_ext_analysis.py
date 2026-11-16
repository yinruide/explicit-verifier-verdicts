"""The per-domain wrapper of the extension contrasts (2026-11-01; analysis/ext_analysis.py):
the domain registry (config, master seed, panel vocabulary), the wrapper's report equal to
ext_contrasts' on the same fake data, math's absent panel, and -- where the
detox runs are -- the detox report byte for byte runs/ext_analysis_20261101T004507Z's."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from analysis import ext_analysis as W
from analysis import ext_contrasts as X

from rafr.core import paths
from rafr.core.config import load_config, master_seed
from rafr.core.seeding import component_seed
from rafr.experiments.ext_config import load_ext
from rafr.logging.writer import read_jsonl

KW = {"n_boot": 200, "seed": 3}


def _dump(report: dict) -> str:
    return json.dumps(X._clean(report), indent=1)


def test_the_registry():
    assert set(W.DOMAINS) == {"detox", "math"}
    assert W.load_domain_ext("detox") == load_ext()
    assert W.domain_master_seed("detox") == master_seed(load_config())
    math_seed = json.loads((paths.ROOT / W.MATH_RUN_CONFIG).read_text("utf-8"))["seed"]
    assert W.domain_master_seed("math") == math_seed
    with pytest.raises(W.ExtDomainError, match="unknown domain"):
        W.domain("chess")
    assert W.DOMAINS["math"].ext_config == "configs/math/ext.yaml"
    assert W.DOMAINS["math"].panel_labels is None


def test_a_missing_or_foreign_config_is_refused(tmp_path: Path):
    with pytest.raises(W.ExtDomainError, match="no extension config"):
        W.load_domain_ext("math", tmp_path / "missing.yaml")
    other = tmp_path / "ext.yaml"
    other.write_text("domain: detox\n", encoding="utf-8")
    with pytest.raises(W.ExtDomainError, match="is for 'detox', not 'math'"):
        W.load_domain_ext("math", other)
    with pytest.raises(W.ExtDomainError, match="for 'detox', not 'math'"):
        W.gather("math", {"domain": "detox"})


def test_detox_on_fake_data_is_ext_contrasts(tmp_path: Path):
    root = tmp_path / "fake"
    ref = X.analyze_fake(root, **KW)
    spec, ext = W.fake_inputs(root, "detox")  # the same data, rewritten in place
    got = W.analyze("detox", ext, **KW, **W.fake_runs(spec))
    assert _dump(got) == _dump(ref)
    assert X.render(got) == X.render(ref)


def test_math_has_no_panel(tmp_path: Path):
    spec, ext = W.fake_inputs(tmp_path / "m", "math")
    assert not list((tmp_path / "m").rglob("panel_*.jsonl"))
    rep = W.analyze("math", ext, **KW, **W.fake_runs(spec))
    assert all(rep["exp2"][c]["panel"] == {} for c in X.COHORTS)
    with pytest.raises(W.ExtDomainError, match="has no panel"):
        W.map_panel([{"x0_id": "x", "arm": "A1", "_panel": {"grok": "SAFE"}}], "math")


def test_math_original_exp2_finals_get_x0_id_and_accepted(tmp_path: Path):
    """2026-11-05: math's original Exp 2 finals name the start question_id and record
    termination_reason; ext_contrasts reads x0_id and accepted. On the same fake data the math
    report's Exp 2 numbers are detox's (math has no panel)."""
    spec_d, ext_d = W.fake_inputs(tmp_path / "d", "detox")
    spec_m, ext_m = W.fake_inputs(tmp_path / "m", "math")
    rows = list(read_jsonl(tmp_path / "m" / "exp2_original" / "trajectories.jsonl"))
    assert rows and all(
        "question_id" in r and "x0_id" not in r and "accepted" not in r for r in rows
    )
    det = W.analyze("detox", ext_d, **KW, **W.fake_runs(spec_d))
    mat = W.analyze("math", ext_m, **KW, **W.fake_runs(spec_m))
    for c in X.COHORTS:
        without_panel = [
            {k: v for k, v in r["exp2"][c].items() if k != "panel"} for r in (det, mat)
        ]
        assert _dump(without_panel[1]) == _dump(without_panel[0])
    row = {"question_id": "q", "arm": "A1", "task_success": False}
    for reason, accepted in (("accepted", True), ("decision_limit", False)):
        (got,) = W.original_finals([{**row, "termination_reason": reason}], "math")
        assert got["x0_id"] == "q" and got["accepted"] is accepted
    with pytest.raises(W.ExtDomainError, match="not one of"):
        W.original_finals([{**row, "termination_reason": "stopped"}], "math")
    with pytest.raises(W.ExtDomainError, match="already has"):
        W.original_finals([{**row, "x0_id": "q", "termination_reason": "accepted"}], "math")
    assert W.original_finals([{"x0_id": "x", "accepted": True}], "detox") == [
        {"x0_id": "x", "accepted": True}
    ]


def test_the_entry_needs_run_directories():
    with pytest.raises(SystemExit):
        W.main(["--domain", "detox"])


# read at import: tests/test_analysis.py::test_step11_and_step12_refuse_while_not_frozen leaves
# paths.RUNS pointing at its tmp directory (two MonkeyPatch objects on the same attribute)
REAL_RUNS = paths.RUNS
REFERENCE = REAL_RUNS / "ext_analysis_20261101T004507Z"
DETOX_RUNS = {
    "exp1_e_incorrect": ["exp1ext_e_incorrect_20261031T112435Z"],
    "exp1_e_correct": ["exp1ext_e_correct_20261031T112435Z"],
    "exp2_e_incorrect": [
        "exp2ext_e_incorrect_bpihat_20261031T152908Z",
        "exp2ext_e_incorrect_verdict_20261031T152908Z",
        "exp2ext_e_incorrect_fill_20261031T152908Z",
        "exp2ext_e_incorrect_nv_20261031T112435Z",
    ],
    "exp2_e_correct": [
        "exp2ext_e_correct_verifier_20261031T152908Z",
        "exp2ext_e_correct_nv_20261031T112435Z",
    ],
}
HAVE_DETOX = (REFERENCE / "report.json").is_file() and all(
    (REAL_RUNS / d).is_dir() for ds in DETOX_RUNS.values() for d in ds
)


@pytest.mark.skipif(not HAVE_DETOX, reason="the detox extension runs are not here")
def test_detox_report_is_the_tagged_scripts_byte_for_byte(tmp_path: Path):
    ext = W.load_domain_ext("detox")
    seed = component_seed(
        W.domain_master_seed("detox"), ext["analysis"]["bootstrap"]["seed_component"]
    )
    runs = {k: [REAL_RUNS / d for d in ds] for k, ds in DETOX_RUNS.items()}
    report = W.analyze("detox", ext, seed=seed, base=REAL_RUNS, **runs)
    files = W.write_outputs(report, tmp_path)
    for f in files:
        assert f.read_bytes() == (REFERENCE / f.name).read_bytes(), f.name
