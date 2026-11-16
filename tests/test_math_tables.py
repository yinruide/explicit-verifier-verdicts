"""pipelines/math_tables.py: math's report.json laid out as detox's preregistered tables. On step
10's FAKE report from the offline stub run: every table of the section map is built, the files are
detox's (FAKE_ prefixed, FAKE inside), every number is the report's, and the formal path refuses
a FAKE report and a run whose files changed since judge."""

from __future__ import annotations

import json

import pytest
from analysis import prereg_tables
from pipelines import math_step06 as s6
from pipelines import math_step10 as s10
from pipelines import math_tables as mt

from math_v2_helpers import make_harness, prepare_gate


@pytest.fixture
def fake_report(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch)
    prepare_gate(h)
    cells = s10.exp1_cells(h.pipeline, tmp_path / "work", count_tokens=len)
    paths = s10.Step10Paths(run=h.root, step10=tmp_path / "step10")
    s6.write_rows(paths.exp1_inputs, cells)
    return s10.fake(paths, n_boot=200)


def test_the_section_map_covers_the_core_plan_and_every_builder():
    detox = [s.section for s in prereg_tables.SECTIONS]
    assert [s.section for s in mt.SECTIONS] == detox  # same plan sections as detox
    listed = [t for s in mt.SECTIONS for t in s.tables]
    assert len(listed) == len(set(listed)) and set(listed) == set(mt.BUILDERS)
    for s in mt.SECTIONS:  # every section lays out something or says why not
        assert s.tables or s.not_implemented, s.section
    # detox's table ids are kept where the analysis is the same
    detox_ids = {t for s in prereg_tables.SECTIONS for t in s.tables}
    assert detox_ids - set(listed) == {"T081_panel", "T082_semantic"}  # M.3, M.9


def test_fake_report_tables_are_the_reports_numbers(fake_report, tmp_path):
    out = tmp_path / "tables"
    n = fake_report["exp1"]["n_questions"]
    written = mt.tabulate(fake_report, out, run_dir=None, fake=True, n_exp1=n, n_per_arm=n // 2)
    names = {p.name for p in written}
    assert {"FAKE_tables.md", "FAKE_sections.json", "FAKE_analysis_full.json"} <= names
    assert all(name.startswith("FAKE_") for name in names)
    t01 = json.loads((out / "FAKE_T01_h1_direction.json").read_text(encoding="utf-8"))
    assert t01["FAKE"] is True
    main = fake_report["exp1"]["primary"]["interaction_high_minus_low"]
    assert t01["rows"][0][:5] == ["main", "A1 - B", main["estimate"], *main["ci95"]]
    contrast = json.loads((out / "FAKE_T08_tfr_contrast.json").read_text(encoding="utf-8"))
    c = fake_report["exp2"]["primary"]["contrast"]
    assert contrast["rows"][0][1:4] == [c["estimate"], *c["ci95"]]
    robust = json.loads((out / "FAKE_T09_robustness.json").read_text(encoding="utf-8"))
    rules = [r[2] for r in robust["rows"] if r[0] == "Exp 2"]
    assert rules == list(mt.EXP2_RULES)
    full = json.loads((out / "FAKE_analysis_full.json").read_text(encoding="utf-8"))
    assert full["exp2"] == json.loads(json.dumps(fake_report["exp2"]))
    md = (out / "FAKE_tables.md").read_text(encoding="utf-8")
    assert md.startswith("# FAKE") and "### FAKE T08_overlength" in md
    assert "Not computed here: math has no panel (M.3)" in md


def test_the_formal_path_refuses_a_fake_report_and_fake_needs_one(fake_report, tmp_path):
    with pytest.raises(mt.MathTablesError, match="refuse a FAKE"):
        mt.tabulate(fake_report, tmp_path / "x", run_dir=tmp_path, fake=False)
    formal = {k: v for k, v in fake_report.items() if k != "FAKE"}
    with pytest.raises(mt.MathTablesError, match="marked FAKE"):
        mt.tabulate(formal, tmp_path / "y", run_dir=None, fake=True)


def test_a_run_whose_files_changed_since_judge_is_refused(tmp_path):
    run = tmp_path / "run"
    (run / "exp1").mkdir(parents=True)
    (run / "exp2").mkdir()
    for name in ("decisions.jsonl", "channel.jsonl"):
        (run / "exp1" / name).write_text("{}\n", encoding="utf-8")
    (run / "exp2" / "trajectories.jsonl").write_text("{}\n", encoding="utf-8")
    sha = mt._sha256  # noqa: SLF001
    (run / "exp1" / "COMPLETE.json").write_text(
        json.dumps(
            {
                "decisions_sha256": sha(run / "exp1" / "decisions.jsonl"),
                "channel_sha256": sha(run / "exp1" / "channel.jsonl"),
                "n_decisions": 9,
            }
        ),
        encoding="utf-8",
    )
    (run / "exp2" / "judge_complete.json").write_text(
        json.dumps({"SMOKE": False, "files": {"trajectories.jsonl": "0" * 64}}), encoding="utf-8"
    )
    with pytest.raises(mt.MathTablesError, match="changed since judge_complete"):
        mt.check_run({}, mt.run_context(run))
