"""Math Exp 1's single-round smoke (pipelines/math_exp1_smoke.py; 2026-10-15, II.4.4 /
M.10) on the offline stub run: ten cells at lambda = high with all three arms, Exp 1's own
prompts under the smoke's seed namespace, engineering checks only, and Exp 1's live release."""

from __future__ import annotations

import json

import pytest

from math_v2_helpers import make_harness, prepare_gate
from rafr.domains.math.storage import IntegrityError, atomic_json, read_json
from rafr.logging.writer import read_jsonl


@pytest.fixture
def harness(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch, experiment_n=12)
    prepare_gate(h)
    return h


def test_ten_cells_at_high_with_every_arm_and_no_defect(harness):
    before = len(harness.provider.calls)
    report = harness.pipeline.exp1_smoke()
    out = harness.root / "exp1_smoke"
    assert report["SMOKE"] is True and report["is_research_result"] is False
    assert report["n_cells"] == 10 and report["level"] == "high"
    assert report["arms"] == {"A1": 4, "B": 3, "permuted_B": 3}
    assert report["engineering_defects"] == [], report
    assert report["prompt_audit"]["n_prompts"] >= 10
    assert sum(report["decisions"].values()) == 10
    assert report["outcome"]["n"] == 12 and report["outcome"]["n_incorrect"] == 12
    assert read_json(out / "outcome_hash.json") == report["outcome"]
    # one call per cell, all live, accounted in the run's ledger
    assert len(harness.provider.calls) - before == report["cost"]["live_calls"] == 10
    assert report["cells_done_before_this_invocation"] == 0
    assert report["cost"]["live_attempts_logged"] == 10
    assert report["cost"]["problems"] == [] and report["cost"]["input_tokens"] > 0
    rows = list(read_jsonl(out / "decisions.jsonl"))
    assert {r["experiment"] for r in rows} == {"dry_run"} and {r["lam"] for r in rows} == {"high"}
    # no arm comparison in the report (M.10)
    text = json.dumps(report)
    assert "FA" not in report and "accept_rate" not in text
    # the smoke's prompts are Exp 1's (the hash leaves the seed out), its seeds are its own
    harness.pipeline.exp1()
    formal = {
        (r["state_id"], r["arm"], r["lam"]): r
        for r in read_jsonl(harness.root / "exp1/decisions.jsonl")
    }
    for r in rows:
        f = formal[r["state_id"], r["arm"], r["lam"]]
        assert f["prompt_hash"] == r["prompt_hash"]
        assert f["attempts"][0]["seed"] != r["attempts"][0]["seed"]
    assert read_json(harness.root / "exp1/outcome_hash.json") == report["outcome"]


def test_rerun_resumes_and_a_tampered_log_is_an_engineering_defect(harness):
    first = harness.pipeline.exp1_smoke()
    before = len(harness.provider.calls)
    again = harness.pipeline.exp1_smoke()
    assert len(harness.provider.calls) == before  # every cell done: no call
    # the cost is still this smoke's own ten calls, found in the ledger by cache key
    assert again["cells_done_before_this_invocation"] == 10
    assert again["cost"] == first["cost"] and again["cost"]["live_calls"] == 10
    assert again["engineering_defects"] == []
    path = harness.root / "exp1_smoke" / "decisions.jsonl"
    rows = list(read_jsonl(path))
    rows[0]["attempts"][0]["prompt_hash"] = "0" * 64
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    broken = harness.pipeline.exp1_smoke()
    assert any(d.startswith("prompt_mismatch:") for d in broken["engineering_defects"])


def test_exp1_is_released_by_the_single_round_smoke_or_the_exp2_smoke(harness):
    pipeline = harness.pipeline
    with pytest.raises((IntegrityError, FileNotFoundError, OSError)):
        pipeline.require_smoke_for_exp1()  # no smoke at all
    pipeline.exp1_smoke()
    out = harness.root / "exp1_smoke"
    report = read_json(out / "smoke_report.json")
    with pytest.raises(IntegrityError):  # an offline stub smoke is not a live one
        pipeline.require_smoke_for_exp1()
    atomic_json(out / "smoke_report.json", {**report, "execution_mode": "live"})
    assert pipeline.require_smoke_for_exp1()["n_cells"] == 10
    for change in (
        {"engineering_defects": ["prompt_mismatch:x:0"]},
        {"arms": {"A1": 10}},
        {"level": "low"},
        {"profile_hash": "other"},
        {"decisions_sha256": "0" * 64},
    ):
        atomic_json(out / "smoke_report.json", {**report, "execution_mode": "live", **change})
        with pytest.raises(IntegrityError):
            pipeline.require_smoke_for_exp1()
    # an Exp 2 smoke report on disk rules instead (and this fixture has none that passes)
    exp2 = harness.root / "exp2_smoke"
    exp2.mkdir()
    atomic_json(exp2 / "smoke_report.json", {"SMOKE": True})
    atomic_json(out / "smoke_report.json", {**report, "execution_mode": "live"})
    with pytest.raises(IntegrityError):
        pipeline.require_smoke_for_exp1()


def test_exp1_report_before_exp2_equals_the_full_reports_exp1_parts(harness):
    """2026-10-15: Exp 1's preregistered report when Exp 2 does not exist yet."""
    import random

    from pipelines import math_step10 as s10
    from pipelines.math_diagnostics import exp1_outcome_diagnostics, protocol_outcome_diagnostics

    pipeline = harness.pipeline
    pipeline.exp1()
    report = pipeline.report_exp1()
    assert read_json(harness.root / "exp1" / "report_exp1.json") == json.loads(json.dumps(report))
    assert report["exp1"]["n_questions"] == 12 and report["exp1"]["n_rows"] == 108
    assert "h2_reference_r_ref" in report["exp1_diagnostics"]
    assert {"exp1_drift", "error_boundary_alignment", "result_shape"} <= set(report)
    assert "exp2" not in report and "transport" not in report
    rows = pipeline._exp1_rows()  # noqa: SLF001
    gate_r = read_json(harness.root / "gate_1.json")["r"]
    qids = sorted({r["question_id"] for r in rows})
    trajectories = s10.fake_trajectories([{"x0_id": q} for q in qids], (0.2, 0.2), random.Random(3))
    full = protocol_outcome_diagnostics(rows, trajectories, gate_r, n_boot=100, seed=1)
    alone = exp1_outcome_diagnostics(rows, gate_r, n_boot=100, seed=1)
    assert alone["exp1_drift"] == full["exp1_drift"]
    assert alone["result_shape"] == full["result_shape"]
    assert alone["error_boundary_alignment"]["exp1"] == full["error_boundary_alignment"]["exp1"]
