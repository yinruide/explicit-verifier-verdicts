"""Math step 10 (pipelines/math_step10.py) on the offline stub run of tests/math_v2_helpers.py:
the Exp 1 inputs are exactly the cells MathPipeline.exp1 runs, and the preregistered analyses run
on FAKE decisions and trajectories with shuffled arms."""

from __future__ import annotations

from collections import Counter

from pipelines import math_step06 as s6
from pipelines import math_step10 as s10

from math_v2_helpers import make_harness, prepare_gate
from rafr.logging.writer import read_jsonl


def _cells(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch)
    prepare_gate(h)
    cells = s10.exp1_cells(h.pipeline, tmp_path / "work", count_tokens=len)
    return h, cells


def test_exp1_inputs_are_the_cells_exp1_runs_with_its_prompts_and_live_calls(tmp_path, monkeypatch):
    h, cells = _cells(tmp_path, monkeypatch)
    before = len(h.provider.calls)
    h.pipeline.exp1()
    live_in_exp1 = len(h.provider.calls) - before
    rows = {
        (r["state_id"], r["arm"], r["lam"]): r for r in read_jsonl(h.root / "exp1/decisions.jsonl")
    }
    assert len(cells) == len(rows) == 54
    for c in cells:
        row = rows[c["state_id"], c["arm"], c["lam"]]
        assert c["prompt_hash"] == row["prompt_hash"]
        assert (c["s1"], c["s2"], c["s1_tilde"], c["s2_tilde"]) == (
            row["s1"],
            row["s2"],
            row["s1_tilde"],
            row["s2_tilde"],
        )
        assert c["permuted_from"] == row.get("permuted_from_state_id")
    # identical requests are cached as in the run: the live count is exp1's own
    assert sum(c["live"] for c in cells) == live_in_exp1
    assert all((c["prompt_tokens_approx"] is None) is not c["live"] for c in cells)
    summary = s10.summarize_inputs(cells)
    assert summary["n_states"] == 6 and summary["n_cells"] == 54
    assert summary["observed_scores_shared_across_arms_within_level"]
    assert summary["donors_never_self"]
    assert summary["live_first_attempts"] + summary["cached_first_attempts"] == 54
    assert set(summary["cells_per_arm_level"].values()) == {6}
    assert not (h.root / "step10").exists()  # nothing written into the run


def test_fake_analyses_run_on_shuffled_arms_and_are_marked_fake(tmp_path, monkeypatch):
    h, cells = _cells(tmp_path, monkeypatch)
    paths = s10.Step10Paths(run=h.root, step10=tmp_path / "step10")
    s6.write_rows(paths.exp1_inputs, cells)
    report = s10.fake(paths, n_boot=200)
    assert report["FAKE"] is True and report["is_research_result"] is False
    exp1, exp2 = report["exp1"], report["exp2"]
    assert exp1["n_questions"] == 6 and exp1["n_rows"] == 54
    for section in ("parse_fail_as_accept", "complete_case"):
        assert section in exp1["sensitivities"]
    assert {"parse_fail_as_accept", "complete_case", "terminal_unresolved_excluded"} <= set(
        exp2["sensitivities"]
    )
    assert paths.fake_report.is_file()
    # the same seed gives the same FAKE data
    again = s10.fake(paths, n_boot=200)
    assert again["exp1"] == exp1 and again["exp2"] == exp2


def test_fake_rows_keep_each_cells_inputs_and_shuffle_only_the_decisions(tmp_path, monkeypatch):
    import random

    _, cells = _cells(tmp_path, monkeypatch)
    rows = s10.fake_exp1_rows(cells, random.Random(1))
    assert len(rows) == 54 and {r["execution_mode"] for r in rows} == {"fake"}
    by_slot = Counter((r["question_id"], r["level"]) for r in rows)
    assert set(by_slot.values()) == {3}
    for r in rows:
        assert (r["action"] is None) is r["parse_fail"]
    trajectories = s10.fake_trajectories(cells, (0.2, 0.001), random.Random(2))
    assert Counter(t["arm"] for t in trajectories) == {"A1": 3, "B": 3}
    assert all(1 <= len(t["rounds"]) <= 4 for t in trajectories)
