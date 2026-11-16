"""paper Sec. 5, App. D production orchestration with Core offline external boundaries."""

from __future__ import annotations

import json
from collections import Counter

import pytest
from pipelines.math_analysis import summarize_exp2
from pipelines.math_pipeline import E_TIMEOUT_UNRESOLVED_REASON

from math_v2_helpers import make_harness, prepare_fit, prepare_gate
from rafr.contracts.schemas import LogRecord, MathLabel, Split
from rafr.data import labels
from rafr.data.splits import load_all_splits
from rafr.domains.math.storage import IntegrityError, read_json
from rafr.logging.writer import read_jsonl


def test_generation_finishes_before_core_split_and_q_excludes_exp1(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch)
    with pytest.raises(IntegrityError):
        h.pipeline.split()
    report = prepare_fit(h)
    splits = load_all_splits(h.root / "splits")
    assert set(report["q_question_ids"]) == set(splits[Split.UNLABELED_POOL])
    assert not set(report["q_question_ids"]) & set(splits[Split.D_EXP1])
    assert report["q_reads_labels"] is False
    assert len(read_json(h.root / "initial_complete.json")["files"]) == 3
    outcomes = list(read_jsonl(h.root / "initial_outcomes.jsonl"))
    assert len(outcomes) == 200 and {row["status"] for row in outcomes} == {"valid"}
    selected = labels.read_labels(
        Split.D_EXP1,
        labels.LabelPurpose.EXP1_OUTCOME,
        labels_dir=h.root / "labels",
        split_dir=h.root / "splits",
    )
    assert len(selected) == 6 and all(row.label is MathLabel.INCORRECT for row in selected)
    all_ids = [qid for ids in splits.values() for qid in ids]
    assert len(all_ids) == len(set(all_ids)) == 200
    for i in range(2):
        expected = sorted(
            h.pipeline._initial_pairs(splits[Split.UNLABELED_POOL])[j][i]
            for j in range(len(splits[Split.UNLABELED_POOL]))
        )
        assert report["channel"]["q"][f"v{i + 1}"]["scores"] == expected


def test_initial_unresolved_retries_same_prompt_fresh_seed_then_excludes(tmp_path, monkeypatch):
    h = make_harness(
        tmp_path, monkeypatch, n=8, initial_failures={0: "unresolved_always", 1: "unresolved_once"}
    )
    summary = h.pipeline.generate()
    assert summary["n_valid"] == 7 and summary["n_excluded"] == 1
    requests = [req for req in h.provider.calls if "Fixture Q000:" in str(req.messages)]
    assert len(requests) == 2
    assert requests[0].messages == requests[1].messages
    assert requests[0].prompt_hash() == requests[1].prompt_hash()
    assert requests[0].seed != requests[1].seed
    assert "temperature" not in requests[0].decoding
    outcomes = {row["question_id"]: row for row in read_jsonl(h.root / "initial_outcomes.jsonl")}
    assert outcomes["fixture:000"]["reason"] == "unresolved"
    assert outcomes["fixture:001"]["status"] == "valid"
    states = list(read_jsonl(h.root / "states.jsonl"))
    assert "fixture:000" not in {row["x0_id"] for row in states}
    assert len(list(read_jsonl(h.root / "generation.jsonl"))) == 10


@pytest.mark.parametrize("fault", ["timeout", ("timeout", "worker_wall_clock_deadline")])
def test_initial_e_error_retries_same_answer_twice_then_drops_question(
    tmp_path, monkeypatch, fault
):
    # The design is unchanged by the paper App. D.1 revision of 2026-10-08: a wall-clock timeout of
    # the first answer still excludes the question; it is not UNRESOLVED here.
    h = make_harness(tmp_path, monkeypatch, n=4, e_faults={0: fault})
    summary = h.pipeline.generate()
    assert summary["n_excluded"] == 1
    assert len([call for call in h.grader.calls if "Q000:" in call[0]]) == 3
    assert len([req for req in h.provider.calls if "Q000:" in str(req.messages)]) == 1
    assert len(list(read_jsonl(h.root / "e_errors.jsonl"))) == 3
    excluded = next(
        row for row in read_jsonl(h.root / "initial_outcomes.jsonl") if row["status"] == "excluded"
    )
    assert excluded["reason"] == "e_error"


def test_failed_first_gate_stops_without_reserve_scoring_or_controller_calls(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch, delta_r=0.25)
    result = h.pipeline.run_all()
    assert result["status"] == "GATE_FAILED_STOPPED" and result["reserve_used"] is False
    assert all(req.schema_name == "MathGenerationOutput" for req in h.provider.calls)
    history = read_json(h.root / "gate_attempts.json")
    assert history["max_attempts"] == 2 and len(history["attempts"]) == 1
    assert history["attempts"][0]["passed"] is False
    assert history["attempts"][0]["split"] == "d_gate"
    assert history["attempts"][0]["descriptive_whole_split"]
    assert not (h.root / "gate_2.json").exists()
    reserve = set(load_all_splits(h.root / "splits")[Split.D_GATE_RESERVE])
    scored = {row["state_id"] for row in read_jsonl(h.root / "scores.jsonl")}
    state_ids = {row["x0_id"]: row["state_id"] for row in read_jsonl(h.root / "states.jsonl")}
    assert scored.isdisjoint({state_ids[qid] for qid in reserve})
    with pytest.raises(IntegrityError, match="the authors"):
        h.pipeline.gate(2)
    for invalid in (
        {"reviewed_by": True, "decision": True},
        {"reviewed_by": "the authors", "decision": "   "},
        ["the authors", "use reserve"],
    ):
        with pytest.raises(IntegrityError, match="the authors"):
            h.pipeline.gate(2, reserve_decision=invalid)
    with pytest.raises(IntegrityError, match="gate not passed"):
        h.pipeline.exp1()


def test_exp1_uses_core_log_nine_cells_shared_scores_and_whole_donor_fields(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch)
    prepare_gate(h)
    result = h.pipeline.exp1()
    assert result == {"n_questions": 6, "n_decisions": 54}
    rows = list(read_jsonl(h.root / "exp1/decisions.jsonl"))
    for row in rows:
        LogRecord.model_validate(row)
    for qid in {row["x0_id"] for row in rows}:
        group = [row for row in rows if row["x0_id"] == qid]
        assert len(group) == 9 and len({row["x_t"] for row in group}) == 1
        for level in {row["lam"] for row in group}:
            cells = [row for row in group if row["lam"] == level]
            assert len({(row["s1_tilde"], row["s2_tilde"]) for row in cells}) == 1
    permutation = read_json(h.root / "exp1/permutation.json")
    for row in rows:
        if row["arm"] != "permuted_B":
            continue
        donor = permutation[row["state_id"]]
        assert donor != row["state_id"] and row["permuted_from_state_id"] == donor
        own_donor = next(
            cell
            for cell in rows
            if cell["state_id"] == donor and cell["lam"] == row["lam"] and cell["arm"] == "B"
        )
        for field in ("y1", "y2", "m1", "m2", "agreement", "conflict_direction"):
            assert row["fields"][field] == own_donor["fields"][field]


def test_smoke_is_thirty_a1_trajectories_without_any_arm_comparison(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch)
    prepare_fit(h)
    report = h.pipeline.smoke()
    assert report["SMOKE"] is True and report["n_questions"] == 30
    assert report["is_research_result"] is False
    assert set(read_json(h.root / "exp2_smoke/assignment.json").values()) == {"A1"}
    encoded = json.dumps(report)
    assert all(word not in encoded for word in ("A1-B", "TFR_A1-TFR_B", "contrast", '"B"'))
    assert not (h.root / "exp1").exists()


def test_exp2_round_synchrony_e_outside_loop_and_fourth_rewrite_final(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch, mode="fourth_rewrite")
    prepare_gate(h)
    h.events.clear()
    h.pipeline.exp2()
    assert not any(event[0] == "grade" for event in h.events)
    # Every active state is scored before the first controller call of that round.
    phases = [event[0] for event in h.events if event[0] in {"verifier", "provider"}]
    assert phases == (["verifier"] * 12 + ["provider"] * 6) * 4
    rows = list(read_jsonl(h.root / "exp2/decisions.jsonl"))
    assert [row["t"] for row in rows] == [1] * 6 + [2] * 6 + [3] * 6 + [4] * 6
    assert len(rows) == 24
    scores = list(read_jsonl(h.root / "exp2/scores.jsonl"))
    for decision in rows:
        matching = [score for score in scores if score["state_id"] == decision["state_id"]]
        assert len(matching) == 2
        by_verifier = {score["verifier_id"]: score["score"] for score in matching}
        assert by_verifier == {decision["v1_id"]: decision["s1"], decision["v2_id"]: decision["s2"]}
    h.pipeline.judge()
    trajectories = list(read_jsonl(h.root / "exp2/trajectories.jsonl"))
    assert all(row["terminal_status"] == "correct" for row in trajectories)
    assert all(row["termination_reason"] == "decision_limit" for row in trajectories)
    assert all("Revision 4" in row["terminal_solution"] for row in trajectories)
    assert len([req for req in h.provider.calls if req.schema_name == "ControllerOutput"]) == 24


def test_unresolved_rewrite_is_normal_action_scored_again_and_terminal_failure(
    tmp_path, monkeypatch
):
    h = make_harness(tmp_path, monkeypatch, mode="terminal_unresolved")
    prepare_gate(h)
    h.pipeline.exp2()
    h.pipeline.judge()
    rows = list(read_jsonl(h.root / "exp2/trajectories.jsonl"))
    assert all(row["terminal_status"] == "unresolved" and not row["task_success"] for row in rows)
    assert all(
        len(row["rounds"]) == 2 and not any(r["parse_fail"] for r in row["rounds"]) for row in rows
    )
    assert any(
        "not supplied an answer" in solution
        for verifier in h.pipeline.verifiers
        for _, solution in verifier.calls
    )


def test_exp2_e_wall_clock_timeout_after_two_retries_is_unresolved_not_a_stop(
    tmp_path, monkeypatch
):
    # paper App. D.1 in-place revision 2026-10-08.
    h = make_harness(tmp_path, monkeypatch)
    prepare_gate(h)
    h.pipeline.exp2()
    target = sorted(load_all_splits(h.root / "splits")[Split.D_EXP1])[0]
    number = int(target.split(":")[1])
    h.grader.faults[number] = ("timeout", "worker_wall_clock_deadline")
    before = len(h.grader.calls)
    h.pipeline.judge()
    # The first answer is E's cached step 4 record; only the rewrite reaches E, three times.
    assert len([call for call in h.grader.calls[before:] if f"Q{number:03}:" in call[0]]) == 3
    trajectories = {
        row["question_id"]: row for row in read_jsonl(h.root / "exp2/trajectories.jsonl")
    }
    row = trajectories[target]
    assert row["terminal_status"] == "unresolved"
    assert row["task_success"] is False and row["is_correct"] is None
    first = row["rounds"][0]
    assert first["input_grade"]["status"] == "incorrect" and not first["input_grade"]["e_timeout"]
    assert first["output_grade"]["status"] == "unresolved" and first["output_grade"]["e_timeout"]
    others = [trajectory for qid, trajectory in trajectories.items() if qid != target]
    assert others and all(trajectory["terminal_status"] == "correct" for trajectory in others)
    assert not any(
        r[field]["e_timeout"]
        for trajectory in others
        for r in trajectory["rounds"]
        for field in ("input_grade", "output_grade")
    )
    errors = list(read_jsonl(h.root / "e_errors.jsonl"))
    assert len(errors) == 3
    assert {(e["question_id"], e["status"], e["detail"]) for e in errors} == {
        (target, "timeout", "worker_wall_clock_deadline")
    }
    records = [
        e
        for e in read_jsonl(h.root / "evaluations.jsonl")
        if e["reason"] == E_TIMEOUT_UNRESOLVED_REASON
    ]
    assert len(records) == 1
    assert records[0]["question_id"] == target
    assert records[0]["label"] == MathLabel.UNRESOLVED.value and records[0]["unresolved"] is True
    # A rerun reads the recorded label: no E call, identical judgments.
    judged = (h.root / "exp2/trajectories.jsonl").read_bytes()
    calls = len(h.grader.calls)
    h.pipeline.judge()
    assert len(h.grader.calls) == calls
    assert (h.root / "exp2/trajectories.jsonl").read_bytes() == judged
    # Counted as a task failure, in the design's unresolved rate and in sensitivity analysis (c).
    summary = summarize_exp2(list(trajectories.values()), n_boot=20, seed=0)
    arm = row["arm"]
    assert summary["complete"] is True and summary["n_terminal_unresolved"] == 1
    assert summary["primary"]["by_arm"][arm]["n_unresolved"] == 1
    excluded = summary["sensitivities"]["terminal_unresolved_excluded"]["by_arm"][arm]
    assert excluded["n_excluded_by_sensitivity"] == 1
    assert summary["unresolved_rates"]["by_arm_round"][arm]["2"]["n_unresolved"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        "backend_error",
        ("timeout", "comparison_timeout"),
        ("backend_error", "worker_exited_without_result"),
    ],
)
def test_exp2_other_e_errors_still_stop_the_judgment_step(tmp_path, monkeypatch, fault):
    # paper Sec. 3 for every E error other than the worker wall-clock timeout (paper App. D.1,
    # 2026-10-08).
    h = make_harness(tmp_path, monkeypatch)
    prepare_gate(h)
    h.pipeline.exp2()
    target = sorted(load_all_splits(h.root / "splits")[Split.D_EXP1])[0]
    h.grader.faults[int(target.split(":")[1])] = fault
    with pytest.raises(IntegrityError, match="E_error after three attempts"):
        h.pipeline.judge()
    assert not (h.root / "exp2/judge_complete.json").exists()
    assert not any(
        e["reason"] == E_TIMEOUT_UNRESOLVED_REASON for e in read_jsonl(h.root / "evaluations.jsonl")
    )


def test_no_reference_in_any_model_or_verifier_input_and_only_e_records_have_gold(
    tmp_path, monkeypatch
):
    h = make_harness(tmp_path, monkeypatch)
    prepare_gate(h)
    h.pipeline.exp1()
    h.pipeline.exp2()
    h.pipeline.judge()
    for request in h.provider.calls:
        assert "PRIVATE_GOLD_" not in json.dumps(request.messages) + (request.system or "")
    for filename in (
        "generation.jsonl",
        "states.jsonl",
        "scores.jsonl",
        "verifier_details.jsonl",
        "exp1/decisions.jsonl",
        "exp2/decisions.jsonl",
    ):
        assert "PRIVATE_GOLD_" not in (h.root / filename).read_text()
    evaluations = list(read_jsonl(h.root / "evaluations.jsonl"))
    assert evaluations and all(
        row["reference_answer"].startswith("PRIVATE_GOLD_") for row in evaluations
    )


def test_complete_run_resume_makes_no_new_provider_calls(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch)
    first = h.pipeline.run_all()
    before = len(h.provider.calls)
    h.reopen()
    second = h.pipeline.run_all()
    assert second == first
    assert len(h.provider.calls) == before
    assert second["is_research_result"] is False
    assert {"exp1_drift", "transport", "error_boundary_alignment"} <= second.keys()
    assert second["exp1_drift"] and second["transport"] and second["error_boundary_alignment"]
    reference = second["exp1_diagnostics"]["h2_reference_r_ref"]
    assert reference["split"] == "d_gate" and reference["channel"] == "none"
    assert set(reference["by_verifier"]) == {"v1", "v2"}
    assert all(value["detection_rate"] == 1 for value in reference["by_verifier"].values())
    assert Counter(row["arm"] for row in read_jsonl(h.root / "exp2/trajectories.jsonl")) == {
        "A1": 3,
        "B": 3,
    }


def test_smoke_audits_actual_hashes_for_three_trajectories_and_all_their_rounds(
    tmp_path, monkeypatch
):
    from pipelines.math_pipeline import write_rows
    from pipelines.math_smoke import audit_smoke

    h = make_harness(tmp_path, monkeypatch, mode="fourth_rewrite")
    prepare_fit(h)
    report = h.pipeline.smoke()
    out = h.root / "exp2_smoke"
    audit = read_json(out / "prompt_audit.json")
    assert report["prompt_audit"]["n_trajectories"] == 3
    assert report["engineering_defects"] == audit["engineering_defects"] == []
    assert len(audit["sampled_question_ids"]) == 3
    assert len(audit["prompts"]) == 12
    assert all(row["actual_hash_matches"] and row["seed_matches"] for row in audit["prompts"])
    for qid in audit["sampled_question_ids"]:
        assert [row["t"] for row in audit["prompts"] if row["question_id"] == qid] == [1, 2, 3, 4]
    decisions = list(read_jsonl(out / "decisions.jsonl"))
    altered = next(row for row in decisions if row["x0_id"] == audit["sampled_question_ids"][0])
    altered["attempts"][0]["prompt_hash"] = "0" * 64
    write_rows(out / "decisions.jsonl", decisions)
    observed = audit_smoke(h.pipeline, out, list(read_jsonl(out / "trajectories.jsonl")))
    assert any(item.startswith("prompt_mismatch:") for item in observed["engineering_defects"])
    with pytest.raises(IntegrityError, match="changed"):
        h.pipeline.report(smoke=True)
