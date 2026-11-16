"""Cross-run and self-consistent corruption must fail before any resumed controller call."""

import json
import shutil
from dataclasses import asdict

import pytest
from pipelines.math_ext_loop import validate_round_chain
from tests.test_math_ext_exp2 import Scorer, harness, run

from rafr.contracts.schemas import Arm
from rafr.domains.math.storage import IntegrityError, file_hash
from rafr.experiments.exp2 import Trajectory
from rafr.logging.writer import read_jsonl


def rewrite_round(directory, name, rows):
    """Recompute the round's own hashes: this deliberately defeats hash-only validation."""
    path = directory / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    marker_path = directory / "COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["files"][name] = file_hash(path)
    if name == "trajectories.jsonl":
        marker["n_trajectories"] = len(rows)
    if name == "decisions.jsonl":
        marker["n_decisions"] = len(rows)
    marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")


def initial_and_options(setup):
    initial = [
        Trajectory(s.x0_id, s.x0, a.value, 1, s.text)
        for s in setup.starts
        for a in setup.plan[s.x0_id]
    ]
    return initial, {
        "run_id": setup.run_id,
        "seed": setup.run_seed,
        "smoke": setup.smoke,
        "taus": setup.taus,
        "verifier_ids": setup.verifier_ids,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "another-run"),
        ("experiment", "exp1"),
        ("seed", 999),
        ("x0_id", "another-question"),
        ("state_id", "other-state"),
        ("x_t", "other-input"),
        ("t", 2),
        ("lam", "low"),
        ("tau1", 0.8),
        ("s1", 0.9),
        ("s1_tilde", 0.9),
        ("v1_id", "wrong-v"),
        ("e_label", "CORRECT"),
        ("c_t", [1, 1]),
        ("r_ref", 0.2),
        ("permuted_from_state_id", "donor"),
        ("action", "rewrite"),
        ("parse_fail", True),
        ("n_parse_retries", 1),
        ("model", "wrong-model"),
        ("prompt_hash", "wrong-prompt"),
    ],
)
def test_resealed_foreign_decision_is_rejected_before_calls(tmp_path, field, value):
    h = harness(tmp_path)
    run(h)
    directory = h[0].run_dir / "rounds/round_1"
    rows = list(read_jsonl(directory / "decisions.jsonl"))
    rows[0][field] = value
    rewrite_round(directory, "decisions.jsonl", rows)
    before = len(h[2].calls)
    with pytest.raises(IntegrityError):
        run(h)
    assert len(h[2].calls) == before
    assert list(read_jsonl(directory / "decisions.jsonl"))[0][field] == value


@pytest.mark.parametrize(
    "field,value",
    [
        ("x0_id", "other"),
        ("arm", "no_verifier"),
        ("x0", "different problem"),
        ("x_t", "different text"),
        ("accepted", False),
        ("accepted_at", 3),
        ("n_rewrites", 2),
        ("n_parse_fail", 1),
        ("t", 2),
        ("x_t_refusal_like", True),
        ("history", []),
    ],
)
def test_resealed_trajectory_changes_are_not_valid_resume(tmp_path, field, value):
    h = harness(tmp_path)
    run(h)
    directory = h[0].run_dir / "rounds/round_1"
    rows = list(read_jsonl(directory / "trajectories.jsonl"))
    rows[0][field] = value
    rewrite_round(directory, "trajectories.jsonl", rows)
    before = len(h[2].calls)
    with pytest.raises(IntegrityError):
        run(h)
    assert len(h[2].calls) == before


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "reorder"])
def test_exact_plan_is_required_even_when_manifest_count_matches(tmp_path, mutation):
    h = harness(tmp_path)
    run(h)
    directory = h[0].run_dir / "rounds/round_2"
    rows = list(read_jsonl(directory / "trajectories.jsonl"))
    rows = (
        rows[:-1]
        if mutation == "drop"
        else rows + [rows[0]]
        if mutation == "duplicate"
        else rows[::-1]
    )
    rewrite_round(directory, "trajectories.jsonl", rows)
    initial, options = initial_and_options(h[0])
    with pytest.raises(IntegrityError, match="plan"):
        validate_round_chain(h[0].run_dir, initial, require_complete=True, **options)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unplanned", "value"])
def test_score_stream_must_join_decisions_exactly(tmp_path, mutation):
    h = harness(tmp_path)
    run(h)
    directory = h[0].run_dir / "rounds/round_1"
    rows = list(read_jsonl(directory / "scores.jsonl"))
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "unplanned":
        rows[0]["state_id"] = "unplanned"
    else:
        rows[0]["score"] = 0.9
    rewrite_round(directory, "scores.jsonl", rows)
    with pytest.raises(IntegrityError):
        run(h)


def test_complete_round_copied_from_other_run_is_rejected(tmp_path):
    target, donor = harness(tmp_path / "target"), harness(tmp_path / "donor")
    donor[0].run_id = "other-run"
    run(target)
    run(donor)
    directory = target[0].run_dir / "rounds/round_1"
    shutil.rmtree(directory)
    shutil.copytree(donor[0].run_dir / "rounds/round_1", directory)
    before = len(target[2].calls)
    with pytest.raises(IntegrityError, match="different run/state"):
        run(target)
    assert len(target[2].calls) == before


@pytest.mark.parametrize("field", ["seed", "prompt_hash", "parsed", "attempt"])
def test_saved_attempt_is_replayed_against_request_and_parser(tmp_path, field):
    h = harness(tmp_path)
    run(h)
    directory = h[0].run_dir / "rounds/round_1"
    rows = list(read_jsonl(directory / "decisions.jsonl"))
    rows[0]["attempts"][0][field] = {
        "seed": 77,
        "prompt_hash": "changed",
        "parsed": {},
        "attempt": 2,
    }[field]
    rewrite_round(directory, "decisions.jsonl", rows)
    with pytest.raises(IntegrityError):
        run(h)


@pytest.mark.parametrize("mode", ["accept", "rewrite", "parse_fail", "overlength", "no_verifier"])
def test_valid_chain_and_complete_resume_make_no_new_calls(tmp_path, mode):
    def responder(req):
        return {
            "accept": '{"action":"accept"}',
            "rewrite": '{"action":"rewrite","revised_text":"next"}',
            "parse_fail": "bad",
            "overlength": '{"action":"rewrite","revised_text":"next"}',
            "no_verifier": '{"action":"accept"}',
        }[mode]

    h = harness(
        tmp_path,
        arms=(Arm.NO_VERIFIER,) if mode == "no_verifier" else (Arm.A1, Arm.B),
        responder=responder,
        scorer=Scorer(overlong=mode == "overlength"),
    )
    result = run(h)
    initial, options = initial_and_options(h[0])
    checked = validate_round_chain(h[0].run_dir, initial, require_complete=True, **options)
    assert [asdict(x) for x in checked] == [asdict(x) for x in result]
    before = len(h[2].calls)
    assert [asdict(x) for x in run(h)] == [asdict(x) for x in result]
    assert len(h[2].calls) == before


def test_terminated_trajectory_cannot_reactivate_in_later_round(tmp_path):
    h = harness(tmp_path)
    run(h)
    directory = h[0].run_dir / "rounds/round_2"
    rows = list(read_jsonl(directory / "trajectories.jsonl"))
    rows[0]["accepted"] = False
    rewrite_round(directory, "trajectories.jsonl", rows)
    with pytest.raises(IntegrityError, match="terminated"):
        run(h)


def test_semantic_audit_does_not_require_complete_chain_for_valid_partial_resume(tmp_path):
    h = harness(tmp_path, responder=lambda req: '{"action":"rewrite","revised_text":"next"}')
    run(h)
    for t in (2, 3, 4):
        shutil.rmtree(h[0].run_dir / f"rounds/round_{t}")
    initial, options = initial_and_options(h[0])
    assert all(tr.t == 2 for tr in validate_round_chain(h[0].run_dir, initial, **options))
    with pytest.raises(IntegrityError, match="all four"):
        validate_round_chain(h[0].run_dir, initial, require_complete=True, **options)
    assert len(run(h)) == 2


@pytest.mark.parametrize("mutation", ["too_short", "wrong_round", "unknown_verifier", "history"])
def test_overlength_termination_cannot_hide_unrelated_state_changes(tmp_path, mutation):
    h = harness(
        tmp_path,
        responder=lambda req: '{"action":"rewrite","revised_text":"next"}',
        scorer=Scorer(overlong=True),
    )
    run(h)
    directory = h[0].run_dir / "rounds/round_2"
    rows = list(read_jsonl(directory / "trajectories.jsonl"))
    detail = rows[0]["history"][-1]["v_input_overlength"]
    if mutation == "too_short":
        detail["tokens"] = detail["L_in"]
    elif mutation == "wrong_round":
        detail["t"] = 3
    elif mutation == "unknown_verifier":
        detail["verifier_id"] = "other-verifier"
    else:
        rows[0]["history"][0]["x_t"] = "different initial solution"
    rewrite_round(directory, "trajectories.jsonl", rows)
    with pytest.raises(IntegrityError):
        run(h)


def test_decision_cannot_be_added_after_acceptance(tmp_path):
    h = harness(tmp_path)
    run(h)
    rows = list(read_jsonl(h[0].run_dir / "rounds/round_1/decisions.jsonl"))
    rows[0]["t"] = 2
    rows[0]["state_id"] = "q1-A1-t2"
    rewrite_round(h[0].run_dir / "rounds/round_2", "decisions.jsonl", rows[:1])
    with pytest.raises(IntegrityError, match="outside active"):
        run(h)


def test_no_verifier_cannot_acquire_score_stream(tmp_path):
    h = harness(tmp_path, arms=(Arm.NO_VERIFIER,))
    run(h)
    row = {
        "state_id": "q1-no_verifier-t1",
        "verifier_id": "v1",
        "model_version": "v1",
        "score": 0.5,
    }
    rewrite_round(h[0].run_dir / "rounds/round_1", "scores.jsonl", [row])
    with pytest.raises(IntegrityError, match="outside active"):
        run(h)
