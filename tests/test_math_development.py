"""Math-specific generation and controller behavior through Core boundaries."""

from __future__ import annotations

import json

import pytest

from math_v2_helpers import make_harness, prepare_gate
from rafr.domains.math.data import MathDataError
from rafr.domains.math.generation import generate_initial
from rafr.domains.math.grading import MathGrader
from rafr.llm.cache import LLMCache
from rafr.llm.client import LLMClient
from rafr.llm.providers.fake import FakeProvider
from rafr.logging.writer import read_jsonl


def test_real_math_verify_in_initial_generation_accepts_nonrational_closed_form(tmp_path):
    answers = ["I have not finished.", r"The result is \boxed{2\sqrt{2}}."]
    provider = FakeProvider(lambda request: json.dumps({"solution": answers[request.attempt]}))
    client = LLMClient({"openai": provider}, cache=LLMCache(tmp_path / "cache"))
    grader = MathGrader(timeout_seconds=10)
    result = generate_initial(
        client,
        {"question_id": "sqrt-eight", "problem": "Compute the square root of eight."},
        lambda text: grader.grade("Compute the square root of eight.", r"\sqrt8", text),
        master_seed=17,
        max_attempts=2,
    )
    assert result["generation_status"] == "valid"
    assert result["grade"].status == "correct"
    assert result["grade"].math_verify_version == "0.9.0"
    assert result["attempts"][0].invalid_reason.value == "unresolved"
    assert len(provider.calls) == 2
    assert provider.calls[0].messages == provider.calls[1].messages
    assert provider.calls[0].seed != provider.calls[1].seed


def test_gold_bearing_public_mapping_rejected_before_any_model_request(tmp_path):
    provider = FakeProvider()
    with pytest.raises(MathDataError, match="private fields"):
        generate_initial(
            LLMClient({"openai": provider}),
            {"question_id": "q", "problem": "Compute.", "reference_answer": "private"},
            lambda text: pytest.fail("E should not be invoked"),
            master_seed=1,
            max_attempts=2,
        )
    assert provider.calls == []


def test_parse_fail_keeps_candidate_and_consumes_four_decisions_with_core_retries(
    tmp_path, monkeypatch
):
    h = make_harness(tmp_path, monkeypatch, mode="parse_fail")
    prepare_gate(h)
    h.pipeline.exp2()
    h.pipeline.judge()
    decisions = list(read_jsonl(h.root / "exp2/decisions.jsonl"))
    assert len(decisions) == 24
    assert all(row["parse_fail"] and len(row["attempts"]) == 3 for row in decisions)
    assert all(row["n_parse_retries"] == 2 for row in decisions)
    trajectories = list(read_jsonl(h.root / "exp2/trajectories.jsonl"))
    assert all(row["terminal_status"] == "incorrect" for row in trajectories)
    for trajectory in trajectories:
        assert len(trajectory["rounds"]) == 4
        assert all(row["input_grade"] == row["output_grade"] for row in trajectory["rounds"])
    requests = [req for req in h.provider.calls if req.schema_name == "ControllerOutput"]
    assert len(requests) == 72
    base, retry1, retry2 = requests[:3]
    assert base.messages != retry1.messages
    assert retry1.messages == retry2.messages
    assert len({req.seed for req in (base, retry1, retry2)}) == 3
    assert all("temperature" not in req.decoding for req in requests)


def test_first_round_acceptance_has_no_later_controller_decision(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch, mode="accept")
    prepare_gate(h)
    h.pipeline.exp2()
    h.pipeline.judge()
    decisions = list(read_jsonl(h.root / "exp2/decisions.jsonl"))
    assert len(decisions) == 6 and {row["t"] for row in decisions} == {1}
    assert all(row["action"] == "accept" for row in decisions)
    trajectories = list(read_jsonl(h.root / "exp2/trajectories.jsonl"))
    assert all(row["termination_reason"] == "accepted" for row in trajectories)
    assert all(row["terminal_status"] == "incorrect" for row in trajectories)


@pytest.mark.parametrize(
    "candidate,expected,all_equivalent",
    [
        (r"An intermediate value is \boxed{1}. The final answer is \boxed{2}.", "correct", False),
        # M.9 (2026-10-03): a self-contradiction is judged by its last box, not arbitrated.
        (r"The final answer is \boxed{3}. The final answer is \boxed{2}.", "correct", False),
        (r"The final answer is \boxed{4/2}. The final answer is \boxed{2}.", "correct", True),
    ],
)
def test_final_answer_claims_do_not_confuse_intermediate_boxes(candidate, expected, all_equivalent):
    result = MathGrader(timeout_seconds=10).grade("Compute one plus one.", "2", candidate)
    assert result.status == expected
    assert result.n_boxes == 2 and result.boxes_all_equivalent is all_equivalent
