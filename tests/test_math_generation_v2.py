"""paper App. D.1, E integration through the unchanged Core controller and LLM client."""

from __future__ import annotations

import json

import pytest

from rafr.contracts.schemas import Arm, GenerationAttempt, InvalidReason, Lambda, State
from rafr.controller.decide import DecisionLabels, decide
from rafr.domains.math.config import MathRunConfig
from rafr.domains.math.dataset import PublicQuestion
from rafr.domains.math.generation import (
    PROMPT_DIR,
    build_controller_config,
    build_generation_request,
    generate_initial,
)
from rafr.domains.math.grading import GradeResult
from rafr.llm.client import LLMClient
from rafr.llm.providers.fake import FakeProvider, FakeReply
from rafr.logging import JsonlWriter, read_jsonl

QUESTION = PublicQuestion("q1", "Solve x^2 = 2 for the positive root.")
SOLUTION = r"Taking the positive root gives \boxed{\sqrt{2}}"


def client_for(outputs):
    replies = iter(outputs)
    provider = FakeProvider(responder=lambda r: next(replies))
    return LLMClient(providers={"openai": provider}), provider


def generation_json(solution=SOLUTION):
    return json.dumps({"solution": solution})


def grade(status="correct"):
    return GradeResult(status, "offline fixture only")


def test_e_logging_failure_stops_instead_of_excluding_question():
    client, provider = client_for([generation_json()])

    def broken_log(solution):
        raise OSError("disk unavailable")

    with pytest.raises(OSError, match="disk unavailable"):
        generate_initial(client, QUESTION, broken_log, master_seed=17, max_attempts=2)
    assert len(provider.calls) == 1


def test_fixed_backbone_and_equal_task_information_for_generation_controller():
    req = build_generation_request(QUESTION, master_seed=9, attempt=0)
    cfg = build_controller_config(9, 4)
    assert req.model == cfg.model == "gpt-5.6-terra"
    assert req.decoding == cfg.decoding
    assert req.decoding["reasoning_effort"] == "medium"
    assert req.decoding["max_output_tokens"] == 16384  # paper App. E in-place revision 2026-10-04
    assert "temperature" not in req.decoding
    task = (PROMPT_DIR / "task.txt").read_text(encoding="utf-8").strip()
    assert task in req.messages[0]["content"]
    assert task in cfg.system_prompt and task in cfg.rewrite_template
    assert "rational" not in task and "variable assignment" not in task
    assert r"\boxed" in task
    assert req.schema.model_json_schema()["additionalProperties"] is False
    assert set(req.schema.model_json_schema()["properties"]) == {"solution"}


def test_math_output_cap_does_not_touch_the_core_controller_configuration():
    from rafr.core.config import load_config
    from rafr.domains.math.generation import frozen_models

    assert frozen_models()["controller"]["decoding"]["max_output_tokens"] == 16384
    # Core (detox, paper Sec. 4.6) keeps its own cap, also within this process.
    assert load_config("models")["controller"]["decoding"]["max_output_tokens"] == 8192


def test_initial_unresolved_regenerates_original_prompt_with_new_seed_and_core_logs(tmp_path):
    client, provider = client_for([generation_json("no final answer"), generation_json()])
    evaluated = []

    def grader(solution):
        evaluated.append(solution)
        return grade("unresolved" if len(evaluated) == 1 else "incorrect")

    with JsonlWriter(tmp_path / "generation.jsonl") as writer:
        result = generate_initial(
            client, QUESTION, grader, master_seed=17, max_attempts=2, writer=writer
        )
    assert result["status"] == "valid" and result["grade"].status == "incorrect"
    assert isinstance(result["state"], State)
    assert result["state"].x0 == QUESTION.problem and result["state"].text == SOLUTION
    assert result["state"].gen_attempt == 1
    assert all(isinstance(a, GenerationAttempt) for a in result["attempts"])
    assert result["attempts"][0].invalid_reason is InvalidReason.UNRESOLVED
    assert provider.calls[0].messages == provider.calls[1].messages
    assert provider.calls[0].prompt_hash() == provider.calls[1].prompt_hash()
    assert provider.calls[0].seed != provider.calls[1].seed
    assert provider.calls[0].cache_key() != provider.calls[1].cache_key()
    assert len(list(read_jsonl(tmp_path / "generation.jsonl"))) == 2


@pytest.mark.parametrize(
    "reply, reason",
    [
        (FakeReply(text=None, refused=True), InvalidReason.REFUSAL),
        (FakeReply(text=""), InvalidReason.EMPTY),
        (FakeReply(text=generation_json(), stop_reason="length"), InvalidReason.TRUNCATED),
        (FakeReply(text='{"solution": 2}'), InvalidReason.STRUCTURED_OUTPUT),
        (
            FakeReply(text='{"solution": "x", "other": "forbidden"}'),
            InvalidReason.STRUCTURED_OUTPUT,
        ),
        (FakeReply(text='{"solution": " "}'), InvalidReason.EMPTY),
    ],
)
def test_initial_invalid_kinds_retry_without_prompt_changes(reply, reason):
    client, provider = client_for([reply] * 2)
    result = generate_initial(
        client, QUESTION, lambda _: pytest.fail("E saw invalid JSON"), master_seed=3, max_attempts=2
    )
    assert result["status"] == "dropped" and result["state"] is None
    assert result["drop_reason"] == reason.value
    assert len(provider.calls) == 2
    assert provider.calls[0].messages == provider.calls[1].messages


def test_unresolved_exhaustion_is_not_incorrect():
    client, _ = client_for([generation_json("no answer")] * 2)
    result = generate_initial(
        client, QUESTION, lambda _: grade("unresolved"), master_seed=3, max_attempts=2
    )
    assert result["status"] == "dropped" and result["state"] is None
    assert result["grade"].status == "unresolved"
    assert result["drop_reason"] == "unresolved"


def test_e_error_retries_twice_then_drops_without_regenerating():
    client, provider = client_for([generation_json()])
    seen = []

    def grader(solution):
        seen.append(solution)
        return grade("timeout")

    result = generate_initial(client, QUESTION, grader, master_seed=2, max_attempts=5)
    assert len(seen) == 3 and len(provider.calls) == 1
    assert result["status"] == "dropped" and result["drop_reason"] == "e_error"
    assert result["attempts"][0].invalid_reason is InvalidReason.E_ERROR


def test_transient_e_error_can_recover_without_regeneration():
    client, provider = client_for([generation_json()])
    grades = iter([grade("backend_error"), grade("correct")])
    result = generate_initial(
        client, QUESTION, lambda _: next(grades), master_seed=2, max_attempts=5
    )
    assert result["status"] == "valid"
    assert len(result["e_attempts"]) == 2 and len(provider.calls) == 1


def test_no_reference_record_can_enter_generation_boundary():
    for key in ("reference_answer", "solution", "answer", "gold", "metadata"):
        with pytest.raises(ValueError):
            build_generation_request(
                {**QUESTION.public_payload(), key: "PRIVATE_CANARY"}, master_seed=1, attempt=0
            )


def test_private_grader_reference_does_not_reach_requests_on_retry():
    private = "PRIVATE_REFERENCE_CANARY"
    client, provider = client_for([generation_json("no final"), generation_json()])
    n = []

    def grader(solution):
        assert private  # The actual callback closes over E-only private reference data.
        n.append(solution)
        return grade("unresolved" if len(n) == 1 else "correct")

    generate_initial(client, QUESTION, grader, master_seed=5, max_attempts=2)
    assert all(private not in str(request) for request in provider.calls)


def test_math_decisions_use_core_retries_and_keep_unresolved_as_a_valid_rewrite():
    client, provider = client_for(
        [
            '{"action":"invalid","revised_text":"x"}',
            '{"action":"rewrite","revised_text":"I have not found a final answer."}',
        ]
    )
    cfg = build_controller_config(19, 4)
    labels = DecisionLabels("s1", Arm.A1, Lambda.NONE, 1, "exp2")
    decision = decide(
        client, cfg, x0=QUESTION.problem, x_t=SOLUTION, observation="checker fixture", labels=labels
    )
    assert not decision.parse_fail
    assert decision.revised_text == "I have not found a final answer."
    assert decision.n_parse_retries == 1
    assert QUESTION.problem in provider.calls[0].messages[0]["content"]
    assert provider.calls[1].messages[0]["content"].endswith(cfg.retry_prompt)
    assert provider.calls[1].seed != provider.calls[0].seed
    assert '"action":"invalid"' not in provider.calls[1].messages[0]["content"]
    assert all(req.schema.__name__ == "ControllerOutput" for req in provider.calls)


def test_core_shared_seed_arms_and_independent_experiment_names():
    client, provider = client_for(['{"action":"accept","revised_text":""}'] * 3)
    cfg = build_controller_config(19, 1)
    for arm, experiment in [(Arm.A1, "exp1"), (Arm.B, "exp1"), (Arm.A1, "exp2")]:
        decide(
            client,
            cfg,
            x0=QUESTION.problem,
            x_t=SOLUTION,
            observation="same",
            labels=DecisionLabels("s", arm, Lambda.LOW, 1, experiment),
        )
    assert provider.calls[0].seed == provider.calls[1].seed
    assert provider.calls[0].seed != provider.calls[2].seed
    assert "at most 1 decision in total" in provider.calls[0].system


def config(**changes):
    return MathRunConfig(
        **{
            "stage": "stub",
            "execution_mode": "offline_stub",
            "source_path": "unused",
            "bundle_path": "unused",
            **changes,
        }
    )


def test_math_configuration_has_no_backbone_or_decoder_escape_hatch():
    for field in (
        "generation",
        "controller",
        "model",
        "temperature",
        "checker_descriptions",
        "tau",
    ):
        with pytest.raises(ValueError):
            config(**{field: {}})
    assert config().seed is None and config().verifiers == []
    with pytest.raises(ValueError, match="seed"):
        config().assert_ready("generation")
    with pytest.raises(ValueError, match="cap"):
        config(seed=3).assert_ready("generation")
    config(seed=3, max_initial_attempts=2).assert_ready("generation")
    with pytest.raises(ValueError, match="fixed"):
        config(stage="formal", execution_mode="live", fit_n=10)


def test_attempt_cap_and_round_budget_must_be_explicit_and_valid():
    client, _ = client_for([])
    for value in (None, 0, True):
        with pytest.raises(ValueError):
            generate_initial(client, QUESTION, lambda _: grade(), master_seed=3, max_attempts=value)
    for value in (0, 2, True):
        with pytest.raises(ValueError):
            build_controller_config(3, value)
    with pytest.raises(ValueError):
        build_controller_config(None, 4)
