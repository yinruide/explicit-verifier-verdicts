"""Math G_fixed and domain prompt configuration, using the Core LLM/controller (paper App. D.1, E).

Only generation validity differs by domain. Decisions, parsing, retries and decision
logging remain in ``rafr.controller``. No backbone override exists here.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from rafr.contracts.schemas import GenerationAttempt, InvalidReason, State
from rafr.controller.decide import ControllerConfig, build_config, with_round_budget
from rafr.core import paths
from rafr.core.config import assert_frozen, load_config
from rafr.core.seeding import attempt_seed, component_seed
from rafr.domains.math.dataset import PublicQuestion, validate_public_question
from rafr.domains.math.grading import GradeResult
from rafr.generation.g_fixed import state_id_for
from rafr.llm.client import LLMClient, LLMRequest, fill_slots
from rafr.logging import JsonlWriter

PROMPT_DIR = Path(__file__).with_name("prompts")
TRUNCATION_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})
# paper App. E in-place revision 2026-10-04: output cap of math G_fixed and controller requests,
# reasoning tokens included. Core (detox, paper Sec. 4.6) keeps 8192 in configs/models.yaml. The
# paper App. D.1 probe's longest G_fixed output was 7,468 tokens; truncation is an invalid output,
# so a tight cap would systematically drop hard questions.
MAX_OUTPUT_TOKENS = 16384


class MathGenerationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    solution: StrictStr


def frozen_models() -> dict[str, Any]:
    """The Core models configuration with the math output cap applied (paper App. E).

    Backbone, effort and temperature are read from Core and drift is rejected, never
    overridden. Only ``max_output_tokens`` is replaced, on a copy: the Core value
    belongs to detox and is not consumed by math.
    """
    models = copy.deepcopy(load_config("models"))
    controller = models["controller"]
    decoding = controller["decoding"]
    expected = {
        "provider": "openai",
        "api": "chat_completions",
        "model": "gpt-5.6-terra",
        "seed_scheme": "per_state",
    }
    if any(controller.get(key) != value for key, value in expected.items()):
        raise ValueError("Core controller configuration differs from Appendix M.8")
    if decoding.get("reasoning_effort") != "medium" or decoding.get("temperature") is not None:
        raise ValueError("math requires Terra / medium / no temperature")
    if not controller.get("version") or controller["version"] == "TBD":
        raise ValueError("Core controller version must be frozen")
    decoding["max_output_tokens"] = MAX_OUTPUT_TOKENS
    return models


def _prompt(name: str) -> str:
    path = PROMPT_DIR / name
    assert_frozen({}, extra_files=[path], only_sections=[])
    return path.read_text(encoding="utf-8").strip()


def read_controller_retry_prompt() -> str:
    """Read the actual shared retry policy for both requests and frozen identities.

    Keeping only its configured path in a run identity misses edits to the file
    at that same path. The returned bytes, after the Core controller's standard
    whitespace normalization, are the policy consumed by every retry request.
    """
    path = paths.resolve(frozen_models()["controller"]["parse_retry_prompt_file"])
    assert_frozen({}, extra_files=[path], only_sections=[])
    text = path.read_text(encoding="utf-8").strip()
    if not text or text == "TBD":
        raise ValueError("Core controller retry prompt must be finalized, not blank or TBD")
    return text


def build_controller_config(master_seed: int, round_budget: int) -> ControllerConfig:
    """Supply only mathematical task text to the unchanged Core controller."""
    if type(master_seed) is not int:
        raise ValueError("an explicit integer master seed is required")
    if type(round_budget) is not int or round_budget not in (1, 4):
        raise ValueError("math decision budget is 1 for Exp 1 or 4 for Exp 2")
    models = frozen_models()
    task = _prompt("task.txt")
    cfg = build_config(
        models,
        {"controller": {"show_x0": True}},
        base_seed=component_seed(master_seed, "controller"),
        system_prompt=fill_slots(_prompt("system.txt"), {"task": task}),
        rewrite_template=fill_slots(_prompt("rewrite.txt"), {"task": task}),
        retry_prompt=read_controller_retry_prompt(),
    )
    # Core transport omits null temperatures; remove the key here as well so no
    # mathematics request contains a temperature parameter.
    cfg.decoding.pop("temperature", None)
    return with_round_budget(cfg, round_budget)


def build_generation_request(
    public_question: PublicQuestion | Mapping[str, Any], *, master_seed: int, attempt: int
) -> LLMRequest:
    """Original prompt is byte-identical across attempts; only seed/attempt change."""
    question = validate_public_question(public_question)
    if type(master_seed) is not int or type(attempt) is not int or attempt < 0:
        raise ValueError("master_seed and nonnegative attempt must be explicit integers")
    controller = frozen_models()["controller"]
    template = fill_slots(_prompt("g_fixed.txt"), {"task": _prompt("task.txt")})
    decoding = dict(controller["decoding"])
    decoding.pop("temperature", None)
    return LLMRequest(
        purpose="g_fixed",
        provider=controller["provider"],
        api=controller["api"],
        model=controller["model"],
        version=controller["version"],
        system=None,
        messages=[{"role": "user", "content": fill_slots(template, {"x0": question.problem})}],
        schema=MathGenerationOutput,
        decoding=decoding,
        network=dict(controller.get("network") or {}),
        seed=attempt_seed(
            component_seed(master_seed, "g_fixed"), question.question_id, attempt=attempt
        ),
        attempt=attempt,
    )


def generate_initial(
    client: LLMClient,
    public_question: PublicQuestion | Mapping[str, Any],
    grader_callback: Callable[[str], GradeResult],
    *,
    master_seed: int,
    max_attempts: int,
    writer: JsonlWriter | None = None,
) -> dict[str, Any]:
    """Generate one valid first answer before any randomization (paper App. D.1).

    The callback owns the private reference and E logging. Reference data is never
    passed to request construction. E errors retry the same candidate at most twice;
    unresolved is a generation failure and regenerates with a fresh seed.
    """
    question = validate_public_question(public_question)
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be explicitly preregistered before generation")
    if not callable(grader_callback):
        raise TypeError("grader_callback must be an independent E boundary")
    attempts: list[GenerationAttempt] = []
    e_attempts: list[dict[str, Any]] = []
    final_grade = None
    for attempt in range(max_attempts):
        request = build_generation_request(question, master_seed=master_seed, attempt=attempt)
        response = client.complete(request)  # Core network retries are not generation retries.
        reason = None
        solution = None
        if response.refused:
            reason = InvalidReason.REFUSAL
        elif response.stop_reason in TRUNCATION_REASONS:
            reason = InvalidReason.TRUNCATED
        elif not response.text or not response.text.strip():
            reason = InvalidReason.EMPTY
        elif response.error:
            reason = InvalidReason.STRUCTURED_OUTPUT
        else:
            try:
                solution = MathGenerationOutput.model_validate_json(response.text).solution
            except (ValidationError, ValueError):
                reason = InvalidReason.STRUCTURED_OUTPUT
            else:
                if not solution.strip():
                    reason = InvalidReason.EMPTY
        if reason is None:
            for e_attempt in range(3):
                # MathGrader represents E failures as GradeResult. Storage and
                # programming exceptions at the callback/logging boundary must
                # stop the run rather than silently exclude a question.
                final_grade = grader_callback(solution)
                if not isinstance(final_grade, GradeResult):
                    raise TypeError("grader_callback must return the shared GradeResult")
                e_attempts.append(
                    {"generation_attempt": attempt, "attempt": e_attempt, "grade": final_grade}
                )
                if final_grade.status not in ("timeout", "backend_error"):
                    break
            if final_grade.status == "unresolved":
                reason = InvalidReason.UNRESOLVED
            elif final_grade.status in ("timeout", "backend_error"):
                reason = InvalidReason.E_ERROR
            elif final_grade.status not in ("correct", "incorrect"):
                raise ValueError("unknown E result; cannot construct a state")
        record = GenerationAttempt(
            x0_id=question.question_id,
            attempt=attempt,
            seed=request.seed,
            raw_output=response.text,
            provider_refused=response.refused,
            valid=reason is None,
            invalid_reason=reason,
            model=request.model,
            model_version=response.model_version,
            prompt_hash=request.prompt_hash(),
            decoding=request.decoding,
            cached=response.cached,
        )
        attempts.append(record)
        if writer is not None:
            writer.write(record)
        if reason is None:
            state = State(
                state_id=state_id_for(question.question_id),
                x0_id=question.question_id,
                x0=question.problem,
                text=solution,
                t=1,
                source="g_fixed",
                gen_seed=request.seed,
                gen_attempt=attempt,
            )
            return {
                "generation_status": "valid",
                "status": "valid",
                "state": state,
                "candidate_solution": solution,
                "grade": final_grade,
                "attempts": attempts,
                "e_attempts": e_attempts,
                "drop_reason": None,
            }
        if reason is InvalidReason.E_ERROR:
            break  # This question is excluded before splitting; do not keep sampling G.
    return {
        "generation_status": "dropped",
        "status": "dropped",
        "state": None,
        "candidate_solution": None,
        "grade": final_grade,
        "attempts": attempts,
        "e_attempts": e_attempts,
        "drop_reason": attempts[-1].invalid_reason.value,
    }
