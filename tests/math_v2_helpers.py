"""Deterministic external boundaries for paper Sec. 5, App. D integration tests only.

The production pipeline, Core controller, cache, ledger, labels, channel, formatter,
round storage and analysis are exercised. No model/API/GPU is used here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pipelines.math_pipeline import MathPipeline

from rafr.contracts.schemas import Split
from rafr.domains.math.config import MathRunConfig
from rafr.domains.math.dataset import DatasetBundle, PrivateReference, PublicQuestion
from rafr.domains.math.grading import GradeResult, extract_final_box
from rafr.domains.math.storage import digest
from rafr.formatter import math_feedback
from rafr.llm.cache import LLMCache
from rafr.llm.client import LLMClient, content_text
from rafr.llm.ledger import Ledger
from rafr.llm.providers.fake import FakeProvider, FakeReply


def q_number(problem):
    return int(re.search(r"Fixture Q(\d+):", problem)[1])


def initial_solution(number):
    value = number + 2 + (1000 if number % 2 else 0)
    return rf"Initial calculation. \boxed{{{value}}}"


class ArithmeticGrader:
    def __init__(self, events=None, faults=None):
        self.events = [] if events is None else events
        self.faults = faults or {}
        self.calls = []

    def configuration(self):
        return {
            "contract_version": "fixture-M-v2",
            "math_verify_version": "0.9.0",
            "execution_mode": "offline_arithmetic_fixture",
        }

    def cache_key(self, problem, reference, candidate):
        return digest([problem, reference, candidate, self.configuration()])

    def grade(self, problem, reference, candidate):
        number = q_number(problem)
        self.calls.append((problem, reference, candidate))
        self.events.append(("grade", number, candidate))
        if number in self.faults:
            # A fault is a status, or (status, detail) to mimic MathGrader's detail string.
            fault = self.faults[number]
            status, detail = fault if isinstance(fault, tuple) else (fault, "injected_E_fault")
            return GradeResult(status, detail)
        assert reference == f"PRIVATE_GOLD_{number:03}"
        try:
            answer, _ = extract_final_box(candidate)
            status = "correct" if int(answer) == number + 2 else "incorrect"
        except (ValueError, TypeError):
            status, answer = "unresolved", None
        return GradeResult(
            status,
            "offline_arithmetic_fixture",
            answer,
            "fixture-M-v2",
            self.cache_key(problem, reference, candidate),
        )


class ArithmeticVerifier:
    def __init__(self, index, events):
        self.index, self.events = index, events
        self.calls = []

    def score(self, problem, solution):
        assert "PRIVATE_GOLD_" not in problem + solution
        number = q_number(problem)
        self.calls.append((problem, solution))
        self.events.append(("verifier", number, solution))
        try:
            answer, _ = extract_final_box(solution)
            wrong = int(answer) != number + 2
        except (ValueError, TypeError):
            wrong = True
        score = (0.87 if wrong else 0.12) + self.index * 0.01
        return {
            "error_score": score,
            "step_scores": [1 - score],
            "execution_mode": "offline_arithmetic_fixture",
        }


class ArithmeticResponder:
    def __init__(self, events, mode="repair_accept", initial_failures=None):
        self.events, self.mode = events, mode
        self.initial_failures = initial_failures or {}
        self.crash_after = None
        self.count = 0

    def __call__(self, request):
        self.count += 1
        if self.crash_after is not None and self.count > self.crash_after:
            raise RuntimeError("injected provider interruption")
        text = "\n".join(content_text(row["content"]) for row in request.messages)
        assert "PRIVATE_GOLD_" not in text + (request.system or "")
        number = q_number(text)
        self.events.append(("provider", number, request.schema_name, request.attempt))
        if request.schema_name == "MathGenerationOutput":
            failure = self.initial_failures.get(number)
            if (
                failure == "unresolved_always"
                or failure == "unresolved_once"
                and not request.attempt
            ):
                solution = "I have not supplied an answer."
            elif failure == "refusal_once" and not request.attempt:
                return FakeReply(None, refused=True)
            else:
                solution = initial_solution(number)
            return json.dumps({"solution": solution})
        assert request.schema_name == "ControllerOutput"
        current = text.split("Current solution:\n---\n", 1)[1].split("\n---", 1)[0]
        if self.mode == "parse_fail":
            return FakeReply(None)
        if self.mode == "accept":
            return json.dumps({"action": "accept", "revised_text": ""})
        revision = re.search(r"Revision (\d+)", current)
        step = int(revision[1]) + 1 if revision else 1
        if self.mode == "repair_accept" and revision:
            return json.dumps({"action": "accept", "revised_text": ""})
        if self.mode == "terminal_unresolved":
            if revision:
                return json.dumps({"action": "accept", "revised_text": ""})
            updated = "Revision 1. I have not supplied an answer."
        else:
            value = number + 2
            if self.mode == "fourth_rewrite" and step < 4:
                value += 1000
            updated = rf"Revision {step}. \boxed{{{value}}}"
        return json.dumps({"action": "rewrite", "revised_text": updated})


@dataclass
class Harness:
    pipeline: MathPipeline
    client: LLMClient
    provider: FakeProvider
    responder: ArithmeticResponder
    grader: ArithmeticGrader
    events: list

    @property
    def root(self):
        return self.pipeline.root

    def reopen(self):
        self.pipeline = MathPipeline(
            self.pipeline.config,
            self.root,
            self.pipeline.bundle,
            client=self.client,
            verifiers=self.pipeline.verifiers,
            grader=self.grader,
        )
        return self.pipeline


def make_harness(
    tmp_path,
    monkeypatch,
    *,
    n=200,
    mode="repair_accept",
    initial_failures=None,
    e_faults=None,
    seed=20261002,
    delta_r=0.05,
    fit_n=20,
    gate_n=40,
    reserve_n=20,
    experiment_n=6,
):
    prompt_dir = Path(tmp_path) / "feedback_prompts"
    prompt_dir.mkdir(exist_ok=True)
    template = Path(math_feedback.__file__).with_name("prompts") / "math" / "a1.txt"
    (prompt_dir / "a1.txt").write_text(template.read_text())
    for name, value in {
        "checker_a.txt": "Checker A evaluates mathematical solution steps.",
        "checker_b.txt": "Checker B evaluates a complete mathematical solution.",
        "score_direction.txt": (
            "Higher scores mean the checker finds the solution more likely wrong."
        ),
    }.items():
        (prompt_dir / name).write_text(value)
    monkeypatch.setattr(math_feedback, "PROMPT_DIR", prompt_dir)
    public, private = [], {}
    for number in range(n):
        qid = f"fixture:{number:03}"
        problem = f"Fixture Q{number:03}: compute {number}+2 exactly."
        public.append(PublicQuestion(qid, problem))
        private[qid] = PrivateReference(
            qid,
            digest(problem),
            f"PRIVATE_GOLD_{number:03}",
            digest([qid, problem]),
            "unknown",
            "fixture",
            None,
        )
    bundle = DatasetBundle(tuple(public), private, {"stage": "stub"}, digest([n, "M-v2-fixture"]))
    config = MathRunConfig(
        stage="stub",
        execution_mode="offline_stub",
        source_path="fixture-only",
        bundle_path="fixture-only",
        seed=seed,
        fit_n=fit_n,
        gate_n=gate_n,
        reserve_n=reserve_n,
        experiment_n=experiment_n,
        delta_r=delta_r,
        bootstrap_n=100,
        max_initial_attempts=2,
        verifiers=[
            {"model_id": "fixture-V1", "revision": "offline-1"},
            {"model_id": "fixture-V2", "revision": "offline-2"},
        ],
    )
    events = []
    grader = ArithmeticGrader(events, e_faults)
    responder = ArithmeticResponder(events, mode, initial_failures)
    provider = FakeProvider(responder, model_version="offline-fixture-not-a-live-model")
    client = LLMClient(
        {"openai": provider},
        cache=LLMCache(Path(tmp_path) / "cache"),
        ledger=Ledger(Path(tmp_path) / "ledger.jsonl"),
    )
    pipeline = MathPipeline(
        config,
        Path(tmp_path) / "run",
        bundle,
        client=client,
        verifiers=[ArithmeticVerifier(i, events) for i in range(2)],
        grader=grader,
    )
    return Harness(pipeline, client, provider, responder, grader, events)


def prepare_fit(harness):
    pipe = harness.pipeline
    pipe.generate()
    pipe.split()
    for split in (Split.D_FIT, Split.UNLABELED_POOL, Split.D_GATE, Split.D_EXP1):
        pipe.score(split)
    return pipe.fit()


def prepare_gate(harness):
    prepare_fit(harness)
    gate = harness.pipeline.gate()
    assert gate["passed"], f"fixed engineering-fixture ladder unexpectedly failed: {gate}"
    return gate
