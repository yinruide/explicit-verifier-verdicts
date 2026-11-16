"""Failure-aware answer grading; fake backends perform no model/network calls."""

from __future__ import annotations

import importlib.metadata
import time
from dataclasses import dataclass
from fractions import Fraction

import pytest

from rafr.domains.math.grading import (
    MathBackendError,
    MathGrader,
    MathParseError,
    MathVerifyAdapter,
)


@dataclass
class NumericBackend:
    def compare(self, reference_answer, prediction_answer):
        try:
            return Fraction(reference_answer) == Fraction(prediction_answer)
        except ValueError as exc:
            raise MathParseError("invalid_numeric_expression") from exc


@dataclass
class FailureBackend:
    mode: str

    def compare(self, reference_answer, prediction_answer):
        if self.mode == "timeout":
            raise TimeoutError("explicit_timeout")
        if self.mode == "slow":
            time.sleep(10)
        if self.mode == "bug":
            raise RuntimeError("not a wrong answer")
        return "not_a_boolean"


def test_grader_only_compares_final_answer_and_preserves_three_valued_correctness():
    grader = MathGrader(NumericBackend())
    correct = grader.grade("1/2", r"There are 12 cases. Final: \boxed{0.5}.")
    assert correct.status == "correct" and correct.is_correct is True
    wrong = grader.grade("1/2", r"\boxed{3/4}")
    assert wrong.status == "incorrect" and wrong.is_correct is False
    parse = grader.grade("1/2", "I considered 1/2 but did not finish.")
    assert parse.status == "unresolved" and parse.is_correct is None
    assert grader.grade("not numeric", r"\boxed{2}").status == "unresolved"


def test_unfinished_final_box_cannot_fall_back_to_an_earlier_answer():
    grader = MathGrader(NumericBackend())
    assert grader.grade("2", r"\boxed{2}; final \boxed{3").status == "unresolved"


@dataclass
class DiagnosticFailureBackend:
    """Fails only when an earlier box ("7") is compared with the last box."""

    mode: str

    def compare(self, reference_answer, prediction_answer):
        if prediction_answer == "7":
            if self.mode == "slow":
                time.sleep(10)
            raise RuntimeError("diagnostic comparison failed")
        return Fraction(reference_answer) == Fraction(prediction_answer)


@pytest.mark.parametrize("mode", ["bug", "slow"])
def test_failed_box_diagnostic_never_changes_the_verdict(mode):
    grader = MathGrader(DiagnosticFailureBackend(mode), timeout_seconds=1.0)
    before = time.monotonic()
    result = grader.grade("2", r"\boxed{7} then \boxed{2}")
    assert result.status == "correct" and result.is_correct is True
    assert result.n_boxes == 2 and result.boxes_all_equivalent is None
    assert time.monotonic() - before < 8


def test_explicit_answer_only_mode():
    grader = MathGrader(NumericBackend(), require_boxed_prediction=False)
    assert grader.grade("1/2", "0.5").status == "correct"


@pytest.mark.parametrize("mode", ["timeout", "slow"])
def test_timeouts_are_visible_and_workers_are_terminated(mode):
    deadline = 0.05 if mode == "slow" else 5.0
    grader = MathGrader(FailureBackend(mode), timeout_seconds=deadline)
    before = time.monotonic()
    result = grader.grade("1", r"\boxed{1}")
    assert result.status == "timeout" and result.is_correct is None
    assert time.monotonic() - before < 8


@pytest.mark.parametrize("mode", ["bug", "invalid_return"])
def test_engineering_failure_does_not_become_incorrect(mode):
    result = MathGrader(FailureBackend(mode)).grade("q", "1", r"\boxed{1}")
    assert result.status == "backend_error" and result.is_correct is None


@pytest.mark.parametrize("deadline", [0, -1, float("inf"), float("nan"), True])
def test_invalid_deadline_is_rejected(deadline):
    with pytest.raises(ValueError):
        MathGrader(NumericBackend(), timeout_seconds=deadline)


def test_missing_or_wrong_backend_dependency_stops_before_grading(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(MathBackendError, match="required"):
        MathGrader()
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.8.0")
    with pytest.raises(MathBackendError, match="expected 0.9.0"):
        MathGrader()


@pytest.mark.parametrize(
    "gold,prediction,expected",
    [
        (r"\frac{1}{2}", r"\boxed{0.5}", "correct"),
        ("2", r"Reasoning used 2, final \boxed{3}.", "incorrect"),
        (r"\sqrt{4}", r"\boxed{2}", "correct"),
        ("2", r"\boxed{???}", "unresolved"),
    ],
)
def test_real_math_verify_offline_adapter(gold, prediction, expected):
    # This is required mathematical acceptance evidence, not an optional GPU test.
    # Missing math dependencies must fail CI instead of silently skipping the grader.
    result = MathGrader(MathVerifyAdapter(), timeout_seconds=10).grade(gold, prediction)
    assert result.status == expected


def test_pin_override_cannot_mislabel_the_evaluator_version():
    with pytest.raises(MathBackendError, match="rebuilding"):
        MathGrader(MathVerifyAdapter(expected_version="0.8.0"))


def test_real_parser_exception_is_e_error_not_unresolved(monkeypatch):
    import math_verify

    def broken(*args, **kwargs):
        raise ValueError("private expression omitted from error detail")

    monkeypatch.setattr(math_verify, "parse", broken)
    with pytest.raises(MathBackendError, match="reference_parse_exception:ValueError"):
        MathVerifyAdapter().compare("2", "2")
