"""Appendix M compatibility names and source eligibility; one Math-Verify evaluator.

The former exact-rational parser and cross-checking judge have been removed.
Closed-form eligibility is determined by the pinned evaluator, not a second CAS.
"""

from __future__ import annotations

import re

from rafr.domains.math.grading import (
    ANSWER_CONTRACT_VERSION as ANSWER_CONTRACT_VERSION,
)
from rafr.domains.math.grading import (
    GradeResult,
    MathBackendError,
    MathGrader,
    MathParseError,
    extract_final_box,
)

StrictMathGrader = MathGrader
StrictGradeResult = GradeResult
AnswerContractError = MathParseError
_APPROXIMATE_TASK = re.compile(
    r"\b(?:approximately|approximate|approximation|nearest)\b"
    r"|\b(?:round|rounded|rounding)\s+to\b"
    r"|\bdecimal\s+places\b|\bsignificant\s+(?:figures|digits)\b",
    re.I,
)


def extract_strict_final_answer(candidate: str) -> str:
    """Compatibility extraction name: the last \\boxed answer (M.9)."""
    return extract_final_box(candidate)[0]


def question_exclusion_reason(question: str) -> str | None:
    if not isinstance(question, str) or not question.strip():
        return "question_empty_or_non_string"
    if _APPROXIMATE_TASK.search(question):
        return "question_approximation_not_supported"
    return None


def reference_exclusion_reason(reference_answer: str) -> str | None:
    """M.2.1: source gold must parse and compare equal to itself using E.

    Explicit E errors stop preparation; they must not silently change which
    questions are eligible. The runtime initial-state E retries are separate.
    """
    if not isinstance(reference_answer, str) or not reference_answer.strip():
        return "reference_empty_or_non_string"
    grader = MathGrader(require_boxed_prediction=False)
    outcome = grader.grade("", reference_answer, reference_answer)
    if outcome.status in {"timeout", "backend_error"}:
        raise MathBackendError(f"source_reference_E_error:{outcome.status}")
    if outcome.status != "correct":
        return "reference_not_parseable_or_self_equivalent"
    return None
