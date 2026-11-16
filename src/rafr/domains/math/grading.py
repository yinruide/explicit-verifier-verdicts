"""Appendix M.9's one evaluator: pinned Math-Verify, outside the decision loop.

Only E receives the private reference. No handwritten arithmetic, second judge,
internal retry or silent conversion of E errors into incorrect answers is used.
The runner owns M.2.3's two E retries and stage-specific unresolved accounting.

Extraction (M.9, revised 2026-10-03): the last ``\\boxed{...}`` is the answer and a
solution without one is UNRESOLVED. Earlier boxes and prose such as "the answer
is/was X" never compete with it; a self-contradictory solution is judged by its
last box, without arbitration. The number of boxes and whether every earlier box
is equivalent to the last one are recorded as diagnostics that never change a label.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import multiprocessing
import re
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Literal, Protocol

MATH_VERIFY_VERSION = "0.9.0"
ANSWER_CONTRACT_VERSION = "math-verify-last-box-2026-10-03"
GradeStatus = Literal["correct", "incorrect", "unresolved", "timeout", "backend_error"]
MAX_SOLUTION_CHARACTERS = 200_000
_BOX = re.compile(r"\\(?:boxed|fbox)\b")  # source references may use either command
_BOXED = re.compile(r"\\boxed\b")  # candidate answers: \boxed only (M.9)


class MathParseError(ValueError):
    """No mathematical answer was extracted; this is UNRESOLVED, not incorrect."""


class MathBackendError(RuntimeError):
    """Dependency/API/parsing/worker exceptions are E_error, not truth labels."""


@dataclass(frozen=True)
class GradeResult:
    status: GradeStatus
    detail: str
    prediction_answer: str | None = None
    contract_version: str = ANSWER_CONTRACT_VERSION
    cache_key: str = ""
    math_verify_version: str = MATH_VERIFY_VERSION
    # Diagnostics only (M.9); they never change status. boxes_all_equivalent is None
    # below two boxes, or when an earlier box could not be read or compared in time.
    n_boxes: int | None = None
    boxes_all_equivalent: bool | None = None

    @property
    def is_correct(self) -> bool | None:
        return self.status == "correct" if self.status in ("correct", "incorrect") else None


class GradingBackend(Protocol):
    def compare(self, reference_answer: str, prediction_answer: str) -> bool:
        """Use one evaluator; raise MathParseError if no answer can be parsed."""
        ...


def _unwrap(value: str) -> str:
    """Remove presentation wrappers only; never rewrite the mathematical expression."""
    text = value.strip()
    pairs = (("**", "**"), ("__", "__"), ("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))
    while True:
        old = text
        for left, right in pairs:
            if text.startswith(left) and text.endswith(right) and len(text) > len(left + right):
                text = text[len(left) : -len(right)].strip()
                break
        if text == old:
            return text


def _box_content(text: str, index: int) -> str:
    """Read the balanced argument of the box command ending at ``index``."""
    while index < len(text) and text[index].isspace():
        index += 1
    if index == len(text) or text[index] != "{":
        raise MathParseError("prediction_malformed_final_box")
    start, depth = index + 1, 1
    index += 1
    while index < len(text) and depth:
        if text[index] == "\\" and index + 1 < len(text) and text[index + 1] in "{}":
            index += 2
            continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    if depth:
        raise MathParseError("prediction_unbalanced_final_box")
    answer = _unwrap(text[start : index - 1])
    if not answer:
        raise MathParseError("prediction_empty_final_box")
    return answer


def extract_final_box(candidate: str) -> tuple[str, tuple[str | None, ...]]:
    """Return the last balanced \\boxed answer and, for diagnostics, the earlier boxes.

    Only the last box is the answer (M.9). Earlier boxes and prose declarations never
    compete with it. An earlier box that cannot be read is returned as None; it never
    makes the candidate unresolved. A malformed last box does: no earlier fallback.
    """
    if not isinstance(candidate, str) or not candidate.strip():
        raise MathParseError("prediction_empty_or_non_string")
    if len(candidate) > MAX_SOLUTION_CHARACTERS:
        raise MathParseError("prediction_length_limit")
    matches = list(_BOXED.finditer(candidate))
    if not matches:
        raise MathParseError("prediction_missing_final_box")
    answer = _box_content(candidate, matches[-1].end())
    earlier: list[str | None] = []
    for match in matches[:-1]:
        try:
            earlier.append(_box_content(candidate, match.end()))
        except MathParseError:
            earlier.append(None)
    return answer, tuple(earlier)


def _last_source_box(value: str) -> str:
    """Source golds may carry their own \\boxed or \\fbox wrapper; read the last one."""
    return _box_content(value, list(_BOX.finditer(value))[-1].end())


@dataclass(frozen=True)
class MathVerifyAdapter:
    """The installed pin is recorded, pending the authors' M.12 version confirmation."""

    expected_version: str = MATH_VERIFY_VERSION
    float_rounding: int = 6
    numeric_precision: int = 15

    def check_available(self) -> None:
        if self.expected_version != MATH_VERIFY_VERSION:
            raise MathBackendError("changing Math-Verify requires rebuilding the versioned E chain")
        try:
            version = importlib.metadata.version("math-verify")
        except importlib.metadata.PackageNotFoundError as exc:
            raise MathBackendError(f"math-verify=={self.expected_version} is required") from exc
        if version != self.expected_version:
            raise MathBackendError(
                f"math-verify version {version}; expected {self.expected_version}"
            )
        try:
            from math_verify import parse, verify
        except ImportError as exc:
            raise MathBackendError("math-verify dependencies could not be imported") from exc
        if any("raise_on_error" not in inspect.signature(fn).parameters for fn in (parse, verify)):
            raise MathBackendError("Math-Verify API cannot preserve E errors")

    def parse_answer(self, value: str, side: str):
        from math_verify import LatexExtractionConfig, parse
        from math_verify.errors import TimeoutException

        if not isinstance(value, str) or not value.strip():
            raise MathParseError(f"{side}_empty")
        wrapped = _unwrap(value)
        # Golds may themselves include a box. Do not nest it in another box.
        if _BOX.search(wrapped):
            wrapped = _last_source_box(wrapped)
        try:
            parsed = parse(
                f"\\boxed{{{wrapped}}}",
                extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
                fallback_mode="no_fallback",
                extraction_mode="first_match",
                parsing_timeout=None,
                raise_on_error=True,
            )
        except TimeoutException as exc:
            raise TimeoutError(f"{side}_parse_timeout") from exc
        except Exception as exc:
            raise MathBackendError(f"{side}_parse_exception:{type(exc).__name__}") from exc
        if not parsed or any(isinstance(item, str) for item in parsed):
            raise MathParseError(f"{side}_not_parsed")
        return parsed

    def compare(self, reference_answer: str, prediction_answer: str) -> bool:
        from math_verify import verify
        from math_verify.errors import TimeoutException

        gold = self.parse_answer(reference_answer, "reference")
        prediction = self.parse_answer(prediction_answer, "prediction")
        try:
            result = verify(
                gold,
                prediction,
                strict=True,
                allow_set_relation_comp=False,
                float_rounding=self.float_rounding,
                numeric_precision=self.numeric_precision,
                timeout_seconds=None,
                raise_on_error=True,
            )
        except TimeoutException as exc:
            raise TimeoutError("comparison_timeout") from exc
        except Exception as exc:
            raise MathBackendError(f"comparison_exception:{type(exc).__name__}") from exc
        if type(result) is not bool:
            raise MathBackendError("comparison returned a non-boolean value")
        return result


def _boxes_all_equivalent(
    backend: GradingBackend, prediction: str, earlier: tuple[str | None, ...]
) -> bool | None:
    """Diagnostic only: is every earlier box equivalent to the last one under the same E?"""
    undetermined = False
    for box in earlier:
        if box is None:
            undetermined = True
            continue
        if box == prediction:
            continue
        try:
            equal = backend.compare(prediction, box)
        except Exception:
            undetermined = True
            continue
        if type(equal) is not bool:
            undetermined = True
        elif not equal:
            return False
    return None if undetermined else True


def _compare_worker(
    connection: Connection,
    backend: GradingBackend,
    reference: str,
    prediction: str,
    earlier: tuple[str | None, ...],
) -> None:
    """Send E's verdict first; the earlier-box diagnostic follows and cannot change it."""
    try:
        try:
            result = backend.compare(reference, prediction)
            if type(result) is not bool:
                raise MathBackendError("comparison returned a non-boolean value")
            verdict = ("correct" if result else "incorrect", "math_verify_answer_equivalence")
        except MathParseError as exc:
            verdict = ("unresolved", str(exc))
        except TimeoutError as exc:
            verdict = ("timeout", str(exc))
        except Exception as exc:
            # Never echo reference answers or exception arguments in operational logs.
            verdict = ("backend_error", type(exc).__name__)
        connection.send(verdict)
        if verdict[0] in ("correct", "incorrect") and earlier:
            connection.send(_boxes_all_equivalent(backend, prediction, earlier))
    finally:
        connection.close()


class MathGrader:
    def __init__(
        self,
        backend: GradingBackend | None = None,
        *,
        timeout_seconds: float = 5.0,
        require_boxed_prediction: bool = True,
        backend_identity: str | None = None,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        self.backend = backend if backend is not None else MathVerifyAdapter()
        self.timeout_seconds = timeout_seconds
        self.require_boxed_prediction = require_boxed_prediction
        versions = {}
        if isinstance(self.backend, MathVerifyAdapter):
            self.backend.check_available()
            versions = {
                name: importlib.metadata.version(name)
                for name in (
                    "math-verify",
                    "sympy",
                    "latex2sympy2_extended",
                    "antlr4-python3-runtime",
                )
            }
        self._configuration = {
            "contract_version": ANSWER_CONTRACT_VERSION,
            "math_verify_version": MATH_VERIFY_VERSION,
            "version_confirmation": "pending_ruide_M12",
            "backend_identity": backend_identity
            or (
                "math-verify"
                if isinstance(self.backend, MathVerifyAdapter)
                else f"injected:{type(self.backend).__module__}.{type(self.backend).__qualname__}"
            ),
            "backend_versions": versions,
            "math_verify_options": asdict(self.backend)
            if isinstance(self.backend, MathVerifyAdapter)
            else None,
            "timeout_seconds": timeout_seconds,
            "require_boxed_prediction": require_boxed_prediction,
            "max_solution_characters": MAX_SOLUTION_CHARACTERS,
            "contract_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def configuration(self) -> dict:
        return json.loads(json.dumps(self._configuration))

    def cache_key(self, question: str, reference_answer: str, candidate_solution: str) -> str:
        if any(not isinstance(x, str) for x in (question, reference_answer, candidate_solution)):
            raise TypeError("question, reference_answer and candidate_solution must be strings")
        value = [question, reference_answer, candidate_solution, self._configuration]
        return hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    def grade(
        self,
        question: str,
        reference_answer: str,
        candidate_solution: str | None = None,
    ) -> GradeResult:
        # Historical offline probe calls grade(gold, prediction); no legacy evaluator is kept.
        if candidate_solution is None:
            question, reference_answer, candidate_solution = "", question, reference_answer
        identity = self.cache_key(question, reference_answer, candidate_solution)
        prediction = None
        n_boxes = len(_BOXED.findall(candidate_solution)) if self.require_boxed_prediction else None
        all_equivalent = None

        def result(status: GradeStatus, detail: str) -> GradeResult:
            return GradeResult(
                status,
                detail,
                prediction,
                ANSWER_CONTRACT_VERSION,
                identity,
                n_boxes=n_boxes,
                boxes_all_equivalent=all_equivalent,
            )

        if not reference_answer.strip():
            return result("unresolved", "reference_empty")
        try:
            if self.require_boxed_prediction:
                prediction, earlier = extract_final_box(candidate_solution)
            else:
                prediction, earlier = _unwrap(candidate_solution), ()
                if not prediction:
                    raise MathParseError("prediction_empty")
        except MathParseError as exc:
            return result("unresolved", str(exc))
        context = multiprocessing.get_context("spawn")
        receive, send = context.Pipe(duplex=False)
        process = context.Process(
            target=_compare_worker,
            args=(send, self.backend, reference_answer, prediction, earlier),
            daemon=True,
        )
        started = False
        try:
            process.start()
            started = True
            send.close()
            if not receive.poll(self.timeout_seconds):
                return result("timeout", "worker_wall_clock_deadline")
            try:
                status, detail = receive.recv()
            except EOFError:
                return result("backend_error", "worker_exited_without_result")
            if status in ("correct", "incorrect") and earlier:
                # Separate deadline: a slow or failed diagnostic leaves it unknown and
                # never turns the verdict already received into an E error.
                try:
                    if receive.poll(self.timeout_seconds):
                        value = receive.recv()
                        all_equivalent = value if isinstance(value, bool) else None
                except (EOFError, OSError):
                    all_equivalent = None
            return result(status, detail)
        except Exception as exc:
            return result("backend_error", type(exc).__name__)
        finally:
            receive.close()
            send.close()
            if started:
                process.join(timeout=0.1)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
                process.close()
