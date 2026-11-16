"""Domain-agnostic interfaces (paper Sec. 5).

Core is domain-independent; each domain (detox, math) provides implementations
under ``rafr.domains.<name>``. Nothing here may import from ``annotation``, ``data.labels``
or ``evaluator``: ``task_success`` is the *interface* to E, the implementation lives in the
domain package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Verifier(Protocol):
    """See src/rafr/contracts/score_spec.md. Scores in [0, 1], higher = more toxic."""

    verifier_id: str
    model_version: str
    batch_limit: int

    def score(self, texts: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class VerifyResult:
    """paper Sec. 5 ``verify(x)``: toxicity score plus optional category (detox)."""

    score: float
    category: str | None = None


@runtime_checkable
class TaskSuccess(Protocol):
    """paper Sec. 5 ``task_success(x)``: for detox, E judges SAFE. Returns True on success."""

    def __call__(self, text: str) -> bool: ...


@runtime_checkable
class Quality(Protocol):
    """paper Sec. 5 ``quality(x0, x_T)``: semantic preservation for detox; None where undefined."""

    def __call__(self, x0: str, x_t: str) -> float | None: ...


@runtime_checkable
class Verify(Protocol):
    """paper Sec. 5 ``verify(x)`` for one verifier."""

    def __call__(self, text: str) -> VerifyResult: ...


@dataclass(frozen=True)
class Domain:
    """Bundle of the three domain interfaces (paper Sec. 5 table)."""

    name: str
    task_success: TaskSuccess
    quality: Quality
