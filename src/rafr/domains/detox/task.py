"""Detox implementations of the core interfaces (I.6 table).

* ``task_success(x)``: E judges SAFE (II.9). Built on :mod:`rafr.evaluator.judge`.
* ``verify(x)``: toxicity score from one verifier (category is None until a verifier that
  emits categories is delivered).
* ``quality(x0, x_T)``: semantic preservation per the TextDetox protocol, in ``semantic.py``
  (``QualityScorer``; Exp 2's judge wires it in). ``quality_not_implemented`` stays for
  :func:`make_domain`, which has no model config, and returns ``None`` so that no caller
  mistakes it for a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from rafr.contracts.schemas import EvalRecord, SafetyLabel
from rafr.core.interfaces import Domain, Verifier, VerifyResult
from rafr.evaluator import judge
from rafr.llm.client import LLMClient


@dataclass
class DetoxTaskSuccess:
    client: LLMClient
    ecfg: judge.EvaluatorConfig
    records: list[EvalRecord]

    def judge(self, text: str) -> EvalRecord:
        rec = judge.judge_text(self.client, self.ecfg, text)
        self.records.append(rec)
        return rec

    def __call__(self, text: str) -> bool:
        return self.judge(text).label is SafetyLabel.SAFE


def quality_not_implemented(x0: str, x_t: str) -> float | None:
    """Semantic preservation is out of scope for this round (semantic.py later)."""
    return None


def make_verify(verifier: Verifier):
    def verify(text: str) -> VerifyResult:
        return VerifyResult(score=verifier.score([text])[0], category=None)

    return verify


def make_domain(client: LLMClient, ecfg: judge.EvaluatorConfig) -> tuple[Domain, DetoxTaskSuccess]:
    ts = DetoxTaskSuccess(client=client, ecfg=ecfg, records=[])
    return Domain(name="detox", task_success=ts, quality=quality_not_implemented), ts
