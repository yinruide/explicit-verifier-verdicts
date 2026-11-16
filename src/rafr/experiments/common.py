"""Shared pieces of the two experiment runners: scoring, log rows, resume helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rafr.contracts.schemas import (
    Action,
    Arm,
    Decision,
    Lambda,
    LogRecord,
    MathLabel,
    Observation,
    SafetyLabel,
    Score,
    UncertaintyFields,
)
from rafr.core.interfaces import Verifier
from rafr.logging.writer import read_jsonl


class PairScorer(Protocol):
    """Scores a batch of texts with V1 and V2. In-process now; the authors' GPU batch flow
    (to_score.jsonl -> scored.jsonl) can be plugged in behind this interface later."""

    v1_id: str
    v2_id: str
    v1_version: str
    v2_version: str

    def score_pair(self, texts: Sequence[str]) -> list[tuple[float, float]]: ...


@dataclass
class InProcessScorer:
    v1: Verifier
    v2: Verifier

    @property
    def v1_id(self) -> str:
        return self.v1.verifier_id

    @property
    def v2_id(self) -> str:
        return self.v2.verifier_id

    @property
    def v1_version(self) -> str:
        return self.v1.model_version

    @property
    def v2_version(self) -> str:
        return self.v2.model_version

    def score_pair(self, texts: Sequence[str]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        texts = list(texts)
        lim = max(1, min(self.v1.batch_limit, self.v2.batch_limit))
        for i in range(0, len(texts), lim):
            chunk = texts[i : i + lim]
            s1 = self.v1.score(chunk)
            s2 = self.v2.score(chunk)
            out.extend(zip(s1, s2, strict=True))
        return out


def score_rows(
    scorer: PairScorer, state_ids: Sequence[str], pairs: Sequence[tuple[float, float]]
) -> list[Score]:
    rows: list[Score] = []
    for sid, (a, b) in zip(state_ids, pairs, strict=True):
        rows.append(
            Score(state_id=sid, verifier_id=scorer.v1_id, model_version=scorer.v1_version, score=a)
        )
        rows.append(
            Score(state_id=sid, verifier_id=scorer.v2_id, model_version=scorer.v2_version, score=b)
        )
    return rows


def make_log_record(
    *,
    run_id: str,
    experiment: str,
    seed: int | None,
    arm: Arm,
    lam: Lambda,
    t: int,
    state_id: str,
    x0_id: str,
    x_t: str,
    obs: Observation,
    fields: UncertaintyFields,
    decision: Decision,
    permuted_from: str | None = None,
    e_label: SafetyLabel | MathLabel | None = None,
    e_score: float | None = None,
) -> LogRecord:
    return LogRecord(
        run_id=run_id,
        experiment=experiment,  # type: ignore[arg-type]
        seed=seed,
        arm=arm,
        lam=lam,
        t=t,
        state_id=state_id,
        x0_id=x0_id,
        x_t=x_t,
        v1_id=obs.v1_id,
        v2_id=obs.v2_id,
        s1=obs.raw_s1,
        s2=obs.raw_s2,
        s1_tilde=obs.s1_obs,
        s2_tilde=obs.s2_obs,
        tau1=obs.tau1,
        tau2=obs.tau2,
        fields=fields,
        permuted_from_state_id=permuted_from,
        attempts=decision.attempts,
        action=decision.action,
        revised_text=decision.revised_text,
        n_parse_retries=decision.n_parse_retries,
        parse_fail=decision.parse_fail,
        refusal_like=decision.refusal_like,
        e_label=e_label,
        e_score=e_score,
        prompt_hash=decision.attempts[-1].prompt_hash if decision.attempts else None,
        model=decision.model,
        model_version=decision.model_version,
        decoding=decision.decoding,
    )


def done_keys(decisions_path: Path) -> set[tuple[str, str, str, int]]:
    """(state_id, arm, lam, t) already on disk, for resume."""
    return {(r["state_id"], r["arm"], r["lam"], int(r["t"])) for r in read_jsonl(decisions_path)}


def accepted(decision: Decision) -> bool:
    """Main-analysis accounting (paper Sec. 4.6): parse_fail counts as NOT accepted."""
    return (not decision.parse_fail) and decision.action is Action.ACCEPT


def parse_fail_cells(
    rows: Iterable[dict[str, Any]], threshold: float, *, by: str = "lam"
) -> dict[str, dict[str, Any]]:
    """parse_fail rate per arm x ``by``; cells above ``threshold`` are flagged.

    paper Sec. 4.6: Exp 1 reports the arm x lambda grid (``by="lam"``), Exp 2 the arm x round
    grid (``by="t"``). ``analysis/parse_fail.py`` produces the full primary report; this
    is the quick in-run version.
    """
    agg: dict[str, list[bool]] = {}
    for r in rows:
        agg.setdefault(f"{r['arm']}|{r[by]}", []).append(bool(r["parse_fail"]))
    out: dict[str, dict[str, Any]] = {}
    for k, v in agg.items():
        rate = sum(v) / len(v) if v else 0.0
        out[k] = {"n": len(v), "parse_fail_rate": rate, "ALERT": rate > threshold}
    return out
