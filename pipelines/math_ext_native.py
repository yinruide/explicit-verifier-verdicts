"""Read-only native math artifacts adapted to extension inputs; no provider or label access."""

from __future__ import annotations

from collections.abc import Sequence

from pipelines.math_exp2_loop import text_key
from pipelines.math_ext_data import MathSource
from rafr.contracts.schemas import Arm, Lambda, LogRecord, Score, State
from rafr.domains.math.storage import IntegrityError


def native_t1(source: MathSource, starts: Sequence[State]) -> dict:
    """Bind original round-one scores to assignment, public x1 and original decision logs.

    The returned export contains no reference or E label and can be sent to the pod.
    Input hashes remain in scoring_identity, alongside the original source filename.
    """
    assignment = source.read_bound_json("exp2/assignment.json")
    ids = {s.x0_id for s in starts}
    if len(ids) != len(starts) or not ids or set(assignment) != ids:
        raise IntegrityError("native t1 assignment must cover exactly the original starts")
    if any(a not in {Arm.A1.value, Arm.B.value} for a in assignment.values()):
        raise IntegrityError("original math assignment must contain only A1/B")
    decision_file = "exp2/rounds/round_1/decisions.jsonl"
    score_file = "exp2/rounds/round_1/scores.jsonl"
    decisions = {}
    for raw in source.read_bound_rows(decision_file):
        row = LogRecord.model_validate(raw)
        if row.x0_id in decisions or row.t != 1:
            raise IntegrityError("duplicate or non-t1 original decision")
        decisions[row.x0_id] = row
    scores = {}
    for raw in source.read_bound_rows(score_file):
        row = Score.model_validate(raw)
        key = row.state_id, row.verifier_id
        if key in scores:
            raise IntegrityError("duplicate native t1 score")
        scores[key] = row
    if set(decisions) != ids or len(scores) != 2 * len(ids):
        raise IntegrityError("native t1 source coverage differs")
    exported = []
    for state in starts:
        row = decisions[state.x0_id]
        expected_id = f"{state.x0_id}-{assignment[state.x0_id]}-t1"
        if (
            row.state_id != expected_id
            or row.arm.value != assignment[state.x0_id]
            or row.x_t != state.text
            or row.lam is not Lambda.NONE
            or row.seed != source.master_seed
            or (row.v1_id, row.v2_id) != source.verifier_ids
            or (row.tau1, row.tau2) != source.taus
        ):
            raise IntegrityError("native t1 question/text/assignment differs")
        pair = []
        for i, verifier in enumerate(source.verifier_ids):
            score = scores.get((expected_id, verifier))
            if score is None or score.model_version != source.verifier_versions[i]:
                raise IntegrityError("native t1 verifier identity or revision differs")
            logged = row.s1 if i == 0 else row.s2
            observed = row.s1_tilde if i == 0 else row.s2_tilde
            if (
                logged is None
                or observed is None
                or (
                    score.score.hex() != float(logged).hex()
                    or score.score.hex() != float(observed).hex()
                )
            ):
                raise IntegrityError("native t1 scores disagree bitwise with original log")
            pair.append(
                score.model_copy(update={"state_id": text_key(state.x0_id, state.text)}).model_dump(
                    mode="json"
                )
            )
        exported.append(
            {"x0_id": state.x0_id, "problem": state.x0, "text": state.text, "scores": pair}
        )
    source.assert_unchanged()
    return {
        "scoring_identity": {
            "domain": "math",
            "source": score_file,
            "source_sha256": source.provenance()["source_sha256"][score_file],
            "original_decisions_sha256": source.provenance()["source_sha256"][decision_file],
            "assignment_sha256": source.provenance()["source_sha256"]["exp2/assignment.json"],
            "verifier_ids": list(source.verifier_ids),
            "verifier_versions": list(source.verifier_versions),
            "reuse_decision": "2026-11-01 handoff 3.5/6.1",
        },
        "states": exported,
    }
