"""Read-only original math request/schema replay. No provider or experiment is run."""

from __future__ import annotations

from pipelines.math_ext_exp1 import reconstruct_original_channel
from pipelines.math_ext_native import native_t1
from rafr.channel import ObservationBuilder
from rafr.channel.identity import IdentityChannel
from rafr.contracts.schemas import Arm, Lambda, LogRecord, Split
from rafr.controller import decide as ctrl
from rafr.domains.math.generation import build_controller_config
from rafr.domains.math.storage import IntegrityError
from rafr.formatter.math_extended import original_feedback


def replay_original(source, *, fixture=False):
    """Return separately counted Exp1 and Exp2-t1 evidence; fixtures are never real replay."""
    if fixture and source.execution_mode != "offline_stub":
        raise IntegrityError("fixture replay requires offline_stub source")
    if not fixture and source.execution_mode != "live":
        raise IntegrityError("real-artifact replay requires live source; a fixture is not evidence")
    starts = source.states(Split.D_EXP1)
    if not fixture and len(starts) != 300:
        raise IntegrityError("formal original replay requires all 300 D_Exp1 starts")
    _, exp1 = reconstruct_original_channel(source, starts)
    exported = native_t1(source, starts)
    by_qid = {s.x0_id: s for s in starts}
    cfg = build_controller_config(source.master_seed, 4)
    builder = ObservationBuilder(IdentityChannel(), *source.taus, *source.verifier_ids)
    attempts = 0
    first = source.read_bound_rows("exp2/rounds/round_1/decisions.jsonl")
    for raw in first:
        row = LogRecord.model_validate(raw)
        state = by_qid[row.x0_id]
        if row.arm not in (Arm.A1, Arm.B) or row.x_t != state.text or row.t != 1:
            raise IntegrityError("original Exp2 first-round input differs")
        obs = builder.build(row.state_id, row.s1, row.s2, Lambda.NONE)
        if row.tau1 != obs.tau1 or row.tau2 != obs.tau2:
            raise IntegrityError("original Exp2 threshold differs")
        if row.model != cfg.model or row.decoding != cfg.decoding:
            raise IntegrityError("original Exp2 controller identity differs")
        feedback = original_feedback(row.arm, obs)
        labels = ctrl.DecisionLabels(row.x0_id, row.arm, Lambda.NONE, 1, "math_exp2")
        seed = ctrl.call_seed(cfg, labels)
        if not row.attempts:
            raise IntegrityError("original Exp2 request has no attempt evidence")
        for i, attempt in enumerate(row.attempts):
            request = ctrl.build_request(
                cfg,
                [
                    {
                        "role": "user",
                        "content": ctrl.render_user_turn(
                            cfg, state.x0, state.text, feedback, attempt=i
                        ),
                    }
                ],
                ctrl.attempt_call_seed(seed, i),
                i,
            )
            if (
                attempt.attempt != i
                or attempt.seed != request.seed
                or attempt.prompt_hash != request.prompt_hash()
            ):
                raise IntegrityError("original Exp2 request prompt/seed differs")
            attempts += 1
        if row.prompt_hash != row.attempts[-1].prompt_hash:
            raise IntegrityError("original Exp2 final prompt hash differs")
    # Require every original decision stream, including aggregate and each round. The
    # caller must bind all hashes; the reader never invents trust from current bytes.
    files = ["exp1/decisions.jsonl", "exp2/decisions.jsonl"] + [
        f"exp2/rounds/round_{t}/decisions.jsonl" for t in range(1, 5)
    ]
    counts = {}
    for name in files:
        rows = source.read_bound_rows(name)
        for row in rows:
            LogRecord.model_validate(row)
        counts[name] = len(rows)
    source.assert_unchanged()
    return {
        "execution": "offline_fixture" if fixture else "read_only_real_artifact_replay",
        "provider_calls": 0,
        "exp1": exp1,
        "exp2": {
            "n_first_round_requests": len(first),
            "n_attempts": attempts,
            "prompt_hash_and_seed_identical": True,
            "t1_identity": exported["scoring_identity"],
        },
        "all_original_log_rows": counts,
        "source_identity": source.provenance(),
    }
