"""Math correction ceiling: E controls stopping; no verifier and no reference in requests.

This is intentionally separate from ordinary Exp2. E's outcome follows the original math
Exp 2 rule (paper App. D.1, ``MathPipeline._grade_after_loop``; 2026-11-05): correct stops the
trajectory; incorrect continues; UNRESOLVED (three wall-clock timeouts of the Math-Verify
worker, ``ExtensionMathEvaluator``) is a task failure and the trajectory continues as after
incorrect, and the run goes on. Any other E outcome stops the affected run for review and
never silently becomes a judgment.
"""

from __future__ import annotations

import re
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from pipelines.math_ext_data import validate_output
from pipelines.math_ext_e import check_e_callback
from pipelines.math_ext_execution import check_execution
from pipelines.math_ext_exp1 import check_client_output_paths
from rafr.contracts.schemas import PARSE_MAX_RETRIES, Action, Arm, Decision, Lambda, LogRecord
from rafr.controller import decide as ctrl
from rafr.controller.parse import parse_attempt
from rafr.domains.math import generation
from rafr.domains.math.storage import IntegrityError, atomic_json, digest, file_hash, read_json
from rafr.experiments.exp2_ext import no_verifier_record
from rafr.formatter.extended import read_block_template
from rafr.formatter.math_extended import build_no_verifier_config
from rafr.logging.writer import JsonlWriter, read_jsonl

FEEDBACK = (
    "A check against the correct answer found that the final answer in the current solution "
    "is wrong. Revise the solution."
)


def ceiling_config(seed: int):
    config = build_no_verifier_config(seed, 4)
    original = read_block_template(generation.PROMPT_DIR / "rewrite.txt")
    _, block = original.rsplit("\n\nChecker output:", 1)
    if block != "\n---\n{observation}\n---":
        raise IntegrityError("original checker block shape changed; review ceiling wording")
    return replace(
        config,
        rewrite_template=config.rewrite_template
        + "\n\nFeedback:"
        + block.replace("{observation}", FEEDBACK),
    )


_NO_EVIDENCE = (
    "v1_id",
    "v2_id",
    "s1",
    "s2",
    "s1_tilde",
    "s2_tilde",
    "tau1",
    "tau2",
    "fields",
    "permuted_from_state_id",
    "e_label",
    "e_score",
    "c_t",
    "r_ref",
)


def _decision(row, state, text, t, cfg, run_id, seed, namespace):
    """Pure replay of request identity and the Core parser; never calls a provider."""
    expected = {
        "run_id": run_id,
        "experiment": "exp2",
        "seed": seed,
        "arm": Arm.ORACLE_FEEDBACK,
        "lam": Lambda.NONE,
        "t": t,
        "state_id": f"{state.x0_id}-oracle_feedback-t{t}",
        "x0_id": state.x0_id,
        "x_t": text,
        "model": cfg.model,
        "decoding": cfg.decoding,
    }
    if any(getattr(row, name) != value for name, value in expected.items()):
        raise IntegrityError("ceiling decision identity differs on resume")
    if any(getattr(row, name) is not None for name in _NO_EVIDENCE):
        raise IntegrityError("ceiling decision contains verifier or E evidence")
    if not 1 <= len(row.attempts) <= PARSE_MAX_RETRIES + 1:
        raise IntegrityError("ceiling decision attempt coverage differs")
    labels = ctrl.DecisionLabels(state.x0_id, Arm.ORACLE_FEEDBACK, Lambda.NONE, t, namespace)
    base_seed = ctrl.call_seed(cfg, labels)
    last = None
    for i, attempt in enumerate(row.attempts):
        request = ctrl.build_request(
            cfg,
            [
                {
                    "role": "user",
                    "content": ctrl.render_user_turn(cfg, state.x0, text, "", attempt=i),
                }
            ],
            ctrl.attempt_call_seed(base_seed, i),
            i,
        )
        if (
            attempt.attempt != i
            or attempt.seed != request.seed
            or attempt.prompt_hash != request.prompt_hash()
        ):
            raise IntegrityError("ceiling decision request hash or seed differs")
        last = parse_attempt(
            SimpleNamespace(text=attempt.raw_output, refused=attempt.provider_refused),
            text,
            cfg.refusal_phrases,
        )
        if attempt.parsed != last.parsed or attempt.parse_fail_reason != last.fail_reason:
            raise IntegrityError("ceiling saved parse result differs from Core parser")
        if i < len(row.attempts) - 1 and last.ok:
            raise IntegrityError("ceiling logged a retry after successful parsing")
    assert last is not None
    if not last.ok and len(row.attempts) != PARSE_MAX_RETRIES + 1:
        raise IntegrityError("ceiling parse failure did not exhaust fixed retries")
    if (
        row.prompt_hash != row.attempts[-1].prompt_hash
        or row.n_parse_retries != len(row.attempts) - 1
        or row.parse_fail != (not last.ok)
        or row.action != last.action
        or row.revised_text != last.revised_text
        or row.refusal_like != last.refusal_like
    ):
        raise IntegrityError("ceiling decision outcome differs from parsed attempts")
    return Decision(
        state_id=state.x0_id,
        arm=Arm.ORACLE_FEEDBACK,
        lam=Lambda.NONE,
        t=t,
        action=last.action,
        revised_text=last.revised_text,
        parse_fail=not last.ok,
        n_parse_retries=row.n_parse_retries,
        refusal_like=last.refusal_like,
        empty_rewrite=last.empty_rewrite,
        attempts=row.attempts,
        model=row.model,
        model_version=row.model_version,
        decoding=row.decoding,
    )


STATUSES = ("correct", "incorrect", "unresolved")  # E's terminal outcomes (original math Exp 2)


def _grade_status(grade):
    if not isinstance(grade, dict) or grade.get("status") not in STATUSES:
        raise IntegrityError("ceiling E returned a non-terminal outcome; new run required")
    status = grade["status"]
    if "task_success" in grade and (
        type(grade["task_success"]) is not bool or grade["task_success"] != (status == "correct")
    ):
        raise IntegrityError("ceiling E status/task_success conflict; new run required")
    return status


def _receipt_body(row, text, candidate, grade, e_identity, previous):
    return {
        "x0_id": row.x0_id,
        "t": row.t,
        "input_text": text,
        "output_text": candidate,
        "input_sha256": digest([row.x0_id, text]),
        "output_sha256": digest([row.x0_id, candidate]),
        "status": _grade_status(grade),
        "grade": grade,
        "e_identity_sha256": digest(e_identity),
        "decision_sha256": digest(row.model_dump(mode="json")),
        "previous_sha256": previous,
    }


def _validate_receipt(step, row, text, candidate, e_identity, previous):
    wanted = _receipt_body(row, text, candidate, step.get("grade"), e_identity, previous)
    if step != {**wanted, "receipt_sha256": digest(wanted)}:
        raise IntegrityError("ceiling receipt identity, grade or hash chain differs")
    return step["receipt_sha256"]


def _checkpoint(path, logged, steps):
    atomic_json(
        path,
        {
            "n_decisions": len(logged),
            "n_steps": len(steps),
            "decisions_sha256": digest([r.model_dump(mode="json") for r in logged.values()]),
            "steps_sha256": digest(list(steps.values())),
        },
    )


def _check_checkpoint(path, logged, steps):
    if not path.exists():
        if len(logged) > 1 or steps:
            raise IntegrityError("partial ceiling run lacks its integrity checkpoint")
        return
    saved = read_json(path)
    nd, ns = saved.get("n_decisions"), saved.get("n_steps")
    if (
        type(nd) is not int
        or type(ns) is not int
        or nd < 0
        or ns < 0
        or len(logged) < nd
        or len(steps) < ns
        or (len(logged) - nd) + (len(steps) - ns) > 1
    ):
        raise IntegrityError("ceiling checkpoint population differs")
    if saved != {
        "n_decisions": nd,
        "n_steps": ns,
        "decisions_sha256": digest([r.model_dump(mode="json") for r in list(logged.values())[:nd]]),
        "steps_sha256": digest(list(steps.values())[:ns]),
    }:
        raise IntegrityError("ceiling checkpoint content differs")


def _audit_partial(starts, logged, steps, cfg, run_id, seed, namespace, e_identity, chain):
    """Every saved row is a contiguous execution prefix, validated before fresh work."""
    if list(steps) != list(logged)[: len(steps)] or len(logged) - len(steps) not in (0, 1):
        raise IntegrityError("ceiling logs/receipts are not a contiguous prefix")
    seen = []
    stopped = False
    for state in starts:
        text = state.text
        for t in range(1, 5):
            key = state.x0_id, t
            if key not in logged:
                stopped = True
                break
            row = logged[key]
            decision = _decision(row, state, text, t, cfg, run_id, seed, namespace)
            seen.append(key)
            candidate = decision.revised_text if decision.action is Action.REWRITE else text
            if key not in steps:
                stopped = True
                break
            step = steps[key]
            chain = _validate_receipt(step, row, text, candidate, e_identity, chain)
            text = candidate
            if step["status"] == "correct":
                break
        if stopped:
            break
    if seen != list(logged):
        raise IntegrityError("unexpected ceiling rows outside reconstructed trajectories")
    return chain


def run_ceiling(
    client,
    starts,
    *,
    run_dir,
    run_id,
    seed,
    initial_identity,
    grade_callback,
    e_identity,
    smoke=False,
    execution_mode="offline_stub",
    confirmed=False,
):
    """Execute the ceiling engine; live transport needs explicit operator confirmation.

    initial_identity certifies starts drawn from the original D_Exp1 (or disjoint wrong
    smoke pool); this function never reads initial labels. The bound input list and source
    evidence must be supplied by the preparation stage, not invented from text content.
    Ceiling smoke uses math_exp2ext_smoke, the general Exp 2 extension smoke
    namespace; oracle_feedback and its static Feedback block distinguish its requests.
    """
    check_execution(
        client,
        execution_mode,
        confirmed=confirmed,
        source_mode=initial_identity.get("source_execution_mode", "offline_stub"),
    )
    if (
        not starts
        or len({s.x0_id for s in starts}) != len(starts)
        or len({s.state_id for s in starts}) != len(starts)
        or any(s.t != 1 or s.source != "g_fixed" for s in starts)
    ):
        raise IntegrityError("ceiling requires unique nonempty starting questions")
    if execution_mode == "live" and not smoke and len(starts) != 300:
        raise IntegrityError("formal ceiling requires all 300 incorrect starts")
    expected_split = "unlabeled_pool" if smoke else "d_exp1"
    if (
        initial_identity.get("split") != expected_split
        or not isinstance(initial_identity.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", initial_identity["sha256"])
        or not isinstance(initial_identity.get("source_root"), str)
        or not initial_identity["source_root"].strip()
        or not e_identity
    ):
        raise IntegrityError("ceiling starting population and E identity must be explicitly bound")
    source_root = Path(initial_identity["source_root"]).resolve()
    rd = validate_output(Path(run_dir), source_root=source_root, allow_existing=True)
    check_e_callback(
        grade_callback,
        run_dir=rd,
        source_root=source_root,
        e_identity=e_identity,
        live=execution_mode == "live",
    )
    if (rd / "STOPPED.json").exists():
        raise IntegrityError("ceiling run is STOPPED; preserve evidence and start a new run")
    check_client_output_paths(client, rd)
    cfg = ceiling_config(seed)
    namespace = "math_exp2ext_smoke" if smoke else "math_ceiling"
    identity = {
        "run_id": run_id,
        "domain": "math",
        "experiment": "correction_ceiling",
        "SMOKE": smoke,
        "execution_mode": execution_mode,
        "seed_namespace": namespace,
        "starts": [s.model_dump(mode="json") for s in starts],
        "controller": asdict(cfg),
        "initial_identity": initial_identity,
        "e_identity": e_identity,
    }
    rd.mkdir(parents=True, exist_ok=True)
    meta = rd / "ext_meta.json"
    if meta.exists() and read_json(meta) != identity:
        raise IntegrityError("ceiling run identity changed")
    if not meta.exists():
        atomic_json(meta, identity)
    marker = rd / "ceiling_complete.json"
    checkpoint = rd / "ceiling_checkpoint.json"
    names = [
        "ext_meta.json",
        "decisions.jsonl",
        "ceiling_steps.jsonl",
        "ceiling_results.json",
        "ceiling_checkpoint.json",
    ]
    if marker.exists():
        if read_json(marker).get("files") != {n: file_hash(rd / n) for n in names}:
            raise IntegrityError("completed ceiling artifacts changed")
        return read_json(rd / "ceiling_results.json")
    logged, steps = {}, {}
    for raw in read_jsonl(rd / "decisions.jsonl"):
        row = LogRecord.model_validate(raw)
        key = row.x0_id, row.t
        if key in logged or row.arm is not Arm.ORACLE_FEEDBACK:
            raise IntegrityError("duplicate or wrong-arm ceiling decision")
        logged[key] = row
    for row in read_jsonl(rd / "ceiling_steps.jsonl"):
        key = row["x0_id"], row["t"]
        if key in steps or key not in logged or row["status"] not in STATUSES:
            raise IntegrityError("invalid ceiling step receipt")
        steps[key] = row
    _check_checkpoint(checkpoint, logged, steps)
    chain = _audit_partial(
        starts, logged, steps, cfg, run_id, seed, namespace, e_identity, digest(identity)
    )
    results, visited = [], set()
    with (
        JsonlWriter(rd / "decisions.jsonl") as decisions,
        JsonlWriter(rd / "ceiling_steps.jsonl") as receipts,
    ):
        for state in starts:
            text, status, history = state.text, "incorrect", []
            for t in range(1, 5):
                key = state.x0_id, t
                visited.add(key)
                if key in logged:
                    row = logged[key]
                    decision = _decision(row, state, text, t, cfg, run_id, seed, namespace)
                else:
                    decision = ctrl.decide(
                        client,
                        cfg,
                        x0=state.x0,
                        x_t=text,
                        observation="",
                        labels=ctrl.DecisionLabels(
                            state.x0_id, Arm.ORACLE_FEEDBACK, Lambda.NONE, t, namespace
                        ),
                    )
                    row = no_verifier_record(
                        run_id=run_id,
                        experiment="exp2",
                        seed=seed,
                        t=t,
                        state_id=f"{state.x0_id}-oracle_feedback-t{t}",
                        x0_id=state.x0_id,
                        x_t=text,
                        decision=decision,
                    ).model_copy(update={"arm": Arm.ORACLE_FEEDBACK})
                    decisions.write(row)
                    logged[key] = row
                    _checkpoint(checkpoint, logged, steps)
                candidate = decision.revised_text if decision.action is Action.REWRITE else text
                if key in steps:
                    step = steps[key]
                    if step["input_text"] != text or step["output_text"] != candidate:
                        raise IntegrityError("ceiling receipt input/output differs")
                else:
                    # A retained text keeps its E outcome (incorrect by design at t = 1, or the
                    # previous round's incorrect / unresolved); no redundant E call.
                    grade = (
                        {"status": status}
                        if candidate == text
                        else grade_callback(state.x0_id, candidate)
                    )
                    try:
                        body = _receipt_body(row, text, candidate, grade, e_identity, chain)
                    except IntegrityError as exc:
                        atomic_json(
                            rd / "STOPPED.json",
                            {
                                "x0_id": state.x0_id,
                                "t": t,
                                "reason": str(exc),
                                "grade": grade,
                            },
                        )
                        raise
                    step = {**body, "receipt_sha256": digest(body)}
                    receipts.write(step)
                    steps[key] = step
                    chain = step["receipt_sha256"]
                    _checkpoint(checkpoint, logged, steps)
                text, status = candidate, step["status"]
                history.append(step)
                # Only correct stops; incorrect and unresolved (a task failure) continue.
                if status == "correct":
                    break
            results.append(
                {
                    "x0_id": state.x0_id,
                    "arm": "oracle_feedback",
                    "x_T": text,
                    "task_success": status == "correct",
                    "terminal_status": status,
                    "unresolved": status == "unresolved",
                    "stopped_by_e": status == "correct",
                    "n_decisions": len(history),
                    "rounds": history,
                    "SMOKE": smoke,
                }
            )
    if set(logged) != visited or set(steps) != visited:
        raise IntegrityError("unexpected ceiling receipts outside reconstructed trajectories")
    atomic_json(rd / "ceiling_results.json", results)
    atomic_json(marker, {"files": {n: file_hash(rd / n) for n in names}, "n": len(results)})
    return results
