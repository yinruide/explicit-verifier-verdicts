"""Math Exp 1's single-round live smoke (2026-10-15; protocol II.4.4 / M.10).

Math's Exp 1 may run before the Exp 2 smoke once this smoke has passed. Ten cells of D_Exp1, drawn
by a fixed seed (component ``math_exp1_smoke``), all at lambda = high, the three arms cycled over
them (A1, B, permuted-B: 4 / 3 / 3); one real controller call per cell (plus II.8.2's parse retries)
with Exp 1's own requests -- the same states, frozen channel, permuted-B donors and controller
configuration as ``MathPipeline.exp1`` -- under their own seed namespace, so the formal run never
reuses a smoke answer. The prompts are byte for byte Exp 1's (the prompt hash leaves the seed out).

It looks only for engineering defects (the 10a disposition table, II.4.4): parsing, decision
extraction, E (D_Exp1's step 5 judgments, hashed before any controller call; Math-Verify on each
rewrite), the log records and the cost accounting. The controller's behaviour is no release
condition and the report holds no arm comparison (M.10). Everything goes to ``<run>/exp1_smoke``,
marked SMOKE; the formal flow reads only its report, through ``require``.
"""

from __future__ import annotations

import json
from collections import Counter

from rafr.channel import ObservationBuilder
from rafr.contracts.schemas import Arm, Lambda, LogRecord
from rafr.controller import decide as controller
from rafr.core.seeding import component_seed
from rafr.domains.math.generation import build_controller_config
from rafr.domains.math.storage import IntegrityError, atomic_json, digest, file_hash, read_json
from rafr.experiments.common import done_keys, make_log_record
from rafr.formatter import fixed_significant
from rafr.formatter.fields import compute_fields
from rafr.formatter.permute import derangement
from rafr.logging.writer import JsonlWriter, read_jsonl

N_CELLS = 10
LEVEL = Lambda.HIGH
ARM_CYCLE = (Arm.A1, Arm.B, Arm.PERMUTED_B)
NAMESPACE = "math_exp1_smoke"
#: Core's log grid has no Exp 1 smoke; a non-formal run logs as ``dry_run`` (II.10).
LOG_EXPERIMENT = "dry_run"
DIR = "exp1_smoke"
#: The II.10 fields every Exp 1 log record must carry.
LOG_FIELDS = (
    "x_t",
    "s1",
    "s2",
    "s1_tilde",
    "s2_tilde",
    "tau1",
    "tau2",
    "prompt_hash",
    "attempts",
    "seed",
    "model_version",
)


def cells(pipeline) -> list[tuple]:
    """The ten (state, arm) cells: D_Exp1 ordered by the smoke seed, arms cycled."""
    states = pipeline._initial_cohort()  # noqa: SLF001 - Exp 1's own cohort
    seed = pipeline.config.seed
    chosen = sorted(states, key=lambda s: digest([seed, NAMESPACE, s.state_id]))[:N_CELLS]
    return [(state, ARM_CYCLE[i % len(ARM_CYCLE)]) for i, state in enumerate(chosen)]


def _ledger_rows(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run(pipeline) -> dict:
    from rafr.core.runtime import prices_for

    pipeline.require_gate()
    if pipeline.config.stage == "formal":
        pipeline.require_diagnostics()
    pipeline._bind("feedback")  # noqa: SLF001 - the profile Exp 1 will run with
    out = pipeline.root / DIR
    out.mkdir(exist_ok=True)
    outcome, labels = pipeline.record_exp1_outcome(out)  # before any controller call
    states = pipeline._initial_cohort()  # noqa: SLF001
    channel = pipeline._channel()  # noqa: SLF001
    ids = [s.state_id for s in states]
    donors = derangement(ids, component_seed(pipeline.config.seed, "permuted_b"))
    pairs = dict(zip(ids, pipeline._initial_pairs([s.x0_id for s in states]), strict=True))  # noqa: SLF001
    builder = ObservationBuilder(
        channel,
        channel.tau1,
        channel.tau2,
        pipeline._verifier_identity(0)[0],  # noqa: SLF001
        pipeline._verifier_identity(1)[0],  # noqa: SLF001
    )
    cfg = build_controller_config(pipeline.config.seed, 1)
    chosen = cells(pipeline)
    ledger = getattr(pipeline.client, "ledger", None)
    if ledger is None:
        raise IntegrityError("the Exp 1 smoke accounts its calls: the client needs a ledger")
    completed = done_keys(out / "decisions.jsonl")
    done_before = sum((s.state_id, a.value, LEVEL.value, 1) in completed for s, a in chosen)
    with JsonlWriter(out / "decisions.jsonl") as log:
        for state, arm in chosen:
            sid = state.state_id
            key = sid, arm.value, LEVEL.value, 1
            if key in completed:
                continue
            obs = builder.build(sid, *pairs[sid], LEVEL)
            donor = builder.build(donors[sid], *pairs[donors[sid]], LEVEL)
            permuted = arm is Arm.PERMUTED_B
            feedback = pipeline.feedback(
                arm,
                obs,
                **(
                    {"donor_observation": donor, "permuted_fields": compute_fields(donor)}
                    if permuted
                    else {}
                ),
            )
            decision = controller.decide(
                pipeline.client,
                cfg,
                x0=state.x0,
                x_t=state.text,
                observation=feedback,
                labels=controller.DecisionLabels(sid, arm, LEVEL, 1, NAMESPACE),
            )
            log.write(
                make_log_record(
                    run_id=pipeline.root.name,
                    experiment=LOG_EXPERIMENT,
                    seed=pipeline.config.seed,
                    arm=arm,
                    lam=LEVEL,
                    t=1,
                    state_id=sid,
                    x0_id=state.x0_id,
                    x_t=state.text,
                    obs=obs,
                    fields=compute_fields(donor if permuted else obs),
                    decision=decision,
                    permuted_from=donor.state_id if permuted else None,
                )
            )
            completed.add(key)
    report, live_keys = audit(pipeline, out, chosen, builder, donors, pairs, cfg, outcome, labels)
    report["cells_done_before_this_invocation"] = done_before
    report["cost"] = cost(_ledger_rows(ledger.path), prices_for(cfg.model), live_keys)
    defects = list(report["engineering_defects"])
    if report["cost"]["problems"]:
        defects += [f"cost_accounting:{p}" for p in report["cost"]["problems"]]
    report["engineering_defects"] = sorted(set(defects))
    report["decisions_sha256"] = file_hash(out / "decisions.jsonl")
    atomic_json(out / "smoke_report.json", report)
    return report


def cost(rows: list[dict], price, live_keys: set[str]) -> dict:
    """This smoke's own controller calls, found in the client's ledger by the cache keys of the
    attempts its log records as live (so a later rerun, which calls nothing, accounts the same),
    priced at list price."""
    live = {}
    for r in rows:
        key = r.get("cache_key")
        if r.get("purpose") == "controller" and key in live_keys and not r.get("cached"):
            live.setdefault(key, r)
    live = list(live.values())
    tokens_in = sum(int(r.get("input_tokens") or 0) for r in live)
    tokens_out = sum(int(r.get("output_tokens") or 0) for r in live)
    problems = []
    if price is None:
        problems.append("no_list_price")
    if any(not r.get("input_tokens") or not r.get("output_tokens") for r in live):
        problems.append("live_call_without_token_counts")
    if len(live) != len(live_keys):
        problems.append("live_attempt_without_ledger_row")
    usd = (tokens_in * price.input_ + tokens_out * price.output) / 1e6 if price else None
    return {
        "live_calls": len(live),
        "live_attempts_logged": len(live_keys),
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "usd_list_price": round(usd, 4) if usd is not None else None,
        "basis": "the client's ledger rows of the attempts this smoke logged as live (cache key)",
        "problems": problems,
    }


def e_status(pipeline, qid: str, text: str) -> str:
    """Math-Verify on a rewrite, not written to the run's E log: M.9's rule -- an E error is
    retried twice; three worker wall-clock timeouts make it unresolved; any other error is an
    engineering defect here (``e_error``)."""
    from pipelines.math_pipeline import WALL_CLOCK_TIMEOUT

    question = pipeline.questions[qid]
    reference = pipeline.bundle.private_references[qid].reference_answer
    results = []
    for _ in range(3):
        result = pipeline.grader.grade(question.problem, reference, text)
        if result.status in ("correct", "incorrect", "unresolved"):
            return result.status
        results.append(result)
    if all(r.status == "timeout" and r.detail == WALL_CLOCK_TIMEOUT for r in results):
        return "unresolved"
    return "e_error"


def audit(pipeline, out, chosen, builder, donors, pairs, cfg, outcome, labels) -> dict:
    """Engineering checks only (II.4.4's table); no controller behaviour is judged."""
    rows = {(r["state_id"], r["arm"]): r for r in read_jsonl(out / "decisions.jsonl")}
    defects, review, prompts, e_rewrites = [], [], [], Counter()
    live_keys: set[str] = set()
    actions = Counter()
    for state, arm in chosen:
        sid, qid = state.state_id, state.x0_id
        row = rows.get((sid, arm.value))
        if row is None:
            defects.append(f"missing_decision:{sid}:{arm.value}")
            continue
        # logging: Core's record, every II.10 field
        try:
            LogRecord.model_validate(row)
        except ValueError:
            defects.append(f"log_record_invalid:{sid}")
        missing = [f for f in LOG_FIELDS if row.get(f) in (None, [], "")]
        if arm is Arm.PERMUTED_B and not row.get("permuted_from_state_id"):
            missing.append("permuted_from_state_id")
        if missing:
            defects.append(f"log_fields_missing:{sid}:{','.join(missing)}")
        # E: the outcome is step 5's judgment of this state (II.9.1), INCORRECT by construction
        if getattr(labels.get(qid), "value", labels.get(qid)) != "INCORRECT":
            defects.append(f"outcome_label_not_incorrect:{qid}")
        # prompts: rebuild each attempt's request; the logged hash and seed must be it
        obs = builder.build(sid, *pairs[sid], LEVEL)
        donor = builder.build(donors[sid], *pairs[donors[sid]], LEVEL)
        permuted = arm is Arm.PERMUTED_B
        observation = pipeline.feedback(
            arm,
            obs,
            **(
                {"donor_observation": donor, "permuted_fields": compute_fields(donor)}
                if permuted
                else {}
            ),
        )
        base_seed = controller.call_seed(
            cfg, controller.DecisionLabels(sid, arm, LEVEL, 1, NAMESPACE)
        )
        if not row.get("attempts"):
            defects.append(f"missing_attempts:{sid}")
        for attempt in row.get("attempts") or []:
            index = attempt["attempt"]
            content = controller.render_user_turn(
                cfg, state.x0, state.text, observation, attempt=index
            )
            request = controller.build_request(
                cfg,
                [{"role": "user", "content": content}],
                controller.attempt_call_seed(base_seed, index),
                index,
            )
            if not attempt.get("cached"):
                live_keys.add(request.cache_key())
            matched = request.prompt_hash() == attempt["prompt_hash"]
            seed_matched = request.seed == attempt["seed"]
            injected = observation in content
            display_ok = all(
                fixed_significant(value, digits=5) in observation
                for value in (obs.s1_obs, obs.s2_obs, obs.tau1, obs.tau2)
            )
            if not (matched and seed_matched and injected and display_ok):
                defects.append(f"prompt_mismatch:{sid}:{index}")
            prompts.append(
                {
                    "state_id": sid,
                    "arm": arm.value,
                    "attempt": index,
                    "system": request.system,
                    "messages": request.messages,
                    "seed": request.seed,
                    "prompt_hash": request.prompt_hash(),
                    "actual_hash_matches": matched,
                    "seed_matches": seed_matched,
                    "observation_injected": injected,
                    "scores_use_core_five_significant_digits": display_ok,
                }
            )
        # decision extraction: the logged action is the last attempt's parsed output
        if row["parse_fail"]:
            actions["parse_fail"] += 1
            review.append(f"parse_fail:{sid}: check the cause (II.4.4: code -> defect)")
            if row.get("action") is not None:
                defects.append(f"parse_fail_with_action:{sid}")
        else:
            actions[row.get("action")] += 1
            last = (row.get("attempts") or [{}])[-1]
            try:
                raw = json.loads(last.get("raw_output") or "")
            except ValueError:
                raw = {}
            if row.get("action") not in ("accept", "rewrite") or raw.get("action") != row.get(
                "action"
            ):
                defects.append(f"decision_extraction:{sid}")
            if row.get("action") == "rewrite":
                text = row.get("revised_text") or raw.get("revised_text")
                if not text:
                    defects.append(f"rewrite_without_text:{sid}")
                else:
                    status = e_status(pipeline, qid, text)
                    e_rewrites[status] += 1
                    if status == "e_error":
                        defects.append(f"e_error:{sid}")
        if row.get("refusal_like"):
            review.append(f"refusal_like:{sid}: check the cause (II.4.4)")
    audit_path = out / "prompt_audit.json"
    atomic_json(
        audit_path,
        {
            "SMOKE": True,
            "method": "Core request reconstruction matched to the actual attempt hash and seed",
            "prompts": prompts,
        },
    )
    arms = Counter(arm.value for _, arm in chosen)
    report = {
        "SMOKE": True,
        "is_research_result": False,
        "kind": "math Exp 1 single-round live smoke (II.4.4 / M.10, 2026-10-15)",
        "stage": pipeline.config.stage,
        "execution_mode": pipeline.config.execution_mode,
        "profile_hash": pipeline.config.profile_hash(),
        "seed_namespace": NAMESPACE,
        "level": LEVEL.value,
        "n_cells": len(chosen),
        "arms": dict(sorted(arms.items())),
        "state_ids": [s.state_id for s, _ in chosen],
        "outcome": outcome,
        "decisions": {k: actions[k] for k in ("accept", "rewrite", "parse_fail")},
        "e_on_rewrites": dict(sorted(e_rewrites.items())),
        "prompt_audit": {
            "file": audit_path.name,
            "sha256": file_hash(audit_path),
            "n_prompts": len(prompts),
        },
        "needs_review": review,
        "engineering_defects": sorted(set(defects)),
        "disposition": "II.4.4 (M.10): controller behaviour never changes the chain; an "
        "engineering defect is fixed and the smoke run again",
        "no_arm_comparison": True,
    }
    return report, live_keys


def require(pipeline) -> dict:
    """Exp 1's live release when no Exp 2 smoke has run (2026-10-15)."""
    out = pipeline.root / DIR
    report = read_json(out / "smoke_report.json")
    arms = report.get("arms", {})
    if (
        report.get("SMOKE") is not True
        or report.get("execution_mode") != "live"
        or report.get("engineering_defects") != []
        or report.get("n_cells") != N_CELLS
        or report.get("level") != LEVEL.value
        or set(arms) != {a.value for a in ARM_CYCLE}
        or report.get("profile_hash") != pipeline.config.profile_hash()
        or report.get("decisions_sha256") != file_hash(out / "decisions.jsonl")
    ):
        raise IntegrityError(
            "II.4.4 / M.10: Exp 1 before the Exp 2 smoke needs a matching single-round live "
            "smoke without engineering defects"
        )
    recorded = read_json(out / "outcome_hash.json")
    if recorded != report["outcome"]:
        raise IntegrityError("the smoke's D_Exp1 outcome record changed")
    return report
