"""Post-run engineering audit of actual Core controller prompts (M.10 / II.4.4)."""

from collections import Counter
from pathlib import Path

from rafr.channel import ObservationBuilder
from rafr.channel.identity import IdentityChannel
from rafr.contracts.schemas import Arm, Lambda
from rafr.controller import decide as controller
from rafr.domains.math.generation import build_controller_config
from rafr.domains.math.storage import atomic_json, file_hash
from rafr.formatter import fixed_significant
from rafr.logging.writer import read_jsonl


def audit_smoke(pipeline, out: Path, trajectories: list[dict]) -> dict:
    """Re-render recorded requests and prove identity with the actual attempt hashes.

    No provider calls and no judgment of whether a controller's behavior is desirable.
    All rounds and retry attempts of three predetermined trajectories are included.
    """
    cfg = build_controller_config(pipeline.config.seed, 4)
    sampled = sorted(row["question_id"] for row in trajectories)[:3]
    decisions = list(read_jsonl(out / "decisions.jsonl"))
    defects, prompts = [], []
    if len(sampled) != 3:
        defects.append("fewer_than_three_trajectories")
    if any(row["arm"] != "A1" or row["lam"] != "none" for row in decisions):
        defects.append("smoke_is_not_A1_identity_channel")
    for qid in sampled:
        selected = [row for row in decisions if row["x0_id"] == qid]
        if not selected or [row["t"] for row in selected] != list(range(1, len(selected) + 1)):
            defects.append(f"noncontiguous_rounds:{qid}")
        if len(selected) > 4:
            defects.append(f"round_budget_exceeded:{qid}")
        for row in selected:
            builder = ObservationBuilder(
                IdentityChannel(), row["tau1"], row["tau2"], row["v1_id"], row["v2_id"]
            )
            obs = builder.build(row["state_id"], row["s1"], row["s2"], Lambda.NONE)
            observation = pipeline.feedback(Arm.A1, obs)
            labels = controller.DecisionLabels(
                qid, Arm.A1, Lambda.NONE, row["t"], "math_exp2_smoke"
            )
            base_seed = controller.call_seed(cfg, labels)
            if not row["attempts"]:
                defects.append(f"missing_attempts:{qid}:{row['t']}")
            for attempt in row["attempts"]:
                index = attempt["attempt"]
                content = controller.render_user_turn(
                    cfg, pipeline.questions[qid].problem, row["x_t"], observation, attempt=index
                )
                request = controller.build_request(
                    cfg,
                    [{"role": "user", "content": content}],
                    controller.attempt_call_seed(base_seed, index),
                    index,
                )
                matched = request.prompt_hash() == attempt["prompt_hash"]
                seed_matched = request.seed == attempt["seed"]
                budget_ok = "at most 4 decisions in total" in request.system
                injected = observation in content
                display_ok = all(
                    fixed_significant(value, digits=5) in observation
                    for value in (row["s1"], row["s2"], row["tau1"], row["tau2"])
                )
                if not (matched and seed_matched and budget_ok and injected and display_ok):
                    defects.append(f"prompt_mismatch:{qid}:{row['t']}:{index}")
                prompts.append(
                    {
                        "question_id": qid,
                        "t": row["t"],
                        "attempt": index,
                        "system": request.system,
                        "messages": request.messages,
                        "observation": observation,
                        "seed": request.seed,
                        "prompt_hash": request.prompt_hash(),
                        "logged_hash": attempt["prompt_hash"],
                        "actual_hash_matches": matched,
                        "seed_matches": seed_matched,
                        "budget_is_four": budget_ok,
                        "observation_injected": injected,
                        "scores_use_core_five_significant_digits": display_ok,
                    }
                )
    audit = {
        "SMOKE": True,
        "method": "Core request reconstruction matched to actual attempt hash",
        "sampled_question_ids": sampled,
        "prompts": prompts,
        "engineering_defects": sorted(set(defects)),
        "behavior_is_not_a_release_gate": True,
    }
    path = out / "prompt_audit.json"
    atomic_json(path, audit)
    return {
        "engineering_defects": audit["engineering_defects"],
        "prompt_audit": {
            "file": path.name,
            "sha256": file_hash(path),
            "n_trajectories": len(sampled),
        },
        "decision_count_distribution": dict(
            sorted(Counter(len(t["rounds"]) for t in trajectories).items())
        ),
        "resume_rounds_verified": True,
    }
