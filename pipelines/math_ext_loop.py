"""Math extension round loop; original math_exp2_loop.py remains byte-for-byte unchanged.

Uses original scoring helpers and Core round storage/controller/logging. E is outside this loop.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pipelines.math_exp2_loop import OVERLENGTH, K, ScoreFn, ended, text_key
from rafr.channel import ObservationBuilder
from rafr.channel.identity import IdentityChannel
from rafr.contracts.schemas import PARSE_MAX_RETRIES, Action, Arm, Lambda, LogRecord, Score
from rafr.controller import decide as controller
from rafr.controller.parse import parse_attempt
from rafr.domains.math.storage import IntegrityError
from rafr.experiments.common import make_log_record
from rafr.experiments.exp2 import (
    Trajectory,
    assemble_streams,
    complete_rounds,
    load_round,
    round_dir,
    write_round,
)
from rafr.experiments.exp2_ext import no_verifier_record
from rafr.formatter.fields import compute_fields
from rafr.logging.writer import read_jsonl


def _replay_decision(row, previous, *, cfg=None, observation="", namespace=None):
    """Replay the Core parser; optionally bind every saved request to the current config."""
    if not 1 <= len(row.attempts) <= PARSE_MAX_RETRIES + 1:
        raise IntegrityError("round decision has invalid attempt coverage")
    if cfg is not None and (row.model != cfg.model or row.decoding != cfg.decoding):
        raise IntegrityError("round decision controller identity differs")
    last = None
    for index, attempt in enumerate(row.attempts):
        if attempt.attempt != index:
            raise IntegrityError("round decision attempt index differs")
        if cfg is not None and namespace is not None:
            labels = controller.DecisionLabels(
                previous.x0_id, row.arm, Lambda.NONE, row.t, namespace
            )
            seed = controller.attempt_call_seed(controller.call_seed(cfg, labels), index)
            request = controller.build_request(
                cfg,
                [
                    {
                        "role": "user",
                        "content": controller.render_user_turn(
                            cfg, previous.x0, previous.x_t, observation, attempt=index
                        ),
                    }
                ],
                seed,
                index,
            )
            if attempt.seed != seed or attempt.prompt_hash != request.prompt_hash():
                raise IntegrityError("round decision request hash or seed differs")
        last = parse_attempt(
            SimpleNamespace(text=attempt.raw_output, refused=attempt.provider_refused),
            previous.x_t,
            cfg.refusal_phrases if cfg is not None else [],
        )
        if attempt.parsed != last.parsed or attempt.parse_fail_reason != last.fail_reason:
            raise IntegrityError("round decision parse result differs from Core")
        if last.ok and index != len(row.attempts) - 1:
            raise IntegrityError("round decision retried after successful parsing")
    if not last.ok and len(row.attempts) != PARSE_MAX_RETRIES + 1:
        raise IntegrityError("round decision parse failure did not exhaust retries")
    if (
        row.action != last.action
        or row.revised_text != last.revised_text
        or row.parse_fail != (not last.ok)
        or row.n_parse_retries != len(row.attempts) - 1
        or row.prompt_hash != row.attempts[-1].prompt_hash
        or (cfg is not None and row.refusal_like != last.refusal_like)
    ):
        raise IntegrityError("round decision outcome differs from saved attempts")
    return last


def validate_round_chain(
    out,
    initial,
    *,
    run_id,
    seed,
    smoke,
    taus,
    verifier_ids,
    verifier_versions=None,
    experiment="exp2",
    k=K,
    require_complete=False,
    cfg=None,
    no_verifier_cfg=None,
    feedback=None,
    namespace=None,
):
    """Audit completed rounds against their input plan, not merely self-reported hashes.

    Pure local validation: no provider, evaluator, scorer or file writer is called. It
    rejects a self-consistent round copied from another run and reconstructs each state
    transition from Core-parsed decisions. Optional controller configs additionally replay
    prompt hashes and call seeds. Partial round files remain untouched for inspection.
    """
    out = Path(out)
    previous = copy.deepcopy(list(initial))
    keys = [(tr.x0_id, tr.arm) for tr in previous]
    if (
        not keys
        or len(keys) != len(set(keys))
        or any(tr.t != 1 or tr.history or tr.accepted for tr in previous)
    ):
        raise IntegrityError("round audit needs unique original starting trajectories")
    done = complete_rounds(out, k)
    for t in range(1, k + 1):
        if round_dir(out, t).exists() and t not in done:
            raise IntegrityError("incomplete or changed Core round: inspect before resume")
    if require_complete and done != list(range(1, k + 1)):
        raise IntegrityError("all four complete rounds are required")
    builder = ObservationBuilder(IdentityChannel(), taus[0], taus[1], *verifier_ids)
    names = {"trajectories.jsonl", "decisions.jsonl", "scores.jsonl", "ledger.jsonl"}
    try:
        for t in done:
            directory = round_dir(out, t)
            marker = json.loads((directory / "COMPLETE.json").read_text(encoding="utf-8"))
            if (
                set(marker.get("files", {})) != names
                or marker.get("t") != t
                or marker.get("SMOKE") is not smoke
                or marker.get("domain") != "math"
                or {p.name for p in directory.iterdir()} != names | {"COMPLETE.json"}
            ):
                raise IntegrityError("round manifest identity or file coverage differs")
            current = load_round(out, t)
            rows = [
                LogRecord.model_validate(row) for row in read_jsonl(directory / "decisions.jsonl")
            ]
            scores = [Score.model_validate(row) for row in read_jsonl(directory / "scores.jsonl")]
            if (
                [(tr.x0_id, tr.arm) for tr in current] != keys
                or marker.get("n_trajectories") != len(current)
                or marker.get("n_decisions") != len(rows)
            ):
                raise IntegrityError("round trajectory plan or counts differ")
            decisions = {(row.x0_id, row.arm.value): row for row in rows}
            scored = {(row.state_id, row.verifier_id): row for row in scores}
            if len(decisions) != len(rows) or len(scored) != len(scores):
                raise IntegrityError("duplicate round decision or score")
            used_decisions, used_scores = set(), set()
            expected = copy.deepcopy(previous)
            for before, after, rebuilt in zip(previous, current, expected, strict=True):
                key = before.x0_id, before.arm
                if before.accepted or ended(before):
                    if asdict(after) != asdict(before):
                        raise IntegrityError("terminated trajectory changed in a later round")
                    continue
                if before.t != t:
                    raise IntegrityError("active trajectory round index differs")
                row = decisions.get(key)
                if row is None:
                    detail = after.history[-1].get(OVERLENGTH) if after.history else None
                    if (
                        before.arm == Arm.NO_VERIFIER.value
                        or not before.history
                        or not isinstance(detail, dict)
                        or detail.get("t") != t
                        or set(detail) - {"t", "tokens", "L_in", "verifier_id"}
                        or type(detail.get("tokens")) is not int
                        or type(detail.get("L_in")) is not int
                        or not 0 < detail["L_in"] < detail["tokens"]
                        or ("verifier_id" in detail and detail["verifier_id"] not in verifier_ids)
                    ):
                        raise IntegrityError(
                            "active trajectory has no decision or valid overlength termination"
                        )
                    rebuilt.history[-1][OVERLENGTH] = detail
                else:
                    wanted = {
                        "run_id": run_id,
                        "experiment": experiment,
                        "seed": seed,
                        "t": t,
                        "state_id": before.state_id,
                        "x0_id": before.x0_id,
                        "x_t": before.x_t,
                        "arm": Arm(before.arm),
                        "lam": Lambda.NONE,
                    }
                    if any(getattr(row, name) != value for name, value in wanted.items()):
                        raise IntegrityError("round decision belongs to a different run/state")
                    if any(
                        getattr(row, name) is not None
                        for name in ("e_label", "e_score", "c_t", "r_ref", "permuted_from_state_id")
                    ):
                        raise IntegrityError(
                            "round decision contains forbidden E/post-hoc evidence"
                        )
                    nv = row.arm is Arm.NO_VERIFIER
                    observation = ""
                    if nv:
                        if any(
                            getattr(row, name) is not None
                            for name in (
                                "v1_id",
                                "v2_id",
                                "s1",
                                "s2",
                                "s1_tilde",
                                "s2_tilde",
                                "tau1",
                                "tau2",
                                "fields",
                            )
                        ):
                            raise IntegrityError("NoVerifier round contains verifier evidence")
                    else:
                        score_keys = [(before.state_id, vid) for vid in verifier_ids]
                        if any(key not in scored for key in score_keys):
                            raise IntegrityError("round verifier decision lacks its score pair")
                        pair = [scored[key] for key in score_keys]
                        if verifier_versions is not None and tuple(
                            score.model_version for score in pair
                        ) != tuple(verifier_versions):
                            raise IntegrityError("persisted round verifier version differs")
                        obs = builder.build(
                            before.state_id, pair[0].score, pair[1].score, Lambda.NONE
                        )
                        wanted = {
                            "v1_id": verifier_ids[0],
                            "v2_id": verifier_ids[1],
                            "s1": pair[0].score,
                            "s2": pair[1].score,
                            "s1_tilde": pair[0].score,
                            "s2_tilde": pair[1].score,
                            "tau1": taus[0],
                            "tau2": taus[1],
                            "fields": compute_fields(obs),
                        }
                        if any(getattr(row, name) != value for name, value in wanted.items()):
                            raise IntegrityError(
                                "round scores/thresholds/fields disagree with decision"
                            )
                        if feedback is not None:
                            observation = feedback(row.arm, obs)
                        used_scores.update(score_keys)
                    parsed = _replay_decision(
                        row,
                        before,
                        cfg=no_verifier_cfg if nv else cfg,
                        observation=observation,
                        namespace=namespace,
                    )
                    rebuilt.history.append(
                        {
                            "t": t,
                            "x_t": before.x_t,
                            "action": row.action.value if row.action else None,
                            "parse_fail": row.parse_fail,
                            "refusal_like": False,
                            "empty_rewrite": parsed.empty_rewrite,
                        }
                    )
                    if row.parse_fail:
                        rebuilt.n_parse_fail += 1
                        rebuilt.t = t + 1
                    elif row.action is Action.ACCEPT:
                        rebuilt.accepted, rebuilt.accepted_at = True, t
                    else:
                        rebuilt.x_t = row.revised_text
                        rebuilt.t = t + 1
                        rebuilt.n_rewrites += 1
                    used_decisions.add(key)
                if asdict(after) != asdict(rebuilt):
                    raise IntegrityError(
                        "round trajectory disagrees with reconstructed decision history"
                    )
            if used_decisions != decisions.keys() or used_scores != scored.keys():
                raise IntegrityError(
                    "round has decisions/scores outside active planned trajectories"
                )
            previous = current
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise IntegrityError(f"malformed persisted round: {exc}") from exc
    return previous


def run_rounds(
    out: Path,
    initial: Sequence[Trajectory],
    *,
    taus: tuple[float, float],
    verifier_ids: tuple[str, str],
    score_texts: ScoreFn,
    client: Any,
    cfg: Any,
    feedback: Callable[..., str],
    namespace: str,
    experiment: str,
    run_id: str,
    seed: int | None,
    smoke: bool,
    k: int = K,
    on_round: Callable[[int], None] | None = None,
    no_verifier_cfg: Any | None = None,
    verifier_versions: tuple[str, str] | None = None,
) -> list[Trajectory]:
    """Rounds 1..k into ``out/rounds``, resuming after the last complete round; then the
    run's assembled streams. ``initial`` is round 1's trajectories (used only when no round
    is complete yet)."""
    if any(Arm(tr.arm) is Arm.NO_VERIFIER for tr in initial) and no_verifier_cfg is None:
        raise IntegrityError("NoVerifier requires its own controller configuration")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    initial = copy.deepcopy(list(initial))
    audit = {
        "run_id": run_id,
        "seed": seed,
        "smoke": smoke,
        "taus": taus,
        "verifier_ids": verifier_ids,
        "verifier_versions": verifier_versions,
        "experiment": experiment,
        "k": k,
        "cfg": cfg,
        "no_verifier_cfg": no_verifier_cfg,
        "feedback": feedback,
        "namespace": namespace,
    }
    trajectories = validate_round_chain(out, initial, **audit)
    done = complete_rounds(out, k)
    builder = ObservationBuilder(IdentityChannel(), taus[0], taus[1], *verifier_ids)
    ledger = getattr(client, "ledger", None)
    for t in range(len(done) + 1, k + 1):
        n_ledger = len(ledger.rows()) if ledger is not None else 0
        decisions, scores = [], []
        active = [tr for tr in trajectories if not tr.accepted and not ended(tr)]
        # Core round barrier: every active trajectory is scored, in batches, before any decision.
        checked = [tr for tr in active if Arm(tr.arm) is not Arm.NO_VERIFIER]
        items = [(tr.x0_id, tr.x0, tr.x_t, text_key(tr.x0_id, tr.x_t)) for tr in checked]
        scored, overlength = score_texts(items) if items else ({}, {})
        observations = {}
        for tr, (_, _, _, key) in zip(checked, items, strict=True):
            if key in overlength:
                # 2026-10-13: the trajectory ends; its current text is its final text
                if not tr.history:
                    raise IntegrityError("a first state is longer than L_in")
                tr.history[-1][OVERLENGTH] = {"t": t, **overlength[key]}
                continue
            rows = scored[key]
            if [row.verifier_id for row in rows] != list(verifier_ids):
                raise IntegrityError("a score row belongs to another verifier")
            # Round streams use the decision's state id so scores and decisions join exactly.
            scores.extend(row.model_copy(update={"state_id": tr.state_id}) for row in rows)
            observations[tr.state_id] = builder.build(
                tr.state_id, rows[0].score, rows[1].score, Lambda.NONE
            )
        for tr in active:
            arm = Arm(tr.arm)
            no_verifier = arm is Arm.NO_VERIFIER
            if not no_verifier and tr.state_id not in observations:
                continue
            obs = observations.get(tr.state_id)
            decision = controller.decide(
                client,
                no_verifier_cfg if no_verifier else cfg,
                x0=tr.x0,
                x_t=tr.x_t,
                observation="" if no_verifier else feedback(arm, obs),
                labels=controller.DecisionLabels(tr.x0_id, arm, Lambda.NONE, t, namespace),
            )
            if no_verifier:
                decisions.append(
                    no_verifier_record(
                        run_id=run_id,
                        experiment=experiment,
                        seed=seed,
                        t=t,
                        state_id=tr.state_id,
                        x0_id=tr.x0_id,
                        x_t=tr.x_t,
                        decision=decision,
                    )
                )
            else:
                decisions.append(
                    make_log_record(
                        run_id=run_id,
                        experiment=experiment,
                        seed=seed,
                        arm=arm,
                        lam=Lambda.NONE,
                        t=t,
                        state_id=tr.state_id,
                        x0_id=tr.x0_id,
                        x_t=tr.x_t,
                        obs=obs,
                        fields=compute_fields(obs),
                        decision=decision,
                    )
                )
            tr.history.append(
                {
                    "t": t,
                    "x_t": tr.x_t,
                    "action": decision.action.value if decision.action else None,
                    "parse_fail": decision.parse_fail,
                    "refusal_like": False,
                    "empty_rewrite": decision.empty_rewrite,
                }
            )
            if decision.parse_fail:
                tr.n_parse_fail += 1
                tr.t = t + 1
            elif decision.action is Action.ACCEPT:
                tr.accepted, tr.accepted_at = True, t
            else:
                tr.x_t = decision.revised_text
                tr.t = t + 1
                tr.n_rewrites += 1
        write_round(
            out,
            t,
            trajectories,
            decisions,
            scores,
            ledger.rows()[n_ledger:] if ledger is not None else [],
            meta={"SMOKE": smoke, "domain": "math"},
        )
        if on_round is not None:
            on_round(t)
    trajectories = validate_round_chain(out, initial, require_complete=True, **audit)
    assemble_streams(out, k)
    return trajectories
