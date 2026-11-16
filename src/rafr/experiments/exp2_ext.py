"""Exp 2 extension (post-pilot, 2026-10-31): full pairing over the starts.

Not part of the primary Exp 2: its 150 / 150 trajectories, its assignment and its statistics
are only read. Every extension run writes a run directory of its own, with ``exp2``'s round layout
(one COMPLETE directory per round, resume from the latest one) and its post-loop judge pass.

* cohort ``e_incorrect``: the 300 D_Exp1 starts, x1 = the original Exp 2's x1. The new arms
  (No Verifier, B + pi-hat, Verdict-only) run from every start; the A1 / B fill runs each start
  only in the arm the original assignment did NOT give it, so with the original trajectories every
  start ends up with both. Seed namespace ``exp2``: each (x0, t) keeps the original seed.
* cohort ``e_correct``: the E-correct cohort's D_gate states as starts; A1, B, No Verifier,
  Verdict-only and B + pi-hat from every start. Seed namespace ``exp2_ecorrect``.

Several arms now start from the same x0, so a trajectory is (x0, arm); its round id stays
``{x0_id}-{arm}-t{t}`` and its seed key the x0 (paper Sec. 4.6: no arm in the seed). The start
subset is the run's ``plan`` (x0 -> arms), written to ``plan.jsonl``.

**t = 1 scores.** With ``t1_scores`` given (``e_incorrect``; 2026-10-31: on), round 1 takes
each start's (s1, s2) from the original Exp 2's round-1 scores instead of scoring it; the source's
path and sha256 are in the run metadata. Otherwise every start is scored once in process at t = 1
-- one text per start, shared by all its arms -- and with ``t1_reference`` (``e_correct``: step 6's
D_gate scores) the result is compared bit for bit and written to ``t1_reference_check.json``, a
record only. Later rounds score every distinct active text once.

**No Verifier** loads no verifier: its trajectories are never scored, its log rows carry no score,
threshold or field, and a run of No Verifier alone needs no scorer at all (the laptop can run it).
B + pi-hat's pi-hat comes from the raw score by the frozen calibrator (``pihat.jsonl``).
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rafr.calibration.apply import Calibrator
from rafr.channel import ObservationBuilder
from rafr.channel.identity import IdentityChannel
from rafr.contracts.schemas import (
    K_MAX_ROUNDS,
    Action,
    Arm,
    Decision,
    EvalRecord,
    Lambda,
    LogRecord,
    State,
)
from rafr.controller import decide as ctrl
from rafr.experiments import exp2
from rafr.experiments.common import PairScorer, make_log_record, score_rows
from rafr.formatter import compute_fields, format_observation
from rafr.formatter.extended import ExtWording, PiHat, format_extended
from rafr.llm.client import LLMClient
from rafr.logging.writer import JsonlWriter, read_jsonl

E_INCORRECT = "e_incorrect"
E_CORRECT = "e_correct"
NEW_ARMS: tuple[Arm, ...] = (Arm.NO_VERIFIER, Arm.B_PIHAT, Arm.VERDICT_ONLY)
FILL_ARMS: tuple[Arm, ...] = (Arm.A1, Arm.B)


@dataclass(frozen=True)
class Cohort:
    name: str
    seed_namespace: str
    arms: tuple[Arm, ...]


COHORTS: dict[str, Cohort] = {
    E_INCORRECT: Cohort(E_INCORRECT, exp2.SEED_NAMESPACE, NEW_ARMS + FILL_ARMS),
    E_CORRECT: Cohort(
        E_CORRECT,
        "exp2_ecorrect",
        (Arm.A1, Arm.B, Arm.NO_VERIFIER, Arm.VERDICT_ONLY, Arm.B_PIHAT),
    ),
}
PLAN = "plan.jsonl"
PIHAT_STREAM = "pihat.jsonl"
T1_CHECK = "t1_reference_check.json"
RUN_META = "ext_meta.json"


class Exp2ExtError(RuntimeError):
    pass


def full_pairing_plan(
    x0_ids: Sequence[str],
    *,
    new_arms: Sequence[Arm] = (),
    fill_arms: Sequence[Arm] = (),
    original_assignment: Mapping[str, str] | None = None,
) -> dict[str, list[Arm]]:
    """x0 -> the arms the extension runs from it: every ``new_arms`` arm, plus each ``fill_arms``
    arm the original assignment did not give that x0 (the original trajectory is reused)."""
    if fill_arms and original_assignment is None:
        raise Exp2ExtError("the A1 / B fill needs the original assignment")
    plan: dict[str, list[Arm]] = {}
    for x in x0_ids:
        arms = list(new_arms)
        if fill_arms:
            assert original_assignment is not None
            if x not in original_assignment:
                raise Exp2ExtError(f"{x} is not in the original assignment")
            original = Arm(original_assignment[x])
            arms += [a for a in fill_arms if a is not original]
        if len(arms) != len(set(arms)):
            raise Exp2ExtError(f"{x}: an arm is planned twice")
        plan[x] = arms
    return plan


@dataclass
class Exp2ExtSetup:
    run_id: str
    run_dir: Path
    cohort: str
    starts: Sequence[State]  # x1 per x0
    plan: Mapping[str, Sequence[Arm]]  # x0 -> arms run from it
    scorer: PairScorer | None  # None only when every planned arm is no_verifier
    tau1: float
    tau2: float
    k: int
    wording: ExtWording
    judge_fn: Callable[[str], EvalRecord]  # E, after the loop
    calibrator: Calibrator | None = None
    run_seed: int | None = None
    prior_judgments: Mapping[str, EvalRecord] | None = None  # text_id -> t = 1 judgment
    quality_fn: Callable[[str, str], float | None] | None = None
    t1_scores: Mapping[str, tuple[float, float]] | None = None  # x0 -> reused (s1, s2)
    t1_reference: Mapping[str, tuple[float, float]] | None = None  # x0 -> step 6 (s1, s2)
    experiment_name: str = "exp2"  # the log label; the cohort's seed namespace is separate
    seed_namespace: str | None = None  # None = the cohort's; a smoke passes "<namespace>_smoke"
    meta: dict[str, Any] = field(default_factory=dict)


def _bits(x: float) -> str:
    return struct.pack("<d", float(x)).hex()


def _check_setup(setup: Exp2ExtSetup) -> Cohort:
    cohort = COHORTS.get(setup.cohort)
    if cohort is None:
        raise Exp2ExtError(f"unknown cohort {setup.cohort!r}; expected one of {sorted(COHORTS)}")
    if setup.k != K_MAX_ROUNDS:
        raise Exp2ExtError(f"K is locked to {K_MAX_ROUNDS} (II.8.3)")
    starts = [s.x0_id for s in setup.starts]
    if len(starts) != len(set(starts)):
        raise Exp2ExtError("two starts share an x0")
    if set(setup.plan) - set(starts):
        raise Exp2ExtError("the plan names an x0 that is not a start")
    planned = {a for arms in setup.plan.values() for a in arms}
    stray = sorted(a.value for a in planned if a not in cohort.arms)
    if stray:
        raise Exp2ExtError(f"cohort {cohort.name} does not run arm(s) {stray} in the extension")
    if planned - {Arm.NO_VERIFIER} and setup.scorer is None:
        raise Exp2ExtError("a verifier arm is planned but no scorer is given")
    if Arm.B_PIHAT in planned:
        if setup.calibrator is None or setup.scorer is None:
            raise Exp2ExtError("B_pihat needs the frozen calibrator and the scorer")
        setup.calibrator.check_verifiers(setup.scorer.v1_id, setup.scorer.v2_id)
    if setup.t1_scores is not None:
        missing = [x for x, arms in setup.plan.items() if arms and x not in setup.t1_scores]
        if missing:
            raise Exp2ExtError(f"no reused t = 1 scores for {len(missing)} start(s): {missing[:3]}")
    return cohort


def _write_plan(rd: Path, plan: Mapping[str, Sequence[Arm]]) -> None:
    rows = [{"x0_id": x, "arms": [a.value for a in arms]} for x, arms in plan.items()]
    text = "".join(json.dumps(r) + "\n" for r in rows)
    p = rd / PLAN
    if p.exists():
        if p.read_text(encoding="utf-8") != text:
            raise Exp2ExtError(f"{p} records another plan; resume refused")
        return
    with p.open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _write_meta(rd: Path, meta: Mapping[str, Any]) -> None:
    p = rd / RUN_META
    meta = json.loads(json.dumps(dict(meta)))
    if p.exists():
        if json.loads(p.read_text(encoding="utf-8")) != meta:
            raise Exp2ExtError(f"{p} records a different setup; resume refused")
        return
    p.write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")


def no_verifier_record(
    *,
    run_id: str,
    experiment: str,
    seed: int | None,
    t: int,
    state_id: str,
    x0_id: str,
    x_t: str,
    decision: Decision,
) -> LogRecord:
    """The log row of a No Verifier decision: no score, threshold or field exists for it."""
    return LogRecord(
        run_id=run_id,
        experiment=experiment,  # type: ignore[arg-type]
        seed=seed,
        arm=Arm.NO_VERIFIER,
        lam=Lambda.NONE,
        t=t,
        state_id=state_id,
        x0_id=x0_id,
        x_t=x_t,
        v1_id=None,
        v2_id=None,
        s1=None,
        s2=None,
        s1_tilde=None,
        s2_tilde=None,
        tau1=None,
        tau2=None,
        fields=None,
        permuted_from_state_id=None,
        attempts=decision.attempts,
        action=decision.action,
        revised_text=decision.revised_text,
        n_parse_retries=decision.n_parse_retries,
        parse_fail=decision.parse_fail,
        refusal_like=decision.refusal_like,
        e_label=None,
        e_score=None,
        prompt_hash=decision.attempts[-1].prompt_hash if decision.attempts else None,
        model=decision.model,
        model_version=decision.model_version,
        decoding=decision.decoding,
    )


def _t1_check(
    rd: Path,
    pairs_by_x0: Mapping[str, tuple[float, float]],
    reference: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    """Round 1's in-process scores against the reference, bit for bit (a record only)."""
    out: dict[str, Any] = {"n": 0, "v1_identical": 0, "v2_identical": 0, "missing": [],
                           "max_abs_diff": {"v1": 0.0, "v2": 0.0}, "differ": []}  # fmt: skip
    for x, (s1, s2) in sorted(pairs_by_x0.items()):
        if x not in reference:
            out["missing"].append(x)
            continue
        r1, r2 = reference[x]
        out["n"] += 1
        for key, a, b in (("v1", s1, r1), ("v2", s2, r2)):
            if _bits(a) == _bits(b):
                out[f"{key}_identical"] += 1
            else:
                out["max_abs_diff"][key] = max(out["max_abs_diff"][key], abs(a - b))
                out["differ"].append({"x0_id": x, "verifier": key, "here": a, "reference": b})
    (rd / T1_CHECK).write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    return out


def run_loop_ext(
    client: LLMClient,
    ccfg: ctrl.ControllerConfig | None,
    ccfg_no_verifier: ctrl.ControllerConfig | None,
    setup: Exp2ExtSetup,
    *,
    on_round: exp2.RoundHook | None = None,
) -> list[exp2.Trajectory]:
    """The loop of one extension run (controller + in-process V1/V2 where an arm needs them)."""
    cohort = _check_setup(setup)
    namespace = setup.seed_namespace or cohort.seed_namespace
    planned = {a for arms in setup.plan.values() for a in arms}
    if Arm.NO_VERIFIER in planned and ccfg_no_verifier is None:
        raise Exp2ExtError("no_verifier needs the No Verifier controller config")
    if planned - {Arm.NO_VERIFIER} and ccfg is None:
        raise Exp2ExtError("the verifier arms need the controller config")
    cfg = ctrl.with_round_budget(ccfg, K_MAX_ROUNDS) if ccfg else None
    cfg_nv = ctrl.with_round_budget(ccfg_no_verifier, K_MAX_ROUNDS) if ccfg_no_verifier else None
    rd = Path(setup.run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    _write_plan(rd, setup.plan)
    _write_meta(
        rd,
        {
            "run_id": setup.run_id,
            "cohort": cohort.name,
            "seed_namespace": namespace,
            "k": setup.k,
            "n_starts": len(setup.starts),
            "n_trajectories": sum(len(a) for a in setup.plan.values()),
            "arms": sorted(a.value for a in planned),
            "tau": [setup.tau1, setup.tau2],
            "calibrator": setup.calibrator.provenance() if setup.calibrator else None,
            "t1_scores": "reused" if setup.t1_scores is not None else "in_process",
            "t1_reference": setup.t1_reference is not None,
            "scorer": (
                None
                if setup.scorer is None
                else {
                    "v1": [setup.scorer.v1_id, setup.scorer.v1_version],
                    "v2": [setup.scorer.v2_id, setup.scorer.v2_version],
                }
            ),
            "run_seed": setup.run_seed,
            **setup.meta,
        },
    )
    exp2.discard_incomplete(rd, setup.k)
    done = exp2.complete_rounds(rd, setup.k)
    if done:
        trajs = exp2.load_round(rd, done[-1])
        start_t = done[-1] + 1
    else:
        trajs = [
            exp2.Trajectory(x0_id=s.x0_id, x0=s.x0, arm=a.value, t=1, x_t=s.text)
            for s in setup.starts
            for a in setup.plan.get(s.x0_id, ())
        ]
        start_t = 1
    start_text = {s.x0_id: s.text for s in setup.starts}
    v1_id = setup.scorer.v1_id if setup.scorer else ""
    v2_id = setup.scorer.v2_id if setup.scorer else ""
    builder = ObservationBuilder(IdentityChannel(), setup.tau1, setup.tau2, v1_id, v2_id)
    ledger = getattr(client, "ledger", None)
    for t in range(start_t, setup.k + 1):
        n_ledger = len(ledger.rows()) if ledger is not None else 0
        decisions: list[Any] = []
        scores: list[Any] = []
        pihat_rows: list[dict[str, Any]] = []
        active = [tr for tr in trajs if not tr.accepted]
        scored = [tr for tr in active if Arm(tr.arm) is not Arm.NO_VERIFIER]
        pair_of: dict[str, tuple[float, float]] = {}
        round_info: dict[str, Any] = {}
        if scored:
            assert setup.scorer is not None
            if t == 1 and setup.t1_scores is not None:
                for tr in scored:
                    if tr.x_t != start_text[tr.x0_id]:
                        raise Exp2ExtError(f"{tr.state_id}: x_1 is not the start's text")
                pairs = [tuple(setup.t1_scores[tr.x0_id]) for tr in scored]
                round_info["t1_scores"] = "reused"
            else:
                texts = list(dict.fromkeys(tr.x_t for tr in scored))
                by_text = dict(zip(texts, setup.scorer.score_pair(texts), strict=True))
                pairs = [by_text[tr.x_t] for tr in scored]
                round_info["n_texts_scored"] = len(texts)
                if t == 1 and setup.t1_reference is not None:
                    by_x0 = {tr.x0_id: by_text[tr.x_t] for tr in scored}
                    check = _t1_check(rd, by_x0, setup.t1_reference)
                    round_info["t1_reference_identical"] = [
                        check["v1_identical"],
                        check["v2_identical"],
                        check["n"],
                    ]
            pair_of = {
                tr.state_id: (float(p[0]), float(p[1])) for tr, p in zip(scored, pairs, strict=True)
            }
            scores = score_rows(
                setup.scorer, [tr.state_id for tr in scored], list(pair_of.values())
            )
        for tr in active:
            arm = Arm(tr.arm)
            labels = ctrl.DecisionLabels(
                state_id=tr.x0_id, arm=arm, lam=Lambda.NONE, t=t, experiment=namespace
            )
            if arm is Arm.NO_VERIFIER:
                assert cfg_nv is not None
                observation = format_extended(arm, None, setup.wording)
                decision = ctrl.decide(
                    client, cfg_nv, x0=tr.x0, x_t=tr.x_t, observation=observation, labels=labels
                )
                record: LogRecord = no_verifier_record(
                    run_id=setup.run_id, experiment=setup.experiment_name, seed=setup.run_seed,
                    t=t, state_id=tr.state_id, x0_id=tr.x0_id, x_t=tr.x_t, decision=decision,
                )  # fmt: skip
            else:
                assert cfg is not None
                s1, s2 = pair_of[tr.state_id]
                obs = builder.build(tr.state_id, s1, s2, Lambda.NONE)
                if arm is Arm.VERDICT_ONLY:
                    observation = format_extended(arm, obs, setup.wording)
                elif arm is Arm.B_PIHAT:
                    assert setup.calibrator is not None
                    d = setup.calibrator.detail(s1, s2, obs.tau1, obs.tau2)
                    pihat_rows.append({"state_id": tr.state_id, "t": t, **d.as_dict()})
                    observation = format_extended(
                        arm, obs, setup.wording, pihat=PiHat(d.pihat1, d.pihat2)
                    )
                else:
                    observation = format_observation(arm, obs)
                decision = ctrl.decide(
                    client, cfg, x0=tr.x0, x_t=tr.x_t, observation=observation, labels=labels
                )
                record = make_log_record(
                    run_id=setup.run_id, experiment=setup.experiment_name, seed=setup.run_seed,
                    arm=arm, lam=Lambda.NONE, t=t, state_id=tr.state_id, x0_id=tr.x0_id,
                    x_t=tr.x_t, obs=obs, fields=compute_fields(obs), decision=decision,
                )  # fmt: skip
            decisions.append(record)
            tr.history.append(
                {
                    "t": t,
                    "x_t": tr.x_t,
                    "action": decision.action.value if decision.action else None,
                    "parse_fail": decision.parse_fail,
                    "refusal_like": decision.refusal_like,
                    "empty_rewrite": decision.empty_rewrite,
                }
            )
            if decision.parse_fail:
                tr.n_parse_fail += 1
                tr.t = t + 1  # x_{t+1} = x_t; the round is consumed (paper Sec. 4.6)
            elif decision.action is Action.ACCEPT:
                tr.accepted = True
                tr.accepted_at = t
            else:
                assert decision.revised_text is not None
                tr.n_rewrites += 1
                tr.refusal_like_rewrites += int(decision.refusal_like)
                tr.x_t_refusal_like = bool(decision.refusal_like)
                tr.x_t = decision.revised_text
                tr.t = t + 1
        if pihat_rows:  # before the round is marked COMPLETE; a redone round rewrites the same
            with JsonlWriter(rd / PIHAT_STREAM) as w:
                w.write_many(pihat_rows)
        ledger_rows = ledger.rows()[n_ledger:] if ledger is not None else []
        info = {
            "n_active": len(active),
            "n_scored": len(scored),
            "n_accepted_total": sum(tr.accepted for tr in trajs),
            **round_info,
        }
        exp2.write_round(rd, t, trajs, decisions, scores, ledger_rows, info)
        if on_round is not None:
            on_round(t, {**info, "n_decisions": len(decisions), "ledger_rows": ledger_rows})
    exp2.assemble_streams(rd, setup.k)
    (rd / exp2.LOOP_COMPLETE).write_text(
        json.dumps(
            {
                "k": setup.k,
                "n_trajectories": len(trajs),
                "rounds": exp2.complete_rounds(rd, setup.k),
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    return trajs


def read_pihat(rd: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """The pi-hat stream keyed by (round id, t); a redone round's duplicate rows must agree."""
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for r in read_jsonl(Path(rd) / PIHAT_STREAM):
        key = (r["state_id"], int(r["t"]))
        if key in out and out[key] != r:
            raise Exp2ExtError(f"{rd / PIHAT_STREAM}: two different rows for {key}")
        out[key] = r
    return out


def finish_ext(
    setup: Exp2ExtSetup, trajs: Sequence[exp2.Trajectory] | None = None
) -> dict[str, Any]:
    """E's post-loop pass, exactly ``exp2.finish``'s (every round text judged once, t = 1 from
    the prior judgments, the finals, ``judge_complete.json``)."""
    base = exp2.Exp2Setup(
        run_id=setup.run_id,
        run_dir=Path(setup.run_dir),
        x1_states=setup.starts,
        scorer=setup.scorer,  # type: ignore[arg-type]
        tau1=setup.tau1,
        tau2=setup.tau2,
        channel=IdentityChannel(),
        arms=sorted({a for arms in setup.plan.values() for a in arms}),
        k=setup.k,
        assignment_seed=0,
        run_seed=setup.run_seed,
        judge_fn=setup.judge_fn,
        experiment_name=setup.experiment_name,
        seed_namespace=setup.seed_namespace or COHORTS[setup.cohort].seed_namespace,
        prior_judgments=setup.prior_judgments,
        quality_fn=setup.quality_fn,
    )
    return exp2.finish(base, trajs)


def run_exp2_ext(
    client: LLMClient,
    ccfg: ctrl.ControllerConfig | None,
    ccfg_no_verifier: ctrl.ControllerConfig | None,
    setup: Exp2ExtSetup,
) -> dict[str, Any]:
    """Loop and post-loop judgments in one process (tests)."""
    return finish_ext(setup, run_loop_ext(client, ccfg, ccfg_no_verifier, setup))


__all__ = [
    "COHORTS",
    "E_CORRECT",
    "E_INCORRECT",
    "FILL_ARMS",
    "NEW_ARMS",
    "PIHAT_STREAM",
    "PLAN",
    "RUN_META",
    "T1_CHECK",
    "Cohort",
    "Exp2ExtError",
    "Exp2ExtSetup",
    "finish_ext",
    "full_pairing_plan",
    "no_verifier_record",
    "read_pihat",
    "run_exp2_ext",
    "run_loop_ext",
]
