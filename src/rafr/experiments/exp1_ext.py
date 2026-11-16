"""Exp 1 extension (post-pilot, 2026-10-31): the four new arms, two cohorts.

Not part of the preregistered Exp 1: ``exp1.py``, its run directory and its results are only read.

* cohort ``e_incorrect`` = the original D_Exp1 (E-UNSAFE). The four new arms only; A1, B and
  permuted-B are the original run's decisions, read only and never rerun. Before any controller
  call every (state, lambda) is passed through the frozen channel again and checked bit for bit
  against the original run's ``channel.jsonl`` -- raw scores, post-channel scores and the
  perturbation assignment -- so the new arms see exactly the s_obs the original arms saw. Seed
  namespace ``exp1``: each state keeps the original run's per-state seed.
* cohort ``e_correct`` = the E-correct cohort drawn from D_gate. All seven arms, the states
  perturbed by the same frozen channel (II.5); permuted-B's table is drawn over this cohort only
  with the original rule (seed component ``permuted_b``). Seed namespace ``exp1_ecorrect``.

Everything else follows ``exp1.run_exp1``: the channel once per (state, lambda), shared by all
arms; one decision per (state, arm, lambda); resume skips cells on disk; E's judgment of every
state is there before the first call, and every state must carry the cohort's label.

pi-hat (``B_pihat`` and the mismatch arm) is computed per (state, lambda) from s_obs by the
frozen calibrator and written to ``pihat.jsonl``. The mismatch arm's donor table is a
derangement over the cohort keyed by its own seed only (component ``ext_pihat_mismatch``), the
same donor at all three levels, written to ``pihat_permutation.jsonl``; at each level the donor's
pi-hat of that level is shown in place of the state's own, nothing else changes. Its log rows name
that donor in ``permuted_from_state_id``. No Verifier gets the empty observation and the No
Verifier controller prompts; its rows still log the (state, lambda) scores it is not shown, as
every row logs B's fields.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rafr.calibration.apply import Calibrator, PiHatDetail
from rafr.channel import Channel, ObservationBuilder
from rafr.contracts.schemas import (
    PARSE_FAIL_ALERT_THRESHOLD,
    Arm,
    EvalRecord,
    Lambda,
    MathLabel,
    Observation,
    SafetyLabel,
    State,
    UncertaintyFields,
)
from rafr.controller import decide as ctrl
from rafr.experiments.common import (
    PairScorer,
    done_keys,
    make_log_record,
    parse_fail_cells,
    score_rows,
)
from rafr.experiments.exp1 import EXP1_DECISIONS, identity_to_low_by_lambda, permutation_for
from rafr.formatter import compute_fields, format_observation
from rafr.formatter.extended import PIHAT_ARMS, ExtWording, PiHat, format_extended
from rafr.llm.client import LLMClient
from rafr.logging.schema import RecordKind, stream_filename
from rafr.logging.writer import JsonlWriter, read_jsonl

E_INCORRECT = "e_incorrect"
E_CORRECT = "e_correct"
NEW_ARMS: tuple[Arm, ...] = (
    Arm.NO_VERIFIER,
    Arm.VERDICT_ONLY,
    Arm.B_PIHAT,
    Arm.B_PIHAT_MISMATCHED,
)
ORIGINAL_ARMS: tuple[Arm, ...] = (Arm.A1, Arm.B, Arm.PERMUTED_B)


@dataclass(frozen=True)
class Cohort:
    name: str
    seed_namespace: str
    arms: tuple[Arm, ...]  # run by the extension
    reused_arms: tuple[Arm, ...]  # the original run's decisions, read only


COHORTS: dict[str, Cohort] = {
    E_INCORRECT: Cohort(E_INCORRECT, "exp1", NEW_ARMS, ORIGINAL_ARMS),
    E_CORRECT: Cohort(E_CORRECT, "exp1_ecorrect", ORIGINAL_ARMS + NEW_ARMS, ()),
}
PIHAT_STREAM = "pihat.jsonl"
PIHAT_PERMUTATION = "pihat_permutation.jsonl"
RUN_META = "ext_meta.json"


class Exp1ExtError(RuntimeError):
    pass


@dataclass
class Exp1ExtSetup:
    run_id: str
    run_dir: Path
    cohort: str
    states: Sequence[State]
    scorer: PairScorer
    tau1: float
    tau2: float
    channel: Channel
    lambdas: Sequence[Lambda]
    arms: Sequence[Arm]
    wording: ExtWording
    e_labels: Mapping[str, EvalRecord]  # state_id -> E judgment (every state)
    expected_label: SafetyLabel | MathLabel  # the cohort's label: every state must carry it
    calibrator: Calibrator | None = None  # needed by the pi-hat arms
    perm_seed: int | None = None  # permuted-B's table (e_correct)
    pihat_perm_seed: int | None = None  # the mismatch arm's table
    run_seed: int | None = None
    # e_incorrect: the original run's channel.jsonl rows, matched bit for bit before any call
    original_channel: Sequence[Mapping[str, Any]] | None = None
    experiment_name: str = "exp1"  # the log label; the cohort's seed namespace is separate
    # None = the cohort's namespace. A smoke run passes "<namespace>_smoke" so that none of its
    # requests shares a seed or a cached answer with the formal run (II.4.4's rule).
    seed_namespace: str | None = None
    # stop point D's smoke (2026-10-31): states outside every formal set, as step 10a, so
    # there is no original channel row to match; the regression of stop point B covers that
    smoke: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def _bits(x: float | None) -> str:
    return "null" if x is None else struct.pack("<d", float(x)).hex()


CHANNEL_KEYS = ("s1", "s2", "s1_tilde", "s2_tilde")


def check_against_original(
    rows: Sequence[Mapping[str, Any]], original: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Every recomputed channel row equals the original run's, bit for bit (scores) and value
    for value (the assignment). The original must cover exactly the same (state, lambda)."""
    orig: dict[tuple[str, str], Mapping[str, Any]] = {}
    for r in original:
        key = (r["state_id"], r["lam"])
        if key in orig and dict(orig[key]) != dict(r):
            raise Exp1ExtError(f"original channel.jsonl has two different rows for {key}")
        orig[key] = r
    mine = {(r["state_id"], r["lam"]): r for r in rows}
    if set(mine) != set(orig):
        raise Exp1ExtError(
            f"the original run covers {len(orig)} (state, lambda), this run {len(mine)}; "
            f"e.g. only here {sorted(set(mine) - set(orig))[:3]}, "
            f"only there {sorted(set(orig) - set(mine))[:3]}"
        )
    bad: list[str] = []
    for key, r in mine.items():
        o = orig[key]
        if set(r) != set(o):
            bad.append(f"{key} fields {sorted(set(r) ^ set(o))}")
            continue
        for k in r:
            same = _bits(r[k]) == _bits(o.get(k)) if k in CHANNEL_KEYS else r[k] == o.get(k)
            if not same:
                bad.append(f"{key} {k}: {r[k]!r} != {o.get(k)!r}")
    if bad:
        raise Exp1ExtError(
            f"{len(bad)} channel value(s) differ from the original run: " + "; ".join(bad[:5])
        )
    return {"n_rows": len(mine), "bitwise_identical": True}


def _write_or_match(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a deterministic side stream once; on resume it must come out identical."""
    text = "".join(json.dumps(dict(r), ensure_ascii=False) + "\n" for r in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise Exp1ExtError(f"{path} on disk differs from this run's recomputation")
        return
    with path.open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _check_setup(setup: Exp1ExtSetup) -> Cohort:
    cohort = COHORTS.get(setup.cohort)
    if cohort is None:
        raise Exp1ExtError(f"unknown cohort {setup.cohort!r}; expected one of {sorted(COHORTS)}")
    stray = [a.value for a in setup.arms if a not in cohort.arms]
    if stray:
        raise Exp1ExtError(f"cohort {cohort.name} does not run arm(s) {stray} in the extension")
    missing = [s.state_id for s in setup.states if s.state_id not in setup.e_labels]
    if missing:
        raise Exp1ExtError(
            f"E judgments missing for {len(missing)} state(s) (they come before any controller "
            f"call), e.g. {missing[:5]}"
        )
    wrong = [
        s.state_id for s in setup.states if setup.e_labels[s.state_id].label != setup.expected_label
    ]
    if wrong:
        raise Exp1ExtError(
            f"{len(wrong)} state(s) of cohort {cohort.name} are not {setup.expected_label.value}"
            f" (e.g. {wrong[:3]})"
        )
    if any(a in PIHAT_ARMS for a in setup.arms):
        if setup.calibrator is None:
            raise Exp1ExtError("the pi-hat arms need the frozen calibrator")
        setup.calibrator.check_verifiers(setup.scorer.v1_id, setup.scorer.v2_id)
    if Arm.B_PIHAT_MISMATCHED in setup.arms and setup.pihat_perm_seed is None:
        raise Exp1ExtError("the pi-hat mismatch arm needs its own permutation seed")
    if Arm.PERMUTED_B in setup.arms and setup.perm_seed is None:
        raise Exp1ExtError("permuted-B needs its permutation seed")
    if setup.smoke and setup.original_channel is not None:
        raise Exp1ExtError("a smoke runs states outside the formal sets: no original channel")
    if cohort.reused_arms and setup.original_channel is None and not setup.smoke:
        raise Exp1ExtError(
            f"cohort {cohort.name} reuses the original decisions: the original channel rows are "
            "needed to check that the new arms see the same s_obs"
        )
    return cohort


def run_exp1_ext(
    client: LLMClient,
    ccfg: ctrl.ControllerConfig,
    ccfg_no_verifier: ctrl.ControllerConfig | None,
    setup: Exp1ExtSetup,
) -> dict[str, Any]:
    """Run the extension arms of one cohort into ``setup.run_dir``; returns the run summary."""
    cohort = _check_setup(setup)
    namespace = setup.seed_namespace or cohort.seed_namespace
    if Arm.NO_VERIFIER in setup.arms and ccfg_no_verifier is None:
        raise Exp1ExtError("no_verifier needs the No Verifier controller config")
    ccfg = ctrl.with_round_budget(ccfg, EXP1_DECISIONS)
    ccfg_nv = ctrl.with_round_budget(ccfg_no_verifier, EXP1_DECISIONS) if ccfg_no_verifier else None
    rd = Path(setup.run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    ids = [s.state_id for s in setup.states]

    # 1) raw scores once per state, then the channel for every (state, lambda), all up front
    pairs = setup.scorer.score_pair([s.text for s in setup.states])
    raw = dict(zip(ids, pairs, strict=True))
    builder = ObservationBuilder(
        setup.channel, setup.tau1, setup.tau2, setup.scorer.v1_id, setup.scorer.v2_id
    )
    obs: dict[tuple[str, Lambda], Observation] = {}
    fields: dict[tuple[str, Lambda], UncertaintyFields] = {}
    channel_rows: list[dict[str, Any]] = []
    for lam in setup.lambdas:
        for sid in ids:
            s1, s2 = raw[sid]
            o = builder.build(sid, s1, s2, lam)
            obs[(sid, lam)] = o
            fields[(sid, lam)] = compute_fields(o)
            channel_rows.append(
                {
                    "state_id": sid,
                    "lam": lam.value,
                    "s1": s1,
                    "s2": s2,
                    "s1_tilde": o.s1_obs,
                    "s2_tilde": o.s2_obs,
                    **builder.assignment(sid, lam),
                }
            )
    check = (
        check_against_original(channel_rows, setup.original_channel)
        if setup.original_channel is not None
        else None
    )
    pihat: dict[tuple[str, Lambda], PiHatDetail] = {}
    if any(a in PIHAT_ARMS for a in setup.arms):
        assert setup.calibrator is not None
        for (sid, lam), o in obs.items():
            pihat[(sid, lam)] = setup.calibrator.detail(o.s1_obs, o.s2_obs, o.tau1, o.tau2)
    perm = (
        permutation_for(rd / stream_filename(RecordKind.PERMUTATION), ids, setup.perm_seed)
        if Arm.PERMUTED_B in setup.arms and setup.perm_seed is not None
        else {}
    )
    pihat_donor = (
        permutation_for(rd / PIHAT_PERMUTATION, ids, setup.pihat_perm_seed)
        if Arm.B_PIHAT_MISMATCHED in setup.arms and setup.pihat_perm_seed is not None
        else {}
    )
    _write_or_match(rd / stream_filename(RecordKind.SCORE), [
        r.model_dump(mode="json", exclude={"timestamp"})
        for r in score_rows(setup.scorer, ids, pairs)
    ])  # fmt: skip
    _write_or_match(rd / stream_filename(RecordKind.CHANNEL), channel_rows)
    if pihat:
        _write_or_match(
            rd / PIHAT_STREAM,
            [{"state_id": sid, "lam": lam.value, **d.as_dict()} for (sid, lam), d in pihat.items()],
        )
    meta = {
        "run_id": setup.run_id,
        "cohort": cohort.name,
        "seed_namespace": namespace,
        "arms": [a.value for a in setup.arms],
        "reused_arms": [a.value for a in cohort.reused_arms],
        "lambdas": [lam.value for lam in setup.lambdas],
        "n_states": len(ids),
        "expected_label": setup.expected_label.value,
        "channel_check_against_original": (
            "not applicable: smoke states are outside D_Exp1" if setup.smoke else check
        ),
        "smoke": setup.smoke,
        "calibrator": setup.calibrator.provenance() if setup.calibrator else None,
        "pihat_clipped": {
            "v1": sum(d.clipped1 for d in pihat.values()),
            "v2": sum(d.clipped2 for d in pihat.values()),
            "n": len(pihat),
        },
        "perm_seed": setup.perm_seed if perm else None,
        "pihat_perm_seed": setup.pihat_perm_seed if pihat_donor else None,
        "run_seed": setup.run_seed,
        **setup.meta,
    }
    meta_path = rd / RUN_META
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old != json.loads(json.dumps(meta)):
            raise Exp1ExtError(f"{meta_path} records a different setup; resume refused")
    else:
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")

    # 2) decisions
    decisions_path = rd / stream_filename(RecordKind.DECISION)
    done = done_keys(decisions_path)
    n_new = 0
    with JsonlWriter(decisions_path) as dec_log:
        for lam in setup.lambdas:
            for s in setup.states:
                o = obs[(s.state_id, lam)]
                for arm in setup.arms:
                    key = (s.state_id, arm.value, lam.value, s.t)
                    if key in done:
                        continue
                    donor: str | None = None
                    run_cfg = ccfg
                    if arm is Arm.NO_VERIFIER:
                        assert ccfg_nv is not None
                        run_cfg = ccfg_nv
                        observation = format_extended(arm, None, setup.wording)
                    elif arm is Arm.VERDICT_ONLY:
                        observation = format_extended(arm, o, setup.wording)
                    elif arm is Arm.B_PIHAT:
                        d = pihat[(s.state_id, lam)]
                        observation = format_extended(
                            arm, o, setup.wording, pihat=PiHat(d.pihat1, d.pihat2)
                        )
                    elif arm is Arm.B_PIHAT_MISMATCHED:
                        donor = pihat_donor[s.state_id]
                        d = pihat[(donor, lam)]
                        observation = format_extended(
                            arm, o, setup.wording, donor_pihat=PiHat(d.pihat1, d.pihat2)
                        )
                    elif arm is Arm.PERMUTED_B:
                        donor = perm[s.state_id]
                        observation = format_observation(
                            arm, o, permuted_fields=fields[(donor, lam)]
                        )
                    else:
                        observation = format_observation(arm, o)
                    decision = ctrl.decide(
                        client,
                        run_cfg,
                        x0=s.x0,
                        x_t=s.text,
                        observation=observation,
                        labels=ctrl.DecisionLabels(
                            state_id=s.state_id, arm=arm, lam=lam, t=s.t, experiment=namespace,
                        ),
                    )  # fmt: skip
                    e = setup.e_labels[s.state_id]
                    dec_log.write(
                        make_log_record(
                            run_id=setup.run_id,
                            experiment=setup.experiment_name,
                            seed=setup.run_seed,
                            arm=arm,
                            lam=lam,
                            t=s.t,
                            state_id=s.state_id,
                            x0_id=s.x0_id,
                            x_t=s.text,
                            obs=o,
                            fields=fields[(s.state_id, lam)],
                            decision=decision,
                            permuted_from=donor,
                            e_label=e.label,
                            e_score=e.score,
                        )
                    )
                    done.add(key)
                    n_new += 1
    summary = summarize_exp1_ext(list(read_jsonl(decisions_path)), cohort.name)
    summary["n_new_decisions"] = n_new
    return summary


def summarize_exp1_ext(rows: Sequence[Mapping[str, Any]], cohort: str) -> dict[str, Any]:
    """Quick run report per arm x lambda (the analysis is analysis/ext_contrasts.py).

    ``e_incorrect``: false accept rate = P(accept); ``e_correct``: false reject rate =
    P(not accepted). parse_fail counts as not accepted (II.8.2's main accounting), so in
    ``e_correct`` it counts towards false reject; the rewrite-only rate is given next to it.
    """
    if cohort not in COHORTS:
        raise Exp1ExtError(f"unknown cohort {cohort!r}")
    cells: dict[str, dict[str, int]] = {}
    for r in rows:
        c = cells.setdefault(
            f"{r['arm']}|{r['lam']}", {"n": 0, "accept": 0, "rewrite": 0, "parse_fail": 0}
        )
        c["n"] += 1
        c["parse_fail"] += int(bool(r["parse_fail"]))
        c["accept"] += int((not r["parse_fail"]) and r["action"] == "accept")
        c["rewrite"] += int((not r["parse_fail"]) and r["action"] == "rewrite")
    out: dict[str, Any] = {"cohort": cohort, "cells": {}}
    for k, c in sorted(cells.items()):
        rates = (
            {"false_accept_rate": c["accept"] / c["n"]}
            if cohort == E_INCORRECT
            else {
                "false_reject_rate": (c["n"] - c["accept"]) / c["n"],
                "rewrite_rate": c["rewrite"] / c["n"],
            }
        )
        out["cells"][k] = {**c, **rates}
    out["parse_fail"] = parse_fail_cells(rows, PARSE_FAIL_ALERT_THRESHOLD)
    out["identical_to_low"] = identity_to_low_by_lambda(rows)
    out["n_rows"] = len(rows)
    return out


__all__ = [
    "COHORTS",
    "E_CORRECT",
    "E_INCORRECT",
    "NEW_ARMS",
    "ORIGINAL_ARMS",
    "PIHAT_PERMUTATION",
    "PIHAT_STREAM",
    "RUN_META",
    "Cohort",
    "Exp1ExtError",
    "Exp1ExtSetup",
    "check_against_original",
    "run_exp1_ext",
    "summarize_exp1_ext",
]
