"""Exp 1 extension (post-pilot, 2026-10-31): the four new arms on two cohorts.

Not part of the primary runs: ``pipelines/step11_exp1.py``, its run directory and its report are
only read. One cohort per run, each in a run directory of its own:

* ``--cohort e_incorrect``: D_Exp1, the arms No Verifier, Verdict-only, B + pi-hat and
  B + pi-hat (mismatched); A1 / B / permuted-B are the original run's decisions (read only). The
  channel is recomputed and checked bit for bit against the original ``channel.jsonl`` before any
  controller call. Outcome: D_Exp1's y* (purpose EXP1_OUTCOME), as step 11.
* ``--cohort e_correct``: the E-correct cohort file (``configs/ext.yaml: cohorts.e_correct``), all
  seven arms. Outcome: those states' y* (purpose ECORRECT_COHORT); every one must be SAFE.

Refused while ``configs/ext.yaml`` has a TBD (the frozen calibrator's and the cohort file's
sha256, the contrast list's confirmation) or an extension prompt file a placeholder; also refused
with a stub controller and without ``--confirm`` (controller calls cost money).

``--smoke N`` is the engineering smoke (2026-10-31; ``pipelines/_ext_smoke.py``):
N states outside every formal set, as step 10a -- E-incorrect from the pool's E-UNSAFE states,
E-correct from D_gate's SAFE states the cohort did not draw -- with their y* as the outcome; the
controller seed namespace ``<namespace>_smoke`` and a cache of its own (no request can share a
seed or a cached answer with the formal run); its own run directory
``runs/exp1ext_smoke_<cohort>_<ts>``. Its states are not D_Exp1's, so the check of the channel
against the original run does not apply (the regression check against the primary runs covers all
2,700 cells).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from datetime import UTC, datetime
from typing import Any

from analysis._common import refuse_fake

from pipelines import _validity_io as io
from pipelines import step11_exp1 as S11
from pipelines._ext_smoke import smoke_states
from rafr.calibration.apply import Calibrator
from rafr.channel.degrade import DegradeChannel
from rafr.contracts.schemas import Arm, Lambda, SafetyLabel, Split, State
from rafr.controller import decide as ctrl
from rafr.controller.stub import is_stub_controller
from rafr.core import paths
from rafr.core.config import assert_frozen, get, load_config, master_seed
from rafr.core.runtime import build_client
from rafr.core.seeding import component_seed
from rafr.data import labels as L
from rafr.evaluator import judge
from rafr.experiments import exp1_ext
from rafr.experiments.ext_cohort import read_cohort
from rafr.experiments.ext_config import (
    assert_ext_frozen,
    checked,
    load_ext,
    no_verifier_config,
)
from rafr.formatter import TEMPLATES_PATH
from rafr.formatter.extended import PIHAT_ARMS, detox_wording
from rafr.logging.snapshot import write_snapshot
from rafr.logging.writer import read_jsonl
from rafr.validity.tau import load_tau


def cohort_states(ext: dict[str, Any], cohort: str) -> list[State]:
    if cohort == exp1_ext.E_INCORRECT:
        return io.load_states(Split(get(ext, "cohorts.e_incorrect.states_split")))
    spec = get(ext, "cohorts.e_correct")
    _, ids = read_cohort(paths.resolve(spec["file"]), spec["file_sha256"])
    by_id = {s.state_id: s for s in io.load_states(Split(spec["source_split"]))}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} cohort state(s) are not D_gate states: {missing[:3]}")
    return [by_id[i] for i in ids]


def cohort_outcome(states: list[State], cohort: str, ecfg: judge.EvaluatorConfig) -> dict:
    """state_id -> E's judgment, recorded before any controller call ."""
    if cohort == exp1_ext.E_INCORRECT:
        return S11.outcome_from_y_star(states, ecfg)
    out = {}
    for y in L.read_labels(Split.D_GATE, L.LabelPurpose.ECORRECT_COHORT):
        judge.check_same_e(judge.record_identity(y.record), ecfg, where="D_gate's y* (step05)")
        out[y.state_id] = y.record
    return {s.state_id: out[s.state_id] for s in states}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cohort", required=True, choices=sorted(exp1_ext.COHORTS))
    ap.add_argument("--arms", nargs="+", default=None, help="a subset of the cohort's arms")
    ap.add_argument("--smoke", type=int, default=None, help="engineering smoke on N states")
    ap.add_argument("--run-id", default=None, help="resume an existing run directory")
    ap.add_argument("--confirm", action="store_true", help="required: controller calls cost money")
    args = ap.parse_args(argv)
    cfgs = load_config()
    ext = load_ext()
    assert_ext_frozen(ext)
    assert_frozen(cfgs, extra_files=[TEMPLATES_PATH], only_sections=S11.GUARD_SECTIONS)
    model = str(get(cfgs["models"], "controller.model", ""))
    if is_stub_controller(model):
        raise SystemExit(f"models.controller.model = {model!r} is a stub; refused (II.4.4)")
    S11.preconditions()
    if not args.confirm:
        raise SystemExit("controller calls cost money: re-run with --confirm after the authors' go")

    cohort = exp1_ext.COHORTS[args.cohort]
    arms = [Arm(a) for a in (args.arms or get(ext, f"exp1.{args.cohort}.arms"))]
    namespace = str(get(ext, f"exp1.{args.cohort}.seed_namespace"))
    if namespace != cohort.seed_namespace:
        raise SystemExit(f"ext.yaml's namespace {namespace} is not the runner's")
    master = master_seed(cfgs)
    ecfg = judge.build_config(cfgs["models"])
    smoke = args.smoke is not None
    if smoke:
        states, e_labels = smoke_states(ext, args.cohort, args.smoke, master, ecfg)
        namespace = f"{namespace}_smoke"
    else:
        states = cohort_states(ext, args.cohort)
        e_labels = cohort_outcome(states, args.cohort, ecfg)
    prefix = str(get(ext, "exp1.run_id_prefix", "exp1ext"))
    kind = f"smoke_{args.cohort}" if smoke else args.cohort
    run_id = args.run_id or f"{prefix}_{kind}_{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}"
    rd = paths.run_dir(run_id)
    refuse_fake(rd, what="ext_exp1")
    write_snapshot(rd, step="ext_exp1", configs={**cfgs, "ext": ext}, extra={"cohort": args.cohort})

    original_channel = None
    if cohort.reused_arms and not smoke:
        spec = get(ext, "exp1.e_incorrect")
        run = paths.resolve(spec["original_run"])
        for name, sha in spec["original_sha256"].items():
            checked(run / name, sha, what=f"original Exp 1 {name}")
        original_channel = list(read_jsonl(run / "channel.jsonl"))
    calibrator = None
    if any(a in PIHAT_ARMS for a in arms):
        cal = ext["calibration"]
        calibrator = Calibrator.load(
            paths.resolve(cal["frozen_file"]), expected_sha256=cal["frozen_sha256"]
        )
    v1, v2 = io.verifier_ids(cfgs["models"])
    s1, s2 = io.load_scores_for(states, v1), io.load_scores_for(states, v2)
    taus = load_tau(paths.FROZEN_TAU)
    refusal = get(cfgs["generation"], "g_fixed.validity.refusal_phrases_file")
    base_seed = component_seed(master, "controller")
    ccfg = ctrl.build_config(
        cfgs["models"], cfgs["exp1"], base_seed=base_seed, refusal_phrases_file=refusal
    )
    ccfg_nv = (
        no_verifier_config(
            cfgs["models"], cfgs["exp1"], ext, base_seed=base_seed, refusal_phrases_file=refusal
        )
        if Arm.NO_VERIFIER in arms
        else None
    )
    prompts = ext["prompts"]
    setup = exp1_ext.Exp1ExtSetup(
        run_id=run_id,
        run_dir=rd,
        cohort=args.cohort,
        states=states,
        scorer=S11.TableScorer(v1, v2, s1, s2, {s.text: s.state_id for s in states}),
        tau1=taus["v1"].tau,
        tau2=taus["v2"].tau,
        channel=DegradeChannel.load(paths.FROZEN_CHANNEL),
        lambdas=[Lambda(x) for x in get(ext, "exp1.lambdas")],
        arms=arms,
        wording=detox_wording(
            verdict_path=paths.resolve(prompts["verdict_only_file"]),
            pihat_path=paths.resolve(prompts["pihat_file"]),
        ),
        e_labels=e_labels,
        expected_label=(
            SafetyLabel.UNSAFE if args.cohort == exp1_ext.E_INCORRECT else SafetyLabel.SAFE
        ),
        calibrator=calibrator,
        perm_seed=component_seed(master, str(get(ext, "exp1.permuted_b_seed_component"))),
        pihat_perm_seed=component_seed(master, str(get(ext, "exp1.pihat_mismatch_seed_component"))),
        run_seed=component_seed(master, "exp1_run"),
        original_channel=original_channel,
        seed_namespace=namespace,
        smoke=smoke,
        meta={"smoke_states": [s.state_id for s in states]} if smoke else {},
    )
    client = build_client(cfgs["models"], run_dir=rd, cache_dir=rd / "cache" if smoke else None)
    summary = exp1_ext.run_exp1_ext(client, ccfg, ccfg_nv, setup)
    (rd / "summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8", newline="\n"
    )
    for cell, c in summary["cells"].items():
        print(f"[ext_exp1] {cell}: {json.dumps(c)}")
    print(f"[ext_exp1] ledger: {json.dumps(client._ledger.summary() if client._ledger else {})}")
    print(f"[ext_exp1] -> {rd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
