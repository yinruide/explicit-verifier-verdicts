"""Exp 2 extension (post-pilot, 2026-10-31): full pairing over the starts. Code only in
this batch: nothing here has been run, and every run waits for the authors' go-ahead.

Not part of the primary runs: ``pipelines/step12_exp2.py``, its run directory, its assignment and
its statistics are only read. Two commands, each run in a directory of its own:

* ``loop --cohort e_incorrect --arms ...``: the 300 D_Exp1 starts (the original x1, sha256
  checked). The new arms (priority P0 ``no_verifier B_pihat``, P1 ``verdict_only``) run from
  every start; ``A1`` / ``B`` (P2) run each start only in the arm the original did not assign it.
  t = 1 scores are the original round-1 scores (``exp2.e_incorrect.t1_scores``; path and sha256
  in the run's metadata). Seed namespace ``exp2``.
* ``loop --cohort e_correct --arms ...``: the E-correct cohort's states as starts (P2), arms A1,
  B, No Verifier, Verdict-only, B + pi-hat; t = 1 scored in process once per start and compared
  bit for bit with step 6's D_gate scores (``t1_reference_check.json``, a record only).
  Seed namespace ``exp2_ecorrect``.
* ``judge --run-id ...``: E's post-loop pass on the laptop, as step 12's (t = 1 from step 5's y*:
  D_Exp1 under EXP2_T1, or D_gate under ECORRECT_COHORT).
* ``panel --run-id...``: every panel member (paper Sec. 3; detox: Grok 4.6) re-judges every x_T of a
  judged run on the laptop, as step 12's panel did for the original Exp 2 -- a sensitivity
  analysis (2026-10-31), read by ``analysis/ext_contrasts.py``.

``loop --smoke N`` is the engineering smoke (2026-10-31;
``pipelines/_ext_smoke.py``): N starts outside every formal set, as step 10a (E-incorrect: the
pool's E-UNSAFE states; E-correct: D_gate's SAFE states the cohort did not draw), every requested
arm from every start, t = 1 never reused (verifier arms score it in process and compare it with
step 6's table), seed namespace ``<namespace>_smoke``, a cache of its own, run directory
``runs/exp2ext_smoke_<cohort>_<ts>`` marked smoke in ``ext_meta.json``. ``judge`` and ``panel``
on a smoke run read its starts' y* from the same source (DRY_RUN / EXT_SMOKE) and keep its cache.

A loop whose arms are all ``no_verifier`` loads no verifier and needs no pod (laptop; controller
key from ``.env``). Any other loop is a pod run, exactly like step 12's: the formal pod check,
the 203-state numeric check against the probe before the first call, the OpenAI key from the pod's
environment (``--key-from-env``). Refused while ``configs/ext.yaml`` has a TBD or an extension
prompt file a placeholder, with a stub controller, and without ``--confirm``.

Three segments, as step 12's (2026-10-31): ``starts --cohort C --arms ... [--smoke N]``
on the laptop writes the loop's start manifest (``state_id``, ``x0_id``, ``x0``, ``text`` and the
arms of each start; no label) and its t = 1 file (the V1 / V2 scores of every start: the original
round 1 to reuse, or step 6's table to compare with) under ``data/ext/exp2_pod/``, each with a
``.sha256``; the RunPod driver (not released) pushes them and runs ``loop --starts-file F
--starts-sha256 H
--t1-file T --t1-sha256 H2 --run-dir D --tau-file P ...`` on the pod, which reads nothing else
of data/ or runs/; ``judge`` and ``panel`` run on the laptop. A loop with verifier arms takes its
inputs from a manifest only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from analysis._common import refuse_fake

from pipelines import _validity_io as io
from pipelines import step12_exp2 as S12
from pipelines._ext_smoke import SOURCES as SMOKE_SOURCES
from pipelines._ext_smoke import smoke_states
from pipelines.ext_prompt_regression import t1_scores_from
from rafr.calibration.apply import Calibrator
from rafr.contracts.schemas import Arm, Split, State
from rafr.controller import decide as ctrl
from rafr.core import paths
from rafr.core.config import assert_frozen, get, load_config, master_seed
from rafr.core.runtime import build_client, build_panel_client, build_pod_client
from rafr.core.seeding import component_seed
from rafr.data import labels as label_gate
from rafr.evaluator import judge
from rafr.evaluator import panel as panel_mod
from rafr.experiments import exp2_ext
from rafr.experiments.common import InProcessScorer
from rafr.experiments.ext_cohort import read_cohort
from rafr.experiments.ext_config import assert_ext_frozen, checked, load_ext, no_verifier_config
from rafr.formatter import TEMPLATES_PATH
from rafr.formatter.extended import detox_wording
from rafr.logging.writer import JsonlWriter, read_jsonl
from rafr.validity.tau import load_tau


def starts_for(ext: Mapping[str, Any], cohort: str) -> list[State]:
    if cohort == exp2_ext.E_INCORRECT:
        spec = get(ext, "exp2.e_incorrect")
        x1 = checked(spec["x1_file"], spec["x1_sha256"], what="Exp 2 x1")
        return [State.model_validate(r) for r in read_jsonl(x1)]
    spec = get(ext, "cohorts.e_correct")
    _, ids = read_cohort(paths.resolve(spec["file"]), spec["file_sha256"])
    by_id = {s.state_id: s for s in io.load_states(Split(spec["source_split"]))}
    return [by_id[i] for i in ids]


def plan_for(
    ext: Mapping[str, Any], cohort: str, arms: list[Arm], starts: list[State]
) -> dict[str, list[Arm]]:
    x0_ids = [s.x0_id for s in starts]
    if cohort == exp2_ext.E_CORRECT:
        return {x: list(arms) for x in x0_ids}
    spec = get(ext, "exp2.e_incorrect")
    new = [a for a in arms if a.value in spec["new_arms"]]
    fill = [a for a in arms if a.value in spec["fill_arms"]]
    original = None
    if fill:
        run = paths.resolve(spec["original_run"])
        f = checked(
            run / "assignment.jsonl", spec["original_sha256"]["assignment.jsonl"], what="assignment"
        )
        original = {r["x0_id"]: r["arm"] for r in read_jsonl(f)}
    return exp2_ext.full_pairing_plan(
        x0_ids, new_arms=new, fill_arms=fill, original_assignment=original
    )


STARTS_DIR_NAME = "exp2_pod"
"""``data/ext/exp2_pod/``: the label-free start manifests and t = 1 files of the pod loops."""
START_FIELDS = ("state_id", "x0_id", "x0", "text")
POD_CHECKS = "pod_checks.jsonl"
"""One row per loop start on the pod: the fixed-pod check and the 203-state numeric check. Kept
out of ``ext_meta.json`` so that a resumed run (same setup) is not refused for a new timestamp."""


def starts_dir() -> Path:
    return paths.DATA / "ext" / STARTS_DIR_NAME


def manifest_name(cohort: str, arms: Sequence[Arm], smoke: bool) -> str:
    return f"{cohort}{'_smoke' if smoke else ''}_{'-'.join(a.value for a in arms)}"


def manifest_paths(name: str) -> tuple[Path, Path]:
    """(start manifest, t = 1 file) of a manifest name."""
    return starts_dir() / f"{name}.starts.jsonl", starts_dir() / f"{name}.t1.jsonl"


def sidecar_sha256(path: Path) -> str:
    side = path.with_name(path.name + ".sha256")
    if not side.is_file():
        raise SystemExit(f"{side} not found: run `ext_exp2.py starts` first")
    return side.read_text(encoding="utf-8").split()[0]


def _write_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Write ``rows`` (JSON lines) unless the file already holds exactly them; refuse a file with
    other content. Writes ``<file>.sha256`` next to it and returns the sha256."""
    text = "".join(json.dumps(dict(r), ensure_ascii=False) + "\n" for r in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{path} exists with other content; move it away to rewrite it")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{sha}  {path.name}\n", encoding="utf-8", newline="\n"
    )
    return sha


def _read_checked(path: str | Path, sha256: str | None, what: str) -> tuple[dict, list[dict]]:
    p = Path(path)
    if not sha256:
        raise SystemExit(f"{what}: give its sha256")
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != sha256:
        raise SystemExit(f"{what}: {p} has sha256 {got}, expected {sha256}")
    rows = list(read_jsonl(p))
    if not rows or not rows[0].get("header"):
        raise SystemExit(f"{what}: {p} has no header row")
    return rows[0], rows[1:]


@dataclass
class LoopInputs:
    cohort: str
    arms: list[Arm]
    smoke: bool
    namespace: str
    starts: list[State]
    plan: dict[str, list[Arm]]
    t1_scores: dict[str, tuple[float, float]] | None
    t1_reference: dict[str, tuple[float, float]] | None
    meta: dict[str, Any]


def _arms(args: argparse.Namespace, ext: Mapping[str, Any], cohort: str) -> list[Arm]:
    allowed = get(ext, f"exp2.{cohort}.arms") or [
        *get(ext, "exp2.e_incorrect.new_arms"),
        *get(ext, "exp2.e_incorrect.fill_arms"),
    ]
    arms = [Arm(a) for a in (args.arms or allowed)]
    stray = [a.value for a in arms if a.value not in allowed]
    if stray:
        raise SystemExit(f"arm(s) {stray} are not in ext.yaml's exp2.{cohort} arms")
    return arms


def laptop_inputs(
    args: argparse.Namespace, cfgs: Mapping[str, Any], ext: Mapping[str, Any]
) -> LoopInputs:
    """Starts, plan and t = 1 scores from the laptop's data/ and runs/ (``starts``, and a
    No Verifier loop on the laptop). The only path that reads y* (a smoke's candidates)."""
    if args.cohort is None:
        raise SystemExit(f"{args.command} needs --cohort")
    cohort = exp2_ext.COHORTS[args.cohort]
    arms = _arms(args, ext, args.cohort)
    namespace = str(get(ext, f"exp2.{args.cohort}.seed_namespace"))
    if namespace != cohort.seed_namespace:
        raise SystemExit(f"ext.yaml's namespace {namespace} is not the runner's")
    verifier_arms = [a for a in arms if a is not Arm.NO_VERIFIER]
    smoke = args.smoke is not None
    if smoke:
        ecfg = judge.build_config(cfgs["models"])
        starts, _ = smoke_states(ext, args.cohort, args.smoke, master_seed(cfgs), ecfg)
        plan = {s.x0_id: list(arms) for s in starts}
        namespace = f"{namespace}_smoke"
    else:
        starts = starts_for(ext, args.cohort)
        plan = plan_for(ext, args.cohort, arms, starts)
    v1_id, v2_id = io.verifier_ids(cfgs["models"])
    t1_scores = t1_reference = None
    meta: dict[str, Any] = {"plan_priority": get(ext, "exp2.priority"), "smoke": smoke}
    if smoke:
        meta["smoke_states"] = [s.state_id for s in starts]
    t1 = get(ext, f"exp2.{args.cohort}.t1_scores")
    if verifier_arms and t1.get("reuse") and not smoke:
        src = checked(t1["source"], t1["source_sha256"], what="t = 1 scores")
        t1_scores = t1_scores_from(src, v1_id, v2_id)
        meta["t1_scores_source"] = {"file": t1["source"], "sha256": t1["source_sha256"]}
    elif verifier_arms and (t1.get("compare_with_step06") or smoke):
        ref1, ref2 = io.load_scores_for(starts, v1_id), io.load_scores_for(starts, v2_id)
        t1_reference = {s.x0_id: (ref1[s.state_id], ref2[s.state_id]) for s in starts}
    return LoopInputs(args.cohort, arms, smoke, namespace, starts, plan, t1_scores,
                      t1_reference, meta)  # fmt: skip


def manifest_inputs(
    args: argparse.Namespace, cfgs: Mapping[str, Any], ext: Mapping[str, Any]
) -> LoopInputs:
    """The pod's inputs: the start manifest and the t = 1 file ``starts`` wrote on the laptop,
    each checked against its sha256. Nothing under data/ or runs/ is read, and no label."""
    header, rows = _read_checked(args.starts_file, args.starts_sha256, "start manifest")
    cohort = str(header["cohort"])
    if args.cohort is not None and args.cohort != cohort:
        raise SystemExit(f"the manifest is cohort {cohort}, --cohort says {args.cohort}")
    arms = [Arm(a) for a in header["arms"]]
    if args.arms and [Arm(a) for a in args.arms] != arms:
        raise SystemExit(f"the manifest's arms are {header['arms']}, --arms says {args.arms}")
    smoke = bool(header["smoke"])
    namespace = str(header["seed_namespace"])
    expected = exp2_ext.COHORTS[cohort].seed_namespace + ("_smoke" if smoke else "")
    if namespace != expected:
        raise SystemExit(f"the manifest's namespace {namespace} is not {expected}")
    starts = [State(**{k: r[k] for k in START_FIELDS}) for r in rows]
    plan = {r["x0_id"]: [Arm(a) for a in r["arms"]] for r in rows}
    meta: dict[str, Any] = {
        "plan_priority": get(ext, "exp2.priority"),
        "smoke": smoke,
        "starts_file": {"name": header["name"], "sha256": args.starts_sha256},
    }
    if smoke:
        meta["smoke_states"] = [s.state_id for s in starts]
    t1_header, t1_rows = _read_checked(args.t1_file, args.t1_sha256, "t = 1 file")
    if t1_header["name"] != header["name"]:
        raise SystemExit(f"the t = 1 file is {t1_header['name']}'s, not {header['name']}'s")
    if [t1_header["v1_id"], t1_header["v2_id"]] != list(io.verifier_ids(cfgs["models"])):
        raise SystemExit("the t = 1 file's verifiers are not the configured V1 / V2")
    pairs = {r["x0_id"]: (float(r["s1"]), float(r["s2"])) for r in t1_rows}
    meta["t1_file"] = {"use": t1_header["use"], "sha256": args.t1_sha256,
                       "source": t1_header["source"]}  # fmt: skip
    t1_scores = pairs if t1_header["use"] == "reuse" else None
    t1_reference = pairs if t1_header["use"] == "reference" else None
    if t1_scores is not None:
        meta["t1_scores_source"] = t1_header["source"]
    return LoopInputs(cohort, arms, smoke, namespace, starts, plan, t1_scores, t1_reference,
                      meta)  # fmt: skip


def cmd_starts(args: argparse.Namespace, cfgs: Mapping[str, Any], ext: Mapping[str, Any]) -> int:
    """Laptop: the start manifest and t = 1 file of one pod loop, under ``data/ext/exp2_pod/``.
    The manifest holds ``state_id``, ``x0_id``, ``x0``, ``text`` and the arms run from each start;
    the t = 1 file the V1 / V2 scores of every start (the original Exp 2's round 1 to reuse, or
    step 6's table to compare with). Neither carries a label."""
    inp = laptop_inputs(args, cfgs, ext)
    if not [a for a in inp.arms if a is not Arm.NO_VERIFIER]:
        raise SystemExit("a No Verifier loop runs on the laptop; it needs no start manifest")
    name = manifest_name(inp.cohort, inp.arms, inp.smoke)
    starts_path, t1_path = manifest_paths(name)
    use = "reuse" if inp.t1_scores is not None else "reference"
    pairs = inp.t1_scores if inp.t1_scores is not None else inp.t1_reference
    assert pairs is not None
    planned = [s for s in inp.starts if inp.plan.get(s.x0_id)]
    header = {
        "header": True,
        "name": name,
        "cohort": inp.cohort,
        "arms": [a.value for a in inp.arms],
        "smoke": inp.smoke,
        "seed_namespace": inp.namespace,
        "n_starts": len(planned),
        "n_trajectories": sum(len(inp.plan[s.x0_id]) for s in planned),
        "t1": use,
    }
    rows = [
        {**{k: getattr(s, k) for k in START_FIELDS}, "arms": [a.value for a in inp.plan[s.x0_id]]}
        for s in planned
    ]
    v1_id, v2_id = io.verifier_ids(cfgs["models"])
    source = (
        inp.meta["t1_scores_source"]
        if use == "reuse"
        else {"file": "step 6's scores tables (data/scores)", "sha256": None}
    )
    t1_header = {"header": True, "name": name, "use": use, "v1_id": v1_id, "v2_id": v2_id,
                 "source": source}  # fmt: skip
    t1_rows = [{"x0_id": s.x0_id, "s1": pairs[s.x0_id][0], "s2": pairs[s.x0_id][1]}
               for s in planned]  # fmt: skip
    sha_s = _write_once(starts_path, [header, *rows])
    sha_t = _write_once(t1_path, [t1_header, *t1_rows])
    print(f"[ext_exp2] {name}: {header['n_starts']} starts, {header['n_trajectories']} "
          f"trajectories, t = 1 {use}")  # fmt: skip
    print(f"[ext_exp2] {starts_path} sha256 {sha_s}")
    print(f"[ext_exp2] {t1_path} sha256 {sha_t}")
    return 0


def cmd_loop(args: argparse.Namespace, cfgs: Mapping[str, Any], ext: Mapping[str, Any]) -> int:
    if args.starts_file:
        if args.smoke is not None:
            raise SystemExit("--smoke comes from the start manifest; drop it")
        inp = manifest_inputs(args, cfgs, ext)
    else:
        inp = laptop_inputs(args, cfgs, ext)
    verifier_arms = [a for a in inp.arms if a is not Arm.NO_VERIFIER]
    if verifier_arms and not args.starts_file:
        raise SystemExit(
            "a loop with verifier arms runs on the pod from a start manifest: run `starts` on "
            "the laptop, then env/ext_exp2_gpu.py (the pod gets no data/, no label)"
        )
    smoke, namespace, arms = inp.smoke, inp.namespace, inp.arms
    master = master_seed(cfgs)
    prefix = str(get(ext, "exp2.run_id_prefix", "exp2ext"))
    kind = f"smoke_{inp.cohort}" if smoke else inp.cohort
    run_id = args.run_id or f"{prefix}_{kind}_{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}"
    if not run_id.startswith(f"{prefix}_smoke_" if smoke else f"{prefix}_"):
        raise SystemExit(f"run id {run_id} does not name a{' smoke' if smoke else 'n'} "
                         f"extension run ({prefix}_{'smoke_' if smoke else ''}...)")  # fmt: skip
    rd = Path(args.run_dir) if args.run_dir else paths.run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    refuse_fake(rd, what="ext_exp2")
    # step 12's snapshot: on the pod (git archive, no .git) the commit is RAFR_GIT_HEAD
    S12.snapshot(rd, step="ext_exp2_loop", cfgs={**dict(cfgs), "ext": dict(ext)}, extra={})
    taus = load_tau(Path(args.tau_file) if args.tau_file else paths.FROZEN_TAU)
    meta = inp.meta
    calibrator = None
    if Arm.B_PIHAT in arms:
        cal = ext["calibration"]
        calibrator = Calibrator.load(
            paths.resolve(cal["frozen_file"]), expected_sha256=cal["frozen_sha256"]
        )
    scorer = None
    v1 = v2 = None
    if verifier_arms:
        if not args.key_from_env:
            raise SystemExit("a loop with verifier arms runs on the pod: add --key-from-env")
        if not (args.numeric_check and args.numeric_check_sha256):
            raise SystemExit("the pod loop needs --numeric-check and --numeric-check-sha256")
        formal_pod = S12.check_pod(cfgs["models"], print)
        records = list(args.records) if args.records else None
        v1 = S12.load_verifier(cfgs["models"], "v1", records=records)
        v2 = S12.load_verifier(cfgs["models"], "v2", records=records)
        numeric = S12.check_numbers([v1, v2], args.numeric_check, args.numeric_check_sha256, print)
        with JsonlWriter(rd / POD_CHECKS) as w:
            w.write({"at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
                     "formal_pod": formal_pod, "numeric_check": numeric})  # fmt: skip
        meta["v1"], meta["v2"] = v1.describe(), v2.describe()
        scorer = InProcessScorer(v1, v2)
        cache = rd / "cache" if smoke else Path(args.cache_dir or rd / "cache")
        client = build_pod_client(cfgs["models"], run_dir=rd, cache_dir=cache)
    else:  # laptop: No Verifier only
        client = build_client(cfgs["models"], run_dir=rd, cache_dir=rd / "cache" if smoke else None)
    refusal = get(cfgs["generation"], "g_fixed.validity.refusal_phrases_file")
    base_seed = component_seed(master, "controller")
    ccfg = (
        ctrl.build_config(
            cfgs["models"], cfgs["exp2"], base_seed=base_seed, refusal_phrases_file=refusal
        )
        if verifier_arms
        else None
    )
    ccfg_nv = (
        no_verifier_config(
            cfgs["models"], cfgs["exp2"], ext, base_seed=base_seed, refusal_phrases_file=refusal
        )
        if Arm.NO_VERIFIER in arms
        else None
    )
    prompts = ext["prompts"]
    setup = exp2_ext.Exp2ExtSetup(
        run_id=run_id,
        run_dir=rd,
        cohort=inp.cohort,
        starts=inp.starts,
        plan=inp.plan,
        scorer=scorer,
        tau1=taus["v1"].tau,
        tau2=taus["v2"].tau,
        k=int(get(ext, "exp2.k")),
        wording=detox_wording(
            verdict_path=paths.resolve(prompts["verdict_only_file"]),
            pihat_path=paths.resolve(prompts["pihat_file"]),
        ),
        judge_fn=S12._no_e,
        calibrator=calibrator,
        run_seed=component_seed(master, "exp2_run"),
        t1_scores=inp.t1_scores,
        t1_reference=inp.t1_reference,
        seed_namespace=namespace,
        meta=meta,
    )
    printer = S12.round_printer(rd, str(get(cfgs["models"], "controller.model")), print)
    try:
        exp2_ext.run_loop_ext(client, ccfg, ccfg_nv, setup, on_round=printer)
    finally:
        for v in (v1, v2):
            if v is not None:
                v.close()
    print(f"[ext_exp2] loop complete: {rd}")
    return 0


def cmd_judge(args: argparse.Namespace, cfgs: Mapping[str, Any], ext: Mapping[str, Any]) -> int:
    if not args.run_id:
        raise SystemExit("judge needs --run-id")
    rd = _existing_ext_run(args.run_id, ext)
    meta = json.loads((rd / exp2_ext.RUN_META).read_text(encoding="utf-8"))
    cohort = str(meta["cohort"])
    smoke = bool(meta.get("smoke"))
    plan = {r["x0_id"]: [Arm(a) for a in r["arms"]] for r in read_jsonl(rd / exp2_ext.PLAN)}
    ecfg = judge.build_config(cfgs["models"])
    if smoke:
        src = SMOKE_SOURCES[cohort]
        by_x0 = {s.x0_id: s for s in io.load_states(src.split)}
        starts = [by_x0[x] for x in plan]
        prior = S12.prior_from_y_star(starts, ecfg, split=src.split, purpose=src.purpose)
    elif cohort == exp2_ext.E_INCORRECT:
        starts = starts_for(ext, cohort)
        prior = S12.prior_from_y_star(starts, ecfg)
    else:
        starts = starts_for(ext, cohort)
        prior = S12.prior_from_y_star(
            starts, ecfg, split=Split.D_GATE, purpose=label_gate.LabelPurpose.ECORRECT_COHORT
        )
    client = build_client(cfgs["models"], run_dir=rd, cache_dir=rd / "cache" if smoke else None)
    setup = exp2_ext.Exp2ExtSetup(
        run_id=args.run_id,
        run_dir=rd,
        cohort=cohort,
        starts=starts,
        plan=plan,
        scorer=None,
        tau1=0.0,
        tau2=0.0,
        k=int(meta["k"]),
        wording=detox_wording(),
        judge_fn=lambda text: judge.judge_text(client, ecfg, text),
        prior_judgments=prior,
        quality_fn=S12.make_quality_fn(cfgs),
        seed_namespace=str(meta["seed_namespace"]),
    )
    summary = exp2_ext.finish_ext(setup)
    (rd / "summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8", newline="\n"
    )
    print(f"[ext_exp2] E judged {rd}: {json.dumps(summary['arms'])}")
    return 0


def _existing_ext_run(run_id: str | None, ext: Mapping[str, Any]) -> Path:
    """A finished loop's run directory: an extension Exp 2 run (``<prefix>_*`` with its
    ``ext_meta.json``); judge and panel never create one."""
    if not run_id:
        raise SystemExit("this command needs --run-id")
    rd = paths.RUNS / run_id
    prefix = str(get(ext, "exp2.run_id_prefix", "exp2ext"))
    if not run_id.startswith(f"{prefix}_") or not (rd / exp2_ext.RUN_META).is_file():
        raise SystemExit(f"{rd} is not an extension Exp 2 run directory ({prefix}_*)")
    return rd


def cmd_panel(args: argparse.Namespace, cfgs: Mapping[str, Any], ext: Mapping[str, Any]) -> int:
    """paper Sec. 3, 4.6: every panel member re-judges every x_T of a judged extension run, exactly
    as step 12's ``panel`` did for the original Exp 2 (2026-10-31: a sensitivity
    analysis). One ``panel_<member>.jsonl`` per member in the run directory; resumable."""
    rd = _existing_ext_run(args.run_id, ext)
    meta = json.loads((rd / exp2_ext.RUN_META).read_text(encoding="utf-8"))
    finals_path = rd / "trajectories.jsonl"
    if not (rd / "judge_complete.json").is_file() or not finals_path.is_file():
        raise SystemExit(f"{rd}: run `judge` first (the panel re-judges the judged x_T)")
    members = panel_mod.build_members(cfgs["models"], allow_tbd=False)
    rule = panel_mod.on_failure_rule(cfgs["models"])
    finals = list(read_jsonl(finals_path))
    smoke = bool(meta.get("smoke"))
    client = build_panel_client(
        cfgs["models"], run_dir=rd, cache_dir=rd / "cache" if smoke else None
    )
    out = panel_mod.rejudge_x_t(client, members, finals, rd, on_failure=rule, echo=print)
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "labels"} for k, v in out.items()}
    (rd / "panel_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8", newline="\n"
    )
    print(f"[ext_exp2] panel re-judged {rd}: {json.dumps(summary)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=("starts", "loop", "judge", "panel"))
    ap.add_argument("--cohort", choices=sorted(exp2_ext.COHORTS), default=None)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--smoke", type=int, default=None, help="starts / loop: smoke on N starts")
    ap.add_argument("--starts-file", default=None, help="loop on the pod: the start manifest")
    ap.add_argument("--starts-sha256", default=None)
    ap.add_argument("--t1-file", default=None, help="loop on the pod: the t = 1 scores file")
    ap.add_argument("--t1-sha256", default=None)
    ap.add_argument("--run-dir", default=None, help="loop on the pod: the remote run directory")
    ap.add_argument("--tau-file", default=None, help="loop on the pod: the pushed frozen tau")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--confirm", action="store_true", help="required: real API calls cost money")
    ap.add_argument("--key-from-env", action="store_true", help="pod: OPENAI_API_KEY from env")
    ap.add_argument("--records", type=Path, nargs="+", help="the probe's records directories")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--numeric-check", default=None)
    ap.add_argument("--numeric-check-sha256", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfgs = load_config()
    ext = load_ext()
    assert_ext_frozen(ext)
    S12.refuse_stub_controller(cfgs["models"])
    if args.command in ("loop", "starts"):
        assert_frozen(cfgs, extra_files=[TEMPLATES_PATH], only_sections=S12.LOOP_SECTIONS)
    elif args.command == "judge":
        assert_frozen(cfgs, only_sections=S12.JUDGE_SECTIONS)
    if args.smoke is not None and args.command not in ("loop", "starts"):
        raise SystemExit("--smoke belongs to starts / loop; judge and panel read it from the run")
    if args.command == "starts":  # writes two files, calls nothing
        return cmd_starts(args, cfgs, ext)
    if not args.confirm:
        raise SystemExit("this command spends money: add --confirm after the authors' go-ahead")
    if args.command == "loop":
        return cmd_loop(args, cfgs, ext)
    if args.command == "panel":
        return cmd_panel(args, cfgs, ext)
    return cmd_judge(args, cfgs, ext)


if __name__ == "__main__":
    sys.exit(main())
