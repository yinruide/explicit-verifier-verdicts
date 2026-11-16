"""Offline-testable math extension loop; E is a separate, post-loop boundary.

The original math round writer, scoring barrier, overlength handling and controller are
shared. This module binds extension inputs, formats added arms and records extra diagnostics.
It does not construct a provider, choose an E implementation, or authorize a live run.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pipelines.math_exp2_loop import OVERLENGTH, K, ScoreFn, text_key
from pipelines.math_ext_data import validate_output
from pipelines.math_ext_e import check_e_callback
from pipelines.math_ext_execution import (
    check_controller,
    check_execution,
    check_extension_config,
    check_mode,
)
from pipelines.math_ext_exp1 import check_client_output_paths
from pipelines.math_ext_loop import run_rounds, validate_round_chain
from pipelines.math_ext_snapshot import snapshot_configs, write_or_match
from rafr.calibration.apply import Calibrator
from rafr.channel import ObservationBuilder
from rafr.channel.identity import IdentityChannel
from rafr.contracts.schemas import Arm, Lambda, Score, State
from rafr.domains.math.storage import IntegrityError, atomic_json, digest, file_hash, read_json
from rafr.experiments.exp2 import ROUND_COMPLETE, Trajectory, complete_rounds, load_round, round_dir
from rafr.experiments.exp2_ext import (
    COHORTS,
    E_CORRECT,
    E_INCORRECT,
    FILL_ARMS,
    NEW_ARMS,
    _t1_check,
    _write_meta,
    _write_plan,
    full_pairing_plan,
)
from rafr.formatter import math_feedback
from rafr.formatter.extended import ExtWording, PiHat, format_extended
from rafr.formatter.fields import compute_fields
from rafr.logging.writer import JsonlWriter, read_jsonl


def plan_for_cohort(ids, cohort, *, original_assignment=None, mode="all"):
    """Core full pairing, with the math cohort/assignment boundary checked first."""
    if not ids or len(ids) != len(set(ids)):
        raise IntegrityError("starts must be nonempty and unique")
    if cohort not in COHORTS or mode not in ("new", "fill", "all"):
        raise IntegrityError("unknown cohort or extension mode")
    if cohort == E_CORRECT:
        if mode != "all" or original_assignment is not None:
            raise IntegrityError("E-correct runs all arms without an original assignment")
        return full_pairing_plan(ids, new_arms=COHORTS[cohort].arms)
    if mode in ("fill", "all"):
        if original_assignment is None or set(original_assignment) != set(ids):
            raise IntegrityError("original assignment must cover exactly these starts")
        if any(Arm(a) not in FILL_ARMS for a in original_assignment.values()):
            raise IntegrityError("original math assignment must be A1/B")
    return full_pairing_plan(
        ids,
        new_arms=NEW_ARMS if mode != "fill" else (),
        fill_arms=FILL_ARMS if mode != "new" else (),
        original_assignment=original_assignment,
    )


@dataclass(frozen=True)
class T1Reuse:
    """Explicit opt-in for an identity-bound export, never inferred from matching text.

    Export schema: scoring_identity and states [{x0_id, problem, text, scores:[Score,Score]}].
    The producing workflow must bind the original source artifacts in scoring_identity;
    matching that identity is required in addition to the export's exact byte hash.
    """

    path: Path
    sha256: str
    approved: bool = False


@dataclass
class MathExtSetup:
    run_id: str
    run_dir: Path
    cohort: str
    starts: Sequence[State]
    plan: Mapping[str, Sequence[Arm]]
    taus: tuple[float, float]
    verifier_ids: tuple[str, str]
    verifier_versions: tuple[str, str]
    score_texts: ScoreFn | None
    wording: ExtWording
    run_seed: int
    # LOCAL E context only. A pod setup leaves this empty; never exported in loop metadata.
    # The post-loop grader binds these original y* separately, through the label-purpose gate.
    initial_grades: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    calibrator: Calibrator | None = None
    smoke: bool = True
    source_identity: dict[str, Any] = field(default_factory=dict)
    scoring_identity: dict[str, Any] = field(default_factory=dict)
    t1_reuse: T1Reuse | None = None
    execution_mode: str = "offline_stub"
    t1_reference: Mapping[str, tuple[float, float]] | None = None
    extension_config: dict[str, Any] | None = None
    channel_config: dict[str, Any] | None = None
    confirmed: bool = False


def setup_from_public_bundle(
    bundle,
    *,
    run_id,
    run_dir,
    taus,
    score_texts,
    wording,
    extension_config,
    channel_config,
    calibrator=None,
    confirmed=False,
):
    """Assemble the loop from validated public pod transport.

    No local references, y* or original-run directory is needed. Live execution additionally
    requires an explicit operator confirmation; fixture transport stays offline.
    """
    manifest = bundle.manifest
    check_mode(manifest["source_mode"], confirmed=confirmed, source_mode=manifest["source_mode"])
    if manifest["fixture"] != (manifest["source_mode"] == "offline_stub"):
        raise IntegrityError("public bundle fixture/source mode mismatch")
    reused = manifest["t1_mode"] == "reuse_original_round1"
    reference = (
        None
        if reused
        else {
            item["x0_id"]: tuple(row["score"] for row in item["scores"])
            for item in bundle.t1_payload["states"]
        }
    )
    return MathExtSetup(
        run_id=run_id,
        run_dir=Path(run_dir),
        cohort=manifest["cohort"],
        starts=bundle.states,
        plan=bundle.plan,
        taus=tuple(taus),
        verifier_ids=tuple(manifest["verifier_ids"]),
        verifier_versions=tuple(manifest["verifier_versions"]),
        score_texts=score_texts,
        wording=wording,
        run_seed=manifest["master_seed"],
        smoke=manifest["smoke"],
        calibrator=calibrator,
        source_identity={
            "source_identity_sha256": manifest["source_identity_sha256"],
            "source_execution_mode": manifest["source_mode"],
        },
        scoring_identity=bundle.t1_payload["scoring_identity"],
        t1_reuse=T1Reuse(bundle.t1_path, manifest["files"]["t1.json"], approved=True)
        if reused
        else None,
        t1_reference=reference,
        extension_config=dict(extension_config),
        channel_config=dict(channel_config),
        execution_mode=manifest["source_mode"],
        confirmed=confirmed,
    )


def _validate_grade(record):
    if not isinstance(record, dict):
        raise IntegrityError("E grade must be a record")
    status = record.get("status")
    if status not in ("correct", "incorrect", "unresolved"):
        raise IntegrityError("E did not return a terminal math grade")
    if record.get("task_success") is not (status == "correct"):
        raise IntegrityError("E task_success disagrees with status")
    if "is_correct" in record and record["is_correct"] is not (
        None if status == "unresolved" else status == "correct"
    ):
        raise IntegrityError("E is_correct disagrees with status")
    allowed = {
        "status",
        "is_correct",
        "task_success",
        "prediction_answer",
        "math_verify_version",
        "n_boxes",
        "boxes_all_equivalent",
        "e_timeout",
    }
    if set(record) - allowed:
        raise IntegrityError("unexpected E grade fields")


def _initial_grade_map(setup):
    if set(setup.initial_grades) != {s.x0_id for s in setup.starts}:
        raise IntegrityError("bound initial grades must cover every start exactly")
    expected = "correct" if setup.cohort == E_CORRECT else "incorrect"
    for record in setup.initial_grades.values():
        _validate_grade(record)
        if record["status"] != expected:
            raise IntegrityError("original initial grade disagrees with selected cohort")
    return {text_key(s.x0_id, s.text): dict(setup.initial_grades[s.x0_id]) for s in setup.starts}


def _reuse_rows(setup):
    reuse = setup.t1_reuse
    if reuse is None:
        return {}
    if not reuse.approved or setup.cohort != E_INCORRECT or setup.smoke:
        raise IntegrityError("math t1 reuse requires explicit approval for E-incorrect")
    if not setup.scoring_identity or file_hash(reuse.path) != reuse.sha256:
        raise IntegrityError("t1 reuse hash or scoring identity is missing/changed")
    payload = read_json(reuse.path)
    if payload["scoring_identity"] != setup.scoring_identity:
        raise IntegrityError("t1 scoring identity differs (prompt/shape/version/source)")
    starts = {s.x0_id: s for s in setup.starts}
    rows = {}
    for item in payload["states"]:
        s = starts.get(item["x0_id"])
        if s is None or (s.x0, s.text) != (item["problem"], item["text"]):
            raise IntegrityError("t1 source question or initial text differs")
        key = text_key(s.x0_id, s.text)
        if key in rows:
            raise IntegrityError("duplicate t1 source state")
        pair = tuple(Score.model_validate(r) for r in item["scores"])
        if len(pair) != 2 or tuple(r.verifier_id for r in pair) != setup.verifier_ids:
            raise IntegrityError("t1 verifier identity differs")
        if tuple(r.model_version for r in pair) != setup.verifier_versions:
            raise IntegrityError("t1 verifier version differs")
        if any(r.state_id != key for r in pair):
            raise IntegrityError("t1 score state key differs from question/text")
        rows[key] = pair
    needed = {
        text_key(s.x0_id, s.text)
        for s in setup.starts
        if set(setup.plan[s.x0_id]) - {Arm.NO_VERIFIER}
    }
    if not needed <= rows.keys():
        raise IntegrityError("t1 reuse is missing planned verifier starts")
    return rows


def run_math_exp2(client, cfg, cfg_no_verifier, setup: MathExtSetup, *, on_round=None):
    """Return final public trajectories after four rounds; no E or references enter here."""
    check_execution(
        client,
        setup.execution_mode,
        confirmed=setup.confirmed,
        source_mode=setup.source_identity.get("source_execution_mode", "offline_stub"),
    )
    if setup.execution_mode == "live":
        check_extension_config(setup.extension_config)
        if setup.channel_config is None:
            raise IntegrityError("live loop requires bound channel configuration for its snapshot")
        if cfg is not None:
            check_controller(cfg, seed=setup.run_seed, rounds=K)
        if cfg_no_verifier is not None:
            check_controller(cfg_no_verifier, seed=setup.run_seed, rounds=K, no_verifier=True)
    ids = [s.x0_id for s in setup.starts]
    if not ids or len(set(ids)) != len(ids) or set(setup.plan) != set(ids):
        raise IntegrityError("plan must cover unique nonempty starts exactly")
    if setup.initial_grades:
        _initial_grade_map(setup)
    if setup.cohort not in COHORTS:
        raise IntegrityError("unknown math extension cohort")
    planned = {Arm(a) for arms in setup.plan.values() for a in arms}
    if any(not arms or len(arms) != len(set(arms)) for arms in setup.plan.values()):
        raise IntegrityError("empty or duplicated trajectory arms")
    if planned - set(COHORTS[setup.cohort].arms):
        raise IntegrityError("arm is not an Exp2 cohort arm")
    if any(s.t != 1 or s.source != "g_fixed" for s in setup.starts):
        raise IntegrityError("extension starts must be G_fixed x1")
    if planned - {Arm.NO_VERIFIER} and (setup.score_texts is None or cfg is None):
        raise IntegrityError("verifier arms require scoring and controller configuration")
    if Arm.NO_VERIFIER in planned and cfg_no_verifier is None:
        raise IntegrityError("NoVerifier needs separate prompts")
    for config in (cfg, cfg_no_verifier):
        if config is not None and config.round_budget != K:
            raise IntegrityError("math Exp2 controller budget must be four")
    if setup.calibrator is not None and setup.calibrator.record.get("domain") != "math":
        raise IntegrityError("math extension requires a math-domain calibrator")
    if Arm.B_PIHAT in planned:
        if setup.calibrator is None:
            raise IntegrityError("B+pihat requires frozen calibration")
        setup.calibrator.check_verifiers(*setup.verifier_ids)
    reused = _reuse_rows(setup)
    if setup.t1_reference is not None:
        if reused:
            raise IntegrityError("t1 reuse and fresh reference comparison are mutually exclusive")
        if set(setup.t1_reference) != set(ids):
            raise IntegrityError("t1 reference must cover exactly the starting questions")
    namespace = "math_exp2" if setup.cohort == E_INCORRECT else "math_exp2_ecorrect"
    if setup.smoke:
        namespace = "math_exp2ext_smoke"
    source_root = setup.source_identity.get("source_root")
    source_digest = setup.source_identity.get("source_identity_sha256", "")
    if (
        not setup.smoke
        and not source_root
        and not (isinstance(source_digest, str) and re.fullmatch(r"[0-9a-f]{64}", source_digest))
    ):
        raise IntegrityError("non-smoke requires bound source provenance")
    rd = validate_output(
        Path(setup.run_dir),
        source_root=Path(source_root) if source_root else None,
        allow_existing=True,
    )
    check_client_output_paths(client, rd)
    if (setup.extension_config is None) != (setup.channel_config is None):
        raise IntegrityError("snapshot needs both extension and channel config")
    if setup.extension_config is not None:
        snapshot_configs(
            rd, setup.extension_config, setup.channel_config, identity=setup.source_identity
        )
    rd.mkdir(parents=True, exist_ok=True)
    feedback_hashes = {
        name: file_hash(math_feedback.PROMPT_DIR / name) for name in math_feedback.PROMPT_FILES
    }
    wording_hash = _wording_identity(setup)
    # Core metadata/plan persistence refuses changed resume identities.
    _write_meta(
        rd,
        {
            "run_id": setup.run_id,
            "cohort": setup.cohort,
            "namespace": namespace,
            "SMOKE": setup.smoke,
            "smoke": setup.smoke,
            "domain": "math",
            "execution_mode": setup.execution_mode,
            "k": K,
            "seed": setup.run_seed,
            "starts": [[s.x0_id, s.x0, s.text] for s in setup.starts],
            "tau": list(setup.taus),
            "verifier_ids": list(setup.verifier_ids),
            "verifier_versions": list(setup.verifier_versions),
            "controller": asdict(cfg) if cfg else None,
            "no_verifier_controller": asdict(cfg_no_verifier) if cfg_no_verifier else None,
            "calibrator": setup.calibrator.provenance() if setup.calibrator else None,
            "source_identity": setup.source_identity,
            "scoring_identity": setup.scoring_identity,
            "t1_reference": dict(setup.t1_reference) if setup.t1_reference is not None else None,
            "t1_reuse": {"path": str(setup.t1_reuse.path), "sha256": setup.t1_reuse.sha256}
            if setup.t1_reuse
            else None,
            "wording_sha256": wording_hash,
            "original_feedback_sha256": feedback_hashes,
        },
    )
    _write_plan(rd, setup.plan)

    def scoring(items):
        # De-duplicate only within a question. Same text under another question is distinct.
        unique = {item[3]: item for item in items}
        missing = [item for key, item in unique.items() if key not in reused]
        fresh, over = setup.score_texts(missing) if missing else ({}, {})
        found = {key: reused[key] for key in unique if key in reused} | dict(fresh)
        if set(found) & set(over) or set(found) | set(over) != set(unique):
            raise IntegrityError("scorer must return exactly one score or overlength per state")
        for key, pair in found.items():
            if tuple(r.model_version for r in pair) != setup.verifier_versions:
                raise IntegrityError("scorer version differs from frozen identity")
            if any(r.state_id != key for r in pair):
                raise IntegrityError("score state key differs from requested question/text")
        return found, over

    initial = [
        Trajectory(s.x0_id, s.x0, Arm(a).value, 1, s.text)
        for s in setup.starts
        for a in setup.plan[s.x0_id]
    ]
    trajectories = run_rounds(
        rd,
        initial,
        taus=setup.taus,
        verifier_ids=setup.verifier_ids,
        score_texts=scoring,
        verifier_versions=setup.verifier_versions,
        client=client,
        cfg=cfg,
        no_verifier_cfg=cfg_no_verifier,
        feedback=lambda arm, obs: _feedback(setup, arm, obs),
        namespace=namespace,
        experiment="exp2",
        run_id=setup.run_id,
        seed=setup.run_seed,
        smoke=setup.smoke,
        k=K,
        on_round=on_round,
    )
    if feedback_hashes != {
        name: file_hash(math_feedback.PROMPT_DIR / name) for name in math_feedback.PROMPT_FILES
    } or (wording_hash != _wording_identity(setup)):
        raise IntegrityError("math feedback changed during loop; results require review")
    _pihat_diagnostics(setup)
    _t1_diagnostics(setup)
    return trajectories


def _feedback(setup, arm, obs):
    """One renderer for execution and request replay at the local E boundary."""
    if arm is Arm.A1:
        return setup.wording.render_a1(obs)
    if arm is Arm.B:
        return setup.wording.render_b(obs)
    pi = None
    if arm is Arm.B_PIHAT:
        d = setup.calibrator.detail(obs.s1_obs, obs.s2_obs, *setup.taus)
        pi = PiHat(d.pihat1, d.pihat2)
    return format_extended(arm, obs, setup.wording, pihat=pi)


def _t1_diagnostics(setup):
    """Use Core's bitwise diagnostic on fresh t1 scores, never replace them with step06."""
    if setup.t1_reference is None:
        return
    rd = Path(setup.run_dir)
    keyed = {}
    for raw in read_jsonl(round_dir(rd, 1) / "scores.jsonl"):
        row = Score.model_validate(raw)
        key = row.state_id, row.verifier_id
        if key in keyed:
            raise IntegrityError("duplicate first-round score row")
        keyed[key] = row.score
    by_question = {}
    for start in setup.starts:
        for arm in setup.plan[start.x0_id]:
            if Arm(arm) is Arm.NO_VERIFIER:
                continue
            sid = f"{start.x0_id}-{Arm(arm).value}-t1"
            pair = tuple(keyed[sid, v] for v in setup.verifier_ids)
            previous = by_question.get(start.x0_id)
            if previous is not None and tuple(v.hex() for v in previous) != tuple(
                v.hex() for v in pair
            ):
                raise IntegrityError("t1 shared scores differ across arms")
            by_question[start.x0_id] = pair
    # Core helper writes its report. Stage it in temporary storage so an existing run
    # report is checked before any replacement, including on a completed-loop resume.
    with tempfile.TemporaryDirectory(prefix="math-ext-t1-") as temporary:
        report = _t1_check(Path(temporary), by_question, setup.t1_reference)
    write_or_match(rd / "t1_reference_check.json", json.dumps(report, indent=1) + "\n")


def _wording_identity(setup):
    builder = ObservationBuilder(IdentityChannel(), *setup.taus, *setup.verifier_ids)
    # Functions themselves cannot be serialized; render all flag/conflict cases plus margins.
    return digest(
        [
            [
                setup.wording.render_a1(obs),
                setup.wording.render_b(obs),
                setup.wording.render_verdict(compute_fields(obs)),
            ]
            for a, b in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.37, 0.63))
            for obs in [builder.build("identity", a, b, Lambda.NONE)]
        ]
        + [setup.wording.pihat_template, setup.wording.separator]
    )


def _pihat_diagnostics(setup):
    if setup.calibrator is None:
        return
    rd = Path(setup.run_dir)
    expected = []
    for row in read_jsonl(rd / "decisions.jsonl"):
        if row["arm"] == Arm.B_PIHAT.value:
            detail = setup.calibrator.detail(row["s1_tilde"], row["s2_tilde"], *setup.taus)
            expected.append(
                {
                    "state_id": row["state_id"],
                    "x0_id": row["x0_id"],
                    "t": row["t"],
                    **detail.as_dict(),
                }
            )
    path = rd / "pihat.jsonl"
    existing = list(read_jsonl(path))
    if existing != expected[: len(existing)]:
        raise IntegrityError("changed pihat side stream")
    with JsonlWriter(path) as writer:
        writer.write_many(expected[len(existing) :])


def grade_math_exp2(setup, trajectories, grade_callback, *, e_identity):
    """Post-loop E callback follows MathPipeline._grade_after_loop(qid,text) result shape.

    Callback MUST own a separate extension E context. Do not bind the original source
    MathPipeline instance: its grade methods append to that original run's evaluation log.
    ``ExtensionMathEvaluator`` supplies the extension-owned Math-Verify context; it is
    exercised here on constructed inputs. A real-run CLI assembly remains separate.
    Original x1 grades are reused from setup.initial_grades; the callback sees only new
    candidate texts. Callback owns retries and private references. Only terminal correct,
    incorrect or unresolved are accepted. Provider/backend failures propagate. Durable
    records join by (question,text) and trajectory, so a resumed completed grade is reused.
    """
    check_mode(
        setup.execution_mode,
        confirmed=setup.confirmed,
        source_mode=setup.source_identity.get("source_execution_mode", "offline_stub"),
    )
    initial = _initial_grade_map(setup)
    source_root = setup.source_identity.get("source_root")
    rd = validate_output(
        Path(setup.run_dir),
        source_root=Path(source_root) if source_root else None,
        allow_existing=True,
    )
    check_e_callback(
        grade_callback,
        run_dir=rd,
        source_root=source_root,
        e_identity=e_identity,
        live=setup.execution_mode == "live",
    )
    if setup.execution_mode == "live":
        check_extension_config(setup.extension_config)
        if setup.channel_config is None:
            raise IntegrityError("live E requires the loop's bound channel configuration")
    if setup.extension_config is not None:
        if not (rd / "snapshot.json").is_file() or not (rd / "input_snapshot.json").is_file():
            raise IntegrityError("E requires the original loop execution snapshot")
        snapshot_configs(
            rd, setup.extension_config, setup.channel_config, identity=setup.source_identity
        )
    if complete_rounds(rd, K) != list(range(1, K + 1)):
        raise IntegrityError("E runs only after all four complete rounds")
    if [asdict(tr) for tr in trajectories] != [asdict(tr) for tr in load_round(rd, K)]:
        raise IntegrityError("E trajectories differ from completed round")
    loop_meta = read_json(rd / "ext_meta.json")
    if loop_meta.get("original_feedback_sha256") != {
        name: file_hash(math_feedback.PROMPT_DIR / name) for name in math_feedback.PROMPT_FILES
    } or loop_meta.get("wording_sha256") != _wording_identity(setup):
        raise IntegrityError("math feedback differs from loop identity")
    if any(
        loop_meta.get(key) != value
        for key, value in {
            "run_id": setup.run_id,
            "cohort": setup.cohort,
            "starts": [[s.x0_id, s.x0, s.text] for s in setup.starts],
            "source_identity": setup.source_identity,
            "execution_mode": setup.execution_mode,
            "SMOKE": setup.smoke,
        }.items()
    ):
        raise IntegrityError("E setup differs from completed loop identity")
    from rafr.controller.decide import ControllerConfig

    cfg = ControllerConfig(**loop_meta["controller"]) if loop_meta["controller"] else None
    nv = (
        ControllerConfig(**loop_meta["no_verifier_controller"])
        if loop_meta["no_verifier_controller"]
        else None
    )
    if setup.execution_mode == "live":
        for controller, no_verifier in ((cfg, False), (nv, True)):
            if controller is not None:
                check_controller(controller, seed=setup.run_seed, rounds=K, no_verifier=no_verifier)
    validate_round_chain(
        rd,
        [
            Trajectory(s.x0_id, s.x0, Arm(a).value, 1, s.text)
            for s in setup.starts
            for a in setup.plan[s.x0_id]
        ],
        run_id=setup.run_id,
        seed=setup.run_seed,
        smoke=setup.smoke,
        taus=setup.taus,
        verifier_ids=setup.verifier_ids,
        require_complete=True,
        verifier_versions=setup.verifier_versions,
        cfg=cfg,
        no_verifier_cfg=nv,
        feedback=lambda arm, obs: _feedback(setup, arm, obs),
        namespace="math_exp2ext_smoke"
        if setup.smoke
        else ("math_exp2" if setup.cohort == E_INCORRECT else "math_exp2_ecorrect"),
    )
    marker = rd / "judge_complete.json"
    sealed_names = [
        "e_identity.json",
        "e_initial_grades.json",
        "math_judgments.jsonl",
        "math_results.json",
        "trajectories.jsonl",
        "ext_meta.json",
        "plan.jsonl",
    ] + [str((round_dir(rd, t) / ROUND_COMPLETE).relative_to(rd)) for t in range(1, K + 1)]
    if marker.exists():
        seal = read_json(marker)
        hashes = seal.get("files", {})
        if set(hashes) != set(sealed_names) or any(
            not (rd / name).is_file() or file_hash(rd / name) != expected
            for name, expected in hashes.items()
        ):
            raise IntegrityError("completed E artifacts changed; resume refused")
    identity_path = rd / "e_identity.json"
    initial_path = rd / "e_initial_grades.json"
    if initial_path.exists() and read_json(initial_path) != dict(setup.initial_grades):
        raise IntegrityError("E setup differs from bound original initial grades")
    if not initial_path.exists():
        atomic_json(initial_path, dict(setup.initial_grades))
    if not e_identity:
        raise IntegrityError("E identity is required")
    if identity_path.exists() and read_json(identity_path) != e_identity:
        raise IntegrityError("E identity changed")
    if not identity_path.exists():
        if (rd / "math_judgments.jsonl").exists():
            raise IntegrityError("persisted E judgments have no E identity")
        atomic_json(identity_path, e_identity)
    path = rd / "math_judgments.jsonl"
    requested = {
        text_key(tr.x0_id, text): tr.x0_id
        for tr in trajectories
        for text in [tr.x_t, *(h["x_t"] for h in tr.history)]
    }

    known = {}
    for row in read_jsonl(path):
        if set(row) != {"text_key", "x0_id", "grade"}:
            raise IntegrityError("unexpected persisted E judgment fields")
        key = row["text_key"]
        if key not in requested or row["x0_id"] != requested[key]:
            raise IntegrityError("persisted E judgment has unexpected question/text key")
        if key in known:
            raise IntegrityError("duplicate E judgment")
        _validate_grade(row["grade"])
        if key in initial and row["grade"] != initial[key]:
            raise IntegrityError("cached initial grade differs from bound original y_star")
        known[key] = row["grade"]
    if marker.exists() and set(known) != set(requested):
        raise IntegrityError("completed E judgment coverage differs from trajectories")
    result = []
    with JsonlWriter(path) as writer:

        def grade(qid, text):
            key = text_key(qid, text)
            if key not in known:
                record = initial[key] if key in initial else dict(grade_callback(qid, text))
                _validate_grade(record)
                writer.write({"text_key": key, "x0_id": qid, "grade": record})
                known[key] = record
            return known[key]

        for tr in trajectories:
            rounds = []
            for i, h in enumerate(tr.history):
                after = tr.history[i + 1]["x_t"] if i + 1 < len(tr.history) else tr.x_t
                rounds.append(
                    {
                        "t": h["t"],
                        "action": h["action"],
                        "parse_fail": h["parse_fail"],
                        "input_solution": h["x_t"],
                        "output_solution": after,
                        "input_grade": grade(tr.x0_id, h["x_t"]),
                        "output_grade": grade(tr.x0_id, after),
                    }
                )
            terminal = grade(tr.x0_id, tr.x_t)
            overlength = tr.history[-1].get(OVERLENGTH) if tr.history else None
            result.append(
                {
                    "x0_id": tr.x0_id,
                    "question_id": tr.x0_id,
                    "arm": tr.arm,
                    "final_text": tr.x_t,
                    "x0": tr.x0,
                    "x_T": tr.x_t,
                    "terminal_solution": tr.x_t,
                    "domain": "math",
                    "execution_mode": setup.execution_mode,
                    "cohort": setup.cohort,
                    "initial_status": setup.initial_grades[tr.x0_id]["status"],
                    "terminal_status": terminal["status"],
                    "rounds": rounds,
                    "termination_reason": "accepted"
                    if tr.accepted
                    else OVERLENGTH
                    if overlength
                    else "decision_limit",
                    "accepted": tr.accepted,
                    "accepted_at": tr.accepted_at,
                    "n_rewrites": tr.n_rewrites,
                    "n_parse_fail": tr.n_parse_fail,
                    OVERLENGTH: overlength,
                    "unresolved": terminal["status"] == "unresolved",
                    **terminal,
                }
            )
    if marker.exists():
        if read_json(rd / "math_results.json") != result:
            raise IntegrityError("completed E results disagree with durable judgments")
    else:
        atomic_json(rd / "math_results.json", result)
        # Consumer contract: the authors' analysis reads this filename, not math_results.
        # Core keeps raw trajectories inside rounds/; assembling decision/score streams
        # does not touch this post-loop export. Partial exports are never consumed.
        partial = rd / "trajectories.jsonl.partial"
        partial.unlink(missing_ok=True)
        with JsonlWriter(partial) as writer:
            writer.write_many(result)
        os.replace(partial, rd / "trajectories.jsonl")
        atomic_json(
            marker,
            {
                "files": {name: file_hash(rd / name) for name in sealed_names},
                "execution_mode": setup.execution_mode,
                "SMOKE": setup.smoke,
                "n_trajectories": len(result),
            },
        )
    return result
