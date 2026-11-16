"""Math post-pilot Exp1 wiring, with read-only replay of the original math artifacts.

Core owns controller/cache/logs/decisions. This entry adapts only State-aware initial
scores, mathematical wording, label purposes and the original math artifact layout.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from pipelines.math_ext_data import MathSource, validate_output
from pipelines.math_ext_execution import check_execution, check_extension_config
from pipelines.math_ext_exp1_runner import (
    COHORTS,
    E_CORRECT,
    E_INCORRECT,
    ORIGINAL_ARMS,
    Exp1ExtError,
    Exp1ExtSetup,
    run_exp1_ext,
)
from pipelines.math_ext_snapshot import snapshot_configs
from pipelines.math_ext_transport import _read_cohort
from rafr.calibration.apply import Calibrator
from rafr.channel import ObservationBuilder
from rafr.contracts.schemas import Arm, Lambda, LogRecord, MathLabel, Split, State
from rafr.controller import decide as ctrl
from rafr.core.seeding import component_seed
from rafr.data.labels import LabelPurpose
from rafr.domains.math.generation import build_controller_config
from rafr.domains.math.storage import IntegrityError, atomic_json, file_hash, read_json
from rafr.formatter import math_feedback
from rafr.formatter.fields import compute_fields
from rafr.formatter.math_extended import (
    PROMPT_DIR,
    PROMPT_FILES,
    SHARED_DIR,
    SHARED_FILES,
    build_no_verifier_config,
    math_wording,
    original_feedback,
)
from rafr.formatter.permute import derangement
from rafr.llm.client import LLMClient


@dataclass(frozen=True)
class BoundScoreIdentity:
    """Metadata for Core score rows; text-only scoring is intentionally impossible."""

    v1_id: str
    v2_id: str
    v1_version: str
    v2_version: str

    def score_pair(self, texts):
        raise Exp1ExtError("math scores require full State identity, never text-only lookup")


def _bits(value) -> bytes:
    return struct.pack("<d", float(value))


def reconstruct_original_channel(
    source: MathSource,
    states: Sequence[State],
    *,
    lambdas: Sequence[Lambda] = (Lambda.LOW, Lambda.MID, Lambda.HIGH),
    controller_cfg: ctrl.ControllerConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """No calls/writes: compare channel observations to original A1 decision rows.

    The original channel.jsonl is an assignment log, not the observed evidence. It is
    deliberately not used as a channel reference. Independently replay
    every original arm's request and permutation to protect prompt and seed identity.
    """
    cfg = controller_cfg or build_controller_config(source.master_seed, 1)
    ids = [s.state_id for s in states]
    if not ids or len(ids) != len(set(ids)) or any(s.t != 1 for s in states):
        raise Exp1ExtError("original replay requires unique t=1 initial states")
    decision_rows = source.read_bound_rows("exp1/decisions.jsonl")
    donors = source.read_bound_json("exp1/permutation.json")
    marker = source.read_bound_json("exp1/COMPLETE.json")
    hashes = source.provenance()["source_sha256"]
    if (
        marker.get("n_questions") != len(ids)
        or marker.get("n_decisions") != len(ids) * len(lambdas) * len(ORIGINAL_ARMS)
        or marker.get("decisions_sha256") != hashes["exp1/decisions.jsonl"]
        or marker.get("channel_sha256") != hashes["exp1/channel.jsonl"]
    ):
        raise Exp1ExtError("original Exp1 completion marker disagrees with bound artifacts")
    expected_donors = derangement(ids, component_seed(source.master_seed, "permuted_b"))
    if donors != expected_donors:
        raise Exp1ExtError("original math donor permutation differs from seeded reconstruction")
    expected = {
        (sid, lam.value, arm.value) for sid in ids for lam in lambdas for arm in ORIGINAL_ARMS
    }

    def indexed(rows, name):
        result = {}
        for row in rows:
            key = row["state_id"], row["lam"], row["arm"]
            if key in result:
                raise Exp1ExtError(f"duplicate original {name} row: {key}")
            result[key] = row
        if set(result) != expected:
            raise Exp1ExtError(f"original {name} coverage differs")
        return result

    decisions = indexed(decision_rows, "decision")
    pairs = dict(zip(ids, source.pairs(states), strict=True))
    builder = ObservationBuilder(source.channel(), *source.taus, *source.verifier_ids)
    obs = {
        (s.state_id, lam): builder.build(s.state_id, *pairs[s.state_id], lam)
        for s in states
        for lam in lambdas
    }
    canonical = []
    n_attempts = 0
    for lam in lambdas:
        for state in states:
            sid = state.state_id
            o = obs[sid, lam]
            assign = builder.assignment(sid, lam)
            canonical.append(
                {
                    "state_id": sid,
                    "lam": lam.value,
                    "s1": o.raw_s1,
                    "s2": o.raw_s2,
                    "s1_tilde": o.s1_obs,
                    "s2_tilde": o.s2_obs,
                    **assign,
                }
            )
            for arm in ORIGINAL_ARMS:
                key = sid, lam.value, arm.value
                row = LogRecord.model_validate(decisions[key])
                donor = obs[donors[sid], lam] if arm is Arm.PERMUTED_B else None
                for name, value in {
                    "s1": o.raw_s1,
                    "s2": o.raw_s2,
                    "s1_tilde": o.s1_obs,
                    "s2_tilde": o.s2_obs,
                    "tau1": o.tau1,
                    "tau2": o.tau2,
                }.items():
                    if getattr(row, name) is None or _bits(getattr(row, name)) != _bits(value):
                        raise Exp1ExtError(f"original decision score differs bitwise: {key} {name}")
                if (
                    row.t != 1
                    or row.x0_id != state.x0_id
                    or row.x_t != state.text
                    or row.seed != source.master_seed
                    or row.model != cfg.model
                    or row.decoding != cfg.decoding
                    or (row.v1_id, row.v2_id) != source.verifier_ids
                    or row.fields != compute_fields(donor or o)
                    or row.permuted_from_state_id != (donor.state_id if donor else None)
                ):
                    raise Exp1ExtError(f"original decision identity/fields differ: {key}")
                feedback = original_feedback(arm, o, donor)
                labels = ctrl.DecisionLabels(sid, arm, lam, 1, "math_exp1")
                seed = ctrl.call_seed(cfg, labels)
                if not row.attempts:
                    raise Exp1ExtError(f"original decision has no attempts: {key}")
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
                        raise Exp1ExtError(
                            f"original request prompt/seed differs: {key} attempt {i}"
                        )
                    n_attempts += 1
                if row.prompt_hash != row.attempts[-1].prompt_hash:
                    raise Exp1ExtError(f"original final prompt hash differs: {key}")
    source.assert_unchanged()
    return canonical, {
        "n_states": len(ids),
        "n_original_decisions": len(decisions),
        "n_attempts": n_attempts,
        "bitwise_scores": True,
        "channel_reference": "exp1/decisions.jsonl:A1 observed evidence",
        "n_channel_observations": len(canonical),
        "permutation_identical": True,
        "request_prompts_and_seeds_identical": True,
        "execution": "read_only_reconstruction_no_provider_calls",
    }


def check_client_output_paths(client: LLMClient, run_dir: Path) -> None:
    """Core transport side effects must be local to this extension run.

    LLMClient exposes ledger publicly; its existing cache/transient paths are inspected
    read-only because Core has no public cache-path accessor. No transport is initialized.
    """
    root = Path(run_dir).resolve()
    cache = getattr(client, "_cache", None)
    ledger = client.ledger
    transient = getattr(client, "_transient_log", None)
    outputs = []
    if cache is not None:
        if not hasattr(cache, "root"):
            raise Exp1ExtError("client cache has no auditable output root")
        outputs.append(("cache", cache.root))
    if ledger is not None:
        if not hasattr(ledger, "path"):
            raise Exp1ExtError("client ledger has no auditable output path")
        outputs.append(("ledger", ledger.path))
    if transient is not None:
        outputs.append(("transient log", transient))
    for kind, value in outputs:
        raw = Path(value).absolute()
        if any(p.is_symlink() for p in (raw, *raw.parents)):
            raise Exp1ExtError(f"client {kind} output traverses a symlink")
        path = raw.resolve()
        if not path.is_relative_to(root) or (path == root and kind != "cache"):
            raise Exp1ExtError(f"client {kind} output must be inside the extension run directory")


def run_math_exp1_extension(
    client: LLMClient,
    source: MathSource,
    *,
    run_dir: Path,
    run_id: str,
    cohort: str,
    state_ids: Sequence[str] | None = None,
    cohort_file: Path | None = None,
    cohort_sha256: str | None = None,
    arms: Sequence[Arm] | None = None,
    lambdas: Sequence[Lambda] = (Lambda.LOW, Lambda.MID, Lambda.HIGH),
    calibrator: Calibrator | None = None,
    prompt_dir: Path | None = None,
    smoke: bool = False,
    allow_draft: bool = False,
    metadata: Mapping[str, Any] | None = None,
    execution_mode: str = "offline_stub",
    extension_config: Mapping[str, Any] | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Execute only approved/injected controller transport; offline drafts require fake-only."""
    source.assert_unchanged()
    run_dir = validate_output(run_dir, source_root=source.root, allow_existing=True)
    check_client_output_paths(client, run_dir)
    completion = run_dir / "math_ext_complete.json"
    if completion.exists():
        files = read_json(completion).get("files", {})
        if not files or any(
            Path(name).name != name
            or not (run_dir / name).is_file()
            or file_hash(run_dir / name) != sha
            for name, sha in files.items()
        ):
            raise Exp1ExtError("completed extension artifacts changed; resume refused")
    if cohort not in COHORTS:
        raise Exp1ExtError("unknown math extension cohort")
    try:
        check_execution(
            client, execution_mode, confirmed=confirmed, source_mode=source.execution_mode
        )
    except IntegrityError as exc:
        raise Exp1ExtError(str(exc)) from exc
    if execution_mode == "live":
        check_extension_config(dict(extension_config) if extension_config is not None else None)
        if allow_draft or (prompt_dir is not None and Path(prompt_dir).resolve() != PROMPT_DIR):
            raise Exp1ExtError("live execution requires the reviewed math extension prompts")
    if not lambdas or len(set(lambdas)) != len(lambdas) or Lambda.NONE in lambdas:
        raise Exp1ExtError("lambda levels must be nonempty and unique")
    if calibrator is not None and calibrator.record.get("domain") != "math":
        raise Exp1ExtError("math extension requires a math-domain calibrator")
    arms = tuple(COHORTS[cohort].arms if arms is None else arms)
    if not arms or len(set(arms)) != len(arms):
        raise Exp1ExtError("arms must be nonempty and unique")
    if (
        execution_mode == "live"
        and not smoke
        and (
            arms != COHORTS[cohort].arms or tuple(lambdas) != (Lambda.LOW, Lambda.MID, Lambda.HIGH)
        )
    ):
        raise Exp1ExtError("formal Exp1 requires every cohort arm and all three channel levels")
    split = (
        (Split.UNLABELED_POOL if smoke else Split.D_EXP1) if cohort == E_INCORRECT else Split.D_GATE
    )
    purpose = (
        (LabelPurpose.DRY_RUN if smoke else LabelPurpose.EXP1_OUTCOME)
        if cohort == E_INCORRECT
        else (LabelPurpose.EXT_SMOKE if smoke else LabelPurpose.ECORRECT_COHORT)
    )
    available = {s.state_id: s for s in source.states(split)}
    ys = {y.state_id: y for y in source.labels(split, purpose)}
    frozen_ids = None
    if cohort == E_CORRECT:
        # Use the same Core cohort-file contract as pod transport. The local Exp1
        # path must not silently replace the frozen sample with hand-picked IDs.
        frozen_ids = _read_cohort(
            cohort_file, cohort_sha256, fixture=execution_mode == "offline_stub"
        )
        if set(frozen_ids) - available.keys() or any(
            ys[sid].label is not MathLabel.CORRECT for sid in frozen_ids
        ):
            raise Exp1ExtError("frozen correct cohort is not E-CORRECT within source D_gate")
        if smoke:
            if state_ids is not None and set(state_ids) & set(frozen_ids):
                raise Exp1ExtError("E-correct smoke overlaps the frozen formal cohort")
        elif state_ids is None:
            state_ids = frozen_ids
        elif list(state_ids) != frozen_ids:
            raise Exp1ExtError("selected correct states differ from the frozen cohort")
    if state_ids is None:
        if cohort == E_CORRECT or smoke:
            raise Exp1ExtError("correct cohort and smoke need explicitly selected state IDs")
        state_ids = list(available)
    if not state_ids or len(set(state_ids)) != len(state_ids) or set(state_ids) - available.keys():
        raise Exp1ExtError("selected states must be a unique nonempty subset of the proper split")
    if cohort == E_INCORRECT and not smoke and list(state_ids) != list(available):
        raise Exp1ExtError("incorrect extension must reuse the complete original cohort in order")
    expected_label = MathLabel.INCORRECT if cohort == E_INCORRECT else MathLabel.CORRECT
    if any(ys[sid].label is not expected_label for sid in state_ids):
        raise Exp1ExtError("selected source y* disagrees with cohort")
    states = [available[sid] for sid in state_ids]
    if execution_mode == "live" and cohort == E_INCORRECT and not smoke and len(states) != 300:
        raise Exp1ExtError("formal incorrect extension requires all 300 original states")
    if any(s.t != 1 for s in states):
        raise Exp1ExtError("math extension only starts at original t=1 states")
    cfg = build_controller_config(source.master_seed, 1)
    cfg_nv = build_no_verifier_config(source.master_seed, 1, prompt_dir, allow_draft=allow_draft)
    wording = math_wording(prompt_dir, allow_draft=allow_draft)
    original, replay = (None, None)
    if cohort == E_INCORRECT and not smoke:
        original, replay = reconstruct_original_channel(
            source, states, lambdas=lambdas, controller_cfg=cfg
        )
    scorer = BoundScoreIdentity(*source.verifier_ids, *source.verifier_versions)
    directory = Path(prompt_dir or PROMPT_DIR)
    prompt_hashes = {
        n: hashlib.sha256((directory / n).read_bytes()).hexdigest() for n in PROMPT_FILES
    }
    feedback_hashes = {
        name: hashlib.sha256((math_feedback.PROMPT_DIR / name).read_bytes()).hexdigest()
        for name in math_feedback.PROMPT_FILES
    }
    prompt_hashes.update(
        {
            f"detox/{n}": hashlib.sha256((SHARED_DIR / n).read_bytes()).hexdigest()
            for n in SHARED_FILES
        }
    )
    namespace = (
        "math_exp1ext_smoke"
        if smoke
        else ("math_exp1" if cohort == E_INCORRECT else "math_exp1_ecorrect")
    )
    setup = Exp1ExtSetup(
        run_id=run_id,
        run_dir=run_dir,
        cohort=cohort,
        states=states,
        scorer=scorer,
        tau1=source.taus[0],
        tau2=source.taus[1],
        channel=source.channel(),
        lambdas=lambdas,
        arms=arms,
        wording=wording,
        e_labels={sid: ys[sid].record for sid in state_ids},
        expected_label=expected_label,
        calibrator=calibrator,
        perm_seed=component_seed(source.master_seed, "permuted_b"),
        pihat_perm_seed=component_seed(source.master_seed, "ext_pihat_mismatch"),
        run_seed=source.master_seed,
        original_channel=original,
        seed_namespace=namespace,
        smoke=smoke,
        score_states=source.pairs,
        original_feedback=original_feedback,
        meta={
            "domain": "math",
            "extension_review": dict(metadata or {}),
            "execution_mode": execution_mode,
            "draft_wording_allowed": allow_draft,
            "source": source.provenance(),
            "ecorrect_cohort_sha256": cohort_sha256 if frozen_ids is not None else None,
            "prompt_sha256": prompt_hashes,
            "original_feedback_sha256": feedback_hashes,
            "controller_identity": asdict(cfg),
            "no_verifier_identity": asdict(cfg_nv),
            "original_replay": replay,
        },
    )
    ext = (
        dict(extension_config)
        if extension_config is not None
        else yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "configs/math/ext.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    snapshot_configs(
        run_dir, ext, source.read_bound_json("fit.json")["channel"], identity=source.provenance()
    )
    result = run_exp1_ext(client, cfg, cfg_nv, setup)
    source.assert_unchanged()
    if frozen_ids is not None:
        if (
            _read_cohort(cohort_file, cohort_sha256, fixture=execution_mode == "offline_stub")
            != frozen_ids
        ):
            raise Exp1ExtError("frozen correct cohort changed during execution")
    current_feedback = {
        name: hashlib.sha256((math_feedback.PROMPT_DIR / name).read_bytes()).hexdigest()
        for name in math_feedback.PROMPT_FILES
    }
    if current_feedback != feedback_hashes:
        raise Exp1ExtError("original feedback prompts changed during extension execution")
    if not completion.exists():
        names = ["ext_meta.json", "decisions.jsonl", "scores.jsonl", "channel.jsonl"]
        names += [
            name
            for name in ("pihat.jsonl", "pihat_permutation.jsonl", "permutation.jsonl")
            if (run_dir / name).exists()
        ]
        atomic_json(
            completion,
            {
                "files": {name: file_hash(run_dir / name) for name in names},
                "n_rows": result["n_rows"],
            },
        )
    return {
        **result,
        "original_replay": replay,
        "seed_namespace": namespace,
        "execution_mode": execution_mode,
        "live_executed": execution_mode == "live",
        "scientific_result": execution_mode == "live" and not smoke,
    }
