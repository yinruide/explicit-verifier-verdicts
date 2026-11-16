"""Math extension pod worker. Public inputs only; E remains on the authors' local machine.

A previously completed, isolated math_step06 pod session supplies pinned weights and its
loopfit/check203 evidence. This worker never downloads, changes those records or terminates
that session. Providers are created only by the explicit ``loop --confirm`` command.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "scripts" / "gpu", ROOT, ROOT / "src"):
    sys.path.insert(0, str(entry))

from rafr.domains.math.storage import IntegrityError, atomic_json, file_hash, read_json

SCHEMA = "math-extension-pod-input-v1"
INPUT_FILES = {
    "transport.json",
    "starts.jsonl",
    "t1.json",
    "ext.yaml",
    "channel.json",
    "formal.json",
    "loop.json",
    "transport.json.sha256",
    "starts.jsonl.sha256",
    "t1.json.sha256",
}
ROUND_FILES = {"trajectories.jsonl", "decisions.jsonl", "scores.jsonl", "ledger.jsonl"}
RUN_FILES = {
    "plan.jsonl",
    "ext_meta.json",
    "snapshot.json",
    "input_snapshot.json",
    "configs/ext.yaml",
    "configs/channel.yaml",
    "pihat.jsonl",
    "t1_check.json",
    "t1_bitwise.json",
    "t1_reference_check.json",
    "decisions.jsonl",
    "scores.jsonl",
    "ledger.jsonl",
    "text_scores.jsonl",
    "text_score_details.jsonl",
    "transient_errors.jsonl",
    "transient_retries.jsonl",
    "loop_complete.json",
    "pod_checks.json",
    "pod_identity.json",
}
REQUIRED_RUN_FILES = {
    "plan.jsonl",
    "ext_meta.json",
    "snapshot.json",
    "input_snapshot.json",
    "configs/ext.yaml",
    "configs/channel.yaml",
    "loop_complete.json",
    "pod_checks.json",
    "pod_identity.json",
}
SHA = re.compile(r"[0-9a-f]{64}\Z")
HEAD = re.compile(r"[0-9a-f]{40}\Z")


def checked_sha(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise IntegrityError("an independent complete SHA256 is required")
    return value


def run_name(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"math_exp2ext_[A-Za-z0-9][A-Za-z0-9_-]{0,99}", value
    ):
        raise IntegrityError("run id must start math_exp2ext_ and contain only safe characters")
    return value


def safe_file(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise IntegrityError(f"missing file or symlink: {path.name}")
    return path


def checked_file(path, expected):
    path = safe_file(path)
    if file_hash(path) != checked_sha(expected):
        raise IntegrityError(f"SHA256 mismatch: {path.name}")
    return path


def timestamp(value):
    try:
        found = datetime.fromisoformat(value)
        if found.tzinfo is None:
            raise ValueError("timezone missing")
        return found
    except (ValueError, TypeError) as exc:
        raise IntegrityError("session/check timestamp must have a timezone") from exc


def read_inputs(directory, expected_sha256):
    """Closed public allowlist plus external receipt binding, before any model/provider."""
    import yaml
    from pipelines.math_ext_execution import check_extension_config
    from pipelines.math_ext_transport import import_public_bundle

    from rafr.calibration.apply import Calibrator
    from rafr.contracts.schemas import Arm
    from rafr.domains.math.config import MathRunConfig

    directory = Path(directory)
    receipt = read_json(checked_file(directory / "receipt.json", expected_sha256))
    fields = {
        "schema",
        "run_id",
        "git_head",
        "execution_sha256",
        "session_started_at",
        "source_identity_sha256",
        "source_mode",
        "verifier_ids",
        "verifier_versions",
        "files",
    }
    if set(receipt) != fields or receipt["schema"] != SCHEMA:
        raise IntegrityError("unexpected pod receipt fields")
    run_name(receipt["run_id"])
    if not isinstance(receipt["git_head"], str) or not HEAD.fullmatch(receipt["git_head"]):
        raise IntegrityError("receipt needs reviewed git HEAD")
    checked_sha(receipt["execution_sha256"])
    timestamp(receipt["session_started_at"])
    files = receipt["files"]
    if not isinstance(files, dict) or set(files) not in (
        INPUT_FILES,
        INPUT_FILES | {"calibration.json"},
    ):
        raise IntegrityError("pod upload allowlist forbids extra or missing files")
    if {p.name for p in directory.iterdir()} != set(files) | {"receipt.json"}:
        raise IntegrityError("public input directory contains an unlisted file")
    for name, sha in files.items():
        checked_file(directory / name, sha)
    bundle = import_public_bundle(
        directory,
        expected_manifest_sha256=files["transport.json"],
        expected_source_identity_sha256=receipt["source_identity_sha256"],
        expected_verifier_ids=tuple(receipt["verifier_ids"]),
        expected_verifier_versions=tuple(receipt["verifier_versions"]),
        expected_source_mode=receipt["source_mode"],
    )
    if receipt["source_mode"] != "live" or bundle.manifest["fixture"]:
        raise IntegrityError("pod inputs must be real, source-bound data; fixtures cannot run live")
    ext = yaml.safe_load((directory / "ext.yaml").read_text(encoding="utf-8"))
    check_extension_config(ext)
    channel = read_json(directory / "channel.json")
    if not isinstance(channel, dict) or not channel.get("levels"):
        raise IntegrityError("missing bound math channel")
    taus = (channel.get("tau1"), channel.get("tau2"))
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in taus):
        raise IntegrityError("frozen finite thresholds required")
    config = MathRunConfig.load(directory / "formal.json")
    if config.execution_mode != "live" or config.stage != "formal":
        raise IntegrityError("pod requires an explicitly formal/live math configuration")
    config.assert_ready("smoke" if bundle.manifest["smoke"] else "experiment")
    if config.seed != bundle.manifest["master_seed"]:
        raise IntegrityError("formal seed differs from public source")
    specs = config.verifiers
    if [v["model_id"] for v in specs] != receipt["verifier_ids"] or [
        v["revision"] for v in specs
    ] != receipt["verifier_versions"]:
        raise IntegrityError("formal verifier identities differ from source")
    arms = {Arm(a) for values in bundle.plan.values() for a in values}
    calibrator = None
    if "calibration.json" in files:
        if ext["calibration"]["frozen_sha256"] != files["calibration.json"]:
            raise IntegrityError("calibrator differs from frozen extension config")
        calibrator = Calibrator.load(
            directory / "calibration.json", expected_sha256=files["calibration.json"]
        )
        calibrator.check_verifiers(*receipt["verifier_ids"])
        if calibrator.record.get("domain") != "math":
            raise IntegrityError("calibrator domain differs")
    if Arm.B_PIHAT in arms and calibrator is None:
        raise IntegrityError("pi-hat segment requires its frozen calibrator")
    return receipt, bundle, ext, channel, config, calibrator


def session_verifiers(base, directory, receipt, config, *, factory=None):
    """Reuse original spec/shape/check203 implementation, requiring this session's check."""
    import math_step06_remote as s6
    import math_vprobe2_remote as r2
    from rafr.domains.math.models import build_verifier

    factory = factory or build_verifier
    paths = s6.layout(Path(base))
    loop = s6.load_loop(Path(directory) / "loop.json")
    specs, evidence = [], {}
    current_env = r2.environment()
    channel = read_json(Path(directory) / "channel.json")
    for index, fixed in enumerate(config.verifiers):
        found = [c for c in r2.load_cfg()["candidates"] if c["spec"] == fixed]
        if len(found) != 1:
            raise IntegrityError("formal verifier is not the frozen math candidate")
        candidate = s6.load_v(found[0]["candidate_id"], Path(directory) / "formal.json")
        cid = candidate["candidate_id"]
        spec = s6.loop_spec(paths, candidate, loop)
        record = s6.require_check(paths, cid, factory(spec).config_hash, loop=True)
        if (
            timestamp(record.get("checked_at")) < timestamp(receipt["session_started_at"])
            or record.get("environment") != current_env
            or record.get("configuration") != "exp2_loop"
        ):
            raise IntegrityError("loopcheck203 is stale or from another environment/configuration")
        if record.get("n") != 203 or record.get("criterion") not in {"bitwise", "alternative"}:
            raise IntegrityError("current loopcheck must cover all 203 reference states")
        if record["criterion"] == "alternative" and record.get("tau") != channel[f"tau{index + 1}"]:
            raise IntegrityError(
                "alternative loopcheck threshold differs from frozen math threshold"
            )
        shape = {key: spec.settings[key] for key in s6.LOOP_ADJUSTABLE}
        if any(type(v) is not int or v <= 0 for v in shape.values()):
            raise IntegrityError("loopfit shape must contain positive integers")
        # require_check validates passed/config_hash; retain its actual criterion, including
        # the math-specific alternative allowed by the original design.
        evidence[cid] = {
            "record": record,
            "shape": shape,
            "check_sha256": file_hash(s6.check_record_path(paths, cid, True)),
            "loopfit_sha256": file_hash(paths["out"] / "loopfit" / f"{cid}.json"),
        }
        specs.append(spec)
    return specs, evidence


def validate_round(directory, t):
    directory = Path(directory)
    marker = read_json(safe_file(directory / "COMPLETE.json"))
    if marker.get("t") != t or set(marker.get("files", {})) != ROUND_FILES:
        raise IntegrityError("incomplete or wrong Core round manifest")
    if {p.name for p in directory.iterdir()} != ROUND_FILES | {"COMPLETE.json"}:
        raise IntegrityError("unexpected round file")
    for name, sha in marker["files"].items():
        checked_file(directory / name, sha)
    rows = [
        json.loads(line)
        for line in (directory / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    decisions = [
        json.loads(line)
        for line in (directory / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != marker.get("n_trajectories") or len(decisions) != marker.get("n_decisions"):
        raise IntegrityError("round record counts disagree")
    return marker


def seal_run(out, receipt_sha256):
    """Terminal hash inventory excludes provider caches and keys; E is deliberately absent."""
    out = Path(out)
    for t in range(1, 5):
        validate_round(out / "rounds" / f"round_{t}", t)
    if not all((out / name).is_file() for name in REQUIRED_RUN_FILES):
        raise IntegrityError("completed loop lacks required delivery artifacts")
    meta = read_json(out / "ext_meta.json")
    if meta.get("t1_reference") is not None and not (out / "t1_reference_check.json").is_file():
        raise IntegrityError("fresh t1 scoring requires its reference comparison report")
    names = {name for name in RUN_FILES if (out / name).is_file()}
    names |= {
        f"rounds/round_{t}/{name}" for t in range(1, 5) for name in ROUND_FILES | {"COMPLETE.json"}
    }
    value = {
        "schema": "math-extension-pod-seal-v1",
        "receipt_sha256": checked_sha(receipt_sha256),
        "run_id": out.name,
        "rounds": [1, 2, 3, 4],
        "files": {name: file_hash(safe_file(out / name)) for name in sorted(names)},
    }
    atomic_json(out / "pod_seal.json", value)
    return value


def validate_public_rounds(out, *, receipt, bundle, channel, require_complete=False):
    from pipelines.math_ext_loop import validate_round_chain

    from rafr.experiments.exp2 import Trajectory

    initial = [
        Trajectory(s.x0_id, s.x0, arm.value, 1, s.text)
        for s in bundle.states
        for arm in bundle.plan[s.x0_id]
    ]
    return validate_round_chain(
        out,
        initial,
        run_id=receipt["run_id"],
        seed=bundle.manifest["master_seed"],
        smoke=bundle.manifest["smoke"],
        taus=(channel["tau1"], channel["tau2"]),
        verifier_ids=tuple(receipt["verifier_ids"]),
        verifier_versions=tuple(receipt["verifier_versions"]),
        require_complete=require_complete,
    )


def validate_delivery(out, *, receipt, receipt_sha256, bundle, channel):
    """Local E precondition: complete byte-bound pod delivery, without any provider."""
    out = Path(out)
    binding = {
        "receipt_sha256": checked_sha(receipt_sha256),
        "git_head": receipt["git_head"],
        "execution_sha256": receipt["execution_sha256"],
        "session_started_at": receipt["session_started_at"],
    }
    if read_json(safe_file(out / "pod_identity.json")) != binding:
        raise IntegrityError("local pod identity differs from externally bound receipt")
    seal = read_json(safe_file(out / "pod_seal.json"))
    rounds = {
        f"rounds/round_{t}/{name}" for t in range(1, 5) for name in ROUND_FILES | {"COMPLETE.json"}
    }
    files = seal.get("files", {})
    if (
        set(seal) != {"schema", "run_id", "receipt_sha256", "rounds", "files"}
        or seal["schema"] != "math-extension-pod-seal-v1"
        or seal["run_id"] != receipt["run_id"]
        or out.name != receipt["run_id"]
        or seal["receipt_sha256"] != receipt_sha256
        or seal["rounds"] != [1, 2, 3, 4]
        or not isinstance(files, dict)
        or set(files) - (RUN_FILES | rounds)
        or not REQUIRED_RUN_FILES | rounds <= set(files)
    ):
        raise IntegrityError("local pod seal identity or artifact set differs")
    for name, sha in files.items():
        checked_file(out / name, sha)
    for t in range(1, 5):
        validate_round(out / "rounds" / f"round_{t}", t)
    meta = read_json(out / "ext_meta.json")
    if (
        meta.get("run_id") != receipt["run_id"]
        or meta.get("execution_mode") != "live"
        or meta.get("source_identity")
        != {
            "source_identity_sha256": receipt["source_identity_sha256"],
            "source_execution_mode": "live",
        }
        or meta.get("verifier_ids") != receipt["verifier_ids"]
        or meta.get("verifier_versions") != receipt["verifier_versions"]
    ):
        raise IntegrityError("loop metadata differs from public receipt")
    if meta.get("t1_reference") is not None and "t1_reference_check.json" not in files:
        raise IntegrityError("sealed delivery omits t1 reference diagnostic")
    validate_public_rounds(
        out, receipt=receipt, bundle=bundle, channel=channel, require_complete=True
    )
    return seal


def loop(
    base,
    directory,
    receipt_sha256,
    *,
    confirmed=False,
    resume=False,
    client=None,
    factory=None,
    check_session=None,
):
    from pipelines.math_ext_execution import check_mode

    check_mode("live", confirmed=confirmed, source_mode="live")
    receipt, bundle, ext, channel, config, calibrator = read_inputs(directory, receipt_sha256)
    from rafr.domains.math.code_identity import code_identity

    identity = code_identity()  # Supports the original, reviewed source-only archive format.
    if (
        identity["git_head"] != receipt["git_head"]
        or identity["source_sha256"] != receipt["execution_sha256"]
    ):
        raise IntegrityError("executing code differs from the reviewed delivery")
    specs, checks = (check_session or session_verifiers)(
        base, directory, receipt, config, factory=factory
    )
    out = ROOT / "runs" / run_name(receipt["run_id"])
    from pipelines.math_ext_data import validate_output
    from pipelines.math_ext_snapshot import write_or_match

    validate_output(out, allow_existing=True)
    binding = {
        "receipt_sha256": receipt_sha256,
        "git_head": identity["git_head"],
        "execution_sha256": identity["source_sha256"],
        "session_started_at": receipt["session_started_at"],
    }
    if out.exists():
        if (
            not resume
            or not (out / "pod_identity.json").is_file()
            or read_json(out / "pod_identity.json") != binding
        ):
            raise IntegrityError(
                "existing run requires explicit resume with unchanged complete identity"
            )
        if (out / "pod_seal.json").exists():
            return validate_delivery(
                out, receipt=receipt, receipt_sha256=receipt_sha256, bundle=bundle, channel=channel
            )
    out.mkdir(parents=True, exist_ok=True)
    write_or_match(out / "pod_identity.json", json.dumps(binding, sort_keys=True, indent=2) + "\n")
    write_or_match(out / "pod_checks.json", json.dumps(checks, sort_keys=True, indent=2) + "\n")
    from pipelines.math_exp2_loop import TextScoreStore
    from pipelines.math_ext_exp2 import run_math_exp2, setup_from_public_bundle

    from math_exp2_remote import release_gpu
    from rafr.core.runtime import build_pod_client
    from rafr.domains.math.generation import build_controller_config, frozen_models
    from rafr.domains.math.models import build_verifier
    from rafr.formatter.math_extended import build_no_verifier_config, math_wording

    def build(i):
        verifier = (factory or build_verifier)(specs[i])
        verifier.load()
        return verifier

    store = TextScoreStore(
        out, [(s.model_id, s.revision) for s in specs], build, release=release_gpu
    )
    setup = setup_from_public_bundle(
        bundle,
        run_id=receipt["run_id"],
        run_dir=out,
        taus=(channel["tau1"], channel["tau2"]),
        score_texts=store,
        wording=math_wording(),
        extension_config=ext,
        channel_config=channel,
        calibrator=calibrator,
        confirmed=True,
    )
    # All configuration/prompt validation precedes provider construction and any paid call.
    cfg = build_controller_config(config.seed, 4)
    nv = build_no_verifier_config(config.seed, 4)
    if client is None:
        client = build_pod_client(frozen_models(), run_dir=out, cache_dir=out / "cache")
        client.set_transient_log(out / "transient_retries.jsonl")
    trajectories = run_math_exp2(
        client,
        cfg,
        nv,
        setup,
        on_round=lambda t: print(f"math extension round {t} complete", flush=True),
    )
    atomic_json(
        out / "loop_complete.json",
        {
            "k": 4,
            "rounds": [1, 2, 3, 4],
            "n_trajectories": len(trajectories),
            "execution_mode": "live",
            "SMOKE": bundle.manifest["smoke"],
        },
    )
    return seal_run(out, receipt_sha256)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["loop"])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            loop(
                args.base,
                args.inputs,
                args.receipt_sha256,
                confirmed=args.confirm,
                resume=args.resume,
            )
        )
    )


if __name__ == "__main__":
    main()
