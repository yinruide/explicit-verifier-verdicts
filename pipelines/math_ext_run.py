"""Safe local entry points for math extension review and synthetic preparation.

Importing this module loads no project provider/runtime. Direct execution explicitly
prefers this checkout's src/root over an editable install from another worktree.
Live calls require an explicit stage and operator confirmation. This module does
not confer execution authorization; default preparation and tests remain offline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "src"):
    _path = str(_entry)
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

from rafr.core import paths  # noqa: E402

BASE_BINDINGS = {
    "states.jsonl",
    "initial_complete.json",
    "splits_complete.json",
    "scores.jsonl",
    "fit.json",
    "snapshot.json",
}
REPLAY_BINDINGS = {
    "exp1/decisions.jsonl",
    "exp1/channel.jsonl",
    "exp1/permutation.json",
    "exp2/assignment.json",
    "exp2/trajectories.jsonl",
    "exp2/decisions.jsonl",
    "exp2/rounds/round_1/scores.jsonl",
    *{f"exp2/rounds/round_{t}/decisions.jsonl" for t in range(1, 5)},
}
SHA = re.compile(r"[0-9a-f]{64}\Z")


def _config(path):
    import yaml

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("domain") != "math":
        raise ValueError("configuration must be a math extension mapping")
    return value


def _manifest(path):
    path = Path(path).resolve()
    body = json.loads(path.read_text(encoding="utf-8"))
    fields = {"source_root", "expected_sha256", "master_seed", "verifier_ids"}
    if not isinstance(body, dict) or set(body) != fields:
        raise ValueError(
            "source manifest requires exactly source_root/expected_sha256/master_seed/verifier_ids"
        )
    if not isinstance(body["source_root"], str) or body["source_root"] in ("", "TBD"):
        raise ValueError("source_manifest.source_root is missing/TBD")
    if type(body["master_seed"]) is not int:
        raise ValueError("source_manifest.master_seed must be an integer")
    ids = body["verifier_ids"]
    if (
        not isinstance(ids, list)
        or len(ids) != 2
        or len(set(ids)) != 2
        or any(not isinstance(v, str) or v in ("", "TBD") for v in ids)
    ):
        raise ValueError("source_manifest.verifier_ids requires two distinct bound IDs")
    hashes = body["expected_sha256"]
    if not isinstance(hashes, dict):
        raise ValueError("source_manifest.expected_sha256 must be a mapping")
    missing = sorted(BASE_BINDINGS - hashes.keys())
    invalid = sorted(k for k, v in hashes.items() if not isinstance(v, str) or not SHA.fullmatch(v))
    if missing or invalid:
        raise ValueError(
            f"missing source bindings: {missing}; missing/TBD/invalid SHA256: {invalid}"
        )
    root = Path(body["source_root"])
    body["source_root"] = root if root.is_absolute() else path.parent / root
    return body


def load_source(manifest):
    """Read and validate bound source bytes; never generate hashes to manufacture trust."""
    from pipelines.math_ext_data import MathSource

    body = _manifest(manifest)
    return MathSource(
        body["source_root"],
        expected_sha256=body["expected_sha256"],
        master_seed=body["master_seed"],
        verifier_ids=tuple(body["verifier_ids"]),
    )


def preflight(
    *,
    manifest=None,
    config=None,
    operation="replay",
    stage="offline_stub",
    cohort="e_incorrect",
    fixture=False,
):
    """Read-only, operation-scoped checks. A clear report never authorizes a run."""
    blockers, missing, source = [], [], None
    cfg = None
    if stage not in {"offline_stub", "real_prepare", "live"}:
        blockers.append("unknown stage")
    try:
        if manifest is None or operation in {"exp1", "exp2", "ceiling", "starts", "loop", "judge"}:
            cfg = _config(config or ROOT / "configs/math/ext.yaml")
        if manifest is None:
            manifest = cfg.get("source_manifest")
            if not isinstance(manifest, str) or manifest in ("", "TBD"):
                missing.append("source_manifest")
                manifest = None
            elif not Path(manifest).is_absolute():
                manifest = paths.resolve(manifest)
        if manifest is not None:
            body = _manifest(manifest)
            if operation == "replay":
                missing.extend(sorted(REPLAY_BINDINGS - body["expected_sha256"].keys()))
            source = load_source(manifest)
            if stage in {"live", "real_prepare"} and source.execution_mode != "live":
                blockers.append("live preparation/execution requires live source artifacts")
            if (
                stage == "offline_stub"
                and operation != "replay"
                and source.execution_mode != "offline_stub"
            ):
                blockers.append("offline execution cannot use live source artifacts")
            if fixture and source.execution_mode != "offline_stub":
                blockers.append("fixture replay cannot use live source artifacts")
            if operation in {"calibration", "cohort"} and stage == "offline_stub":
                if source.execution_mode != "offline_stub":
                    blockers.append("offline_stub preparation requires synthetic source artifacts")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        blockers.append(str(exc))
    if cfg is not None and operation in {"exp1", "exp2", "loop", "judge", "starts", "ceiling"}:
        if cfg.get("calibration", {}).get("frozen_sha256") in (None, "", "TBD"):
            missing.append("calibration.frozen_sha256")
        if cohort == "e_correct" and cfg.get("cohorts", {}).get("e_correct", {}).get(
            "file_sha256"
        ) in (None, "", "TBD"):
            missing.append("cohorts.e_correct.file_sha256")
        if source is not None and stage == "live":
            try:
                from pipelines.math_ext_runtime import load_extension

                load_extension(
                    config or ROOT / "configs/math/ext.yaml", source, operation=operation, live=True
                )
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
                blockers.append(str(exc))
    if source is not None:
        source.assert_unchanged()
    return {
        "status": "BLOCKED" if blockers or missing else "READY_FOR_REVIEW",
        "operation": operation,
        "stage": stage,
        "cohort": cohort,
        "read_only": True,
        "provider_calls": 0,
        "live_executed": False,
        "execution_authorized": False,
        "missing_bindings": sorted(set(missing)),
        "blockers": blockers,
        "source": source.provenance() if source is not None else None,
        "evidence_scope": "synthetic fixture" if fixture else "bound local artifacts only",
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="read-only stage-scoped binding report; no outputs")
    pre.add_argument("--manifest", type=Path)
    pre.add_argument("--config", type=Path, default=ROOT / "configs/math/ext.yaml")
    pre.add_argument(
        "--operation",
        choices=(
            "replay",
            "calibration",
            "cohort",
            "exp1",
            "exp2",
            "ceiling",
            "starts",
            "loop",
            "judge",
        ),
        default="replay",
    )
    pre.add_argument("--stage", choices=("offline_stub", "real_prepare", "live"), required=True)
    pre.add_argument("--cohort", choices=("e_incorrect", "e_correct"), default="e_incorrect")
    pre.add_argument("--fixture", action="store_true")
    rep = sub.add_parser("replay", help="read-only original request/schema replay")
    rep.add_argument("--manifest", type=Path, required=True)
    rep.add_argument(
        "--fixture", action="store_true", help="mark synthetic fixture, never real replay"
    )
    for name in ("calibrate", "cohort"):
        cmd = sub.add_parser(name, help="candidate preparation; real_prepare requires confirmation")
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--stage", choices=("offline_stub", "real_prepare"), required=True)
        cmd.add_argument("--confirm", action="store_true")
        if name == "calibrate":
            cmd.add_argument("--n-boot", type=int, default=10000)
        else:
            cmd.add_argument("--n", type=int, default=300)
    for name in ("starts", "exp1", "loop", "judge", "ceiling"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--config", type=Path, default=ROOT / "configs/math/ext.yaml")
        cmd.add_argument("--stage", choices=("offline_stub", "live"), required=True)
        cmd.add_argument("--confirm", action="store_true")
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--run-id", required=name != "starts", default=None)
        cmd.add_argument("--cohort", choices=("e_incorrect", "e_correct"), default="e_incorrect")
        cmd.add_argument("--smoke", type=int, default=0, metavar="N")
        if name in {"starts", "exp1"}:
            cmd.add_argument("--arms", nargs="+")
        if name == "starts":
            cmd.add_argument(
                "--fill-original",
                action="store_true",
                help="add only each question's missing A1/B arm",
            )
        if name in {"loop", "judge"}:
            cmd.add_argument("--bundle", type=Path, required=True)
            cmd.add_argument("--bundle-sha256", required=True)
        if name in {"judge", "ceiling"}:
            group = cmd.add_mutually_exclusive_group(required=True)
            group.add_argument("--references", type=Path)
            group.add_argument(
                "--dataset-bundle",
                type=Path,
                help="original private dataset bundle bound by source snapshot",
            )
            cmd.add_argument("--references-sha256")
        if name == "judge":
            cmd.add_argument("--pod-receipt", type=Path)
            cmd.add_argument("--pod-receipt-sha256")
    cmd = sub.add_parser("freeze", help="explicit M-C candidate freeze, writes a new file only")
    cmd.add_argument("--kind", choices=("calibration", "cohort"), required=True)
    cmd.add_argument("--manifest", type=Path, required=True)
    cmd.add_argument("--stage", choices=("offline_stub", "real_prepare"), required=True)
    cmd.add_argument("--candidate", type=Path, required=True)
    cmd.add_argument("--candidate-sha256", required=True)
    cmd.add_argument("--receipt", type=Path)
    cmd.add_argument("--receipt-sha256")
    cmd.add_argument("--diagnostics", type=Path)
    cmd.add_argument("--diagnostics-sha256")
    cmd.add_argument(
        "--out",
        type=Path,
        default=None,
        help="offline_stub only; real_prepare writes frozen/calibration_math.json or "
        "data/ext/ecorrect_cohort_math.jsonl (configs/math/ext.yaml)",
    )
    cmd.add_argument("--confirm", action="store_true")
    return p


def main(argv=None, *, provider_factory=None, grader=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(
                manifest=args.manifest,
                config=args.config,
                operation=args.operation,
                stage=args.stage,
                cohort=args.cohort,
                fixture=args.fixture,
            )
        elif args.command == "replay":
            from pipelines.math_ext_regression import replay_original

            source = load_source(args.manifest)
            result = replay_original(source, fixture=args.fixture)
        elif args.command in {"calibrate", "cohort"}:
            if args.stage == "real_prepare" and not args.confirm:
                raise ValueError(
                    "real_prepare is not authorized/enabled without explicit --confirm"
                )
            from pipelines.math_ext_data import prepare_calibration, prepare_correct_cohort

            source = load_source(args.manifest)
            if args.stage == "real_prepare" and source.execution_mode != "live":
                raise ValueError("real_prepare requires live source artifacts")
            stage = "offline_candidate" if args.stage == "real_prepare" else "offline_stub"
            if args.command == "calibrate":
                result = prepare_calibration(source, args.out, n_boot=args.n_boot, stage=stage)
            else:
                result = prepare_correct_cohort(source, args.out, n=args.n, stage=stage)
        elif args.command == "freeze":
            from pipelines.math_ext_runtime import freeze

            result = freeze(args)
        else:
            from pipelines.math_ext_runtime import execute

            if args.smoke < 0:
                raise ValueError("smoke sample count must be positive")
            result = execute(args, provider_factory=provider_factory, grader=grader)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if result.get("status") == "BLOCKED" else 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "error": str(exc),
                    "execution_complete": False,
                    "execution_note": (
                        "Inspect isolated run receipts: a failed run may contain completed calls."
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
