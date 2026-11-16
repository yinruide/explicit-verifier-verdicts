"""paper Sec. 5, App. D math stages. Preflight is local; model calls require --confirm-live."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

# Match Core pipeline entrypoints: direct invocation also works from a foreign cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.math_initial import DEFAULT_WORKERS
from pipelines.math_pipeline import finite_json
from pipelines.math_revision import initialize_revision, writer_lock_root
from pipelines.math_runtime import (
    build_pipeline,
    close_pipeline,
    preflight,
    resolve_path,
    validate_output_root,
)
from rafr.contracts.schemas import Split
from rafr.domains.math.config import MathRunConfig
from rafr.domains.math.dataset import SourceMetadata, prepare_dataset_bundle
from rafr.domains.math.storage import atomic_json, exclusive_lock, read_json

COMMANDS = (
    "prepare",
    "preflight",
    "generate",
    "split",
    "score",
    "fit",
    "diagnostics",
    "stability",
    "channel-check",
    "gate",
    "exp1",
    "exp1-smoke",
    "exp2",
    "smoke",
    "judge",
    "report",
    "report-exp1",
    "all",
    "revise-for-reserve",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--prior-run", type=Path, help="failed prior run for a reserve redesign")
    parser.add_argument(
        "--phase",
        choices=["generation", "scoring", "stability", "gate", "experiment", "smoke", "analysis"],
        default="generation",
        help="phase inspected by preflight",
    )
    parser.add_argument(
        "--split",
        choices=[value.value for value in Split],
        help="score one Core split; defaults to the four active splits",
    )
    parser.add_argument("--attempt", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--reserve-decision",
        type=Path,
        help="the authors' explicit redesign/reserve decision record for gate attempt 2",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"questions generated in parallel by generate / all (default {DEFAULT_WORKERS}, "
        "as detox step04); files are written in input order either way, 1 is the serial run",
    )
    parser.add_argument("--confirm-live", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    config = MathRunConfig.load(args.config)
    if args.command == "prepare":
        if args.metadata is None:
            parser.error("prepare requires --metadata")
        result = prepare_dataset_bundle(
            resolve_path(config.source_path),
            resolve_path(config.bundle_path),
            metadata=SourceMetadata(**read_json(args.metadata)),
            stage=config.stage,
            development_source_path=resolve_path(config.development_source_path)
            if config.development_source_path
            else None,
            additional_development_paths=[
                resolve_path(p) for p in config.additional_development_paths
            ],
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "preflight":
        result = preflight(config, phase=args.phase)
        if args.output:
            atomic_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["ready_for_requested_phase"] else 2
    if args.run_dir is None:
        parser.error("execution stages require --run-dir")
    root = validate_output_root(config, args.run_dir)
    if args.command == "revise-for-reserve" and (
        args.prior_run is None or args.reserve_decision is None
    ):
        parser.error("revise-for-reserve requires --prior-run and --reserve-decision")
    prior_root = resolve_path(args.prior_run) if args.command == "revise-for-reserve" else None
    lineage_root = writer_lock_root(root, prior_root)
    # This is only a writer mutex, not a reserve lock or a new storage/logging format.
    # Transfers also hold the target mutex; sort both paths to prevent parent/child
    # lock inversion. Every descendant serializes shared Core history through the
    # exact same general writer mutex as its original cohort.
    lock_paths = {base.parent / f".{base.name}.writer.lock" for base in (root, lineage_root)}
    with ExitStack() as locks:
        for lock_path in sorted(lock_paths):
            locks.enter_context(exclusive_lock(lock_path))
        if writer_lock_root(root, prior_root) != lineage_root:
            raise ValueError("gate lineage changed while acquiring writer mutexes; retry")
        pipeline = build_pipeline(
            config, root, command=args.command, confirm_live=args.confirm_live
        )
        try:
            if args.command == "revise-for-reserve":
                result = initialize_revision(pipeline, prior_root, read_json(args.reserve_decision))
            elif args.command == "score":
                selected = (
                    [Split(args.split)]
                    if args.split
                    else [
                        Split.D_FIT,
                        Split.UNLABELED_POOL,
                        Split.D_GATE,
                        Split.D_EXP1,
                    ]
                )
                result = [pipeline.score(split) for split in selected]
            elif args.command == "gate":
                result = pipeline.gate(
                    args.attempt,
                    reserve_decision=read_json(args.reserve_decision)
                    if args.reserve_decision
                    else None,
                )
            elif args.command == "generate":
                result = pipeline.generate(workers=args.workers)
            elif args.command == "all":
                result = pipeline.run_all(workers=args.workers)
            elif args.command == "report-exp1":
                result = pipeline.report_exp1()
            elif args.command == "exp1-smoke":
                result = pipeline.exp1_smoke()
            elif args.command == "channel-check":
                result = pipeline.channel_check()
            elif args.command in {"diagnostics", "stability"}:
                result = pipeline.diagnostics(stability=args.command == "stability")
            elif args.command in {"judge", "report"}:
                result = getattr(pipeline, args.command)(smoke=config.stage == "smoke")
            else:
                result = getattr(pipeline, args.command)()
            print(
                json.dumps(
                    finite_json(
                        {
                            "result": result,
                            "request_ledger": pipeline.client.ledger.summary()
                            if pipeline.client.ledger is not None
                            else {},
                        }
                    ),
                    ensure_ascii=False,
                )
            )
        finally:
            close_pipeline(pipeline)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, TypeError) as exc:
        print(f"math stage stopped: {exc}", file=sys.stderr)
        sys.exit(1)
