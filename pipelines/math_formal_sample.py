"""paper App. D.1 formal sample: N eligible DeepMath-103K questions by seed, outside MATH-500 and
the probe.

    python pipelines/math_formal_sample.py --raw-dir data/raw/math/DeepMath-103K \
        --development-source data/raw/math/MATH-500/test.jsonl \
        --additional-development data/math/m21_probe/probe_public.jsonl \
        --additional-development data/math/m21_probe_2/probe_public.jsonl \
        --out data/math/formal_n9000 [--n 9000] [--workers 6]

No model call. Paper App. D.1 (2026-10-04): the formal generation draws
N = 9000 questions by seed from the eligible population; the probed questions are not in
it. Eligibility is the formal loader's own screen, ``dataset._screen_source`` with the
same development exclusions the formal config declares (MATH-500 and the probe files, by
ID and normalized problem): deduplication, task screens, question rule, verdict /
option-letter / label references. Its last rule, the E reference check, is the only
expensive one (one Math-Verify process per distinct reference), so it is deferred here:
screen-passing questions are visited in SHA256(seed NUL question_id) order
(``data.stable_sample``'s key) and checked until n pass. That is exactly the n eligible
questions with the smallest keys, i.e. a uniform sample without replacement from the
eligible population, not stratified by difficulty. ``--workers`` only runs the checks of
a chunk concurrently; results are consumed in key order, so the sample does not depend
on it. ``math_run.py prepare`` later repeats every check on the sample as the bundle's
own evidence.

The seed is component ``math_formal_sample`` of the Core master seed
(``configs/data.yaml``), derived like ``component_seed`` without registering a math name
in Core.

Writes (refuses an existing --out):
* source.jsonl: DeepMath-native rows (question, final_answer, difficulty, topic) of the
  sample in key order -- the formal config's ``source_path``;
* source_metadata.json: SourceMetadata for ``math_run.py prepare --metadata``;
* sample_manifest.json: revision, shard hashes, seed, counts and the ordered sample IDs.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq

from pipelines.math_probe_sample import (
    COLUMNS,
    LICENSE_ID,
    LICENSE_URL,
    REPO,
    REVISION,
    SHARDS,
    _write_json,
    _write_jsonl,
)
from rafr.core.config import load_config
from rafr.core.seeding import derive_seed
from rafr.domains.math import dataset
from rafr.domains.math.answer_contract import question_exclusion_reason, reference_exclusion_reason
from rafr.domains.math.data import stable_sample
from rafr.domains.math.dataset import SourceMetadata, content_hash, file_hash

SEED_COMPONENT = "math_formal_sample"
PARTITION = "formal_n{n}"
PURPOSE = "protocol v1.1 M.2.1 formal generation sample (step 4 source)"
CHUNK = 240


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--development-source", type=Path, required=True, help="MATH-500 test.jsonl")
    ap.add_argument(
        "--additional-development",
        type=Path,
        action="append",
        default=[],
        help="probe_public.jsonl of a probe; the formal config lists the same files",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=9000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--partition", help=f"source_partition (default {PARTITION})")
    ap.add_argument("--purpose", default=PURPOSE, help="manifest purpose line")
    args = ap.parse_args(argv)
    if args.n < 1 or args.workers < 1:
        ap.error("need --n >= 1 and --workers >= 1")
    if args.out.exists():
        ap.error(f"{args.out} exists; keep it and choose a new directory")

    for name, expected in SHARDS.items():
        if file_hash(args.raw_dir / name) != expected:
            raise SystemExit(f"{name}: SHA256 differs from {REPO}@{REVISION}")
    master = load_config("data")["seeds"]["master"]
    seed = derive_seed(master, "component", SEED_COMPONENT)
    full = SourceMetadata(
        REPO,
        REVISION,
        content_hash(SHARDS),
        LICENSE_ID,
        LICENSE_URL,
        "approved",
        None,
        "deepmath",
        "all_shards",
    )
    rows = []
    for name in SHARDS:
        rows.extend(pq.read_table(args.raw_dir / name, columns=COLUMNS).to_pylist())
    questions, source_exclusions = dataset.native_questions(rows, full)
    development, development_audit = dataset._development_exclusions(
        args.development_source, args.additional_development
    )
    # The formal loader's screen with the E reference check deferred (None = pass here).
    screened, exclusions = dataset._screen_source(
        questions,
        lambda answer: None,
        question_exclusion_reason,
        development,
        allow_unknown_original_split=True,
        source_exclusions=source_exclusions,
    )
    screened_out = Counter(row["reason"] for row in exclusions)
    print(
        f"[formal] {len(rows)} source rows from {len(SHARDS)} verified shards; "
        f"{len(screened)} pass the screens; screened out {dict(sorted(screened_out.items()))}"
    )

    order = stable_sample(screened, len(screened), seed)
    checks: dict[str, str | None] = {}
    sample, visited, failed = [], 0, Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(order), CHUNK):
            if len(sample) == args.n:
                break
            chunk = order[start : start + CHUNK]
            pending = [
                a for a in dict.fromkeys(q.reference_answer for q in chunk) if a not in checks
            ]
            # E errors raise in reference_exclusion_reason and stop the sample.
            checks.update(zip(pending, pool.map(reference_exclusion_reason, pending), strict=True))
            for question in chunk:
                if len(sample) == args.n:
                    break
                visited += 1
                reason = checks[question.reference_answer]
                if reason:
                    failed[reason] += 1
                else:
                    sample.append(question)
            print(f"[formal] visited {visited}, sampled {len(sample)}", flush=True)
    if len(sample) < args.n:
        raise SystemExit(f"only {len(sample)} eligible questions; requested {args.n}")
    reference_pass_rate = len(sample) / visited

    args.out.mkdir(parents=True)
    source = args.out / "source.jsonl"
    _write_jsonl(
        source,
        [
            {
                "question": q.problem,
                "final_answer": q.reference_answer,
                "difficulty": q.difficulty,
                "topic": q.subject,
            }
            for q in sample
        ],
    )
    metadata = SourceMetadata(
        REPO,
        REVISION,
        file_hash(source),
        LICENSE_ID,
        LICENSE_URL,
        "approved",
        None,
        "deepmath",
        args.partition or PARTITION.format(n=args.n),
    )
    _write_json(args.out / "source_metadata.json", asdict(metadata))
    _write_json(
        args.out / "sample_manifest.json",
        {
            "purpose": args.purpose,
            "source": {"repo": REPO, "revision": REVISION, "shards": SHARDS},
            "license": {"id": LICENSE_ID, "url": LICENSE_URL},
            "seed": {"master": master, "component": SEED_COMPONENT, "value": seed},
            "development_exclusion": development_audit,
            "method": (
                "formal loader screen (dataset._screen_source) over the whole source with "
                "MATH-500 and the probe files as development exclusions; screen-passing "
                "questions visited in SHA256(seed NUL question_id) order with the E reference "
                "check until n are eligible; uniform without replacement, not stratified"
            ),
            "whole_source_exact": {
                "source_rows": len(rows),
                "pass_screens": len(screened),
                "screened_out": dict(sorted(screened_out.items())),
            },
            "walk": {
                "visited": visited,
                "eligible": len(sample),
                "reference_failed": dict(sorted(failed.items())),
                "reference_pass_rate": reference_pass_rate,
                "distinct_reference_strings_checked": len(checks),
            },
            "eligible_population_estimate": {
                "value": len(screened) * reference_pass_rate,
                "formula": "pass_screens (exact) x reference_pass_rate (walk)",
            },
            "counts": {"n": len(sample)},
            "sample": [
                {"rank": rank, "question_id": q.question_id, "difficulty": q.difficulty}
                for rank, q in enumerate(sample)
            ],
            "files": {source.name: metadata.source_sha256},
        },
    )
    print(f"[formal] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
