"""Step 00 — prepare data (step 0, paper Sec. 4.2).

    download ParaDetox -> clean + dedupe -> E prefilter of EVERY x0 (keep those judged UNSAFE)

**No split happens here any more**: the splits are drawn in step05, after E has judged
every state. So there is no candidate cap, no "enough x0 to fill the splits" stop and no
"the pool cannot fill the splits" projection: the whole deduplicated pool is judged.

**One E** (paper Sec. 3). E is built from ``models.evaluator`` by ``judge.build_config``, which
refuses to run when the prompt file's sha256 is not the pinned one. Every row of
``data/pools/prefilter.jsonl`` names the E that made it (model, version, decoding, prompt
sha256). A re-run reuses the judgments of the SAME E only; a file holding any judgment by a
different E stops the step (exit 5): withdrew 's "the prefilter is never redone"
and an E change means archiving the old chain and rebuilding it, never mixing.

Spending controls (2026-09-20 / 2026-09-23):

* ``--budget`` is required on any run that can call E; the run stops when the live tokens it
  has bought reach it.
* ``--limit N`` stops after N NEW judgments (reused ones are free and do not count) -- the dry
  run of a paid segment is ``--limit N`` and prints the measured price per x0 and the
  projected cost of the whole pool.
* it stops after ``--max-consecutive-e-failures`` failures in a row (default 10), which is what
  a provider-side outage looks like from in here.

Parallel calls (2026-09-23): ``--workers`` (default 4) E calls in flight on a thread
pool; the results are written in input order, so the files are byte-for-byte those of a serial
run (``--workers 1``) and every request and cache key is the same. A 429, a connection error or
a 5xx is retried by the LLM client with an exponential backoff (``models.evaluator.network``)
and counted -- it is not an E attempt, not an E failure and not part of the consecutive-failure
count; only when one request's backoff runs out does the run stop (``transient_errors``), to be
resumed later. Each such event is a row of ``runs/step00_<ts>/transient_errors.jsonl``.

An x0 on which E still fails after its 2 retries is dropped with the reason recorded
(paper Sec. 4.2); a later run retries it, because a failure is not a judgment. Only when the loop
has gone through every x0 is the prefilter complete: then ``data/pools/x0_prefiltered.jsonl``
(the input of step04) is written and the manifest says ``prefilter.complete: true``.

The processing order is a seeded shuffle of the whole pool (component ``sampling``), so a
``--limit`` batch is a random sample of ParaDetox, not its first block of rows.

The first run also pins the split plan (seed + procedure) in the manifest, before any
state exists (paper Sec. 4.2); see ``rafr.data.splits.split_plan``.

Usage::

    python pipelines/step00_prepare_data.py --budget 1 --limit 100    # dry run: price + estimate
    python pipelines/step00_prepare_data.py --workers 4 --budget 85 --skip-download  # the rest
    python pipelines/step00_prepare_data.py --fake --skip-download    # offline self-check
"""

from __future__ import annotations

# A direct ``python pipelines/<step>.py`` puts pipelines/ on sys.path, not the repo root,
# so the repo-root packages (analysis, annotation, pipelines) would not import; rafr is
# installed and unaffected. Covered by tests/test_pipeline_entrypoints.py.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from rafr.core import paths
from rafr.core.config import assert_frozen, get, load_config, master_seed
from rafr.core.runtime import (  # shared with step04 / step05: the steps that buy thousands
    PRICES,
    Spend,
    Stop,
    build_client,
    cost_projection,
    fmt_duration,
    prices_for,
    render_projection,
)
from rafr.core.seeding import component_seed, py_rng
from rafr.core.textio import crlf_append_notice
from rafr.data import paradetox
from rafr.data import splits as splits_mod
from rafr.data.manifest import Manifest, SplitPlanChangedError
from rafr.evaluator import judge
from rafr.llm.client import TransientRetriesExhaustedError
from rafr.logging.snapshot import write_snapshot
from rafr.logging.writer import JsonlWriter

GUARD_SECTIONS: tuple[str, ...] = (
    "data.seeds",
    "data.source",
    "data.cleaning",
    "data.prefilter",
    "data.splits",  # the split plan is pinned here, before any state exists (paper Sec. 4.2)
    "models.evaluator",
)
"""Config sections step00 actually reads; the formal-run guard checks these and nothing else
(same rule as step11 / step12). Named here rather than inlined so that a test can check the
guard without running the step -- running it without ``--fake`` makes live E calls."""

PROGRESS_EVERY = 50
"""Items between progress lines."""

CHECKPOINT_EVERY = 500
"""Items between checkpoint summaries; each one also saves the manifest, so a crash cannot
cost the pinned dataset revision or the prefilter counts."""

DEFAULT_WORKERS = 4
"""Parallel E calls (2026-09-23: 4, as in annotation/annotator_selection). Results are
written in input order whatever the number; ``--workers 1`` is the serial run."""

MAX_CONSECUTIVE_E_FAILURES = 10
"""Default for ``--max-consecutive-e-failures``. Each failure is already 3 attempts (paper Sec. 3),
so a run of 10 is 30 calls spent on what is most likely a provider-side outage."""

PREFILTER_FILE = "prefilter.jsonl"
PREFILTERED_FILE = "x0_prefiltered.jsonl"


def _ts() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def processing_order(x0s: list[dict], seed: int) -> list[dict]:
    """The whole pool in a seeded random order (component ``sampling``).

    Every x0 is judged whatever the order; the order only decides which x0 a ``--limit`` batch
    reaches first. ``train.tsv`` is laid out in blocks by source corpus, so file order would make
    a dry run a sample of one source (2026-09-20).
    """
    out = list(x0s)
    py_rng(seed, "candidates").shuffle(out)
    return out


@dataclass
class PrefilterResult:
    kept: list[dict]
    e_failed: list[dict]  # {"x0_id", "reason"}: E still failed after its retries -> x0 dropped
    n_reused: int = 0  # judgments by the same E taken from the file, not re-bought
    n_judged: int = 0  # x0 sent to E in this run (including the ones that failed)
    n_processed: int = 0  # x0 looked at (reused + judged)
    n_candidates: int = 0  # x0 offered to the loop: the whole deduplicated pool
    n_parse_retries: int = 0  # re-sends after a refusal / unreadable answer (paper Sec. 3)
    failed_attempts: dict[str, int] = field(default_factory=dict)  # by reason, all attempts
    n_transient_gave_up: int = 0  # x0 whose 429 / network retries ran out: not judged
    transient: dict[str, float] = field(default_factory=dict)  # the client's transient_stats
    workers: int = 1
    stop: Stop = field(default_factory=lambda: Stop("complete"))
    elapsed_s: float = 0.0
    spend_usd: float | None = None
    tokens: dict[str, int] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Every x0 went through the loop (judged, reused or failed)."""
        return self.stop.reason == "complete" and self.n_processed == self.n_candidates

    @property
    def n_labeled(self) -> int:
        return self.n_processed - len(self.e_failed)

    @property
    def pass_rate(self) -> float | None:
        """kept / judged: the measured prefilter pass rate."""
        return len(self.kept) / self.n_labeled if self.n_labeled else None

    def summary_dict(self) -> dict:
        return {
            "n_candidates": self.n_candidates,
            "n_processed": self.n_processed,
            "n_reused": self.n_reused,
            "n_judged_this_run": self.n_judged,
            "n_labeled": self.n_labeled,
            "n_kept": len(self.kept),
            "n_e_failed": len(self.e_failed),
            "pass_rate": self.pass_rate,
            "n_parse_retries_this_run": self.n_parse_retries,
            "failed_attempts_by_reason_this_run": dict(sorted(self.failed_attempts.items())),
            "n_transient_gave_up_this_run": self.n_transient_gave_up,
            "transient_this_run": self.transient,
            "workers": self.workers,
            "complete": self.complete,
            "stop_reason": self.stop.reason,
            "stop_detail": self.stop.detail,
            "elapsed_s": round(self.elapsed_s, 1),
            "spend_usd": None if self.spend_usd is None else round(self.spend_usd, 4),
            "tokens": self.tokens,
        }


def load_prefilter_rows(path: Path) -> dict[tuple[str, str], dict]:
    """Existing prefilter rows keyed by ``(x0_id, text)``; the last row per key wins.

    The text is part of the key on purpose: a row whose x0_id was reused for different text is
    not a judgment of the current text, so it is re-judged rather than silently carried over.
    """
    if not path.exists():
        return {}
    rows: dict[tuple[str, str], dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[(row["x0_id"], row["text"])] = row
    return rows


def row_identity(row: dict) -> dict:
    """The E identity a prefilter row names (see :func:`judge.check_same_e`)."""
    return {
        "model": row.get("e_model"),
        "version": row.get("e_version"),
        "decoding": row.get("e_decoding"),
        "prompt_sha256": row.get("e_prompt_sha256"),
    }


def check_prefilter_file(path: Path, ecfg: judge.EvaluatorConfig) -> int:
    """Refuse a prefilter file that any other E has written to (paper Sec. 3).

    Every row -- judgments and failures alike -- must name the configured E; a row from an
    older schema names no E at all and is refused the same way. Returns the number of rows.
    """
    rows = load_prefilter_rows(path)
    for row in rows.values():
        judge.check_same_e(row_identity(row), ecfg, where=str(path))
    return len(rows)


def e_row_fields(ecfg: judge.EvaluatorConfig) -> dict:
    return {
        "e_model": ecfg.model,
        "e_version": ecfg.version,
        "e_decoding": ecfg.decoding,
        "e_prompt_sha256": ecfg.prompt_sha256,
    }


def prefilter(  # noqa: C901 - one loop with the stop conditions it must not outrun
    client,
    ecfg,
    x0s,
    *,
    keep_label: str,
    out_path: Path,
    log: JsonlWriter,
    limit: int | None = None,
    spend: Spend | None = None,
    budget_usd: float | None = None,
    max_consecutive_e_failures: int = MAX_CONSECUTIVE_E_FAILURES,
    workers: int = 1,
    progress_every: int = PROGRESS_EVERY,
    checkpoint_every: int = CHECKPOINT_EVERY,
    on_checkpoint: Callable[[PrefilterResult], None] | None = None,
    echo: Callable[[str], None] = print,
) -> PrefilterResult:
    """Judge every x0 with E; keep those labelled ``keep_label``. Writes all judgments.

    Reuse: an x0 already judged in ``out_path`` keeps that judgment -- the caller has checked
    with :func:`check_prefilter_file` that every row there comes from the configured E. Only
    x0 without a judgment reach E; a recorded failure is retried.

    Concurrency (2026-09-23): up to ``workers`` live E calls are in flight on a
    thread pool, as in ``annotation/annotator_selection``. Results are consumed and written in
    the INPUT order -- the head of the queue is waited for, whatever finishes first -- so the
    file is byte-for-byte what a serial run writes, and each request (hence its cache key) is
    the one a serial run would send. ``workers=1`` is the serial run.

    Stopping: the checks run before every submission -- ``limit`` new judgments, ``spend``
    at ``budget_usd``, ``max_consecutive_e_failures`` failures in a row (in input order).
    When one fires nothing new is submitted, and the calls already in flight are still
    written: a bought judgment is never thrown away, so ``--limit`` / ``--budget`` can be
    passed by at most ``workers - 1`` calls. A request whose transient retries (429, network,
    5xx; handled by the client, see ``rafr.llm.client``) ran out stops the run with reason
    ``transient_errors``: it is not an E failure, does not enter the streak, and its x0 gets no
    row, so the next run judges it. Every stop leaves a resumable state on disk: judgments are
    appended as they are consumed, and ``on_checkpoint`` (the manifest) runs every
    ``checkpoint_every`` x0.
    """
    x0s = list(x0s)
    cache = load_prefilter_rows(out_path)
    ident = e_row_fields(ecfg) if ecfg is not None else {}
    res = PrefilterResult(kept=[], e_failed=[], n_candidates=len(x0s), workers=max(1, workers))
    recent: deque[bool] = deque(maxlen=progress_every)
    streak = 0
    t0 = time.monotonic()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def snapshot() -> PrefilterResult:
        res.elapsed_s = time.monotonic() - t0
        if spend is not None:
            res.spend_usd = spend.usd
            res.tokens = spend.tokens
        stats = getattr(client, "transient_stats", None)
        if stats is not None:
            res.transient = dict(stats)
        return res

    def progress() -> None:
        snapshot()
        pct = 100.0 * res.n_processed / max(1, res.n_candidates)
        pr = "n/a" if res.pass_rate is None else f"{res.pass_rate:.1%}"
        last = f"{sum(recent) / len(recent):.1%}" if recent else "n/a"
        fail_rate = len(res.e_failed) / max(1, res.n_judged)
        per_item = (time.monotonic() - t0) / max(1, res.n_processed)
        eta = per_item * (res.n_candidates - res.n_processed)
        tr = res.transient
        transient = (
            f" | 429 {int(tr.get('rate_limit', 0))} net {int(tr.get('network', 0))} "
            f"5xx {int(tr.get('server', 0))}"
            if tr and (tr.get("rate_limit") or tr.get("network") or tr.get("server"))
            else ""
        )
        echo(
            f"[step00] {res.n_processed}/{res.n_candidates} ({pct:.1f}%) | "
            f"kept {len(res.kept)} ({pr}, last {len(recent)}: {last}) | "
            f"E fail {len(res.e_failed)} ({fail_rate:.1%}) | retries {res.n_parse_retries} | "
            + (spend.describe() if spend is not None else "no ledger")
            + transient
            + f" | {per_item:.2f}s/item | ETA {fmt_duration(eta)}"
        )

    def checkpoint() -> None:
        r = snapshot()
        echo(
            f"[step00] --- checkpoint at {r.n_processed}/{r.n_candidates}: "
            f"kept {len(r.kept)}, reused {r.n_reused}, judged {r.n_judged}, "
            f"E failed {len(r.e_failed)}, elapsed {fmt_duration(r.elapsed_s)}"
            + (f", spend ${r.spend_usd:,.2f}" if r.spend_usd is not None else "")
            + " (manifest saved)"
        )
        if on_checkpoint is not None:
            on_checkpoint(r)

    def tick() -> None:
        if res.n_processed % progress_every == 0:
            progress()
        if res.n_processed % checkpoint_every == 0:
            checkpoint()

    def may_submit() -> bool:
        """The stop checks, before a live call; sets ``res.stop`` when one fires."""
        if limit is not None and res.n_judged >= limit:
            res.stop = Stop(
                "limit",
                f"--limit {limit} reached ({res.n_judged} new judgments this run)",
                "re-run: the x0 already processed are reused for free and the run "
                "continues from there",
            )
            return False
        if budget_usd is not None and spend is not None:
            spent = spend.usd
            if spent is not None and spent >= budget_usd:
                res.stop = Stop(
                    "budget",
                    f"--budget ${budget_usd:,.2f} reached (spent ${spent:,.2f} on "
                    f"{spend.live_calls} live calls)",
                    "re-run with a higher --budget; judgments already bought are reused",
                )
                return False
        if streak >= max_consecutive_e_failures:
            res.stop = Stop(
                "e_failures",
                f"{streak} E failures in a row (limit {max_consecutive_e_failures}): "
                "this looks like a provider-side outage, not the data",
                "fix the API side, then re-run: every successful judgment is reused",
            )
            return False
        return True

    def write(out: dict) -> None:
        fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        fh.flush()

    # (row, reused judgment | None, future | None), in input order
    queue: deque[tuple[dict, dict | None, Future | None]] = deque()
    source = iter(x0s)
    in_flight = 0
    submitting = True
    n_workers = max(1, workers)
    # APPEND: rows already on disk stay untouched and each new judgment lands as soon as it
    # is consumed, so a crash never costs a judgment that has been paid for and written.
    # load_prefilter_rows keeps the last row per (x0_id, text): a retried failure supersedes
    # its null row.
    crlf_append_notice(out_path)  # written before 2026-09-27: CRLF
    with (
        out_path.open("a", encoding="utf-8", newline="\n") as fh,
        ThreadPoolExecutor(n_workers) as pool,
    ):
        while True:
            while submitting and in_flight < n_workers:
                row = next(source, None)
                if row is None:
                    submitting = False
                    break
                cached = cache.get((row["x0_id"], row["text"]))
                if cached is not None and cached.get("label") is not None:
                    queue.append((row, cached, None))
                    continue
                if not may_submit():
                    submitting = False
                    break
                res.n_judged += 1
                fut = pool.submit(judge.judge_with_failures, client, ecfg, row["text"])
                queue.append((row, None, fut))
                in_flight += 1
            if not queue:
                break
            row, cached, fut = queue.popleft()
            if fut is None:  # reused judgment by the same E
                assert cached is not None
                res.n_reused += 1
                res.n_processed += 1
                hit = cached["label"] == keep_label
                recent.append(hit)
                if hit:
                    res.kept.append(row)
                tick()
                continue
            in_flight -= 1
            try:
                rec, failures = fut.result()
            except judge.EvaluatorError as e:
                streak += 1
                res.n_processed += 1
                res.n_parse_retries += max(0, len(e.failures) - 1)
                for reason in e.failures:
                    res.failed_attempts[reason] = res.failed_attempts.get(reason, 0) + 1
                res.e_failed.append({"x0_id": row["x0_id"], "reason": str(e)})
                write(
                    {
                        "x0_id": row["x0_id"],
                        "text": row["text"],
                        "label": None,
                        "e_error": str(e),
                        **ident,
                    }
                )
                tick()
                continue
            except TransientRetriesExhaustedError as e:
                # not a judgment, not an E failure, no row: the next run judges this x0
                res.n_judged -= 1
                res.n_transient_gave_up += 1
                if submitting or res.stop.reason == "complete":
                    res.stop = Stop(
                        "transient_errors",
                        f"x0 {row['x0_id']}: {e}",
                        "wait for the provider to recover, then re-run: every judgment "
                        "already written is reused",
                    )
                submitting = False
                continue
            streak = 0
            res.n_processed += 1
            res.n_parse_retries += len(failures)
            for reason in failures:
                res.failed_attempts[reason] = res.failed_attempts.get(reason, 0) + 1
            log.write(rec)
            write(
                {
                    "x0_id": row["x0_id"],
                    "text": row["text"],
                    "label": rec.label.value,
                    "score": rec.score,
                    "reason": rec.reason,
                    **ident,
                    "e_model_version": rec.model_version,
                    "e_request_hash": rec.prompt_hash,
                }
            )
            hit = rec.label.value == keep_label
            recent.append(hit)
            if hit:
                res.kept.append(row)
            tick()

    out = snapshot()
    if out.stop.reason != "complete":
        echo(f"[step00] STOP ({out.stop.reason}): {out.stop.detail}")
    echo(
        f"[step00] prefilter: {out.n_processed}/{out.n_candidates} x0, kept {len(out.kept)}"
        + (f" ({out.pass_rate:.1%} of judged)" if out.pass_rate is not None else "")
        + f", reused {out.n_reused}, judged now {out.n_judged}, E failed {len(out.e_failed)}, "
        f"parse retries {out.n_parse_retries}, workers {out.workers}, "
        f"elapsed {fmt_duration(out.elapsed_s)}"
        + (f", spend ${out.spend_usd:,.2f}" if out.spend_usd is not None else "")
    )
    if out.transient:
        echo(f"[step00] transient provider errors (not E failures): {out.transient}")
    if out.tokens:
        echo(f"[step00] tokens (live calls only): {out.tokens}")
    if out.stop.resume:
        echo(f"[step00] resume: {out.stop.resume}")
    return out


def write_prefiltered_pool(kept: list[dict]) -> Path:
    """``data/pools/x0_prefiltered.jsonl``: every x0 E kept -- step04's input. Written only
    when the prefilter is complete, so step04 never builds states from part of the pool."""
    path = paths.DATA_POOLS / PREFILTERED_FILE
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - a linear script with guards
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", type=Path, default=None, help="local ParaDetox snapshot dir")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after N NEW E judgments; reused judgments are free and do not count",
    )
    ap.add_argument(
        "--budget",
        type=float,
        default=None,
        help="USD ceiling for this run's live E calls (required unless --fake)",
    )
    ap.add_argument(
        "--max-consecutive-e-failures",
        type=int,
        default=MAX_CONSECUTIVE_E_FAILURES,
        help="stop after this many E failures in a row (provider-side outage)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"parallel E calls (default {DEFAULT_WORKERS}); results are written in input "
        "order either way, and 1 is the serial run",
    )
    ap.add_argument(
        "--fake", action="store_true", help="fake LLM provider (engineering check only)"
    )
    args = ap.parse_args(argv)
    if args.workers < 1:
        ap.error("--workers must be >= 1")

    cfgs = load_config()
    data_cfg, models_cfg = cfgs["data"], cfgs["models"]
    # Formal-run protection: this step consumes data.* and the evaluator config.
    assert_frozen(
        cfgs,
        only_sections=GUARD_SECTIONS,
        allow_placeholders=args.fake,
        allow_tbd=args.fake,
    )
    # The one E; refuses a prompt file whose sha256 is not the pinned one (paper Sec. 3).
    ecfg = judge.build_config(models_cfg, allow_tbd=args.fake)
    prices = prices_for(ecfg.model)
    if not args.fake:
        if args.budget is None:
            print(
                "[step00] refusing to run: --budget is required for a run that calls E. "
                "It is the only ceiling on what this step can spend; pass a number of USD "
                "(--fake makes no live calls).",
                file=sys.stderr,
            )
            return 2
        if prices is None:
            print(
                f"[step00] refusing to run: no price list for model {ecfg.model!r}, so "
                f"--budget cannot be enforced. Known models: {sorted(PRICES)}. Add the "
                "model's list price to PRICES in rafr/core/runtime.py.",
                file=sys.stderr,
            )
            return 2

    if args.fake:
        # A --fake run is an engineering self-check: it writes nothing a formal run reads.
        root = paths.use_fake_data_root()
        print(f"[step00] --fake: every output goes to {root}; no formal path is touched")
    paths.assert_fake_isolation(args.fake)

    run_dir = paths.run_dir(f"step00_{_ts()}")
    write_snapshot(
        run_dir,
        step="step00",
        configs=cfgs,
        extra={
            "fake": args.fake,
            "limit": args.limit,
            "budget_usd": args.budget,
            "max_consecutive_e_failures": args.max_consecutive_e_failures,
            "workers": args.workers,
            "prices_usd_per_mtok": None if prices is None else asdict(prices),
            "e_identity": ecfg.identity(),
        },
    )
    paths.ensure_dirs()
    manifest = Manifest.load()
    manifest.note_run(step="step00", fake=args.fake, run_dir=run_dir)
    try:
        manifest.pin_split_plan(splits_mod.split_plan(cfgs))
    except (SplitPlanChangedError, splits_mod.SplitError) as e:
        print(f"[step00] refusing to run: {e}", file=sys.stderr)
        return 5

    # E identity of what is already on disk (paper Sec. 3): refuse before spending a cent
    pre_path = paths.DATA_POOLS / PREFILTER_FILE
    try:
        n_rows = check_prefilter_file(pre_path, ecfg)
    except judge.EChangedError as e:
        print(f"[step00] refusing to run: {e}", file=sys.stderr)
        return 5
    print(
        f"[step00] E: {ecfg.model} version {ecfg.version}, effort "
        f"{ecfg.decoding.get('effort')}, prompt sha256 {ecfg.prompt_sha256[:12]} "
        f"({n_rows} prefilter rows on disk, all by this E)"
    )
    if ecfg.prompt_cache and ecfg.prefix_too_short_to_cache():
        print(
            "[step00] note: E's stable prompt prefix is far below the 512-token cache minimum, "
            "so every call is billed uncached (expected; the prompt is not changed for it)"
        )

    # 1) download
    raw_dir = args.raw_dir or (paths.DATA_RAW / "paradetox")
    # The dataset revision is pinned by the code, not configured (decision, 2026-09-17):
    # the first run resolves the repo's current commit and records it; every later run reuses
    # that pin, so the pool never follows the upstream head by accident.
    pinned = (manifest.source or {}).get("hf_revision")
    if not args.skip_download:
        repo = get(data_cfg, "source.hf_repo")
        if pinned is None:
            pinned = paradetox.resolve_revision(repo)
            print(f"[step00] pinning {repo} at revision {pinned} (first run)")
        else:
            print(f"[step00] reusing the pinned revision {pinned} from the manifest")
        raw_dir = paradetox.download(repo_id=repo, revision=pinned, dest_dir=raw_dir)
    files = paradetox.find_data_files(raw_dir)
    manifest.update_section(
        "source",
        {
            "hf_repo": get(data_cfg, "source.hf_repo"),
            "hf_revision": pinned,
            "raw_dir": str(raw_dir),
            "files": {str(p.relative_to(raw_dir)): paradetox.sha256_file(p) for p in files},
        },
    )

    # 2) clean + dedupe; the whole pool is judged, in a seeded random order
    rows = paradetox.load_raw(raw_dir, text_field=get(data_cfg, "source.text_field"))
    cleaned = paradetox.clean(
        rows,
        text_field=get(data_cfg, "source.text_field"),
        min_chars=int(get(data_cfg, "cleaning.min_chars", 1)),
    )
    order_seed = component_seed(master_seed(cfgs), "sampling")
    x0s = processing_order(cleaned.x0s, order_seed)
    manifest.update_section(
        "sampling",
        {
            "method": "every deduplicated x0 is judged; this is only the processing order",
            "seed": order_seed,
            "seed_component": "sampling",
            "rng": "random.Random via rafr.core.seeding.py_rng(seed, 'candidates').shuffle",
            "n_pool": len(cleaned.x0s),
            "reason": (
                "a --limit batch (the dry run) should be a random sample of ParaDetox, and "
                "train.tsv is ordered in blocks by source corpus"
            ),
        },
    )
    pool_path = paths.DATA_POOLS / "x0_clean.jsonl"
    paths.DATA_POOLS.mkdir(parents=True, exist_ok=True)
    with pool_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in x0s:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest.update_section(
        "cleaning",
        {**cleaned.stats, "file": str(pool_path), "sha256": paradetox.sha256_file(pool_path)},
    )
    print(f"[step00] cleaned: {cleaned.stats}; all {len(x0s)} x0 go to the E prefilter")

    # 3) E prefilter of every x0 (paper Sec. 4.2): before any split, before any randomization
    if not bool(get(data_cfg, "prefilter.enabled", True)):
        print("[step00] refusing to run: data.prefilter.enabled is false (II.2.3 requires it)")
        return 2
    keep_label = get(data_cfg, "prefilter.keep_label")
    if args.limit is not None and x0s and ecfg.provider == "anthropic":
        # the dry run shows one complete request (2026-09-23): exactly the SDK
        # arguments, no key; saved next to the run's ledger as well
        body = json.dumps(judge.request_body(ecfg, x0s[0]["text"]), ensure_ascii=False, indent=2)
        (run_dir / "e_request_body.json").write_text(body + "\n", encoding="utf-8", newline="\n")
        print(f"[step00] E request body for the first x0 (no key):\n{body}")
    client = build_client(models_cfg, run_dir=run_dir, fake=args.fake)
    client.set_transient_log(run_dir / "transient_errors.jsonl")  # one row per 429 / net / 5xx
    meta_path = paths.DATA_POOLS / "prefilter_meta.json"
    spend = Spend(client.ledger, prices, judge.PURPOSE)

    def save_state(r: PrefilterResult) -> None:
        """Checkpoint: prefilter_meta + manifest, so a crash cannot cost the pin."""
        _write_prefilter_state(
            manifest, ecfg, meta_path, pre_path, r, keep_label=keep_label, run_dir=run_dir
        )

    with JsonlWriter(run_dir / "evaluations.jsonl") as log:
        res = prefilter(
            client,
            ecfg,
            x0s,
            keep_label=keep_label,
            out_path=pre_path,
            log=log,
            limit=args.limit,
            spend=spend,
            budget_usd=args.budget,
            max_consecutive_e_failures=args.max_consecutive_e_failures,
            workers=args.workers,
            on_checkpoint=save_state,
        )
    if res.complete:
        out = write_prefiltered_pool(res.kept)
        print(
            f"[step00] prefilter COMPLETE: {len(res.kept)} of {res.n_candidates} x0 kept "
            f"({res.pass_rate:.1%} of the {res.n_labeled} judged), {len(res.e_failed)} "
            f"dropped after E failed -> {out}"
        )
    else:
        print(
            "[step00] prefilter not complete yet: x0_prefiltered.jsonl is written only when "
            "every x0 has been through E, so step04 cannot start before that"
        )
    save_state(res)
    proj = cost_projection(
        spend,
        items_bought=res.n_judged,
        items_total=res.n_candidates,
        items_left=res.n_candidates - res.n_processed + len(res.e_failed),
    )
    manifest.update_section("prefilter", {"last_run_cost": proj})
    manifest.save()
    print(render_projection(proj, step="step00", item="x0"))
    print(f"[step00] manifest -> {paths.MANIFEST}; run dir -> {run_dir}")
    return 4 if res.stop.is_error else 0


def _write_prefilter_state(
    manifest: Manifest,
    ecfg: judge.EvaluatorConfig,
    meta_path: Path,
    pre_path: Path,
    res: PrefilterResult,
    *,
    keep_label: str,
    run_dir: Path,
) -> None:
    """prefilter_meta.json + the manifest's prefilter section, from a (possibly partial) run.

    Called on every checkpoint and once at the end, so a crash leaves the revision pin, the E
    record and the counts on disk (review, 2026-09-20).
    """
    summary = res.summary_dict()
    e_record = {
        "e_config": ecfg.config_hash_payload(),
        "e_identity": ecfg.identity(),
        "e_identity_hash": ecfg.identity_hash(),
    }
    meta_path.write_text(
        json.dumps(
            {
                **e_record,
                **summary,
                "e_failed": res.e_failed,
                "keep_label": keep_label,
                "run_dir": str(run_dir),
                "rule": (
                    "II.9.1 (v0.9): prefilter, y* and outcome come from one E; a re-run "
                    "reuses judgments of the same E only, and a file written by another E "
                    "stops step00 (C.15)"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest.update_section(
        "prefilter",
        {
            **e_record,
            **summary,
            "keep_label": keep_label,
            "file": str(pre_path),
            "sha256": paradetox.sha256_file(pre_path) if pre_path.exists() else None,
            "x0_prefiltered": str(paths.DATA_POOLS / PREFILTERED_FILE) if res.complete else None,
        },
    )
    manifest.save()


if __name__ == "__main__":
    sys.exit(main())
