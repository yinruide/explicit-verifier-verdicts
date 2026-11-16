"""Step 04 — build the state pool with G_fixed (step 4, paper Sec. 4.2).

One state for EVERY x0 that passed the E prefilter (the states are built before the
split, which step05 draws after E has judged them). Invalid outputs are regenerated with the
next seed up to ``generation.g_fixed.max_attempts``; then the x0 is dropped. Every attempt is
logged. G_fixed itself -- model, prompt, decoding, seed derivation, invalid-output rules -- is
unchanged from round 1; its config hash is printed and recorded so that can be checked.

Preconditions: the step00 prefilter is complete (``manifest.prefilter.complete``), so the
states are never built from part of the pool.

Formal-run protection: refuses while the controller / G_fixed backbone, decoding, seed,
regeneration cap, non-English threshold, refusal phrase list or G_fixed prompt are TBD /
PLACEHOLDER. There is no dry-run exception here (the authors).

Spending controls (2026-09-20 / 2026-09-23):

* ``--budget`` is required. The run stops once the live tokens it has bought reach it.
* ``--limit N`` stops after N x0 have been generated -- the dry run of this paid segment; it
  prints the measured price per x0 and the projected cost of the whole pool.
* it stops after ``--max-consecutive-failures`` x0 in a row produce nothing valid (default
  10), which is what a provider-side problem looks like from in here.
* a progress line every 50 x0 and a checkpoint every 500 that saves the manifest.

Parallel calls (2026-09-23, as step00): ``--workers`` (default 4) x0 in flight on a
thread pool; each x0's attempts run in order inside its worker, and the results are written in
input order, so every file is byte-for-byte that of a serial run (``--workers 1``) and every
request and cache key is the same. A 429, a connection error or a 5xx is retried by the LLM
client with the counted backoff (``models.controller.network``; the OpenAI SDK's own retries
are off) -- it is not an attempt, not an invalid output and not part of the consecutive-failure
count; only when one request's backoff runs out does the run stop (``transient_errors``), to
be resumed later. Each such event is a row of ``runs/step04_<ts>/transient_errors.jsonl``.

**Resume is free.** Every state is appended the moment it is consumed, and a re-run skips x0
that already have one; x0 already recorded in ``dropped.jsonl`` are skipped too, so the
regeneration attempts they cost are not bought twice.

Outputs: ``data/pools/states.jsonl`` (one state per x0, no split yet),
``data/pools/generation_attempts.jsonl``, ``data/pools/dropped.jsonl``, manifest section
``states``; snapshot in ``runs/step04_<ts>/``.
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
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from rafr.contracts.schemas import State
from rafr.core import paths
from rafr.core.config import assert_frozen, load_config, master_seed
from rafr.core.runtime import (  # shared with step00 / step05: the steps that buy thousands
    PRICES,
    Spend,
    Stop,
    build_client,
    cost_projection,
    fmt_duration,
    prices_for,
    render_projection,
    transient_suffix,
)
from rafr.core.seeding import component_seed
from rafr.core.textio import crlf_append_notice
from rafr.data.manifest import Manifest
from rafr.data.paradetox import sha256_file
from rafr.generation import g_fixed
from rafr.llm.client import TransientRetriesExhaustedError
from rafr.logging.snapshot import write_snapshot
from rafr.logging.writer import JsonlWriter, read_jsonl

PROGRESS_EVERY = 50
"""x0 between progress lines."""

CHECKPOINT_EVERY = 500
"""x0 between checkpoint summaries; each one also saves the manifest."""

DEFAULT_WORKERS = 4
"""x0 generated in parallel (2026-09-23: 4, as step00). Results are written in input
order whatever the number; ``--workers 1`` is the serial run."""

MAX_CONSECUTIVE_FAILURES = 10
"""Default for ``--max-consecutive-failures``: x0 in a row that produced nothing valid.
Each one already costs ``max_attempts`` calls, so 10 in a row is 30 calls spent on what is
most likely a provider-side problem rather than the data."""

STATES_FILE = "states.jsonl"


def _ts() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def load_x0s(path: Path) -> list[tuple[str, str]]:
    """``(x0_id, text)`` in file order (step00 wrote the kept x0 in its processing order)."""
    return [(r["x0_id"], r["text"]) for r in read_jsonl(path)]


def states_path() -> Path:
    return paths.DATA_POOLS / STATES_FILE


def dropped_path() -> Path:
    return paths.DATA_POOLS / "dropped.jsonl"


GUARD_SECTIONS: tuple[str, ...] = (
    "models.controller",
    "models.g_fixed",
    "generation",
    "refusal_phrases",
)
"""Config sections step04 actually reads; the formal-run guard checks these and nothing else
(same rule as step11 / step12). Named here so a test can check the guard without running the
step -- running it without ``--fake`` builds the real providers."""


def load_existing() -> dict[str, State]:
    p = states_path()
    return {r["x0_id"]: State.model_validate(r) for r in read_jsonl(p)} if p.exists() else {}


def load_dropped() -> dict[str, dict]:
    """x0 that already ran out of attempts, keyed by x0_id (paper Sec. 4.2: they are dropped).

    A re-run skips them instead of buying ``max_attempts`` more calls on each.
    """
    p = dropped_path()
    return {r["x0_id"]: r for r in read_jsonl(p)} if p.exists() else {}


@dataclass
class BuildResult:
    """What this run did."""

    n_todo: int = 0  # x0 with no state and no drop record when the run started
    n_done: int = 0  # x0 processed this run (generated or dropped)
    n_generated: int = 0
    n_dropped: int = 0
    n_attempts: int = 0
    n_invalid: int = 0  # attempts that came back invalid (refusal / empty / same / non-EN)
    invalid_by_reason: dict[str, int] = field(default_factory=dict)
    n_reused_states: int = 0  # already on disk, not re-bought
    n_skipped_dropped: int = 0  # already out of attempts on an earlier run
    n_transient_gave_up: int = 0  # x0 whose 429 / network / 5xx retries ran out: no row
    transient: dict[str, float] = field(default_factory=dict)  # the client's transient_stats
    workers: int = 1
    stop: Stop = field(default_factory=lambda: Stop("complete"))
    elapsed_s: float = 0.0
    spend_usd: float | None = None
    tokens: dict[str, int] = field(default_factory=dict)

    @property
    def n_regenerations(self) -> int:
        """Attempts beyond the first one for an x0 (the design's "next seed" re-rolls)."""
        return max(0, self.n_attempts - self.n_done)

    def summary_dict(self) -> dict:
        return {
            "n_todo": self.n_todo,
            "n_done": self.n_done,
            "n_generated": self.n_generated,
            "n_dropped": self.n_dropped,
            "n_attempts": self.n_attempts,
            "n_invalid_attempts": self.n_invalid,
            "invalid_by_reason": dict(sorted(self.invalid_by_reason.items())),
            "n_regenerations": self.n_regenerations,
            "n_reused_states": self.n_reused_states,
            "n_skipped_dropped": self.n_skipped_dropped,
            "n_transient_gave_up": self.n_transient_gave_up,
            "transient": self.transient,
            "workers": self.workers,
            "stop_reason": self.stop.reason,
            "stop_detail": self.stop.detail,
            "elapsed_s": round(self.elapsed_s, 1),
            "spend_usd": None if self.spend_usd is None else round(self.spend_usd, 4),
            "tokens": self.tokens,
        }


def build_states(  # noqa: C901 - one loop with the stop conditions it must not outrun
    client,
    cfg,
    work: list[tuple[str, str]],
    *,
    write_state: Callable[[State], None],
    write_dropped: Callable[[dict], None],
    attempts_log: JsonlWriter | None = None,
    limit: int | None = None,
    spend: Spend | None = None,
    budget_usd: float | None = None,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
    workers: int = 1,
    n_reused_states: int = 0,
    n_skipped_dropped: int = 0,
    progress_every: int = PROGRESS_EVERY,
    checkpoint_every: int = CHECKPOINT_EVERY,
    on_checkpoint: Callable[[BuildResult], None] | None = None,
    echo: Callable[[str], None] = print,
) -> BuildResult:
    """Generate one state per ``(x0_id, text)`` in ``work``, stopping when told to.

    ``work`` holds only the x0 that still need a state: the caller has already taken out the
    ones with a state on disk and the ones already dropped, and counted them into the result.
    Each state is handed to ``write_state`` as soon as it is consumed, so a stop -- for any of
    the reasons below -- leaves nothing in memory that is not also on disk.

    Concurrency (2026-09-23, as step00's prefilter): up to ``workers`` x0 are in
    flight on a thread pool, each running :func:`g_fixed.generate_state` -- its attempts in
    order, with the seeds and requests they always had. Results are consumed in the INPUT
    order (the head of the queue is waited for, whatever finishes first), and only then are the
    attempts logged and the state or drop record written, so every file is byte-for-byte what
    a serial run writes. ``workers=1`` is the serial run.

    Stopping: the checks run before every submission -- ``limit`` x0 submitted this run,
    ``spend`` at ``budget_usd``, ``max_consecutive_failures`` x0 in a row with nothing valid
    (in input order). When one fires nothing new is submitted, and the x0 already in flight
    are still written: a bought state is never thrown away, so ``--budget`` and the failure
    stop can be passed by at most ``workers - 1`` x0 (``--limit`` counts submissions and is
    exact). An x0 whose transient retries (429, network, 5xx; handled by the client, see
    ``rafr.llm.client``) ran out stops the run with reason ``transient_errors``: it is not an
    invalid attempt, does not enter the streak, and gets no state, no drop record and no logged
    attempt -- the next run generates it, its earlier attempts coming from the LLM cache.
    """
    res = BuildResult(
        n_todo=len(work),
        n_reused_states=n_reused_states,
        n_skipped_dropped=n_skipped_dropped,
        workers=max(1, workers),
    )
    by_reason: Counter[str] = Counter()
    streak = 0
    n_submitted = 0
    t0 = time.monotonic()

    def snapshot() -> BuildResult:
        res.elapsed_s = time.monotonic() - t0
        res.invalid_by_reason = dict(by_reason)
        if spend is not None:
            res.spend_usd = spend.usd
            res.tokens = spend.tokens
        stats = getattr(client, "transient_stats", None)
        if stats is not None:
            res.transient = dict(stats)
        return res

    def progress() -> None:
        snapshot()
        pct = 100.0 * res.n_done / max(1, res.n_todo)
        per_item = (time.monotonic() - t0) / max(1, res.n_done)
        eta = per_item * (res.n_todo - res.n_done)
        echo(
            f"[step04] {res.n_done}/{res.n_todo} ({pct:.1f}%) | "
            f"states {res.n_generated} | invalid {res.n_invalid} "
            f"(regen {res.n_regenerations}) | dropped {res.n_dropped} | "
            + (spend.describe() if spend is not None else "no ledger")
            + transient_suffix(res.transient)
            + f" | {per_item:.2f}s/x0 | ETA {fmt_duration(eta)}"
        )

    def checkpoint() -> None:
        r = snapshot()
        echo(
            f"[step04] --- checkpoint at {r.n_done}/{r.n_todo}: states {r.n_generated}, "
            f"dropped {r.n_dropped}, attempts {r.n_attempts}, "
            f"elapsed {fmt_duration(r.elapsed_s)}"
            + (f", spend ${r.spend_usd:,.2f}" if r.spend_usd is not None else "")
            + " (manifest saved)"
        )
        if on_checkpoint is not None:
            on_checkpoint(r)

    def may_submit() -> bool:
        """The stop checks, before a live x0; sets ``res.stop`` when one fires."""
        if limit is not None and n_submitted >= limit:
            res.stop = Stop(
                "limit",
                f"--limit {limit} reached ({n_submitted} x0 generated this run)",
                "run step04 again: the states already written are reused for free and the "
                "run continues from there",
            )
            return False
        if budget_usd is not None and spend is not None:
            spent = spend.usd
            if spent is not None and spent >= budget_usd:
                res.stop = Stop(
                    "budget",
                    f"--budget ${budget_usd:,.2f} reached (spent ${spent:,.2f} on "
                    f"{spend.live_calls} live calls)",
                    "run step04 again with a higher --budget; the states already written "
                    "are reused for free",
                )
                return False
        if streak >= max_consecutive_failures:
            res.stop = Stop(
                "generation_failures",
                f"{streak} x0 in a row produced nothing valid (limit "
                f"{max_consecutive_failures}): that looks like the backbone, not the data",
                "fix the API side, then run step04 again: every state already written is "
                "reused, and the x0 dropped so far are not retried",
            )
            return False
        return True

    # (x0_id, future of generate_state), in input order; every entry is a live x0
    queue: deque[tuple[str, Future]] = deque()
    source = iter(work)
    submitting = True
    n_workers = max(1, workers)
    with ThreadPoolExecutor(n_workers) as pool:
        while True:
            while submitting and len(queue) < n_workers:
                item = next(source, None)
                if item is None:
                    submitting = False
                    break
                # --- everything below costs money; the stop conditions come first ---
                if not may_submit():
                    submitting = False
                    break
                n_submitted += 1
                x0_id, x0 = item
                # no log here: the attempts are logged below, in input order
                queue.append((x0_id, pool.submit(g_fixed.generate_state, client, cfg, x0_id, x0)))
            if not queue:
                break
            x0_id, fut = queue.popleft()
            try:
                state, attempts = fut.result()
            except TransientRetriesExhaustedError as e:
                # not an attempt, not a drop, no row: the next run generates this x0
                res.n_transient_gave_up += 1
                if submitting or res.stop.reason == "complete":
                    res.stop = Stop(
                        "transient_errors",
                        f"x0 {x0_id}: {e}",
                        "wait for the provider to recover, then run step04 again: every "
                        "state already written is reused",
                    )
                submitting = False
                continue
            res.n_done += 1
            res.n_attempts += len(attempts)
            for a in attempts:
                if attempts_log is not None:
                    attempts_log.write(a)
                if not a.valid:
                    res.n_invalid += 1
                    by_reason[a.invalid_reason.value if a.invalid_reason else "unknown"] += 1
            if state is None:
                streak += 1
                res.n_dropped += 1
                write_dropped(
                    {
                        "x0_id": x0_id,
                        "reasons": [a.invalid_reason.value for a in attempts if a.invalid_reason],
                    }
                )
            else:
                streak = 0
                res.n_generated += 1
                write_state(state)
            if res.n_done % progress_every == 0:
                progress()
            if res.n_done % checkpoint_every == 0:
                checkpoint()

    out = snapshot()
    if out.stop.reason != "complete":
        echo(f"[step04] STOP ({out.stop.reason}): {out.stop.detail}")
    echo(
        f"[step04] generation: {out.n_done}/{out.n_todo} x0 this run, "
        f"{out.n_generated} states, {out.n_dropped} dropped, {out.n_attempts} attempts "
        f"({out.n_invalid} invalid {out.invalid_by_reason}, {out.n_regenerations} "
        f"regenerations), workers {out.workers}, elapsed {fmt_duration(out.elapsed_s)}"
        + (f", spend ${out.spend_usd:,.2f}" if out.spend_usd is not None else "")
    )
    if out.n_reused_states or out.n_skipped_dropped:
        echo(
            f"[step04] reused: {out.n_reused_states} states already on disk, "
            f"{out.n_skipped_dropped} x0 already dropped; neither was bought again"
        )
    if out.transient:
        echo(f"[step04] transient provider errors (not invalid attempts): {out.transient}")
    if out.tokens:
        echo(f"[step04] tokens (live calls only): {out.tokens}")
    if out.stop.resume:
        echo(f"[step04] resume: {out.stop.resume}")
    return out


def attempt_totals(path: Path) -> dict:
    """Invalid attempts by reason over every attempt ever logged (all runs), for the manifest
    and the segment report; this run's own counts are in :class:`BuildResult`."""
    by_reason: Counter[str] = Counter()
    n = 0
    if path.exists():
        for r in read_jsonl(path):
            n += 1
            if not r.get("valid"):
                by_reason[r.get("invalid_reason") or "unknown"] += 1
    return {"n_attempts": n, "invalid_by_reason": dict(sorted(by_reason.items()))}


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - a linear script with guards
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--x0-file", type=Path, default=None, help="default data/pools/x0_prefiltered")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after N x0 have been generated this run (states already on disk are free "
        "and do not count)",
    )
    ap.add_argument(
        "--budget",
        type=float,
        default=None,
        help="USD ceiling for this run's live G_fixed calls (required)",
    )
    ap.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=MAX_CONSECUTIVE_FAILURES,
        help="stop after this many x0 in a row produce nothing valid",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"x0 generated in parallel (default {DEFAULT_WORKERS}); results are written in "
        "input order either way, and 1 is the serial run",
    )
    args = ap.parse_args(argv)
    if args.workers < 1:
        ap.error("--workers must be >= 1")

    cfgs = load_config()
    assert_frozen(
        cfgs,
        only_sections=GUARD_SECTIONS,
        extra_files=[cfgs["generation"]["g_fixed"]["validity"]["refusal_phrases_file"]],
    )
    cfg = g_fixed.build_config(
        cfgs["models"],
        cfgs["generation"],
        allow_tbd=False,
        base_seed=component_seed(master_seed(cfgs), "g_fixed"),
    )
    gf_hash = g_fixed.config_hash(cfg)
    prices = prices_for(cfg.model)
    if args.budget is None:
        print(
            "[step04] refusing to run: --budget is required. It is the only ceiling on what "
            "this step can spend (one call per x0, up to "
            f"{cfg.max_attempts} for the awkward ones); pass a number of USD.",
            file=sys.stderr,
        )
        return 2
    if prices is None:
        print(
            f"[step04] refusing to run: no price list for model {cfg.model!r}, so --budget "
            f"cannot be enforced. Known models: {sorted(PRICES)}. Add the model's list price "
            "to PRICES in rafr/core/runtime.py (it is code, not frozen config).",
            file=sys.stderr,
        )
        return 2
    paths.assert_fake_isolation(False)  # step04 has no fake mode and must never write one
    manifest = Manifest.load()
    if not (manifest.prefilter or {}).get("complete"):
        print(
            "[step04] refusing to run: the step00 prefilter is not complete (manifest "
            "prefilter.complete is not true). States are built for EVERY x0 E kept, so the "
            "whole pool has to be judged first.",
            file=sys.stderr,
        )
        return 2
    x0_file = args.x0_file or (paths.DATA_POOLS / "x0_prefiltered.jsonl")
    if not x0_file.exists():
        print(f"[step04] refusing to run: {x0_file} does not exist (run step00)", file=sys.stderr)
        return 2

    run_dir = paths.run_dir(f"step04_{_ts()}")
    # The design's `temperature > 0` is now satisfied by the provider default rather than by a
    # configured value; the snapshot says so instead of just carrying a bare null.
    temp_rec = g_fixed.temperature_record(cfg)
    write_snapshot(
        run_dir,
        step="step04",
        configs=cfgs,
        extra={
            "g_fixed_temperature": temp_rec,
            "g_fixed_config_hash": gf_hash,
            "g_fixed_config": g_fixed.config_hash_payload(cfg),
            "limit": args.limit,
            "budget_usd": args.budget,
            "max_consecutive_failures": args.max_consecutive_failures,
            "workers": args.workers,
            "prices_usd_per_mtok": asdict(prices),
        },
    )
    print(f"[step04] G_fixed config hash {gf_hash} ({cfg.model} {cfg.version})")
    print(f"[step04] temperature: sent={temp_rec['sent']}; {temp_rec['satisfied_by']}")
    paths.ensure_dirs()
    client = build_client(cfgs["models"], run_dir=run_dir, fake=False)
    client.set_transient_log(run_dir / "transient_errors.jsonl")  # one row per 429 / net / 5xx
    manifest.note_run(step="step04", fake=False, run_dir=run_dir)
    spend = Spend(client.ledger, prices, g_fixed.PURPOSE)

    # --- the work list: what has no state yet and has not already been dropped -----------
    x0s = load_x0s(x0_file)
    have = load_existing()
    dropped_already = load_dropped()
    work = [(i, t) for i, t in x0s if i not in have and i not in dropped_already]
    n_reused = sum(1 for i, _ in x0s if i in have)
    n_skipped = sum(1 for i, _ in x0s if i not in have and i in dropped_already)
    print(
        f"[step04] {len(x0s)} prefiltered x0: {n_reused} have a state, {n_skipped} already "
        f"dropped, {len(work)} to generate; budget ${args.budget:,.2f}; workers {args.workers}"
    )

    # --- generate, appending every state as it is produced --------------------------------
    attempts_path = paths.DATA_POOLS / "generation_attempts.jsonl"
    attempts_log = JsonlWriter(attempts_path)
    dropped_log = JsonlWriter(dropped_path())
    crlf_append_notice(states_path())  # written before 2026-09-27: CRLF
    states_fh = states_path().open("a", encoding="utf-8", newline="\n")

    def write_state(state: State) -> None:
        states_fh.write(state.model_dump_json() + "\n")
        states_fh.flush()

    def save_state(r: BuildResult) -> None:
        _write_states_section(manifest, r, len(x0s), run_dir, cfg, gf_hash, attempts_path)

    try:
        res = build_states(
            client,
            cfg,
            work,
            write_state=write_state,
            write_dropped=dropped_log.write,
            attempts_log=attempts_log,
            limit=args.limit,
            spend=spend,
            budget_usd=args.budget,
            max_consecutive_failures=args.max_consecutive_failures,
            workers=args.workers,
            n_reused_states=n_reused,
            n_skipped_dropped=n_skipped,
            on_checkpoint=save_state,
        )
    finally:
        attempts_log.close()
        dropped_log.close()
        states_fh.close()
    save_state(res)
    proj = cost_projection(
        spend, items_bought=res.n_done, items_total=len(x0s), items_left=len(work) - res.n_done
    )
    manifest.update_section("states", {"last_run_cost": proj})
    manifest.save()
    print(render_projection(proj, step="step04", item="x0"))
    print(f"[step04] ledger: {json.dumps(client.ledger.summary() if client.ledger else {})}")
    print(f"[step04] manifest -> {paths.MANIFEST}; run dir -> {run_dir}")
    return 4 if res.stop.is_error else 0


def _write_states_section(
    manifest: Manifest,
    res: BuildResult,
    n_x0: int,
    run_dir: Path,
    cfg,
    gf_hash: str,
    attempts_path: Path,
) -> None:
    """Manifest ``states``: pool totals (from the files, all runs) plus this run's counts."""
    n_states = len(load_existing())
    n_dropped = len(load_dropped())
    manifest.update_section(
        "states",
        {
            "n_x0": n_x0,
            "n_states": n_states,
            "n_dropped": n_dropped,
            "complete": n_states + n_dropped == n_x0,
            "file": str(states_path()),
            "sha256": sha256_file(states_path()) if states_path().exists() else None,
            "attempts": attempt_totals(attempts_path),
            "g_fixed_config_hash": gf_hash,
            "g_fixed_model": cfg.model,
            "g_fixed_version": cfg.version,
            "run_dir": str(run_dir),
            "run": res.summary_dict(),
        },
    )
    manifest.save()


if __name__ == "__main__":
    sys.exit(main())
