"""Step 05 — E judges every state (= y*), then the split (protocol v0.9: 0.5 step 5, II.2.3,
II.2.4, II.3).

    python pipelines/step05_label_and_split.py judge --workers 4 --budget 1 --limit 20  # dry run
    python pipelines/step05_label_and_split.py judge --workers 4 --budget 30  # every state
    python pipelines/step05_label_and_split.py split                          # no API calls

``judge`` -- one E call per state (II.2.3: a single judgment, and it IS y*, II.3). Every
judgment is appended to ``data/labels/state_judgments.jsonl`` the moment it is made, with the
E that made it; a re-run reuses them and retries only failures. A state on which E still fails
after its 2 retries is dropped together with its x0 (II.9.1), before the split. When every
state has been through E the file is complete and its sha256 goes into the manifest
(``labels``) -- after that it is never written again. Same spending controls as step00 /
step04: ``--budget`` required, ``--limit`` for the dry run (it prints the measured price per
state and the projected total), stop after ``--max-consecutive-e-failures`` failures in a row.
Same concurrency as step00 (2026-09-23): ``--workers`` (default 4) E calls in flight,
results written in input order, so the file is byte-for-byte that of a serial run
(``--workers 1``) and every request and cache key is the same. A 429, a connection error or a
5xx is retried by the LLM client with the counted backoff (``models.evaluator.network``) -- not
an E attempt, not an E failure, not part of the consecutive-failure count; only when one
request's backoff runs out does the run stop (``transient_errors``), to be resumed later. Each
such event is a row of ``runs/step05_judge_<ts>/transient_errors.jsonl``.

``split`` -- no API calls. Checks the judgment file against the recorded sha256 and the split
plan against the one pinned in the manifest before any state was judged (II.2.4), then draws
(rafr.data.splits.PROCEDURE): D_fit, D_gate, D_gate_reserve at random from all states, D_Exp1
at random from the remaining E-UNSAFE states, the rest to the unlabeled pool. Too few E-UNSAFE
states left -> exit 3 and nothing written; no other split is shrunk. Writes
``data/splits/<split>.jsonl`` and ``data/pools/states_<split>.jsonl`` and reports, per split,
sizes, and the E-UNSAFE count and share of D_fit only (2026-09-23): no other split's
y* is opened, so the reserve's count cannot be recovered from the reports (II.6.1).

Formal-run protection covers exactly what this step reads: the seeds, the split plan, the
labels section and the one E (whose prompt sha256 is re-checked, II.9.1).
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

from rafr.contracts.schemas import SafetyLabel, Split, State
from rafr.core import paths
from rafr.core.config import assert_frozen, load_config
from rafr.core.runtime import (
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
from rafr.core.textio import crlf_append_notice
from rafr.data import labels as L
from rafr.data import splits as S
from rafr.data.manifest import Manifest, SplitPlanChangedError
from rafr.data.paradetox import sha256_file
from rafr.evaluator import judge
from rafr.llm.client import TransientRetriesExhaustedError
from rafr.logging.snapshot import write_snapshot
from rafr.logging.writer import read_jsonl

GUARD_SECTIONS: tuple[str, ...] = (
    "data.seeds",
    "data.splits",
    "data.labels",
    "models.evaluator",
)
"""Config sections step05 reads; the formal-run guard checks these and nothing else."""

PROGRESS_EVERY = 50
CHECKPOINT_EVERY = 500
MAX_CONSECUTIVE_E_FAILURES = 10

DEFAULT_WORKERS = 4
"""Parallel E calls of ``judge`` (2026-09-23: 4, as step00). Results are written in
input order whatever the number; ``--workers 1`` is the serial run."""


def _ts() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def load_states() -> list[State]:
    p = paths.DATA_POOLS / "states.jsonl"
    return [State.model_validate(r) for r in read_jsonl(p)] if p.exists() else []


def split_states_path(split: Split) -> Path:
    return paths.DATA_POOLS / f"states_{split.value}.jsonl"


# ---------------------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------------------


@dataclass
class JudgeResult:
    n_states: int = 0
    n_reused: int = 0  # already judged by this E on an earlier run
    n_judged: int = 0  # sent to E in this run (failures included)
    n_processed: int = 0
    failed: list[dict] = field(default_factory=list)  # this run's failures
    n_parse_retries: int = 0  # re-sends after a refusal / unreadable answer (II.9.1)
    failed_attempts: dict[str, int] = field(default_factory=dict)  # by reason, all attempts
    n_transient_gave_up: int = 0  # states whose 429 / network / 5xx retries ran out: no row
    transient: dict[str, float] = field(default_factory=dict)  # the client's transient_stats
    workers: int = 1
    stop: Stop = field(default_factory=lambda: Stop("complete"))
    elapsed_s: float = 0.0
    spend_usd: float | None = None
    tokens: dict[str, int] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.stop.reason == "complete" and self.n_processed == self.n_states

    def summary_dict(self) -> dict:
        return {
            "n_states": self.n_states,
            "n_processed": self.n_processed,
            "n_reused": self.n_reused,
            "n_judged_this_run": self.n_judged,
            "n_e_failed_this_run": len(self.failed),
            "n_parse_retries_this_run": self.n_parse_retries,
            "failed_attempts_by_reason_this_run": dict(sorted(self.failed_attempts.items())),
            "n_transient_gave_up_this_run": self.n_transient_gave_up,
            "transient_this_run": self.transient,
            "workers": self.workers,
            "stop_reason": self.stop.reason,
            "stop_detail": self.stop.detail,
            "elapsed_s": round(self.elapsed_s, 1),
            "spend_usd": None if self.spend_usd is None else round(self.spend_usd, 4),
            "tokens": self.tokens,
        }


def check_rows_same_e(rows: dict[str, L.StateJudgment], ecfg: judge.EvaluatorConfig) -> None:
    for r in rows.values():
        judge.check_same_e(r.e_identity, ecfg, where=str(L.y_star_path()))


def judge_states(  # noqa: C901 - one loop with the stop conditions it must not outrun
    client,
    ecfg: judge.EvaluatorConfig,
    states: list[State],
    *,
    out_path: Path,
    limit: int | None = None,
    spend: Spend | None = None,
    budget_usd: float | None = None,
    max_consecutive_e_failures: int = MAX_CONSECUTIVE_E_FAILURES,
    workers: int = 1,
    progress_every: int = PROGRESS_EVERY,
    checkpoint_every: int = CHECKPOINT_EVERY,
    on_checkpoint: Callable[[JudgeResult], None] | None = None,
    echo: Callable[[str], None] = print,
) -> JudgeResult:
    """One E judgment per state, appended to ``out_path`` as it is consumed (II.2.3, II.3).

    States with a judgment by this E are reused; a recorded failure is retried.

    Concurrency (2026-09-23, as step00's prefilter): up to ``workers`` live E calls
    are in flight on a thread pool. Results are consumed and written in the INPUT order -- the
    head of the queue is waited for, whatever finishes first -- so the file is byte-for-byte
    what a serial run writes, and each request (hence its cache key) is the one a serial run
    would send. ``workers=1`` is the serial run.

    Stopping: the checks run before every submission -- ``limit`` new judgments, ``spend`` at
    ``budget_usd``, ``max_consecutive_e_failures`` failures in a row (in input order). When
    one fires nothing new is submitted, and the calls already in flight are still written: a
    bought judgment is never thrown away, so ``--budget`` and the failure stop can be passed by
    at most ``workers - 1`` calls (``--limit`` counts submissions and is exact). A request whose
    transient retries (429, network, 5xx; handled by the client, see ``rafr.llm.client``) ran
    out stops the run with reason ``transient_errors``: it is not an E failure, does not enter
    the streak, and its state gets no row, so the next run judges it.
    """
    existing = L.load_judgment_rows(out_path)
    ident = ecfg.identity()
    res = JudgeResult(n_states=len(states), workers=max(1, workers))
    streak = 0
    t0 = time.monotonic()
    n_unsafe = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def snapshot() -> JudgeResult:
        res.elapsed_s = time.monotonic() - t0
        if spend is not None:
            res.spend_usd = spend.usd
            res.tokens = spend.tokens
        stats = getattr(client, "transient_stats", None)
        if stats is not None:
            res.transient = dict(stats)
        return res

    def tick() -> None:
        if res.n_processed % progress_every == 0:
            snapshot()
            per = res.elapsed_s / max(1, res.n_processed)
            eta = fmt_duration(per * (res.n_states - res.n_processed))
            labelled = res.n_processed - len(res.failed)
            echo(
                f"[step05] {res.n_processed}/{res.n_states} states | UNSAFE {n_unsafe} "
                f"({n_unsafe / max(1, labelled):.1%}) | E fail {len(res.failed)} | "
                f"retries {res.n_parse_retries} | "
                + (spend.describe() if spend is not None else "no ledger")
                + transient_suffix(res.transient)
                + f" | {per:.2f}s/state | ETA {eta}"
            )
        if res.n_processed % checkpoint_every == 0 and on_checkpoint is not None:
            on_checkpoint(snapshot())

    def may_submit() -> bool:
        """The stop checks, before a live call; sets ``res.stop`` when one fires."""
        if limit is not None and res.n_judged >= limit:
            res.stop = Stop(
                "limit",
                f"--limit {limit} reached ({res.n_judged} states judged this run)",
                "re-run: judged states are reused for free",
            )
            return False
        if budget_usd is not None and spend is not None:
            spent = spend.usd
            if spent is not None and spent >= budget_usd:
                res.stop = Stop(
                    "budget",
                    f"--budget ${budget_usd:,.2f} reached (spent ${spent:,.2f})",
                    "re-run with a higher --budget; judged states are reused for free",
                )
                return False
        if streak >= max_consecutive_e_failures:
            res.stop = Stop(
                "e_failures",
                f"{streak} E failures in a row (limit {max_consecutive_e_failures})",
                "fix the API side, then re-run: every judgment already made is reused",
            )
            return False
        return True

    def count_failures(reasons: list[str]) -> None:
        for reason in reasons:
            res.failed_attempts[reason] = res.failed_attempts.get(reason, 0) + 1

    # (state, reused judgment | None, future | None), in input order
    queue: deque[tuple[State, L.StateJudgment | None, Future | None]] = deque()
    source = iter(states)
    in_flight = 0
    submitting = True
    n_workers = max(1, workers)
    # APPEND: rows already on disk stay untouched and each new judgment lands as soon as it is
    # consumed; load_judgment_rows keeps the last row per state, so a retried failure
    # supersedes its null row.
    crlf_append_notice(out_path)  # written before 2026-09-27: CRLF (D-2026-09-27)
    with (
        out_path.open("a", encoding="utf-8", newline="\n") as fh,
        ThreadPoolExecutor(n_workers) as pool,
    ):
        while True:
            while submitting and in_flight < n_workers:
                st = next(source, None)
                if st is None:
                    submitting = False
                    break
                row = existing.get(st.state_id)
                if row is not None and row.label is not None:
                    queue.append((st, row, None))
                    continue
                if not may_submit():
                    submitting = False
                    break
                res.n_judged += 1
                fut = pool.submit(judge.judge_with_failures, client, ecfg, st.text)
                queue.append((st, None, fut))
                in_flight += 1
            if not queue:
                break
            st, row, fut = queue.popleft()
            if fut is None:  # reused judgment by the same E
                assert row is not None
                res.n_reused += 1
                res.n_processed += 1
                n_unsafe += row.label is SafetyLabel.UNSAFE
                tick()
                continue
            in_flight -= 1
            try:
                rec, failures = fut.result()
            except judge.EvaluatorError as e:
                streak += 1
                res.n_processed += 1
                res.n_parse_retries += max(0, len(e.failures) - 1)
                count_failures(e.failures)
                res.failed.append({"state_id": st.state_id, "x0_id": st.x0_id, "reason": str(e)})
                out = L.StateJudgment(
                    state_id=st.state_id,
                    x0_id=st.x0_id,
                    label=None,
                    e_error=str(e),
                    e_identity=ident,
                )
            except TransientRetriesExhaustedError as e:
                # not a judgment, not an E failure, no row: the next run judges this state
                res.n_judged -= 1
                res.n_transient_gave_up += 1
                if submitting or res.stop.reason == "complete":
                    res.stop = Stop(
                        "transient_errors",
                        f"state {st.state_id}: {e}",
                        "wait for the provider to recover, then re-run: every judgment "
                        "already written is reused",
                    )
                submitting = False
                continue
            else:
                streak = 0
                res.n_processed += 1
                res.n_parse_retries += len(failures)
                count_failures(failures)
                n_unsafe += rec.label is SafetyLabel.UNSAFE
                out = L.StateJudgment(
                    state_id=st.state_id,
                    x0_id=st.x0_id,
                    label=rec.label,
                    record=rec,
                    e_identity=ident,
                )
            fh.write(out.model_dump_json() + "\n")
            fh.flush()
            tick()

    out = snapshot()
    if out.stop.reason != "complete":
        echo(f"[step05] STOP ({out.stop.reason}): {out.stop.detail}")
    echo(
        f"[step05] judge: {out.n_processed}/{out.n_states} states, reused {out.n_reused}, "
        f"judged now {out.n_judged}, E failed now {len(out.failed)}, "
        f"parse retries {out.n_parse_retries}, workers {out.workers}, "
        f"elapsed {fmt_duration(out.elapsed_s)}"
        + (f", spend ${out.spend_usd:,.2f}" if out.spend_usd is not None else "")
    )
    if out.transient:
        echo(f"[step05] transient provider errors (not E failures): {out.transient}")
    if out.stop.resume:
        echo(f"[step05] resume: {out.stop.resume}")
    return out


def judgment_totals(rows: dict[str, L.StateJudgment]) -> dict:
    """Totals over every state before any split (the authors' segment-3 report)."""
    labelled = [r for r in rows.values() if r.label is not None]
    n_unsafe = sum(1 for r in labelled if r.label is SafetyLabel.UNSAFE)
    failed = [r for r in rows.values() if r.label is None]
    return {
        "n_states": len(rows),
        "n_labelled": len(labelled),
        "n_unsafe": n_unsafe,
        "unsafe_rate": n_unsafe / len(labelled) if labelled else None,
        "n_dropped_e_failed": len(failed),
        "dropped": [
            {"state_id": r.state_id, "x0_id": r.x0_id, "reason": r.e_error} for r in failed
        ],
    }


def cmd_judge(args, cfgs, ecfg: judge.EvaluatorConfig, manifest: Manifest) -> int:
    prices = prices_for(ecfg.model)
    if args.budget is None:
        print("[step05] refusing to run: --budget is required for `judge`", file=sys.stderr)
        return 2
    if prices is None:
        print(
            f"[step05] refusing to run: no price list for {ecfg.model!r} (known: {sorted(PRICES)})",
            file=sys.stderr,
        )
        return 2
    if (manifest.labels or {}).get("complete"):
        print(
            "[step05] every state is already judged and the file's sha256 is recorded "
            f"({manifest.labels.get('sha256', '')[:12]}); y* is frozen, nothing to do. "
            "Next: `split`."
        )
        return 0
    if any(S.load_all_splits(paths.DATA_SPLITS).values()):
        print(
            "[step05] refusing to judge: data/splits/ already holds a split, and y* must be "
            "complete and hashed before any split exists (II.2.4)",
            file=sys.stderr,
        )
        return 5
    states = load_states()
    if not states or not (manifest.states or {}).get("complete"):
        print(
            "[step05] refusing to run: step04 has not built a state for every prefiltered x0 "
            "(manifest states.complete is not true)",
            file=sys.stderr,
        )
        return 2
    out_path = L.y_star_path()
    try:
        check_rows_same_e(L.load_judgment_rows(out_path), ecfg)
    except judge.EChangedError as e:
        print(f"[step05] refusing to run: {e}", file=sys.stderr)
        return 5

    run_dir = paths.run_dir(f"step05_judge_{_ts()}")
    write_snapshot(
        run_dir,
        step="step05_judge",
        configs=cfgs,
        extra={
            "limit": args.limit,
            "budget_usd": args.budget,
            "max_consecutive_e_failures": args.max_consecutive_e_failures,
            "workers": args.workers,
            "e_identity": ecfg.identity(),
            "prices_usd_per_mtok": asdict(prices),
        },
    )
    manifest.note_run(step="step05_judge", fake=False, run_dir=run_dir)
    client = build_client(cfgs["models"], run_dir=run_dir, fake=False)
    client.set_transient_log(run_dir / "transient_errors.jsonl")  # one row per 429 / net / 5xx
    spend = Spend(client.ledger, prices, judge.PURPOSE)
    print(
        f"[step05] E {ecfg.model} {ecfg.version} effort {ecfg.decoding.get('effort')} prompt "
        f"{ecfg.prompt_sha256[:12]}; {len(states)} states; budget ${args.budget:,.2f}; "
        f"workers {args.workers}"
    )
    if args.limit is not None and ecfg.provider == "anthropic":
        # the dry run shows one complete request (2026-09-23): no key
        body = json.dumps(judge.request_body(ecfg, states[0].text), ensure_ascii=False, indent=2)
        (run_dir / "e_request_body.json").write_text(body + "\n", encoding="utf-8", newline="\n")
        print(f"[step05] E request body for the first state (no key):\n{body}")

    def save(r: JudgeResult) -> None:
        manifest.update_section(
            "labels",
            {
                "complete": False,
                "file": str(out_path),
                "e_identity": ecfg.identity(),
                "e_identity_hash": ecfg.identity_hash(),
                "run_dir": str(run_dir),
                "n_states": r.n_states,
                "n_processed_last_run": r.n_processed,
                "run": r.summary_dict(),
            },
        )
        manifest.save()

    res = judge_states(
        client,
        ecfg,
        states,
        out_path=out_path,
        limit=args.limit,
        spend=spend,
        budget_usd=args.budget,
        max_consecutive_e_failures=args.max_consecutive_e_failures,
        workers=args.workers,
        on_checkpoint=save,
    )
    save(res)
    rows = L.load_judgment_rows(out_path)
    if res.complete:
        totals = judgment_totals(rows)
        digest = sha256_file(out_path)
        manifest.update_section(
            "labels",
            {
                "complete": True,
                "sha256": digest,
                "recorded_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
                **totals,
            },
        )
        print(
            f"[step05] y* COMPLETE: {totals['n_states']} states, {totals['n_unsafe']} E-UNSAFE "
            f"({totals['unsafe_rate']:.1%} of {totals['n_labelled']} judged), "
            f"{totals['n_dropped_e_failed']} dropped with their x0 (E failed); "
            f"sha256 {digest} recorded. Next: `split`."
        )
    proj = cost_projection(
        spend,
        items_bought=res.n_judged,
        items_total=res.n_states,
        items_left=res.n_states - res.n_processed + len(res.failed),
    )
    manifest.update_section("labels", {"last_run_cost": proj})
    manifest.save()
    print(render_projection(proj, step="step05", item="state"))
    print(f"[step05] manifest -> {paths.MANIFEST}; run dir -> {run_dir}")
    return 4 if res.stop.is_error else 0


# ---------------------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------------------


def cmd_split(args, cfgs, ecfg: judge.EvaluatorConfig, manifest: Manifest) -> int:  # noqa: C901
    lab = manifest.labels or {}
    y_path = L.y_star_path()
    if not lab.get("complete") or not lab.get("sha256"):
        print(
            "[step05] refusing to split: not every state has been judged yet (run `judge`; "
            "the split needs y* of every state, II.2.4)",
            file=sys.stderr,
        )
        return 2
    digest = sha256_file(y_path) if y_path.exists() else None
    if digest != lab["sha256"]:
        print(
            f"[step05] refusing to split: {y_path} hashes to {digest}, but the manifest "
            f"recorded {lab['sha256']} when judging finished. y* may not change after that.",
            file=sys.stderr,
        )
        return 5
    rows = L.load_judgment_rows(y_path)
    try:
        check_rows_same_e(rows, ecfg)
    except judge.EChangedError as e:
        print(f"[step05] refusing to split: {e}", file=sys.stderr)
        return 5
    plan = manifest.splits.get("plan")
    seed = int(plan["seed"])
    sizes = dict(plan["sizes"])

    by_x0 = {r.x0_id: r for r in rows.values()}
    dropped = [r for r in rows.values() if r.label is None]
    judged_x0 = [x for x, r in by_x0.items() if r.label is not None]
    states = {s.x0_id: s for s in load_states()}
    missing = [x for x in judged_x0 if x not in states]
    if missing:
        print(
            f"[step05] refusing to split: {len(missing)} judged x0 have no state "
            f"(e.g. {missing[0]})",
            file=sys.stderr,
        )
        return 2

    def is_unsafe(x0_id: str) -> bool:
        return by_x0[x0_id].label is SafetyLabel.UNSAFE

    try:
        split_map = S.make_splits(judged_x0, is_unsafe, sizes, seed)
    except S.InsufficientUnsafeError as e:
        print(f"[step05] STOP: {e}", file=sys.stderr)
        print(
            f"[step05] states with a y*: {len(judged_x0)}; after the three random splits "
            f"({sum(int(sizes[sp.value]) for sp in S.RANDOM_SPLITS)}) {e.n_remaining} remain, "
            f"{e.have} of them E-UNSAFE, D_Exp1 needs {e.need}. Nothing was written.",
            file=sys.stderr,
        )
        return 3

    existing = S.load_all_splits(paths.DATA_SPLITS)
    if any(existing.values()):
        if existing == split_map:
            print("[step05] data/splits/ already holds exactly this split; nothing rewritten")
        else:
            print(
                "[step05] refusing to split: data/splits/ holds a DIFFERENT split. The seed and "
                "procedure were pinned before judging and y* is hashed, so a recomputation "
                "should reproduce it exactly -- find out why before touching anything.",
                file=sys.stderr,
            )
            return 5
    else:
        paths.assert_fake_isolation(False)
        run_dir = paths.run_dir(f"step05_split_{_ts()}")
        write_snapshot(run_dir, step="step05_split", configs=cfgs, extra={"plan": plan})
        manifest.note_run(step="step05_split", fake=False, run_dir=run_dir)
        written = S.save_splits(split_map, paths.DATA_SPLITS)
        state_files = {}
        for sp, ids in split_map.items():
            p = split_states_path(sp)
            with p.open("w", encoding="utf-8", newline="\n") as fh:
                for x0 in ids:
                    fh.write(states[x0].model_copy(update={"split": sp}).model_dump_json() + "\n")
            state_files[sp.value] = {"path": str(p), "sha256": sha256_file(p)}
        manifest.update_section(
            "splits",
            {
                "run_mode": "formal",
                "sizes": {sp.value: len(ids) for sp, ids in split_map.items()},
                "files": {
                    k: {"path": str(p), "sha256": sha256_file(p)} for k, p in written.items()
                },
                "state_files": state_files,
                "y_star_sha256": lab["sha256"],
                "n_x0_with_y_star": len(judged_x0),
                "n_dropped_before_split": len(dropped),
                "dropped_before_split": [
                    {"state_id": r.state_id, "x0_id": r.x0_id, "reason": r.e_error} for r in dropped
                ],
                "run_dir": str(run_dir),
            },
        )
    report = L.split_report(split_map)
    manifest.update_section("splits", {"report": report})
    manifest.save()
    print(
        f"[step05] split ({plan['procedure']}, seed {seed} = component {plan['seed_component']}):"
    )
    for name, r in report.items():
        if "n_unsafe" in r:
            rate = "n/a" if r["unsafe_rate"] is None else f"{r['unsafe_rate']:.1%}"
            print(f"    {name:16s} n={r['n']:5d}  E-UNSAFE {r['n_unsafe']:5d} ({rate})")
        else:
            print(f"    {name:16s} n={r['n']:5d}")
    print(f"[step05] {len(dropped)} state(s) dropped with their x0 before the split (E failed)")
    print(f"[step05] manifest -> {paths.MANIFEST}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)
    j = sub.add_parser("judge", help="E judges every state (paid)")
    j.add_argument("--budget", type=float, default=None, help="USD ceiling (required)")
    j.add_argument("--limit", type=int, default=None, help="stop after N new judgments")
    j.add_argument("--max-consecutive-e-failures", type=int, default=MAX_CONSECUTIVE_E_FAILURES)
    j.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"parallel E calls (default {DEFAULT_WORKERS}); results are written in input "
        "order either way, and 1 is the serial run",
    )
    sub.add_parser("split", help="draw the five splits from the judged states (no API calls)")
    args = ap.parse_args(argv)
    if args.command == "judge" and args.workers < 1:
        ap.error("--workers must be >= 1")

    cfgs = load_config()
    assert_frozen(cfgs, only_sections=GUARD_SECTIONS)
    ecfg = judge.build_config(cfgs["models"])  # re-checks the prompt sha256 (II.9.1)
    paths.ensure_dirs()
    manifest = Manifest.load()
    try:
        manifest.pin_split_plan(S.split_plan(cfgs))
    except (SplitPlanChangedError, S.SplitError) as e:
        print(f"[step05] refusing to run: {e}", file=sys.stderr)
        return 5
    pre = manifest.prefilter or {}
    if not pre.get("e_identity"):
        print("[step05] refusing to run: no prefilter record in the manifest (step00)")
        return 2
    try:
        judge.check_same_e(pre["e_identity"], ecfg, where="the step00 prefilter")
    except judge.EChangedError as e:
        print(f"[step05] refusing to run: {e}", file=sys.stderr)
        return 5
    if args.command == "judge":
        return cmd_judge(args, cfgs, ecfg, manifest)
    return cmd_split(args, cfgs, ecfg, manifest)


if __name__ == "__main__":
    sys.exit(main())
