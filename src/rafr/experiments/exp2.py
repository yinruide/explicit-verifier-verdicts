"""Exp 2 — naturalistic sequential validation (0.2, II.8.3, II.9.3; decisions 4c, 9).

* x1 for every x0 is produced beforehand by G_fixed (same function as the state pool) —
  the runner receives the x1 states and never generates them;
* representation (A1 / B) is randomly assigned per x0 by a balanced permutation with a
  logged seed; no channel exists here (identity only, enforced);
* K = 4 controller decisions starting at t = 1; round-synchronous batching: all active
  trajectories are scored by V1/V2 as one batch, then decided, then the round is written to
  ``rounds/round_<t>/`` -- the trajectory checkpoint, that round's decision and score rows, its
  ledger rows and cost -- and only then marked complete (``COMPLETE.json``, the sha256 of every
  file). A run resumes from its latest COMPLETE round; a round that was cut off is discarded and
  redone whole, so no decision is ever logged twice (II.8.3). ``decisions.jsonl`` and
  ``scores.jsonl`` at the top of the run are assembled from the round directories;
* :func:`run_loop` is the part that runs on the rented pod (controller + in-process V1/V2,
  II.8.3); :func:`finish` -- E's round judgments and the finals -- runs on the laptop after the
  loop, from the round directories pulled back (E never takes part in the loop);
* parse_fail: x_{t+1} = x_t and the round is consumed (II.8.2);
* E never takes part in the loop. After the loop (decision, group 6d, 2026-09-16) E judges
  EVERY round text of EVERY trajectory in both arms — x_1..x_T plus the terminal text — and
  the judgments are written to ``round_judgments.jsonl`` (one row per (x0, arm, t)) and
  ``evaluations.jsonl``. The t = 1 texts are the D_Exp1 G_fixed states, so their judgments
  are step 5's y* (``prior_judgments``, II.8.3 v0.9) instead of being re-bought. These judgments
  feed c_t / R_t^(a) (II.10, II.1.1), decision accuracy, rewrite improvement rate and the
  "parse_fail as accept" sensitivity (analysis/);
* TFR = P(task_success(x_T) = False) over ALL randomized trajectories (II.9.3, C.10);
  non-convergence rate is reported separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rafr.channel import Channel, ObservationBuilder
from rafr.contracts.schemas import (
    K_MAX_ROUNDS,
    PARSE_FAIL_ALERT_THRESHOLD,
    Action,
    Arm,
    EvalRecord,
    Lambda,
    SafetyLabel,
    State,
    text_hash,
)
from rafr.controller import decide as ctrl
from rafr.core.seeding import py_rng
from rafr.experiments.common import PairScorer, make_log_record, parse_fail_cells, score_rows
from rafr.formatter import compute_fields, format_observation
from rafr.llm.client import LLMClient
from rafr.logging.schema import RecordKind, stream_filename
from rafr.logging.writer import JsonlWriter, read_jsonl

SEED_NAMESPACE = "exp2"
"""Controller seed namespace (decision, group 3); see ``exp1.SEED_NAMESPACE``."""


class Exp2Error(RuntimeError):
    pass


@dataclass
class Trajectory:
    x0_id: str
    x0: str
    arm: str
    t: int  # index of the current text x_t (1 = G_fixed output)
    x_t: str
    accepted: bool = False
    accepted_at: int | None = None
    n_rewrites: int = 0
    n_parse_fail: int = 0
    refusal_like_rewrites: int = 0
    # does the CURRENT text carry the refusal_like flag? It travels with x_t: set by the
    # rewrite that produced it, unchanged by accept or parse_fail (x_1 comes from G_fixed,
    # whose refusals are invalid outputs, so it is False there). At the end of the loop this
    # is the flag of x_T, which is what the II.8.2 sensitivity analysis conditions on.
    x_t_refusal_like: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def state_id(self) -> str:
        return f"{self.x0_id}-{self.arm}-t{self.t}"


@dataclass
class Exp2Setup:
    run_id: str
    run_dir: Path
    x1_states: Sequence[State]  # one per x0, from G_fixed (decision 4c)
    scorer: PairScorer
    tau1: float
    tau2: float
    channel: Channel  # must be identity (II.1.3)
    arms: Sequence[Arm]
    k: int
    assignment_seed: int
    run_seed: int | None
    judge_fn: Callable[[str], EvalRecord]  # E, text only; called after the loop
    experiment_name: str = "exp2"
    # The controller's seed namespace (and so its cache key), II.8.2. The formal run keeps
    # ``SEED_NAMESPACE``; the step 10a smoke passes its own id (II.4.4, 2026-09-25) so that no
    # smoke request can share a seed or a cached answer with Exp 1 or Exp 2.
    seed_namespace: str = SEED_NAMESPACE
    # text_id -> EvalRecord already bought elsewhere (step 5's y* of D_Exp1 covers t = 1)
    prior_judgments: Mapping[str, EvalRecord] | None = None
    # quality(x0, x_T) (I.6): semantic preservation of the terminal text, recorded on the
    # finals as ``semantic_score`` for the descriptive "low preservation among SAFE" report
    # (2026-09-18). None -> not measured (semantic.py is a later round); the finals
    # then carry ``semantic_score: null`` and the report shows counts only.
    quality_fn: Callable[[str, str], float | None] | None = None


def balanced_assignment(x0_ids: Sequence[str], arms: Sequence[Arm], seed: int) -> dict[str, Arm]:
    """Balanced permutation (decision 9): arm sizes differ by at most one."""
    ids = list(x0_ids)
    rng = py_rng(seed, "exp2_assignment")
    rng.shuffle(ids)
    return {x: arms[i % len(arms)] for i, x in enumerate(ids)}


ROUNDS_DIR = "rounds"
ROUND_COMPLETE = "COMPLETE.json"
ROUND_TRAJECTORIES = "trajectories.jsonl"
LOOP_COMPLETE = "loop_complete.json"


class LoopNotFinishedError(Exp2Error):
    """The loop has not written its last round yet; E's judgments wait for it (II.8.3)."""


def round_dir(rd: Path, t: int) -> Path:
    return Path(rd) / ROUNDS_DIR / f"round_{t}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonl(path: Path, rows: Sequence[Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            row = r.model_dump(mode="json") if hasattr(r, "model_dump") else r
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_round(
    rd: Path,
    t: int,
    trajs: Sequence[Trajectory],
    decisions: Sequence[Any],
    scores: Sequence[Any],
    ledger_rows: Sequence[Mapping[str, Any]] = (),
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write round ``t`` into a temporary directory, mark it COMPLETE (sha256 of every file),
    then move it into place: a round directory without COMPLETE.json never exists."""
    final = round_dir(rd, t)
    tmp = final.with_name(final.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    _jsonl(tmp / ROUND_TRAJECTORIES, [asdict(tr) for tr in trajs])
    _jsonl(tmp / stream_filename(RecordKind.DECISION), decisions)
    _jsonl(tmp / stream_filename(RecordKind.SCORE), scores)
    _jsonl(tmp / stream_filename(RecordKind.LEDGER), list(ledger_rows))
    names = [ROUND_TRAJECTORIES, *(stream_filename(k) for k in (
        RecordKind.DECISION, RecordKind.SCORE, RecordKind.LEDGER))]  # fmt: skip
    (tmp / ROUND_COMPLETE).write_text(
        json.dumps(
            {
                "t": t,
                "files": {n: _sha256(tmp / n) for n in names},
                "n_trajectories": len(trajs),
                "n_decisions": len(decisions),
                **dict(meta or {}),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
        newline="\n",
    )
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    return final


def verify_round(d: Path) -> dict[str, Any] | None:
    """The COMPLETE record of a round directory if every file matches its sha256, else None."""
    marker = Path(d) / ROUND_COMPLETE
    if not marker.is_file():
        return None
    info = json.loads(marker.read_text(encoding="utf-8"))
    for name, digest in (info.get("files") or {}).items():
        f = Path(d) / name
        if not f.is_file() or _sha256(f) != digest:
            return None
    return info


def complete_rounds(rd: Path, k: int) -> list[int]:
    """Rounds 1..t that are all COMPLETE and verified, in order (a gap ends the run of them)."""
    done: list[int] = []
    for t in range(1, k + 1):
        if verify_round(round_dir(rd, t)) is None:
            break
        done.append(t)
    return done


def load_round(rd: Path, t: int) -> list[Trajectory]:
    return [Trajectory(**r) for r in read_jsonl(round_dir(rd, t) / ROUND_TRAJECTORIES)]


def discard_incomplete(rd: Path, k: int) -> list[str]:
    """Remove every round directory after the last COMPLETE one (and any ``.partial``)."""
    keep = set(complete_rounds(rd, k))
    removed: list[str] = []
    base = Path(rd) / ROUNDS_DIR
    if not base.is_dir():
        return removed
    for d in sorted(base.iterdir()):
        name = d.name
        t = name.removeprefix("round_").removesuffix(".partial")
        if name.endswith(".partial") or (t.isdigit() and int(t) not in keep):
            shutil.rmtree(d) if d.is_dir() else d.unlink()
            removed.append(name)
    return removed


def assemble_streams(rd: Path, k: int) -> None:
    """``decisions.jsonl`` / ``scores.jsonl`` of the run = the rounds' rows, in round order."""
    for kind in (RecordKind.DECISION, RecordKind.SCORE):
        rows = [
            r
            for t in complete_rounds(rd, k)
            for r in read_jsonl(round_dir(rd, t) / stream_filename(kind))
        ]
        _jsonl(Path(rd) / stream_filename(kind), rows)


def _ensure_assignment(setup: Exp2Setup, rd: Path) -> dict[str, Arm]:
    assign_path = rd / "assignment.jsonl"
    x0_ids = [s.x0_id for s in setup.x1_states]
    if assign_path.exists():
        assignment = {r["x0_id"]: Arm(r["arm"]) for r in read_jsonl(assign_path)}
        if set(assignment) != set(x0_ids):
            raise Exp2Error(f"{assign_path} does not cover exactly this run's x0")
        return assignment
    assignment = balanced_assignment(x0_ids, setup.arms, setup.assignment_seed)
    with assign_path.open("w", encoding="utf-8", newline="\n") as fh:
        for x, a in assignment.items():
            fh.write(json.dumps({"x0_id": x, "arm": a.value, "seed": setup.assignment_seed}) + "\n")
    return assignment


RoundHook = Callable[[int, dict[str, Any]], None]


def run_loop(
    client: LLMClient,
    ccfg: ctrl.ControllerConfig,
    setup: Exp2Setup,
    *,
    on_round: RoundHook | None = None,
) -> list[Trajectory]:
    """The loop alone (II.8.3): controller + in-process V1/V2, one COMPLETE directory per round.

    Resumes from the latest COMPLETE round; anything after it is discarded and redone. After
    round K the run's decision / score streams are assembled and ``loop_complete.json`` written.
    ``on_round(t, info)`` is called after each round is on disk (the pod prints the running cost
    there). No E call is made here.
    """
    if getattr(setup.channel, "name", None) != "identity":
        raise Exp2Error("Exp 2 has no channel (II.1.3): only the identity channel is allowed")
    if setup.k != K_MAX_ROUNDS:
        raise Exp2Error(f"K is locked to {K_MAX_ROUNDS} (II.8.3)")
    ccfg = ctrl.with_round_budget(ccfg, K_MAX_ROUNDS)  # the {k} the controller is told about
    rd = setup.run_dir
    rd.mkdir(parents=True, exist_ok=True)
    assignment = _ensure_assignment(setup, rd)
    discard_incomplete(rd, setup.k)
    done = complete_rounds(rd, setup.k)
    if done:
        trajs = load_round(rd, done[-1])
        start_t = done[-1] + 1
    else:
        trajs = [
            Trajectory(x0_id=s.x0_id, x0=s.x0, arm=assignment[s.x0_id].value, t=1, x_t=s.text)
            for s in setup.x1_states
        ]
        start_t = 1

    builder = ObservationBuilder(
        setup.channel, setup.tau1, setup.tau2, setup.scorer.v1_id, setup.scorer.v2_id
    )
    ledger = getattr(client, "ledger", None)
    for t in range(start_t, setup.k + 1):
        n_ledger = len(ledger.rows()) if ledger is not None else 0
        decisions: list[Any] = []
        scores: list[Any] = []
        active = [tr for tr in trajs if not tr.accepted]
        if active:
            # round-synchronous scoring of every active text as one batch
            pairs = setup.scorer.score_pair([tr.x_t for tr in active])
            scores = score_rows(setup.scorer, [tr.state_id for tr in active], pairs)
            for tr, (s1, s2) in zip(active, pairs, strict=True):
                obs = builder.build(tr.state_id, s1, s2, Lambda.NONE)
                fields = compute_fields(obs)
                arm = Arm(tr.arm)
                observation = format_observation(arm, obs)
                decision = ctrl.decide(
                    client,
                    ccfg,
                    x0=tr.x0,
                    x_t=tr.x_t,
                    observation=observation,
                    # II.8.2: the seed is keyed by (experiment, state, t, attempt) with no arm
                    # and no lambda; the state key is the x0 id, not the round id
                    # {x0_id}-{arm}-t{t}, which carries the arm (D-2026-09-25g). The log keeps
                    # the round id (make_log_record below).
                    labels=ctrl.DecisionLabels(
                        state_id=tr.x0_id, arm=arm, lam=Lambda.NONE, t=t,
                        experiment=setup.seed_namespace,
                    ),
                )  # fmt: skip
                decisions.append(
                    make_log_record(
                        run_id=setup.run_id,
                        experiment=setup.experiment_name,
                        seed=setup.run_seed,
                        arm=arm,
                        lam=Lambda.NONE,
                        t=t,
                        state_id=tr.state_id,
                        x0_id=tr.x0_id,
                        x_t=tr.x_t,
                        obs=obs,
                        fields=fields,
                        decision=decision,
                    )
                )
                tr.history.append(
                    {
                        "t": t,
                        "x_t": tr.x_t,
                        "action": decision.action.value if decision.action else None,
                        "parse_fail": decision.parse_fail,
                        "refusal_like": decision.refusal_like,
                        "empty_rewrite": decision.empty_rewrite,
                    }
                )
                if decision.parse_fail:
                    tr.n_parse_fail += 1
                    tr.t = t + 1  # x_{t+1} = x_t; the round is consumed (II.8.2)
                elif decision.action is Action.ACCEPT:
                    tr.accepted = True
                    tr.accepted_at = t
                else:
                    assert decision.revised_text is not None
                    tr.n_rewrites += 1
                    tr.refusal_like_rewrites += int(decision.refusal_like)
                    tr.x_t_refusal_like = bool(decision.refusal_like)
                    tr.x_t = decision.revised_text
                    tr.t = t + 1
        ledger_rows = ledger.rows()[n_ledger:] if ledger is not None else []
        info = {"n_active": len(active), "n_accepted_total": sum(tr.accepted for tr in trajs)}
        write_round(rd, t, trajs, decisions, scores, ledger_rows, info)
        if on_round is not None:
            on_round(t, {**info, "n_decisions": len(decisions), "ledger_rows": ledger_rows})
    assemble_streams(rd, setup.k)
    (rd / LOOP_COMPLETE).write_text(
        json.dumps(
            {"k": setup.k, "n_trajectories": len(trajs), "rounds": complete_rounds(rd, setup.k)}
        ),
        encoding="utf-8",
        newline="\n",
    )
    return trajs


def final_trajectories(rd: Path, k: int) -> list[Trajectory]:
    """The trajectories after round K, from a finished loop (on the laptop, after the pull)."""
    done = complete_rounds(rd, k)
    if done != list(range(1, k + 1)) or not (Path(rd) / LOOP_COMPLETE).is_file():
        raise LoopNotFinishedError(
            f"{rd}: complete rounds {done}, loop_complete {(Path(rd) / LOOP_COMPLETE).is_file()}"
        )
    return load_round(rd, k)


class JudgeNotCompleteError(Exp2Error):
    """E's post-loop judgments of this run are missing, unfinished or changed since."""


def finish(setup: Exp2Setup, trajs: Sequence[Trajectory] | None = None) -> dict[str, Any]:
    """After the loop (laptop): E judges every round text, the finals and the summary.

    II.9.1: if E gives no verdict on a text after its retries (:class:`rafr.evaluator.judge.
    EvaluatorError`), the whole pass stops and no round or terminal judgment is written:
    ``round_judgments.jsonl`` and ``trajectories.jsonl`` are written as ``*.partial`` and
    renamed only once every text is judged. ``judge_complete.json`` is removed when a pass
    starts and written when it ends, with both files' sha256; ``report`` requires it
    (:func:`require_judge_complete`). ``evaluations.jsonl`` keeps every judgment bought, so a
    rerun after a failure buys nothing twice."""
    rd = setup.run_dir
    (rd / JUDGE_COMPLETE).unlink(missing_ok=True)
    trajs = list(trajs) if trajs is not None else final_trajectories(rd, setup.k)
    finals = judge_rounds(setup, trajs, rd)
    _write_jsonl_atomic(rd / TRAJECTORIES, finals)
    summary = summarize_exp2(finals, list(read_jsonl(rd / stream_filename(RecordKind.DECISION))))
    (rd / JUDGE_COMPLETE).write_text(
        json.dumps(
            {
                "n_trajectories": len(finals),
                "round_judgments_sha256": _file_sha256(rd / ROUND_JUDGMENTS),
                "trajectories_sha256": _file_sha256(rd / TRAJECTORIES),
                "completed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            },
            indent=1,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return summary


def require_judge_complete(rd: Path) -> dict[str, Any]:
    """The judge pass of this run finished and its two files are the ones it wrote."""
    rd = Path(rd)
    marker = rd / JUDGE_COMPLETE
    if not marker.is_file():
        raise JudgeNotCompleteError(f"no {JUDGE_COMPLETE}: E's judgments are missing or unfinished")
    info = json.loads(marker.read_text(encoding="utf-8"))
    for name, key in (
        (ROUND_JUDGMENTS, "round_judgments_sha256"),
        (TRAJECTORIES, "trajectories_sha256"),
    ):
        p = rd / name
        if not p.is_file():
            raise JudgeNotCompleteError(f"{name} missing although {JUDGE_COMPLETE} exists")
        if _file_sha256(p) != info.get(key):
            raise JudgeNotCompleteError(f"{name} changed since the judge pass that wrote it")
    return info


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    tmp = path.with_name(path.name + PARTIAL_SUFFIX)
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def run_exp2(client: LLMClient, ccfg: ctrl.ControllerConfig, setup: Exp2Setup) -> dict[str, Any]:
    """Loop and post-loop judgments in one process (the dry run and the tests)."""
    return finish(setup, run_loop(client, ccfg, setup))


ROUND_JUDGMENTS = "round_judgments.jsonl"
TRAJECTORIES = "trajectories.jsonl"
JUDGE_COMPLETE = "judge_complete.json"
PARTIAL_SUFFIX = ".partial"


def round_texts(tr: Trajectory, k: int) -> list[dict[str, Any]]:
    """The texts of one trajectory that E must judge: x_t of every decided round and the
    terminal text. If the trajectory was accepted, the terminal text IS the accepted round's
    text; otherwise it is x_{K+1}, the last rewrite, which no round decided on."""
    rows = [
        {"t": int(h["t"]), "x_t": h["x_t"], "terminal": tr.accepted and h["t"] == tr.accepted_at}
        for h in tr.history
    ]
    if not tr.accepted:
        rows.append({"t": k + 1, "x_t": tr.x_t, "terminal": True})
    return rows


def judge_rounds(setup: Exp2Setup, trajs: Sequence[Trajectory], rd: Path) -> list[dict[str, Any]]:
    """Judge every round text once (deduplicated by text), reusing prior judgments, and write
    ``evaluations.jsonl`` (every EvalRecord used, appended as it is obtained) +
    ``round_judgments.jsonl`` (written as ``.partial``, renamed once every text is judged; on
    any failure the partial file is removed and the error passes through, II.9.1)."""
    eval_path = rd / stream_filename(RecordKind.EVALUATION)
    # judgments bought elsewhere (step 5's y* of the D_Exp1 states covers t = 1) ...
    external = dict(setup.prior_judgments or {})
    # ... and judgments this run already bought before being interrupted: appending only the
    # new ones keeps evaluations.jsonl duplicate-free across a resumed post-loop pass.
    on_disk = {r["text_id"]: EvalRecord.model_validate(r) for r in read_jsonl(eval_path)}
    used: dict[str, EvalRecord] = {}
    source: dict[str, str] = {}
    elog = JsonlWriter(eval_path)

    def judge(text: str) -> EvalRecord:
        tid = text_hash(text)
        if tid not in used:
            if tid in external:
                used[tid], source[tid] = external[tid], "reused"
            elif tid in on_disk:
                used[tid], source[tid] = on_disk[tid], "exp2"
            else:
                used[tid], source[tid] = setup.judge_fn(text), "exp2"
            if tid not in on_disk:  # write as soon as it is bought: a crash loses nothing
                elog.write(used[tid])
                on_disk[tid] = used[tid]
        return used[tid]

    finals: list[dict[str, Any]] = []
    partial = rd / (ROUND_JUDGMENTS + PARTIAL_SUFFIX)
    try:
        finals = _judge_into(setup, trajs, judge, source, elog, partial)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, rd / ROUND_JUDGMENTS)
    return finals


def _judge_into(
    setup: Exp2Setup,
    trajs: Sequence[Trajectory],
    judge: Callable[[str], EvalRecord],
    source: Mapping[str, str],
    elog: JsonlWriter,
    partial: Path,
) -> list[dict[str, Any]]:
    finals: list[dict[str, Any]] = []
    with elog, partial.open("w", encoding="utf-8", newline="\n") as fh:
        for tr in trajs:
            labels_by_t: dict[str, str] = {}
            terminal: EvalRecord | None = None
            for r in round_texts(tr, setup.k):
                rec = judge(r["x_t"])
                tid = text_hash(r["x_t"])
                labels_by_t[str(r["t"])] = rec.label.value
                if r["terminal"]:
                    terminal = rec
                fh.write(
                    json.dumps(
                        {
                            "x0_id": tr.x0_id,
                            "arm": tr.arm,
                            "t": r["t"],
                            "state_id": f"{tr.x0_id}-{tr.arm}-t{r['t']}",
                            "text_id": tid,
                            "e_label": rec.label.value,
                            "e_score": rec.score,
                            "terminal": bool(r["terminal"]),
                            "source": source[tid],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            assert terminal is not None
            first_pf = next((int(h["t"]) for h in tr.history if h["parse_fail"]), None)
            finals.append(
                {
                    "x0_id": tr.x0_id,
                    "x0": tr.x0,
                    "arm": tr.arm,
                    "x_T": tr.x_t,
                    # quality(x0, x_T); None when no quality function is configured
                    "semantic_score": (
                        setup.quality_fn(tr.x0, tr.x_t) if setup.quality_fn is not None else None
                    ),
                    "accepted": tr.accepted,
                    "accepted_at": tr.accepted_at,
                    "n_rewrites": tr.n_rewrites,
                    "n_parse_fail": tr.n_parse_fail,
                    "first_parse_fail_t": first_pf,
                    "refusal_like_rewrites": tr.refusal_like_rewrites,  # descriptive count
                    # II.8.2's sensitivity analysis conditions on this one, not on the count
                    "x_T_refusal_like": tr.x_t_refusal_like,
                    "e_label": terminal.label.value,
                    "e_score": terminal.score,
                    "task_success": terminal.label is SafetyLabel.SAFE,
                    "round_e_labels": labels_by_t,  # {t: label}; terminal t = K+1 if not accepted
                }
            )
    return finals


def summarize_exp2(
    finals: list[dict[str, Any]], decision_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Quick run report (preregistered analysis: analysis/exp2_tfr.py)."""
    per_arm: dict[str, dict[str, Any]] = {}
    for f in finals:
        a = per_arm.setdefault(
            f["arm"],
            {"n": 0, "failures": 0, "non_converged": 0, "rewrites": [], "refusal_like_failures": 0},
        )
        a["n"] += 1
        a["failures"] += int(not f["task_success"])
        a["non_converged"] += int(not f["accepted"])
        a["rewrites"].append(f["n_rewrites"])
        # sensitivity (II.8.2): a trajectory whose x_T carries the refusal_like flag counts as
        # a terminal failure. A refusal-looking rewrite in the middle of a trajectory whose
        # final output is fine does NOT count.
        a["refusal_like_failures"] += int(
            (not f["task_success"]) or bool(f.get("x_T_refusal_like"))
        )
    out: dict[str, Any] = {"arms": {}}
    for arm, a in per_arm.items():
        rw = sorted(a["rewrites"])
        out["arms"][arm] = {
            "n": a["n"],
            "TFR": a["failures"] / a["n"] if a["n"] else None,
            "TFR_refusal_like_as_failure": a["refusal_like_failures"] / a["n"] if a["n"] else None,
            "non_convergence_rate": a["non_converged"] / a["n"] if a["n"] else None,
            "rewrite_rounds_median": rw[len(rw) // 2] if rw else None,
        }
    # II.8.2 (v0.8): Exp 2 reports parse_fail by arm x ROUND, not by arm x lambda
    out["parse_fail"] = parse_fail_cells(decision_rows, PARSE_FAIL_ALERT_THRESHOLD, by="t")
    out["n_trajectories"] = len(finals)
    return out


__all__ = [
    "JUDGE_COMPLETE",
    "ROUND_JUDGMENTS",
    "TRAJECTORIES",
    "Exp2Error",
    "Exp2Setup",
    "JudgeNotCompleteError",
    "LoopNotFinishedError",
    "Trajectory",
    "balanced_assignment",
    "complete_rounds",
    "final_trajectories",
    "finish",
    "judge_rounds",
    "require_judge_complete",
    "round_texts",
    "run_exp2",
    "run_loop",
    "summarize_exp2",
    "verify_round",
]


def load_assignment(path: Path) -> Mapping[str, str]:
    return {r["x0_id"]: r["arm"] for r in read_jsonl(path)}
