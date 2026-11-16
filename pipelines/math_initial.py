"""Math initial-state producer and split writer (M.2.3–M.2.4).

This is the sole math producer allowed to inspect the complete initial judgment file.
All post-split consumers go through the Core label-purpose gate.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from rafr.contracts.schemas import MathLabel, State
from rafr.core.seeding import component_seed
from rafr.data import labels as labels_module
from rafr.data.splits import make_splits, save_splits
from rafr.domains.math.generation import generate_initial
from rafr.domains.math.storage import IntegrityError, atomic_json, file_hash, read_json
from rafr.logging.writer import JsonlWriter, read_jsonl

DEFAULT_WORKERS = 4
"""Questions generated in parallel by the CLI (2026-10-04: 4, as detox step04)."""


def initial_paths(root: Path) -> tuple[Path, Path, Path]:
    return root / "states.jsonl", root / "initial_outcomes.jsonl", root / "initial_complete.json"


def require_initial(root: Path) -> dict:
    states, outcomes, marker = initial_paths(root)
    info = read_json(marker)
    for name, expected in info["files"].items():
        if file_hash(root / name) != expected:
            raise IntegrityError("initial-state files changed after completion")
    return info


def initial_e_records_for_revision(root: Path):
    """Copy initial E evidence without judging again or revealing split statistics.

    This is part of the initial-state producer's transfer boundary. Post-split
    analysis still reads labels exclusively through the Core purpose gate.
    """
    require_initial(root)
    judgments = labels_module.load_judgment_rows(labels_module.y_star_path(root / "labels"))
    records = {}
    for row in judgments.values():
        if row.record is None or row.label not in {MathLabel.CORRECT, MathLabel.INCORRECT}:
            raise IntegrityError("a reserve revision requires resolved initial E evidence")
        record = row.record
        if record.label != row.label or record.question_id != row.x0_id:
            raise IntegrityError("initial judgment and E evidence disagree")
        if record.text_id in records and records[record.text_id] != record:
            raise IntegrityError("conflicting initial E records")
        records[record.text_id] = record
    return list(records.values())


def _generate_one(pipeline, qid, question, max_attempts):
    """One question's G_fixed attempts and E calls, run in a worker thread; writes nothing.

    The attempts come back in the result and the E log rows in ``deferred``; generate_all
    writes both on the main thread, in input order.
    """
    deferred = []
    result = generate_initial(
        pipeline.client,
        question.public_payload(),
        lambda candidate: pipeline.grade_once(qid, candidate, deferred=deferred),
        master_seed=pipeline.config.seed,
        max_attempts=max_attempts,
    )
    return result, deferred


def generate_all(pipeline, *, workers=1) -> dict:
    """G_fixed + E for every question without an outcome row (M.2.3).

    Concurrency as detox step04 (2026-10-04): up to ``workers`` questions are in
    flight on a thread pool, each running its attempts in order with the seeds and requests
    a serial run sends. Results are consumed in input order (the head of the queue is waited
    for, whatever finishes first), and only then are the attempts, E rows, state, judgment
    and outcome written, so every file is what a serial run (``workers=1``) writes. The Core
    client, cache and ledger are shared between threads. If a question raises, nothing after
    it is written; the questions still in flight finish into the cache, so a resume does not
    buy them again.
    """
    root, config = pipeline.root, pipeline.config
    states_path, outcomes_path, marker = initial_paths(root)
    if marker.exists():
        return require_initial(root)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    existing = {row["question_id"]: row for row in read_jsonl(outcomes_path)}
    states = {row["x0_id"]: State.model_validate(row) for row in read_jsonl(states_path)}
    judgments = labels_module.load_judgment_rows(labels_module.y_star_path(root / "labels"))
    max_attempts = config.max_initial_attempts
    if max_attempts is None:
        raise IntegrityError("M.12: max_initial_attempts must be preregistered before generation")
    work = []
    for qid, question in pipeline.questions.items():
        if qid in existing:
            row = existing[qid]
            if row["status"] == "valid" and (
                qid not in states or states[qid].state_id not in judgments
            ):
                raise IntegrityError("completed initial row lacks its Core state or judgment")
            continue
        work.append((qid, question))
    queue: deque[tuple[str, object, Future]] = deque()
    source = iter(work)
    with (
        JsonlWriter(root / "generation.jsonl") as attempts,
        JsonlWriter(states_path) as state_writer,
        JsonlWriter(outcomes_path) as outcome_writer,
        JsonlWriter(labels_module.y_star_path(root / "labels")) as label_writer,
        ThreadPoolExecutor(workers) as pool,
    ):
        while True:
            while len(queue) < workers and (item := next(source, None)) is not None:
                qid, question = item
                future = pool.submit(_generate_one, pipeline, qid, question, max_attempts)
                queue.append((qid, question, future))
            if not queue:
                break
            qid, question, future = queue.popleft()
            result, deferred = future.result()
            attempts.write_many(result["attempts"])
            pipeline.commit_grades(deferred)
            if result["generation_status"] == "valid":
                state = result["state"]
                if not isinstance(state, State):
                    state = State.model_validate(state)
                if state.x0_id != qid or state.x0 != question.problem:
                    raise IntegrityError("generation returned a state bound to another question")
                record = pipeline.e_record(qid, state.text)
                if record.label not in {MathLabel.CORRECT, MathLabel.INCORRECT}:
                    raise IntegrityError("unresolved initial state cannot enter the split")
                if qid not in states:
                    state_writer.write(state)
                    states[qid] = state
                elif states[qid].model_dump() != state.model_dump():
                    # A resumed cached call may get a new timestamp, but not a new answer.
                    if states[qid].text != state.text or states[qid].state_id != state.state_id:
                        raise IntegrityError("resumed initial candidate differs")
                    state = states[qid]
                if state.state_id not in judgments:
                    judgment = labels_module.StateJudgment(
                        state_id=state.state_id,
                        x0_id=qid,
                        label=record.label,
                        record=record,
                        e_identity=pipeline.grader.configuration(),
                    )
                    label_writer.write(judgment)
                    judgments[state.state_id] = judgment
                row = {"question_id": qid, "status": "valid", "state_id": state.state_id}
            else:
                row = {
                    "question_id": qid,
                    "status": "excluded",
                    "reason": result.get("drop_reason", result.get("generation_status")),
                }
            outcome_writer.write(row)
            existing[qid] = row
    info = {
        "n_questions": len(pipeline.questions),
        "n_valid": len(states),
        "n_incorrect": sum(row.label is MathLabel.INCORRECT for row in judgments.values()),
        "n_correct": sum(row.label is MathLabel.CORRECT for row in judgments.values()),
        "n_excluded": sum(row["status"] == "excluded" for row in existing.values()),
        "files": {
            path.relative_to(root).as_posix(): file_hash(path)
            for path in (states_path, outcomes_path, labels_module.y_star_path(root / "labels"))
        },
    }
    atomic_json(marker, info)
    return info


def split_completed(pipeline) -> dict:
    root, config = pipeline.root, pipeline.config
    require_initial(root)
    states = {row["x0_id"]: State.model_validate(row) for row in read_jsonl(root / "states.jsonl")}
    judgments = labels_module.load_judgment_rows(labels_module.y_star_path(root / "labels"))
    by_question = {row.x0_id: row for row in judgments.values()}
    if set(states) != set(by_question):
        raise IntegrityError("initial states and completed y* coverage differ")
    for qid, state in states.items():
        row = by_question[qid]
        if row.state_id != state.state_id or row.label not in {
            MathLabel.CORRECT,
            MathLabel.INCORRECT,
        }:
            raise IntegrityError("all states must have valid initial labels before splitting")
    sizes = {
        "d_fit": config.fit_n,
        "d_gate": config.gate_n,
        "d_gate_reserve": config.reserve_n,
        "d_exp1": config.experiment_n,
        "unlabeled_pool": "rest",
    }
    splits = make_splits(
        states,
        lambda qid: by_question[qid].label is MathLabel.INCORRECT,
        sizes,
        component_seed(config.seed, "splits_v09"),
    )
    marker = root / "splits_complete.json"
    if marker.exists():
        info = read_json(marker)
        if info["sizes"] != sizes:
            raise IntegrityError("split configuration changed")
        for name, expected in info["files"].items():
            if file_hash(root / name) != expected:
                raise IntegrityError("split files changed")
        return info
    saved = save_splits(splits, root / "splits")
    info = {
        "procedure": "random_then_e_unsafe_v09",
        "seed": component_seed(config.seed, "splits_v09"),
        "sizes": sizes,
        "counts": {split.value: len(ids) for split, ids in splits.items()},
        "files": {path.relative_to(root).as_posix(): file_hash(path) for path in saved.values()},
    }
    atomic_json(marker, info)
    # Only D_fit counts are disclosed; the Core report never reads reserve labels.
    report = labels_module.split_report(
        splits, labels_dir=root / "labels", split_dir=root / "splits", domain="math"
    )
    atomic_json(root / "split_report.json", report)
    return info
