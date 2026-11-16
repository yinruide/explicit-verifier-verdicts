"""Explicit reserve redesign transfer with the original Core gate-attempt history.

Only immutable cohort evidence is copied. Scores, fitting, diagnostics and controller
work start fresh, while every descendant shares the original two-attempt limit.
The CLI serializes transfers and gate writes with its existing general writer mutex.
"""

from __future__ import annotations

import os
from pathlib import Path

from pipelines.math_initial import initial_e_records_for_revision, require_initial
from rafr.contracts.schemas import Split
from rafr.domains.math.storage import IntegrityError, atomic_json, digest, file_hash, read_json
from rafr.logging.writer import JsonlWriter
from rafr.validity.gate import next_attempt_split, read_attempts


def gate_history_path(root: Path) -> Path:
    root = Path(root).resolve()
    snapshot = root / "snapshot.json"
    value = read_json(snapshot).get("math_gate_history_path") if snapshot.exists() else None
    if value is not None and (not isinstance(value, str) or not Path(value).is_absolute()):
        raise IntegrityError("shared gate history must have a canonical absolute path")
    return Path(value).resolve() if value else root / "gate_attempts.json"


def writer_lock_root(root: Path, prior_root: Path | None = None) -> Path:
    """Use the same general writer mutex for every run sharing Core gate history."""
    root = Path(root).resolve()
    current = gate_history_path(root)
    if prior_root is None:
        return current.parent
    inherited = gate_history_path(Path(prior_root))
    snapshot = root / "snapshot.json"
    if snapshot.exists() and read_json(snapshot).get("math_gate_history_path"):
        if current != inherited:
            raise IntegrityError("revision target is bound to another gate history")
    return inherited.parent


def _relative_file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise IntegrityError("cohort evidence contains an absent or escaping file")
    return path


def _copy_once(source: Path, destination: Path):
    """Resume exact copies; never overwrite different evidence."""
    expected = file_hash(source)
    if destination.exists():
        if file_hash(destination) != expected:
            raise IntegrityError(f"revision target already contains different {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".revision-copy.partial")
    try:
        with temporary.open("wb") as handle:
            handle.write(source.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def initialize_revision(pipeline, prior_root: Path, decision: dict):
    """Initialize one reviewed redesign; no new G_fixed, E, V or controller calls.

    The new run must have the same data/E identity and generation/split settings.
    Its verifier and channel choices may differ. A completed transfer is safe to
    repeat; a partial transfer is resumed only when every existing byte agrees.
    """
    root, prior_root = pipeline.root.resolve(), Path(prior_root).resolve()
    if root == prior_root or root in prior_root.parents or prior_root in root.parents:
        raise IntegrityError("reserve revisions require separate, non-nested run directories")
    if not isinstance(decision, dict) or any(
        not isinstance(decision.get(key), str) or not decision[key].strip()
        for key in ("reviewed_by", "decision")
    ):
        raise IntegrityError(
            "the authors must provide an explicit reviewed reserve redesign decision"
        )

    prior_snapshot = read_json(prior_root / "snapshot.json")
    target_snapshot = read_json(root / "snapshot.json")
    if prior_snapshot.get("math_identity") != target_snapshot.get("math_identity"):
        raise IntegrityError(
            "reserve redesign cannot change the cohort, E, stage or execution mode"
        )
    inherited = prior_snapshot.get("math_phase_bindings", {})
    for phase in ("generation", "splits"):
        if inherited.get(phase) != pipeline._phase_values(phase):
            raise IntegrityError(f"reserve redesign cannot change frozen {phase} settings")
    target_bindings = target_snapshot.get("math_phase_bindings", {})
    if set(target_bindings) - {"generation", "splits"}:
        raise IntegrityError("reserve revision target has already consumed scientific settings")

    history_path = gate_history_path(prior_root)
    attempts = read_attempts(history_path)
    if len(attempts) != 1 or attempts[0].get("passed") is not False:
        raise IntegrityError("reserve redesign requires exactly one failed Core gate attempt")
    if next_attempt_split(history_path) is not Split.D_GATE_RESERVE:
        raise IntegrityError("Core gate history does not permit the reserve attempt")
    gate_path = prior_root / "gate_1.json"
    gate_hash = file_hash(gate_path)
    if prior_snapshot.get("math_artifact_hashes", {}).get("gate_1.json") != gate_hash:
        raise IntegrityError("first failed gate evidence is not bound to the prior snapshot")
    gate_report = read_json(gate_path)
    if (
        gate_report.get("passed") is not False
        or gate_report.get("attempt") != 1
        or gate_report.get("split") != Split.D_GATE.value
    ):
        raise IntegrityError("prior gate report is not the failed first D_gate attempt")
    if attempts[0].get("gate_report_sha256", gate_hash) != gate_hash:
        raise IntegrityError("shared Core history disagrees with first gate evidence")

    initial = require_initial(prior_root)
    splits = read_json(prior_root / "splits_complete.json")
    cohort_files = {**initial["files"], **splits["files"]}
    for name, expected in cohort_files.items():
        if file_hash(_relative_file(prior_root, name)) != expected:
            raise IntegrityError("prior cohort or split evidence changed")
    for name in ("initial_complete.json", "splits_complete.json"):
        cohort_files[name] = file_hash(prior_root / name)
    records = initial_e_records_for_revision(prior_root)
    record_bytes = {record.text_id: record.model_dump(mode="json") for record in records}
    marker = {
        "prior_root": str(prior_root),
        "history_path": str(history_path),
        "decision": decision,
        "first_attempt_sha256": digest(attempts[0]),
        "first_gate_sha256": gate_hash,
        "cohort_files": cohort_files,
        "cohort_sha256": digest(cohort_files),
        "initial_e_records_sha256": digest(record_bytes),
        "n_initial_e_records": len(records),
        "new_model_calls": 0,
    }
    expected_files = set(cohort_files) | {
        "snapshot.json",
        "reserve_revision.json",
        "gate_1.json",
        "evaluations.jsonl",
        "reserve_revision_complete.json",
    }
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        if name.startswith("configs/"):
            continue
        if name.endswith(".revision-copy.partial"):
            name = name.removesuffix(".revision-copy.partial")
        if name not in expected_files:
            raise IntegrityError(f"reserve revision target is not fresh: {name}")
    if target_snapshot.get("math_gate_history_path", str(history_path)) != str(history_path):
        raise IntegrityError("target is bound to another Core gate history")
    pipeline._save_fixed(root / "reserve_revision.json", marker)
    target_snapshot = read_json(root / "snapshot.json")
    target_snapshot["math_gate_history_path"] = str(history_path)
    target_snapshot["math_phase_bindings"] = {
        phase: inherited[phase] for phase in ("generation", "splits")
    }
    atomic_json(root / "snapshot.json", target_snapshot)

    for name in cohort_files:
        _copy_once(prior_root / name, root / name)
    _copy_once(gate_path, root / "gate_1.json")
    pipeline._save_fixed(root / "gate_1.json", gate_report)

    # Export only prior y* records, never fresh E decisions or downstream judgments.
    evaluation_path = root / "evaluations.jsonl"
    temporary = evaluation_path.with_name(evaluation_path.name + ".revision-copy.partial")
    temporary.unlink(missing_ok=True)
    try:
        with JsonlWriter(temporary) as writer:
            writer.write_many(records)
        if evaluation_path.exists():
            if file_hash(temporary) != file_hash(evaluation_path):
                raise IntegrityError("revision target has different initial E cache evidence")
        else:
            temporary.replace(evaluation_path)
    finally:
        temporary.unlink(missing_ok=True)
    pipeline._evaluations = {record.text_id: record for record in records}
    pipeline._save_fixed(
        root / "reserve_revision_complete.json",
        {
            "revision_sha256": file_hash(root / "reserve_revision.json"),
            "cohort_sha256": marker["cohort_sha256"],
            "initial_e_cache_sha256": file_hash(evaluation_path),
            "n_initial_e_records": len(records),
        },
    )
    return {
        "status": "RESERVE_REVISION_INITIALIZED",
        "prior_root": str(prior_root),
        "run_dir": str(root),
        "history_path": str(history_path),
        "next_gate_attempt": 2,
        "cohort_sha256": marker["cohort_sha256"],
        "new_model_calls": 0,
    }
