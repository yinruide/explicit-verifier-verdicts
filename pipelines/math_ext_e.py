"""Extension-owned Math-Verify context; never constructs the original MathPipeline.

Private references live only in this local E boundary and Core EvalRecord stream. The
ordinary extension loop calls this context only after all rounds; correction ceiling
uses the same callable but owns its separate rule for stopping on nonbinary E output.
No language-model provider, network client, model download or original-run writer exists here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

from pipelines.math_ext_data import validate_output
from rafr.contracts.schemas import EvalRecord, MathLabel, utc_now_iso
from rafr.domains.math.grading import GradeResult, MathGrader
from rafr.domains.math.storage import IntegrityError, atomic_json, digest, file_hash, read_json
from rafr.logging.writer import JsonlWriter, read_jsonl

WALL_CLOCK_TIMEOUT = "worker_wall_clock_deadline"
TIMEOUT_UNRESOLVED = f"e_timeout_after_retries:{WALL_CLOCK_TIMEOUT}"
TERMINAL = {"correct", "incorrect", "unresolved"}
MAX_ATTEMPTS = 3


def check_e_callback(callback, *, run_dir, source_root, e_identity, live=False):
    """Reject a source-writing evaluator before any grading side effect.

    Plain functions remain an explicit offline test seam. Stateful evaluator contexts
    must own this extension's output and agree with the identity recorded by the caller.
    This check does not sandbox arbitrary user code or authorize a real experiment.
    """
    while isinstance(callback, partial):
        callback = callback.func
    owner = getattr(callback, "__self__", None)
    if owner is None and callable(callback):
        owner = callback
    if any(cls.__module__ == "pipelines.math_pipeline" for cls in type(owner).__mro__):
        raise IntegrityError("extension grader must not bind the original MathPipeline/source")
    root = getattr(owner, "root", None)
    if root is not None:
        if source_root and Path(root).resolve() == Path(source_root).resolve():
            raise IntegrityError("extension grader must not bind the original MathPipeline/source")
        target = validate_output(
            Path(root),
            source_root=Path(source_root) if source_root else None,
            allow_existing=True,
        )
        if target != Path(run_dir).resolve():
            raise IntegrityError("E context output must belong to this extension run")
    actual = getattr(owner, "identity", None)
    if actual is not None and actual != e_identity:
        raise IntegrityError("E callback identity differs from the recorded E identity")
    if live:
        from rafr.domains.math.grading import MathVerifyAdapter

        if type(owner) is not ExtensionMathEvaluator or callback is not owner:
            raise IntegrityError("live E requires the extension-owned Math-Verify context")
        grader = owner.grader
        if (
            type(grader) is not MathGrader
            or type(grader.backend) is not MathVerifyAdapter
            or grader.backend != MathVerifyAdapter()
            or grader.timeout_seconds != 5.0
            or grader.require_boxed_prediction is not True
            or grader.configuration().get("backend_identity") != "math-verify"
        ):
            raise IntegrityError("live E differs from the frozen Math-Verify configuration")
        owner._assert_identity()


class ExtensionMathEvaluator:
    """A resumable, question/reference/text/config-bound local evaluator.

    ``source_identity`` is the caller's independently bound source provenance. Inputs are
    copied at construction; no reference is read back from a controller/pod artifact.
    Original x1 labels must be reused by the runner, rather than sent here for rejudging.

    Every completed attempt is written before its evaluation record. On interruption
    between these writes, reopening reconstructs the missing Core record from the attempt
    without running Math-Verify again. An ordinary exception propagates immediately and
    blocks automatic continuation for that candidate; returned backend errors receive the
    original three-attempt budget, which persists across process restarts.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        questions: Mapping[str, str],
        private_references: Mapping[str, str],
        source_identity: Mapping[str, Any],
        source_root: Path,
        grader: MathGrader | None = None,
    ) -> None:
        self.root = validate_output(run_dir, source_root=source_root, allow_existing=True)
        self.questions = dict(questions)
        self._references = dict(private_references)
        if not self.questions or set(self.questions) != set(self._references):
            raise IntegrityError("E requires exactly one private reference per question")
        if any(
            not isinstance(v, str) or not v.strip()
            for v in [*self.questions.keys(), *self.questions.values(), *self._references.values()]
        ):
            raise IntegrityError(
                "E question IDs, questions and references must be nonempty strings"
            )
        if not source_identity:
            raise IntegrityError("E requires bound source provenance")
        self.grader = grader if grader is not None else MathGrader()
        self.settings = self.grader.configuration()
        self.identity = {
            "schema": "math-extension-e-v1",
            "source_root": str(Path(source_root).resolve()),
            "source_identity": dict(source_identity),
            "private_inputs_sha256": digest([self.questions, self._references]),
            "grader": self.settings,
            "retry_policy": "three_attempts_wall_clock_timeout_unresolved",
        }
        # Defensive deep copy also validates serializability and forbids nonfinite settings.
        self._identity_digest = digest(self.identity)
        self.identity_path = self.root / "extension_e_identity.json"
        self.attempts_path = self.root / "extension_e_attempts.jsonl"
        self.records_path = self.root / "extension_e_evaluations.jsonl"
        self._check_children()
        if self.identity_path.exists():
            if read_json(self.identity_path) != self.identity:
                raise IntegrityError("extension E identity changed; use a separate run")
        elif self.attempts_path.exists() or self.records_path.exists():
            raise IntegrityError("extension E records have no bound identity")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            atomic_json(self.identity_path, self.identity)
        self._attempts: dict[str, list[dict[str, Any]]] = {}
        self._records: dict[str, EvalRecord] = {}
        self._head = ""
        self._load()
        self._stream_hashes = self._current_hashes()

    def _current_hashes(self):
        return {
            str(p): file_hash(p) if p.exists() else None
            for p in (self.attempts_path, self.records_path)
        }

    def _check_children(self) -> None:
        for path in (self.identity_path, self.attempts_path, self.records_path):
            if path.is_symlink():
                raise IntegrityError("extension E output may not be a symlink")

    def _assert_identity(self) -> None:
        self._check_children()
        if (
            digest(self.identity) != self._identity_digest
            or read_json(self.identity_path) != self.identity
            or self.grader.configuration() != self.settings
            or digest([self.questions, self._references]) != self.identity["private_inputs_sha256"]
        ):
            raise IntegrityError("extension E inputs/configuration changed")
        if self._current_hashes() != self._stream_hashes:
            raise IntegrityError("extension E stream changed while context was open")

    def _key(self, qid: str, candidate: str) -> str:
        if qid not in self.questions or not isinstance(candidate, str):
            raise IntegrityError("unknown question or non-string E candidate")
        return digest(
            [qid, self.grader.cache_key(self.questions[qid], self._references[qid], candidate)]
        )

    def _load(self) -> None:
        try:
            entries = list(read_jsonl(self.attempts_path))
            records = list(read_jsonl(self.records_path))
        except (ValueError, OSError) as exc:
            raise IntegrityError("damaged extension E stream; inspect before resuming") from exc
        for row in entries:
            body = {k: v for k, v in row.items() if k != "sha256"}
            if row.get("previous_sha256") != self._head or row.get("sha256") != digest(body):
                raise IntegrityError("extension E attempt stream changed")
            self._validate_attempt(row)
            key = row["text_id"]
            previous = self._attempts.setdefault(key, [])
            if len(previous) >= MAX_ATTEMPTS or row["attempt"] != len(previous) + 1:
                raise IntegrityError("extension E attempt count changed")
            if previous and (
                previous[-1].get("exception_type") or previous[-1]["result"]["status"] in TERMINAL
            ):
                raise IntegrityError("E attempted again after terminal result or exception")
            previous.append(row)
            self._head = row["sha256"]
        expected = {}
        for key, rows in self._attempts.items():
            result = self._terminal(rows)
            if result is not None:
                expected[key] = self._eval_record(rows[-1], result)
        for raw in records:
            record = EvalRecord.model_validate(raw)
            if record.text_id in self._records or record.text_id not in expected:
                raise IntegrityError("unexpected or duplicated extension E evaluation")
            if record.model_dump(mode="json") != expected[record.text_id].model_dump(mode="json"):
                raise IntegrityError("extension E evaluation disagrees with durable attempt")
            self._records[record.text_id] = record
        # Recover a success journaled before its Core record was flushed.
        for key, record in expected.items():
            if key not in self._records:
                self._commit_record(record)

    def _validate_attempt(self, row) -> None:
        qid, candidate = row.get("question_id"), row.get("candidate")
        if row.get("text_id") != self._key(qid, candidate):
            raise IntegrityError("extension E attempt question/reference/text identity changed")
        if bool(row.get("exception_type")) == (row.get("result") is not None):
            raise IntegrityError("E attempt must contain one result or exception")
        if row.get("result") is not None:
            result = GradeResult(**row["result"])
            expected = self.grader.cache_key(self.questions[qid], self._references[qid], candidate)
            if result.cache_key != expected:
                raise IntegrityError("E backend result belongs to another candidate")
            if result.status not in TERMINAL | {"timeout", "backend_error"}:
                raise IntegrityError("E returned an unknown status")
            if (
                result.math_verify_version != self.settings["math_verify_version"]
                or result.contract_version != self.settings["contract_version"]
            ):
                raise IntegrityError("E backend version/contract differs from bound identity")

    @staticmethod
    def _terminal(rows) -> GradeResult | None:
        if rows[-1].get("exception_type"):
            return None
        last = GradeResult(**rows[-1]["result"])
        if last.status in TERMINAL:
            return last
        if len(rows) == MAX_ATTEMPTS and all(
            r["result"]["status"] == "timeout" and r["result"]["detail"] == WALL_CLOCK_TIMEOUT
            for r in rows
        ):
            return GradeResult(
                **{**asdict(last), "status": "unresolved", "detail": TIMEOUT_UNRESOLVED}
            )
        return None

    def _eval_record(self, row, result) -> EvalRecord:
        key, qid = row["text_id"], row["question_id"]
        version = self.settings["math_verify_version"]
        return EvalRecord(
            text_id=key,
            text=row["candidate"],
            label=MathLabel(result.status.upper()),
            model="Math-Verify",
            model_version=version,
            prompt_hash=key,
            e_prompt_sha256=digest(self.settings),
            e_version=version,
            decoding=self.settings,
            reason=result.detail,
            question_id=qid,
            reference_answer=self._references[qid],
            extracted_answer=result.prediction_answer,
            unresolved=result.status == "unresolved",
            math_verify_version=version,
            n_boxes=result.n_boxes,
            boxes_all_equivalent=result.boxes_all_equivalent,
            created_at=row["created_at"],
        )

    def _commit_record(self, record) -> None:
        with JsonlWriter(self.records_path) as writer:
            writer.write(record)
        self._records[record.text_id] = record
        self._stream_hashes = self._current_hashes()

    def __call__(self, qid: str, candidate: str) -> dict[str, Any]:
        self._assert_identity()
        key = self._key(qid, candidate)
        if key not in self._records:
            attempts = self._attempts.setdefault(key, [])
            if attempts and attempts[-1].get("exception_type"):
                raise IntegrityError(
                    "previous E exception requires inspection; no automatic replay"
                )
            if attempts and self._terminal(attempts) is not None:
                self._commit_record(self._eval_record(attempts[-1], self._terminal(attempts)))
            while key not in self._records and len(attempts) < MAX_ATTEMPTS:
                row = {
                    "text_id": key,
                    "question_id": qid,
                    "candidate": candidate,
                    "attempt": len(attempts) + 1,
                    "created_at": utc_now_iso(),
                    "previous_sha256": self._head,
                    "result": None,
                    "exception_type": None,
                }
                try:
                    result = self.grader.grade(
                        self.questions[qid], self._references[qid], candidate
                    )
                except Exception as exc:
                    row["exception_type"] = type(exc).__name__
                    self._append_attempt(row)
                    raise
                row["result"] = asdict(result)
                self._validate_attempt(row)
                self._append_attempt(row)
                terminal = self._terminal(attempts)
                if terminal is not None:
                    self._commit_record(self._eval_record(attempts[-1], terminal))
                    break
            if key not in self._records:
                raise IntegrityError(f"E_error after three attempts: {qid}")
        rec = self._records[key]
        status = rec.label.value.lower()
        return {
            "status": status,
            "is_correct": status == "correct" if status != "unresolved" else None,
            "task_success": status == "correct",
            "prediction_answer": rec.extracted_answer,
            "math_verify_version": rec.math_verify_version,
            "n_boxes": rec.n_boxes,
            "boxes_all_equivalent": rec.boxes_all_equivalent,
            "e_timeout": rec.reason == TIMEOUT_UNRESOLVED,
        }

    def _append_attempt(self, row) -> None:
        row["sha256"] = digest(row)
        with JsonlWriter(self.attempts_path) as writer:
            writer.write(row)
        self._attempts.setdefault(row["text_id"], []).append(row)
        self._head = row["sha256"]
        self._stream_hashes = self._current_hashes()

    def record(self, qid: str, candidate: str) -> EvalRecord:
        """Return the persisted Core E record, only after this candidate has been graded."""
        self._assert_identity()
        return self._records[self._key(qid, candidate)].model_copy(deep=True)
