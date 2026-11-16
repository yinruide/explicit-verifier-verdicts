"""Local Math-Verify expressions and fault injection; no external model or source writes."""

from __future__ import annotations

import json

import pytest
from pipelines.math_ext_e import TIMEOUT_UNRESOLVED, WALL_CLOCK_TIMEOUT, ExtensionMathEvaluator

from rafr.contracts.schemas import EvalRecord, MathLabel
from rafr.domains.math.grading import GradeResult, MathGrader
from rafr.domains.math.storage import IntegrityError, file_hash
from rafr.logging.writer import read_jsonl


class OfflineBackend:
    def compare(self, reference, prediction):
        return reference == prediction


class ScriptGrader:
    def __init__(self, script):
        self.base = MathGrader(OfflineBackend(), backend_identity="fixture-math-e-v1")
        self.script = list(script)
        self.calls = []

    def configuration(self):
        return self.base.configuration()

    def cache_key(self, *args):
        return self.base.cache_key(*args)

    def grade(self, question, reference, candidate):
        self.calls.append((question, reference, candidate))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        status, detail = item if isinstance(item, tuple) else (item, "fixture")
        return GradeResult(
            status,
            detail,
            prediction_answer="2",
            n_boxes=1,
            cache_key=self.cache_key(question, reference, candidate),
        )


def context(tmp_path, grader=None, **kwargs):
    source = tmp_path / "original"
    source.mkdir(exist_ok=True)
    if not (source / "sentinel.txt").exists():
        (source / "sentinel.txt").write_text("original evidence", encoding="utf-8")
    arguments = {
        "questions": {"q1": "What is one plus one?", "q2": "What is two plus one?"},
        "private_references": {"q1": "2", "q2": "3"},
        "source_identity": {"snapshot_sha256": file_hash(source / "sentinel.txt")},
        "source_root": source,
        "grader": grader,
    }
    arguments.update(kwargs)
    return ExtensionMathEvaluator(tmp_path / "extension", **arguments)


def test_real_math_verify_trivial_expressions_last_box_and_context_cache(tmp_path):
    e = context(tmp_path)
    source_hash = file_hash(tmp_path / "original" / "sentinel.txt")
    text = r"Compute carefully. \boxed{2}"
    assert e("q1", text)["status"] == "correct"
    assert e("q2", text)["status"] == "incorrect"
    assert e("q1", r"\boxed{99}, corrected to \boxed{1+1}")["status"] == "correct"
    assert e("q1", "I cannot find the answer")["status"] == "unresolved"
    count = len(list(read_jsonl(e.attempts_path)))
    restarted = context(tmp_path)
    assert restarted("q1", text)["status"] == "correct"
    assert len(list(read_jsonl(e.attempts_path))) == count
    records = [EvalRecord.model_validate(r) for r in read_jsonl(e.records_path)]
    assert len(records) == 4 and {r.math_verify_version for r in records} == {"0.9.0"}
    assert file_hash(tmp_path / "original" / "sentinel.txt") == source_hash
    assert e.record("q1", text).label is MathLabel.CORRECT


def test_backend_errors_retry_twice_then_success_and_reopen_cached(tmp_path):
    grader = ScriptGrader(["backend_error", "timeout", "correct"])
    e = context(tmp_path, grader)
    result = e("q1", r"\boxed{2}")
    assert result["task_success"] and len(grader.calls) == 3
    reopened = context(tmp_path, ScriptGrader([]))
    assert reopened("q1", r"\boxed{2}") == result
    assert len(list(read_jsonl(e.attempts_path))) == 3


def test_only_three_wall_clock_timeouts_become_unresolved(tmp_path):
    grader = ScriptGrader([("timeout", WALL_CLOCK_TIMEOUT)] * 3)
    e = context(tmp_path, grader)
    result = e("q1", r"\boxed{2}")
    assert result["status"] == "unresolved" and result["e_timeout"]
    assert result["task_success"] is False
    assert e.record("q1", r"\boxed{2}").reason == TIMEOUT_UNRESOLVED
    assert len(grader.calls) == 3
    assert context(tmp_path, ScriptGrader([]))("q1", r"\boxed{2}") == result


@pytest.mark.parametrize(
    "script",
    [
        ["backend_error"] * 3,
        [("timeout", "parser_timeout")] * 3,
        [("timeout", WALL_CLOCK_TIMEOUT), "backend_error", ("timeout", WALL_CLOCK_TIMEOUT)],
    ],
)
def test_other_exhausted_errors_stop_and_do_not_reset_retry_budget(tmp_path, script):
    e = context(tmp_path, ScriptGrader(script))
    with pytest.raises(IntegrityError, match="three attempts"):
        e("q1", r"\boxed{2}")
    assert list(read_jsonl(e.records_path)) == []
    fresh = ScriptGrader(["correct"])
    with pytest.raises(IntegrityError, match="three attempts"):
        context(tmp_path, fresh)("q1", r"\boxed{2}")
    assert fresh.calls == []


def test_unknown_exception_preserved_and_no_automatic_replay(tmp_path):
    fault = RuntimeError("private backend diagnostic must not be in operational log")
    e = context(tmp_path, ScriptGrader([fault]))
    with pytest.raises(RuntimeError) as caught:
        e("q1", r"\boxed{2}")
    assert caught.value is fault
    assert "private backend diagnostic" not in e.attempts_path.read_text(encoding="utf-8")
    fresh = ScriptGrader(["correct"])
    with pytest.raises(IntegrityError, match="previous E exception"):
        context(tmp_path, fresh)("q1", r"\boxed{2}")
    assert fresh.calls == []


def test_attempt_survives_interruption_before_core_record(tmp_path, monkeypatch):
    e = context(tmp_path, ScriptGrader(["correct"]))

    def crash(record):
        raise RuntimeError("crash after durable attempt")

    monkeypatch.setattr(e, "_commit_record", crash)
    with pytest.raises(RuntimeError):
        e("q1", r"\boxed{2}")
    assert not e.records_path.exists()
    fresh = ScriptGrader([])
    restarted = context(tmp_path, fresh)
    assert restarted("q1", r"\boxed{2}")["task_success"]
    assert fresh.calls == [] and len(list(read_jsonl(restarted.records_path))) == 1


def test_retry_budget_survives_interrupted_error_attempt(tmp_path, monkeypatch):
    e = context(tmp_path, ScriptGrader(["backend_error"]))
    append = e._append_attempt

    def crash(row):
        append(row)
        raise RuntimeError("interrupted after error record")

    monkeypatch.setattr(e, "_append_attempt", crash)
    with pytest.raises(RuntimeError):
        e("q1", r"\boxed{2}")
    fresh = ScriptGrader(["backend_error", "correct"])
    assert context(tmp_path, fresh)("q1", r"\boxed{2}")["task_success"]
    assert len(fresh.calls) == 2


@pytest.mark.parametrize("part", ["question", "reference", "config", "source"])
def test_identity_change_refused_before_backend_call(tmp_path, part):
    context(tmp_path, ScriptGrader(["correct"]))
    kwargs = {}
    if part == "question":
        kwargs["questions"] = {"q1": "changed", "q2": "What is two plus one?"}
    elif part == "reference":
        kwargs["private_references"] = {"q1": "7", "q2": "3"}
    elif part == "source":
        kwargs["source_identity"] = {"snapshot_sha256": "changed"}
    grader = ScriptGrader([])
    if part == "config":
        grader.base._configuration["backend_identity"] = "different-backend"
    with pytest.raises(IntegrityError, match="identity changed"):
        context(tmp_path, grader, **kwargs)
    assert grader.calls == []


@pytest.mark.parametrize("stream", ["attempts", "evaluations"])
def test_persisted_result_tamper_refused_open_and_reopened(tmp_path, stream):
    e = context(tmp_path, ScriptGrader(["correct"]))
    e("q1", r"\boxed{2}")
    path = e.attempts_path if stream == "attempts" else e.records_path
    rows = list(read_jsonl(path))
    if stream == "attempts":
        rows[0]["result"]["status"] = "incorrect"
    else:
        rows[0]["label"] = "INCORRECT"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(IntegrityError, match="stream changed"):
        e("q1", r"\boxed{2}")
    with pytest.raises(IntegrityError):
        context(tmp_path, ScriptGrader([]))


def test_private_references_only_in_local_e_record_and_output_isolated(tmp_path):
    secret = "PRIVATE_REFERENCE_SENTINEL"
    e = context(tmp_path, ScriptGrader(["correct"]), private_references={"q1": secret, "q2": "3"})
    e("q1", r"\boxed{2}")
    assert secret not in e.identity_path.read_text(encoding="utf-8") + e.attempts_path.read_text(
        encoding="utf-8"
    )
    assert e.record("q1", r"\boxed{2}").reference_answer == secret
    with pytest.raises(IntegrityError, match="overlaps"):
        ExtensionMathEvaluator(
            e.root,
            questions=e.questions,
            private_references={"q1": secret, "q2": "3"},
            source_identity={"fixture": 1},
            source_root=e.root,
            grader=ScriptGrader([]),
        )


def test_child_symlink_and_missing_identity_refused(tmp_path, make_symlink):
    e = context(tmp_path, ScriptGrader(["correct"]))
    e("q1", r"\boxed{2}")
    e.identity_path.unlink()
    with pytest.raises(IntegrityError, match="no bound identity"):
        context(tmp_path, ScriptGrader([]))
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    make_symlink(e.identity_path, outside)
    with pytest.raises(IntegrityError, match="symlink"):
        context(tmp_path, ScriptGrader([]))


def test_same_context_retry_after_record_write_failure_does_not_regrade(tmp_path, monkeypatch):
    grader = ScriptGrader(["correct"])
    e = context(tmp_path, grader)
    commit = e._commit_record

    def crash(record):
        raise OSError("disk interruption")

    monkeypatch.setattr(e, "_commit_record", crash)
    with pytest.raises(OSError):
        e("q1", r"\boxed{2}")
    monkeypatch.setattr(e, "_commit_record", commit)
    assert e("q1", r"\boxed{2}")["task_success"]
    assert len(grader.calls) == 1


def test_backend_result_for_different_candidate_rejected(tmp_path):
    grader = ScriptGrader(["correct"])
    original = grader.grade

    def wrong(question, reference, candidate):
        return original(question, reference, "a different candidate")

    grader.grade = wrong
    e = context(tmp_path, grader)
    with pytest.raises(IntegrityError, match="another candidate"):
        e("q1", r"\boxed{2}")
    assert list(read_jsonl(e.records_path)) == []


def test_live_configuration_mutation_is_rejected(tmp_path):
    grader = ScriptGrader(["correct"])
    e = context(tmp_path, grader)
    grader.base._configuration["backend_identity"] = "changed"
    with pytest.raises(IntegrityError, match="configuration changed"):
        e("q1", r"\boxed{2}")
    assert grader.calls == []


def test_live_e_rejects_plain_callable_and_injected_backend(tmp_path):
    from pipelines.math_ext_e import check_e_callback

    e = context(tmp_path, ScriptGrader([]))
    kwargs = dict(
        run_dir=e.root, source_root=e.identity["source_root"], e_identity=e.identity, live=True
    )
    with pytest.raises(IntegrityError, match="extension-owned Math-Verify"):
        check_e_callback(lambda *a: {"status": "correct"}, **kwargs)
    with pytest.raises(IntegrityError, match="frozen Math-Verify configuration"):
        check_e_callback(e, **kwargs)


def test_live_e_accepts_real_local_math_verify_without_calling_it(tmp_path):
    from pipelines.math_ext_e import check_e_callback

    e = context(tmp_path)
    check_e_callback(
        e, run_dir=e.root, source_root=e.identity["source_root"], e_identity=e.identity, live=True
    )
    assert not e.attempts_path.exists()
