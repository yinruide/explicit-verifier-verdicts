import json
import re

import pytest
from pipelines.math_ext_ceiling import FEEDBACK, ceiling_config, run_ceiling

from rafr.contracts.schemas import Arm, State
from rafr.controller.decide import render_user_turn
from rafr.domains.math import generation
from rafr.domains.math.storage import IntegrityError
from rafr.formatter.extended import read_block_template
from rafr.formatter.math_extended import PROMPT_DIR, build_no_verifier_config, math_wording
from rafr.llm.cache import LLMCache
from rafr.llm.client import LLMClient
from rafr.llm.providers.fake import FakeProvider
from rafr.logging.writer import read_jsonl


def client(tmp_path, response):
    provider = FakeProvider(response)
    return LLMClient({"openai": provider}, LLMCache(tmp_path / "cache")), provider


def inputs(tmp_path):
    # A fixed created_at: State's default is the current time, and the ceiling's run identity
    # includes the starts, so two calls in different seconds looked like another run.
    start = State(
        state_id="q-g1",
        x0_id="q",
        x0="Compute 1+1",
        text=r"\boxed{3}",
        created_at="2026-11-05T00:00:00+00:00",
    )
    return dict(
        starts=[start],
        run_dir=tmp_path,
        run_id="OFFLINE-ceiling",
        seed=17,
        initial_identity={
            "split": "d_exp1",
            "sha256": "f" * 64,
            "source_root": str(tmp_path.parent / (tmp_path.name + "_source")),
        },
        e_identity={"mode": "offline_fixture"},
    )


def test_ceiling_block_exactly_preserves_no_verifier_prefix_and_original_separators():
    nv = build_no_verifier_config(17, 4)
    cfg = ceiling_config(17)
    block = "\n\nFeedback:\n---\n" + FEEDBACK + "\n---"
    assert cfg.rewrite_template == nv.rewrite_template + block
    original = read_block_template(generation.PROMPT_DIR / "rewrite.txt")
    original_block = "\n\nChecker output:" + original.rsplit("\n\nChecker output:", 1)[1]
    assert block == original_block.replace("Checker output:", "Feedback:").replace(
        "{observation}", FEEDBACK
    )
    assert cfg.system_prompt == nv.system_prompt
    assert Arm.ORACLE_FEEDBACK.value == "oracle_feedback"


def test_no_verifier_has_the_recorded_edits_and_no_checker_vocabulary():
    from tests.test_math_ext_contracts import MATH_NO_VERIFIER_EDITS

    # The ceiling reads both files the way the controller does (read_block_template).
    for name in ("system", "rewrite"):
        expected = read_block_template(generation.PROMPT_DIR / f"{name}.txt")
        for old, replacement in MATH_NO_VERIFIER_EDITS[name]:
            expected = expected.replace(old, replacement)
        assert read_block_template(PROMPT_DIR / f"no_verifier_{name}.txt") == expected
    cfg = build_no_verifier_config(17, 4)
    request = cfg.system_prompt + render_user_turn(cfg, "Compute 1+1", r"\boxed{3}", "")
    assert not re.search(r"checker|score|threshold|flag", request, re.I)
    assert "{observation}" not in cfg.rewrite_template
    math_wording()  # Final shared detox blocks need no draft escape hatch.


def test_correct_rewrite_stops_even_without_controller_accept_and_resume_no_calls(tmp_path):
    c, p = client(
        tmp_path, lambda req: json.dumps({"action": "rewrite", "revised_text": r"\boxed{2}"})
    )
    seen = []

    def grade(qid, text):
        seen.append((qid, text))
        return {"status": "correct"}

    args = inputs(tmp_path)
    result = run_ceiling(c, grade_callback=grade, **args)
    assert result[0]["task_success"] and result[0]["n_decisions"] == 1
    assert len(p.calls) == len(seen) == 1
    assert run_ceiling(c, grade_callback=grade, **args) == result
    assert len(p.calls) == len(seen) == 1
    rows = list(read_jsonl(tmp_path / "decisions.jsonl"))
    assert rows[0]["arm"] == "oracle_feedback" and rows[0]["s1"] is None


def test_incorrect_accept_does_not_override_e_and_has_four_decisions(tmp_path):
    c, p = client(tmp_path, lambda req: '{"action":"accept","revised_text":null}')
    result = run_ceiling(
        c, grade_callback=lambda *a: pytest.fail("unchanged x1 rejudged"), **inputs(tmp_path)
    )
    assert not result[0]["task_success"] and result[0]["n_decisions"] == 4
    assert len(p.calls) == 4


UNRESOLVED = {"status": "unresolved", "is_correct": None, "task_success": False, "e_timeout": True}


def counting_rewrites():
    count = 0

    def response(req):
        nonlocal count
        count += 1
        return json.dumps({"action": "rewrite", "revised_text": rf"\boxed{{{count + 10}}}"})

    return response


def test_unresolved_e_is_a_failure_the_trajectory_continues_and_resume_works(tmp_path):
    """2026-11-05, the original math Exp 2 rule (M.9): E's UNRESOLVED (three wall-clock
    timeouts) is a task failure; the ceiling goes on as after incorrect, never STOPPED."""
    c, p = client(tmp_path, counting_rewrites())
    seen = []

    def interrupted(qid, text):
        seen.append(text)
        if len(seen) == 1:
            return UNRESOLVED
        raise RuntimeError("offline interrupted grading")

    with pytest.raises(RuntimeError, match="interrupted grading"):
        run_ceiling(c, grade_callback=interrupted, **inputs(tmp_path))
    assert not (tmp_path / "STOPPED.json").exists()
    assert [s["status"] for s in read_jsonl(tmp_path / "ceiling_steps.jsonl")] == ["unresolved"]
    assert len(p.calls) == 2  # t = 1 decided and judged; t = 2 decided, its E interrupted
    result = run_ceiling(c, grade_callback=lambda *a: dict(UNRESOLVED), **inputs(tmp_path))
    assert len(p.calls) == 4  # resume: t = 1, 2 replayed from the logs, t = 3, 4 new
    assert not (tmp_path / "STOPPED.json").exists()
    (r,) = result
    assert r["n_decisions"] == 4 and r["terminal_status"] == "unresolved" and r["unresolved"]
    assert r["task_success"] is False and r["stopped_by_e"] is False
    assert [s["status"] for s in r["rounds"]] == ["unresolved"] * 4
    assert run_ceiling(c, grade_callback=lambda *a: pytest.fail("rejudged"), **inputs(tmp_path))
    assert len(p.calls) == 4


def test_unresolved_then_correct_stops_on_correct(tmp_path):
    c, p = client(tmp_path, counting_rewrites())
    grades = iter([dict(UNRESOLVED), {"status": "correct"}])
    (r,) = run_ceiling(c, grade_callback=lambda *a: next(grades), **inputs(tmp_path))
    assert r["task_success"] and r["n_decisions"] == 2 and r["terminal_status"] == "correct"
    assert [s["status"] for s in r["rounds"]] == ["unresolved", "correct"] and len(p.calls) == 2


@pytest.mark.parametrize("grade", [None, {"status": "correct", "task_success": False}])
def test_unknown_e_sticky_stop_requires_new_run(tmp_path, grade):
    c, p = client(
        tmp_path, lambda req: json.dumps({"action": "rewrite", "revised_text": r"\boxed{2}"})
    )
    with pytest.raises(IntegrityError):
        run_ceiling(c, grade_callback=lambda *a: grade, **inputs(tmp_path))
    assert len(p.calls) == 1 and (tmp_path / "STOPPED.json").exists()
    with pytest.raises(IntegrityError, match="STOPPED"):
        run_ceiling(
            c, grade_callback=lambda *a: pytest.fail("E called after STOPPED"), **inputs(tmp_path)
        )
    assert len(p.calls) == 1
    fresh = tmp_path / "fresh"
    c2, _ = client(
        fresh, lambda req: json.dumps({"action": "rewrite", "revised_text": r"\boxed{2}"})
    )
    result = run_ceiling(c2, grade_callback=lambda *a: {"status": "correct"}, **inputs(fresh))
    assert result[0]["task_success"]


def test_completed_artifact_tamper_refused(tmp_path):
    c, _ = client(tmp_path, lambda req: '{"action":"accept","revised_text":null}')
    run_ceiling(c, grade_callback=lambda *a: None, **inputs(tmp_path))
    with (tmp_path / "decisions.jsonl").open("a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(IntegrityError, match="artifacts changed"):
        run_ceiling(c, grade_callback=lambda *a: None, **inputs(tmp_path))


def pending_grade(*args):
    raise RuntimeError("offline interrupted grading")


def partial_run(tmp_path, *, with_receipt=False):
    count = 0

    def response(req):
        nonlocal count
        count += 1
        return json.dumps({"action": "rewrite", "revised_text": f"candidate {count}"})

    def grade(qid, text):
        if with_receipt and text == "candidate 1":
            return {"status": "incorrect", "task_success": False}
        return pending_grade(qid, text)

    c, p = client(tmp_path, response)
    with pytest.raises(RuntimeError, match="offline interrupted"):
        run_ceiling(c, grade_callback=grade, **inputs(tmp_path))
    assert not (tmp_path / "STOPPED.json").exists()
    return c, p


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def rebind_fixture_checkpoint(path):
    # Deliberately re-bind this test fixture's checkpoint, so semantic guards (rather
    # than just byte identity) must reject the tampering below. No real artifacts used.
    from rafr.contracts.schemas import LogRecord
    from rafr.domains.math.storage import atomic_json, digest

    logs = [
        LogRecord.model_validate(r).model_dump(mode="json")
        for r in read_jsonl(path / "decisions.jsonl")
    ]
    steps = list(read_jsonl(path / "ceiling_steps.jsonl"))
    atomic_json(
        path / "ceiling_checkpoint.json",
        {
            "n_decisions": len(logs),
            "n_steps": len(steps),
            "decisions_sha256": digest(logs),
            "steps_sha256": digest(steps),
        },
    )


def test_partial_pending_grade_resume_does_not_reissue_controller(tmp_path):
    c, p = partial_run(tmp_path)
    seen = []

    def grade(qid, text):
        seen.append((qid, text))
        return {"status": "correct", "task_success": True}

    result = run_ceiling(c, grade_callback=grade, **inputs(tmp_path))
    assert result[0]["task_success"]
    assert len(p.calls) == 1 and seen == [("q", "candidate 1")]


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 99),
        ("model", "wrong-model"),
        ("run_id", "other-run"),
        ("decoding", {}),
        ("x_t", "other text"),
        ("s1", 0.1),
        ("tau1", 0.3),
        ("e_score", 1.0),
        ("prompt_hash", "f" * 64),
        ("x0_id", "other-q"),
        ("state_id", "other-state"),
    ],
)
def test_partial_request_identity_tamper_refused_before_calls(tmp_path, field, value):
    c, p = partial_run(tmp_path)
    rows = list(read_jsonl(tmp_path / "decisions.jsonl"))
    rows[0][field] = value
    write_rows(tmp_path / "decisions.jsonl", rows)
    rebind_fixture_checkpoint(tmp_path)
    with pytest.raises(IntegrityError):
        run_ceiling(
            c, grade_callback=lambda *a: pytest.fail("tampered E invoked"), **inputs(tmp_path)
        )
    assert len(p.calls) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 999),
        ("prompt_hash", "f" * 64),
        ("attempt", 1),
        ("raw_output", '{"action":"accept"}'),
    ],
)
def test_partial_attempt_identity_or_parser_tamper_refused(tmp_path, field, value):
    c, p = partial_run(tmp_path)
    rows = list(read_jsonl(tmp_path / "decisions.jsonl"))
    rows[0]["attempts"][0][field] = value
    write_rows(tmp_path / "decisions.jsonl", rows)
    rebind_fixture_checkpoint(tmp_path)
    with pytest.raises(IntegrityError):
        run_ceiling(c, grade_callback=lambda *a: pytest.fail("E invoked"), **inputs(tmp_path))
    assert len(p.calls) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "correct"),
        ("grade", {"status": "incorrect", "task_success": True}),
        ("e_identity_sha256", "f" * 64),
        ("input_sha256", "f" * 64),
        ("output_sha256", "f" * 64),
        ("previous_sha256", "f" * 64),
        ("decision_sha256", "f" * 64),
        ("receipt_sha256", "f" * 64),
        ("output_text", "other text"),
    ],
)
def test_partial_receipt_semantics_or_binding_tamper_refused(tmp_path, field, value):
    c, p = partial_run(tmp_path, with_receipt=True)
    rows = list(read_jsonl(tmp_path / "ceiling_steps.jsonl"))
    rows[0][field] = value
    write_rows(tmp_path / "ceiling_steps.jsonl", rows)
    rebind_fixture_checkpoint(tmp_path)
    with pytest.raises(IntegrityError):
        run_ceiling(c, grade_callback=lambda *a: pytest.fail("E invoked"), **inputs(tmp_path))
    assert len(p.calls) == 2


def test_partial_checkpoint_catches_raw_output_mutation(tmp_path):
    c, p = partial_run(tmp_path)
    rows = list(read_jsonl(tmp_path / "decisions.jsonl"))
    rows[0]["attempts"][0]["raw_output"] += " "
    write_rows(tmp_path / "decisions.jsonl", rows)
    with pytest.raises(IntegrityError, match="checkpoint content"):
        run_ceiling(c, grade_callback=pending_grade, **inputs(tmp_path))
    assert len(p.calls) == 1


def test_partial_receipt_then_pending_grade_resume_preserves_hash_chain(tmp_path):
    c, p = partial_run(tmp_path, with_receipt=True)
    result = run_ceiling(c, grade_callback=lambda *a: {"status": "correct"}, **inputs(tmp_path))
    assert result[0]["task_success"] and len(p.calls) == 2
    rows = list(read_jsonl(tmp_path / "ceiling_steps.jsonl"))
    assert rows[1]["previous_sha256"] == rows[0]["receipt_sha256"]


@pytest.mark.parametrize("sha", ["", "f" * 63, "g" * 64, "f" * 65, 123])
def test_initial_source_hash_strict_sha256(tmp_path, sha):
    c, p = client(tmp_path, lambda req: pytest.fail("provider invoked"))
    args = inputs(tmp_path)
    args["initial_identity"]["sha256"] = sha
    with pytest.raises(IntegrityError, match="explicitly bound"):
        run_ceiling(c, grade_callback=pending_grade, **args)
    assert not p.calls


def test_initial_source_root_required_and_cannot_overlap_output(tmp_path):
    c, _ = client(tmp_path, lambda req: pytest.fail("provider invoked"))
    args = inputs(tmp_path)
    args["initial_identity"].pop("source_root")
    with pytest.raises(IntegrityError, match="explicitly bound"):
        run_ceiling(c, grade_callback=pending_grade, **args)
    args["initial_identity"]["source_root"] = str(tmp_path)
    with pytest.raises(IntegrityError):
        run_ceiling(c, grade_callback=pending_grade, **args)


@pytest.mark.parametrize("by_module", [False, True])
def test_bound_original_math_grader_rejected(tmp_path, by_module):
    from functools import partial
    from pathlib import Path

    class Grader:
        def grade(self, *args):
            pytest.fail("original grader invoked")

    args = inputs(tmp_path)
    grader = Grader()
    if by_module:
        Grader.__module__ = "pipelines.math_pipeline"
    else:
        grader.root = Path(args["initial_identity"]["source_root"])
    c, p = client(tmp_path, lambda req: pytest.fail("provider invoked"))
    with pytest.raises(IntegrityError, match="original MathPipeline"):
        run_ceiling(c, grade_callback=partial(grader.grade), **args)
    assert not p.calls


def test_smoke_uses_general_exp2_extension_namespace(tmp_path):
    from rafr.contracts.schemas import Lambda
    from rafr.controller import decide as ctrl

    c, p = client(tmp_path, lambda req: '{"action":"accept","revised_text":null}')
    args = inputs(tmp_path)
    args["initial_identity"]["split"] = "unlabeled_pool"
    run_ceiling(c, grade_callback=pending_grade, smoke=True, **args)
    meta = json.loads((tmp_path / "ext_meta.json").read_text(encoding="utf-8"))
    assert meta["seed_namespace"] == "math_exp2ext_smoke"
    cfg = ceiling_config(17)
    labels = ctrl.DecisionLabels("q", Arm.ORACLE_FEEDBACK, Lambda.NONE, 1, "math_exp2ext_smoke")
    assert p.calls[0].seed == ctrl.attempt_call_seed(ctrl.call_seed(cfg, labels), 0)


def test_parse_failure_prefix_replays_core_retries_then_resumes(tmp_path):
    from rafr.contracts.schemas import PARSE_MAX_RETRIES

    n = 0

    def response(req):
        nonlocal n
        n += 1
        if n <= PARSE_MAX_RETRIES + 1:
            return "invalid JSON"
        raise RuntimeError("offline interrupted provider")

    c, p = client(tmp_path, response)
    with pytest.raises(RuntimeError, match="interrupted provider"):
        run_ceiling(c, grade_callback=pending_grade, **inputs(tmp_path))
    rows = list(read_jsonl(tmp_path / "decisions.jsonl"))
    assert len(rows) == 1 and rows[0]["parse_fail"]
    assert len(rows[0]["attempts"]) == PARSE_MAX_RETRIES + 1
    c2, p2 = client(tmp_path, lambda req: '{"action":"accept","revised_text":null}')
    result = run_ceiling(c2, grade_callback=pending_grade, **inputs(tmp_path))
    assert not result[0]["task_success"] and result[0]["n_decisions"] == 4
    assert len(p2.calls) == 3 and len(p.calls) == PARSE_MAX_RETRIES + 2


def test_actual_request_has_only_public_text_and_static_feedback(tmp_path):
    seen = []

    def response(req):
        seen.append(req)
        return json.dumps({"action": "rewrite", "revised_text": "public candidate"})

    args = inputs(tmp_path)
    # Private source/E identity receipts must never be forwarded into a request.
    args["e_identity"]["private_reference"] = "E_ONLY_SENTINEL"
    args["initial_identity"]["label_receipt"] = "PRIVATE_LABEL_SENTINEL"
    c, _ = client(tmp_path, response)
    result = run_ceiling(c, grade_callback=lambda qid, text: {"status": "correct"}, **args)
    assert result[0]["task_success"]
    request = seen[0]
    rendered = request.system + "\n" + json.dumps(request.messages)
    assert FEEDBACK in rendered and "Compute 1+1" in rendered
    assert "E_ONLY_SENTINEL" not in rendered and "PRIVATE_LABEL_SENTINEL" not in rendered
    assert not re.search(r"checker|threshold|score|flag", rendered, re.I)


def test_live_formal_ceiling_refuses_incomplete_population_before_e_or_model(tmp_path):
    class Transport:
        def complete(self, request):
            pytest.fail("model must not run")

    c = LLMClient({"openai": Transport()})
    args = inputs(tmp_path)
    args["initial_identity"]["source_execution_mode"] = "live"
    with pytest.raises(IntegrityError, match="all 300"):
        run_ceiling(
            c,
            grade_callback=lambda *a: pytest.fail("E must not run"),
            execution_mode="live",
            confirmed=True,
            **args,
        )
