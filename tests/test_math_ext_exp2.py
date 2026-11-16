"""Synthetic-only extension loop regression tests; every output is under tmp_path."""

import json
from dataclasses import replace

import pytest
from pipelines.math_exp2_loop import text_key
from pipelines.math_ext_exp2 import (
    MathExtSetup,
    T1Reuse,
    grade_math_exp2,
    plan_for_cohort,
    run_math_exp2,
)

from rafr.calibration.apply import Calibrator
from rafr.contracts.schemas import Arm, Score, State
from rafr.controller.decide import ControllerConfig
from rafr.domains.math.storage import IntegrityError, file_hash
from rafr.experiments.exp2 import round_dir
from rafr.formatter.extended import ExtWording
from rafr.llm.cache import LLMCache
from rafr.llm.client import LLMClient
from rafr.llm.ledger import Ledger
from rafr.llm.providers.fake import FakeProvider
from rafr.logging.writer import read_jsonl


def config(nv=False):
    return ControllerConfig(
        provider="fake",
        api=None,
        model="stub",
        version="stub-v1",
        decoding={},
        base_seed=123,
        seed_scheme="per_state",
        system_prompt="Offline math, four decisions",
        rewrite_template="Q:{x0}\nANSWER:{x_t}" + ("" if nv else "\nOBS:{observation}"),
        retry_prompt="Retry JSON.",
        round_budget=4,
    )


def calibration():
    return Calibrator(
        {
            "kind": "platt_logit",
            "domain": "math",
            "targets": "platt_smoothed",
            "epsilon": 1e-7,
            "verifiers": {
                f"v{i}": {"verifier_id": f"v{i}", "a": 1.0, "b": 0.0, "converged": True}
                for i in (1, 2)
            },
        },
        "frozen-fixture",
    )


class Scorer:
    def __init__(self, overlong=False):
        self.calls = []
        self.overlong = overlong

    def __call__(self, items):
        self.calls.append(list(items))
        result, over = {}, {}
        for qid, problem, text, key in items:
            assert "PRIVATE_REFERENCE" not in problem + text
            if self.overlong and text != "same":
                over[key] = {"tokens": 5000, "L_in": 4096}
                continue
            value = 0.8 if "B-rewrite" in text or qid == "q2" else 0.2
            result[key] = tuple(
                Score(state_id=key, verifier_id=f"v{i}", model_version="v1", score=value)
                for i in (1, 2)
            )
        return result, over


def harness(tmp_path, arms=(Arm.A1, Arm.B), n=1, responder=None, scorer=None):
    states = [
        State(state_id=f"q{i}-g1", x0_id=f"q{i}", x0=f"question {i}", text="same")
        for i in range(1, n + 1)
    ]
    wording = ExtWording(
        lambda o: f"A1:{o.s1_obs}",
        lambda o: f"B:{o.s1_obs}",
        lambda f: f"flag:{f.y1}",
        "pi:{pihat1}/{pihat2}",
        "\n",
    )
    scorer = scorer or Scorer()
    setup = MathExtSetup(
        "fixture",
        tmp_path / "loop",
        "e_correct",
        states,
        {s.x0_id: list(arms) for s in states},
        (0.5, 0.5),
        ("v1", "v2"),
        ("v1", "v1"),
        scorer,
        wording,
        123,
        {s.x0_id: {"status": "correct", "task_success": True} for s in states},
        calibration(),
    )
    provider = FakeProvider(responder or (lambda r: '{"action":"accept","revised_text":null}'))
    client = LLMClient(
        {"fake": provider},
        LLMCache(setup.run_dir / "cache"),
        Ledger(setup.run_dir / "ledger.jsonl"),
    )
    return setup, client, provider, scorer


def run(h, **kwargs):
    s, c, _, _ = h
    return run_math_exp2(c, config(), config(True), s, **kwargs)


def test_context_and_trajectory_keys_do_not_cross_contaminate(tmp_path):
    def respond(req):
        prompt = req.messages[-1]["content"]
        if "ANSWER:same" in prompt:
            text = "B-rewrite" if "OBS:B:" in prompt else "A-rewrite"
            return json.dumps({"action": "rewrite", "revised_text": text})
        return '{"action":"accept","revised_text":null}'

    h = harness(tmp_path, n=2, responder=respond)
    run(h)
    rows = list(read_jsonl(h[0].run_dir / "decisions.jsonl"))
    by = {(r["x0_id"], r["arm"], r["t"]): r for r in rows}
    assert by["q1", "A1", 1]["s1"] == 0.2
    assert by["q2", "A1", 1]["s1"] == 0.8
    assert by["q1", "A1", 2]["s1"] == 0.2
    assert by["q1", "B", 2]["s1"] == 0.8
    assert len(h[3].calls[0]) == 2  # one state per question, not one per text or arm
    for row in rows:
        assert row["state_id"] == f"{row['x0_id']}-{row['arm']}-t{row['t']}"


def test_no_verifier_has_no_scoring_or_score_fields(tmp_path):
    h = harness(tmp_path, arms=(Arm.NO_VERIFIER,))
    h[0].score_texts = None
    run(h)
    assert h[3].calls == []
    assert list(read_jsonl(h[0].run_dir / "scores.jsonl")) == []
    for r in read_jsonl(h[0].run_dir / "decisions.jsonl"):
        assert all(r[key] is None for key in ("s1", "s2", "s1_tilde", "tau1", "fields", "v1_id"))
    assert all("OBS:" not in str(r.messages) for r in h[2].calls)


def test_wrong_domain_calibration_refuses_before_calls(tmp_path):
    h = harness(tmp_path, arms=(Arm.B_PIHAT,))
    h[0].calibrator.record["domain"] = "detox"
    with pytest.raises(IntegrityError, match="math-domain"):
        run(h)
    assert h[2].calls == [] and h[3].calls == []


def test_frozen_pihat_and_resume_zero_paid_replay(tmp_path):
    h = harness(tmp_path, arms=(Arm.B_PIHAT,))
    run(h)
    n_calls = len(h[2].calls)
    n_score = len(h[3].calls)
    run(h)
    assert (len(h[2].calls), len(h[3].calls)) == (n_calls, n_score)
    side = list(read_jsonl(h[0].run_dir / "pihat.jsonl"))
    assert len(side) == 1 and side[0]["pihat1"] == pytest.approx(0.2)
    assert "pi:0.20000/0.20000" in str(h[2].calls[0].messages)


def test_overlength_retained_and_e_outside_loop(tmp_path):
    h = harness(
        tmp_path,
        arms=(Arm.A1,),
        scorer=Scorer(overlong=True),
        responder=lambda r: '{"action":"rewrite","revised_text":"long"}',
    )
    trajectories = run(h)
    calls = []

    def grade(qid, text):
        calls.append((qid, text))
        return {"status": "unresolved", "task_success": False, "is_correct": None}

    assert not (h[0].run_dir / "math_judgments.jsonl").exists()
    result = grade_math_exp2(h[0], trajectories, grade, e_identity={"fixture": 1})
    assert len(calls) == 1 and result[0]["termination_reason"] == "v_input_overlength"
    assert result[0]["unresolved"] and not result[0]["task_success"]
    grade_math_exp2(h[0], trajectories, grade, e_identity={"fixture": 1})
    assert len(calls) == 1


def test_fourth_rewrite_is_final_and_graded_after_loop(tmp_path):
    def respond(req):
        text = req.messages[-1]["content"].split("ANSWER:")[1].split("\n")[0]
        return json.dumps({"action": "rewrite", "revised_text": text + "!"})

    h = harness(tmp_path, arms=(Arm.A1,), responder=respond)
    trs = run(h)
    assert trs[0].x_t == "same!!!!" and trs[0].n_rewrites == 4
    seen = []

    def grade(qid, text):
        seen.append(text)
        return {
            "status": "correct" if text.endswith("!!!!") else "incorrect",
            "task_success": text.endswith("!!!!"),
        }

    result = grade_math_exp2(h[0], trs, grade, e_identity={"fixture": 1})
    assert seen[-1] == "same!!!!" and len(seen) == 4
    assert result[0]["task_success"] and len(result[0]["rounds"]) == 4


def test_interruption_after_round_resumes_but_changed_round_refused(tmp_path):
    h = harness(tmp_path)

    def stop(t):
        if t == 1:
            raise RuntimeError("interrupted after complete round")

    with pytest.raises(RuntimeError):
        run(h, on_round=stop)
    paid = len(h[2].calls)
    run(h)
    assert len(h[2].calls) == paid
    path = round_dir(h[0].run_dir, 2) / "decisions.jsonl"
    path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(IntegrityError, match="incomplete or changed"):
        run(h)


def test_changed_setup_and_provider_failure_do_not_silently_resume(tmp_path):
    h = harness(tmp_path)
    run(h)
    h[0].starts[0].x0 = "different question"
    with pytest.raises(RuntimeError, match="different setup"):
        run(h)

    def fail(req):
        raise RuntimeError("unknown provider error")

    h2 = harness(tmp_path / "second", responder=fail)
    with pytest.raises(RuntimeError, match="unknown provider"):
        run(h2)
    assert list(read_jsonl(h2[0].run_dir / "decisions.jsonl")) == []


def test_plans_and_namespace(tmp_path):
    plan = plan_for_cohort(["q1", "q2"], "e_incorrect", original_assignment={"q1": "A1", "q2": "B"})
    assert Arm.B in plan["q1"] and Arm.A1 not in plan["q1"]
    assert Arm.A1 in plan["q2"] and Arm.B not in plan["q2"]
    with pytest.raises(IntegrityError):
        plan_for_cohort(["q1", "q1"], "e_correct")
    h = harness(tmp_path)
    run(h)
    meta = json.loads((h[0].run_dir / "ext_meta.json").read_text(encoding="utf-8"))
    assert meta["namespace"] == "math_exp2ext_smoke" and meta["smoke"]


def test_t1_reuse_requires_explicit_matching_identity(tmp_path):
    h = harness(tmp_path, arms=(Arm.A1,))
    setup = h[0]
    setup.cohort = "e_incorrect"
    setup.smoke = False
    setup.source_identity = {"source_root": str(tmp_path / "original")}
    setup.initial_grades = {"q1": {"status": "incorrect", "task_success": False}}
    setup.scoring_identity = {"shape": 4096, "prompt_sha256": "fixture", "source_sha256": "source"}
    key = text_key("q1", "same")
    pair = h[3]([("q1", "question 1", "same", key)])[0][key]
    export = tmp_path / "reuse.json"
    export.write_text(
        json.dumps(
            {
                "scoring_identity": setup.scoring_identity,
                "states": [
                    {
                        "x0_id": "q1",
                        "problem": "question 1",
                        "text": "same",
                        "scores": [r.model_dump(mode="json") for r in pair],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    setup.t1_reuse = T1Reuse(export, file_hash(export))
    with pytest.raises(IntegrityError, match="explicit approval"):
        run(h)
    setup.t1_reuse = replace(setup.t1_reuse, approved=True)
    before = len(h[3].calls)
    run(h)
    assert len(h[3].calls) == before
    setup.scoring_identity["shape"] = 1408
    with pytest.raises(IntegrityError, match="scoring identity differs"):
        run(h)


def test_live_and_output_overlap_refused_before_calls(tmp_path):
    h = harness(tmp_path)
    h[0].execution_mode = "live"
    with pytest.raises(IntegrityError, match="--confirm"):
        run(h)
    assert not (h[0].run_dir / "ext_meta.json").exists() and not h[2].calls
    h[0].execution_mode = "offline_stub"
    h[0].source_identity = {"source_root": str(tmp_path)}
    with pytest.raises(IntegrityError, match="overlaps"):
        run(h)
    assert not (h[0].run_dir / "ext_meta.json").exists() and not h[2].calls


@pytest.mark.parametrize("mode", ["accept", "parse_fail", "overlength"])
def test_grades_flow_through_shared_math_extension_analysis(tmp_path, mode):
    from analysis.ext_contrasts import ExtAnalysisError, exp2_finals

    response = {
        "accept": '{"action":"accept","revised_text":null}',
        "parse_fail": "invalid JSON",
        "overlength": '{"action":"rewrite","revised_text":"long"}',
    }[mode]
    h = harness(
        tmp_path,
        arms=(Arm.A1,),
        responder=lambda r: response,
        scorer=Scorer(overlong=mode == "overlength"),
    )
    trajectories = run(h)
    rows = grade_math_exp2(
        h[0],
        trajectories,
        lambda qid, text: {"status": "unresolved", "task_success": False},
        e_identity={"fixture": "PRIVATE_REFERENCE_ONLY_IN_E"},
    )
    exported = list(read_jsonl(h[0].run_dir / "trajectories.jsonl"))
    assert exported == rows and len(exported) == 1
    assert exported[0]["x_T"] == trajectories[0].x_t
    # Smoke output must never enter real comparison tables.
    with pytest.raises(ExtAnalysisError):
        exp2_finals("e_correct", [h[0].run_dir], base=tmp_path, prefix="math_exp2ext")
    assert all("PRIVATE_REFERENCE" not in str(req.messages) for req in h[2].calls)


def test_mid_round_restart_reuses_core_cache_and_question_score_store(tmp_path):
    from pipelines.math_exp2_loop import TextScoreStore

    attempts = []

    def respond(req):
        attempts.append(req)
        if len(attempts) == 2:
            raise RuntimeError("process interrupted between decisions")
        return '{"action":"accept","revised_text":null}'

    h = harness(tmp_path, n=2, responder=respond)
    batches = []

    class V:
        def score_batch(self, inputs):
            batches.append(inputs)
            return [{"error_score": 0.2} for _ in inputs]

    def store():
        return TextScoreStore(tmp_path / "scores", [("v1", "v1"), ("v2", "v1")], lambda i: V())

    h[0].score_texts = store()
    with pytest.raises(RuntimeError, match="between decisions"):
        run(h)
    assert len(batches) == 2 and len(attempts) == 2
    h[0].score_texts = store()  # recreate durable scorer, as a restarted process would
    run(h)
    assert len(batches) == 2  # neither V is executed again
    assert len(attempts) == 5  # 4 distinct decisions + one failed call; first is cached
    assert any(r["cached"] for r in h[1].ledger.rows())


def test_nonfake_provider_refused_before_output(tmp_path):
    h = harness(tmp_path)
    client = LLMClient({"other": object()})
    with pytest.raises(IntegrityError, match="FakeProvider"):
        run_math_exp2(client, config(), config(True), h[0])
    assert not (h[0].run_dir / "ext_meta.json").exists()


@pytest.mark.parametrize(
    "changed",
    [
        "math_judgments.jsonl",
        "math_results.json",
        "trajectories.jsonl",
        "e_identity.json",
        "rounds/round_4/COMPLETE.json",
    ],
)
def test_completed_grade_tamper_refused_without_replacing_marker(tmp_path, changed):
    h = harness(tmp_path)
    trs = run(h)

    def grade(qid, text):
        return {"status": "correct", "task_success": True}

    grade_math_exp2(h[0], trs, grade, e_identity={"fixture": 1})
    marker = h[0].run_dir / "judge_complete.json"
    before = marker.read_bytes()
    path = h[0].run_dir / changed
    # Whitespace keeps JSON semantics intact but changes its sealed bytes.
    path.write_bytes(path.read_bytes() + b" \n")
    with pytest.raises(IntegrityError, match="changed"):
        grade_math_exp2(h[0], trs, grade, e_identity={"fixture": 1})
    assert marker.read_bytes() == before


@pytest.mark.parametrize("change", ["qid", "key", "status", "success", "duplicate"])
def test_partial_grade_resume_validates_persisted_rows(tmp_path, change):
    h = harness(tmp_path, n=2, responder=lambda r: '{"action":"rewrite","revised_text":"new"}')
    trs = run(h)

    def fail_second(qid, text):
        if qid == "q2":
            raise RuntimeError("E interrupted")
        return {"status": "correct", "task_success": True}

    with pytest.raises(RuntimeError):
        grade_math_exp2(h[0], trs, fail_second, e_identity={"fixture": 1})
    path = h[0].run_dir / "math_judgments.jsonl"
    rows = list(read_jsonl(path))
    assert len(rows) == 3
    if change == "qid":
        rows[0]["x0_id"] = "q2"
    elif change == "key":
        rows[0]["text_key"] = "unrequested"
    elif change == "status":
        rows[0]["grade"]["status"] = "timeout"
    elif change == "success":
        rows[0]["grade"]["task_success"] = False
    else:
        rows *= 2
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    calls = []
    with pytest.raises(IntegrityError):
        grade_math_exp2(h[0], trs, lambda q, t: calls.append((q, t)), e_identity={"fixture": 1})
    assert calls == [] and not (h[0].run_dir / "judge_complete.json").exists()


def test_feedback_file_change_during_loop_refused(tmp_path, monkeypatch):
    from rafr.formatter import math_feedback

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name in math_feedback.PROMPT_FILES:
        (prompts / name).write_bytes((math_feedback.PROMPT_DIR / name).read_bytes())
    monkeypatch.setattr(math_feedback, "PROMPT_DIR", prompts)
    h = harness(tmp_path)

    def changed(t):
        if t == 1:
            (prompts / "a1.txt").write_text("changed", encoding="utf-8")

    with pytest.raises(IntegrityError, match="feedback changed"):
        run(h, on_round=changed)


def test_fresh_swapped_score_rows_refused(tmp_path):
    h = harness(tmp_path, n=2)
    scorer = h[3]

    def swapped(items):
        rows, over = scorer(items)
        keys = list(rows)
        rows[keys[0]], rows[keys[1]] = rows[keys[1]], rows[keys[0]]
        return rows, over

    h[0].score_texts = swapped
    with pytest.raises(IntegrityError, match="score state key"):
        run(h)
    assert h[2].calls == []


def test_t1_export_with_wrong_state_key_refused(tmp_path):
    h = harness(tmp_path, arms=(Arm.A1,))
    setup = h[0]
    setup.cohort = "e_incorrect"
    setup.smoke = False
    setup.source_identity = {"source_root": str(tmp_path / "original")}
    setup.initial_grades = {"q1": {"status": "incorrect", "task_success": False}}
    setup.scoring_identity = {"source_sha256": "fixture", "shape": 4096}
    pair = h[3]([("q1", "question 1", "same", "wrong-key")])[0]["wrong-key"]
    path = tmp_path / "export.json"
    path.write_text(
        json.dumps(
            {
                "scoring_identity": setup.scoring_identity,
                "states": [
                    {
                        "x0_id": "q1",
                        "problem": "question 1",
                        "text": "same",
                        "scores": [r.model_dump(mode="json") for r in pair],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    setup.t1_reuse = T1Reuse(path, file_hash(path), approved=True)
    with pytest.raises(IntegrityError, match="t1 score state key"):
        run(h)
    assert h[2].calls == []


def test_initial_grades_reused_without_calling_e_and_identity_frozen(tmp_path):
    h = harness(tmp_path)
    trs = run(h)

    def never(qid, text):
        raise AssertionError("original x1 must not be rejudged")

    rows = grade_math_exp2(h[0], trs, never, e_identity={"fixture": 1})
    assert all(row["initial_status"] == row["terminal_status"] == "correct" for row in rows)
    h[0].initial_grades["q1"] = {"status": "correct", "task_success": True, "n_boxes": 1}
    with pytest.raises(IntegrityError, match="setup differs"):
        grade_math_exp2(h[0], trs, never, e_identity={"fixture": 1})


@pytest.mark.parametrize("grade", [{}, {"q1": {"status": "incorrect", "task_success": False}}])
def test_missing_or_wrong_cohort_original_grade_refused(tmp_path, grade):
    h = harness(tmp_path)
    h[0].initial_grades = grade
    if grade:
        with pytest.raises(IntegrityError, match="initial grade"):
            run(h)
        assert h[2].calls == []
    else:
        trajectories = run(h)  # Public pod loop does not need labels.
        with pytest.raises(IntegrityError, match="initial grade"):
            grade_math_exp2(
                h[0], trajectories, lambda *a: pytest.fail("E called"), e_identity={"fixture": 1}
            )


def test_client_cache_outside_extension_run_refused(tmp_path):
    from pipelines.math_ext_exp1_runner import Exp1ExtError

    h = harness(tmp_path)
    c = LLMClient({"fake": h[2]}, LLMCache(tmp_path / "outside-cache"))
    with pytest.raises(Exp1ExtError, match="inside the extension run"):
        run_math_exp2(c, config(), config(True), h[0])
    assert h[2].calls == []


@pytest.mark.parametrize("kind", ["original_root", "pipeline_subclass", "other_run", "identity"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_e_context_is_bound_to_this_run_before_any_judge_output(tmp_path, kind, wrapped):
    from functools import partial

    class Grader:
        def grade(self, *args):
            pytest.fail("unbound E must not run")

    h = harness(tmp_path, arms=(Arm.NO_VERIFIER,))
    h[0].source_identity = {"source_root": str(tmp_path / "original")}
    trajectories = run(h)
    grader = Grader()
    if kind == "original_root":
        grader.root = tmp_path / "original"
    elif kind == "pipeline_subclass":
        Grader.__module__ = "pipelines.math_pipeline"

        class Subclass(Grader):
            pass

        grader = Subclass()
    elif kind == "other_run":
        grader.root = tmp_path / "other_extension"
    else:
        grader.root = h[0].run_dir
        grader.identity = {"fixture": "a different grader"}
    callback = partial(grader.grade) if wrapped else grader.grade
    before = {p: p.read_bytes() for p in h[0].run_dir.rglob("*") if p.is_file()}
    with pytest.raises(IntegrityError):
        grade_math_exp2(h[0], trajectories, callback, e_identity={"fixture": 1})
    assert before == {p: p.read_bytes() for p in h[0].run_dir.rglob("*") if p.is_file()}


def test_post_loop_e_rejects_cross_run_round_even_with_self_consistent_hashes(tmp_path):
    import shutil

    from rafr.experiments.exp2 import load_round

    current = harness(tmp_path / "current")
    donor = harness(tmp_path / "donor")
    donor[0].run_id = "foreign-run"
    run(current)
    run(donor)
    target = current[0].run_dir / "rounds/round_1"
    shutil.rmtree(target)
    shutil.copytree(donor[0].run_dir / "rounds/round_1", target)
    with pytest.raises(IntegrityError, match="different run/state"):
        grade_math_exp2(
            current[0],
            load_round(current[0].run_dir, 4),
            lambda *args: pytest.fail("E must not run on another run's rounds"),
            e_identity={"stub": True},
        )
    assert not (current[0].run_dir / "math_judgments.jsonl").exists()


def test_complete_resume_rejects_changed_verifier_revision(tmp_path):
    from tests.test_math_ext_round_integrity import rewrite_round

    h = harness(tmp_path)
    run(h)
    directory = h[0].run_dir / "rounds/round_1"
    rows = list(read_jsonl(directory / "scores.jsonl"))
    rows[0]["model_version"] = "another-revision"
    rewrite_round(directory, "scores.jsonl", rows)
    before = len(h[2].calls)
    with pytest.raises(IntegrityError, match="verifier version"):
        run(h)
    assert len(h[2].calls) == before
