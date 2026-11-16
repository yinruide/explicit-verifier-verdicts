"""Math extension request-boundary tests using original math artifacts made OFFLINE."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pipelines.math_ext_data import MathSource, prepare_calibration, prepare_correct_cohort
from pipelines.math_ext_exp1 import (
    BoundScoreIdentity,
    check_client_output_paths,
    reconstruct_original_channel,
    run_math_exp1_extension,
)
from pipelines.math_ext_exp1_runner import (
    E_CORRECT,
    E_INCORRECT,
    Exp1ExtError,
    Exp1ExtSetup,
    run_exp1_ext,
)
from tests.math_v2_helpers import make_harness, prepare_gate

from rafr.calibration.apply import Calibrator, file_sha256
from rafr.contracts.schemas import (
    Arm,
    EvalRecord,
    Lambda,
    MathLabel,
    Observation,
    Split,
    State,
    text_hash,
)
from rafr.controller import decide as ctrl
from rafr.data.labels import LabelPurpose
from rafr.domains.math.generation import build_controller_config
from rafr.domains.math.storage import IntegrityError
from rafr.experiments.ext_cohort import CohortError, read_cohort, write_cohort
from rafr.formatter.extended import PiHat, format_extended
from rafr.formatter.math_extended import (
    build_no_verifier_config,
    math_wording,
    original_feedback,
)
from rafr.llm.client import LLMClient
from rafr.llm.providers.fake import FakeProvider
from rafr.logging.writer import read_jsonl


def bind(root):
    names = {
        "states.jsonl",
        "initial_complete.json",
        "splits_complete.json",
        "scores.jsonl",
        "fit.json",
        "snapshot.json",
    }
    for marker in ("initial_complete.json", "splits_complete.json"):
        names.update(json.loads((root / marker).read_text(encoding="utf-8"))["files"])
    names.update(
        f"exp1/{n}"
        for n in ("channel.jsonl", "decisions.jsonl", "permutation.json", "COMPLETE.json")
    )
    return {name: file_sha256(root / name) for name in names}


def source_for(h):
    return MathSource(
        h.root,
        expected_sha256=bind(h.root),
        master_seed=h.pipeline.config.seed,
        verifier_ids=("fixture-V1", "fixture-V2"),
    )


def bound_correct_cohort(source, tmp_path, n=3):
    prepared = prepare_correct_cohort(source, tmp_path / "cohort", n=n, stage="offline_stub")
    path, sha = Path(prepared["cohort_file"]), prepared["cohort_sha256"]
    _, ids = read_cohort(path, sha)
    return path, sha, ids


@pytest.fixture
def original(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch)
    prepare_gate(h)
    h.pipeline.exp1()
    return h


def obs(sid="s", s1=0.75, s2=0.2):
    return Observation(
        state_id=sid,
        s1_obs=s1,
        s2_obs=s2,
        tau1=0.5,
        tau2=0.5,
        lam=Lambda.LOW,
        raw_s1=s1,
        raw_s2=s2,
        v1_id="v1",
        v2_id="v2",
    )


def test_wording_guards_and_exact_math_prefix():
    wording = math_wording()
    build_no_verifier_config(17, 1)
    o = obs()
    b = original_feedback(Arm.B, o)
    shown = format_extended(Arm.B_PIHAT, o, wording, pihat=PiHat(0.1, 0.2))
    assert shown.startswith(b + "\n\n")
    verdict = format_extended(Arm.VERDICT_ONLY, o, wording)
    assert verdict.startswith(original_feedback(Arm.A1, o))
    assert "distance from" not in verdict and "Distances" not in verdict
    assert "Checker A flags this text: yes" in verdict
    donor = obs("donor", 0.1, 0.8)
    permuted = original_feedback(Arm.PERMUTED_B, o, donor)
    assert permuted.startswith(original_feedback(Arm.A1, o))
    assert "Checker A flags this text: no" in permuted
    with pytest.raises(ValueError):
        original_feedback(Arm.PERMUTED_B, o, o)
    with pytest.raises(ValueError):
        original_feedback(Arm.A1, o, donor)


def test_original_math_replay_requests_and_resume(original, tmp_path):
    source = source_for(original)
    before = bind(original.root)
    states = source.states(Split.D_EXP1)
    rows, evidence = reconstruct_original_channel(source, states)
    assert len(rows) == 18 and evidence["n_original_decisions"] == 54
    assert evidence["request_prompts_and_seeds_identical"]
    assert all("arm" not in row for row in rows)
    provider = FakeProvider(lambda req: '{"action":"accept","revised_text":""}')
    client = LLMClient({"openai": provider})
    kw = dict(
        run_dir=tmp_path / "extension",
        run_id="offline",
        cohort=E_INCORRECT,
        arms=[Arm.NO_VERIFIER, Arm.VERDICT_ONLY],
        allow_draft=True,
        metadata={"review": "NOT_APPROVED_OFFLINE_ONLY"},
    )
    result = run_math_exp1_extension(client, source, **kw)
    assert result["n_rows"] == 36 and result["seed_namespace"] == "math_exp1"
    assert len(provider.calls) == 36
    orig_seeds = {
        r["state_id"]: r["attempts"][0]["seed"]
        for r in read_jsonl(original.root / "exp1/decisions.jsonl")
    }
    new_rows = list(read_jsonl(kw["run_dir"] / "decisions.jsonl"))
    assert all(r["attempts"][0]["seed"] == orig_seeds[r["state_id"]] for r in new_rows)
    for req in provider.calls:
        text = json.dumps(req.messages) + (req.system or "")
        assert "PRIVATE_GOLD_" not in text
        assert req.model == "gpt-5.6-terra" and req.decoding["reasoning_effort"] == "medium"
        assert req.decoding["max_output_tokens"] == 16384 and "temperature" not in req.decoding
        if "Checker output:" not in text:
            assert "Checker A" not in text and "CHECKER OUTPUT" not in text
            assert "{observation}" not in text and "Fixture Q" in text
    assert run_math_exp1_extension(client, source, **kw)["n_new_decisions"] == 0
    assert len(provider.calls) == 36
    with pytest.raises(Exp1ExtError, match="different setup"):
        run_math_exp1_extension(client, source, **{**kw, "metadata": {"review": "CHANGED"}})
    assert len(provider.calls) == 36 and bind(original.root) == before


@pytest.mark.parametrize("tamper", ["score", "donor", "prompt", "duplicate"])
def test_replay_detects_rebound_bad_original_before_calls(original, tamper):
    root = original.root / "exp1"
    if tamper == "donor":
        p = root / "permutation.json"
        value = json.loads(p.read_text(encoding="utf-8"))
        first = next(iter(value))
        value[first] = first
        p.write_text(json.dumps(value), encoding="utf-8")
    else:
        p = root / ("channel.jsonl" if tamper == "assignment" else "decisions.jsonl")
        rows = list(read_jsonl(p))
        if tamper == "score":
            rows[0]["s1_tilde"] += 0.000000000001
        elif tamper == "assignment":
            rows[0]["z"] = "different"
        elif tamper == "prompt":
            rows[0]["attempts"][0]["prompt_hash"] = "invented"
        else:
            rows.append(rows[0])
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        marker = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
        marker["channel_sha256" if tamper == "assignment" else "decisions_sha256"] = file_sha256(p)
        (root / "COMPLETE.json").write_text(json.dumps(marker), encoding="utf-8")
    source = source_for(
        original
    )  # The fixture consciously rebinds bytes; semantic replay still fails.
    with pytest.raises(Exp1ExtError):
        reconstruct_original_channel(source, source.states(Split.D_EXP1))


def test_correct_cohort_all_arms_and_no_private_label_request(original, tmp_path):
    source = source_for(original)
    fitted = prepare_calibration(source, tmp_path / "cal", n_boot=25, stage="offline_stub")
    cal = Calibrator.load(
        Path(fitted["calibrator_file"]), expected_sha256=fitted["calibrator_sha256"]
    )
    cohort_file, cohort_sha256, ids = bound_correct_cohort(source, tmp_path)
    provider = FakeProvider(lambda req: '{"action":"accept","revised_text":""}')
    result = run_math_exp1_extension(
        LLMClient({"openai": provider}),
        source,
        run_dir=tmp_path / "correct",
        run_id="stub",
        cohort=E_CORRECT,
        state_ids=ids,
        cohort_file=cohort_file,
        cohort_sha256=cohort_sha256,
        calibrator=cal,
        allow_draft=True,
    )
    assert result["n_rows"] == 63 and result["seed_namespace"] == "math_exp1_ecorrect"
    assert all("PRIVATE_GOLD_" not in json.dumps(r.messages) for r in provider.calls)
    rows = list(read_jsonl(tmp_path / "correct/decisions.jsonl"))
    own = {(r["state_id"], r["lam"]): r["fields"] for r in rows if r["arm"] == "B"}
    for row in rows:
        if row["arm"] == "permuted_B":
            assert row["fields"] == own[row["permuted_from_state_id"], row["lam"]]
    cfg = build_controller_config(source.master_seed, 1)
    for row in rows:
        labels = ctrl.DecisionLabels(
            row["state_id"], Arm(row["arm"]), Lambda(row["lam"]), 1, "math_exp1_ecorrect"
        )
        assert row["attempts"][0]["seed"] == ctrl.call_seed(cfg, labels)


@pytest.mark.parametrize(
    "fault", ["missing", "hash", "changed_file", "different_ids", "smoke_overlap", "wrong_label"]
)
def test_correct_selection_refuses_before_outputs_or_calls(original, tmp_path, fault):
    source = source_for(original)
    path, sha, ids = bound_correct_cohort(source, tmp_path)
    kwargs = dict(cohort_file=path, cohort_sha256=sha, state_ids=ids)
    if fault == "missing":
        kwargs.pop("cohort_file")
    elif fault == "hash":
        kwargs["cohort_sha256"] = "0" * 64
    elif fault == "changed_file":
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="\n")
    elif fault == "different_ids":
        kwargs["state_ids"] = ids[:-1]
    elif fault == "smoke_overlap":
        kwargs["smoke"] = True
    elif fault == "wrong_label":
        incorrect = next(
            y.state_id
            for y in source.labels(Split.D_GATE, LabelPurpose.ECORRECT_COHORT)
            if y.label is MathLabel.INCORRECT
        )
        bad = tmp_path / "wrong_cohort.jsonl"
        kwargs.update(
            cohort_file=bad,
            cohort_sha256=write_cohort(bad, [incorrect], {"domain": "math"}),
            state_ids=[incorrect],
        )
    fp = FakeProvider()
    run_dir = tmp_path / "rejected"
    with pytest.raises((IntegrityError, CohortError, Exp1ExtError)):
        run_math_exp1_extension(
            LLMClient({"openai": fp}),
            source,
            run_dir=run_dir,
            run_id="stub",
            cohort=E_CORRECT,
            arms=[Arm.NO_VERIFIER],
            **kwargs,
        )
    assert not fp.calls
    assert not run_dir.exists()


def test_correct_smoke_excludes_frozen_sample_and_binds_resume_identity(original, tmp_path):
    source = source_for(original)
    path, sha, ids = bound_correct_cohort(source, tmp_path)
    smoke_ids = [
        y.state_id
        for y in source.labels(Split.D_GATE, LabelPurpose.EXT_SMOKE)
        if y.label is MathLabel.CORRECT and y.state_id not in ids
    ][:2]
    assert len(smoke_ids) == 2
    fp = FakeProvider(lambda req: '{"action":"accept","revised_text":""}')
    run_dir = tmp_path / "correct_smoke"
    result = run_math_exp1_extension(
        LLMClient({"openai": fp}),
        source,
        run_dir=run_dir,
        run_id="stub",
        cohort=E_CORRECT,
        arms=[Arm.NO_VERIFIER],
        state_ids=smoke_ids,
        smoke=True,
        cohort_file=path,
        cohort_sha256=sha,
    )
    assert result["n_rows"] == 6
    assert len(fp.calls) == 6
    assert result["seed_namespace"] == "math_exp1ext_smoke"
    meta = json.loads((run_dir / "ext_meta.json").read_text(encoding="utf-8"))
    assert meta["ecorrect_cohort_sha256"] == sha


def test_correct_cohort_file_supplies_exact_order_when_ids_omitted(original, tmp_path):
    source = source_for(original)
    path, sha, ids = bound_correct_cohort(source, tmp_path)
    fp = FakeProvider(lambda req: '{"action":"accept","revised_text":""}')
    run_dir = tmp_path / "correct_from_file"
    result = run_math_exp1_extension(
        LLMClient({"openai": fp}),
        source,
        run_dir=run_dir,
        run_id="stub",
        cohort=E_CORRECT,
        arms=[Arm.NO_VERIFIER],
        cohort_file=path,
        cohort_sha256=sha,
    )
    assert result["n_rows"] == 9
    records = list(read_jsonl(run_dir / "decisions.jsonl"))
    assert [r["state_id"] for r in records if r["lam"] == "low"] == ids


def test_state_callback_separates_identical_texts(tmp_path):
    class Identity:
        name = "identity"

        def __call__(self, state_id, a, b, lam):
            return a, b

    states = [
        State(state_id=f"s{i}", x0_id=f"q{i}", x0=f"different problem {i}", text="same solution")
        for i in range(2)
    ]
    records = {
        s.state_id: EvalRecord(
            text_id=text_hash(s.text),
            text=s.text,
            label=MathLabel.CORRECT,
            model="offline",
            model_version=None,
            prompt_hash="fixture",
            e_prompt_sha256="fixture",
            e_version="fixture",
            decoding={},
        )
        for s in states
    }
    pairs = {"s0": (0.1, 0.2), "s1": (0.8, 0.9)}
    setup = Exp1ExtSetup(
        run_id="stub",
        run_dir=tmp_path / "duplicate-text",
        cohort=E_CORRECT,
        states=states,
        scorer=BoundScoreIdentity("a", "b", "1", "2"),
        tau1=0.5,
        tau2=0.5,
        channel=Identity(),
        lambdas=[Lambda.LOW],
        arms=[Arm.A1],
        wording=math_wording(allow_draft=True),
        e_labels=records,
        expected_label=MathLabel.CORRECT,
        score_states=lambda ss: [pairs[s.state_id] for s in ss],
        original_feedback=original_feedback,
        seed_namespace="math_exp1_ecorrect",
    )
    provider = FakeProvider(lambda req: '{"action":"accept","revised_text":""}')
    run_exp1_ext(LLMClient({"openai": provider}), build_controller_config(17, 1), None, setup)
    rows = list(read_jsonl(setup.run_dir / "decisions.jsonl"))
    assert [r["s1"] for r in rows] == [0.1, 0.8]
    assert provider.calls[0].prompt_hash() != provider.calls[1].prompt_hash()


def test_draft_cannot_reach_nonfake_provider(original, tmp_path):
    class NotFake:
        def complete(self, request):
            raise AssertionError("provider must never be reached")

    with pytest.raises(Exp1ExtError, match="FakeProvider"):
        run_math_exp1_extension(
            LLMClient({"openai": NotFake()}),
            source_for(original),
            run_dir=tmp_path / "bad",
            run_id="bad",
            cohort=E_INCORRECT,
            allow_draft=True,
        )
    assert not (tmp_path / "bad").exists()


def test_reviewed_fixture_slots_and_no_verifier_request(tmp_path, monkeypatch):
    from rafr.formatter import math_extended
    from rafr.formatter.math_extended import PROMPT_DIR, PROMPT_FILES, SHARED_DIR, SHARED_FILES

    directory = tmp_path / "reviewed_test_fixture_not_real_approval"
    directory.mkdir()
    for name in PROMPT_FILES:
        text = (PROMPT_DIR / name).read_text(encoding="utf-8")
        if text.startswith("PLACEHOLDER"):
            text = text.split("\n\n", 1)[1]
        (directory / name).write_text(text, encoding="utf-8")
    wording = math_wording(directory)
    assert "PLACEHOLDER" not in wording.pihat_template
    cfg = build_no_verifier_config(17, 4, directory)
    assert "at most 4 decisions" in cfg.system_prompt
    user = ctrl.render_user_turn(cfg, "problem contains {observation}", "solution", "MUST_NOT_SHOW")
    assert "problem contains {observation}" in user  # literal public input survives one-pass render
    assert "MUST_NOT_SHOW" not in user and "Checker output" not in user
    shared = tmp_path / "shared"
    shared.mkdir()
    for name in SHARED_FILES:
        (shared / name).write_bytes((SHARED_DIR / name).read_bytes())
    monkeypatch.setattr(math_extended, "SHARED_DIR", shared)
    pihat = shared / "pihat.txt"
    pihat.write_text(
        pihat.read_text(encoding="utf-8") + "\n{private_reference}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unfilled"):
        math_wording(directory)


def test_wrapper_output_and_execution_mode_guards(original, tmp_path):
    source = source_for(original)
    provider = FakeProvider()
    client = LLMClient({"openai": provider})
    kwargs = dict(run_id="stub", cohort=E_INCORRECT, arms=[Arm.VERDICT_ONLY], allow_draft=True)
    with pytest.raises(Exp1ExtError, match="live is not authorized"):
        run_math_exp1_extension(
            client, source, run_dir=tmp_path / "live", execution_mode="live", **kwargs
        )
    with pytest.raises(IntegrityError, match="overlaps"):
        run_math_exp1_extension(client, source, run_dir=source.root / "extension", **kwargs)
    assert not provider.calls
    ids = [
        y.state_id
        for y in source.labels(Split.UNLABELED_POOL, LabelPurpose.DRY_RUN)
        if y.label is MathLabel.INCORRECT
    ][:2]
    result = run_math_exp1_extension(
        client, source, run_dir=tmp_path / "smoke", smoke=True, state_ids=ids, **kwargs
    )
    assert result["seed_namespace"] == "math_exp1ext_smoke" and result["n_rows"] == 6
    assert result["original_replay"] is None
    assert not (source.root / "extension").exists()


def test_client_cache_ledger_and_transient_outputs_are_run_local(tmp_path):
    from rafr.llm.cache import LLMCache
    from rafr.llm.ledger import Ledger

    run_dir = tmp_path / "extension"
    fp = FakeProvider()
    valid = LLMClient(
        {"openai": fp},
        cache=LLMCache(run_dir / "cache"),
        ledger=Ledger(run_dir / "ledger.jsonl"),
        transient_log=run_dir / "transient.jsonl",
    )
    check_client_output_paths(valid, run_dir)
    wrong = (
        LLMClient({"openai": fp}, cache=LLMCache(tmp_path / "original_cache")),
        LLMClient({"openai": fp}, ledger=Ledger(tmp_path / "original_ledger.jsonl")),
        LLMClient({"openai": fp}, transient_log=tmp_path / "original_transient.jsonl"),
    )
    for client in wrong:
        with pytest.raises(Exp1ExtError, match="inside the extension"):
            check_client_output_paths(client, run_dir)
    assert not fp.calls
    assert not (tmp_path / "original_ledger.jsonl").exists()
