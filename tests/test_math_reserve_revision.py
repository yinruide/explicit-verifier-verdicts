"""Approved reserve redesigns preserve cohort evidence and one Core attempt budget."""

from dataclasses import replace

import pytest
from pipelines.math_pipeline import MathPipeline
from pipelines.math_revision import initialize_revision, writer_lock_root

from math_v2_helpers import make_harness, prepare_fit
from rafr.contracts.schemas import Split
from rafr.domains.math.storage import IntegrityError, digest, file_hash, read_json
from rafr.validity.gate import GateError, next_attempt_split, read_attempts

DECISION = {
    "reviewed_by": "the authors (offline engineering fixture)",
    "decision": "Reviewed channel redesign for one reserve attempt; no new initial cohort.",
}


def failed_run(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch, delta_r=0.25)
    prepare_fit(h)
    assert h.pipeline.gate()["passed"] is False
    return h


def revision(h, root, **changes):
    config = h.pipeline.config.model_copy(deep=True, update={"delta_r": 0.05, **changes})
    return MathPipeline(
        config,
        root,
        h.pipeline.bundle,
        client=h.client,
        verifiers=h.pipeline.verifiers,
        grader=h.grader,
    )


def test_revision_reuses_exact_initial_cohort_without_new_g_or_e_calls(tmp_path, monkeypatch):
    h = failed_run(tmp_path, monkeypatch)
    child = revision(
        h,
        tmp_path / "reviewed-revision",
        verifiers=[
            {"model_id": "reviewed-new-V1", "revision": "offline-1"},
            {"model_id": "reviewed-new-V2", "revision": "offline-2"},
        ],
    )
    calls = len(h.provider.calls), len(h.grader.calls)
    first = read_attempts(h.root / "gate_attempts.json")[0]

    result = initialize_revision(child, h.root, DECISION)

    assert result["status"] == "RESERVE_REVISION_INITIALIZED"
    assert (len(h.provider.calls), len(h.grader.calls)) == calls
    snapshot = read_json(child.root / "snapshot.json")
    assert set(snapshot["math_phase_bindings"]) == {"generation", "splits"}
    marker = child._read_fixed(child.root / "reserve_revision.json")
    assert marker["first_attempt_sha256"] == digest(first)
    assert marker["history_path"] == str((h.root / "gate_attempts.json").resolve())
    assert not (child.root / "gate_attempts.json").exists()
    for name, checksum in marker["cohort_files"].items():
        assert file_hash(child.root / name) == file_hash(h.root / name) == checksum
    for forbidden in (
        "scores.jsonl",
        "fit.json",
        "cache",
        "exp1",
        "exp2_smoke",
        "channel_check.json",
    ):
        assert not (child.root / forbidden).exists()
    child.generate()
    child.split()
    for qid, state in child._states().items():
        assert child.grade_once(qid, state.text).status in {"correct", "incorrect"}
    assert (len(h.provider.calls), len(h.grader.calls)) == calls
    assert initialize_revision(child, h.root, DECISION) == result


@pytest.mark.parametrize(
    "decision", [None, {}, {"reviewed_by": "the authors"}, {"reviewed_by": " ", "decision": "yes"}]
)
def test_revision_requires_actual_explicit_decision(tmp_path, monkeypatch, decision):
    h = failed_run(tmp_path, monkeypatch)
    child = revision(h, tmp_path / "child")
    with pytest.raises(IntegrityError, match="explicit reviewed"):
        initialize_revision(child, h.root, decision)
    assert not (child.root / "reserve_revision.json").exists()


def test_revision_rejects_changed_seed_or_e_cohort_identity(tmp_path, monkeypatch):
    h = failed_run(tmp_path, monkeypatch)
    child = revision(h, tmp_path / "seed-change", seed=h.pipeline.config.seed + 1)
    with pytest.raises(IntegrityError, match="generation settings"):
        initialize_revision(child, h.root, DECISION)
    changed_bundle = replace(h.pipeline.bundle, manifest_sha256="different-source")
    child = MathPipeline(
        h.pipeline.config,
        tmp_path / "other-cohort",
        changed_bundle,
        client=h.client,
        verifiers=h.pipeline.verifiers,
        grader=h.grader,
    )
    with pytest.raises(IntegrityError, match="cohort, E"):
        initialize_revision(child, h.root, DECISION)


def test_revision_refuses_corrupted_parent_or_nonempty_target(tmp_path, monkeypatch):
    h = failed_run(tmp_path, monkeypatch)
    child = revision(h, tmp_path / "child")
    (child.root / "scores.jsonl").write_text("{}\n")
    with pytest.raises(IntegrityError, match="not fresh"):
        initialize_revision(child, h.root, DECISION)
    (child.root / "scores.jsonl").unlink()
    path = h.root / "states.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(IntegrityError, match="initial-state files changed"):
        initialize_revision(child, h.root, DECISION)


def test_revision_copy_resumes_after_interruption_without_overwriting_conflicts(
    tmp_path, monkeypatch
):
    from pipelines import math_revision

    h = failed_run(tmp_path, monkeypatch)
    child = revision(h, tmp_path / "child")
    original = math_revision._copy_once
    copied = []

    def interrupt(source, destination):
        original(source, destination)
        copied.append(destination)
        if len(copied) == 2:
            raise OSError("injected copy interruption")

    monkeypatch.setattr(math_revision, "_copy_once", interrupt)
    with pytest.raises(OSError, match="copy interruption"):
        initialize_revision(child, h.root, DECISION)
    assert not (child.root / "reserve_revision_complete.json").exists()
    calls = len(h.provider.calls), len(h.grader.calls)
    for action in (child.generate, child.split, lambda: child.score(Split.D_FIT)):
        with pytest.raises(IntegrityError, match="incomplete"):
            action()
    assert (len(h.provider.calls), len(h.grader.calls)) == calls
    monkeypatch.setattr(math_revision, "_copy_once", original)
    copied[0].write_text("conflicting partial target")
    with pytest.raises(IntegrityError, match="different"):
        initialize_revision(child, h.root, DECISION)
    copied[0].unlink()
    assert initialize_revision(child, h.root, DECISION)["next_gate_attempt"] == 2
    assert (child.root / "reserve_revision_complete.json").exists()
    with pytest.raises(IntegrityError, match="staged execution"):
        child.run_all()


@pytest.mark.parametrize("command", ["generate", "score"])
def test_incomplete_revision_runtime_stops_before_verifier_or_provider_creation(
    tmp_path, monkeypatch, command
):
    from pipelines import math_runtime

    from rafr.llm.providers import openai

    h = make_harness(tmp_path, monkeypatch, n=4)
    h.pipeline._save_fixed(h.root / "reserve_revision.json", {"incomplete_fixture": True})
    # Only bypass the live permission/data boundary, never the startup integrity guard.
    monkeypatch.setattr(math_runtime, "assert_command_allowed", lambda *a, **k: None)
    monkeypatch.setattr(math_runtime, "validate_output_root", lambda config, root: root)
    monkeypatch.setattr(math_runtime, "load_bundle", lambda config: h.pipeline.bundle)
    monkeypatch.setattr(math_runtime, "MathGrader", lambda *a, **k: h.grader)
    for module, name in ((math_runtime, "build_verifier"), (openai, "OpenAIProvider")):
        monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail("created external boundary"))
    with pytest.raises(IntegrityError, match="incomplete"):
        math_runtime.build_pipeline(h.pipeline.config, h.root, command=command)


def test_sibling_revisions_share_the_single_remaining_core_gate_attempt(tmp_path, monkeypatch):
    h = failed_run(tmp_path, monkeypatch)
    first, sibling = (revision(h, tmp_path / name) for name in ("first", "sibling"))
    initialize_revision(first, h.root, DECISION)
    initialize_revision(sibling, h.root, DECISION)
    assert writer_lock_root(first.root) == writer_lock_root(sibling.root) == h.root.resolve()
    assert writer_lock_root(tmp_path / "future", h.root) == h.root.resolve()
    assert first.gate_attempts_path == sibling.gate_attempts_path == h.root / "gate_attempts.json"
    for split in (Split.D_FIT, Split.UNLABELED_POOL, Split.D_GATE_RESERVE):
        first.score(split)
    first.fit()
    result = first.gate(2, reserve_decision=DECISION)
    assert result["attempt"] == 2
    assert len(read_attempts(first.gate_attempts_path)) == 2
    with pytest.raises(GateError, match="already attempted"):
        next_attempt_split(sibling.gate_attempts_path)
    with pytest.raises(IntegrityError):
        sibling.gate(2, reserve_decision=DECISION)
    fresh = revision(h, tmp_path / "after-budget-used")
    with pytest.raises(IntegrityError, match="exactly one failed"):
        initialize_revision(fresh, h.root, DECISION)
