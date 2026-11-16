"""Adversarial execution-order checks for Appendix M and Core step 8/12."""

import pytest

from math_v2_helpers import make_harness, prepare_fit, prepare_gate
from rafr.contracts.schemas import Split
from rafr.data import labels
from rafr.domains.math.storage import IntegrityError, atomic_json, file_hash, read_json
from rafr.logging.writer import read_jsonl


def test_unreachable_channel_stops_before_heldout_labels_and_does_not_unlock_reserve(
    tmp_path, monkeypatch
):
    h = make_harness(tmp_path, monkeypatch, delta_r=0.95)
    prepare_fit(h)
    accesses = []
    original = h.pipeline._labels

    def track(split, purpose):
        accesses.append((split, purpose))
        return original(split, purpose)

    monkeypatch.setattr(h.pipeline, "_labels", track)
    report = h.pipeline.channel_check()
    assert report["status"] == "CHANNEL_CHECK_FAILED" and report["passed"] is False
    assert all(row["expected_target_shortfall"] > 0 for row in report["per_verifier"].values())
    assert report["realized_ladder"]["ci"]
    with pytest.raises(IntegrityError, match="before held-out gate"):
        h.pipeline.gate()
    assert accesses == [(Split.D_FIT, labels.LabelPurpose.CHANNEL_CHECK)] * 2
    assert not (h.root / "gate_attempts.json").exists()
    assert not (h.root / "gate_1.json").exists()
    assert not labels.reserve_unlocked(h.root / "gate_attempts.json")
    assert "gate" not in read_json(h.root / "snapshot.json")["math_phase_bindings"]


def test_delta_may_be_reviewed_from_d_fit_then_freezes_before_gate_labels(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch, delta_r=0.95)
    prepare_fit(h)
    assert not h.pipeline.channel_check()["passed"]
    h.pipeline.config.delta_r = 0.05
    h.reopen()
    assert h.pipeline.channel_check()["passed"]
    original = h.pipeline._labels

    def check_binding(split, purpose):
        if split is Split.D_GATE:
            bound = read_json(h.root / "snapshot.json")["math_phase_bindings"]["gate"]
            assert bound["delta_r"] == 0.05
            assert read_json(h.root / "channel_check.json")["passed"]
        return original(split, purpose)

    monkeypatch.setattr(h.pipeline, "_labels", check_binding)
    assert h.pipeline.gate()["passed"]
    h.pipeline.config.delta_r = 0.04
    with pytest.raises(IntegrityError, match="frozen gate configuration changed"):
        h.reopen()


def test_formal_exp2_requires_intact_exp1_before_any_controller_call(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch)
    prepare_gate(h)
    calls = len(h.provider.calls)
    # Exercise the formal sequencing guard using the small offline cohort. No
    # real-run configuration or live boundary is created by this test.
    h.pipeline.config.stage = "formal"
    with pytest.raises(IntegrityError, match="COMPLETE.json"):
        h.pipeline.exp2()
    assert len(h.provider.calls) == calls
    h.pipeline.config.stage = "stub"
    h.pipeline.exp1()
    assert h.pipeline.require_exp1()["n_decisions"] == 54
    path = h.root / "exp1/decisions.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(IntegrityError, match="unchanged completed Exp1"):
        h.pipeline.require_exp1()


def test_formal_exp1_requires_full_diagnostics_before_controller_calls(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch)
    prepare_gate(h)
    calls = len(h.provider.calls)
    h.pipeline.config.stage = "formal"
    with pytest.raises(IntegrityError, match="COMPLETE.json"):
        h.pipeline.exp1()
    assert len(h.provider.calls) == calls


def test_round_checkpoints_include_the_core_ledger_calls_for_that_round(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch, mode="fourth_rewrite")
    prepare_gate(h)
    before = len(h.client.ledger.rows())
    h.pipeline.exp2()
    all_rows = []
    for t in range(1, 5):
        rows = list(read_jsonl(h.root / f"exp2/rounds/round_{t}/ledger.jsonl"))
        assert len(rows) == 6
        assert all(row["purpose"] == "controller" for row in rows)
        all_rows.extend(rows)
    assert all_rows == h.client.ledger.rows()[before:]
    h.reopen()
    h.pipeline.exp2()
    assert h.client.ledger.rows()[before:] == all_rows


def test_completed_gate_recovers_interrupted_core_history_without_reading_labels_again(
    tmp_path, monkeypatch
):
    from rafr.validity import gate

    h = make_harness(tmp_path, monkeypatch)
    prepare_fit(h)
    original = gate.record_attempt

    def interrupt(*args, **kwargs):
        raise OSError("interrupted before Core attempt publication")

    monkeypatch.setattr(gate, "record_attempt", interrupt)
    with pytest.raises(OSError, match="publication"):
        h.pipeline.gate()
    completed = read_json(h.root / "gate_1.json")
    assert not (h.root / "gate_attempts.json").exists()
    monkeypatch.setattr(gate, "record_attempt", original)
    h.reopen()
    calls = len(h.provider.calls)

    def forbid_labels(*args, **kwargs):
        raise AssertionError("completed gate must not inspect labels again")

    monkeypatch.setattr(h.pipeline, "_labels", forbid_labels)
    assert h.pipeline.gate() == completed
    history = read_json(h.root / "gate_attempts.json")
    assert len(history["attempts"]) == 1
    entry = history["attempts"][0]
    assert entry["gate_report_sha256"] == file_hash(h.root / "gate_1.json")
    assert entry["descriptive_whole_split"] == completed["descriptive"]
    assert len(h.provider.calls) == calls
    h.pipeline.require_gate()
    entry["gate_report_sha256"] = "0" * 64
    atomic_json(h.root / "gate_attempts.json", history)
    with pytest.raises(IntegrityError, match="contradicts"):
        h.pipeline.require_gate()
