"""Core storage/cache/resume in the mathematical pipeline; no parallel RunStore."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from math_v2_helpers import make_harness, prepare_gate
from rafr.data.labels import y_star_path
from rafr.domains.math import storage
from rafr.domains.math.storage import IntegrityError, atomic_json, read_json
from rafr.llm.cache import LLMCache
from rafr.llm.client import LLMClient, LLMRequest, TransientProviderError
from rafr.llm.ledger import Ledger
from rafr.llm.providers.fake import FakeProvider
from rafr.logging.writer import read_jsonl


def test_math_storage_contains_no_parallel_run_store():
    assert not hasattr(storage, "RunStore")


def test_core_cache_replays_same_prompt_and_ledger_counts_no_new_tokens(tmp_path):
    provider = FakeProvider(lambda request: "answer")
    ledger = Ledger(tmp_path / "ledger.jsonl")
    client = LLMClient({"offline": provider}, cache=LLMCache(tmp_path / "cache"), ledger=ledger)
    request = LLMRequest(
        "controller",
        "offline",
        "fixture",
        "fixture-v1",
        [{"role": "user", "content": "fixed input"}],
        seed=17,
    )
    first, second = client.complete(request), client.complete(request)
    assert first.cached is False and second.cached is True
    assert len(provider.calls) == 1
    assert ledger.totals.live_calls == 1 and ledger.totals.cached_calls == 1
    assert ledger.totals.input_tokens == 10 and ledger.totals.output_tokens == 5
    client.complete(replace(request, seed=18, attempt=1))
    assert len(provider.calls) == 2


def test_core_transient_retry_repeats_identical_request_without_parse_attempt(tmp_path):
    count, requests, delays = 0, [], []

    def responder(request):
        nonlocal count
        count += 1
        requests.append(request)
        if count == 1:
            raise TransientProviderError("network", "injected transient")
        return "answer"

    client = LLMClient({"offline": FakeProvider(responder)}, sleep=delays.append)
    request = LLMRequest(
        "controller",
        "offline",
        "fixture",
        "fixture-v1",
        [{"role": "user", "content": "fixed input"}],
        seed=17,
        network={"transient_retries": 1, "backoff_base_s": 0},
    )
    client.complete(request)
    assert len(requests) == 2 and requests[0] is requests[1]
    assert requests[0].attempt == requests[1].attempt == 0
    assert delays == [0.0] and client.transient_stats["retries"] == 1


def test_generation_interruption_resumes_core_states_without_duplicate_completed_calls(
    tmp_path, monkeypatch
):
    h = make_harness(tmp_path, monkeypatch, n=10)
    h.responder.crash_after = 3
    with pytest.raises(RuntimeError, match="interruption"):
        h.pipeline.generate()
    assert len(list(read_jsonl(h.root / "states.jsonl"))) == 3
    assert not (h.root / "initial_complete.json").exists()
    h.responder.crash_after = None
    h.reopen()
    result = h.pipeline.generate()
    assert result["n_valid"] == 10
    assert len(list(read_jsonl(h.root / "states.jsonl"))) == 10
    assert len(list(read_jsonl(h.root / "initial_outcomes.jsonl"))) == 10
    assert len(h.provider.calls) == 11  # One interrupted request plus ten successful states.


def _timeless(value):
    if isinstance(value, dict):
        return {k: _timeless(v) for k, v in value.items() if k not in {"created_at", "ts"}}
    if isinstance(value, list):
        return [_timeless(v) for v in value]
    return value


def _live_ledger_keys(path):
    return [row["cache_key"] for row in read_jsonl(path) if not row["cached"]]


def test_concurrent_generation_writes_what_a_serial_run_writes(tmp_path, monkeypatch):
    options = {
        "n": 12,
        "initial_failures": {1: "unresolved_once", 2: "unresolved_always", 3: "refusal_once"},
        "e_faults": {4: "timeout"},
    }
    runs = {}
    for name, workers in (("serial", 1), ("parallel", 4)):
        (tmp_path / name).mkdir()
        h = make_harness(tmp_path / name, monkeypatch, **options)
        summary = h.pipeline.generate(workers=workers)
        runs[name] = (h, summary)
    (serial, a), (parallel, b) = runs["serial"], runs["parallel"]
    assert {k: v for k, v in a.items() if k != "files"} == {
        k: v for k, v in b.items() if k != "files"
    }
    files = ["generation.jsonl", "evaluations.jsonl", "e_errors.jsonl", "states.jsonl"]
    files += ["initial_outcomes.jsonl", y_star_path(Path("labels")).as_posix()]
    for name in files:
        assert (serial.root / name).is_file(), name
        assert _timeless(list(read_jsonl(serial.root / name))) == _timeless(
            list(read_jsonl(parallel.root / name))
        ), name
    keys = [sorted(r.cache_key() for r in h.provider.calls) for h in (serial, parallel)]
    assert keys[0] == keys[1]
    for h in (serial, parallel):
        live = _live_ledger_keys(h.pipeline.root.parent / "ledger.jsonl")
        assert (
            len(live) == len(h.provider.calls) == len(list(read_jsonl(h.root / "generation.jsonl")))
        )
        assert len(set(live)) == len(live)


def test_concurrent_interruption_resumes_without_buying_any_request_twice(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch, n=10)
    h.responder.crash_after = 3
    with pytest.raises(RuntimeError, match="interruption"):
        h.pipeline.generate(workers=4)
    assert not (h.root / "initial_complete.json").exists()
    h.responder.crash_after = None
    h.reopen()
    assert h.pipeline.generate(workers=4)["n_valid"] == 10
    ids = [row["question_id"] for row in read_jsonl(h.root / "initial_outcomes.jsonl")]
    assert ids == [f"fixture:{number:03}" for number in range(10)]
    live = _live_ledger_keys(tmp_path / "ledger.jsonl")
    assert len(live) == len(set(live)) == 10  # each question's request bought exactly once


def test_resume_after_e_write_before_state_commit_uses_cached_grade_and_provider_response(
    tmp_path, monkeypatch
):
    h = make_harness(tmp_path, monkeypatch, n=3)
    original = h.pipeline.e_record
    failed = False

    def interruption(*args):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("after E write")
        return original(*args)

    monkeypatch.setattr(h.pipeline, "e_record", interruption)
    with pytest.raises(RuntimeError, match="after E write"):
        h.pipeline.generate()
    assert len(h.provider.calls) == 1
    assert len(list(read_jsonl(h.root / "evaluations.jsonl"))) == 1
    h.reopen()
    assert h.pipeline.generate()["n_valid"] == 3
    assert len(h.provider.calls) == 3
    assert len(h.grader.calls) == 3
    assert len(list(read_jsonl(h.root / "states.jsonl"))) == 3


def test_completed_initial_and_split_artifacts_detect_mutation(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch)
    h.pipeline.generate()
    h.pipeline.split()
    initial = h.root / "states.jsonl"
    original = initial.read_bytes()
    initial.write_bytes(original + b"\n")
    with pytest.raises(IntegrityError, match="changed"):
        h.pipeline.generate()
    initial.write_bytes(original)
    marker = read_json(h.root / "splits_complete.json")
    split_path = h.root / next(iter(marker["files"]))
    split_path.write_bytes(split_path.read_bytes() + b"\n")
    with pytest.raises(IntegrityError, match="changed"):
        h.pipeline.split()


def test_incomplete_core_round_is_not_silently_overwritten(tmp_path, monkeypatch):
    from rafr.experiments.exp2 import round_dir

    h = make_harness(tmp_path, monkeypatch)
    prepare_gate(h)
    directory = round_dir(h.root / "exp2", 1)
    directory.mkdir(parents=True)
    atomic_json(directory / "incomplete.json", {"reason": "injected interruption"})
    with pytest.raises(IntegrityError, match="incomplete|changed"):
        h.pipeline.exp2()


def test_downstream_scoring_rechecks_frozen_initial_artifacts(tmp_path, monkeypatch):
    from rafr.contracts.schemas import Split

    h = make_harness(tmp_path, monkeypatch)
    h.pipeline.generate()
    h.pipeline.split()
    path = h.root / "states.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(IntegrityError, match="changed"):
        h.pipeline.score(Split.D_FIT)
    assert all(verifier.calls == [] for verifier in h.pipeline.verifiers)


def test_label_read_rejects_changed_population_before_opening_y_star(tmp_path, monkeypatch):
    from rafr.contracts.schemas import Split
    from rafr.data import labels

    h = make_harness(tmp_path, monkeypatch)
    h.pipeline.generate()
    h.pipeline.split()
    marker = read_json(h.root / "splits_complete.json")
    path = h.root / next(iter(marker["files"]))
    path.write_bytes(path.read_bytes() + b"\n")
    opened = []
    monkeypatch.setattr(labels, "read_labels", lambda *a, **kw: opened.append(True))
    with pytest.raises(IntegrityError, match="split files changed"):
        h.pipeline._labels(Split.D_FIT, labels.LabelPurpose.TAU_FIT)
    assert opened == []


def test_relabeling_offline_smoke_report_cannot_promote_it_to_live(tmp_path, monkeypatch):
    from rafr.core import paths
    from rafr.domains.math.generation import frozen_models

    h = make_harness(tmp_path, monkeypatch)
    prepare_gate(h)
    h.pipeline.smoke()
    path = h.root / "exp2_smoke/smoke_report.json"
    report = read_json(path)
    report["execution_mode"] = "live"
    atomic_json(path, report)
    # Relative configured artifact paths stay repository-relative even when the
    # CLI is invoked from a foreign working directory.
    # A relocated repository fixture must include the now-frozen Core retry
    # prompt; otherwise the test stops at the missing-prompt guard before it can
    # exercise the independent smoke provenance check below.
    retry_path = frozen_models()["controller"]["parse_retry_prompt_file"]
    retry_bytes = paths.resolve(retry_path).read_bytes()
    relocated_retry = tmp_path / retry_path
    relocated_retry.parent.mkdir(parents=True, exist_ok=True)
    relocated_retry.write_bytes(retry_bytes)
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    h.pipeline.config.live_smoke_run = str(path.parent.relative_to(tmp_path))
    elsewhere = tmp_path / "foreign"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with pytest.raises(IntegrityError, match="producing run"):
        h.pipeline.require_smoke()


@pytest.mark.parametrize("change", [{"passed": False}, {"attempt": 2}, {"delta_r": 0.9}])
def test_gate_marker_and_actual_gate_report_must_agree(tmp_path, monkeypatch, change):
    h = make_harness(tmp_path, monkeypatch)
    prepare_gate(h)
    path = h.root / "gate_1.json"
    gate = read_json(path)
    atomic_json(path, {**gate, **change})
    with pytest.raises(IntegrityError, match="gate|Gate|changed"):
        h.pipeline.require_gate()
