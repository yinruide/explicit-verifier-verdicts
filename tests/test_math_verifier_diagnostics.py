"""Core diagnostic reuse, scientific label scope and mathematical perturbation safety."""

import json
from collections import Counter

import pytest
from pipelines import math_verifier_diagnostics as md

from math_v2_helpers import make_harness, q_number
from rafr.contracts.schemas import MathLabel, Split
from rafr.data.labels import LabelPurpose
from rafr.domains.math.storage import IntegrityError, file_hash, read_json
from rafr.llm.client import content_text
from rafr.llm.providers.fake import FakeReply
from rafr.logging.writer import read_jsonl
from rafr.verifiers.perturb import PerturbError


def prepare(tmp_path, monkeypatch, *, n=200, **sizes):
    h = make_harness(tmp_path, monkeypatch, n=n, **sizes)
    h.pipeline.generate()
    h.pipeline.split()
    for split in (Split.D_FIT, Split.UNLABELED_POOL):
        h.pipeline.score(split)
    prompt_dir = tmp_path / "perturb_prompts"
    prompt_dir.mkdir()
    (prompt_dir / "perturb_prefixes.txt").write_text("Quick note:\nBy the way,\nOne more thing:\n")
    (prompt_dir / "perturb_paraphrase.txt").write_text(
        "Keep every step and conclusion; change wording only.\nProblem: {problem}\n"
        'Solution: {text}\nReturn JSON {"solution": "complete reworded solution"}.\n'
    )
    monkeypatch.setattr(md, "PROMPT_DIR", prompt_dir)
    return h, prompt_dir


def test_baseline_uses_three_populations_only_allowed_labels_and_no_model_calls(
    tmp_path, monkeypatch
):
    h, prompt_dir = prepare(tmp_path, monkeypatch)
    # Baseline diagnostics must not require the still-pending stability prompt.
    (prompt_dir / "perturb_paraphrase.txt").write_text("__PLACEHOLDER__")
    accesses = []
    original = h.pipeline._labels

    def track(split, purpose):
        accesses.append((split, purpose))
        return original(split, purpose)

    monkeypatch.setattr(h.pipeline, "_labels", track)
    before = len(h.provider.calls), len(h.grader.calls)
    report = md.run_diagnostics(h.pipeline)
    assert before == (len(h.provider.calls), len(h.grader.calls))
    assert accesses == [(split, LabelPurpose.VERIFIER_DIAGNOSTICS) for split in md.POPULATIONS]
    assert set(report["single"]) == set(report["pair"]) == {"d_fit", "unlabeled_pool", "combined"}
    assert report["tau_source"] == "provisional_D_fit_F1"
    assert report["p_incorrect"] == 0.5
    assert report["stability_status"] == "NOT_RUN"
    assert "n_incorrect" in report["single"]["d_fit"]["v1"]
    assert "n_unsafe" not in json.dumps(report)
    fit = h.pipeline.fit()
    frozen = md.run_diagnostics(h.pipeline)
    assert frozen["tau"] == report["tau"] == [row["tau"] for row in fit["tau"]]
    assert frozen["tau_source"] == "frozen_D_fit_F1"


def test_stability_balanced_core_sample_math_labels_and_public_only_requests(tmp_path, monkeypatch):
    h, _ = prepare(tmp_path, monkeypatch)
    calls = len(h.provider.calls)
    report = md.run_diagnostics(h.pipeline, stability=True, n_per_class=3)
    assert report["stability_status"] == "COMPLETE"
    sample = read_json(h.root / "verifier_diagnostics/sample.json")
    assert Counter(row["e_label"] for row in sample) == {"INCORRECT": 3, "CORRECT": 3}
    # M.4 (2026-10-13): the stability sample comes from D_fit alone
    assert {row["split"] for row in sample} == {"d_fit"}
    requests = h.provider.calls[calls:]
    assert len(requests) == 6
    for request in requests:
        assert request.purpose == "math_verifier_paraphrase"
        assert request.model == "gpt-5.6-terra"
        assert request.decoding["reasoning_effort"] == "medium"
        assert request.decoding["max_output_tokens"] == 16384
        assert "temperature" not in request.decoding
        assert request.schema_name == "MathGenerationOutput" and request.attempt == 0
        assert "PRIVATE_GOLD_" not in json.dumps(request.messages)
    complete = h.pipeline._read_fixed(h.root / "verifier_diagnostics/COMPLETE.json")
    assert complete["report_sha256"] == file_hash(h.root / "verifier_diagnostics/report.json")
    assert complete["n_per_class"] == 3 and complete["n_sampled"] == 6
    assert complete["taus"] == report["tau"]
    assert report["stability"]["prefix"]["by_verifier"]["v1"]["e_unchanged"] is None
    assert report["stability"]["paraphrase"]["n_e_unchanged"] == 6
    before = len(h.provider.calls), len(h.grader.calls)
    h.reopen()
    assert md.run_diagnostics(h.pipeline, stability=True, n_per_class=3) == report
    assert before == (len(h.provider.calls), len(h.grader.calls))


def test_invalid_and_unresolved_paraphrases_dropped_once_before_verifier_scoring(
    tmp_path, monkeypatch
):
    h, _ = prepare(tmp_path, monkeypatch)
    counts = Counter()

    def answer(request):
        content = content_text(request.messages[0]["content"])
        number = q_number(content)
        counts[number] += 1
        if number % 2:
            return FakeReply(None, refused=True)
        return json.dumps({"solution": "Reworded but no answer declaration."})

    h.provider._responder = answer
    before_e = len(h.grader.calls)
    report = md.run_diagnostics(h.pipeline, stability=True, n_per_class=3)
    assert all(value == 1 for value in counts.values()) and len(counts) == 6
    para = report["stability"]["paraphrase"]
    assert para["n_valid"] == 0
    assert para["invalid_by_reason"] == {"refusal": 3, "unresolved": 3}
    assert para["e_status_counts"] == {"unresolved": 3}
    assert para["e_changed"] == {}
    assert para["by_verifier"]["v1"]["all"]["n"] == 0
    assert para["by_verifier"]["v1"]["e_unchanged"]["n"] == 0
    assert len(h.grader.calls) - before_e == 3
    assert all("Reworded" in call[2] for call in h.grader.calls[before_e:])
    assert not any("Reworded" in solution for _, solution in h.pipeline.verifiers[0].calls)
    judgments = list(read_jsonl(h.root / "verifier_diagnostics/paraphrase_judgments.jsonl"))
    assert {row["label"] for row in judgments} == {MathLabel.UNRESOLVED.value}


@pytest.mark.parametrize(
    "template", ["__PLACEHOLDER__ {problem} {text}", "Only {text}", "{problem} {text} {reference}"]
)
def test_unfilled_or_wrong_prompt_slots_stop_before_calls(tmp_path, monkeypatch, template):
    h, prompt_dir = prepare(tmp_path, monkeypatch)
    (prompt_dir / "perturb_paraphrase.txt").write_text(template)
    before = len(h.provider.calls)
    with pytest.raises((IntegrityError, PerturbError)):
        md.run_diagnostics(h.pipeline, stability=True, n_per_class=3)
    assert len(h.provider.calls) == before


def test_prompt_changes_and_completed_record_tampering_rejected(tmp_path, monkeypatch):
    h, prompt_dir = prepare(tmp_path, monkeypatch)
    md.run_diagnostics(h.pipeline, stability=True, n_per_class=3)
    template = prompt_dir / "perturb_paraphrase.txt"
    original = template.read_text()
    template.write_text(original + "Changed instruction.")
    with pytest.raises(IntegrityError, match="prompts.json"):
        md.run_diagnostics(h.pipeline, stability=True, n_per_class=3)
    template.write_text(original)
    path = h.root / "verifier_diagnostics/paraphrases.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(IntegrityError, match="diagnostic files changed"):
        md.run_diagnostics(h.pipeline, stability=True, n_per_class=3)


def test_live_cannot_reduce_preregistered_sample_size(tmp_path, monkeypatch):
    h, _ = prepare(tmp_path, monkeypatch)
    h.pipeline.config.stage = "development"
    h.pipeline.config.execution_mode = "live"
    with pytest.raises(IntegrityError, match="150 CORRECT and 150 INCORRECT"):
        md.run_diagnostics(h.pipeline, stability=True, n_per_class=3)


def test_insufficient_balanced_sample_stops_before_any_paraphrase_calls(tmp_path, monkeypatch):
    h, _ = prepare(tmp_path, monkeypatch)
    before = len(h.provider.calls)
    with pytest.raises(PerturbError, match="need 150"):
        md.run_diagnostics(h.pipeline, stability=True)
    assert len(h.provider.calls) == before


def test_interrupted_stability_resumes_without_regenerating_saved_paraphrases(
    tmp_path, monkeypatch
):
    h, _ = prepare(tmp_path, monkeypatch)
    original = h.provider._responder
    count = 0

    def interrupt(request):
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("injected stability interruption")
        return original(request)

    h.provider._responder = interrupt
    with pytest.raises(RuntimeError, match="stability interruption"):
        md.run_diagnostics(h.pipeline, stability=True, n_per_class=3)
    assert len(list(read_jsonl(h.root / "verifier_diagnostics/paraphrases.jsonl"))) == 2
    h.provider._responder = original
    before = len(h.provider.calls)
    h.reopen()
    report = md.run_diagnostics(h.pipeline, stability=True, n_per_class=3)
    assert report["stability_status"] == "COMPLETE"
    assert len(h.provider.calls) - before == 4


def test_default_stability_really_uses_three_hundred_balanced_states(tmp_path, monkeypatch):
    h, _ = prepare(tmp_path, monkeypatch, n=700, fit_n=360)
    before = len(h.provider.calls)
    report = md.run_diagnostics(h.pipeline, stability=True)
    assert report["sample"]["n"] == 300 and report["sample"]["n_per_class"] == 150
    sample = read_json(h.root / "verifier_diagnostics/sample.json")
    assert Counter(row["e_label"] for row in sample) == {"CORRECT": 150, "INCORRECT": 150}
    assert len(h.provider.calls) - before == 300


def test_interruption_after_raw_response_before_e_does_not_buy_a_second_paraphrase(
    tmp_path, monkeypatch
):
    h, _ = prepare(tmp_path, monkeypatch)
    original = md._judge_paraphrase

    def interrupt(*args):
        raise OSError("injected E persistence failure")

    monkeypatch.setattr(md, "_judge_paraphrase", interrupt)
    with pytest.raises(OSError, match="E persistence failure"):
        md.run_diagnostics(h.pipeline, stability=True, n_per_class=3)
    out = h.root / "verifier_diagnostics"
    assert len(list(read_jsonl(out / "generation_attempts.jsonl"))) == 1
    assert not list(read_jsonl(out / "paraphrases.jsonl"))
    before = len(h.provider.calls)
    monkeypatch.setattr(md, "_judge_paraphrase", original)
    h.reopen()
    md.run_diagnostics(h.pipeline, stability=True, n_per_class=3)
    assert len(h.provider.calls) - before == 5
    assert len(list(read_jsonl(out / "generation_attempts.jsonl"))) == 6
    assert len(list(read_jsonl(out / "paraphrases.jsonl"))) == 6
