"""paper Sec. 5, App. D startup/configuration guards; no provider or model execution."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from math_v2_helpers import make_harness, prepare_gate
from rafr.domains.math.config import MathRunConfig
from rafr.domains.math.models import MathModelError
from rafr.domains.math.storage import IntegrityError


def config(**changes):
    return MathRunConfig.model_validate(
        {
            "stage": "stub",
            "execution_mode": "offline_stub",
            "source_path": "unused",
            "bundle_path": "unused",
            **changes,
        }
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("generation", {"model": "different-backbone"}),
        ("controller", {"temperature": 0.0}),
        ("checker_descriptions", ["embedded", "descriptions"]),
        ("answer_contract_version", "exact-rational-v1"),
    ],
)
def test_math_configuration_rejects_parallel_core_and_retired_scientific_fields(field, value):
    with pytest.raises(ValidationError, match="Extra inputs"):
        config(**{field: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"stage": "formal", "execution_mode": "offline_stub"},
        {"stage": "stub", "execution_mode": "live"},
        {"stage": "formal", "execution_mode": "live", "experiment_n": 10},
        {"stage": "formal", "execution_mode": "live", "bootstrap_n": 100},
        {"stage": "smoke", "execution_mode": "live", "smoke_n": 2},
    ],
)
def test_stub_cannot_be_promoted_and_formal_sizes_are_fixed(changes):
    with pytest.raises(ValidationError):
        config(**changes)


def test_pending_generation_choices_are_not_silently_defaulted():
    pending = config()
    assert pending.seed is None and pending.max_initial_attempts is None
    assert pending.delta_r is None and pending.verifiers == []
    with pytest.raises(ValueError, match="seed"):
        pending.assert_ready("generation")
    seeded = config(seed=5)
    with pytest.raises(ValueError, match="attempt cap"):
        seeded.assert_ready("generation")
    seeded.max_initial_attempts = 3
    seeded.assert_ready("generation")
    with pytest.raises(MathModelError):
        seeded.assert_ready("scoring")


def test_live_confirmation_and_review_are_distinct_from_offline_readiness():
    candidate = config(stage="smoke", execution_mode="live", seed=5, max_initial_attempts=2)
    with pytest.raises(ValueError, match="confirm-live"):
        candidate.assert_live_allowed(confirm_live=False)
    with pytest.raises(ValueError, match="the authors"):
        candidate.assert_live_allowed(confirm_live=True)
    candidate.reviewed_by = "test-fixture-operator"
    candidate.research_configuration_approved = True
    candidate.pricing_reviewed = True
    candidate.data_reviewed = True
    candidate.assert_live_allowed(confirm_live=True)


def test_consumed_generation_identity_cannot_change_on_resume(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch, n=4)
    h.pipeline.generate()
    h.pipeline.config.seed += 1
    with pytest.raises(IntegrityError, match="identity|changed|frozen"):
        h.reopen()


def test_e_version_identity_cannot_claim_another_math_verify(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch, n=4)
    h.pipeline.config.math_verify_version = "different-version"
    with pytest.raises(IntegrityError, match="version|identity|changed"):
        h.reopen()


def test_unconsumed_verifiers_and_delta_can_be_filled_after_generation(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch, n=4)
    h.pipeline.config.verifiers = []
    h.pipeline.config.delta_r = None
    h.pipeline.generate()
    h.pipeline.config.verifiers = [
        {"model_id": "new-fixture-A", "revision": "fixture-1"},
        {"model_id": "new-fixture-B", "revision": "fixture-2"},
    ]
    h.pipeline.config.delta_r = 0.1
    h.reopen()
    assert h.pipeline.generate()["n_valid"] == 4


def test_verifier_configuration_freezes_at_first_score(tmp_path, monkeypatch):
    from rafr.contracts.schemas import Split

    h = make_harness(tmp_path, monkeypatch)
    h.pipeline.generate()
    h.pipeline.split()
    h.pipeline.score(Split.D_FIT)
    h.pipeline.config.verifiers[0]["revision"] = "changed-after-scoring"
    with pytest.raises(IntegrityError, match="scoring.*changed"):
        h.reopen()


def test_delta_freezes_at_first_gate(tmp_path, monkeypatch):
    h = make_harness(tmp_path, monkeypatch)
    prepare_gate(h)
    h.pipeline.config.delta_r = 0.08
    with pytest.raises(IntegrityError, match="gate.*changed"):
        h.reopen()
