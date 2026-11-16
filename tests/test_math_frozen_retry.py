"""A frozen controller run must bind the contents of the shared retry policy."""

from copy import deepcopy

import pytest

from math_v2_helpers import make_harness, prepare_gate
from rafr.domains.math import generation
from rafr.domains.math.storage import IntegrityError


def setup_retry_file(tmp_path, monkeypatch):
    retry = tmp_path / "core_retry.txt"
    retry.write_text("Return the required JSON without commentary.\n")
    models = deepcopy(generation.frozen_models())
    models["controller"]["parse_retry_prompt_file"] = str(retry)
    monkeypatch.setattr(generation, "frozen_models", lambda: deepcopy(models))
    return retry


def test_retry_content_change_invalidates_consumed_feedback_and_smoke_identity(
    tmp_path, monkeypatch
):
    retry = setup_retry_file(tmp_path, monkeypatch)
    h = make_harness(tmp_path, monkeypatch)
    prepare_gate(h)
    h.pipeline.exp1()
    old_identity = h.pipeline.config.profile_hash()
    old_prompt = generation.build_controller_config(h.pipeline.config.seed, 1).retry_prompt
    calls = len(h.provider.calls)

    retry.write_text("Return the required JSON and always choose rewrite.\n")

    assert h.pipeline.config.profile_hash() != old_identity
    assert generation.build_controller_config(h.pipeline.config.seed, 1).retry_prompt != old_prompt
    with pytest.raises(IntegrityError, match="frozen feedback configuration changed"):
        h.reopen()
    assert len(h.provider.calls) == calls


def test_generation_does_not_freeze_a_retry_policy_it_does_not_consume(tmp_path, monkeypatch):
    retry = setup_retry_file(tmp_path, monkeypatch)
    h = make_harness(tmp_path, monkeypatch, n=4)
    h.pipeline.generate()
    calls = len(h.provider.calls)

    retry.write_text("Return valid JSON with the required keys.\n")

    h.reopen()
    h.pipeline.generate()
    assert len(h.provider.calls) == calls


@pytest.mark.parametrize("value", ["TBD\n", "\n"])
def test_unfilled_retry_policy_cannot_be_bound_to_a_controller_run(tmp_path, monkeypatch, value):
    retry = setup_retry_file(tmp_path, monkeypatch)
    h = make_harness(tmp_path, monkeypatch)
    retry.write_text(value)

    with pytest.raises(ValueError, match="TBD"):
        h.pipeline.config.profile_hash()
    with pytest.raises(ValueError, match="TBD"):
        generation.build_controller_config(h.pipeline.config.seed, 1)
