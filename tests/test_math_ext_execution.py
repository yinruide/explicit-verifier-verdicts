"""Execution admission checks create no provider and perform no model call."""

from dataclasses import replace

import pytest
import yaml
from pipelines.math_ext_execution import (
    check_controller,
    check_execution,
    check_extension_config,
    check_mode,
)

from rafr.core.config import NotFrozenError
from rafr.domains.math.generation import build_controller_config
from rafr.domains.math.storage import IntegrityError
from rafr.formatter.math_extended import build_no_verifier_config
from rafr.llm.client import LLMClient
from rafr.llm.providers.fake import FakeProvider


class UncalledTransport:
    def generate(self, *args, **kwargs):
        pytest.fail("admission must never call the transport")


@pytest.mark.parametrize("confirmed", [False, None, "yes", 1])
def test_live_needs_explicit_boolean_confirmation(confirmed):
    with pytest.raises(IntegrityError, match="--confirm"):
        check_mode("live", confirmed=confirmed, source_mode="live")


@pytest.mark.parametrize("mode,source", [("live", "offline_stub"), ("offline_stub", "live")])
def test_source_mode_cannot_be_relabelled(mode, source):
    with pytest.raises(IntegrityError, match="source mode mismatch"):
        check_mode(mode, confirmed=True, source_mode=source)


def test_fixture_and_production_transport_are_separated_without_calling_either():
    fake = LLMClient({"openai": FakeProvider()})
    boundary = LLMClient({"openai": UncalledTransport()})
    check_execution(fake, "offline_stub", source_mode="offline_stub")
    check_execution(boundary, "live", confirmed=True, source_mode="live")
    with pytest.raises(IntegrityError, match="FakeProvider is not live"):
        check_execution(fake, "live", confirmed=True, source_mode="live")
    with pytest.raises(IntegrityError, match="requires explicitly injected FakeProvider"):
        check_execution(boundary, "offline_stub", source_mode="offline_stub")
    with pytest.raises(IntegrityError, match="OpenAI-only"):
        check_execution(
            LLMClient({"openai": UncalledTransport(), "anthropic": UncalledTransport()}),
            "live",
            confirmed=True,
            source_mode="live",
        )


@pytest.mark.parametrize("no_verifier", [False, True])
@pytest.mark.parametrize("mutation", ["model", "temperature", "effort", "prompt", "budget"])
def test_live_controller_cannot_drift(no_verifier, mutation):
    config = (build_no_verifier_config if no_verifier else build_controller_config)(17, 4)
    check_controller(config, seed=17, rounds=4, no_verifier=no_verifier)
    if mutation == "model":
        config = replace(config, model="other")
    elif mutation == "temperature":
        config = replace(config, decoding={**config.decoding, "temperature": 0.0})
    elif mutation == "effort":
        config = replace(config, decoding={**config.decoding, "reasoning_effort": "low"})
    elif mutation == "prompt":
        config = replace(config, system_prompt=config.system_prompt + " changed")
    else:
        config = replace(config, round_budget=1)
    with pytest.raises(IntegrityError, match="frozen math configuration"):
        check_controller(config, seed=17, rounds=4, no_verifier=no_verifier)


def test_unbound_extension_config_refuses_before_any_client(math_ext_unbound):
    # The repository config is bound (2026-11-05): each artifact binding back to TBD on its own
    # still refuses.
    with pytest.raises(NotFrozenError):
        check_extension_config(math_ext_unbound())
    for binding in math_ext_unbound.bindings:
        with pytest.raises(NotFrozenError):
            check_extension_config(math_ext_unbound([binding]))
    with pytest.raises(IntegrityError, match="math extension config"):
        check_extension_config(None)


def populated_config():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/math/ext.yaml").read_text(encoding="utf-8"))

    def fill(value):
        if isinstance(value, dict):
            return {k: fill(v) for k, v in value.items()}
        if isinstance(value, list):
            return [fill(v) for v in value]
        return "a" * 64 if value == "TBD" else value

    return fill(cfg)


@pytest.mark.parametrize(
    "path,value",
    [
        (("exp2", "k"), 3),
        (("exp2", "e_incorrect", "seed_namespace"), "wrong"),
        (("calibration", "epsilon"), 0.01),
        (("calibration", "bootstrap", "n_boot"), 100),
        (("cohorts", "e_correct", "n"), 299),
        (("exp1", "e_incorrect", "arms"), ["no_verifier"]),
        (("ceiling", "feedback"), "new hint"),
    ],
)
def test_populated_config_cannot_change_scientific_settings(path, value):
    cfg = populated_config()
    check_extension_config(cfg)
    node = cfg
    for name in path[:-1]:
        node = node[name]
    node[path[-1]] = value
    with pytest.raises(IntegrityError, match="scientific settings"):
        check_extension_config(cfg)


def test_original_hash_map_cannot_be_removed_to_bypass_binding():
    cfg = populated_config()
    cfg["exp1"]["e_incorrect"]["original_sha256"] = {}
    with pytest.raises(IntegrityError, match="binding keys"):
        check_extension_config(cfg)
