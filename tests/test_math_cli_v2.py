"""V2 production entrypoint wiring with mocked external boundaries; no live calls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pipelines import math_run, math_runtime

from rafr.domains.math.config import MathRunConfig
from rafr.domains.math.storage import atomic_json
from rafr.llm.client import LLMClient


def config(tmp_path, **changes):
    source = tmp_path / "source.jsonl"
    source.write_text("fixture input; no preflight parsing")
    bundle = tmp_path / "bundle"
    bundle.mkdir(exist_ok=True)
    atomic_json(
        bundle / "manifest.json", {"domain": "math", "stage": "development", "eligible_count": 3}
    )
    return MathRunConfig(
        **{
            "stage": "development",
            "execution_mode": "live",
            "source_path": str(source),
            "bundle_path": str(bundle),
            "seed": 5,
            "max_initial_attempts": 3,
            "reviewed_by": "offline test fixture only",
            "research_configuration_approved": True,
            "pricing_reviewed": True,
            "data_reviewed": True,
            **changes,
        }
    )


def test_preflight_never_constructs_a_model_grader_provider_or_reads_a_key(tmp_path, monkeypatch):
    from rafr.llm.providers import openai

    cfg = config(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for module, name in (
        (openai, "OpenAIProvider"),
        (math_runtime, "MathGrader"),
        (math_runtime, "build_verifier"),
        (math_runtime, "load_bundle"),
    ):
        monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail("preflight invoked boundary"))
    report = math_runtime.preflight(cfg)
    assert report["ready_for_requested_phase"]
    assert report["controller"]["max_output_tokens"] == 16384
    assert report["eligible_questions"] == 3
    assert report["source_revalidation"] == report["api_key_lookup"] == "NOT_RUN"
    assert report["real_model_calls"] == 0
    assert report["execution_authorized"] is False


def test_preflight_reports_pending_without_network_or_invented_defaults(tmp_path):
    cfg = config(tmp_path, seed=None, max_initial_attempts=None)
    report = math_runtime.preflight(cfg)
    assert not report["ready_for_requested_phase"]
    assert any("seed" in reason for reason in report["pending_or_failed_checks"])
    assert cfg.seed is None and cfg.verifiers == []


def test_formal_initial_generation_does_not_require_future_smoke(tmp_path):
    cfg = config(tmp_path, stage="formal")
    assert cfg.live_smoke_run is None
    math_runtime.assert_command_allowed(cfg, "generate", confirm_live=True)
    with pytest.raises(ValueError, match="confirm-live"):
        math_runtime.assert_command_allowed(cfg, "generate", confirm_live=False)
    with pytest.raises(ValueError, match="review"):
        math_runtime.assert_command_allowed(
            cfg.model_copy(update={"reviewed_by": None}), "generate", confirm_live=True
        )


def test_smoke_phase_does_not_require_delta_and_cannot_run_arm_comparisons(tmp_path, monkeypatch):
    import rafr.formatter.math_feedback as feedback

    cfg = config(
        tmp_path,
        stage="smoke",
        delta_r=None,
        verifiers=[
            {
                "form": "prm",
                "vendor": "one",
                "base_model": "one",
                "model_id": "one",
                "revision": "pin",
                "implementation": "reviewed",
            },
            {
                "form": "orm",
                "vendor": "two",
                "base_model": "two",
                "model_id": "two",
                "revision": "pin",
                "implementation": "reviewed",
            },
        ],
    )
    monkeypatch.setattr(
        feedback,
        "load_prompts",
        lambda: {
            "checker_a": "Checker A evaluates steps.",
            "checker_b": "Checker B evaluates solutions.",
            "score_direction": "Higher scores indicate more errors.",
        },
    )
    math_runtime.assert_command_allowed(cfg, "smoke", confirm_live=True)
    for phase in ("exp1", "exp2"):
        with pytest.raises(ValueError, match="arm comparison"):
            math_runtime.assert_command_allowed(cfg, phase, confirm_live=True)


def test_runtime_uses_core_client_and_builds_snapshot_before_cache(tmp_path, monkeypatch):
    from pipelines import math_pipeline

    from rafr.llm.providers import openai

    cfg = config(tmp_path)
    root = tmp_path / "runs" / "first"
    calls = []

    class Pipeline:
        def __init__(self, config, root, bundle, *, client, verifiers, grader):
            assert not root.exists() or not any(root.iterdir())
            root.mkdir(parents=True, exist_ok=True)
            (root / "snapshot.json").write_text("{}")
            self.client, self.verifiers = client, verifiers

    monkeypatch.setattr(math_pipeline, "MathPipeline", Pipeline)
    monkeypatch.setattr(math_runtime, "load_bundle", lambda cfg: object())
    monkeypatch.setattr(math_runtime, "MathGrader", lambda *a, **k: object())
    monkeypatch.setattr(openai, "OpenAIProvider", lambda **kw: calls.append(kw) or object())
    run = math_runtime.build_pipeline(cfg, root, command="generate", confirm_live=True)
    assert isinstance(run.client, LLMClient)
    assert "openai" in run.client.providers
    assert run.client.ledger.path == root / "ledger.jsonl"
    assert (root / "cache").is_dir()
    assert calls == [{"max_retries": 0, "timeout_s": 120.0}]
    math_runtime.close_pipeline(run)


def test_analysis_runtime_does_not_create_openai_provider(tmp_path, monkeypatch):
    from pipelines import math_pipeline

    from rafr.llm.providers import openai

    cfg = config(tmp_path, max_initial_attempts=None)
    root = tmp_path / "runs" / "analysis"

    class Pipeline:
        def __init__(self, config, root, bundle, **kwargs):
            root.mkdir(parents=True, exist_ok=True)
            self.verifiers = []

    monkeypatch.setattr(math_pipeline, "MathPipeline", Pipeline)
    monkeypatch.setattr(math_runtime, "load_bundle", lambda cfg: object())
    monkeypatch.setattr(math_runtime, "MathGrader", lambda *a, **k: object())
    monkeypatch.setattr(
        openai, "OpenAIProvider", lambda **kw: pytest.fail("analysis requested key")
    )
    run = math_runtime.build_pipeline(cfg, root, command="report")
    assert run.client.providers == {}


def test_output_isolation_checks_run_before_model_construction(tmp_path):
    cfg = config(tmp_path)
    for root in (tmp_path, Path(cfg.bundle_path), Path(cfg.bundle_path) / "run"):
        with pytest.raises(ValueError, match="separate"):
            math_runtime.validate_output_root(cfg, root)


def test_cli_preflight_from_foreign_cwd_without_key(tmp_path):
    cfg = config(tmp_path)
    config_path = tmp_path / "math.json"
    atomic_json(config_path, cfg.model_dump())
    root = Path(__file__).resolve().parents[1]
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "OPENAI_API_KEY"}
    }
    result = subprocess.run(
        [
            sys.executable,
            str(root / "pipelines/math_run.py"),
            "preflight",
            "--config",
            str(config_path),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["api_key_lookup"] == "NOT_RUN"


def test_smoke_cli_report_uses_smoke_flag(tmp_path, monkeypatch, capsys):
    cfg = config(tmp_path, stage="smoke")
    config_path = tmp_path / "math.json"
    atomic_json(config_path, cfg.model_dump())
    observed = []
    fake = SimpleNamespace(
        report=lambda **kw: observed.append(kw) or {"SMOKE": True},
        client=SimpleNamespace(ledger=SimpleNamespace(summary=lambda: {})),
        verifiers=[],
    )
    monkeypatch.setattr(math_run, "build_pipeline", lambda *a, **kw: fake)
    assert (
        math_run.main(["report", "--config", str(config_path), "--run-dir", str(tmp_path / "run")])
        == 0
    )
    assert observed == [{"smoke": True}]
    assert '"SMOKE": true' in capsys.readouterr().out


def reviewed_pair():
    return [
        {
            "form": "prm",
            "vendor": "source-A",
            "base_model": "backbone-A",
            "model_id": "model-A",
            "revision": "pin",
            "implementation": "reviewed",
            "settings": {
                "license_reviewed": True,
                "license_review_reference": "fixture",
                "local_files_only": True,
            },
        },
        {
            "form": "orm",
            "vendor": "source-B",
            "base_model": "backbone-B",
            "model_id": "model-B",
            "revision": "pin",
            "implementation": "reviewed",
            "settings": {
                "license_reviewed": True,
                "license_review_reference": "fixture",
                "local_files_only": True,
            },
        },
    ]


@pytest.mark.parametrize(
    ("command", "expected"),
    [("diagnostics", False), ("stability", True), ("channel-check", "channel")],
)
def test_diagnostic_cli_dispatch(tmp_path, monkeypatch, command, expected):
    cfg = config(tmp_path)
    path = tmp_path / "math.json"
    atomic_json(path, cfg.model_dump())
    calls = []
    fake = SimpleNamespace(
        diagnostics=lambda **kw: calls.append(kw["stability"]),
        channel_check=lambda: calls.append("channel"),
        client=SimpleNamespace(ledger=SimpleNamespace(summary=lambda: {})),
        verifiers=[],
    )
    monkeypatch.setattr(math_run, "build_pipeline", lambda *a, **kw: fake)
    assert math_run.main([command, "--config", str(path), "--run-dir", str(tmp_path / "run")]) == 0
    assert calls == [expected]


def test_stability_pending_prompt_blocks_before_model_loading(tmp_path, monkeypatch):
    import shutil

    from rafr.domains.math import generation

    # The repository's rewrite instruction is final since 2026-10-13 (paper Sec. 5) and passes the
    # readiness check; a pending copy must still block before any model is loaded.
    (tmp_path / "real").mkdir()
    config(tmp_path / "real", verifiers=reviewed_pair()).assert_ready("stability")
    pending = tmp_path / "prompts"
    shutil.copytree(generation.PROMPT_DIR, pending)
    (pending / "perturb" / "perturb_paraphrase.txt").write_text(
        "__PLACEHOLDER__: pending.\n{problem}\n{text}\n", encoding="utf-8"
    )
    monkeypatch.setattr(generation, "PROMPT_DIR", pending)
    cfg = config(tmp_path, verifiers=reviewed_pair())
    monkeypatch.setattr(
        math_runtime, "build_verifier", lambda *a, **kw: pytest.fail("loaded pending model")
    )
    with pytest.raises(ValueError, match="confirm-live"):
        math_runtime.build_pipeline(cfg, tmp_path / "run", command="stability")
    with pytest.raises(RuntimeError, match="placeholder"):
        math_runtime.build_pipeline(cfg, tmp_path / "run", command="stability", confirm_live=True)
    report = math_runtime.preflight(cfg, phase="stability")
    assert not report["ready_for_requested_phase"]
    assert report["real_model_calls"] == 0 and report["api_key_lookup"] == "NOT_RUN"


@pytest.mark.parametrize("failure_at", ["second_verifier", "snapshot", "provider"])
def test_runtime_failure_closes_all_already_constructed_verifiers(
    tmp_path, monkeypatch, failure_at
):
    from pipelines import math_pipeline

    import rafr.formatter.math_feedback as feedback
    from rafr.llm.providers import openai

    cfg = config(tmp_path, verifiers=reviewed_pair())
    root = tmp_path / "runs" / "failed"
    constructed, closed = [], []
    monkeypatch.setattr(
        feedback,
        "load_prompts",
        lambda: {
            "checker_a": "Checker A evaluates steps.",
            "checker_b": "Checker B evaluates solutions.",
            "score_direction": "Higher scores indicate more errors.",
        },
    )
    monkeypatch.setattr(math_runtime, "load_bundle", lambda cfg: object())
    monkeypatch.setattr(math_runtime, "MathGrader", lambda *a, **k: object())

    def verifier(spec):
        if failure_at == "second_verifier" and constructed:
            raise RuntimeError("second verifier failed")
        value = SimpleNamespace(close=lambda: closed.append(spec.model_id))
        constructed.append(value)
        return value

    class Pipeline:
        def __init__(self, config, root, bundle, **kwargs):
            if failure_at == "snapshot":
                raise RuntimeError("snapshot failed")
            root.mkdir(parents=True, exist_ok=True)
            self.verifiers = kwargs["verifiers"]

    def provider(**kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(math_runtime, "build_verifier", verifier)
    monkeypatch.setattr(math_pipeline, "MathPipeline", Pipeline)
    monkeypatch.setattr(openai, "OpenAIProvider", provider)
    with pytest.raises(RuntimeError, match="failed"):
        math_runtime.build_pipeline(cfg, root, command="smoke", confirm_live=True)
    expected = {"snapshot": 0, "second_verifier": 1, "provider": 2}[failure_at]
    assert len(closed) == len(constructed) == expected


def test_preflight_reports_verifier_interfaces_as_pending_not_ready(tmp_path):
    cfg = config(tmp_path, verifiers=reviewed_pair())
    report = math_runtime.preflight(cfg, phase="scoring")
    assert not report["ready_for_requested_phase"]
    assert all(row["status"] == "INTERFACE_ONLY" for row in report["verifier_adapters"])
    assert any("backend implementation" in value for value in report["pending_or_failed_checks"])


def installed_pair(second_form):
    from rafr.domains.math.models import HF_IMPLEMENTATIONS

    first, second = reviewed_pair()
    first["implementation"] = "qwen_prm"
    second["form"] = second_form
    second["implementation"] = HF_IMPLEMENTATIONS[second_form]
    return [first, second]


@pytest.mark.parametrize("second_form", ["orm", "llm_judge"])
def test_preflight_reports_installed_adapters_of_every_form(tmp_path, second_form):
    cfg = config(tmp_path, verifiers=installed_pair(second_form))
    report = math_runtime.preflight(cfg, phase="scoring")
    assert [row["form"] for row in report["verifier_adapters"]] == ["prm", second_form]
    assert all(
        row["status"] == "ADAPTER_AVAILABLE_NOT_LOADED" for row in report["verifier_adapters"]
    )
    assert not any(
        "backend implementation" in value for value in report["pending_or_failed_checks"]
    )
    assert report["real_model_loading"] == "NOT_RUN"


@pytest.mark.parametrize("second_form", ["orm", "llm_judge"])
@pytest.mark.parametrize(
    "change",
    [{"local_files_only": False}, {"license_reviewed": False}, {"license_review_reference": " "}],
)
def test_every_form_needs_review_and_local_only_weights_before_it_is_built(
    tmp_path, monkeypatch, second_form, change
):
    from pipelines import math_pipeline

    import rafr.formatter.math_feedback as feedback

    pair = installed_pair(second_form)
    pair[1]["settings"] = {**pair[1]["settings"], **change}
    cfg = config(tmp_path, verifiers=pair)
    built = []
    monkeypatch.setattr(
        feedback,
        "load_prompts",
        lambda: {
            "checker_a": "Checker A evaluates steps.",
            "checker_b": "Checker B evaluates solutions.",
            "score_direction": "Higher scores indicate more errors.",
        },
    )
    monkeypatch.setattr(math_runtime, "load_bundle", lambda cfg: object())
    monkeypatch.setattr(math_runtime, "MathGrader", lambda *a, **k: object())
    monkeypatch.setattr(math_runtime, "build_verifier", lambda spec: built.append(spec.form))

    class Pipeline:
        def __init__(self, config, root, bundle, **kwargs):
            root.mkdir(parents=True, exist_ok=True)
            self.verifiers = kwargs["verifiers"]

    monkeypatch.setattr(math_pipeline, "MathPipeline", Pipeline)
    with pytest.raises(ValueError, match=f"{second_form} checkpoint review and local-only"):
        math_runtime.build_pipeline(
            cfg, tmp_path / "runs" / "x", command="score", confirm_live=True
        )
    assert built == ["prm"]
