"""Thin CLI wiring for the Appendix M pipeline and the shared Core infrastructure.

Preflight reads configuration and installed-package metadata only. Model construction,
API-key lookup and GPU loading are never part of preflight.
"""

from __future__ import annotations

import importlib.metadata
from contextlib import suppress
from pathlib import Path
from typing import Any

from rafr.core import paths
from rafr.domains.math.config import MathRunConfig
from rafr.domains.math.dataset import load_dataset_bundle
from rafr.domains.math.generation import frozen_models
from rafr.domains.math.grading import MATH_VERIFY_VERSION, MathGrader, MathVerifyAdapter
from rafr.domains.math.models import (
    GENERATIVE_IMPLEMENTATIONS,
    HF_IMPLEMENTATIONS,
    MODEL_REPOSITORIES,
    build_verifier,
    validate_verifier_pair,
)
from rafr.domains.math.storage import IntegrityError, read_json
from rafr.llm.cache import LLMCache
from rafr.llm.client import LLMClient
from rafr.llm.ledger import Ledger

MODEL_COMMANDS = frozenset(
    {"generate", "score", "stability", "exp1", "exp1-smoke", "exp2", "smoke", "all"}
)
LLM_COMMANDS = frozenset({"generate", "stability", "exp1", "exp1-smoke", "exp2", "smoke", "all"})
VERIFIER_COMMANDS = frozenset({"score", "stability", "exp2", "smoke", "all"})
COMMAND_PHASE = {
    "generate": "generation",
    "split": "analysis",
    "score": "scoring",
    "fit": "scoring",
    "diagnostics": "scoring",
    "stability": "stability",
    "channel-check": "gate",
    "gate": "gate",
    "exp1": "experiment",
    "exp1-smoke": "experiment",
    "exp2": "experiment",
    "smoke": "smoke",
    "judge": "analysis",
    "report": "analysis",
    "report-exp1": "analysis",
    "revise-for-reserve": "analysis",
    "all": "experiment",
}
PACKAGES = ("pydantic", "numpy", "openai", "math-verify", "sympy", "latex2sympy2_extended")


def resolve_path(value: str | Path) -> Path:
    return paths.resolve(value).resolve()


def preflight(config: MathRunConfig, *, phase: str = "generation") -> dict[str, Any]:
    """Inventory, not an API/GPU probe or permission to run an experiment."""
    packages = {}
    problems = []
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
            problems.append(f"missing dependency: {name}")
    if packages["math-verify"] != config.math_verify_version:
        problems.append("installed Math-Verify differs from configured version")
    if config.math_verify_version != MATH_VERIFY_VERSION:
        problems.append("changing Math-Verify requires rebuilding the E chain")
    try:
        controller = frozen_models()["controller"]
    except (ValueError, OSError, KeyError) as exc:
        controller = None
        problems.append(str(exc))
    try:
        config.assert_ready(phase)
    except (ValueError, RuntimeError, OSError, TypeError) as exc:
        problems.append(str(exc))
    verifier_adapters = []
    if phase in {"scoring", "stability", "experiment", "smoke"}:
        try:
            specs = validate_verifier_pair(config.verifiers)
        except (ValueError, RuntimeError, TypeError):
            specs = ()  # The configuration check above already reports an incomplete pair.
        for spec in specs:
            # M.4 v1.2: a generative adapter (generative.py) implements its own form.
            implemented = GENERATIVE_IMPLEMENTATIONS.get(spec.implementation) == spec.form or (
                spec.implementation in MODEL_REPOSITORIES
                if spec.form == "prm"
                else spec.implementation == HF_IMPLEMENTATIONS.get(spec.form)
            )
            verifier_adapters.append(
                {
                    "form": spec.form,
                    "implementation": spec.implementation,
                    "status": "ADAPTER_AVAILABLE_NOT_LOADED" if implemented else "INTERFACE_ONLY",
                }
            )
            if not implemented:
                problems.append(f"reviewed {spec.form} backend implementation is still required")
    manifest_path = resolve_path(config.bundle_path) / "manifest.json"
    manifest = None
    if manifest_path.is_file():
        try:
            manifest = read_json(manifest_path)
        except (ValueError, OSError) as exc:
            problems.append(f"bundle manifest unreadable: {type(exc).__name__}")
    else:
        problems.append("dataset bundle has not been prepared")
    if not resolve_path(config.source_path).is_file():
        problems.append("source file is missing")
    if manifest is not None and (
        manifest.get("domain") != "math" or manifest.get("stage") != config.stage
    ):
        problems.append("bundle domain/stage differs from configuration")
    return {
        "domain": "math",
        "protocol": "v1.1 Appendix M",
        "phase": phase,
        "configuration_schema_valid": True,
        "stage": config.stage,
        "execution_mode": config.execution_mode,
        "packages": packages,
        "controller": {
            "model": controller["model"],
            "version": controller["version"],
            "reasoning_effort": "medium",
            "temperature_sent": False,
            "max_output_tokens": controller["decoding"]["max_output_tokens"],
        }
        if controller
        else None,
        "verifier_adapters": verifier_adapters,
        "bundle_manifest_present": manifest is not None,
        "eligible_questions": manifest.get("eligible_count") if manifest else None,
        "source_revalidation": "NOT_RUN",
        "ready_for_requested_phase": not problems,
        "pending_or_failed_checks": problems,
        "real_model_loading": "NOT_RUN",
        "api_key_lookup": "NOT_RUN",
        "real_model_calls": 0,
        "live_smoke": "NOT_RUN",
        "execution_authorized": False,
    }


def load_bundle(config: MathRunConfig):
    return load_dataset_bundle(
        resolve_path(config.bundle_path),
        stage=config.stage,
        source_path=resolve_path(config.source_path),
        development_source_path=resolve_path(config.development_source_path)
        if config.development_source_path
        else None,
        additional_development_paths=[resolve_path(p) for p in config.additional_development_paths],
    )


def validate_output_root(config: MathRunConfig, root: Path) -> Path:
    root = resolve_path(root)
    if root in {paths.ROOT.resolve(), paths.DATA.resolve(), paths.RUNS.resolve()}:
        raise IntegrityError("use a dedicated math run directory")
    protected = [resolve_path(config.source_path), resolve_path(config.bundle_path)]
    if config.development_source_path:
        protected.append(resolve_path(config.development_source_path))
    protected.extend(resolve_path(p) for p in config.additional_development_paths)
    if any(root == path or root in path.parents or path in root.parents for path in protected):
        raise IntegrityError("run output must be separate from input files and dataset bundles")
    if paths.is_archived(root):
        raise IntegrityError("active math runs cannot write into archive")
    return root


def assert_command_allowed(config: MathRunConfig, command: str, *, confirm_live: bool) -> None:
    if command not in COMMAND_PHASE:
        raise ValueError("unknown math execution command")
    if config.stage == "stub" or config.execution_mode != "live":
        raise ValueError("the CLI runs reviewed live boundaries; offline fixtures belong in tests")
    if config.stage == "smoke" and command in {"exp1", "exp1-smoke", "exp2"}:
        raise ValueError("a smoke run may execute only A1 smoke, never an arm comparison")
    if command in MODEL_COMMANDS:
        config.assert_live_allowed(confirm_live=confirm_live)
    phase = "smoke" if command == "all" and config.stage == "smoke" else COMMAND_PHASE[command]
    config.assert_ready(phase)
    if command in {"generate", "all"}:
        config.assert_ready("generation")
    if command == "all" and config.stage == "formal":
        config.assert_ready("stability")
    if config.math_verify_version != MATH_VERIFY_VERSION:
        raise ValueError("changing Math-Verify requires rebuilding the E chain")


def build_pipeline(config: MathRunConfig, root: Path, *, command: str, confirm_live: bool = False):
    """Wire Core objects only after phase-specific review guards have passed.

    Every verifier form (PRM, ORM, LLM judge) needs a recorded checkpoint review and
    local-only weights before it is built; an unknown implementation fails in
    build_verifier. Both happen before the API provider is created. No fake backend
    can be selected by a production configuration.
    """
    from pipelines.math_pipeline import MathPipeline

    assert_command_allowed(config, command, confirm_live=confirm_live)
    root = validate_output_root(config, root)
    bundle = load_bundle(config)
    grader = MathGrader(MathVerifyAdapter(expected_version=config.math_verify_version))
    verifiers = []
    try:
        # Establish and validate the run before loading weights or constructing a
        # provider. Only the transfer command may reopen an incomplete revision.
        pipeline = MathPipeline(
            config,
            root,
            bundle,
            client=LLMClient(providers={}),
            verifiers=[],
            grader=grader,
        )
        snapshot = root / "snapshot.json"
        is_revision = (root / "reserve_revision.json").exists() or (
            snapshot.exists() and read_json(snapshot).get("math_gate_history_path") is not None
        )
        if command != "revise-for-reserve" and is_revision:
            _ = pipeline.gate_attempts_path
        if command in VERIFIER_COMMANDS:
            specs = validate_verifier_pair(config.verifiers)
            for spec in specs:
                if (
                    not spec.settings.get("license_reviewed")
                    or not spec.settings.get("license_review_reference", "").strip()
                    or not spec.settings.get("local_files_only", True)
                ):
                    raise ValueError(
                        f"{spec.form} checkpoint review and local-only weights are required"
                    )
                verifiers.append(build_verifier(spec))
        pipeline.verifiers = verifiers
        providers = {}
        if command in LLM_COMMANDS:
            from rafr.llm.providers.openai import OpenAIProvider

            network = frozen_models()["controller"].get("network", {})
            providers["openai"] = OpenAIProvider(
                max_retries=int(network.get("max_retries", 0)),
                timeout_s=float(network.get("timeout_s", 120)),
            )
        if command == "revise-for-reserve":
            return pipeline
        pipeline.client = LLMClient(
            providers=providers,
            cache=LLMCache(root / "cache"),
            ledger=Ledger(root / "ledger.jsonl"),
            transient_log=root / "transient_retries.jsonl",
        )
        return pipeline
    except BaseException:
        _close_verifiers(verifiers)
        raise


def _close_verifiers(verifiers) -> None:
    for verifier in verifiers:
        close = getattr(verifier, "close", None)
        if callable(close):
            # On initialization failure preserve the original exception and attempt
            # cleanup of every already-created verifier, even if another close fails.
            with suppress(Exception):
                close()


def close_pipeline(pipeline) -> None:
    _close_verifiers(pipeline.verifiers)
