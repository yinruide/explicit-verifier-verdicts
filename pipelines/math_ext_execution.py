"""Explicit execution boundaries for math extensions; imports never create providers.

An operator's --confirm is required for live calls. Implementing this path does not
authorize this development session to run it. Offline fixtures stay visibly offline.
"""

from __future__ import annotations

from dataclasses import asdict

from rafr.domains.math.storage import IntegrityError
from rafr.llm.providers.fake import FakeProvider


def check_mode(execution_mode, *, confirmed=False, source_mode=None):
    if execution_mode not in {"offline_stub", "live"}:
        raise IntegrityError("execution mode must be offline_stub or live")
    if execution_mode == "live" and confirmed is not True:
        raise IntegrityError("live is not authorized without explicit --confirm")
    if source_mode != execution_mode:
        raise IntegrityError("execution/source mode mismatch; never relabel fixture evidence")


def check_execution(client, execution_mode, *, confirmed=False, source_mode=None):
    check_mode(execution_mode, confirmed=confirmed, source_mode=source_mode)
    providers = getattr(client, "providers", {})
    if not providers:
        raise IntegrityError("an explicit controller provider is required")
    if execution_mode == "offline_stub":
        if any(not isinstance(provider, FakeProvider) for provider in providers.values()):
            raise IntegrityError("offline_stub requires explicitly injected FakeProvider")
    elif set(providers) != {"openai"} or any(
        isinstance(provider, FakeProvider) for provider in providers.values()
    ):
        raise IntegrityError("live math needs an OpenAI-only client; FakeProvider is not live")


def check_controller(config, *, seed, rounds, no_verifier=False):
    """Live controllers retain the exact frozen model, prompts, seed and decoding."""
    from rafr.domains.math.generation import build_controller_config
    from rafr.formatter.math_extended import build_no_verifier_config

    expected = (build_no_verifier_config if no_verifier else build_controller_config)(seed, rounds)
    if config is None or asdict(config) != asdict(expected):
        raise IntegrityError("live controller differs from frozen math configuration")


def check_extension_config(config):
    from copy import deepcopy
    from pathlib import Path

    import yaml

    from rafr.experiments.ext_config import assert_ext_frozen
    from rafr.formatter.math_extended import PROMPT_DIR, PROMPT_FILES, SHARED_DIR, SHARED_FILES

    if not isinstance(config, dict) or config.get("domain") != "math":
        raise IntegrityError("live execution requires a frozen math extension config")
    assert_ext_frozen(config)
    root = Path(__file__).resolve().parents[1]
    # Only artifact bindings are operator-fillable. A fully populated configuration
    # must not silently contradict the fixed algorithms, populations or seed scheme.
    template = yaml.safe_load((root / "configs/math/ext.yaml").read_text(encoding="utf-8"))
    variable = (
        "source_manifest",
        "prompts",
        "calibration.frozen_file",
        "calibration.frozen_sha256",
        "calibration.diagnostics_dir",
        "cohorts.e_correct.file",
        "cohorts.e_correct.file_sha256",
        "exp1.e_incorrect.original_run",
        "exp1.e_incorrect.original_sha256",
        "exp2.e_incorrect.original_run",
        "exp2.e_incorrect.original_sha256",
        "exp2.e_incorrect.x1_file",
        "exp2.e_incorrect.x1_sha256",
        "exp2.e_incorrect.t1_scores.source",
        "exp2.e_incorrect.t1_scores.source_sha256",
        "analysis.cited_not_recomputed.exp1.e_incorrect.tables_dir",
        "analysis.cited_not_recomputed.exp1.e_incorrect.sha256",
    )
    actual = deepcopy(config)
    for field in variable:
        expected_node, actual_node = template, actual
        *parents, leaf = field.split(".")
        try:
            for parent in parents:
                expected_node, actual_node = expected_node[parent], actual_node[parent]
            value, expected_value = actual_node[leaf], expected_node[leaf]
            if isinstance(expected_value, dict):
                if not isinstance(value, dict) or set(value) != set(expected_value):
                    raise IntegrityError(f"extension binding keys differ: {field}")
            elif not isinstance(value, str) or not value.strip():
                raise IntegrityError(f"empty extension binding: {field}")
            actual_node[leaf] = expected_node[leaf]
        except (KeyError, TypeError) as exc:
            raise IntegrityError(f"missing extension binding: {field}") from exc
    if actual != template:
        raise IntegrityError("extension scientific settings differ from the reviewed math template")
    wanted = {
        "no_verifier_system_file": PROMPT_DIR / PROMPT_FILES[0],
        "no_verifier_rewrite_file": PROMPT_DIR / PROMPT_FILES[1],
        "verdict_only_file": SHARED_DIR / SHARED_FILES[0],
        "pihat_file": SHARED_DIR / SHARED_FILES[1],
    }
    for name, expected in wanted.items():
        path = Path(config["prompts"][name])
        path = path if path.is_absolute() else root / path
        if path.resolve() != expected.resolve():
            raise IntegrityError("extension prompt configuration differs from executed files")
