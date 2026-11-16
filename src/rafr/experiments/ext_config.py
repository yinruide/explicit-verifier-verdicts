"""configs/ext.yaml and the extension's formal-run guard (post-pilot, 2026-10-31).

The extension has a config file of its own, outside ``rafr.core.config.CONFIG_FILES``: nothing the
preregistered steps check moves because of it, and its TBDs stop extension runs only.

* :func:`assert_ext_frozen` refuses while any ext.yaml value is TBD or any extension prompt file
  (the No Verifier controller prompts, detox's pi-hat block) still carries the placeholder marker;
* :func:`checked` returns a preregistered file the extension reads, after checking it has the
  sha256 written in ext.yaml (a changed file is refused, never used);
* :func:`no_verifier_config` is the No Verifier controller: the formal controller config with the
  two No Verifier prompt files in place of system.txt / rewrite.txt.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rafr.calibration.apply import file_sha256
from rafr.controller import decide as ctrl
from rafr.core import paths
from rafr.core.config import NotFrozenError, find_placeholders, find_tbd, get, load_config

EXT_CONFIG = "ext"
EXT_PROMPT_KEYS: tuple[str, ...] = (
    "prompts.no_verifier_system_file",
    "prompts.no_verifier_rewrite_file",
    "prompts.verdict_only_file",
    "prompts.pihat_file",
)


class ExtConfigError(RuntimeError):
    pass


def load_ext(config_dir: Path | None = None) -> dict[str, Any]:
    return load_config(EXT_CONFIG, config_dir)


def ext_prompt_files(ext: Mapping[str, Any]) -> list[str]:
    files: list[str] = []
    for key in EXT_PROMPT_KEYS:
        value = get(ext, key)
        if not isinstance(value, str) or not value.strip():
            raise ExtConfigError(f"ext.{key} is not set")
        files.append(value)
    return files


def assert_ext_frozen(ext: Mapping[str, Any]) -> None:
    """Every extension run refuses while ext.yaml has a TBD or a prompt file a placeholder."""
    tbd = find_tbd(ext, EXT_CONFIG)
    placeholders = find_placeholders(ext_prompt_files(ext))
    if tbd or placeholders:
        raise NotFrozenError(tbd, placeholders)


def checked(path: str | Path, expected_sha256: str | None, *, what: str) -> Path:
    """``path`` resolved, once its sha256 is the one ext.yaml records for it."""
    p = paths.resolve(path)
    if not p.is_file():
        raise ExtConfigError(f"{what}: {p} does not exist")
    if expected_sha256 in (None, "", "TBD"):
        raise ExtConfigError(f"{what}: no sha256 recorded for {p}")
    got = file_sha256(p)
    if got != expected_sha256:
        raise ExtConfigError(f"{what}: {p} has sha256 {got}, ext.yaml records {expected_sha256}")
    return p


def no_verifier_config(
    models_cfg: Mapping[str, Any],
    exp_cfg: Mapping[str, Any],
    ext: Mapping[str, Any],
    *,
    base_seed: int | None,
    refusal_phrases_file: str | None = None,
    allow_tbd: bool = False,
) -> ctrl.ControllerConfig:
    """The controller of the No Verifier arm: same model, decoding, seed scheme, retry prompt
    and show_x0 as the formal controller; only the system prompt and the user-turn template are
    the No Verifier files (no checker paragraph, no observation slot)."""
    system = paths.resolve(get(ext, "prompts.no_verifier_system_file"))
    rewrite = paths.resolve(get(ext, "prompts.no_verifier_rewrite_file"))
    template = rewrite.read_text(encoding="utf-8")
    if "{observation}" in template:
        raise ExtConfigError(f"{rewrite} has an observation slot; No Verifier is shown nothing")
    return ctrl.build_config(
        models_cfg,
        exp_cfg,
        base_seed=base_seed,
        allow_tbd=allow_tbd,
        system_prompt=system.read_text(encoding="utf-8"),
        rewrite_template=template,
        refusal_phrases_file=refusal_phrases_file,
    )


__all__ = [
    "EXT_CONFIG",
    "EXT_PROMPT_KEYS",
    "ExtConfigError",
    "assert_ext_frozen",
    "checked",
    "ext_prompt_files",
    "load_ext",
    "no_verifier_config",
]
