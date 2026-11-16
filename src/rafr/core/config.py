"""Config loading and formal-run protection.

* :func:`load_config` reads one or all ``configs/*.yaml`` files.
* :func:`find_tbd` lists every dotted path whose value is the ``TBD`` sentinel.
* :data:`PROMPT_FILES` is the explicit list of config entries that point at prompt / template /
  phrase files the authors write; :func:`find_placeholders` lists those that still contain
  ``PLACEHOLDER`` (or do not exist). Nothing is inferred from key names (decision, 2026-09-17):
  data files such as ``exp1.outcome.judgments_file`` or ``channel.gate.attempts_path`` are
  never treated as prompts.
* :func:`assert_frozen` is the formal-run guard: step04 and later call it and refuse to run
  while anything is still undecided. ``pipelines/step03_dry_run.py`` is the only caller
  allowed to pass ``allow_placeholders=True``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from rafr.contracts.schemas import PLACEHOLDER, TBD
from rafr.core import paths

CONFIG_FILES = (
    "models",
    "data",
    "generation",
    "channel",
    "exp1",
    "exp2",
    "exp2_smoke",  # 0.5 step 10a: the Exp 2 smoke (II.4.4, 2026-09-25)
    "refusal_phrases",
)


class NotFrozenError(RuntimeError):
    """Raised by :func:`assert_frozen` with the full list of missing items."""

    def __init__(
        self, tbd: list[str], placeholders: list[str], missing: list[str] | None = None
    ) -> None:
        self.tbd = tbd
        self.placeholders = placeholders
        self.missing = list(missing or [])  # PROMPT_FILES entries absent from the configs
        lines = ["Refusing to run a formal step: configuration is not frozen."]
        if tbd:
            lines.append(f"  {len(tbd)} config value(s) still TBD:")
            lines += [f"    - {p}" for p in tbd]
        if placeholders:
            lines.append(f"  {len(placeholders)} file(s) still contain {PLACEHOLDER}:")
            lines += [f"    - {p}" for p in placeholders]
        if self.missing:
            lines.append(f"  {len(self.missing)} prompt-file entry(ies) not set in configs/:")
            lines += [f"    - {p}" for p in self.missing]
        super().__init__("\n".join(lines))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def load_config(name: str | None = None, config_dir: Path | None = None) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` or, with ``name=None``, all files keyed by name."""
    cdir = config_dir or paths.CONFIGS
    if name is not None:
        return load_yaml(cdir / f"{name}.yaml")
    return {n: load_yaml(cdir / f"{n}.yaml") for n in CONFIG_FILES}


def master_seed(configs: Mapping[str, Any]) -> int:
    """``configs/data.yaml: seeds.master`` — the one seed every component derives from."""
    v = (configs.get("data") or {}).get("seeds", {}).get("master")
    if v is None or v == TBD:
        raise NotFrozenError(["data.seeds.master"], [])
    return int(v)


def find_tbd(obj: Any, prefix: str = "") -> list[str]:
    """Dotted paths of every value equal to the ``TBD`` sentinel (case-sensitive)."""
    found: list[str] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            found += find_tbd(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += find_tbd(v, f"{prefix}[{i}]")
    elif isinstance(obj, str) and obj == TBD:
        found.append(prefix)
    return found


PROMPT_FILES: tuple[str, ...] = (
    "models.controller.system_prompt_file",
    "models.controller.rewrite_prompt_file",
    "models.controller.parse_retry_prompt_file",
    "models.g_fixed.prompt_file",
    "models.evaluator.prompt_file",
    "generation.g_fixed.validity.refusal_phrases_file",
)
"""Dotted ``<config>.<path>`` of every file the formal-run guard checks for ``PLACEHOLDER``.
Adding a prompt file means adding it here; ``tests/test_core.py`` checks the list against
the real configs."""


def _prompt_entries(
    configs: Mapping[str, Any], only_sections: Iterable[str] | None = None
) -> tuple[list[str], list[str]]:
    """``(file paths, missing entries)`` for the :data:`PROMPT_FILES` entries in scope.

    An entry in scope whose key is absent (or not a non-empty string) is reported as
    missing, never skipped: a config that simply lacks ``rewrite_prompt_file`` must not pass
    the guard as if that prompt had been checked (review, 2026-09-18).
    """
    secs = None if only_sections is None else tuple(only_sections)
    files: list[str] = []
    missing: list[str] = []
    for dotted in PROMPT_FILES:
        cname, _, rest = dotted.partition(".")
        if secs is not None and not any(
            dotted == sec or dotted.startswith(sec + ".") for sec in secs
        ):
            continue
        value = get(configs.get(cname, {}), rest)
        if isinstance(value, str) and value.strip():
            files.append(value)
        else:
            missing.append(dotted)
    return files, missing


def prompt_files(
    configs: Mapping[str, Any], only_sections: Iterable[str] | None = None
) -> list[str]:
    """The file paths of the :data:`PROMPT_FILES` entries in scope (all of them, or those
    under ``only_sections`` -- ``"<config>.<top-level-key>"`` prefixes).

    Raises :class:`NotFrozenError` (``.missing``) when an entry in scope is not set.
    """
    files, missing = _prompt_entries(configs, only_sections)
    if missing:
        raise NotFrozenError([], [], missing)
    return files


class ArchiveReferenceError(RuntimeError):
    """A config value points into ``archive/``, which the formal pipeline must never read
    (protocol v0.9, II.10 / App. B.1)."""

    def __init__(self, found: list[str]):
        self.found = list(found)
        super().__init__(
            "Refusing to run: the formal pipeline may not read archive/ (II.10), but these "
            "config values resolve into it:\n" + "\n".join(f"    - {f}" for f in self.found)
        )


def archived_config_paths(obj: Any, prefix: str = "") -> list[str]:
    """``"<dotted.key> = <value>"`` for every string in ``obj`` that resolves into ``archive/``.

    Every string leaf is tried as a repo-relative path, not only keys that look like paths:
    a model id such as ``claude-opus-5`` resolves to ``<root>/claude-opus-5`` and is harmless,
    while any spelling that lands in ``archive/`` is caught whatever its key is called.
    """
    found: list[str] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            found += archived_config_paths(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += archived_config_paths(v, f"{prefix}[{i}]")
    elif isinstance(obj, str) and obj.strip():
        try:
            hit = paths.is_archived(obj)
        except (OSError, ValueError):
            hit = False
        if hit:
            found.append(f"{prefix} = {obj}")
    return found


def assert_no_archive(configs: Mapping[str, Any]) -> None:
    """Raise :class:`ArchiveReferenceError` if any config value resolves into ``archive/``."""
    found = archived_config_paths(configs)
    if found:
        raise ArchiveReferenceError(found)


def file_has_placeholder(path: Path) -> bool:
    if not path.exists():
        return True  # a missing prompt file is as blocking as a placeholder
    return PLACEHOLDER in path.read_text(encoding="utf-8")


def find_placeholders(files: Iterable[str | Path]) -> list[str]:
    return [str(f) for f in files if file_has_placeholder(paths.resolve(f))]


def assert_frozen(
    configs: Mapping[str, Any],
    *,
    extra_files: Iterable[str | Path] = (),
    allow_placeholders: bool = False,
    allow_tbd: bool = False,
    only_sections: Iterable[str] | None = None,
) -> None:
    """Formal-run protection.

    Raises :class:`NotFrozenError` listing every ``TBD`` config value, every prompt file
    of :data:`PROMPT_FILES` (plus ``extra_files``) that still contains ``PLACEHOLDER`` or is
    missing on disk, and every :data:`PROMPT_FILES` entry in scope that the configs do not
    set at all. ``only_sections`` restricts the TBD scan to
    ``"<config>.<top-level-key>"`` entries (e.g. ``["models.controller", "generation"]``)
    so that a step is blocked only by the values it actually consumes.

    Independently of all that, and with no dry-run exception, the whole config is checked for
    paths into ``archive/`` (:class:`ArchiveReferenceError`; v0.9, II.10).
    """
    assert_no_archive(configs)
    tbd: list[str] = []
    if not allow_tbd:
        if only_sections is None:
            for cname, cobj in configs.items():
                tbd += find_tbd(cobj, cname)
        else:
            for sec in only_sections:
                # "<config>" (whole file), "<config>.<key>" or any deeper dotted path
                cname, _, key = sec.partition(".")
                cobj = configs.get(cname, {})
                sub = get(cobj, key) if key else cobj
                tbd += find_tbd(sub, sec)
    placeholders: list[str] = []
    missing: list[str] = []
    if not allow_placeholders:
        files, missing = _prompt_entries(configs, only_sections)
        placeholders = find_placeholders(dict.fromkeys(list(extra_files) + files))
    if tbd or placeholders or missing:
        raise NotFrozenError(tbd, placeholders, missing)


def get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """``get(cfg, "controller.decoding.seed")``."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur
