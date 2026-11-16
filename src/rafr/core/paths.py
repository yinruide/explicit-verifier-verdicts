"""Repository paths. Everything resolves relative to the repo root, never the cwd."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Locate the repo root: ``RAFR_ROOT`` env var, else walk up from this file."""
    env = os.environ.get("RAFR_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError("cannot locate repo root (no pyproject.toml + configs/ above core/paths.py)")


ROOT = repo_root()


def data_root() -> Path:
    """Directory that holds ``data/``, ``runs/`` and ``frozen/``.

    ``RAFR_DATA_DIR`` env var if set (the released data directory, kept outside git), else the
    repo root.
    """
    env = os.environ.get("RAFR_DATA_DIR")
    return Path(env).resolve() if env else ROOT


DATA_ROOT = data_root()
DATA_PREFIXES = ("data", "runs", "frozen")
"""First path components that :func:`resolve` maps to :data:`DATA_ROOT` instead of the repo."""

CONFIGS = ROOT / "configs"
DATA = DATA_ROOT / "data"
RUNS = DATA_ROOT / "runs"
FROZEN = DATA_ROOT / "frozen"
SRC = ROOT / "src" / "rafr"

DATA_RAW = DATA / "raw"
DATA_POOLS = DATA / "pools"
DATA_SPLITS = DATA / "splits"
DATA_LABELS = DATA / "labels"
DATA_SCORES = DATA / "scores"
DATA_CACHE = DATA / "cache"
MANIFEST = DATA / "manifest.json"

FROZEN_TAU = FROZEN / "tau.json"
FROZEN_CHANNEL = FROZEN / "channel.json"
FROZEN_GATE_ATTEMPTS = FROZEN / "gate_attempts.json"

ARCHIVE = ROOT / "archive"
"""Retired rounds (protocol v0.9, II.10). The formal pipeline never reads anything under it
(App. B.1); :func:`is_archived` is what the config guard and the tests check against. Not in
``_REDIRECTED``: nothing may write here either, fake or formal."""


def is_archived(path: str | os.PathLike[str]) -> bool:
    """Does ``path`` (config-relative or absolute) resolve into ``archive/``?"""
    return resolve(path).resolve().is_relative_to(ARCHIVE.resolve())


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_POOLS, DATA_SPLITS, DATA_LABELS, DATA_SCORES, DATA_CACHE, RUNS):
        p.mkdir(parents=True, exist_ok=True)


def resolve(path: str | os.PathLike[str]) -> Path:
    """Resolve a config-relative path (e.g. ``src/rafr/controller/prompts/system.txt``).

    A relative path whose first component is ``data``, ``runs`` or ``frozen`` resolves under
    :data:`DATA_ROOT`; every other relative path under the repo root.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in DATA_PREFIXES:
        return DATA_ROOT / p
    return ROOT / p


def shown(path: str | os.PathLike[str]) -> str:
    """``path`` as written in reports: relative to :data:`DATA_ROOT` or the repo root, else as
    is."""
    p = Path(path)
    for base in (DATA_ROOT, ROOT):
        try:
            return p.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return p.as_posix()


def run_dir(run_id: str) -> Path:
    """Create and return the directory a new run or analysis writes into.

    ``RAFR_OUT_DIR`` env var, when set, receives new directories instead of :data:`RUNS`, so that
    regenerating tables from the released data never writes into the data directory.
    """
    out = os.environ.get("RAFR_OUT_DIR")
    d = (Path(out) if out else RUNS) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


FAKE_ROOT_NAME = "_fake"
"""Name of the isolated root a ``--fake`` run writes under (2026-09-20).

A ``--fake`` run is an engineering self-check. On 2026-09-20 one of them overwrote the real
splits of a finished formal run -- 200/200/200/300/3000 became 1/1/1/1/1 -- because the fake
run wrote to the same ``data/`` paths as the real one. It cost a restore from a backup kept
outside the repo. The rule now is structural: a fake run redirects every data, runs and
frozen path under ``data/_fake/`` and cannot reach a formal path at all.
"""

_REDIRECTED = (
    "DATA",
    "DATA_RAW",
    "DATA_POOLS",
    "DATA_SPLITS",
    "DATA_LABELS",
    "DATA_SCORES",
    "DATA_CACHE",
    "MANIFEST",
    "RUNS",
    "FROZEN",
    "FROZEN_TAU",
    "FROZEN_CHANNEL",
    "FROZEN_GATE_ATTEMPTS",
)
"""Every module path a fake run must not be able to write. Kept as a list so that the
isolation check and the tests' sandbox fixture cannot drift apart from the redirect."""


def is_fake_root() -> bool:
    """Is this process currently writing under the isolated fake root?"""
    return DATA.name == FAKE_ROOT_NAME


def use_fake_data_root() -> Path:
    """Redirect every path in :data:`_REDIRECTED` under ``<data>/_fake`` and return it.

    Idempotent: calling it twice does not nest another ``_fake``. Call it once, at the top of
    a step's ``main()``, before anything resolves a path.
    """
    global DATA, DATA_RAW, DATA_POOLS, DATA_SPLITS, DATA_LABELS, DATA_SCORES, DATA_CACHE
    global MANIFEST, RUNS, FROZEN, FROZEN_TAU, FROZEN_CHANNEL, FROZEN_GATE_ATTEMPTS
    root = DATA if is_fake_root() else DATA / FAKE_ROOT_NAME
    DATA = root
    DATA_RAW = root / "raw"
    DATA_POOLS = root / "pools"
    DATA_SPLITS = root / "splits"
    DATA_LABELS = root / "labels"
    DATA_SCORES = root / "scores"
    DATA_CACHE = root / "cache"
    MANIFEST = root / "manifest.json"
    RUNS = root / "runs"
    FROZEN = root / "frozen"
    FROZEN_TAU = FROZEN / "tau.json"
    FROZEN_CHANNEL = FROZEN / "channel.json"
    FROZEN_GATE_ATTEMPTS = FROZEN / "gate_attempts.json"
    return root


class FakeIsolationError(RuntimeError):
    """A fake run is about to touch a formal path, or a formal run the fake root."""


def assert_fake_isolation(fake: bool) -> None:
    """Check the redirect actually happened (or did not), before anything is written.

    Cheap insurance at the dangerous write sites: the redirect is one call at the top of a
    step, and this is what notices when a later edit drops it.
    """
    if fake and not is_fake_root():
        raise FakeIsolationError(
            f"--fake run would write to the formal data root {DATA}: "
            "call paths.use_fake_data_root() before writing anything"
        )
    if not fake and is_fake_root():
        raise FakeIsolationError(
            f"formal run is pointed at the fake root {DATA}; refusing to write formal "
            "results into it"
        )
