"""Run snapshots and freeze tags (II.10).

* :func:`write_snapshot` stores ``git rev-parse HEAD``, the dirty flag, a verbatim copy of
  ``configs/``, ``sys.argv`` and timestamps into the run directory.
* :func:`create_freeze_tag` is a helper for the three freeze tags (``tau-frozen``,
  ``channel-frozen``, ``analysis-preregistered``). It creates a LOCAL annotated tag only;
  pushing is the authors' job (App. A.1).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from rafr.contracts.schemas import utc_now_iso
from rafr.core import paths
from rafr.core.textio import copy_tree_lf

FREEZE_TAGS = ("tau-frozen", "channel-frozen", "analysis-preregistered")


def _git(*args: str, cwd: Path | None = None) -> str:
    out = subprocess.run(
        ["git", *args],
        cwd=str(cwd or paths.ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def git_head(cwd: Path | None = None) -> str:
    return _git("rev-parse", "HEAD", cwd=cwd)


def git_dirty(cwd: Path | None = None) -> bool:
    return bool(_git("status", "--porcelain", cwd=cwd))


def git_describe(cwd: Path | None = None) -> str | None:
    try:
        return _git("describe", "--tags", "--always", "--dirty", cwd=cwd)
    except RuntimeError:
        return None


def write_snapshot(
    run_dir: Path,
    *,
    step: str,
    configs: dict[str, Any],
    extra: dict[str, Any] | None = None,
    config_dir: Path | None = None,
    repo: Path | None = None,
) -> dict[str, Any]:
    """Write ``snapshot.json`` and copy ``configs/`` into ``run_dir``. Returns the snapshot."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cdir = config_dir or paths.CONFIGS
    dst = run_dir / "configs"
    if dst.exists():
        shutil.rmtree(dst)
    copy_tree_lf(cdir, dst)  # LF whatever the checkout holds (D-2026-09-27)
    snap = {
        "step": step,
        "started_at": utc_now_iso(),
        "git_head": git_head(repo),
        "git_dirty": git_dirty(repo),
        "git_describe": git_describe(repo),
        "argv": sys.argv,
        "python": sys.version,
        "configs": configs,  # parsed values as loaded at launch
        **(extra or {}),
    }
    with (run_dir / "snapshot.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(snap, fh, ensure_ascii=False, indent=2, default=str)
    return snap


def create_freeze_tag(name: str, message: str, *, repo: Path | None = None) -> str:
    """Create a local annotated tag. ``name`` must be one of :data:`FREEZE_TAGS`.

    The message must say what was frozen and on which data (II.10). Never pushes.
    """
    if name not in FREEZE_TAGS:
        raise ValueError(f"{name!r} is not a freeze tag; expected one of {FREEZE_TAGS}")
    if not message.strip():
        raise ValueError("freeze tag message must not be empty")
    if git_dirty(repo):
        raise RuntimeError("working tree is dirty; commit before tagging a freeze point")
    _git("tag", "-a", name, "-m", message, cwd=repo)
    return git_head(repo)
