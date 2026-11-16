"""Extension-only snapshots shaped for the authors' read-only analysis consumers."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import yaml

from pipelines.math_ext_data import validate_output
from rafr.domains.math.storage import IntegrityError, digest, file_hash
from rafr.formatter import math_feedback
from rafr.formatter.math_extended import PROMPT_DIR, PROMPT_FILES, SHARED_DIR, SHARED_FILES


def _execution_identity():
    """Reuse math's execution-file allowlist, bind content as well as the commit."""
    from rafr.domains.math.code_identity import ROOT, _execution_files
    from rafr.logging.snapshot import git_head

    try:
        head = git_head()
    except RuntimeError:
        head = os.environ.get("RAFR_GIT_HEAD")
    if not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise IntegrityError("gitless snapshot requires RAFR_GIT_HEAD")
    hashes = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in _execution_files()}
    return {"git_head": head, "execution_sha256": digest(hashes)}


def write_or_match(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise IntegrityError(f"snapshot identity changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as out:
        out.write(text)


def snapshot_configs(out: Path, ext: dict, channel: dict, *, identity: dict) -> None:
    """Caller supplies bound math channel and config; never copy detox channel values."""
    if ext.get("domain") != "math" or not channel.get("levels"):
        raise IntegrityError("math snapshot requires math ext config and bound channel levels")
    configs = {
        "ext": ext,
        "channel": {
            "degrade": {
                "levels": channel["levels"],
                "q": {"kind": "independent_resampling"},
            }
        },
    }
    out = validate_output(Path(out), allow_existing=True)
    hashes = {
        f"src/rafr/formatter/prompts/{group}/{name}": file_hash(directory / name)
        for directory, names, group in (
            (PROMPT_DIR, PROMPT_FILES, "math_ext"),
            (SHARED_DIR, SHARED_FILES, "detox"),
            (math_feedback.PROMPT_DIR, math_feedback.PROMPT_FILES, "math"),
        )
        for name in names
    }
    bound = {
        "domain": "math",
        "source_identity": identity,
        "prompt_sha256": hashes,
        "configs": configs,
        "execution": _execution_identity(),
    }
    if (out / "snapshot.json").exists():
        existing = json.loads((out / "snapshot.json").read_text(encoding="utf-8"))
        if (
            existing.get("math_extension_identity") != bound
            or existing.get("git_head") != bound["execution"]["git_head"]
        ):
            raise IntegrityError("Core snapshot identity changed")
    else:
        # Reuse Core's laptop/git-archive snapshot path without overwriting the math
        # analysis-facing config files with the default detox config tree.
        from pipelines.step12_exp2 import snapshot

        out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="math-ext-snapshot-") as directory:
            temporary = Path(directory)
            snapshot(
                temporary,
                step="math_extension",
                cfgs=configs,
                extra={"math_extension_identity": bound, "domain": "math"},
            )
            generated = json.loads((temporary / "snapshot.json").read_text(encoding="utf-8"))
            if not generated.get("git_head"):
                raise IntegrityError("gitless snapshot requires RAFR_GIT_HEAD")
            # Write to the destination filesystem, then rename atomically.
            target = out / "snapshot.json.partial"
            target.write_bytes((temporary / "snapshot.json").read_bytes())
            os.replace(target, out / "snapshot.json")
    for name, value in configs.items():
        write_or_match(out / "configs" / f"{name}.yaml", yaml.safe_dump(value, sort_keys=False))
    write_or_match(
        out / "input_snapshot.json",
        json.dumps(
            bound,
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
