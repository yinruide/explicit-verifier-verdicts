"""Existing archive identity utility, relocated during the Core migration.

Identity/CRLF engineering is reserved for the authors; behavior is unchanged here.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from rafr.domains.math.storage import IntegrityError, digest, file_hash, read_json

ROOT = Path(__file__).resolve().parents[4]
ARCHIVE_NAME = "ARCHIVE_IDENTITY.json"


def _execution_files():
    files = []
    for top in ("src", "pipelines", "env"):
        for path in (ROOT / top).rglob("*"):
            # Editable setuptools installations regenerate these metadata files.
            # They are neither reviewed source nor portable between installations;
            # dependency versions are bound separately. Ignore only the known
            # metadata names, so unexpected executable files remain detectable.
            if path.parent == ROOT / "src" / "rafr.egg-info" and path.name in {
                "PKG-INFO",
                "SOURCES.txt",
                "dependency_links.txt",
                "requires.txt",
                "top_level.txt",
                "entry_points.txt",
                "zip-safe",
                "not-zip-safe",
            }:
                continue
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            ):
                if not path.resolve().is_relative_to(ROOT.resolve()):
                    raise IntegrityError("execution source may not escape the code root")
                files.append(path)
    if (ROOT / "pyproject.toml").is_file():
        files.append(ROOT / "pyproject.toml")
    files.extend(ROOT.glob("*.py"))
    return sorted(files)


def code_identity():
    files = _execution_files()
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in files}
    if (ROOT / ".git").exists():
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise IntegrityError("the code root is not a readable full Git checkout") from exc
    else:
        if not (ROOT / ARCHIVE_NAME).is_file():
            raise IntegrityError(
                "use a full Git checkout or a reviewed archive with ARCHIVE_IDENTITY.json; "
                "an ordinary copied source folder cannot establish code identity"
            )
        archive = read_json(ROOT / ARCHIVE_NAME)
        if (
            archive.get("schema") != "math-code-archive-v1"
            or not isinstance(archive.get("git_head"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", archive["git_head"])
            or not hashes
            or archive.get("files") != hashes
        ):
            raise IntegrityError(
                "archive identity mismatch: use a full clean reviewed Git checkout or an "
                "externally reviewed git archive with exact ARCHIVE_IDENTITY.json hashes"
            )
        commit = archive["git_head"]
    return {"git_head": commit, "source_sha256": digest(hashes), "files": hashes}
