"""Text files written by the pipeline: LF on every platform (decisions.md D-2026-09-27).

Every text-mode write in the code passes ``newline="\\n"`` (csv writers ``lineterminator="\\n"``),
so a file has the same bytes -- and the same raw-byte sha256 -- whether the laptop or the pod
wrote it; ``tests/test_line_endings.py`` scans the code for it. Hashes stay raw-byte hashes.

Files written on the laptop before 2026-09-27 are CRLF, and the sha256 values registered for them
(the manifest, prereg analysis_plan section 15, progress.md) are raw-byte hashes of those CRLF
bytes. They are left as they are. The two helpers below wrap the writes that can replace or
extend one of them, so that its raw-byte hash never changes silently: the old and the new hash
are printed to stderr.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

CRLF = b"\r\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def has_crlf(path: Path) -> bool:
    """True for an existing file that contains a CRLF line ending."""
    p = Path(path)
    return p.is_file() and CRLF in p.read_bytes()


@contextmanager
def crlf_rewrite_notice(path: Path, *, stream: TextIO | None = None) -> Iterator[None]:
    """Around a write that replaces or extends ``path``: if it existed with CRLF line endings,
    report its raw-byte sha256 before and after, so a registered hash never changes silently."""
    p = Path(path)
    before = p.read_bytes() if p.is_file() else None
    yield
    if before is None or CRLF not in before:
        return
    after = p.read_bytes() if p.is_file() else b""
    print(
        f"[line-endings] {p}: a CRLF file (written before 2026-09-27) was rewritten or extended "
        f"in LF; raw-byte sha256 {_sha256(before)} -> {_sha256(after)}. A hash registered for "
        "the old bytes no longer matches this file (decisions.md D-2026-09-27).",
        file=stream or sys.stderr,
    )


TEXT_SUFFIXES = frozenset({".yaml", ".yml", ".json", ".jsonl", ".txt", ".md", ".csv"})


def copy_tree_lf(src: Path, dst: Path) -> None:
    """``shutil.copytree`` for the configs snapshot, with text files written LF: a Windows
    checkout may hold CRLF, and what the pipeline writes into runs/ is LF (D-2026-09-27)."""
    src, dst = Path(src), Path(dst)
    for p in sorted(src.rglob("*")):
        target = dst / p.relative_to(src)
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        data = p.read_bytes()
        target.write_bytes(data.replace(CRLF, b"\n") if p.suffix in TEXT_SUFFIXES else data)


def crlf_append_notice(path: Path, *, stream: TextIO | None = None) -> None:
    """Before appending LF lines to ``path`` for a long run: if it exists with CRLF line endings,
    say so with its current raw-byte sha256 (the file will end up with mixed line endings)."""
    p = Path(path)
    if not p.is_file():
        return
    data = p.read_bytes()
    if CRLF in data:
        print(
            f"[line-endings] {p}: appending LF lines to a CRLF file (written before 2026-09-27); "
            f"its raw-byte sha256 {_sha256(data)} will change and a hash registered for it will "
            "no longer match (decisions.md D-2026-09-27).",
            file=stream or sys.stderr,
        )
