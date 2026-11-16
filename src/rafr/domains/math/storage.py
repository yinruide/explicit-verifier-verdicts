"""Small file utilities for math preparation and probes; experiment logs use Core."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class IntegrityError(ValueError):
    """Refuse inconsistent identities, incomplete stages or damaged artifacts."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise IntegrityError(f"cannot read JSON artifact: {path}") from exc


def atomic_json(path: Path, value: Any) -> None:
    """Same-filesystem replace; interrupted temporaries never count as artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with tmp.open("xb") as fh:
            fh.write(canonical(value) + b"\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def exclusive_lock(path: Path):
    """OS-released process lock: a crash does not leave a stale ownership claim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as fh:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            if not fh.read(1):
                fh.write(b"0")
                fh.flush()
            fh.seek(0)
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise IntegrityError("another process owns this run") from exc
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise IntegrityError("another process owns this run") from exc
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
