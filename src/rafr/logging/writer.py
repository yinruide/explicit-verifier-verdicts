"""Append-only JSONL writer with per-row flush (a crash never loses a completed row)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rafr.core.textio import crlf_append_notice


def _to_jsonable(row: Any) -> Any:
    if isinstance(row, BaseModel):
        return row.model_dump(mode="json")
    return row


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        crlf_append_notice(self.path)  # a CRLF file from before 2026-09-27 (D-2026-09-27)
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        self.count = 0

    def write(self, row: Any) -> None:
        self._fh.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")
        self._fh.flush()
        self.count += 1

    def write_many(self, rows: Any) -> None:
        for r in rows:
            self.write(r)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl_models(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    return [model.model_validate(r) for r in read_jsonl(path)]
