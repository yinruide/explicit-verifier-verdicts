"""Disk cache for LLM responses, keyed by :meth:`rafr.llm.client.LLMRequest.cache_key`.

Layout: ``<root>/<key[:2]>/<key>.json`` with the response dict plus a ``_meta`` block.
The cache never stores API keys or request headers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rafr.contracts.schemas import utc_now_iso


class LLMCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.pop("_meta", None)
        return data

    def put(self, key: str, response: dict[str, Any], meta: dict[str, Any] | None = None) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(response)
        payload["_meta"] = {"stored_at": utc_now_iso(), **(meta or {})}
        tmp = p.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=None)
        tmp.replace(p)

    def __contains__(self, key: str) -> bool:
        return self._path(key).exists()

    def __len__(self) -> int:
        return sum(1 for _ in self.root.glob("*/*.json"))
