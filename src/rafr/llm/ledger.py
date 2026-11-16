"""Token ledger. One JSONL row per LLM call (cached calls are rows with zero new tokens).

No prices (decision, 2026-09-17): the ledger records what a run actually produces —
calls, cache hits and input / output tokens per purpose. Turning tokens into dollars needed a
price list in the frozen config, which meant no formal step could run until four more numbers
had been typed in, all for a multiplication that can be done afterwards.

Two kinds of "cached" meet here and must not be confused:

* ``cached`` (the column) — the response came from **our** on-disk cache in
  ``data/cache/llm``. No request left the machine, so nothing was bought; the row
  replays the token counts of the original call so that the ledger still says what the call
  would cost to reproduce.
* ``cache_creation_input_tokens`` / ``cache_read_input_tokens`` — **provider-side** prompt
  caching on a live call (Anthropic). Both are input tokens that were really sent and really
  billed, only at a different rate (write 1.25x, read 0.1x of the input price). A live call
  with a cache read is NOT a ``cached`` row.

``Ledger.totals`` keeps the same numbers in memory as they are written, so a long step can
print running cost without re-reading the file on every call (2026-09-20).
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rafr.contracts.schemas import utc_now_iso

TOKEN_FIELDS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


@dataclass(frozen=True)
class Usage:
    """Token counts of one call, in the provider's own accounting.

    ``input_tokens`` is the **uncached remainder** when prompt caching is in play: the full
    prompt is ``input_tokens + cache_creation_input_tokens + cache_read_input_tokens``
    (:attr:`prompt_tokens`). Providers without prompt caching leave the two cache fields 0.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        """Every input token of the call, whatever rate it was billed at."""
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.output_tokens


@dataclass
class Totals:
    """Running totals of one :class:`Ledger` instance (this process, this run).

    Token counts come from **live** calls only: a row served from our on-disk cache bought
    nothing, so adding its replayed tokens here would invent spending that never happened.
    ``cached_calls`` counts those rows so the two are still visible separately.
    """

    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    by_purpose: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, row: dict[str, Any]) -> None:
        self.calls += 1
        per = self.by_purpose.setdefault(
            row["purpose"], {"calls": 0, "cached_calls": 0, **dict.fromkeys(TOKEN_FIELDS, 0)}
        )
        per["calls"] += 1
        if row.get("cached"):
            self.cached_calls += 1
            per["cached_calls"] += 1
            return
        for k in TOKEN_FIELDS:
            v = int(row.get(k) or 0)
            setattr(self, k, getattr(self, k) + v)
            per[k] += v

    @property
    def live_calls(self) -> int:
        return self.calls - self.cached_calls


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.totals = Totals()
        self._lock = threading.Lock()

    def record(
        self,
        *,
        purpose: str,
        provider: str,
        model: str,
        model_version: str | None,
        usage: Usage,
        cached: bool,
        cache_key: str | None = None,
    ) -> dict[str, Any]:
        row = {
            "ts": utc_now_iso(),
            "purpose": purpose,
            "provider": provider,
            "model": model,
            "model_version": model_version,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            # provider-side prompt caching (Anthropic); 0 everywhere else
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens,
            "cached": cached,  # served from data/cache/llm — nothing was bought
            "cache_key": cache_key,
        }
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:  # step00 judges with parallel workers (2026-09-23)
            with self.path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(line)
            self.totals.add(row)
        return row

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def summary(self) -> dict[str, dict[str, Any]]:
        """Per-purpose totals read back from the file: calls, cached calls, every token field."""
        agg: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "cached_calls": 0, **dict.fromkeys(TOKEN_FIELDS, 0)}
        )
        for r in self.rows():
            a = agg[r["purpose"]]
            a["calls"] += 1
            a["cached_calls"] += int(bool(r["cached"]))
            for k in TOKEN_FIELDS:
                a[k] += int(r.get(k) or 0)
        return {k: dict(a) for k, a in agg.items()}
