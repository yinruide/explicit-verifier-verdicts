"""Unified LLM client (paper Sec. 4.6 structured output, reproducibility).

Flow: ``LLMRequest`` -> cache lookup (key = model + version + decoding + seed + prompt hash +
attempt) -> provider call -> cache write -> ledger row -> ``LLMResponse``.

Network / rate-limit retries are NOT the parse retries of paper Sec. 4.6 (which the caller drives by
issuing a new request with ``attempt + 1``) and never count as one. Two layers exist:

* inside the provider SDKs (``network.max_retries``), invisible here -- switched off (0) in
  the frozen config for both providers since 2026-09-23;
* here, for a provider that raises :class:`TransientProviderError` (both the Anthropic
  provider, i.e. E, and the OpenAI one, i.e. G_fixed and the controller, since 2026-09-23):
  :meth:`LLMClient.complete` re-sends the same request after an exponential backoff
  (``network.transient_retries`` / ``backoff_base_s`` / ``backoff_cap_s``), counts every
  event in :attr:`LLMClient.transient_stats`, and raises :class:`TransientRetriesExhaustedError`
  only when the retries run out. The request and its cache key are unchanged by a retry: the
  network block is not part of the key.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rafr.llm.cache import LLMCache
from rafr.llm.ledger import Ledger, Usage
from rafr.llm.providers import Provider


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fill_slots(template: str, values: Mapping[str, str]) -> str:
    """Fill ``{name}`` slots in ONE left-to-right scan of the template.

    Every slot is replaced by looking at the template's own text, never at the text just
    inserted, so a value that happens to contain ``{x0}`` or ``{text}`` (a sentence under
    test, a checker output, a scale) is copied verbatim and can neither pull in another
    field nor be expanded twice (review, 2026-09-18: chained ``str.replace`` calls did
    exactly that). Only the names in ``values`` are slots; any other ``{...}`` -- e.g. the
    JSON example in a prompt -- is left untouched.
    """
    if not values:
        return template
    pattern = re.compile("|".join(re.escape("{" + k + "}") for k in values))
    return pattern.sub(lambda m: values[m.group(0)[1:-1]], template)


def content_text(content: Any) -> str:
    """The plain text of a message ``content``, whether it is a string or a block list.

    A content block list (used when part of a prompt carries a ``cache_control`` marker) is
    concatenated in order, so the result is the exact text the model receives either way.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(b.get("text", "")) for b in content if isinstance(b, Mapping))
    return str(content)


@dataclass
class LLMRequest:
    """Everything that determines one call. All fields go into the cache key except ``purpose``
    and ``network`` (they do not change the model's output)."""

    purpose: str  # "controller" | "g_fixed" | "evaluator" | "dry_run" | ...
    provider: str  # "openai" | "anthropic" | "fake"
    model: str
    version: str | None
    # [{"role": "user"|"assistant", "content": ...}]; ``content`` is a plain string, or a list
    # of provider content blocks when a call needs per-block markers -- E's judge prompt splits
    # into a cacheable prefix block plus the text under judgment (2026-09-20). The blocks are
    # part of the prompt hash like any other prompt bytes.
    messages: list[dict[str, Any]]
    system: str | None = None
    schema: type[BaseModel] | None = None
    decoding: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    attempt: int = 0
    api: str | None = None  # provider-specific: "chat_completions" | "messages"
    network: dict[str, Any] = field(default_factory=dict)

    @property
    def schema_name(self) -> str | None:
        return self.schema.__name__ if self.schema else None

    def prompt_hash(self) -> str:
        """Hash of system + messages + output schema (``prompt hash``)."""
        payload = {
            "system": self.system,
            "messages": self.messages,
            "schema": self.schema.model_json_schema() if self.schema else None,
        }
        return sha256_hex(canonical_json(payload))

    def cache_key(self) -> str:
        """model & version, all decoding params, seed, prompt hash, attempt index."""
        payload = {
            "provider": self.provider,
            "api": self.api,
            "model": self.model,
            "version": self.version,
            "decoding": self.decoding,
            "seed": self.seed,
            "prompt_hash": self.prompt_hash(),
            "attempt": self.attempt,
        }
        return sha256_hex(canonical_json(payload))


@dataclass
class LLMResponse:
    text: str | None  # raw model text (None only if the provider returned no text at all)
    parsed: dict[str, Any] | None  # schema-validated JSON, None if absent/invalid/refused
    refused: bool  # provider-level refusal signal
    stop_reason: str | None
    model_version: str | None  # model id / fingerprint as reported by the server
    usage: Usage
    prompt_hash: str
    cache_key: str
    cached: bool = False
    request_id: str | None = None
    error: str | None = None  # non-raising provider errors (e.g. content filter)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["usage"] = asdict(self.usage)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LLMResponse:
        d = dict(d)
        d["usage"] = Usage(**d.get("usage", {}))
        return cls(**d)


TRANSIENT_KINDS: tuple[str, ...] = ("rate_limit", "network", "server")


class TransientProviderError(RuntimeError):
    """A provider call that failed for a reason unrelated to the request: rate limited (429),
    the connection failed or timed out, or the server erred (5xx / overloaded). Re-sending the
    same request is the remedy; the model's answer was never seen."""

    def __init__(
        self,
        kind: str,
        message: str = "",
        *,
        retry_after: float | None = None,
        status: int | None = None,
    ) -> None:
        if kind not in TRANSIENT_KINDS:
            raise ValueError(f"kind must be one of {TRANSIENT_KINDS}, got {kind!r}")
        self.kind, self.retry_after, self.status = kind, retry_after, status
        super().__init__(f"{kind}{f' ({status})' if status else ''}: {message}")


class TransientRetriesExhaustedError(RuntimeError):
    """The transient retries of one request ran out. Not a judgment and not a parse failure:
    a step stops and is resumed later (the request is re-sent then)."""

    def __init__(self, last: TransientProviderError, attempts: int) -> None:
        self.last, self.attempts = last, attempts
        super().__init__(f"gave up after {attempts} transient failures; last: {last}")


@dataclass(frozen=True)
class BackoffPolicy:
    """Exponential backoff for :class:`TransientProviderError` (from ``request.network``)."""

    retries: int = 0  # 0 = no retries at this layer (the SDK's own may still apply)
    base_s: float = 2.0
    cap_s: float = 60.0
    jitter: float = 0.2  # +-20%, so parallel workers do not retry in lockstep

    @classmethod
    def from_network(cls, network: Mapping[str, Any] | None) -> BackoffPolicy:
        n = network or {}
        return cls(
            retries=int(n.get("transient_retries", 0) or 0),
            base_s=float(n.get("backoff_base_s", 2.0)),
            cap_s=float(n.get("backoff_cap_s", 60.0)),
        )

    def delay(self, k: int, retry_after: float | None = None, u: float = 0.5) -> float:
        """Wait before retry ``k`` (0-based): base * 2**k, capped, jittered by ``u`` in [0, 1],
        and never shorter than the server's own ``retry-after``."""
        d = min(self.cap_s, self.base_s * (2**k)) * (1 + self.jitter * (2 * u - 1))
        return max(d, float(retry_after or 0.0))


class LLMClient:
    """Provider-agnostic entry point. One instance per process; providers are registered
    by name so that tests can register only the fake one. Safe to share between threads:
    the cache writes one file per key, and the ledger and the transient counters lock."""

    def __init__(
        self,
        providers: dict[str, Provider],
        cache: LLMCache | None = None,
        ledger: Ledger | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        transient_log: Path | None = None,
    ) -> None:
        self._providers = providers
        self._cache = cache
        self._ledger = ledger
        self._sleep = sleep
        self._transient_log = Path(transient_log) if transient_log else None
        self._lock = threading.Lock()
        self._rand = random.SystemRandom()
        self.transient_stats: dict[str, float] = {
            **dict.fromkeys(TRANSIENT_KINDS, 0),
            "retries": 0,
            "exhausted": 0,
            "wait_s": 0.0,
        }
        """Every transient event of this client, by kind; ``retries`` = re-sends made,
        ``exhausted`` = requests given up on, ``wait_s`` = total backoff slept."""

    def set_transient_log(self, path: Path | None) -> None:
        """Append one JSON row per transient event (429 / network / 5xx) to ``path``."""
        self._transient_log = Path(path) if path else None

    @property
    def providers(self) -> dict[str, Provider]:
        return dict(self._providers)

    @property
    def ledger(self) -> Ledger | None:
        """The run's ledger, for steps that print running token counts / cost as they go."""
        return self._ledger

    def complete(self, request: LLMRequest, *, use_cache: bool = True) -> LLMResponse:
        key = request.cache_key()
        if use_cache and self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                resp = LLMResponse.from_dict(hit)
                resp.cached = True
                self._record(request, resp)
                return resp
        provider = self._providers.get(request.provider)
        if provider is None:
            raise KeyError(f"no provider registered under {request.provider!r}")
        resp = self._call_with_backoff(provider, request)
        resp.prompt_hash = request.prompt_hash()
        resp.cache_key = key
        resp.cached = False
        if use_cache and self._cache is not None:
            self._cache.put(key, resp.to_dict(), meta={"purpose": request.purpose})
        self._record(request, resp)
        return resp

    def _call_with_backoff(self, provider: Provider, request: LLMRequest) -> LLMResponse:
        """``provider.complete``, re-sent unchanged after a transient failure (see module doc)."""
        policy = BackoffPolicy.from_network(request.network)
        k = 0
        while True:
            try:
                return provider.complete(request)
            except TransientProviderError as e:
                give_up = k >= policy.retries
                wait = 0.0 if give_up else policy.delay(k, e.retry_after, self._rand.random())
                self._note_transient(request, e, k, wait, give_up)
                if give_up:
                    raise TransientRetriesExhaustedError(e, k + 1) from e
                self._sleep(wait)
                k += 1

    def _note_transient(
        self, request: LLMRequest, e: TransientProviderError, k: int, wait: float, give_up: bool
    ) -> None:
        with self._lock:
            s = self.transient_stats
            s[e.kind] += 1
            s["exhausted" if give_up else "retries"] += 1
            s["wait_s"] += wait
            if self._transient_log is not None:
                row = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "purpose": request.purpose,
                    "kind": e.kind,
                    "status": e.status,
                    "retry_after": e.retry_after,
                    "retry_index": k,
                    "wait_s": round(wait, 3),
                    "gave_up": give_up,
                    "error": str(e)[:300],
                }
                self._transient_log.parent.mkdir(parents=True, exist_ok=True)
                with self._transient_log.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _record(self, request: LLMRequest, resp: LLMResponse) -> None:
        if self._ledger is None:
            return
        self._ledger.record(
            purpose=request.purpose,
            provider=request.provider,
            model=request.model,
            model_version=resp.model_version,
            usage=resp.usage,
            cached=resp.cached,
            cache_key=resp.cache_key,
        )
