"""Anthropic provider (evaluator E; II.9).

A request with a ``schema`` goes through ``messages.parse`` with a pydantic ``output_format``
(structured outputs); a request without one through ``messages.create``, and the caller parses
the text itself. E is the latter since 2026-09-23: it is called exactly as the retest called it
(plain prompt, own parser; see ``rafr.evaluator.judge``). The Messages
API has no ``seed``; ``request.seed`` is ignored here but still part of the cache key so that
a config change is never silently served from cache. A refusal is ``stop_reason == "refusal"``.

Sampling parameters are not available on this model generation: ``temperature`` / ``top_p`` /
``top_k`` are rejected by the API (and are not even parameters of ``messages.parse`` in the
SDK). Per decision 14 a model that does not accept ``temperature`` must fail loudly, so a
configured value raises :class:`UnsupportedDecodingParamError` here instead of being dropped; the
frozen config sets them to null (= not sent, recorded as null). Thinking depth is steered by
``decoding.effort`` -> ``output_config.effort`` instead (II.8.2 records it like any other
decoding parameter); ``messages.parse`` merges the output schema into the same
``output_config``.

Prompt caching is requested by the caller, not here: a message whose ``content`` is a list of
blocks may carry ``cache_control`` on a block, and this provider passes the blocks through
untouched. What the model receives is the concatenation of those blocks, so caching changes
billing and latency only. ``cache_creation_input_tokens`` / ``cache_read_input_tokens`` come
back on every response and go into the ledger (2026-09-20).
"""

from __future__ import annotations

from typing import Any

from rafr.llm.client import LLMRequest, LLMResponse, TransientProviderError, Usage
from rafr.llm.providers import load_api_key

_ENV_VAR = "ANTHROPIC_API_KEY"

UNSUPPORTED_SAMPLING_PARAMS: tuple[str, ...] = ("temperature", "top_p", "top_k")
EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


class UnsupportedDecodingParamError(ValueError):
    """A decoding parameter the target model does not accept (never silently dropped)."""


def decoding_params(req: LLMRequest) -> dict[str, Any]:
    """model / max_tokens / output_config / system, as sent to the SDK."""
    d = req.decoding or {}
    for name in UNSUPPORTED_SAMPLING_PARAMS:
        if d.get(name) is not None:
            raise UnsupportedDecodingParamError(
                f"{name}={d[name]!r} was configured for {req.model!r}, but this model "
                f"rejects {name}. Set it to null (= not sent, recorded as null) rather "
                "than expecting the call to silently ignore it (decision 14)."
            )
    params: dict[str, Any] = {"model": req.model}
    params["max_tokens"] = int(d.get("max_output_tokens") or 1024)
    effort = d.get("effort")
    if effort is not None:
        if effort not in EFFORT_LEVELS:
            raise UnsupportedDecodingParamError(f"effort={effort!r} is not one of {EFFORT_LEVELS}")
        params["output_config"] = {"effort": effort}
    if req.system:
        params["system"] = req.system
    return params


def as_transient(exc: BaseException) -> TransientProviderError | None:
    """The SDK error as a :class:`TransientProviderError`, or None if it is not transient.

    429 -> ``rate_limit`` (with the server's ``retry-after``, seconds, when it sends one);
    connection failures and timeouts -> ``network``; 5xx including 529 overloaded ->
    ``server``. Anything else (400, 401, 404, ...) is a real error and is not retried.
    """
    import anthropic

    if isinstance(exc, anthropic.APIConnectionError):  # includes APITimeoutError
        return TransientProviderError("network", str(exc))
    if isinstance(exc, anthropic.APIStatusError):
        status = int(exc.status_code)
        retry_after: float | None = None
        try:
            raw = exc.response.headers.get("retry-after")
            retry_after = float(raw) if raw is not None else None
        except (AttributeError, TypeError, ValueError):
            retry_after = None
        if status == 429:
            return TransientProviderError(
                "rate_limit", str(exc), retry_after=retry_after, status=status
            )
        if status >= 500:
            return TransientProviderError(
                "server", str(exc), retry_after=retry_after, status=status
            )
    return None


def request_kwargs(req: LLMRequest) -> dict[str, Any]:
    """Exactly the keyword arguments :meth:`AnthropicProvider.complete` hands to the SDK, minus
    the structured-output schema (``output_format``) when there is one. Holds no key, so it
    can be printed -- the dry run of a paid step shows one of these (2026-09-23)."""
    return {**decoding_params(req), "messages": req.messages}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None, max_retries: int = 5, timeout_s: float = 120):
        import anthropic

        key = api_key or load_api_key(_ENV_VAR)
        self._client = anthropic.Anthropic(api_key=key, max_retries=max_retries, timeout=timeout_s)
        del key

    def _params(self, req: LLMRequest) -> dict[str, Any]:
        return decoding_params(req)

    def complete(self, req: LLMRequest) -> LLMResponse:
        kwargs = request_kwargs(req)
        parsed: dict[str, Any] | None = None
        error: str | None = None
        try:
            if req.schema is not None:
                msg = self._client.messages.parse(output_format=req.schema, **kwargs)
                out = msg.parsed_output
                if out is not None:
                    parsed = out.model_dump()
            else:
                msg = self._client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - classified below, re-raised if not transient
            transient = as_transient(exc)
            if transient is None:
                raise
            raise transient from exc
        text_parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        text = "\n".join(text_parts) if text_parts else None
        refused = msg.stop_reason == "refusal"
        if req.schema is not None and parsed is None and not refused:
            error = "schema_validation: no parsed output"
        # With prompt caching on, `input_tokens` is only the uncached remainder: the cached
        # prefix is reported separately as written (1.25x) or read (0.1x). All three are
        # recorded so the ledger can price a run and so a silent cache miss is visible.
        usage = Usage(
            input_tokens=getattr(msg.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(msg.usage, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
        )
        return LLMResponse(
            text=text,
            parsed=parsed,
            refused=refused,
            stop_reason=msg.stop_reason,
            model_version=msg.model,
            usage=usage,
            prompt_hash="",
            cache_key="",
            request_id=msg.id,
            error=error,
        )
