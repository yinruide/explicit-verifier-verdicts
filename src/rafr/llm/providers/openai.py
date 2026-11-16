"""OpenAI provider (controller and G_fixed; II.8.2).

Uses Chat Completions with ``response_format = json_schema (strict)`` rather than the
Responses API, because only Chat Completions exposes ``seed``. The raw ``content`` string of
every attempt is returned untouched so that II.10's "raw output of each attempt" is on disk
even when parsing fails. A provider-level refusal is the ``message.refusal`` field.

A 429, a connection failure or a 5xx is raised as :class:`TransientProviderError`, as the
Anthropic provider does (2026-09-23): the SDK's own retries are switched off in
config (``models.controller.network.max_retries: 0``) and ``rafr.llm.client`` retries with a
counted backoff instead, so that no rate limit is absorbed unseen.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from rafr.llm.client import LLMRequest, LLMResponse, TransientProviderError, Usage
from rafr.llm.providers import load_api_key, strict_json_schema

_ENV_VAR = "OPENAI_API_KEY"

NOT_TRANSIENT_429_CODES: frozenset[str] = frozenset({"insufficient_quota"})
"""429s that waiting does not cure: OpenAI answers an exhausted account with a 429 carrying
this code. Raised as-is so the step stops at once instead of backing off for minutes."""


def as_transient(exc: BaseException) -> TransientProviderError | None:
    """The SDK error as a :class:`TransientProviderError`, or None if it is not transient.

    429 -> ``rate_limit`` (with the server's ``retry-after``, seconds, when it sends one),
    except an ``insufficient_quota`` 429; connection failures and timeouts -> ``network``;
    5xx -> ``server``. Anything else (400, 401, 404, ...) is a real error and is not retried.
    """
    import openai

    if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
        return TransientProviderError("network", str(exc))
    if isinstance(exc, openai.APIStatusError):
        status = int(exc.status_code)
        retry_after: float | None = None
        try:
            raw = exc.response.headers.get("retry-after")
            retry_after = float(raw) if raw is not None else None
        except (AttributeError, TypeError, ValueError):
            retry_after = None
        if status == 429:
            if getattr(exc, "code", None) in NOT_TRANSIENT_429_CODES:
                return None
            return TransientProviderError(
                "rate_limit", str(exc), retry_after=retry_after, status=status
            )
        if status >= 500:
            return TransientProviderError(
                "server", str(exc), retry_after=retry_after, status=status
            )
    return None


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str | None = None, max_retries: int = 5, timeout_s: float = 120):
        import openai  # local import keeps the SDK optional for tests

        key = api_key or load_api_key(_ENV_VAR)
        self._client = openai.OpenAI(api_key=key, max_retries=max_retries, timeout=timeout_s)
        del key

    def _params(self, req: LLMRequest) -> dict[str, Any]:
        d = req.decoding or {}
        params: dict[str, Any] = {"model": req.model}
        if d.get("temperature") is not None:
            params["temperature"] = float(d["temperature"])
        if d.get("top_p") is not None:
            params["top_p"] = float(d["top_p"])
        if d.get("max_output_tokens") is not None:
            params["max_completion_tokens"] = int(d["max_output_tokens"])
        if d.get("reasoning_effort") is not None:
            params["reasoning_effort"] = d["reasoning_effort"]
        if req.seed is not None:
            params["seed"] = int(req.seed)
        return params

    def complete(self, req: LLMRequest) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages += req.messages
        params = self._params(req)
        if req.schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": req.schema.__name__,
                    "schema": strict_json_schema(req.schema),
                    "strict": True,
                },
            }
        try:
            completion = self._client.chat.completions.create(messages=messages, **params)
        except Exception as exc:  # noqa: BLE001 - classified below, re-raised if not transient
            transient = as_transient(exc)
            if transient is None:
                raise
            raise transient from exc
        choice = completion.choices[0]
        msg = choice.message
        refused = msg.refusal is not None
        text = msg.content if msg.content is not None else msg.refusal
        parsed: dict[str, Any] | None = None
        error: str | None = None
        if req.schema is not None and not refused and text:
            try:
                parsed = req.schema.model_validate_json(text).model_dump()
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                error = f"schema_validation: {type(e).__name__}"
        usage = Usage(
            input_tokens=getattr(completion.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(completion.usage, "completion_tokens", 0) or 0,
        )
        fingerprint = getattr(completion, "system_fingerprint", None)
        version = completion.model + (f"@{fingerprint}" if fingerprint else "")
        return LLMResponse(
            text=text,
            parsed=parsed,
            refused=refused,
            stop_reason=choice.finish_reason,
            model_version=version,
            usage=usage,
            prompt_hash="",  # filled by LLMClient
            cache_key="",
            request_id=completion.id,
            error=error,
        )
