"""xAI (Grok) and Alibaba DashScope (Qwen): the two panel members' providers (paper Sec. 3).

The panel re-judges Exp 2's x_T with E's own rubric-free prompt (paper Sec. 4.6); nothing else in
the pipeline calls these two providers. Both speak OpenAI-compatible Chat Completions, so the
``openai`` SDK is pointed at their base URLs, as the annotator-selection test did
(``annotation/annotator_selection/providers.py``, whose calling conventions these copy):

* one user message holding the whole rendered prompt, no system prompt, no structured output;
  the caller parses the answer text (``rafr.evaluator.judge.parse_response``);
* ``decoding.max_output_tokens`` -> ``max_tokens``; ``decoding.temperature`` is sent only when
  it is not null; ``decoding.reasoning_params`` (a dict) goes into ``extra_body`` only when it
  is non-empty -- ``{}`` sends nothing and leaves the provider default in effect;
* DashScope is STREAMED: the hybrid-thinking Qwen models serve thinking mode over a stream only;
  the chunks are re-assembled into one response;
* billed output = max(``completion_tokens``, ``total_tokens - prompt_tokens``): xAI reports
  thinking tokens outside ``completion_tokens`` (measured 2026-09-22 in the selection test).

A provider-side content block (DashScope's input moderation, ``data_inspection_failed``, a 400)
is not an exception here but a refusal: ``refused=True``, ``stop_reason="blocked"``. The caller
treats it like any refusal (two retries, then a recorded failure). 429 / connection errors /
5xx are raised as :class:`rafr.llm.client.TransientProviderError` exactly as the OpenAI provider
does, so ``rafr.llm.client`` retries them with the counted backoff.

API keys come from ``.env`` only (``XAI_API_KEY``, ``DASHSCOPE_API_KEY``; ``DASHSCOPE_BASE_URL``
optionally overrides the region endpoint), as for every other provider.
"""

from __future__ import annotations

from typing import Any

from rafr.llm.client import LLMRequest, LLMResponse, Usage, content_text
from rafr.llm.providers import load_api_key
from rafr.llm.providers.openai import as_transient

XAI_BASE_URL = "https://api.x.ai/v1"
DASHSCOPE_DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_BASE_URL_ENV = "DASHSCOPE_BASE_URL"
XAI_ENV_VAR = "XAI_API_KEY"
DASHSCOPE_ENV_VAR = "DASHSCOPE_API_KEY"

#: Substrings that mark a 4xx as a provider-side content block rather than a bug.
BLOCK_MARKERS: tuple[str, ...] = (
    "data_inspection",
    "inappropriate",
    "content_filter",
    "moderation",
    "safety",
)


def _int(x: Any) -> int:
    try:
        return int(x or 0)
    except (TypeError, ValueError):
        return 0


def billed_output_tokens(usage: Any) -> int:
    """Every output token the provider bills, thinking included (see module doc)."""
    completion = _int(getattr(usage, "completion_tokens", 0))
    total = _int(getattr(usage, "total_tokens", 0))
    prompt = _int(getattr(usage, "prompt_tokens", 0))
    return max(completion, total - prompt) if total else completion


def request_params(req: LLMRequest) -> dict[str, Any]:
    """model / max_tokens / temperature / extra_body, as handed to the SDK (no key)."""
    d = req.decoding or {}
    params: dict[str, Any] = {"model": req.model}
    if d.get("max_output_tokens") is not None:
        params["max_tokens"] = int(d["max_output_tokens"])
    if d.get("temperature") is not None:
        params["temperature"] = float(d["temperature"])
    reasoning = d.get("reasoning_params") or {}
    if not isinstance(reasoning, dict):
        raise ValueError(f"decoding.reasoning_params must be a mapping, got {reasoning!r}")
    if reasoning:
        params["extra_body"] = dict(reasoning)
    return params


def request_messages(req: LLMRequest) -> list[dict[str, str]]:
    """System prompt (if any) + the messages, each content flattened to one string."""
    out: list[dict[str, str]] = []
    if req.system:
        out.append({"role": "system", "content": req.system})
    out += [{"role": m["role"], "content": content_text(m["content"])} for m in req.messages]
    return out


def blocked(exc: BaseException) -> bool:
    """A 4xx whose text says the provider blocked the content (not a bug in the request)."""
    status = getattr(exc, "status_code", None)
    text = f"{type(exc).__name__}: {exc}".lower()
    return status is not None and 400 <= int(status) < 500 and any(m in text for m in BLOCK_MARKERS)


class OpenAICompatibleProvider:
    """Chat Completions against a non-OpenAI base URL. ``stream`` re-assembles a stream."""

    name = "openai_compatible"
    stream = False

    def __init__(
        self, *, api_key: str, base_url: str, max_retries: int = 0, timeout_s: float = 120
    ) -> None:
        import openai

        self.base_url = base_url
        self._client = openai.OpenAI(
            api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=timeout_s
        )
        del api_key

    def complete(self, req: LLMRequest) -> LLMResponse:
        params = request_params(req)
        messages = request_messages(req)
        try:
            if self.stream:
                return self._complete_stream(messages, params)
            completion = self._client.chat.completions.create(messages=messages, **params)
        except Exception as exc:  # noqa: BLE001 - classified below, re-raised if not handled
            if blocked(exc):
                return LLMResponse(
                    text=None,
                    parsed=None,
                    refused=True,
                    stop_reason="blocked",
                    model_version=None,
                    usage=Usage(),
                    prompt_hash="",
                    cache_key="",
                    request_id=getattr(exc, "request_id", None),
                    error=f"blocked: {type(exc).__name__}: {str(exc)[:300]}",
                )
            transient = as_transient(exc)
            if transient is None:
                raise
            raise transient from exc
        choice = completion.choices[0]
        msg = choice.message
        refusal = getattr(msg, "refusal", None)
        fingerprint = getattr(completion, "system_fingerprint", None)
        return LLMResponse(
            text=msg.content if msg.content is not None else refusal,
            parsed=None,
            refused=refusal is not None,
            stop_reason=choice.finish_reason,
            model_version=completion.model + (f"@{fingerprint}" if fingerprint else ""),
            usage=Usage(
                input_tokens=_int(getattr(completion.usage, "prompt_tokens", 0)),
                output_tokens=billed_output_tokens(completion.usage),
            ),
            prompt_hash="",
            cache_key="",
            request_id=completion.id,
        )

    def _complete_stream(
        self, messages: list[dict[str, str]], params: dict[str, Any]
    ) -> LLMResponse:
        chunks: list[str] = []
        refusal_parts: list[str] = []
        finish: str | None = None
        usage: Any = None
        model: str | None = None
        request_id: str | None = None
        fingerprint: str | None = None
        stream = self._client.chat.completions.create(
            messages=messages, stream=True, stream_options={"include_usage": True}, **params
        )
        for chunk in stream:
            model = getattr(chunk, "model", None) or model
            request_id = getattr(chunk, "id", None) or request_id
            fingerprint = getattr(chunk, "system_fingerprint", None) or fingerprint
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            for ch in chunk.choices or []:
                delta = ch.delta
                if delta is not None and delta.content:
                    chunks.append(delta.content)
                if delta is not None and getattr(delta, "refusal", None):
                    refusal_parts.append(delta.refusal)
                if ch.finish_reason:
                    finish = ch.finish_reason
        text = "".join(chunks) if chunks else None
        refusal = "".join(refusal_parts) if refusal_parts else None
        version = (model or "") + (f"@{fingerprint}" if fingerprint else "")
        return LLMResponse(
            text=text if text is not None else refusal,
            parsed=None,
            refused=refusal is not None,
            stop_reason=finish,
            model_version=version or None,
            usage=Usage(
                input_tokens=_int(getattr(usage, "prompt_tokens", 0)),
                output_tokens=billed_output_tokens(usage),
            ),
            prompt_hash="",
            cache_key="",
            request_id=request_id,
        )


class XAIProvider(OpenAICompatibleProvider):
    name = "xai"

    def __init__(
        self, api_key: str | None = None, max_retries: int = 0, timeout_s: float = 120
    ) -> None:
        key = api_key or load_api_key(XAI_ENV_VAR)
        super().__init__(
            api_key=key, base_url=XAI_BASE_URL, max_retries=max_retries, timeout_s=timeout_s
        )
        del key


class DashScopeProvider(OpenAICompatibleProvider):
    name = "dashscope"
    stream = True

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 0,
        timeout_s: float = 120,
        base_url: str | None = None,
    ) -> None:
        key = api_key or load_api_key(DASHSCOPE_ENV_VAR)
        super().__init__(
            api_key=key,
            base_url=base_url or dashscope_base_url(),
            max_retries=max_retries,
            timeout_s=timeout_s,
        )
        del key


def dashscope_base_url() -> str:
    """``DASHSCOPE_BASE_URL`` from ``.env`` (the key is bound to one region), else the
    international endpoint."""
    from dotenv import dotenv_values

    from rafr.core import paths

    env_path = paths.ROOT / ".env"
    env = dotenv_values(env_path) if env_path.exists() else {}
    return (env.get(DASHSCOPE_BASE_URL_ENV) or "").strip() or DASHSCOPE_DEFAULT_BASE_URL
