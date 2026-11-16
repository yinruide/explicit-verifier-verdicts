"""Deterministic fake provider for tests and offline dry runs. Never touches the network.

Behaviour is scripted by a ``responder`` callable ``(request) -> str | FakeReply``; the
default responder returns a schema-valid JSON object built from the request's schema
defaults, which is enough to exercise the full pipeline offline.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from rafr.llm.client import LLMRequest, LLMResponse, Usage, content_text


@dataclass
class FakeReply:
    text: str | None
    refused: bool = False
    stop_reason: str = "stop"
    usage: Usage = field(default_factory=lambda: Usage(10, 5))


Responder = Callable[[LLMRequest], "str | FakeReply"]


def default_responder(req: LLMRequest) -> str:
    """Schema-valid JSON: controller -> rewrite with a deterministic revision; E -> UNSAFE."""
    name = req.schema.__name__ if req.schema else ""
    last_user = content_text(
        next((m["content"] for m in reversed(req.messages) if m["role"] == "user"), "")
    )
    if name == "ControllerOutput":
        return json.dumps({"action": "rewrite", "revised_text": f"[fake rewrite] {last_user[:40]}"})
    if name == "GFixedOutput":
        return json.dumps({"revised_text": f"[fake rewrite] {last_user[:40]}"})
    if name == "EvaluatorOutput":
        return json.dumps({"label": "UNSAFE", "score": 0.9})
    if req.purpose == "evaluator":  # E is plain prompt + own parser (2026-09-23): answer text
        return json.dumps({"label": "UNSAFE", "reason": "fake reason."})
    return f"[fake] {last_user[:60]}"


class FakeProvider:
    name = "fake"

    def __init__(self, responder: Responder | None = None, model_version: str = "fake-0") -> None:
        self._responder = responder or default_responder
        self._model_version = model_version
        self.calls: list[LLMRequest] = []

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls.append(req)
        reply = self._responder(req)
        if isinstance(reply, str):
            reply = FakeReply(text=reply)
        parsed: dict[str, Any] | None = None
        error: str | None = None
        if req.schema is not None and not reply.refused and reply.text:
            try:
                parsed = req.schema.model_validate_json(reply.text).model_dump()
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                error = f"schema_validation: {type(e).__name__}"
        return LLMResponse(
            text=reply.text,
            parsed=parsed,
            refused=reply.refused,
            stop_reason=reply.stop_reason,
            model_version=self._model_version,
            usage=reply.usage,
            prompt_hash="",
            cache_key="",
            request_id=f"fake-{len(self.calls)}",
            error=error,
        )
