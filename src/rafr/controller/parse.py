"""Parse one controller attempt (II.8.2).

Failure categories (preregistered): refusal; action not in {accept, rewrite}; action ==
rewrite with empty revised_text. ``action == rewrite`` with ``revised_text == x_t`` (under
the whitespace-normalized equality rule) is a VALID empty rewrite. ``MALFORMED_JSON`` is the
guard category for truncated output (see ``ParseFailReason``).

``refusal_like`` (decision 1c) is a flag only: a phrase-list match on ``revised_text`` never
changes the parse outcome.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rafr.contracts.schemas import Action, ParseFailReason
from rafr.generation.validity import matches_refusal, normalize_ws, texts_equal
from rafr.llm.client import LLMResponse


@dataclass(frozen=True)
class ParsedAttempt:
    action: Action | None
    revised_text: str | None
    fail_reason: ParseFailReason | None
    empty_rewrite: bool = False
    refusal_like: bool = False
    parsed: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.fail_reason is None


def parse_attempt(resp: LLMResponse, x_t: str, refusal_phrases: list[str]) -> ParsedAttempt:
    if resp.refused:
        return ParsedAttempt(None, None, ParseFailReason.REFUSAL)
    if not resp.text or not resp.text.strip():
        return ParsedAttempt(None, None, ParseFailReason.MALFORMED_JSON)
    try:
        data = json.loads(resp.text)
    except (json.JSONDecodeError, ValueError):
        return ParsedAttempt(None, None, ParseFailReason.MALFORMED_JSON)
    if not isinstance(data, dict):
        return ParsedAttempt(None, None, ParseFailReason.BAD_ACTION, parsed=None)
    action_raw = data.get("action")
    if action_raw not in (Action.ACCEPT.value, Action.REWRITE.value):
        return ParsedAttempt(None, None, ParseFailReason.BAD_ACTION, parsed=data)
    action = Action(action_raw)
    revised = data.get("revised_text")
    revised = str(revised) if revised is not None else ""
    if action is Action.ACCEPT:
        return ParsedAttempt(action, None, None, parsed=data)
    if not normalize_ws(revised):
        return ParsedAttempt(None, None, ParseFailReason.EMPTY_REVISED_TEXT, parsed=data)
    return ParsedAttempt(
        action,
        revised,
        None,
        empty_rewrite=texts_equal(revised, x_t),
        refusal_like=matches_refusal(revised, refusal_phrases),
        parsed=data,
    )
