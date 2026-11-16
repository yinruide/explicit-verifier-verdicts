"""Provider adapters. Each exposes ``complete(request) -> LLMResponse`` without caching.

API keys are read from ``.env`` only (never from logs, never printed). The fake provider is
the only one tests may use.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dotenv import dotenv_values
from pydantic import BaseModel

from rafr.core import paths


@runtime_checkable
class Provider(Protocol):
    name: str

    def complete(self, request: Any) -> Any: ...


class MissingAPIKeyError(RuntimeError):
    pass


def load_api_key(var_name: str, env_file: Path | None = None) -> str:
    """Read ``var_name`` from ``.env`` at the repo root. ``.env`` is the only source.

    Rationale (coding instructions, section 6): keys are never taken from the process
    environment so that they cannot leak through shell history or CI logs by accident.
    """
    env_path = env_file or (paths.ROOT / ".env")
    values = dotenv_values(env_path) if env_path.exists() else {}
    key = values.get(var_name) or ""
    if not key:
        raise MissingAPIKeyError(
            f"{var_name} not found in {env_path}. Copy .env.example to .env and fill it in."
        )
    return key


def redact(value: str) -> str:
    """Never let a key reach a log; show only that something was set."""
    return "<set>" if value else "<unset>"


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema adjusted for provider strict modes.

    Strict structured output (OpenAI) requires ``additionalProperties: false`` and every
    property listed in ``required`` at every object level; optional fields keep their
    ``null`` alternative so the model can emit ``null``.
    """
    schema = model.model_json_schema()

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
                node.pop("default", None)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(schema)
    # strict mode rejects "default" keys on properties
    for prop in schema.get("properties", {}).values():
        prop.pop("default", None)
    for d in schema.get("$defs", {}).values():
        for prop in d.get("properties", {}).values():
            prop.pop("default", None)
    return schema


def is_ci() -> bool:
    return bool(os.environ.get("CI"))
