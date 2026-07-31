"""Environment-block handling for harness MCP server entries.

Each harness passes environment variables to a stdio MCP server under its own
key: ``env`` almost everywhere, ``environment`` for the OpenCode-schema
harnesses (OpenCode and KiloCode). Managed updates merge; unrelated keys the
user placed in the block are preserved.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from typing import Any

from .config_files import SERVER_NAME, _jsonc_as_json

ENV_KEYS: dict[str, str] = {
    "codex": "env",
    "claude-code": "env",
    "kimi-code": "env",
    "claude-desktop": "env",
    "opencode": "environment",
    "kilocode": "environment",
}

_OBJECT_KEYS: dict[str, str] = {
    "claude-code": "mcpServers",
    "kimi-code": "mcpServers",
    "claude-desktop": "mcpServers",
    "opencode": "mcp",
    "kilocode": "mcp",
}


def entry_from_text(slug: str, text: str) -> dict[str, Any] | None:
    """Parse the Code Indexing MCP server entry out of a harness config's text."""
    servers: Any
    try:
        if slug == "codex":
            servers = tomllib.loads(text).get("mcp_servers")
        else:
            object_key = _OBJECT_KEYS.get(slug)
            if object_key is None:
                return None
            servers = json.loads(_jsonc_as_json(text)).get(object_key)
    except ValueError:
        return None
    if not isinstance(servers, dict):
        return None
    entry = servers.get(SERVER_NAME)
    return dict(entry) if isinstance(entry, dict) else None


def env_from_entry(slug: str, entry: Mapping[str, Any]) -> dict[str, str]:
    """Return the entry's environment block under this harness's key."""
    raw = entry.get(ENV_KEYS[slug])
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def merge_env(existing: Mapping[str, str], updates: Mapping[str, str | None]) -> dict[str, str]:
    """Apply managed updates to an existing block; a None value deletes the key."""
    merged = dict(existing)
    for key, value in updates.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged
