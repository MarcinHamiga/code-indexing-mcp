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
from typing import Any, Literal, NamedTuple

from .config_files import SERVER_NAME, InstallerError, _jsonc_as_json


class HarnessSchema(NamedTuple):
    """The common server-entry shape shared by installer read/write paths."""

    environment_key: str
    object_key: str | None
    command_style: Literal["string", "argv"] = "string"


HARNESS_SCHEMAS: dict[str, HarnessSchema] = {
    "codex": HarnessSchema("env", None),
    "claude-code": HarnessSchema("env", "mcpServers"),
    "kimi-code": HarnessSchema("env", "mcpServers"),
    "claude-desktop": HarnessSchema("env", "mcpServers"),
    "opencode": HarnessSchema("environment", "mcp", "argv"),
    "kilocode": HarnessSchema("environment", "mcp", "argv"),
    "antigravity": HarnessSchema("env", "mcpServers"),
    "antigravity-cli": HarnessSchema("env", "mcpServers"),
    "muse-code": HarnessSchema("env", "mcpServers"),
    "tabnine": HarnessSchema("env", "mcpServers"),
    "tabnine-cli": HarnessSchema("env", "mcpServers"),
}

# Keep these mappings as compatibility views for callers that only need one
# field; the registry above is the source of truth for all shared schemas.
ENV_KEYS = {slug: schema.environment_key for slug, schema in HARNESS_SCHEMAS.items()}
OBJECT_KEYS = {
    slug: schema.object_key
    for slug, schema in HARNESS_SCHEMAS.items()
    if schema.object_key is not None
}


def harness_schema(slug: str) -> HarnessSchema | None:
    return HARNESS_SCHEMAS.get(slug)


def entry_from_text(slug: str, text: str) -> dict[str, Any] | None:
    """Parse the Code Indexing MCP server entry out of a harness config's text."""
    servers: Any
    try:
        if slug == "codex":
            servers = tomllib.loads(text).get("mcp_servers")
        else:
            schema = harness_schema(slug)
            if schema is None or schema.object_key is None:
                return None
            servers = json.loads(_jsonc_as_json(text)).get(schema.object_key)
    except ValueError:
        return None
    if not isinstance(servers, dict):
        return None
    entry = servers.get(SERVER_NAME)
    return dict(entry) if isinstance(entry, dict) else None


def env_from_entry(slug: str, entry: Mapping[str, Any]) -> dict[str, str]:
    """Return the entry's environment block under this harness's key."""
    schema = harness_schema(slug)
    if schema is None:
        raise InstallerError(f"Unknown harness {slug!r}")
    raw = entry.get(schema.environment_key)
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def command_from_entry(slug: str, entry: Mapping[str, Any]) -> str | None:
    """Return the executable the entry launches, or None if it names none.

    The OpenCode-schema harnesses put the command and its arguments in one list;
    everywhere else ``command`` is the executable on its own.
    """

    raw = entry.get("command")
    if isinstance(raw, str):
        return raw or None
    if isinstance(raw, list) and raw and isinstance(raw[0], str):
        return raw[0] or None
    return None


def merge_env(existing: Mapping[str, str], updates: Mapping[str, str | None]) -> dict[str, str]:
    """Apply managed updates to an existing block; a None value deletes the key."""
    merged = dict(existing)
    for key, value in updates.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged
