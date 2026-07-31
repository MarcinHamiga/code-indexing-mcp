"""Tests for harness environment-block reading, merging, and writing."""

import json
import tomllib
from pathlib import Path

from incode_mcp.installer.env_blocks import entry_from_text, env_from_entry, merge_env
from incode_mcp.installer.harnesses import configure_harness, read_server_entry

SERVER_COMMAND = str(Path("/opt/ci-mcp"))


def test_entry_from_text_reads_jsonc_with_comments() -> None:
    text = (
        '{\n  // mine\n  "mcpServers": {"code-indexing-mcp": '
        '{"command": "/old", "args": ["serve"],}}\n}\n'
    )
    entry = entry_from_text("kimi-code", text)
    assert entry == {"command": "/old", "args": ["serve"]}


def test_entry_from_text_reads_codex_toml() -> None:
    text = (
        '[mcp_servers.code-indexing-mcp]\ncommand = "/old"\nargs = ["serve"]\n'
        'env = { INCODE_OFFLINE = "1" }\n'
    )
    assert entry_from_text("codex", text) == {
        "command": "/old",
        "args": ["serve"],
        "env": {"INCODE_OFFLINE": "1"},
    }


def test_entry_from_text_returns_none_for_missing_or_invalid() -> None:
    assert entry_from_text("kimi-code", "{}\n") is None
    assert entry_from_text("kimi-code", "not json") is None
    assert entry_from_text("codex", "not = = toml") is None


def test_env_from_entry_uses_the_per_harness_key() -> None:
    assert env_from_entry("opencode", {"environment": {"A": "1"}, "env": {"B": "2"}}) == {"A": "1"}
    assert env_from_entry("kimi-code", {"env": {"B": "2"}}) == {"B": "2"}
    assert env_from_entry("kimi-code", {}) == {}


def test_merge_env_applies_updates_deletions_and_preserves_unknown_keys() -> None:
    merged = merge_env(
        {"KEEP": "x", "INCODE_OFFLINE": "1", "INCODE_BROKER": "off"},
        {
            "INCODE_OFFLINE": "0",
            "INCODE_BROKER": None,
        },
    )
    assert merged == {"KEEP": "x", "INCODE_OFFLINE": "0"}


def test_configure_harness_writes_env_and_preserves_unmanaged_keys(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "code-indexing-mcp": {
                        "command": "/old",
                        "args": ["serve"],
                        "env": {"KEEP": "x", "INCODE_BROKER": "off"},
                    }
                }
            }
        )
    )
    configure_harness(
        "kimi-code",
        Path(SERVER_COMMAND),
        env={"INCODE_BROKER": None, "INCODE_INDEX_MODE": "eager"},
        environment={"KIMI_CODE_HOME": str(tmp_path)},
    )
    entry = json.loads(config.read_text())["mcpServers"]["code-indexing-mcp"]
    assert entry == {
        "command": SERVER_COMMAND,
        "args": ["serve"],
        "env": {"KEEP": "x", "INCODE_INDEX_MODE": "eager"},
    }


def test_configure_harness_opencode_uses_environment_key(tmp_path: Path) -> None:
    configure_harness(
        "opencode",
        Path(SERVER_COMMAND),
        env={"INCODE_OFFLINE": "1"},
        environment={"OPENCODE_CONFIG_DIR": str(tmp_path)},
    )
    entry = json.loads((tmp_path / "opencode.json").read_text())["mcp"]["code-indexing-mcp"]
    assert entry == {
        "type": "local",
        "command": [SERVER_COMMAND, "serve"],
        "enabled": True,
        "environment": {"INCODE_OFFLINE": "1"},
    }
    assert "env" not in entry


def test_configure_harness_codex_writes_toml_env_table(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    configure_harness(
        "codex",
        Path(SERVER_COMMAND),
        env={"INCODE_OFFLINE": "1"},
        environment={"CODEX_HOME": str(tmp_path)},
    )
    parsed = tomllib.loads(path.read_text())
    assert parsed["mcp_servers"]["code-indexing-mcp"] == {
        "command": SERVER_COMMAND,
        "args": ["serve"],
        "env": {"INCODE_OFFLINE": "1"},
    }


def test_configure_harness_codex_update_preserves_unmanaged_env(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[mcp_servers.code-indexing-mcp]\ncommand = "/old"\nargs = ["serve"]\n'
        'env = { KEEP = "x" }\n'
    )
    configure_harness(
        "codex",
        Path(SERVER_COMMAND),
        env={"INCODE_OFFLINE": "1"},
        environment={"CODEX_HOME": str(tmp_path)},
    )
    parsed = tomllib.loads(path.read_text())
    assert parsed["mcp_servers"]["code-indexing-mcp"]["env"] == {
        "KEEP": "x",
        "INCODE_OFFLINE": "1",
    }


def test_configure_harness_without_env_reproduces_the_legacy_entries(tmp_path: Path) -> None:
    configure_harness(
        "kimi-code", Path(SERVER_COMMAND), environment={"KIMI_CODE_HOME": str(tmp_path)}
    )
    entry = json.loads((tmp_path / "mcp.json").read_text())["mcpServers"]["code-indexing-mcp"]
    assert entry == {"command": SERVER_COMMAND, "args": ["serve"]}


def test_read_server_entry_returns_none_when_unconfigured(tmp_path: Path) -> None:
    assert read_server_entry("kimi-code", environment={"KIMI_CODE_HOME": str(tmp_path)}) is None
