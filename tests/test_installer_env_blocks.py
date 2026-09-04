"""Tests for harness environment-block reading, merging, and writing."""

import json
import tomllib
from pathlib import Path

from code_indexing_mcp.installer.env_blocks import entry_from_text, env_from_entry, merge_env
from code_indexing_mcp.installer.harnesses import configure_harness, read_server_entry

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
        'env = { CODE_INDEXING_OFFLINE = "1" }\n'
    )
    assert entry_from_text("codex", text) == {
        "command": "/old",
        "args": ["serve"],
        "env": {"CODE_INDEXING_OFFLINE": "1"},
    }


def test_entry_from_text_returns_none_for_missing_or_invalid() -> None:
    assert entry_from_text("kimi-code", "{}\n") is None
    assert entry_from_text("kimi-code", "not json") is None
    assert entry_from_text("codex", "not = = toml") is None


def test_read_server_entry_survives_a_non_utf8_configuration(tmp_path: Path) -> None:
    """An unreadable config is worth nothing to prefill, but must not raise."""

    directory = tmp_path / ".kimi-code"
    directory.mkdir()
    (directory / "mcp.json").write_bytes(b'{"mcpServers": {"code-indexing-mcp": "\xff\xfe"}}')
    assert read_server_entry("kimi-code", home=tmp_path, environment={}) is None


def test_env_from_entry_uses_the_per_harness_key() -> None:
    assert env_from_entry("opencode", {"environment": {"A": "1"}, "env": {"B": "2"}}) == {"A": "1"}
    assert env_from_entry("kimi-code", {"env": {"B": "2"}}) == {"B": "2"}
    assert env_from_entry("kimi-code", {}) == {}


def test_merge_env_applies_updates_deletions_and_preserves_unknown_keys() -> None:
    merged = merge_env(
        {"KEEP": "x", "CODE_INDEXING_OFFLINE": "1", "CODE_INDEXING_BROKER": "off"},
        {
            "CODE_INDEXING_OFFLINE": "0",
            "CODE_INDEXING_BROKER": None,
        },
    )
    assert merged == {"KEEP": "x", "CODE_INDEXING_OFFLINE": "0"}


def test_configure_harness_writes_env_and_preserves_unmanaged_keys(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "code-indexing-mcp": {
                        "command": "/old",
                        "args": ["serve"],
                        "env": {"KEEP": "x", "CODE_INDEXING_BROKER": "off"},
                    }
                }
            }
        )
    )
    configure_harness(
        "kimi-code",
        Path(SERVER_COMMAND),
        env={"CODE_INDEXING_BROKER": None, "CODE_INDEXING_INDEX_MODE": "eager"},
        environment={"KIMI_CODE_HOME": str(tmp_path)},
    )
    entry = json.loads(config.read_text())["mcpServers"]["code-indexing-mcp"]
    assert entry == {
        "command": SERVER_COMMAND,
        "args": ["serve"],
        "env": {"KEEP": "x", "CODE_INDEXING_INDEX_MODE": "eager"},
    }


def test_configure_harness_opencode_uses_environment_key(tmp_path: Path) -> None:
    configure_harness(
        "opencode",
        Path(SERVER_COMMAND),
        env={"CODE_INDEXING_OFFLINE": "1"},
        environment={"OPENCODE_CONFIG_DIR": str(tmp_path)},
    )
    entry = json.loads((tmp_path / "opencode.json").read_text())["mcp"]["code-indexing-mcp"]
    assert entry == {
        "type": "local",
        "command": [SERVER_COMMAND, "serve"],
        "enabled": True,
        "environment": {"CODE_INDEXING_OFFLINE": "1"},
    }
    assert "env" not in entry


def test_configure_harness_antigravity_writes_gemini_config(tmp_path: Path) -> None:
    configure_harness(
        "antigravity",
        Path(SERVER_COMMAND),
        env={"CODE_INDEXING_OFFLINE": "1"},
        environment={"ANTIGRAVITY_HOME": str(tmp_path)},
    )
    entry = json.loads((tmp_path / "mcp_config.json").read_text())["mcpServers"][
        "code-indexing-mcp"
    ]
    assert entry == {
        "command": SERVER_COMMAND,
        "args": ["serve"],
        "env": {"CODE_INDEXING_OFFLINE": "1"},
    }
    reread = read_server_entry("antigravity", environment={"ANTIGRAVITY_HOME": str(tmp_path)})
    assert reread is not None
    assert env_from_entry("antigravity", reread) == {"CODE_INDEXING_OFFLINE": "1"}


def test_configure_harness_antigravity_cli_writes_mcp_config(tmp_path: Path) -> None:
    configure_harness(
        "antigravity-cli",
        Path(SERVER_COMMAND),
        env={"CODE_INDEXING_OFFLINE": "1"},
        environment={"ANTIGRAVITY_CLI_HOME": str(tmp_path)},
    )
    entry = json.loads((tmp_path / "mcp_config.json").read_text())["mcpServers"][
        "code-indexing-mcp"
    ]
    assert entry == {
        "command": SERVER_COMMAND,
        "args": ["serve"],
        "env": {"CODE_INDEXING_OFFLINE": "1"},
    }
    reread = read_server_entry(
        "antigravity-cli", environment={"ANTIGRAVITY_CLI_HOME": str(tmp_path)}
    )
    assert reread is not None
    assert env_from_entry("antigravity-cli", reread) == {"CODE_INDEXING_OFFLINE": "1"}


def test_configure_harness_muse_code_writes_settings_with_schema_version(
    tmp_path: Path,
) -> None:
    configure_harness(
        "muse-code",
        Path(SERVER_COMMAND),
        env={"CODE_INDEXING_OFFLINE": "1"},
        environment={"XDG_CONFIG_HOME": str(tmp_path)},
    )
    parsed = json.loads((tmp_path / "muse" / "settings.json").read_text())
    assert parsed["schema_version"] == 1
    assert parsed["mcpServers"]["code-indexing-mcp"] == {
        "command": SERVER_COMMAND,
        "args": ["serve"],
        "env": {"CODE_INDEXING_OFFLINE": "1"},
    }
    reread = read_server_entry("muse-code", environment={"XDG_CONFIG_HOME": str(tmp_path)})
    assert reread is not None
    assert env_from_entry("muse-code", reread) == {"CODE_INDEXING_OFFLINE": "1"}


def test_configure_harness_codex_writes_toml_env_table(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    configure_harness(
        "codex",
        Path(SERVER_COMMAND),
        env={"CODE_INDEXING_OFFLINE": "1"},
        environment={"CODEX_HOME": str(tmp_path)},
    )
    parsed = tomllib.loads(path.read_text())
    assert parsed["mcp_servers"]["code-indexing-mcp"] == {
        "command": SERVER_COMMAND,
        "args": ["serve"],
        "env": {"CODE_INDEXING_OFFLINE": "1"},
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
        env={"CODE_INDEXING_OFFLINE": "1"},
        environment={"CODEX_HOME": str(tmp_path)},
    )
    parsed = tomllib.loads(path.read_text())
    assert parsed["mcp_servers"]["code-indexing-mcp"]["env"] == {
        "KEEP": "x",
        "CODE_INDEXING_OFFLINE": "1",
    }


def test_configure_harness_without_env_reproduces_the_legacy_entries(tmp_path: Path) -> None:
    configure_harness(
        "kimi-code", Path(SERVER_COMMAND), environment={"KIMI_CODE_HOME": str(tmp_path)}
    )
    entry = json.loads((tmp_path / "mcp.json").read_text())["mcpServers"]["code-indexing-mcp"]
    assert entry == {"command": SERVER_COMMAND, "args": ["serve"]}


def test_read_server_entry_returns_none_when_unconfigured(tmp_path: Path) -> None:
    assert read_server_entry("kimi-code", environment={"KIMI_CODE_HOME": str(tmp_path)}) is None
