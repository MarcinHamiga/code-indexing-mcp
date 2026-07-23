import importlib.util
import json
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

INSTALLER_PATH = Path(__file__).parents[1] / "install.py"


def load_installer() -> ModuleType:
    assert INSTALLER_PATH.exists(), "install.py does not exist"
    spec = importlib.util.spec_from_file_location("code_indexing_mcp_installer", INSTALLER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_merge_creates_parent_and_top_level_object(tmp_path: Path) -> None:
    installer = load_installer()
    path = tmp_path / "nested" / "config.json"

    changed = installer.merge_json_object_entry(
        path,
        "mcpServers",
        "code-indexing-mcp",
        {"command": "/opt/ci-mcp", "args": ["serve"]},
    )

    assert changed is True
    assert json.loads(path.read_text()) == {
        "mcpServers": {
            "code-indexing-mcp": {
                "command": "/opt/ci-mcp",
                "args": ["serve"],
            }
        }
    }
    assert not path.with_name("config.json.bak").exists()


def test_jsonc_merge_preserves_comments_trailing_commas_and_unrelated_entries(
    tmp_path: Path,
) -> None:
    installer = load_installer()
    path = tmp_path / "config.jsonc"
    original = """{
  // This setting belongs to the user.
  "theme": "dark",
  "mcp": {
    "existing": {
      "enabled": false,
    },
    "code-indexing-mcp": {"old": true}, // keep this note
  },
}
"""
    path.write_text(original)

    changed = installer.merge_json_object_entry(
        path,
        "mcp",
        "code-indexing-mcp",
        {
            "type": "local",
            "command": ["/opt/ci-mcp", "serve"],
            "enabled": True,
        },
    )

    updated = path.read_text()
    assert changed is True
    assert "// This setting belongs to the user." in updated
    assert '"theme": "dark"' in updated
    assert '"existing": {' in updated
    assert "// keep this note" in updated
    assert '"old"' not in updated
    assert updated.count('"code-indexing-mcp"') == 1
    assert path.with_name("config.jsonc.bak").read_text() == original


def test_jsonc_merge_is_idempotent(tmp_path: Path) -> None:
    installer = load_installer()
    path = tmp_path / "config.jsonc"
    entry = {"command": "/opt/ci-mcp", "args": ["serve"]}

    assert installer.merge_json_object_entry(path, "mcpServers", "server", entry) is True
    first = path.read_text()
    assert installer.merge_json_object_entry(path, "mcpServers", "server", entry) is False

    assert path.read_text() == first


def test_jsonc_merge_rejects_invalid_input_without_modifying_it(tmp_path: Path) -> None:
    installer = load_installer()
    path = tmp_path / "config.jsonc"
    original = '{"mcp": [}'
    path.write_text(original)

    with pytest.raises(installer.InstallerError, match="Invalid JSON"):
        installer.merge_json_object_entry(path, "mcp", "server", {"enabled": True})

    assert path.read_text() == original
    assert not path.with_name("config.jsonc.bak").exists()


def test_codex_merge_creates_server_table(tmp_path: Path) -> None:
    installer = load_installer()
    path = tmp_path / "config.toml"

    changed = installer.merge_codex_server(path, Path("/opt/ci-mcp"))

    assert changed is True
    assert path.read_text() == (
        "[mcp_servers.code-indexing-mcp]\n"
        'command = "/opt/ci-mcp"\n'
        'args = ["serve"]\n'
    )


def test_codex_merge_replaces_only_target_table_and_subtables(tmp_path: Path) -> None:
    installer = load_installer()
    path = tmp_path / "config.toml"
    original = """# Keep this comment.
model = "gpt-5"

[mcp_servers.other]
command = "other"

[mcp_servers.code-indexing-mcp]
command = "old"
args = ["old"]

[mcp_servers.code-indexing-mcp.env]
OLD = "value"

[features]
example = true
"""
    path.write_text(original)

    changed = installer.merge_codex_server(path, Path("/new/ci-mcp"))

    updated = path.read_text()
    assert changed is True
    assert "# Keep this comment." in updated
    assert '[mcp_servers.other]\ncommand = "other"' in updated
    assert "[features]\nexample = true" in updated
    assert 'command = "old"' not in updated
    assert "OLD" not in updated
    assert updated.count("[mcp_servers.code-indexing-mcp]") == 1
    assert 'command = "/new/ci-mcp"' in updated
    assert path.with_name("config.toml.bak").read_text() == original


def test_codex_merge_is_idempotent(tmp_path: Path) -> None:
    installer = load_installer()
    path = tmp_path / "config.toml"

    assert installer.merge_codex_server(path, Path("/opt/ci-mcp")) is True
    first = path.read_text()
    assert installer.merge_codex_server(path, Path("/opt/ci-mcp")) is False

    assert path.read_text() == first


def test_harness_menu_combines_codex_cli_and_desktop() -> None:
    installer = load_installer()

    assert [(choice.slug, choice.label) for choice in installer.HARNESS_CHOICES] == [
        ("codex", "Codex (CLI + Desktop)"),
        ("claude-code", "Claude Code"),
        ("kimi-code", "Kimi Code"),
        ("claude-desktop", "Claude Desktop"),
        ("opencode", "OpenCode"),
        ("kilocode", "KiloCode"),
    ]


def test_harness_selection_accepts_numbers_slugs_duplicates_and_all() -> None:
    installer = load_installer()

    assert installer.parse_harness_selection("1, 3, codex, opencode") == [
        "codex",
        "kimi-code",
        "opencode",
    ]
    assert installer.parse_harness_selection("all") == [
        choice.slug for choice in installer.HARNESS_CHOICES
    ]
    assert installer.parse_harness_selection("") == []
    with pytest.raises(installer.InstallerError, match="Unknown harness"):
        installer.parse_harness_selection("7")


def test_configuration_paths_honor_client_home_overrides(tmp_path: Path) -> None:
    installer = load_installer()
    environment = {
        "CODEX_HOME": str(tmp_path / "codex-home"),
        "KIMI_CODE_HOME": str(tmp_path / "kimi-home"),
        "OPENCODE_CONFIG": str(tmp_path / "custom-opencode.jsonc"),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
        "APPDATA": str(tmp_path / "appdata"),
    }

    assert installer.configuration_path(
        "codex", home=tmp_path, environment=environment, platform_name="darwin"
    ) == tmp_path / "codex-home" / "config.toml"
    assert installer.configuration_path(
        "kimi-code", home=tmp_path, environment=environment, platform_name="darwin"
    ) == tmp_path / "kimi-home" / "mcp.json"
    assert installer.configuration_path(
        "opencode", home=tmp_path, environment=environment, platform_name="darwin"
    ) == tmp_path / "custom-opencode.jsonc"
    assert installer.configuration_path(
        "kilocode", home=tmp_path, environment=environment, platform_name="darwin"
    ) == tmp_path / "xdg" / "kilo" / "kilo.jsonc"
    assert installer.configuration_path(
        "claude-desktop", home=tmp_path, environment=environment, platform_name="win32"
    ) == tmp_path / "appdata" / "Claude" / "claude_desktop_config.json"


@pytest.mark.parametrize(
    ("slug", "object_key", "relative_path", "expected_entry"),
    [
        (
            "claude-code",
            "mcpServers",
            ".claude.json",
            {
                "type": "stdio",
                "command": "/opt/ci-mcp",
                "args": ["serve"],
                "env": {},
            },
        ),
        (
            "kimi-code",
            "mcpServers",
            ".kimi-code/mcp.json",
            {"command": "/opt/ci-mcp", "args": ["serve"]},
        ),
        (
            "claude-desktop",
            "mcpServers",
            "Library/Application Support/Claude/claude_desktop_config.json",
            {"command": "/opt/ci-mcp", "args": ["serve"]},
        ),
        (
            "opencode",
            "mcp",
            ".config/opencode/opencode.json",
            {
                "type": "local",
                "command": ["/opt/ci-mcp", "serve"],
                "enabled": True,
            },
        ),
        (
            "kilocode",
            "mcp",
            ".config/kilo/kilo.jsonc",
            {
                "type": "local",
                "command": ["/opt/ci-mcp", "serve"],
                "enabled": True,
            },
        ),
    ],
)
def test_configure_json_harnesses(
    tmp_path: Path,
    slug: str,
    object_key: str,
    relative_path: str,
    expected_entry: dict[str, object],
) -> None:
    installer = load_installer()

    path = installer.configure_harness(
        slug,
        Path("/opt/ci-mcp"),
        home=tmp_path,
        environment={},
        platform_name="darwin",
    )

    assert path == tmp_path / relative_path
    parsed = json.loads(path.read_text())
    assert parsed[object_key]["code-indexing-mcp"] == expected_entry


def test_configure_codex_uses_shared_toml(tmp_path: Path) -> None:
    installer = load_installer()

    path = installer.configure_harness(
        "codex",
        Path("/opt/ci-mcp"),
        home=tmp_path,
        environment={},
        platform_name="darwin",
    )

    assert path == tmp_path / ".codex" / "config.toml"
    parsed = tomllib.loads(path.read_text())
    assert parsed["mcp_servers"]["code-indexing-mcp"] == {
        "command": "/opt/ci-mcp",
        "args": ["serve"],
    }


def test_claude_desktop_rejects_platform_without_standard_config(tmp_path: Path) -> None:
    installer = load_installer()

    with pytest.raises(installer.InstallerError, match="not supported on linux"):
        installer.configuration_path(
            "claude-desktop",
            home=tmp_path,
            environment={},
            platform_name="linux",
        )
