import importlib.util
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

from incode_mcp.installer.accelerator import (
    ACCELERATOR_ENVIRONMENT_DIRECTORY,
    ACCELERATOR_EXTRAS,
    AcceleratorPlan,
    accelerator_lock_fingerprint,
    configure_accelerator,
    plan_accelerator,
    probe_accelerator,
    sync_accelerator_environment,
    write_accelerator_record,
)
from incode_mcp.installer.config_files import (
    InstallerError,
    merge_codex_server,
    merge_json_object_entry,
)
from incode_mcp.installer.harnesses import (
    HARNESS_CHOICES,
    configuration_path,
    configure_harness,
    configure_selected_harnesses,
    install_skills,
    parse_harness_selection,
    skill_directory,
)

INSTALLER_PATH = Path(__file__).parents[1] / "install.py"
SHELL_INSTALLER_PATH = Path(__file__).parents[1] / "install.sh"

# The installer stringifies Path objects with the native separator, so expected
# values must go through the same conversion to stay correct on Windows.
SERVER_BINARY = Path("/opt/ci-mcp")
SERVER_COMMAND = str(SERVER_BINARY)


def load_installer() -> ModuleType:
    """Load the stdlib-only bootstrap by path; only its own surface is tested here."""
    assert INSTALLER_PATH.exists(), "install.py does not exist"
    spec = importlib.util.spec_from_file_location("code_indexing_mcp_installer", INSTALLER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_merge_creates_parent_and_top_level_object(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "config.json"

    changed = merge_json_object_entry(
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

    changed = merge_json_object_entry(
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
    path = tmp_path / "config.jsonc"
    entry = {"command": "/opt/ci-mcp", "args": ["serve"]}

    assert merge_json_object_entry(path, "mcpServers", "server", entry) is True
    first = path.read_text()
    assert merge_json_object_entry(path, "mcpServers", "server", entry) is False

    assert path.read_text() == first


def test_jsonc_merge_rejects_invalid_input_without_modifying_it(tmp_path: Path) -> None:
    path = tmp_path / "config.jsonc"
    original = '{"mcp": [}'
    path.write_text(original)

    with pytest.raises(InstallerError, match="Invalid JSON"):
        merge_json_object_entry(path, "mcp", "server", {"enabled": True})

    assert path.read_text() == original
    assert not path.with_name("config.jsonc.bak").exists()


def test_jsonc_merge_validates_unrelated_nested_values(tmp_path: Path) -> None:
    path = tmp_path / "config.jsonc"
    original = '{"unrelated": {"broken": nope}}\n'
    path.write_text(original)

    with pytest.raises(InstallerError, match="Invalid JSON"):
        merge_json_object_entry(path, "mcp", "server", {"enabled": True})

    assert path.read_text() == original
    assert not path.with_name("config.jsonc.bak").exists()


def test_codex_merge_creates_server_table(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"

    changed = merge_codex_server(path, SERVER_BINARY)

    encoded_command = json.dumps(SERVER_COMMAND, ensure_ascii=False)
    assert changed is True
    assert path.read_text() == (
        f'[mcp_servers.code-indexing-mcp]\ncommand = {encoded_command}\nargs = ["serve"]\n'
    )


def test_codex_merge_replaces_only_target_table_and_subtables(tmp_path: Path) -> None:
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

# Keep the feature explanation too.
[features]
example = true
"""
    path.write_text(original)

    changed = merge_codex_server(path, Path("/new/ci-mcp"))

    updated = path.read_text()
    assert changed is True
    assert "# Keep this comment." in updated
    assert '[mcp_servers.other]\ncommand = "other"' in updated
    assert "# Keep the feature explanation too." in updated
    assert "[features]\nexample = true" in updated
    assert 'command = "old"' not in updated
    assert "OLD" not in updated
    assert updated.count("[mcp_servers.code-indexing-mcp]") == 1
    assert f"command = {json.dumps(str(Path('/new/ci-mcp')), ensure_ascii=False)}" in updated
    assert path.with_name("config.toml.bak").read_text() == original


def test_codex_merge_preserves_following_array_tables(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = """[mcp_servers.code-indexing-mcp]
command = "old"

[[skills.config]]
path = "/tmp/skill"
enabled = false
"""
    path.write_text(original)

    merge_codex_server(path, Path("/new/ci-mcp"))

    parsed = tomllib.loads(path.read_text())
    assert parsed["skills"]["config"] == [
        {
            "path": "/tmp/skill",
            "enabled": False,
        }
    ]
    assert path.with_name("config.toml.bak").read_text() == original


def test_codex_merge_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"

    assert merge_codex_server(path, Path("/opt/ci-mcp")) is True
    first = path.read_text()
    assert merge_codex_server(path, Path("/opt/ci-mcp")) is False

    assert path.read_text() == first


def test_codex_merge_rejects_inline_target_without_corrupting_config(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = '[mcp_servers]\ncode-indexing-mcp = { command = "old", args = ["old"] }\n'
    path.write_text(original)

    with pytest.raises(InstallerError, match="inline or dotted"):
        merge_codex_server(path, Path("/opt/ci-mcp"))

    assert path.read_text() == original
    assert not path.with_name("config.toml.bak").exists()


def test_harness_menu_combines_codex_cli_and_desktop() -> None:

    assert [(choice.slug, choice.label) for choice in HARNESS_CHOICES] == [
        ("codex", "Codex (CLI + Desktop)"),
        ("claude-code", "Claude Code"),
        ("kimi-code", "Kimi Code"),
        ("claude-desktop", "Claude Desktop"),
        ("opencode", "OpenCode"),
        ("kilocode", "KiloCode"),
    ]


def test_harness_selection_accepts_numbers_slugs_duplicates_and_all() -> None:

    assert parse_harness_selection("1, 3, codex, opencode") == [
        "codex",
        "kimi-code",
        "opencode",
    ]
    assert parse_harness_selection("all") == [choice.slug for choice in HARNESS_CHOICES]
    assert parse_harness_selection("") == []
    with pytest.raises(InstallerError, match="Unknown harness"):
        parse_harness_selection("7")


def test_configuration_paths_honor_client_home_overrides(tmp_path: Path) -> None:
    environment = {
        "CODEX_HOME": str(tmp_path / "codex-home"),
        "KIMI_CODE_HOME": str(tmp_path / "kimi-home"),
        "OPENCODE_CONFIG": str(tmp_path / "custom-opencode.jsonc"),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
        "APPDATA": str(tmp_path / "appdata"),
    }

    assert (
        configuration_path("codex", home=tmp_path, environment=environment, platform_name="darwin")
        == tmp_path / "codex-home" / "config.toml"
    )
    assert (
        configuration_path(
            "kimi-code", home=tmp_path, environment=environment, platform_name="darwin"
        )
        == tmp_path / "kimi-home" / "mcp.json"
    )
    assert (
        configuration_path(
            "opencode", home=tmp_path, environment=environment, platform_name="darwin"
        )
        == tmp_path / "custom-opencode.jsonc"
    )
    assert (
        configuration_path(
            "kilocode", home=tmp_path, environment=environment, platform_name="darwin"
        )
        == tmp_path / "xdg" / "kilo" / "kilo.jsonc"
    )
    assert (
        configuration_path(
            "claude-desktop", home=tmp_path, environment=environment, platform_name="win32"
        )
        == tmp_path / "appdata" / "Claude" / "claude_desktop_config.json"
    )


def test_claude_code_honors_config_directory_override(tmp_path: Path) -> None:
    config_directory = tmp_path / "claude-config"

    assert (
        configuration_path(
            "claude-code",
            home=tmp_path,
            environment={"CLAUDE_CONFIG_DIR": str(config_directory)},
            platform_name="linux",
        )
        == config_directory / ".claude.json"
    )


def test_kilocode_honors_config_file_override(tmp_path: Path) -> None:
    config_file = tmp_path / "custom-kilo.jsonc"

    assert (
        configuration_path(
            "kilocode",
            home=tmp_path,
            environment={
                "KILO_CONFIG": str(config_file),
                "KILO_CONFIG_DIR": str(tmp_path / "kilo-config"),
            },
            platform_name="linux",
        )
        == config_file
    )


def test_kilocode_honors_config_directory_override(tmp_path: Path) -> None:
    config_directory = tmp_path / "kilo-config"

    assert (
        configuration_path(
            "kilocode",
            home=tmp_path,
            environment={"KILO_CONFIG_DIR": str(config_directory)},
            platform_name="linux",
        )
        == config_directory / "kilo.jsonc"
    )


@pytest.mark.parametrize(
    ("slug", "object_key", "relative_path", "expected_entry"),
    [
        (
            "claude-code",
            "mcpServers",
            ".claude.json",
            {
                "type": "stdio",
                "command": SERVER_COMMAND,
                "args": ["serve"],
            },
        ),
        (
            "kimi-code",
            "mcpServers",
            ".kimi-code/mcp.json",
            {"command": SERVER_COMMAND, "args": ["serve"]},
        ),
        (
            "claude-desktop",
            "mcpServers",
            "Library/Application Support/Claude/claude_desktop_config.json",
            {"command": SERVER_COMMAND, "args": ["serve"]},
        ),
        (
            "opencode",
            "mcp",
            ".config/opencode/opencode.json",
            {
                "type": "local",
                "command": [SERVER_COMMAND, "serve"],
                "enabled": True,
            },
        ),
        (
            "kilocode",
            "mcp",
            ".config/kilo/kilo.jsonc",
            {
                "type": "local",
                "command": [SERVER_COMMAND, "serve"],
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

    path = configure_harness(
        slug,
        SERVER_BINARY,
        home=tmp_path,
        environment={},
        platform_name="darwin",
    )

    assert path == tmp_path / relative_path
    parsed = json.loads(path.read_text())
    assert parsed[object_key]["code-indexing-mcp"] == expected_entry


def test_configure_codex_uses_shared_toml(tmp_path: Path) -> None:

    path = configure_harness(
        "codex",
        SERVER_BINARY,
        home=tmp_path,
        environment={},
        platform_name="darwin",
    )

    assert path == tmp_path / ".codex" / "config.toml"
    parsed = tomllib.loads(path.read_text())
    assert parsed["mcp_servers"]["code-indexing-mcp"] == {
        "command": SERVER_COMMAND,
        "args": ["serve"],
    }


def test_claude_desktop_uses_linux_config_directory(tmp_path: Path) -> None:

    assert (
        configuration_path(
            "claude-desktop",
            home=tmp_path,
            environment={},
            platform_name="linux",
        )
        == tmp_path / ".config" / "Claude" / "claude_desktop_config.json"
    )


def run_git(*arguments: str, cwd: Path | None = None) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def create_test_remote(tmp_path: Path, name: str = "remote") -> tuple[Path, Path]:
    remote = tmp_path / f"{name}.git"
    publisher = tmp_path / f"{name}-publisher"
    run_git("init", "--bare", "--initial-branch=main", str(remote))
    run_git("init", "--initial-branch=main", str(publisher))
    run_git("config", "user.name", "Installer Tests", cwd=publisher)
    run_git("config", "user.email", "installer@example.test", cwd=publisher)
    (publisher / "version.txt").write_text("one\n")
    run_git("add", "version.txt", cwd=publisher)
    run_git("commit", "-m", "initial", cwd=publisher)
    run_git("remote", "add", "origin", str(remote), cwd=publisher)
    run_git("push", "-u", "origin", "main", cwd=publisher)
    return remote, publisher


def test_repository_is_cloned_then_fast_forwarded_on_update(tmp_path: Path) -> None:
    installer = load_installer()
    remote, publisher = create_test_remote(tmp_path)
    checkout = tmp_path / "installed" / "code-indexing-mcp"

    assert installer.clone_or_update_repository(str(remote), checkout) == "installed"
    assert (checkout / "version.txt").read_text() == "one\n"

    (publisher / "version.txt").write_text("two\n")
    run_git("add", "version.txt", cwd=publisher)
    run_git("commit", "-m", "update", cwd=publisher)
    run_git("push", cwd=publisher)

    assert installer.clone_or_update_repository(str(remote), checkout) == "updated"
    assert (checkout / "version.txt").read_text() == "two\n"


def test_repository_update_rejects_non_repo_dirty_and_mismatched_targets(
    tmp_path: Path,
) -> None:
    installer = load_installer()
    remote, _ = create_test_remote(tmp_path, "expected")
    other_remote, _ = create_test_remote(tmp_path, "other")

    non_repo = tmp_path / "not-a-repo"
    non_repo.mkdir()
    with pytest.raises(installer.InstallerError, match="not a Git repository"):
        installer.clone_or_update_repository(str(remote), non_repo)

    checkout = tmp_path / "checkout"
    installer.clone_or_update_repository(str(remote), checkout)
    (checkout / "version.txt").write_text("local edit\n")
    with pytest.raises(installer.InstallerError, match="uncommitted changes"):
        installer.clone_or_update_repository(str(remote), checkout)

    run_git("restore", "version.txt", cwd=checkout)
    with pytest.raises(installer.InstallerError, match="origin does not match"):
        installer.clone_or_update_repository(str(other_remote), checkout)


def test_sync_environment_runs_locked_sync_and_finds_server(tmp_path: Path) -> None:
    """The serving environment is pinned to the CPU extra plus the TUI extra."""
    installer = load_installer()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    expected = installer.server_executable(checkout)
    if sys.platform == "win32":
        fake_uv = tmp_path / "uv.bat"
        fake_uv.write_text(
            "@echo off\r\n"
            'if not "%~1"=="sync" exit /b 1\r\n'
            'if not "%~2"=="--locked" exit /b 1\r\n'
            'if not "%~3"=="--extra" exit /b 1\r\n'
            'if not "%~4"=="cpu" exit /b 1\r\n'
            'if not "%~5"=="--extra" exit /b 1\r\n'
            'if not "%~6"=="tui" exit /b 1\r\n'
            "md .venv\\Scripts\r\n"
            "type nul > .venv\\Scripts\\code-indexing-mcp.exe\r\n",
            newline="",
        )
    else:
        fake_uv = tmp_path / "uv"
        fake_uv.write_text(
            "#!/bin/sh\n"
            'test "$1" = "sync"\n'
            'test "$2" = "--locked"\n'
            'test "$3" = "--extra"\n'
            'test "$4" = "cpu"\n'
            'test "$5" = "--extra"\n'
            'test "$6" = "tui"\n'
            "mkdir -p .venv/bin\n"
            "touch .venv/bin/code-indexing-mcp\n"
        )
        fake_uv.chmod(0o755)

    command = installer.sync_environment(checkout, uv_executable=str(fake_uv))

    assert command == expected


def test_configure_selected_harnesses_isolates_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempted: list[str] = []

    def fake_configure(slug: str, command: Path, **kwargs: object) -> Path:
        attempted.append(slug)
        if slug == "claude-code":
            raise InstallerError("broken config")
        return tmp_path / f"{slug}.json"

    monkeypatch.setattr("incode_mcp.installer.harnesses.configure_harness", fake_configure)

    successes, failures = configure_selected_harnesses(
        ["codex", "claude-code", "kimi-code"],
        Path("/opt/ci-mcp"),
        home=tmp_path,
        environment={},
        platform_name="darwin",
    )

    assert attempted == ["codex", "claude-code", "kimi-code"]
    assert [slug for slug, _ in successes] == ["codex", "kimi-code"]
    assert failures == [("claude-code", "broken config")]


def test_configure_selected_harnesses_isolates_non_utf8_config(tmp_path: Path) -> None:
    (tmp_path / ".claude.json").write_bytes(b"\xff")

    successes, failures = configure_selected_harnesses(
        ["claude-code", "kimi-code"],
        Path("/opt/ci-mcp"),
        home=tmp_path,
        environment={},
        platform_name="darwin",
    )

    assert [slug for slug, _ in successes] == ["kimi-code"]
    assert len(failures) == 1
    assert failures[0][0] == "claude-code"
    assert "UTF-8" in failures[0][1]
    assert (tmp_path / ".kimi-code" / "mcp.json").exists()


def test_main_delegates_to_the_module_cli_with_forwarded_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installer = load_installer()
    checkout = tmp_path / "checkout"
    monkeypatch.setattr(installer, "clone_or_update_repository", lambda url, directory: "installed")
    monkeypatch.setattr(installer, "sync_environment", lambda directory: checkout / "server")
    monkeypatch.setattr(installer, "tui_available", lambda: False)
    delegated: list[list[str]] = []
    monkeypatch.setattr(installer, "_delegate", lambda directory, tail: delegated.append(tail) or 0)

    code = installer.main(
        [
            "--install-dir",
            str(checkout),
            "--accelerator",
            "mlx",
            "--harnesses",
            "codex,kimi-code",
            "--set",
            "INCODE_OFFLINE=1",
            "--unset",
            "INCODE_BROKER",
            "--offline",
        ]
    )

    assert code == 0
    (tail,) = delegated
    assert tail[:4] == ["--install-dir", str(checkout), "--accelerator", "mlx"]
    for fragment in (
        ["--harnesses", "codex,kimi-code"],
        ["--set", "INCODE_OFFLINE=1"],
        ["--unset", "INCODE_BROKER"],
        ["--offline"],
    ):
        assert any(tail[index : index + len(fragment)] == fragment for index in range(len(tail)))
    assert "--tui" not in tail
    assert "Installed repository" in capsys.readouterr().out


def test_main_adds_tui_flag_on_a_capable_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer = load_installer()
    monkeypatch.setattr(installer, "clone_or_update_repository", lambda url, directory: "updated")
    monkeypatch.setattr(installer, "sync_environment", lambda directory: tmp_path / "server")
    monkeypatch.setattr(installer, "tui_available", lambda: True)
    delegated: list[list[str]] = []
    monkeypatch.setattr(installer, "_delegate", lambda directory, tail: delegated.append(tail) or 0)

    assert installer.main(["--install-dir", str(tmp_path / "checkout")]) == 0
    assert "--tui" in delegated[0]


def test_main_no_tui_flag_suppresses_the_wizard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer = load_installer()
    monkeypatch.setattr(installer, "clone_or_update_repository", lambda url, directory: "updated")
    monkeypatch.setattr(installer, "sync_environment", lambda directory: tmp_path / "server")
    monkeypatch.setattr(installer, "tui_available", lambda: True)
    delegated: list[list[str]] = []
    monkeypatch.setattr(installer, "_delegate", lambda directory, tail: delegated.append(tail) or 0)

    assert installer.main(["--install-dir", str(tmp_path / "checkout"), "--no-tui"]) == 0
    assert "--tui" not in delegated[0]


@pytest.mark.parametrize(
    "flags",
    [
        ["--harnesses", "codex"],
        ["--set", "INCODE_OFFLINE=1"],
        ["--unset", "INCODE_BROKER"],
        ["--no-prompt"],
    ],
)
def test_main_does_not_open_the_wizard_over_scripted_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str]
) -> None:
    """A run that already says what to do must not stop for a keypress."""

    installer = load_installer()
    monkeypatch.setattr(installer, "clone_or_update_repository", lambda url, directory: "updated")
    monkeypatch.setattr(installer, "sync_environment", lambda directory: tmp_path / "server")
    monkeypatch.setattr(installer, "tui_available", lambda: True)
    delegated: list[list[str]] = []
    monkeypatch.setattr(installer, "_delegate", lambda directory, tail: delegated.append(tail) or 0)

    assert installer.main(["--install-dir", str(tmp_path / "checkout"), *flags]) == 0
    assert "--tui" not in delegated[0]


def test_explicit_tui_flag_wins_over_scripted_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer = load_installer()
    monkeypatch.setattr(installer, "clone_or_update_repository", lambda url, directory: "updated")
    monkeypatch.setattr(installer, "sync_environment", lambda directory: tmp_path / "server")
    monkeypatch.setattr(installer, "tui_available", lambda: False)
    delegated: list[list[str]] = []
    monkeypatch.setattr(installer, "_delegate", lambda directory, tail: delegated.append(tail) or 0)

    assert (
        installer.main(
            ["--install-dir", str(tmp_path / "checkout"), "--tui", "--harnesses", "codex"]
        )
        == 0
    )
    assert "--tui" in delegated[0]
    assert "codex" in delegated[0]


def test_main_reports_actionable_installer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installer = load_installer()

    def fail(repository_url: str, install_directory: Path) -> str:
        raise installer.InstallerError("Git is required")

    monkeypatch.setattr(installer, "clone_or_update_repository", fail)

    status = installer.main(["--install-dir", str(tmp_path / "checkout")])

    assert status == 1
    assert "Error: Git is required" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("tty", "term", "expected"),
    [
        (True, "xterm-256color", True),
        (True, "dumb", False),
        (True, "", False),
        (False, "xterm", False),
    ],
)
def test_tui_available_detects_capable_terminals(
    monkeypatch: pytest.MonkeyPatch, tty: bool, term: str, expected: bool
) -> None:
    installer = load_installer()
    monkeypatch.setattr(installer.sys.stdin, "isatty", lambda: tty)
    monkeypatch.setattr(installer.sys.stdout, "isatty", lambda: tty)
    monkeypatch.setenv("TERM", term)
    assert installer.tui_available() is expected


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="install.sh is a POSIX bootstrap; Windows sh receives a backslash $0 and misbehaves",
)
def test_posix_bootstrap_has_valid_syntax_and_runs_adjacent_installer() -> None:
    assert SHELL_INSTALLER_PATH.exists(), "install.sh does not exist"

    subprocess.run(["sh", "-n", str(SHELL_INSTALLER_PATH)], check=True)
    completed = subprocess.run(
        ["sh", str(SHELL_INSTALLER_PATH), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--harnesses" in completed.stdout


def _skills_source(tmp_path: Path, names: tuple[str, ...] = ("alpha", "beta")) -> Path:
    root = tmp_path / "repo" / "src" / "incode_mcp" / "skills"
    for name in names:
        skill = root / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: test\n---\n", encoding="utf-8"
        )
    return tmp_path / "repo"


def test_skill_directories_cover_supported_harnesses(tmp_path: Path) -> None:
    assert (
        skill_directory("claude-code", home=tmp_path, environment={})
        == tmp_path / ".claude" / "skills"
    )
    assert (
        skill_directory("codex", home=tmp_path, environment={}) == tmp_path / ".agents" / "skills"
    )
    assert (
        skill_directory("kimi-code", home=tmp_path, environment={})
        == tmp_path / ".agents" / "skills"
    )
    assert (
        skill_directory(
            "opencode", home=tmp_path, environment={"XDG_CONFIG_HOME": str(tmp_path / "xdg")}
        )
        == tmp_path / "xdg" / "opencode" / "skills"
    )
    assert skill_directory("claude-desktop", home=tmp_path, environment={}) is None
    assert skill_directory("kilocode", home=tmp_path, environment={}) is None


def test_skill_directory_honors_claude_config_dir(tmp_path: Path) -> None:
    environment = {"CLAUDE_CONFIG_DIR": str(tmp_path / "custom-claude")}
    assert (
        skill_directory("claude-code", home=tmp_path, environment=environment)
        == tmp_path / "custom-claude" / "skills"
    )


def test_install_skills_links_bundled_skills(tmp_path: Path) -> None:
    repo = _skills_source(tmp_path)

    results = install_skills(["claude-code"], repo, home=tmp_path, environment={})

    skills_dir = tmp_path / ".claude" / "skills"
    for name in ("alpha", "beta"):
        link = skills_dir / name
        assert link.is_symlink()
        assert link.resolve() == (repo / "src" / "incode_mcp" / "skills" / name).resolve()
    assert len(results) == 1
    slug, message = results[0]
    assert slug == "claude-code"
    assert "2 linked" in message


def test_install_skills_is_idempotent(tmp_path: Path) -> None:
    repo = _skills_source(tmp_path)
    install_skills(["codex"], repo, home=tmp_path, environment={})

    results = install_skills(["codex"], repo, home=tmp_path, environment={})

    _slug, message = results[0]
    assert "0 linked" in message
    assert "already installed" in message


def test_install_skills_is_idempotent_across_equivalent_source_paths(tmp_path: Path) -> None:
    """The already-installed check must compare real paths, not raw link text.

    Windows readlink returns an extended-length "\\\\?\\C:\\..." path that never
    equals the plain path the link was made from; a symlinked parent directory
    reproduces the same mismatch on POSIX.
    """
    repo = _skills_source(tmp_path, names=("alpha",))
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    install_skills(["codex"], alias, home=tmp_path, environment={})

    results = install_skills(["codex"], repo, home=tmp_path, environment={})

    _slug, message = results[0]
    assert "0 linked" in message
    assert "1 already installed" in message


def test_install_skills_backs_up_clashing_directory(tmp_path: Path) -> None:
    repo = _skills_source(tmp_path)
    clash = tmp_path / ".agents" / "skills" / "alpha"
    clash.mkdir(parents=True)
    (clash / "SKILL.md").write_text("old", encoding="utf-8")

    install_skills(["kimi-code"], repo, home=tmp_path, environment={})

    assert (tmp_path / ".agents" / "skills" / "alpha").is_symlink()
    backup = tmp_path / ".agents" / "skills" / "alpha.bak"
    assert backup.is_dir() and not backup.is_symlink()
    assert (backup / "SKILL.md").read_text(encoding="utf-8") == "old"


def test_install_skills_backs_up_clashing_symlink(tmp_path: Path) -> None:
    repo = _skills_source(tmp_path)
    elsewhere = tmp_path / "dotfiles" / "alpha"
    elsewhere.mkdir(parents=True)
    (elsewhere / "SKILL.md").write_text("mine", encoding="utf-8")
    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "alpha").symlink_to(elsewhere, target_is_directory=True)

    install_skills(["kimi-code"], repo, home=tmp_path, environment={})

    link = skills_dir / "alpha"
    assert link.is_symlink()
    assert link.resolve() == (repo / "src" / "incode_mcp" / "skills" / "alpha").resolve()
    backup = skills_dir / "alpha.bak"
    assert backup.is_symlink()
    assert backup.resolve() == elsewhere.resolve()


def test_install_skills_skips_unsupported_harness(tmp_path: Path) -> None:
    repo = _skills_source(tmp_path)

    results = install_skills(["claude-desktop"], repo, home=tmp_path, environment={})

    slug, message = results[0]
    assert slug == "claude-desktop"
    assert "skipped" in message
    assert not (tmp_path / ".claude").exists()


def test_install_skills_reports_missing_source(tmp_path: Path) -> None:
    results = install_skills(["codex"], tmp_path / "empty-repo", home=tmp_path, environment={})

    _slug, message = results[0]
    assert "skipped" in message
    assert not (tmp_path / ".agents").exists()


def test_reinstall_from_a_new_checkout_keeps_the_original_backup(tmp_path: Path) -> None:
    """Relinking must not treat an earlier install's own link as user content."""
    first = _skills_source(tmp_path / "first", names=("alpha",))
    second = _skills_source(tmp_path / "second", names=("alpha",))
    skills_dir = tmp_path / ".agents" / "skills"
    mine = skills_dir / "alpha"
    mine.mkdir(parents=True)
    (mine / "SKILL.md").write_text("mine", encoding="utf-8")

    install_skills(["codex"], first, home=tmp_path, environment={})
    install_skills(["codex"], second, home=tmp_path, environment={})

    assert (skills_dir / "alpha").resolve() == (
        second / "src" / "incode_mcp" / "skills" / "alpha"
    ).resolve()
    assert (skills_dir / "alpha.bak" / "SKILL.md").read_text(encoding="utf-8") == "mine"
    assert not (skills_dir / "alpha.bak.2").exists()


def test_install_skills_never_overwrites_an_existing_backup(tmp_path: Path) -> None:
    repo = _skills_source(tmp_path, names=("alpha",))
    skills_dir = tmp_path / ".agents" / "skills"
    for name, content in (("alpha", "current"), ("alpha.bak", "older")):
        entry = skills_dir / name
        entry.mkdir(parents=True)
        (entry / "SKILL.md").write_text(content, encoding="utf-8")

    install_skills(["codex"], repo, home=tmp_path, environment={})

    assert (skills_dir / "alpha").is_symlink()
    assert (skills_dir / "alpha.bak" / "SKILL.md").read_text(encoding="utf-8") == "older"
    assert (skills_dir / "alpha.bak.2" / "SKILL.md").read_text(encoding="utf-8") == "current"


def test_install_skills_leaves_existing_skills_in_place_when_symlinks_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows without the symlink privilege must not displace a user's skill."""
    repo = _skills_source(tmp_path, names=("alpha",))
    mine = tmp_path / ".agents" / "skills" / "alpha"
    mine.mkdir(parents=True)
    (mine / "SKILL.md").write_text("mine", encoding="utf-8")

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(Path, "symlink_to", refuse)

    results = install_skills(["codex"], repo, home=tmp_path, environment={})

    _slug, message = results[0]
    assert "skipped" in message
    assert (mine / "SKILL.md").read_text(encoding="utf-8") == "mine"
    assert not (tmp_path / ".agents" / "skills" / "alpha.bak").exists()
    assert not (tmp_path / ".agents" / "skills" / "alpha.incoming").exists()


# A platform CUDA wheels are published for, so detection reaches the steps
# these tests are about rather than stopping at "no wheels for this machine".
CUDA_PLATFORM = "win32" if sys.platform == "win32" else "linux"
CUDA_MACHINE = "AMD64" if sys.platform == "win32" else "x86_64"


def _fake_uv(tmp_path: Path) -> Path:
    """A uv stand-in that creates the environment layout a real sync would."""
    if sys.platform == "win32":
        fake_uv = tmp_path / "uv.bat"
        fake_uv.write_text(
            "@echo off\r\n"
            'if not "%~1"=="sync" exit /b 1\r\n'
            "md %UV_PROJECT_ENVIRONMENT%\\Scripts\r\n"
            "type nul > %UV_PROJECT_ENVIRONMENT%\\Scripts\\python.exe\r\n",
            newline="",
        )
    else:
        fake_uv = tmp_path / "uv"
        fake_uv.write_text(
            "#!/bin/sh\n"
            'test "$1" = "sync"\n'
            'mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"\n'
            'printf "" > "$UV_PROJECT_ENVIRONMENT/bin/python"\n'
        )
        fake_uv.chmod(0o755)
    return fake_uv


@pytest.mark.parametrize(
    ("requested", "platform_name", "machine", "report", "expected", "reason"),
    [
        ("cpu", "linux", "x86_64", "550.54.14, A100", "cpu", "CPU was requested"),
        # Apple Silicon on macOS 14+, which is what platform_version says below.
        ("auto", "darwin", "arm64", None, "mlx", "the locked MLX build is available"),
        ("auto", "linux", "x86_64", None, "cpu", "no usable NVIDIA driver"),
        ("auto", "linux", "x86_64", "", "cpu", "no usable NVIDIA driver"),
        ("auto", "linux", "x86_64", "470.10, Tesla T4", "cpu", "is below the 525.60"),
        ("auto", "linux", "x86_64", "550.54.14, A100", "cuda", "satisfies the pinned"),
        ("auto", "win32", "AMD64", "560.1, RTX 4090", "cuda", "satisfies the pinned"),
        ("auto", "win32", "AMD64", "527.40, RTX 4090", "cpu", "is below the 527.41"),
        # The cuda extra's marker is platform_machine == 'x86_64' on Linux, so a
        # machine calling itself amd64 there would sync an environment with no
        # embedding runtime in it at all. Refused up front instead.
        ("auto", "linux", "amd64", "550.54.14, A100", "cpu", "no CUDA wheels"),
        ("cuda", "darwin", "arm64", None, "cpu", "CUDA was requested but"),
        ("coreml", "darwin", "arm64", None, "cpu", "INCODE_EMBED_ACCELERATOR=coreml"),
        ("webgpu", "linux", "x86_64", None, "webgpu", "locked WebGPU"),
        ("migraphx", "linux", "x86_64", None, "webgpu", "falling back to WebGPU"),
    ],
)
def test_accelerator_detection_nominates_only_a_supported_pinned_combination(
    requested: str,
    platform_name: str,
    machine: str,
    report: str | None,
    expected: str,
    reason: str,
) -> None:

    plan = plan_accelerator(
        requested,
        platform_name=platform_name,
        machine=machine,
        nvidia_report=lambda: report,
        rocm_report=lambda: None,
        python_version="3.12",
        platform_version="14.0",
    )

    assert plan.accelerator == expected
    assert reason in plan.reason


@pytest.mark.parametrize(
    ("platform_name", "machine", "platform_version", "expected"),
    [
        ("darwin", "arm64", "14.0", "webgpu"),
        ("darwin", "x86_64", "14.6", "cpu"),
        ("darwin", "arm64", "13.6", "cpu"),
        ("linux", "x86_64", "", "webgpu"),
        ("win32", "AMD64", "", "webgpu"),
        ("linux", "aarch64", "", "cpu"),
    ],
)
def test_webgpu_is_prepared_only_where_the_locked_plugin_has_a_wheel(
    platform_name: str,
    machine: str,
    platform_version: str,
    expected: str,
) -> None:

    plan = plan_accelerator(
        "webgpu",
        platform_name=platform_name,
        machine=machine,
        platform_version=platform_version,
        python_version="3.12",
        nvidia_report=lambda: None,
        rocm_report=lambda: None,
    )

    assert plan.accelerator == expected
    assert plan.honored is (expected == "webgpu")


@pytest.mark.parametrize(
    ("platform_name", "machine", "python_version", "rocm", "expected", "honored"),
    [
        ("linux", "x86_64", "3.12", "7.2.1, AMD Radeon PRO W7900", "migraphx", True),
        ("linux", "x86_64", "3.12", "7.2, AMD Radeon PRO W7900", "webgpu", False),
        ("linux", "x86_64", "3.13", "7.2.1, AMD Radeon PRO W7900", "webgpu", False),
        ("linux", "aarch64", "3.12", "7.2.1, AMD GPU", "cpu", False),
    ],
)
def test_migraphx_uses_only_the_pinned_rocm_python_matrix_then_falls_back(
    platform_name: str,
    machine: str,
    python_version: str,
    rocm: str,
    expected: str,
    honored: bool,
) -> None:

    plan = plan_accelerator(
        "migraphx",
        platform_name=platform_name,
        machine=machine,
        platform_version="",
        python_version=python_version,
        nvidia_report=lambda: None,
        rocm_report=lambda: rocm,
    )

    assert plan.accelerator == expected
    assert plan.honored is honored
    if expected == "migraphx":
        assert plan.driver_version == "7.2.1"
        assert plan.device_name == "AMD Radeon PRO W7900"
    else:
        assert "MIGraphX was requested but" in plan.reason


@pytest.mark.parametrize(
    ("platform_name", "machine", "platform_version", "expected"),
    [
        ("darwin", "arm64", "14.0", "mlx"),
        ("darwin", "arm64", "26.5.2", "mlx"),
        ("darwin", "arm64", "13.6", "cpu"),
        ("darwin", "x86_64", "15.1", "cpu"),
        ("linux", "x86_64", "", "cpu"),
        ("win32", "AMD64", "", "cpu"),
    ],
)
def test_mlx_is_prepared_only_on_apple_silicon_with_a_published_wheel(
    platform_name: str,
    machine: str,
    platform_version: str,
    expected: str,
) -> None:
    """MLX also ships CPU-only Linux and Windows wheels.

    Nominating one of those would prepare a "Metal" environment with no Metal in
    it, which would pass its own probe and then lose to the CPU it really is.
    """

    plan = plan_accelerator(
        "mlx",
        platform_name=platform_name,
        machine=machine,
        platform_version=platform_version,
        python_version="3.12",
        nvidia_report=lambda: None,
        rocm_report=lambda: None,
    )

    assert plan.accelerator == expected
    assert plan.honored is (expected == "mlx")
    if expected == "mlx":
        # The OS version is part of the probe cache key, so an upgrade under a
        # prepared environment retires the verdict recorded before it.
        assert plan.driver_version == platform_version
    else:
        assert "MLX was requested but" in plan.reason


def test_an_unsupported_mlx_request_does_not_fall_through_to_webgpu() -> None:
    """MIGraphX degrades to WebGPU because both are cross-vendor GPU paths.

    A request for Metal on a machine with no Metal is not a request for Vulkan,
    so this reports CPU and says why instead.
    """

    plan = plan_accelerator(
        "mlx",
        platform_name="linux",
        machine="x86_64",
        platform_version="",
        python_version="3.12",
        nvidia_report=lambda: None,
        rocm_report=lambda: None,
    )

    assert plan.accelerator == "cpu"
    assert "WebGPU" not in plan.reason


def test_auto_prepares_mlx_on_apple_silicon() -> None:
    """MLX passed the gates CUDA passed, so `auto` prepares it unasked."""

    plan = plan_accelerator(
        "auto",
        platform_name="darwin",
        machine="arm64",
        platform_version="26.5.2",
        python_version="3.12",
        nvidia_report=lambda: None,
        rocm_report=lambda: None,
    )

    assert plan.accelerator == "mlx"
    assert plan.honored is True
    assert plan.driver_version == "26.5.2"


def test_auto_on_an_unsupported_mac_reports_the_same_reason_it_always_did() -> None:

    plan = plan_accelerator(
        "auto",
        platform_name="darwin",
        machine="x86_64",
        platform_version="15.1",
        python_version="3.12",
        nvidia_report=lambda: None,
        rocm_report=lambda: None,
    )

    assert plan.accelerator == "cpu"
    assert plan.honored is True
    assert "no CUDA wheels" in plan.reason


def test_an_explicit_cuda_request_is_never_answered_with_mlx() -> None:
    """An override names a backend, not "whatever this machine has"."""

    plan = plan_accelerator(
        "cuda",
        platform_name="darwin",
        machine="arm64",
        platform_version="26.5.2",
        python_version="3.12",
        nvidia_report=lambda: None,
        rocm_report=lambda: None,
    )

    assert plan.accelerator == "cpu"
    assert plan.honored is False


def test_all_accelerator_environments_have_a_locked_extra() -> None:

    assert ACCELERATOR_EXTRAS == {
        "cuda": "cuda",
        "mlx": "mlx",
        "webgpu": "webgpu",
        "migraphx": "migraphx",
    }


@pytest.mark.parametrize(
    ("requested", "platform_name", "honored"),
    [
        # Landing on CPU is not by itself a request denied: these three asked
        # for exactly what they got, whatever the hardware turned out to be.
        ("cpu", "linux", True),
        ("auto", "darwin", True),
        ("coreml", "darwin", True),
        # These named something this installation cannot give them.
        ("cuda", "darwin", False),
        ("webgpu", "linux", False),
        ("migraphx", "linux", False),
    ],
)
def test_only_a_denied_request_is_reported_as_a_problem(
    requested: str, platform_name: str, honored: bool
) -> None:

    plan = plan_accelerator(
        requested,
        platform_name=platform_name,
        machine="arm64",
        # Old enough that MLX has no wheel for it, so every case here still
        # lands on CPU and the question stays whether that is reported as a
        # problem -- rather than depending on the macOS this test runs on.
        platform_version="13.0",
        nvidia_report=lambda: None,
    )

    assert plan.accelerator == "cpu"
    assert plan.honored is honored


def test_a_cpu_installation_retracts_an_earlier_accelerator_offer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine reinstalled as CPU-only must stop advertising a GPU it had."""
    data = tmp_path / "data"
    data.mkdir()
    record = data / "accelerator.json"
    record.write_text('{"schema_version": 1}', encoding="utf-8")
    monkeypatch.setattr(
        "incode_mcp.installer.accelerator.runtime_record_path",
        lambda python: data / "accelerator.json",
    )

    plan = configure_accelerator(tmp_path / "checkout", "cpu")

    assert plan.accelerator == "cpu"
    assert not record.exists()


def test_a_prepared_accelerator_is_recorded_only_after_its_probe_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    data = tmp_path / "data"
    probed: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        "incode_mcp.installer.accelerator.runtime_record_path",
        lambda python: data / "accelerator.json",
    )
    monkeypatch.setattr(
        "incode_mcp.installer.accelerator.interpreter_version", lambda python: "3.12"
    )

    def fake_probe(python: Path, accelerator: str, *, offline: bool = False) -> dict[str, object]:
        probed.append((python, accelerator))
        return {
            "ok": True,
            "interpreter": str(python),
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "runtime_version": "1.23.2",
            "python_version": "3.12",
            "device": "cuda:0",
            "detail": "probed 2 passages on CUDAExecutionProvider",
        }

    monkeypatch.setattr("incode_mcp.installer.accelerator.probe_accelerator", fake_probe)

    plan = configure_accelerator(
        checkout,
        "cuda",
        uv_executable=str(_fake_uv(tmp_path)),
        nvidia_report=lambda: "550.54.14, NVIDIA A100",
        platform_name=CUDA_PLATFORM,
        machine=CUDA_MACHINE,
    )

    assert plan.accelerator == "cuda"
    assert plan.driver_version == "550.54.14"
    assert len(probed) == 1
    assert probed[0][1] == "cuda"
    written = json.loads((data / "accelerator.json").read_text())
    assert written["accelerator"] == "cuda"
    assert written["driver_version"] == "550.54.14"
    assert written["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert written["lock_fingerprint"]


@pytest.mark.parametrize(
    ("requested", "rocm", "expected", "platform_name", "machine"),
    [
        ("webgpu", None, "webgpu", "linux", "x86_64"),
        ("migraphx", "7.2.1, AMD Radeon PRO W7900", "migraphx", "linux", "x86_64"),
        ("mlx", None, "mlx", "darwin", "arm64"),
    ],
)
def test_experimental_accelerators_sync_their_own_locked_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    rocm: str | None,
    expected: str,
    platform_name: str,
    machine: str,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    record = tmp_path / "data" / "accelerator.json"
    interpreter = checkout / ACCELERATOR_ENVIRONMENT_DIRECTORY / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("", encoding="utf-8")
    synced: list[str] = []
    monkeypatch.setattr(
        "incode_mcp.installer.accelerator.runtime_record_path", lambda python: record
    )
    monkeypatch.setattr(
        "incode_mcp.installer.accelerator.interpreter_version", lambda python: "3.12"
    )

    def fake_sync(
        install_directory: Path,
        extra: str,
        **_kwargs: object,
    ) -> Path:
        synced.append(extra)
        return interpreter

    monkeypatch.setattr("incode_mcp.installer.accelerator.sync_accelerator_environment", fake_sync)
    monkeypatch.setattr(
        "incode_mcp.installer.accelerator.probe_accelerator",
        lambda python, accelerator, *, offline=False: {
            "ok": True,
            "interpreter": str(python),
            "providers": [f"{accelerator} provider", "CPUExecutionProvider"],
            "runtime_version": "tested",
            "python_version": "3.12",
            "device": accelerator,
            "detail": f"probed {accelerator}",
        },
    )

    plan = configure_accelerator(
        checkout,
        requested,
        platform_name=platform_name,
        machine=machine,
        python_version="3.12",
        platform_version="14.0",
        rocm_report=lambda: rocm,
    )

    assert plan.accelerator == expected
    assert synced == [expected]
    assert json.loads(record.read_text(encoding="utf-8"))["accelerator"] == expected


def test_the_installer_record_is_the_shape_the_runtime_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """install.py is stdlib-only, so the schema it writes is pinned by this test."""
    from incode_mcp.accelerator_env import load_environment

    interpreter = tmp_path / "python"
    interpreter.write_text("", encoding="utf-8")
    data = tmp_path / "data"
    plan = AcceleratorPlan("cuda", "detected", driver_version="550.54.14")

    write_accelerator_record(
        data / "accelerator.json",
        plan,
        {
            "interpreter": str(interpreter),
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "runtime_version": "1.23.2",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "device": "cuda:0",
            "detail": "probed 2 passages",
        },
    )

    status = load_environment(data)
    assert status.reason is None
    assert status.environment is not None
    assert status.environment.accelerator.value == "cuda"
    assert status.environment.interpreter == interpreter
    assert status.environment.driver_version == "550.54.14"


def test_a_failed_probe_rolls_the_installation_back_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-built environment must leave nothing the server could pick up."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    stale = data / "accelerator.json"
    stale.write_text('{"schema_version": 1}', encoding="utf-8")
    monkeypatch.setattr(
        "incode_mcp.installer.accelerator.runtime_record_path",
        lambda python: data / "accelerator.json",
    )
    monkeypatch.setattr(
        "incode_mcp.installer.accelerator.interpreter_version", lambda python: "3.12"
    )

    def failing_probe(
        python: Path, accelerator: str, *, offline: bool = False
    ) -> dict[str, object]:
        raise InstallerError("The accelerator probe failed: no CUDA-capable device")

    monkeypatch.setattr("incode_mcp.installer.accelerator.probe_accelerator", failing_probe)

    plan = configure_accelerator(
        checkout,
        "cuda",
        uv_executable=str(_fake_uv(tmp_path)),
        nvidia_report=lambda: "550.54.14, NVIDIA A100",
        platform_name=CUDA_PLATFORM,
        machine=CUDA_MACHINE,
    )

    assert plan.accelerator == "cpu"
    assert "could not be prepared" in plan.reason
    assert "no CUDA-capable device" in plan.reason
    assert not stale.exists()
    assert not (checkout / ACCELERATOR_ENVIRONMENT_DIRECTORY).exists()


def test_an_unresolvable_data_directory_never_fails_the_installation(tmp_path: Path) -> None:

    plan = configure_accelerator(tmp_path / "checkout", "auto")

    assert plan.accelerator == "cpu"
    assert "data directory could not be resolved" in plan.reason


def test_the_probe_refuses_an_environment_that_does_not_offer_the_provider() -> None:
    """Run the real probe in this CPU environment: it must refuse CUDA, and say so.

    This is the pairing that matters -- the probe's report format and the
    installer's reading of it -- exercised end to end without a GPU.
    """

    with pytest.raises(InstallerError) as failure:
        probe_accelerator(Path(sys.executable), "cuda")

    message = str(failure.value)
    assert "The accelerator probe failed" in message
    assert "CUDAExecutionProvider is not offered" in message


@pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX shell script")
def test_a_probe_that_reports_nothing_is_a_failure_not_a_pass(tmp_path: Path) -> None:
    silent = tmp_path / "silent-python"
    silent.write_text("#!/bin/sh\nexit 5\n", encoding="utf-8")
    silent.chmod(0o755)

    with pytest.raises(InstallerError, match="returned no report"):
        probe_accelerator(silent, "cuda")


def _prepared_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """A checkout whose accelerator environment an earlier run already built."""
    checkout = tmp_path / "checkout"
    (checkout / "uv.lock").parent.mkdir(parents=True)
    (checkout / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    accelerator_env = checkout / ACCELERATOR_ENVIRONMENT_DIRECTORY
    accelerator_env.mkdir(parents=True)
    interpreter = accelerator_env / "python"
    interpreter.write_text("", encoding="utf-8")
    record = tmp_path / "data" / "accelerator.json"
    write_accelerator_record(
        record,
        AcceleratorPlan(
            "cuda",
            "detected",
            driver_version="550.54.14",
            lock_fingerprint=accelerator_lock_fingerprint(checkout, "cuda"),
        ),
        {
            "interpreter": str(interpreter),
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "runtime_version": "1.23.2",
            "python_version": "3.12",
            "device": "cuda:0",
            "detail": "probed 2 passages",
        },
    )
    monkeypatch.setattr(
        "incode_mcp.installer.accelerator.runtime_record_path", lambda python: record
    )
    monkeypatch.setattr(
        "incode_mcp.installer.accelerator.interpreter_version", lambda python: "3.12"
    )
    return checkout, record, accelerator_env


def test_an_unchanged_machine_reuses_its_environment_instead_of_rebuilding_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running the installer must not re-download CUDA and re-probe the GPU."""
    checkout, record, _ = _prepared_checkout(tmp_path, monkeypatch)
    before = record.read_text(encoding="utf-8")

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the environment was rebuilt or re-probed")

    monkeypatch.setattr("incode_mcp.installer.accelerator.sync_accelerator_environment", refuse)
    monkeypatch.setattr("incode_mcp.installer.accelerator.probe_accelerator", refuse)

    plan = configure_accelerator(
        checkout,
        "cuda",
        nvidia_report=lambda: "550.54.14, NVIDIA A100",
        platform_name=CUDA_PLATFORM,
        machine=CUDA_MACHINE,
    )

    assert plan.accelerator == "cuda"
    assert "reusing the environment" in plan.reason
    assert record.read_text(encoding="utf-8") == before


def test_an_unremovable_accelerator_environment_stops_the_build_not_the_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A locked .venv-accel degrades to CPU rather than taking the installer down.

    Windows locks directories that antivirus or a live process still holds, and
    the caller degrades on InstallerError alone -- a raw OSError here would
    abort an installation that had every reason to finish on CPU.
    """
    stale = tmp_path / ACCELERATOR_ENVIRONMENT_DIRECTORY
    stale.mkdir(parents=True)

    def locked(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(shutil, "rmtree", locked)

    with pytest.raises(InstallerError) as caught:
        sync_accelerator_environment(tmp_path, "cuda", python_version="3.12", uv_executable="uv")

    # Rebuilding over the leftovers is what the removal exists to prevent, so a
    # failed removal must not be swallowed into a build that then proceeds.
    assert "Could not remove the existing accelerator environment" in str(caught.value)
    assert stale.is_dir()


@pytest.mark.parametrize(
    ("driver", "python_version"),
    [("560.35.03", "3.12"), ("550.54.14", "3.13")],
)
def test_a_driver_or_python_that_moved_forces_the_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str, python_version: str
) -> None:
    """The record vouches for one combination; anything else has to be re-proven."""
    checkout, _, _ = _prepared_checkout(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "incode_mcp.installer.accelerator.interpreter_version", lambda python: python_version
    )
    rebuilt: list[str] = []

    def fake_sync(*_args: object, **_kwargs: object) -> Path:
        rebuilt.append("sync")
        return checkout / ACCELERATOR_ENVIRONMENT_DIRECTORY / "python"

    monkeypatch.setattr("incode_mcp.installer.accelerator.sync_accelerator_environment", fake_sync)
    monkeypatch.setattr(
        "incode_mcp.installer.accelerator.probe_accelerator",
        lambda python, accelerator, *, offline=False: {
            "ok": True,
            "interpreter": str(python),
            "providers": ["CUDAExecutionProvider"],
            "python_version": python_version,
        },
    )

    plan = configure_accelerator(
        checkout,
        "cuda",
        nvidia_report=lambda: f"{driver}, NVIDIA A100",
        platform_name=CUDA_PLATFORM,
        machine=CUDA_MACHINE,
    )

    assert plan.accelerator == "cuda"
    assert rebuilt == ["sync"]


def test_a_changed_lockfile_forces_the_accelerator_environment_to_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, _, _ = _prepared_checkout(tmp_path, monkeypatch)
    (checkout / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    rebuilt: list[str] = []

    def fake_sync(*_args: object, **_kwargs: object) -> Path:
        rebuilt.append("sync")
        return checkout / ACCELERATOR_ENVIRONMENT_DIRECTORY / "python"

    monkeypatch.setattr("incode_mcp.installer.accelerator.sync_accelerator_environment", fake_sync)
    monkeypatch.setattr(
        "incode_mcp.installer.accelerator.probe_accelerator",
        lambda python, accelerator, *, offline=False: {
            "ok": True,
            "interpreter": str(python),
            "providers": ["CUDAExecutionProvider"],
            "runtime_version": "1.23.2",
            "python_version": "3.12",
        },
    )

    plan = configure_accelerator(
        checkout,
        "cuda",
        nvidia_report=lambda: "550.54.14, NVIDIA A100",
        platform_name=CUDA_PLATFORM,
        machine=CUDA_MACHINE,
    )

    assert plan.accelerator == "cuda"
    assert rebuilt == ["sync"]


def test_a_checkout_without_a_lockfile_cannot_fingerprint_an_accelerator(
    tmp_path: Path,
) -> None:
    """A missing lock is a broken checkout, never silently some other lockfile."""

    with pytest.raises(InstallerError, match="lockfile cannot be read"):
        accelerator_lock_fingerprint(tmp_path, "cuda")


def test_migraphx_detection_uses_the_serving_interpreter_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "incode_mcp.installer.accelerator.interpreter_version", lambda python: "3.13"
    )
    monkeypatch.setattr(
        "incode_mcp.installer.accelerator.accelerator_record_path",
        lambda *args, **kwargs: tmp_path / "accelerator.json",
    )

    def capture_plan(requested: str, **kwargs: object) -> object:
        captured.update(kwargs)
        return AcceleratorPlan("cpu", "test plan")

    monkeypatch.setattr("incode_mcp.installer.accelerator.plan_accelerator", capture_plan)

    configure_accelerator(
        checkout,
        "migraphx",
        platform_name="linux",
        machine="x86_64",
    )

    assert captured["python_version"] == "3.13"


def test_a_cpu_installation_reclaims_the_environment_it_no_longer_points_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record and the gigabytes behind it are retired together."""
    checkout, record, accelerator_env = _prepared_checkout(tmp_path, monkeypatch)

    plan = configure_accelerator(checkout, "cpu")

    assert plan.accelerator == "cpu"
    assert not record.exists()
    assert not accelerator_env.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX shell script")
def test_a_probe_that_never_finishes_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output is captured, so an unbounded probe is indistinguishable from a hang."""
    monkeypatch.setattr("incode_mcp.installer.accelerator.PROBE_TIMEOUT_SECONDS", 1)
    stalled = tmp_path / "stalled-python"
    stalled.write_text("#!/bin/sh\nexec sleep 30\n", encoding="utf-8")
    stalled.chmod(0o755)

    with pytest.raises(InstallerError, match="did not finish within"):
        probe_accelerator(stalled, "cuda")
