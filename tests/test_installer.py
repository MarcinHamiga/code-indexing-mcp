import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

INSTALLER_PATH = Path(__file__).parents[1] / "install.py"
SHELL_INSTALLER_PATH = Path(__file__).parents[1] / "install.sh"

# The installer stringifies Path objects with the native separator, so expected
# values must go through the same conversion to stay correct on Windows.
SERVER_BINARY = Path("/opt/ci-mcp")
SERVER_COMMAND = str(SERVER_BINARY)


def load_installer() -> ModuleType:
    assert INSTALLER_PATH.exists(), "install.py does not exist"
    spec = importlib.util.spec_from_file_location("code_indexing_mcp_installer", INSTALLER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = load_installer()


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


def test_jsonc_merge_validates_unrelated_nested_values(tmp_path: Path) -> None:
    installer = load_installer()
    path = tmp_path / "config.jsonc"
    original = '{"unrelated": {"broken": nope}}\n'
    path.write_text(original)

    with pytest.raises(installer.InstallerError, match="Invalid JSON"):
        installer.merge_json_object_entry(path, "mcp", "server", {"enabled": True})

    assert path.read_text() == original
    assert not path.with_name("config.jsonc.bak").exists()


def test_codex_merge_creates_server_table(tmp_path: Path) -> None:
    installer = load_installer()
    path = tmp_path / "config.toml"

    changed = installer.merge_codex_server(path, SERVER_BINARY)

    encoded_command = json.dumps(SERVER_COMMAND, ensure_ascii=False)
    assert changed is True
    assert path.read_text() == (
        f'[mcp_servers.code-indexing-mcp]\ncommand = {encoded_command}\nargs = ["serve"]\n'
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

# Keep the feature explanation too.
[features]
example = true
"""
    path.write_text(original)

    changed = installer.merge_codex_server(path, Path("/new/ci-mcp"))

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
    installer = load_installer()
    path = tmp_path / "config.toml"
    original = """[mcp_servers.code-indexing-mcp]
command = "old"

[[skills.config]]
path = "/tmp/skill"
enabled = false
"""
    path.write_text(original)

    installer.merge_codex_server(path, Path("/new/ci-mcp"))

    parsed = tomllib.loads(path.read_text())
    assert parsed["skills"]["config"] == [
        {
            "path": "/tmp/skill",
            "enabled": False,
        }
    ]
    assert path.with_name("config.toml.bak").read_text() == original


def test_codex_merge_is_idempotent(tmp_path: Path) -> None:
    installer = load_installer()
    path = tmp_path / "config.toml"

    assert installer.merge_codex_server(path, Path("/opt/ci-mcp")) is True
    first = path.read_text()
    assert installer.merge_codex_server(path, Path("/opt/ci-mcp")) is False

    assert path.read_text() == first


def test_codex_merge_rejects_inline_target_without_corrupting_config(tmp_path: Path) -> None:
    installer = load_installer()
    path = tmp_path / "config.toml"
    original = '[mcp_servers]\ncode-indexing-mcp = { command = "old", args = ["old"] }\n'
    path.write_text(original)

    with pytest.raises(installer.InstallerError, match="inline or dotted"):
        installer.merge_codex_server(path, Path("/opt/ci-mcp"))

    assert path.read_text() == original
    assert not path.with_name("config.toml.bak").exists()


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

    assert (
        installer.configuration_path(
            "codex", home=tmp_path, environment=environment, platform_name="darwin"
        )
        == tmp_path / "codex-home" / "config.toml"
    )
    assert (
        installer.configuration_path(
            "kimi-code", home=tmp_path, environment=environment, platform_name="darwin"
        )
        == tmp_path / "kimi-home" / "mcp.json"
    )
    assert (
        installer.configuration_path(
            "opencode", home=tmp_path, environment=environment, platform_name="darwin"
        )
        == tmp_path / "custom-opencode.jsonc"
    )
    assert (
        installer.configuration_path(
            "kilocode", home=tmp_path, environment=environment, platform_name="darwin"
        )
        == tmp_path / "xdg" / "kilo" / "kilo.jsonc"
    )
    assert (
        installer.configuration_path(
            "claude-desktop", home=tmp_path, environment=environment, platform_name="win32"
        )
        == tmp_path / "appdata" / "Claude" / "claude_desktop_config.json"
    )


def test_claude_code_honors_config_directory_override(tmp_path: Path) -> None:
    installer = load_installer()
    config_directory = tmp_path / "claude-config"

    assert (
        installer.configuration_path(
            "claude-code",
            home=tmp_path,
            environment={"CLAUDE_CONFIG_DIR": str(config_directory)},
            platform_name="linux",
        )
        == config_directory / ".claude.json"
    )


def test_kilocode_honors_config_file_override(tmp_path: Path) -> None:
    installer = load_installer()
    config_file = tmp_path / "custom-kilo.jsonc"

    assert (
        installer.configuration_path(
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
    installer = load_installer()
    config_directory = tmp_path / "kilo-config"

    assert (
        installer.configuration_path(
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
                "env": {},
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
    installer = load_installer()

    path = installer.configure_harness(
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
    installer = load_installer()

    path = installer.configure_harness(
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
    installer = load_installer()

    assert (
        installer.configuration_path(
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
            "mkdir -p .venv/bin\n"
            "touch .venv/bin/code-indexing-mcp\n"
        )
        fake_uv.chmod(0o755)

    command = installer.sync_environment(checkout, uv_executable=str(fake_uv))

    assert command == expected


def test_configure_selected_harnesses_isolates_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer = load_installer()
    attempted: list[str] = []

    def fake_configure(slug: str, command: Path, **kwargs: object) -> Path:
        attempted.append(slug)
        if slug == "claude-code":
            raise installer.InstallerError("broken config")
        return tmp_path / f"{slug}.json"

    monkeypatch.setattr(installer, "configure_harness", fake_configure)

    successes, failures = installer.configure_selected_harnesses(
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
    installer = load_installer()
    (tmp_path / ".claude.json").write_bytes(b"\xff")

    successes, failures = installer.configure_selected_harnesses(
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


def test_main_runs_noninteractive_install_and_reports_harness_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer = load_installer()
    checkout = tmp_path / "checkout"
    command = checkout / ".venv" / "bin" / "code-indexing-mcp"
    calls: list[object] = []
    output: list[str] = []
    errors: list[str] = []

    def fake_repository(repository_url: str, install_directory: Path) -> str:
        calls.append(("repository", repository_url, install_directory))
        return "installed"

    def fake_sync(install_directory: Path) -> Path:
        calls.append(("sync", install_directory))
        return command

    def fake_configure(
        slugs: list[str], server_command: Path
    ) -> tuple[list[tuple[str, Path]], list[tuple[str, str]]]:
        calls.append(("configure", slugs, server_command))
        return [("codex", tmp_path / "config.toml")], [("kimi-code", "invalid JSON")]

    monkeypatch.setattr(installer, "clone_or_update_repository", fake_repository)
    monkeypatch.setattr(installer, "sync_environment", fake_sync)
    monkeypatch.setattr(installer, "configure_selected_harnesses", fake_configure)

    status = installer.main(
        [
            "--repo-url",
            "https://example.test/repo.git",
            "--install-dir",
            str(checkout),
            "--harnesses",
            "codex,kimi-code",
        ],
        output_fn=output.append,
        error_fn=errors.append,
    )

    assert status == 1
    assert calls == [
        ("repository", "https://example.test/repo.git", checkout),
        ("sync", checkout),
        ("configure", ["codex", "kimi-code"], command),
    ]
    assert any("Installed repository" in line for line in output)
    assert any("Configured Codex" in line for line in output)
    assert errors == ["Failed to configure Kimi Code: invalid JSON"]


def test_main_prompts_for_harnesses_when_option_is_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer = load_installer()
    selected: list[str] = []
    output: list[str] = []

    monkeypatch.setattr(
        installer,
        "clone_or_update_repository",
        lambda repository_url, install_directory: "updated",
    )
    monkeypatch.setattr(
        installer,
        "sync_environment",
        lambda install_directory: tmp_path / "server",
    )

    def fake_configure(
        slugs: list[str], command: Path
    ) -> tuple[list[tuple[str, Path]], list[tuple[str, str]]]:
        selected.extend(slugs)
        return [], []

    monkeypatch.setattr(installer, "configure_selected_harnesses", fake_configure)

    status = installer.main(
        ["--install-dir", str(tmp_path / "checkout")],
        input_fn=lambda prompt: "1,3",
        output_fn=output.append,
        error_fn=output.append,
    )

    assert status == 0
    assert selected == ["codex", "kimi-code"]
    assert any("Codex (CLI + Desktop)" in line for line in output)
    assert any("Updated repository" in line for line in output)


def test_main_reports_actionable_installer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer = load_installer()
    errors: list[str] = []

    def fail(repository_url: str, install_directory: Path) -> str:
        raise installer.InstallerError("Git is required")

    monkeypatch.setattr(installer, "clone_or_update_repository", fail)

    status = installer.main(
        ["--install-dir", str(tmp_path / "checkout"), "--harnesses", ""],
        output_fn=lambda message: None,
        error_fn=errors.append,
    )

    assert status == 1
    assert errors == ["Error: Git is required"]


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
        installer.skill_directory("claude-code", home=tmp_path, environment={})
        == tmp_path / ".claude" / "skills"
    )
    assert (
        installer.skill_directory("codex", home=tmp_path, environment={})
        == tmp_path / ".agents" / "skills"
    )
    assert (
        installer.skill_directory("kimi-code", home=tmp_path, environment={})
        == tmp_path / ".agents" / "skills"
    )
    assert (
        installer.skill_directory(
            "opencode", home=tmp_path, environment={"XDG_CONFIG_HOME": str(tmp_path / "xdg")}
        )
        == tmp_path / "xdg" / "opencode" / "skills"
    )
    assert installer.skill_directory("claude-desktop", home=tmp_path, environment={}) is None
    assert installer.skill_directory("kilocode", home=tmp_path, environment={}) is None


def test_skill_directory_honors_claude_config_dir(tmp_path: Path) -> None:
    environment = {"CLAUDE_CONFIG_DIR": str(tmp_path / "custom-claude")}
    assert (
        installer.skill_directory("claude-code", home=tmp_path, environment=environment)
        == tmp_path / "custom-claude" / "skills"
    )


def test_install_skills_links_bundled_skills(tmp_path: Path) -> None:
    repo = _skills_source(tmp_path)

    results = installer.install_skills(["claude-code"], repo, home=tmp_path, environment={})

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
    installer.install_skills(["codex"], repo, home=tmp_path, environment={})

    results = installer.install_skills(["codex"], repo, home=tmp_path, environment={})

    _slug, message = results[0]
    assert "0 linked" in message
    assert "already installed" in message


def test_install_skills_backs_up_clashing_directory(tmp_path: Path) -> None:
    repo = _skills_source(tmp_path)
    clash = tmp_path / ".agents" / "skills" / "alpha"
    clash.mkdir(parents=True)
    (clash / "SKILL.md").write_text("old", encoding="utf-8")

    installer.install_skills(["kimi-code"], repo, home=tmp_path, environment={})

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

    installer.install_skills(["kimi-code"], repo, home=tmp_path, environment={})

    link = skills_dir / "alpha"
    assert link.is_symlink()
    assert link.resolve() == (repo / "src" / "incode_mcp" / "skills" / "alpha").resolve()
    backup = skills_dir / "alpha.bak"
    assert backup.is_symlink()
    assert backup.resolve() == elsewhere.resolve()


def test_install_skills_skips_unsupported_harness(tmp_path: Path) -> None:
    repo = _skills_source(tmp_path)

    results = installer.install_skills(["claude-desktop"], repo, home=tmp_path, environment={})

    slug, message = results[0]
    assert slug == "claude-desktop"
    assert "skipped" in message
    assert not (tmp_path / ".claude").exists()


def test_install_skills_reports_missing_source(tmp_path: Path) -> None:
    results = installer.install_skills(
        ["codex"], tmp_path / "empty-repo", home=tmp_path, environment={}
    )

    _slug, message = results[0]
    assert "skipped" in message
    assert not (tmp_path / ".agents").exists()
