"""Tests for the launcher and the shell-profile PATH entry."""

import os
import sys
from pathlib import Path

import pytest

from code_indexing_mcp.installer import shell_path
from code_indexing_mcp.installer.accelerator import server_executable
from code_indexing_mcp.installer.config_files import InstallerError


def _checkout(tmp_path: Path, *, platform_name: str | None = None) -> Path:
    """A checkout whose venv holds a stand-in for the built server command."""

    directory = tmp_path / "checkout"
    command = server_executable(directory, platform_name=platform_name)
    command.parent.mkdir(parents=True, exist_ok=True)
    command.touch(mode=0o755)
    return directory


def _path(*entries: Path) -> dict[str, str]:
    return {"PATH": os.pathsep.join(str(entry) for entry in entries)}


# --- where the launcher goes -------------------------------------------------


def test_default_bin_directory_prefers_the_explicit_override(tmp_path: Path) -> None:
    environment = {
        "CODE_INDEXING_MCP_BIN_DIR": str(tmp_path / "explicit"),
        "XDG_BIN_HOME": str(tmp_path / "xdg"),
    }
    assert shell_path.default_bin_directory(home=tmp_path, environment=environment) == (
        tmp_path / "explicit"
    )


def test_default_bin_directory_falls_back_through_xdg_to_local_bin(tmp_path: Path) -> None:
    xdg = shell_path.default_bin_directory(
        home=tmp_path, environment={"XDG_BIN_HOME": str(tmp_path / "xdg")}
    )
    assert xdg == tmp_path / "xdg"
    assert shell_path.default_bin_directory(home=tmp_path, environment={}) == (
        tmp_path / ".local" / "bin"
    )


def test_launcher_path_is_a_cmd_shim_on_windows(tmp_path: Path) -> None:
    assert shell_path.launcher_path(tmp_path, platform_name="linux").name == "code-indexing-mcp"
    assert shell_path.launcher_path(tmp_path, platform_name="win32").name == "code-indexing-mcp.cmd"


# --- creating the launcher ---------------------------------------------------


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX symlink launcher")
def test_install_launcher_creates_a_symlink_to_the_server_executable(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    result = shell_path.install_launcher(checkout, tmp_path / "bin")

    assert result.status == "created"
    assert result.path.is_symlink()
    assert result.path.resolve() == server_executable(checkout).resolve()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX symlink launcher")
def test_install_launcher_is_idempotent(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    shell_path.install_launcher(checkout, tmp_path / "bin")
    again = shell_path.install_launcher(checkout, tmp_path / "bin")

    assert again.status == "current"


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX symlink launcher")
def test_install_launcher_backs_up_a_file_it_did_not_create(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    theirs = bin_directory / "code-indexing-mcp"
    theirs.write_text("#!/bin/sh\necho mine\n", encoding="utf-8")

    result = shell_path.install_launcher(checkout, bin_directory)

    assert result.status == "replaced"
    # The name in the message is the name the file actually ended up under.
    assert (bin_directory / result.detail.rsplit(" ", 1)[-1]).read_text(
        encoding="utf-8"
    ) == "#!/bin/sh\necho mine\n"


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX symlink launcher")
def test_install_launcher_replaces_its_own_link_without_a_backup(tmp_path: Path) -> None:
    first = _checkout(tmp_path)
    second = tmp_path / "second"
    command = server_executable(second)
    command.parent.mkdir(parents=True, exist_ok=True)
    command.touch(mode=0o755)
    bin_directory = tmp_path / "bin"
    shell_path.install_launcher(first, bin_directory)

    result = shell_path.install_launcher(second, bin_directory)

    assert result.status == "created"
    assert result.path.resolve() == command.resolve()
    assert not (bin_directory / "code-indexing-mcp.bak").exists()


def test_install_launcher_reports_a_checkout_with_no_built_environment(tmp_path: Path) -> None:
    result = shell_path.install_launcher(tmp_path / "never-built", tmp_path / "bin")

    assert result.status == "failed"
    assert not result.ok
    assert "no server executable" in result.detail


def test_install_launcher_writes_a_batch_shim_on_windows(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path, platform_name="win32")
    bin_directory = tmp_path / "bin"

    result = shell_path.install_launcher(checkout, bin_directory, platform_name="win32")

    assert result.status == "created"
    content = result.path.read_text(encoding="utf-8")
    assert content.startswith("@echo off")
    assert str(server_executable(checkout, platform_name="win32")) in content
    assert (
        shell_path.install_launcher(checkout, bin_directory, platform_name="win32").status
        == "current"
    )


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX symlink launcher")
def test_remove_launcher_takes_back_only_its_own(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    bin_directory = tmp_path / "bin"
    shell_path.install_launcher(checkout, bin_directory)

    assert shell_path.remove_launcher(bin_directory) == bin_directory / "code-indexing-mcp"
    assert shell_path.remove_launcher(bin_directory) is None

    theirs = bin_directory / "code-indexing-mcp"
    theirs.write_text("mine\n", encoding="utf-8")
    assert shell_path.remove_launcher(bin_directory) is None
    assert theirs.is_file()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX symlink launcher")
def test_remove_launcher_leaves_a_link_into_another_venv_alone(tmp_path: Path) -> None:
    """Pointing at *a* virtual environment is not evidence of pointing at ours."""

    ours = _checkout(tmp_path)
    theirs = tmp_path / "their-project"
    their_command = server_executable(theirs)
    their_command.parent.mkdir(parents=True)
    their_command.touch(mode=0o755)
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    (bin_directory / "code-indexing-mcp").symlink_to(their_command)

    assert shell_path.remove_launcher(bin_directory, ours) is None
    assert (bin_directory / "code-indexing-mcp").is_symlink()

    assert shell_path.remove_launcher(bin_directory, theirs) == bin_directory / "code-indexing-mcp"


def test_remove_launcher_checks_the_shim_names_this_checkout_on_windows(tmp_path: Path) -> None:
    ours = _checkout(tmp_path, platform_name="win32")
    theirs = tmp_path / "their-project"
    their_command = server_executable(theirs, platform_name="win32")
    their_command.parent.mkdir(parents=True)
    their_command.touch()
    bin_directory = tmp_path / "bin"
    shell_path.install_launcher(theirs, bin_directory, platform_name="win32")

    assert shell_path.remove_launcher(bin_directory, ours, platform_name="win32") is None
    assert shell_path.remove_launcher(bin_directory, theirs, platform_name="win32") is not None


# --- PATH detection ----------------------------------------------------------


def test_is_on_path_compares_directories_not_strings(tmp_path: Path) -> None:
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    assert shell_path.is_on_path(bin_directory, environment=_path(bin_directory))
    assert shell_path.is_on_path(
        bin_directory, environment=_path(tmp_path / "bin" / "." / "..", bin_directory)
    )
    assert not shell_path.is_on_path(bin_directory, environment=_path(tmp_path / "other"))


def test_is_on_path_survives_a_malformed_entry(tmp_path: Path) -> None:
    bin_directory = tmp_path / "bin"
    environment = {"PATH": os.pathsep.join(["", "\0broken", str(bin_directory)])}
    assert shell_path.is_on_path(bin_directory, environment=environment)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX executable bit")
def test_shadowing_executable_finds_an_earlier_entry(tmp_path: Path) -> None:
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    other = earlier / "code-indexing-mcp"
    other.write_text("#!/bin/sh\n", encoding="utf-8")
    other.chmod(0o755)
    ours = tmp_path / "bin"

    assert shell_path.shadowing_executable(ours, environment=_path(earlier, ours)) == other
    # Ours comes first, so nothing shadows it.
    assert shell_path.shadowing_executable(ours, environment=_path(ours, earlier)) is None


# --- shell profiles ----------------------------------------------------------


def test_shell_profiles_leads_with_the_login_shell(tmp_path: Path) -> None:
    profiles = shell_path.shell_profiles(
        home=tmp_path, environment={"SHELL": "/bin/zsh"}, platform_name="linux"
    )
    assert profiles[0] == tmp_path / ".zshrc"


def test_shell_profiles_honours_zdotdir(tmp_path: Path) -> None:
    zdotdir = tmp_path / "config" / "zsh"
    profiles = shell_path.shell_profiles(
        home=tmp_path,
        environment={"SHELL": "/bin/zsh", "ZDOTDIR": str(zdotdir)},
        platform_name="linux",
    )
    assert profiles[0] == zdotdir / ".zshrc"


def test_shell_profiles_adds_bash_profile_on_macos(tmp_path: Path) -> None:
    profiles = shell_path.shell_profiles(
        home=tmp_path, environment={"SHELL": "/bin/bash"}, platform_name="darwin"
    )
    assert profiles[:2] == (tmp_path / ".bashrc", tmp_path / ".bash_profile")


def test_shell_profiles_includes_other_profiles_that_already_exist(tmp_path: Path) -> None:
    (tmp_path / ".bashrc").write_text("", encoding="utf-8")
    profiles = shell_path.shell_profiles(
        home=tmp_path, environment={"SHELL": "/bin/zsh"}, platform_name="linux"
    )
    assert profiles[0] == tmp_path / ".zshrc"
    assert tmp_path / ".bashrc" in profiles


def test_shell_profiles_is_empty_on_windows(tmp_path: Path) -> None:
    assert shell_path.shell_profiles(home=tmp_path, environment={}, platform_name="win32") == ()


# --- the PATH block ----------------------------------------------------------


def test_update_profile_appends_a_marked_block_once(tmp_path: Path) -> None:
    profile = tmp_path / ".zshrc"
    profile.write_text("alias ll='ls -l'\n", encoding="utf-8")
    bin_directory = tmp_path / ".local" / "bin"

    assert shell_path.update_profile(profile, bin_directory, home=tmp_path) is True
    assert shell_path.update_profile(profile, bin_directory, home=tmp_path) is False

    text = profile.read_text(encoding="utf-8")
    assert text.count(shell_path.BLOCK_START) == 1
    assert text.startswith("alias ll='ls -l'\n")
    assert text.rstrip().endswith(shell_path.BLOCK_END)
    # Written relative to $HOME so the line keeps working from a moved home.
    assert 'export PATH="$HOME/.local/bin:$PATH"' in text
    assert (profile.with_name(".zshrc.bak")).read_text(encoding="utf-8") == "alias ll='ls -l'\n"


def test_update_profile_adds_a_newline_before_the_block(tmp_path: Path) -> None:
    profile = tmp_path / ".zshrc"
    profile.write_text("no trailing newline", encoding="utf-8")

    shell_path.update_profile(profile, tmp_path / "bin", home=tmp_path)

    assert profile.read_text(encoding="utf-8").startswith("no trailing newline\n")


def test_update_profile_creates_a_missing_profile(tmp_path: Path) -> None:
    profile = tmp_path / ".profile"

    assert shell_path.update_profile(profile, tmp_path / "bin", home=tmp_path) is True
    assert shell_path.BLOCK_START in profile.read_text(encoding="utf-8")


def test_update_profile_leaves_a_hand_written_path_line_alone(tmp_path: Path) -> None:
    profile = tmp_path / ".zshrc"
    profile.write_text('export PATH="$HOME/.local/bin:$PATH"\n', encoding="utf-8")

    assert shell_path.update_profile(profile, tmp_path / ".local" / "bin", home=tmp_path) is False
    assert shell_path.BLOCK_START not in profile.read_text(encoding="utf-8")


def test_update_profile_uses_fish_syntax_for_a_fish_config(tmp_path: Path) -> None:
    profile = tmp_path / ".config" / "fish" / "config.fish"
    profile.parent.mkdir(parents=True)
    profile.write_text("", encoding="utf-8")

    shell_path.update_profile(profile, tmp_path / ".local" / "bin", home=tmp_path)

    assert 'fish_add_path "$HOME/.local/bin"' in profile.read_text(encoding="utf-8")


def test_update_profile_quotes_a_directory_with_a_space(tmp_path: Path) -> None:
    """--bin-dir takes whatever the user types; the profile has to survive it."""

    profile = tmp_path / ".zshrc"
    profile.write_text("", encoding="utf-8")
    fish = tmp_path / "config.fish"
    fish.write_text("", encoding="utf-8")
    awkward = tmp_path / "my bin"

    shell_path.update_profile(profile, awkward, home=tmp_path)
    shell_path.update_profile(fish, awkward, home=tmp_path)

    assert 'export PATH="$HOME/my bin:$PATH"' in profile.read_text(encoding="utf-8")
    assert 'fish_add_path "$HOME/my bin"' in fish.read_text(encoding="utf-8")


def test_update_profile_escapes_shell_metacharacters_in_the_directory(tmp_path: Path) -> None:
    profile = tmp_path / ".zshrc"
    profile.write_text("", encoding="utf-8")

    shell_path.update_profile(profile, Path('/opt/a"b`c$d'), home=tmp_path)

    line = next(
        text
        for text in profile.read_text(encoding="utf-8").splitlines()
        if text.startswith("export")
    )
    assert line == 'export PATH="/opt/a\\"b\\`c\\$d:$PATH"'


def test_update_profile_does_not_mistake_a_neighbouring_directory_for_this_one(
    tmp_path: Path,
) -> None:
    """A substring match here would leave the user with no PATH entry at all."""

    profile = tmp_path / ".zshrc"
    profile.write_text('export PATH="$HOME/bin2:$PATH"\n', encoding="utf-8")

    assert shell_path.update_profile(profile, tmp_path / "bin", home=tmp_path) is True
    assert shell_path.BLOCK_START in profile.read_text(encoding="utf-8")


def test_update_profile_honours_a_tilde_spelled_path_line(tmp_path: Path) -> None:
    profile = tmp_path / ".zshrc"
    profile.write_text('export PATH="~/.local/bin:$PATH"\n', encoding="utf-8")

    assert shell_path.update_profile(profile, tmp_path / ".local" / "bin", home=tmp_path) is False


def test_update_profile_reports_a_profile_it_cannot_decode(tmp_path: Path) -> None:
    """Silently returning False here would read as "already configured"."""

    profile = tmp_path / ".zshrc"
    profile.write_bytes(b"\xff\xfe not utf-8 \x00\n")

    with pytest.raises(InstallerError):
        shell_path.update_profile(profile, tmp_path / "bin", home=tmp_path)

    written, failures = shell_path.update_profiles(tmp_path / "bin", [profile], home=tmp_path)
    assert written == []
    assert [path for path, _ in failures] == [profile]


def test_update_profiles_isolates_one_files_failure(tmp_path: Path) -> None:
    good = tmp_path / ".zshrc"
    # A directory where a file belongs: writing it fails, the other still lands.
    bad = tmp_path / ".bashrc"
    bad.mkdir()

    written, failures = shell_path.update_profiles(tmp_path / "bin", [bad, good], home=tmp_path)

    assert written == [good]
    assert [path for path, _ in failures] == [bad]


def test_remove_path_block_takes_back_exactly_what_was_added(tmp_path: Path) -> None:
    profile = tmp_path / ".zshrc"
    original = "alias ll='ls -l'\nexport EDITOR=vim\n"
    profile.write_text(original, encoding="utf-8")
    shell_path.update_profile(profile, tmp_path / "bin", home=tmp_path)

    assert shell_path.remove_path_block(profile) is True
    assert profile.read_text(encoding="utf-8") == original
    assert shell_path.remove_path_block(profile) is False


def test_remove_path_block_refuses_a_block_the_user_broke(tmp_path: Path) -> None:
    profile = tmp_path / ".zshrc"
    profile.write_text(f"{shell_path.BLOCK_START}\nexport PATH=x\n", encoding="utf-8")

    # No end marker: removing to end-of-file would take the user's edits with it.
    assert shell_path.remove_path_block(profile) is False


# --- the summary the wizard renders -----------------------------------------


def test_inspect_reports_the_full_situation(tmp_path: Path) -> None:
    bin_directory = tmp_path / ".local" / "bin"
    profile = tmp_path / ".zshrc"
    profile.write_text("", encoding="utf-8")
    environment = {"SHELL": "/bin/zsh", "PATH": str(tmp_path / "elsewhere")}

    state = shell_path.inspect(
        bin_directory, home=tmp_path, environment=environment, platform_name="linux"
    )

    assert state.on_path is False
    assert state.profiles_current is False
    assert profile in state.profiles
    assert state.launcher == bin_directory / "code-indexing-mcp"

    shell_path.update_profile(profile, bin_directory, home=tmp_path)
    refreshed = shell_path.inspect(
        bin_directory, home=tmp_path, environment=environment, platform_name="linux"
    )
    assert refreshed.profiles_current is True


def test_activation_hint_matches_the_shell(tmp_path: Path) -> None:
    fish = tmp_path / "config.fish"
    assert shell_path.activation_hint([fish], environment={}) == "exec fish"
    assert (
        shell_path.activation_hint([tmp_path / ".zshrc"], environment={"SHELL": "/bin/zsh"})
        == "exec /bin/zsh -l"
    )
