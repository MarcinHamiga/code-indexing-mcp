"""Tests for the post-install checks."""

import json
import sys
from pathlib import Path

import pytest

from code_indexing_mcp.installer import verify
from code_indexing_mcp.installer.accelerator import server_executable
from code_indexing_mcp.installer.shell_path import LauncherResult

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="the checks run POSIX shell stand-ins"
)


def _checkout(tmp_path: Path, *, script: str = "#!/bin/sh\nexit 0\n") -> Path:
    directory = tmp_path / "checkout"
    command = server_executable(directory)
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text(script, encoding="utf-8")
    command.chmod(0o755)
    return directory


def _check(checks: tuple[verify.Check, ...], name: str) -> verify.Check:
    return next(check for check in checks if check.name == name)


# --- the executable ----------------------------------------------------------


def test_a_runnable_executable_passes(tmp_path: Path) -> None:
    checks = verify.run_checks(_checkout(tmp_path), accelerator_was_prepared=False)
    assert _check(checks, "server executable").ok


def test_a_missing_executable_fails(tmp_path: Path) -> None:
    check = _check(
        verify.run_checks(tmp_path / "never-built", accelerator_was_prepared=False),
        "server executable",
    )
    assert check.status == "fail"
    assert "missing" in check.detail


def test_an_executable_that_errors_fails_with_its_message(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path, script="#!/bin/sh\necho 'broken install' >&2\nexit 3\n")

    check = _check(verify.run_checks(checkout, accelerator_was_prepared=False), "server executable")

    assert check.status == "fail"
    assert "exited 3" in check.detail
    assert "broken install" in check.detail


def test_an_executable_that_hangs_fails_instead_of_hanging_the_wizard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verify, "HELP_TIMEOUT_SECONDS", 0.5)
    checkout = _checkout(tmp_path, script="#!/bin/sh\nsleep 30\n")

    check = _check(verify.run_checks(checkout, accelerator_was_prepared=False), "server executable")

    assert check.status == "fail"
    assert "did not answer" in check.detail


# --- the launcher on PATH ----------------------------------------------------


def test_a_launcher_on_path_passes(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    launcher = bin_directory / "code-indexing-mcp"
    launcher.symlink_to(server_executable(checkout))

    checks = verify.run_checks(
        checkout,
        launcher=LauncherResult(launcher, "created"),
        accelerator_was_prepared=False,
        environment={"PATH": str(bin_directory)},
    )

    assert _check(checks, "command on PATH").ok


def test_a_launcher_awaiting_a_new_shell_is_a_warning_that_says_so(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    launcher = tmp_path / "bin" / "code-indexing-mcp"

    check = _check(
        verify.run_checks(
            checkout,
            launcher=LauncherResult(launcher, "created"),
            profiles_updated=(tmp_path / ".zshrc",),
            accelerator_was_prepared=False,
            environment={"PATH": "", "SHELL": "/bin/zsh"},
        ),
        "command on PATH",
    )

    assert check.status == "warn"
    assert "new shell" in check.detail
    assert "exec /bin/zsh -l" in check.detail


def test_a_launcher_nothing_will_ever_find_is_a_warning(tmp_path: Path) -> None:
    check = _check(
        verify.run_checks(
            _checkout(tmp_path),
            launcher=LauncherResult(tmp_path / "bin" / "code-indexing-mcp", "created"),
            accelerator_was_prepared=False,
            environment={"PATH": ""},
        ),
        "command on PATH",
    )

    assert check.status == "warn"
    assert "not on PATH" in check.detail


def test_a_failed_launcher_is_reported_as_such(tmp_path: Path) -> None:
    check = _check(
        verify.run_checks(
            _checkout(tmp_path),
            launcher=LauncherResult(tmp_path / "bin" / "x", "failed", "no room"),
            accelerator_was_prepared=False,
            environment={"PATH": ""},
        ),
        "command on PATH",
    )
    assert check.status == "warn"
    assert "no room" in check.detail


# --- harness entries ---------------------------------------------------------


def _configure(tmp_path: Path, checkout: Path) -> tuple[str, Path]:
    from code_indexing_mcp.installer.harnesses import configure_harness

    path = configure_harness(
        "kimi-code",
        server_executable(checkout),
        env={},
        home=tmp_path,
        environment={},
    )
    return ("kimi-code", path)


def test_a_freshly_written_harness_entry_passes(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    configured = (_configure(tmp_path, checkout),)

    checks = verify.run_checks(
        checkout,
        configured,
        accelerator_was_prepared=False,
        home=tmp_path,
        environment={},
    )

    assert _check(checks, "kimi-code configuration").ok


def test_an_entry_naming_a_different_command_warns(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    slug, path = _configure(tmp_path, checkout)
    # Someone else's entry, or a stale one left by an install elsewhere.
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mcpServers"]["code-indexing-mcp"]["command"] = "/somewhere/else"
    path.write_text(json.dumps(payload), encoding="utf-8")

    check = _check(
        verify.run_checks(
            checkout,
            ((slug, path),),
            accelerator_was_prepared=False,
            home=tmp_path,
            environment={},
        ),
        "kimi-code configuration",
    )

    assert check.status == "warn"
    assert "/somewhere/else" in check.detail


def test_a_vanished_entry_warns(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    slug, path = _configure(tmp_path, checkout)
    path.write_text("{}", encoding="utf-8")

    check = _check(
        verify.run_checks(
            checkout,
            ((slug, path),),
            accelerator_was_prepared=False,
            home=tmp_path,
            environment={},
        ),
        "kimi-code configuration",
    )

    assert check.status == "warn"
    assert "no server entry" in check.detail


# --- accelerator record and skills -------------------------------------------


def test_a_missing_accelerator_record_warns_only_when_one_was_prepared(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)

    without = verify.run_checks(checkout, accelerator_was_prepared=False)
    assert not any(check.name == "accelerator record" for check in without)

    with_prepared = verify.run_checks(checkout, accelerator_was_prepared=True)
    assert _check(with_prepared, "accelerator record").status == "warn"


def test_a_broken_skill_link_warns(tmp_path: Path) -> None:
    from code_indexing_mcp.installer.harnesses import skill_directory

    checkout = _checkout(tmp_path)
    slug, path = _configure(tmp_path, checkout)
    skills = skill_directory(slug, home=tmp_path, environment={})
    assert skills is not None
    skills.mkdir(parents=True)
    (skills / "gone").symlink_to(tmp_path / "never-existed", target_is_directory=True)

    check = _check(
        verify.run_checks(
            checkout,
            ((slug, path),),
            accelerator_was_prepared=False,
            home=tmp_path,
            environment={},
        ),
        "kimi-code skills",
    )

    assert check.status == "warn"
    assert "gone" in check.detail


def test_format_check_marks_a_failure_distinctly() -> None:
    assert verify.format_check(verify.Check("a", "ok", "fine")) == "ok   - a: fine"
    assert verify.format_check(verify.Check("b", "fail")) == "FAIL - b"
