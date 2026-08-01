"""The two-phase updater: what it refuses, what it pulls, and what it finalizes."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import pytest
from conftest import create_test_remote, run_git
from filelock import FileLock

from code_indexing_mcp.installer import update
from code_indexing_mcp.installer.accelerator import (
    ACCELERATOR_EXTRAS,
    AcceleratorPlan,
    accelerator_lock_fingerprint,
    environment_python,
)

# The fake uv and the fake server executable are POSIX shell scripts; the
# Windows shield tests below stay out of this and drive plain files instead.
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX shell stand-ins")


class Installation(NamedTuple):
    checkout: Path
    remote: Path
    publisher: Path
    uv: Path
    log: Path


def _install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Installation:
    remote, publisher = create_test_remote(tmp_path)
    checkout = tmp_path / "install"
    run_git("clone", str(remote), str(checkout))
    run_git("config", "user.name", "Updater Tests", cwd=checkout)
    run_git("config", "user.email", "updater@example.test", cwd=checkout)
    (checkout / "uv.lock").write_text("version = 1\n")

    binaries = checkout / ".venv" / "bin"
    binaries.mkdir(parents=True)
    server = binaries / "code-indexing-mcp"
    server.write_text("#!/bin/sh\nexit 0\n")
    server.chmod(0o755)

    log = tmp_path / "events.log"
    uv = tmp_path / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        "{\n"
        "  printf 'sync'\n"
        '  for argument in "$@"; do printf \' %s\' "$argument"; done\n'
        '  printf \' cwd=%s version=%s\\n\' "$(pwd)" "$(cat version.txt)"\n'
        f'}} >> "{log}"\n'
    )
    uv.chmod(0o755)

    monkeypatch.setenv("CODE_INDEXING_MCP_REPO_URL", str(remote))
    monkeypatch.setenv("CODE_INDEXING_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CODE_INDEXING_CACHE_DIR", str(tmp_path / "cache"))
    return Installation(checkout, remote, publisher, uv, log)


def _publish(installation: Installation, content: str) -> None:
    (installation.publisher / "version.txt").write_text(content)
    run_git("add", "version.txt", cwd=installation.publisher)
    run_git("commit", "-m", content, cwd=installation.publisher)
    run_git("push", cwd=installation.publisher)


def _head(directory: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _recorder(log: Path, returncode: int = 0) -> tuple[list[tuple[list[str], Path]], Any]:
    calls: list[tuple[list[str], Path]] = []

    def spawn(argv: list[str], cwd: Path) -> int:
        calls.append((argv, cwd))
        with log.open("a", encoding="utf-8") as handle:
            handle.write("spawn\n")
        return returncode

    return calls, spawn


def _run_update(installation: Installation, spawn: Any, **overrides: Any) -> int:
    arguments: dict[str, Any] = {
        "install_dir": str(installation.checkout),
        "check": False,
        "skip_accelerator": False,
        "finalize": False,
        "previous_sha": None,
        "uv_executable": str(installation.uv),
        "spawn": spawn,
    }
    arguments.update(overrides)
    return update.update_main(**arguments)


def _events(installation: Installation) -> list[str]:
    if not installation.log.exists():
        return []
    return installation.log.read_text(encoding="utf-8").splitlines()


@posix_only
def test_update_refuses_a_directory_without_an_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    (installation.checkout / ".venv" / "bin" / "code-indexing-mcp").unlink()
    calls, spawn = _recorder(installation.log)

    assert _run_update(installation, spawn) == 1

    assert "no installation found at" in capsys.readouterr().err
    assert calls == []


@posix_only
def test_update_refuses_while_another_update_holds_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    calls, spawn = _recorder(installation.log)
    lock = FileLock(installation.checkout / update.UPDATE_LOCK_NAME)
    lock.acquire(timeout=0)
    try:
        assert _run_update(installation, spawn) == 1
    finally:
        lock.release()

    assert "another update is already running" in capsys.readouterr().err
    assert calls == []


@posix_only
def test_update_refuses_without_git_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    calls, spawn = _recorder(installation.log)
    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name, *args, **kwargs: None if name == "git" else real_which(name, *args, **kwargs),
    )

    assert _run_update(installation, spawn) == 1

    assert "git is required" in capsys.readouterr().err
    assert calls == []


@posix_only
def test_update_refuses_a_directory_that_is_not_a_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    shutil.rmtree(installation.checkout / ".git")
    calls, spawn = _recorder(installation.log)

    assert _run_update(installation, spawn) == 1

    assert "is not a git checkout" in capsys.readouterr().err
    assert calls == []


@posix_only
def test_update_refuses_a_checkout_of_another_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    other, _ = create_test_remote(tmp_path, "other")
    monkeypatch.setenv("CODE_INDEXING_MCP_REPO_URL", str(other))
    head = _head(installation.checkout)
    calls, spawn = _recorder(installation.log)

    assert _run_update(installation, spawn) == 1

    assert "tracks" in capsys.readouterr().err
    assert _head(installation.checkout) == head
    assert calls == []


@posix_only
def test_update_refuses_a_checkout_on_another_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    run_git("switch", "-c", "experiment", cwd=installation.checkout)
    calls, spawn = _recorder(installation.log)

    assert _run_update(installation, spawn) == 1

    assert "not main" in capsys.readouterr().err
    assert calls == []


@posix_only
def test_update_refuses_a_dirty_checkout_and_names_the_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    (installation.checkout / "version.txt").write_text("edited\n")
    head = _head(installation.checkout)
    calls, spawn = _recorder(installation.log)

    assert _run_update(installation, spawn) == 1

    message = capsys.readouterr().err
    assert "uncommitted changes" in message
    assert "version.txt" in message
    assert _head(installation.checkout) == head
    assert calls == []


@posix_only
def test_update_refuses_before_pulling_when_uv_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    _publish(installation, "two\n")
    head = _head(installation.checkout)
    calls, spawn = _recorder(installation.log)
    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name, *args, **kwargs: None if name == "uv" else real_which(name, *args, **kwargs),
    )

    assert _run_update(installation, spawn, uv_executable=None) == 1

    assert "uv is required" in capsys.readouterr().err
    assert _head(installation.checkout) == head
    assert calls == []


@posix_only
def test_update_refuses_when_the_remote_cannot_be_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    shutil.rmtree(installation.remote)
    head = _head(installation.checkout)
    calls, spawn = _recorder(installation.log)

    assert _run_update(installation, spawn) == 1

    assert "could not fetch origin/main" in capsys.readouterr().err
    assert _head(installation.checkout) == head
    assert calls == []


@posix_only
def test_update_refuses_a_diverged_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    _publish(installation, "two\n")
    (installation.checkout / "local.txt").write_text("local\n")
    run_git("add", "local.txt", cwd=installation.checkout)
    run_git("commit", "-m", "local", cwd=installation.checkout)
    head = _head(installation.checkout)
    calls, spawn = _recorder(installation.log)

    assert _run_update(installation, spawn) == 1

    assert "diverged" in capsys.readouterr().err
    assert _head(installation.checkout) == head
    assert calls == []
    assert _events(installation) == []


@posix_only
def test_update_merges_then_syncs_then_hands_off_to_the_new_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installation = _install(tmp_path, monkeypatch)
    previous = _head(installation.checkout)
    _publish(installation, "two\n")
    calls, spawn = _recorder(installation.log, returncode=7)

    assert _run_update(installation, spawn) == 7

    assert (installation.checkout / "version.txt").read_text() == "two\n"
    events = _events(installation)
    assert len(events) == 2
    # The sync ran after the merge (it saw the new file) and before the hand-off.
    assert events[0].startswith("sync sync --locked --extra cpu --extra tui")
    assert f"cwd={installation.checkout}" in events[0]
    assert "version=two" in events[0]
    assert events[1] == "spawn"
    argv, cwd = calls[0]
    assert argv == [
        str(environment_python(installation.checkout / ".venv")),
        "-m",
        "code_indexing_mcp",
        "update",
        "--finalize",
        "--previous-sha",
        previous,
        "--install-dir",
        str(installation.checkout),
    ]
    assert cwd == installation.checkout


@posix_only
def test_update_passes_skip_accelerator_to_the_finalize_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installation = _install(tmp_path, monkeypatch)
    _publish(installation, "two\n")
    calls, spawn = _recorder(installation.log)

    assert _run_update(installation, spawn, skip_accelerator=True) == 0

    assert "--skip-accelerator" in calls[0][0]


@posix_only
def test_an_up_to_date_checkout_still_syncs_and_finalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An interrupted update leaves the checkout current and everything after it undone."""

    installation = _install(tmp_path, monkeypatch)
    calls, spawn = _recorder(installation.log)

    assert _run_update(installation, spawn) == 0

    assert "Already up to date." in capsys.readouterr().out
    assert [event.split()[0] for event in _events(installation)] == ["sync", "spawn"]
    assert len(calls) == 1


@posix_only
def test_a_failed_sync_reports_that_the_environment_is_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    _publish(installation, "two\n")
    installation.uv.write_text("#!/bin/sh\necho 'resolution failed' >&2\nexit 1\n")
    installation.uv.chmod(0o755)
    calls, spawn = _recorder(installation.log)

    assert _run_update(installation, spawn) == 1

    message = capsys.readouterr().err
    assert "resolution failed" in message
    assert "re-run `code-indexing-mcp update`" in message
    assert calls == []


def _record_accelerator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkout: Path, fingerprint: str
) -> Path:
    record = tmp_path / "accelerator.json"
    record.write_text(
        json.dumps({"accelerator": "cuda", "lock_fingerprint": fingerprint}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "code_indexing_mcp.installer.accelerator.accelerator_record_path",
        lambda directory, **kwargs: record,
    )
    monkeypatch.setattr(update, "accelerator_record_path", lambda directory, **kwargs: record)
    return record


def _quiet_finalize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update, "load_prefill", lambda: _Prefill())
    monkeypatch.setattr(update, "daemon_supported", lambda: False)


class _Prefill:
    configured_slugs: tuple[str, ...] = ()


@posix_only
def test_finalize_rebuilds_the_accelerator_when_the_lockfile_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    _record_accelerator(tmp_path, monkeypatch, installation.checkout, "stale")
    _quiet_finalize(monkeypatch)
    rebuilt: list[tuple[Path, str]] = []

    def configure(directory: Path, requested: str, **kwargs: Any) -> AcceleratorPlan:
        rebuilt.append((directory, requested))
        return AcceleratorPlan("cuda", "rebuilt against the new lockfile")

    monkeypatch.setattr(update, "configure_accelerator", configure)

    assert (
        update.update_main(
            install_dir=str(installation.checkout),
            check=False,
            skip_accelerator=False,
            finalize=True,
            previous_sha="0" * 40,
        )
        == 0
    )

    assert rebuilt == [(installation.checkout, "cuda")]
    output = capsys.readouterr().out
    assert "[accelerator] ok: cuda (rebuilt against the new lockfile)" in output
    assert "Restart your MCP clients to load the updated server." in output


@posix_only
def test_finalize_leaves_a_matching_accelerator_environment_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    fingerprint = accelerator_lock_fingerprint(installation.checkout, ACCELERATOR_EXTRAS["cuda"])
    _record_accelerator(tmp_path, monkeypatch, installation.checkout, fingerprint)
    _quiet_finalize(monkeypatch)

    def refuse(*args: Any, **kwargs: Any) -> AcceleratorPlan:
        raise AssertionError("the accelerator environment must not be rebuilt")

    monkeypatch.setattr(update, "configure_accelerator", refuse)

    assert (
        update.update_main(
            install_dir=str(installation.checkout),
            check=False,
            skip_accelerator=False,
            finalize=True,
            previous_sha=None,
        )
        == 0
    )

    assert "[accelerator] skipped: the cuda environment is unchanged" in capsys.readouterr().out


@posix_only
def test_finalize_warns_that_a_deferred_accelerator_keeps_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    _record_accelerator(tmp_path, monkeypatch, installation.checkout, "stale")
    _quiet_finalize(monkeypatch)

    def refuse(*args: Any, **kwargs: Any) -> AcceleratorPlan:
        raise AssertionError("--skip-accelerator must not rebuild anything")

    monkeypatch.setattr(update, "configure_accelerator", refuse)

    assert (
        update.update_main(
            install_dir=str(installation.checkout),
            check=False,
            skip_accelerator=True,
            finalize=True,
            previous_sha=None,
        )
        == 0
    )

    message = capsys.readouterr().err
    assert "keeps serving" in message
    assert "code-indexing-mcp configure --accelerator cuda" in message


class _FakeBroker:
    stops: ClassVar[list[Path]] = []

    def __init__(self, paths: Any) -> None:
        self._paths = paths

    def stop(self) -> dict[str, Any]:
        _FakeBroker.stops.append(self._paths.data)
        return {"stopped": True}


@posix_only
def test_finalize_stops_a_running_daemon_after_a_real_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    monkeypatch.setattr(update, "prepared_accelerator", lambda directory: None)
    monkeypatch.setattr(update, "load_prefill", lambda: _Prefill())
    monkeypatch.setattr(update, "daemon_supported", lambda: True)
    _FakeBroker.stops = []
    states = [{"running": True}, {"running": False}]
    monkeypatch.setattr(update, "daemon_status", lambda paths: states.pop(0) if states else states)
    monkeypatch.setattr(update, "BrokerApplication", _FakeBroker)

    assert (
        update.update_main(
            install_dir=str(installation.checkout),
            check=False,
            skip_accelerator=False,
            finalize=True,
            previous_sha="0" * 40,
        )
        == 0
    )

    assert len(_FakeBroker.stops) == 1
    assert "[daemon] ok: stopped" in capsys.readouterr().out


@posix_only
def test_finalize_leaves_the_daemon_alone_when_nothing_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    monkeypatch.setattr(update, "prepared_accelerator", lambda directory: None)
    monkeypatch.setattr(update, "load_prefill", lambda: _Prefill())
    monkeypatch.setattr(update, "daemon_supported", lambda: True)
    _FakeBroker.stops = []
    monkeypatch.setattr(update, "daemon_status", lambda paths: {"running": True})
    monkeypatch.setattr(update, "BrokerApplication", _FakeBroker)

    assert (
        update.update_main(
            install_dir=str(installation.checkout),
            check=False,
            skip_accelerator=False,
            finalize=True,
            previous_sha=_head(installation.checkout),
        )
        == 0
    )

    assert _FakeBroker.stops == []
    assert "[daemon] skipped:" in capsys.readouterr().out


@posix_only
def test_finalize_silences_the_update_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installation = _install(tmp_path, monkeypatch)
    monkeypatch.setattr(update, "prepared_accelerator", lambda directory: None)
    _quiet_finalize(monkeypatch)

    assert (
        update.update_main(
            install_dir=str(installation.checkout),
            check=False,
            skip_accelerator=False,
            finalize=True,
            previous_sha="0" * 40,
        )
        == 0
    )

    cached = json.loads(
        (tmp_path / "cache" / "update-check.json").read_text(encoding="utf-8"),
    )
    head = _head(installation.checkout)
    assert cached["local_sha"] == head
    assert cached["remote_sha"] == head


def _windows_launcher(tmp_path: Path) -> Path:
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    executable = scripts / "code-indexing-mcp.exe"
    executable.write_text("image")
    return executable


def test_the_windows_shield_renames_the_running_launcher_aside(tmp_path: Path) -> None:
    executable = _windows_launcher(tmp_path)

    shielded = update._shield_running_script(tmp_path, platform_name="win32")

    assert shielded is not None
    assert update.STALE_SUFFIX in shielded.name
    assert not executable.exists()


def test_the_windows_shield_restores_a_launcher_uv_did_not_replace(tmp_path: Path) -> None:
    executable = _windows_launcher(tmp_path)
    shielded = update._shield_running_script(tmp_path, platform_name="win32")

    update._unshield(shielded, tmp_path, platform_name="win32")

    assert executable.read_text() == "image"
    assert shielded is not None and not shielded.exists()


def test_the_windows_shield_drops_the_copy_uv_replaced(tmp_path: Path) -> None:
    executable = _windows_launcher(tmp_path)
    shielded = update._shield_running_script(tmp_path, platform_name="win32")
    executable.write_text("new image")

    update._unshield(shielded, tmp_path, platform_name="win32")

    assert executable.read_text() == "new image"
    assert shielded is not None and not shielded.exists()


def test_stale_launchers_are_discarded_on_the_next_update(tmp_path: Path) -> None:
    executable = _windows_launcher(tmp_path)
    stale = executable.with_name(f"{executable.name}{update.STALE_SUFFIX}42")
    stale.write_text("leftover")

    update._discard_stale_scripts(tmp_path, platform_name="win32")
    update._discard_stale_scripts(tmp_path / "missing", platform_name="win32")

    assert not stale.exists()
    assert executable.exists()


def test_the_shield_does_nothing_off_windows(tmp_path: Path) -> None:
    _windows_launcher(tmp_path)

    assert update._shield_running_script(tmp_path, platform_name="linux") is None


@posix_only
def test_check_reports_an_up_to_date_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)

    code = update.update_main(
        install_dir=str(installation.checkout),
        check=True,
        skip_accelerator=False,
        finalize=False,
        previous_sha=None,
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["update_available"] is False
    assert payload["local_sha"] == payload["remote_sha"] == _head(installation.checkout)
    assert (tmp_path / "cache" / "update-check.json").is_file()


@posix_only
def test_check_reports_an_available_update_with_its_own_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    _publish(installation, "two\n")

    code = update.update_main(
        install_dir=str(installation.checkout),
        check=True,
        skip_accelerator=False,
        finalize=False,
        previous_sha=None,
    )

    assert code == update.CHECK_UPDATE_AVAILABLE_EXIT
    payload = json.loads(capsys.readouterr().out)
    assert payload["update_available"] is True
    assert payload["remote_sha"] == _head(installation.publisher)


@posix_only
def test_check_fails_when_the_remote_cannot_be_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installation = _install(tmp_path, monkeypatch)
    shutil.rmtree(installation.remote)

    code = update.update_main(
        install_dir=str(installation.checkout),
        check=True,
        skip_accelerator=False,
        finalize=False,
        previous_sha=None,
    )

    assert code == 1
    assert "could not reach origin" in capsys.readouterr().err
