import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from code_indexing_mcp import update_check
from code_indexing_mcp.update_check import (
    CACHE_FILENAME,
    CHECK_INTERVAL_SECONDS,
    DISABLE_VARIABLE,
    UpdateStatus,
)

LOCAL_SHA = "1111111aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
REMOTE_SHA = "2222222bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _checkout(
    root: Path, *, head: str = "ref: refs/heads/main\n", ref: str | None = LOCAL_SHA
) -> Path:
    git = root / ".git"
    git.mkdir(parents=True)
    (git / "HEAD").write_text(head, encoding="utf-8")
    if ref is not None:
        reference = git / "refs" / "heads" / "main"
        reference.parent.mkdir(parents=True)
        reference.write_text(f"{ref}\n", encoding="utf-8")
    return root


def _managed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make *tmp_path*/install look like the install this interpreter runs from."""
    root = _checkout(tmp_path / "install")
    monkeypatch.setenv("CODE_INDEXING_MCP_INSTALL_DIR", str(root))
    monkeypatch.setattr(sys, "prefix", str(root / ".venv"))
    monkeypatch.delenv(DISABLE_VARIABLE, raising=False)
    return root


def _runner(
    *, sha: str = REMOTE_SHA, error: BaseException | None = None, calls: list[object] | None = None
) -> update_check._Runner:
    def run(command: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append((command, cwd, timeout))
        if error is not None:
            raise error
        return subprocess.CompletedProcess(command, 0, f"{sha}\trefs/heads/main\n", "")

    return run


# --- install context ---


def test_install_context_returns_the_directory_of_a_managed_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _managed(tmp_path, monkeypatch)

    assert update_check.install_context() == root.resolve()


def test_install_context_is_none_without_a_git_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "install"
    root.mkdir()
    monkeypatch.setattr(sys, "prefix", str(root / ".venv"))

    assert (
        update_check.install_context(environment={"CODE_INDEXING_MCP_INSTALL_DIR": str(root)})
        is None
    )


def test_install_context_is_none_when_the_interpreter_lives_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _checkout(tmp_path / "install")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "elsewhere" / ".venv"))

    assert (
        update_check.install_context(environment={"CODE_INDEXING_MCP_INSTALL_DIR": str(root)})
        is None
    )


# --- head reading ---


def test_checkout_head_reads_a_detached_head(tmp_path: Path) -> None:
    root = _checkout(tmp_path / "repo", head=f"{LOCAL_SHA}\n", ref=None)

    assert update_check.checkout_head(root) == LOCAL_SHA


def test_checkout_head_follows_a_symbolic_ref(tmp_path: Path) -> None:
    root = _checkout(tmp_path / "repo")

    assert update_check.checkout_head(root) == LOCAL_SHA


def test_checkout_head_falls_back_to_packed_refs(tmp_path: Path) -> None:
    root = _checkout(tmp_path / "repo", ref=None)
    (root / ".git" / "packed-refs").write_text(
        f"# pack-refs with: peeled fully-peeled sorted\n{LOCAL_SHA} refs/heads/main\n",
        encoding="utf-8",
    )

    assert update_check.checkout_head(root) == LOCAL_SHA


def test_checkout_head_is_none_without_a_checkout(tmp_path: Path) -> None:
    assert update_check.checkout_head(tmp_path) is None


# --- cache ---


def test_write_cache_round_trips_through_read_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache" / "nested"
    status = UpdateStatus(checked_at=1000.0, local_sha=LOCAL_SHA, remote_sha=REMOTE_SHA)

    update_check.write_cache(cache, status)

    assert update_check.read_cache(cache) == status
    assert [path.name for path in cache.iterdir()] == [CACHE_FILENAME]


@pytest.mark.parametrize(
    "payload",
    ["not json at all", json.dumps({"schema_version": 99, "checked_at": 1.0}), json.dumps([])],
)
def test_read_cache_treats_an_unusable_file_as_absent(tmp_path: Path, payload: str) -> None:
    (tmp_path / CACHE_FILENAME).write_text(payload, encoding="utf-8")

    assert update_check.read_cache(tmp_path) is None


def test_read_cache_is_none_when_nothing_was_written(tmp_path: Path) -> None:
    assert update_check.read_cache(tmp_path) is None


# --- remote check ---


def test_check_remote_asks_origin_for_the_main_branch(tmp_path: Path) -> None:
    root = _checkout(tmp_path / "repo")
    calls: list[object] = []

    status = update_check.check_remote(root, timeout=3.0, run_command=_runner(calls=calls))

    assert calls == [(["git", "ls-remote", "origin", "refs/heads/main"], root, 3.0)]
    assert status.local_sha == LOCAL_SHA
    assert status.remote_sha == REMOTE_SHA
    assert status.update_available


# --- refresh ---


def test_refresh_if_due_writes_a_status_when_no_cache_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DISABLE_VARIABLE, raising=False)
    root = _checkout(tmp_path / "repo")
    cache = tmp_path / "cache"

    update_check.refresh_if_due(root, cache, now=1000.0, run_command=_runner())

    cached = update_check.read_cache(cache)
    assert cached is not None and cached.remote_sha == REMOTE_SHA


def test_refresh_if_due_honours_the_throttle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DISABLE_VARIABLE, raising=False)
    root = _checkout(tmp_path / "repo")
    cache = tmp_path / "cache"
    update_check.write_cache(cache, UpdateStatus(checked_at=990.0, local_sha="a", remote_sha="a"))
    calls: list[object] = []

    update_check.refresh_if_due(root, cache, now=1000.0, run_command=_runner(calls=calls))

    assert calls == []
    cached = update_check.read_cache(cache)
    assert cached is not None and cached.checked_at == 990.0


def test_refresh_if_due_rechecks_once_the_interval_expired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DISABLE_VARIABLE, raising=False)
    root = _checkout(tmp_path / "repo")
    cache = tmp_path / "cache"
    update_check.write_cache(cache, UpdateStatus(checked_at=1000.0, local_sha="a", remote_sha="a"))

    update_check.refresh_if_due(
        root, cache, now=1000.0 + CHECK_INTERVAL_SECONDS + 1, run_command=_runner()
    )

    cached = update_check.read_cache(cache)
    assert cached is not None and cached.remote_sha == REMOTE_SHA


@pytest.mark.parametrize("value", ["off", "OFF", "0", "false", "No", " off "])
def test_refresh_if_due_is_disabled_by_the_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(DISABLE_VARIABLE, value)
    root = _checkout(tmp_path / "repo")
    cache = tmp_path / "cache"
    calls: list[object] = []

    update_check.refresh_if_due(root, cache, now=1000.0, run_command=_runner(calls=calls))

    assert calls == []
    assert not cache.exists()


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired(["git"], 5.0),
        FileNotFoundError("git"),
        subprocess.CalledProcessError(128, ["git"]),
    ],
)
def test_refresh_if_due_stays_silent_when_git_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    monkeypatch.delenv(DISABLE_VARIABLE, raising=False)
    root = _checkout(tmp_path / "repo")
    cache = tmp_path / "cache"

    update_check.refresh_if_due(root, cache, now=1000.0, run_command=_runner(error=error))

    assert update_check.read_cache(cache) is None


def test_refresh_if_due_overwrites_a_corrupt_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DISABLE_VARIABLE, raising=False)
    root = _checkout(tmp_path / "repo")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / CACHE_FILENAME).write_text("{not json", encoding="utf-8")

    update_check.refresh_if_due(root, cache, now=1000.0, run_command=_runner())

    cached = update_check.read_cache(cache)
    assert cached is not None and cached.remote_sha == REMOTE_SHA


# --- background refresh ---


def test_start_background_refresh_returns_none_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _managed(tmp_path, monkeypatch)
    monkeypatch.setenv(DISABLE_VARIABLE, "off")

    assert update_check.start_background_refresh(tmp_path / "cache") is None


def test_start_background_refresh_returns_none_without_a_managed_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DISABLE_VARIABLE, raising=False)
    monkeypatch.setenv("CODE_INDEXING_MCP_INSTALL_DIR", str(tmp_path / "missing"))

    assert update_check.start_background_refresh(tmp_path / "cache") is None


def test_start_background_refresh_returns_none_when_the_cache_is_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _managed(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    update_check.write_cache(
        cache, UpdateStatus(checked_at=time.time(), local_sha=LOCAL_SHA, remote_sha=LOCAL_SHA)
    )

    assert update_check.start_background_refresh(cache) is None


def test_start_background_refresh_starts_a_daemon_thread_when_due(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _managed(tmp_path, monkeypatch)
    cache = tmp_path / "cache"

    thread = update_check.start_background_refresh(cache)

    assert thread is not None
    assert thread.daemon
    thread.join(timeout=10.0)
    # The fake checkout has no "origin", so the check fails -- silently.
    assert not thread.is_alive()


# --- notice ---


def test_notice_is_silent_when_the_live_head_matches_the_cached_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _managed(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    update_check.write_cache(
        cache, UpdateStatus(checked_at=1000.0, local_sha="0" * 40, remote_sha=LOCAL_SHA)
    )

    assert update_check.notice(cache) is None


def test_notice_reports_a_newer_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _managed(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    update_check.write_cache(
        cache, UpdateStatus(checked_at=1000.0, local_sha=LOCAL_SHA, remote_sha=REMOTE_SHA)
    )

    message = update_check.notice(cache)

    assert message == (
        f"A code-indexing-mcp update is available ({LOCAL_SHA[:7]} -> {REMOTE_SHA[:7]}). "
        "Run: code-indexing-mcp update"
    )


def test_notice_is_silent_without_a_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _managed(tmp_path, monkeypatch)

    assert update_check.notice(tmp_path / "cache") is None


def test_notice_is_silent_without_a_managed_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODE_INDEXING_MCP_INSTALL_DIR", str(tmp_path / "missing"))
    cache = tmp_path / "cache"
    update_check.write_cache(
        cache, UpdateStatus(checked_at=1000.0, local_sha=LOCAL_SHA, remote_sha=REMOTE_SHA)
    )

    assert update_check.notice(cache) is None
