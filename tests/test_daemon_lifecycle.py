"""Daemon lifecycle regressions (review track I, DMN-2 .. DMN-12).

Every test here uses an isolated data directory and a per-test endpoint, waits on
events or polls with a deadline rather than sleeping, and shuts its in-process
daemon down in a ``finally`` so a failure cannot leak a thread holding the
lifetime lock into the parallel suite.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from filelock import FileLock
from support import DeterministicEmbedder

from code_indexing_mcp import daemon
from code_indexing_mcp.application import Application, RuntimePaths
from code_indexing_mcp.daemon import (
    BrokerApplication,
    DaemonServer,
    daemon_endpoint,
    daemon_supported,
    receive_frame,
    send_frame,
)
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode

requires_local_sockets = pytest.mark.skipif(
    not daemon_supported(),
    reason="the shared daemon requires Unix domain sockets",
)


def _paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")


def _application(paths: RuntimePaths, tmp_path: Path) -> Application:
    return Application(paths, embedder=DeterministicEmbedder(), cwd=tmp_path)


def _wait_for(condition: Callable[[], bool], *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@contextlib.contextmanager
def _serving(
    paths: RuntimePaths, application: Application, *, idle_timeout_seconds: float = 60
) -> Iterator[tuple[DaemonServer, threading.Thread]]:
    server = DaemonServer(paths, application=application, idle_timeout_seconds=idle_timeout_seconds)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:
        assert server.ready.wait(timeout=5)
        yield server, thread
    finally:
        server.request_stop()
        thread.join(timeout=10)
        assert not thread.is_alive()


def _lock_path(paths: RuntimePaths) -> Path:
    return paths.data / "locks" / "daemon.lock"


@contextlib.contextmanager
def _held_lifetime_lock(paths: RuntimePaths) -> Iterator[FileLock]:
    _lock_path(paths).parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(_lock_path(paths), thread_local=False)
    lock.acquire(timeout=0)
    try:
        yield lock
    finally:
        if lock.is_locked:
            lock.release()


class _FakeProcess:
    def __init__(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code
        self.returncode = exit_code
        self.waited = threading.Event()

    def poll(self) -> int | None:
        return self.exit_code

    def wait(self, timeout: float | None = None) -> int:
        self.waited.set()
        return self.exit_code or 0


def _write_token(paths: RuntimePaths, token: str = "shared-token") -> None:
    paths.data.mkdir(parents=True, exist_ok=True)
    (paths.data / "daemon.token").write_text(token)


# --- DMN-3 ---------------------------------------------------------------


def test_stopped_means_the_lifetime_lock_is_free_not_that_the_socket_refuses(
    tmp_path: Path,
) -> None:
    """DMN-3: a daemon that closed its listener but still holds the lock is not stopped."""
    paths = _paths(tmp_path)
    releaser: threading.Timer | None = None
    with _held_lifetime_lock(paths) as lock:
        assert not daemon._lifetime_lock_free(paths)
        stopped = daemon.stop_daemon_and_wait(
            paths,
            timeout_seconds=0.3,
            poll_interval=0.01,
            status=lambda _paths: {"running": False},
            stop=lambda: None,
            initial_running=True,
        )
        assert stopped is False

        releaser = threading.Timer(0.1, lock.release)
        releaser.start()
        try:
            stopped = daemon.stop_daemon_and_wait(
                paths,
                timeout_seconds=5,
                poll_interval=0.01,
                status=lambda _paths: {"running": False},
                stop=lambda: None,
                initial_running=True,
            )
        finally:
            releaser.join()
    assert stopped is True
    assert daemon._lifetime_lock_free(paths)


@requires_local_sockets
def test_a_new_daemon_waits_for_the_lifetime_lock_held_by_a_stopping_one(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    application = _application(paths, tmp_path)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    with _held_lifetime_lock(paths) as lock:
        thread.start()
        try:
            # The replacement must be waiting, not failed and not serving.
            assert not server.ready.wait(timeout=0.2)
            assert thread.is_alive()
            lock.release()
            assert server.ready.wait(timeout=5)
        finally:
            server.request_stop()
            thread.join(timeout=10)
    assert not thread.is_alive()


@requires_local_sockets
def test_a_new_daemon_reports_busy_when_the_lifetime_lock_never_frees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon, "LIFETIME_LOCK_WAIT_SECONDS", 0.2)
    paths = _paths(tmp_path)
    server = DaemonServer(paths, application=_application(paths, tmp_path))
    with _held_lifetime_lock(paths), pytest.raises(CodeIndexingError) as failure:
        server.serve()
    assert failure.value.code is ErrorCode.INDEX_BUSY


@requires_local_sockets
def test_ensure_daemon_waits_for_a_draining_daemon_before_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    spawned_with_lock_free: list[bool] = []

    def fake_popen(*args: object, **kwargs: object) -> _FakeProcess:
        spawned_with_lock_free.append(daemon._lifetime_lock_free(paths))
        return _FakeProcess(exit_code=1)

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)
    with _held_lifetime_lock(paths) as lock:
        releaser = threading.Timer(0.15, lock.release)
        releaser.start()
        try:
            with pytest.raises(CodeIndexingError):
                daemon.ensure_daemon(paths, timeout_seconds=2)
        finally:
            releaser.join()
    assert spawned_with_lock_free == [True]


@requires_local_sockets
def test_ensure_daemon_gives_up_when_the_previous_daemon_never_releases_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon, "DAEMON_STOP_WAIT_SECONDS", 0.2)

    def fail_popen(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not spawn a daemon that cannot take the lock")

    monkeypatch.setattr(daemon.subprocess, "Popen", fail_popen)
    paths = _paths(tmp_path)
    with _held_lifetime_lock(paths), pytest.raises(CodeIndexingError) as failure:
        daemon.ensure_daemon(paths, timeout_seconds=2)
    assert failure.value.code is ErrorCode.DAEMON_UNAVAILABLE
    assert "shutting down" in str(failure.value)


@requires_local_sockets
def test_ensure_daemon_fails_fast_when_the_spawned_daemon_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    process = _FakeProcess(exit_code=3)
    monkeypatch.setattr(daemon.subprocess, "Popen", lambda *a, **k: process)
    started = time.monotonic()
    with pytest.raises(CodeIndexingError) as failure:
        daemon.ensure_daemon(paths, timeout_seconds=30)
    assert time.monotonic() - started < 10
    assert failure.value.code is ErrorCode.DAEMON_UNAVAILABLE
    assert "exited with status 3" in str(failure.value)


# --- DMN-4 ---------------------------------------------------------------


def _blocking_list_projects(
    application: Application, entered: threading.Event, release: threading.Event
) -> None:
    original = application.list_projects

    def blocking() -> Any:
        entered.set()
        assert release.wait(timeout=20)
        return original()

    application.list_projects = blocking  # type: ignore[method-assign]


def _call_in_thread(call: Callable[[], Any], outcome: list[Any]) -> threading.Thread:
    def run() -> None:
        try:
            outcome.append(call())
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _listener_closed(paths: RuntimePaths) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(1)
            probe.connect(str(daemon_endpoint(paths)))
    except (ConnectionRefusedError, FileNotFoundError):
        return True
    except OSError:
        return False
    return False


@requires_local_sockets
def test_stop_drains_an_in_flight_request_before_the_daemon_exits(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    application = _application(paths, tmp_path)
    entered, release = threading.Event(), threading.Event()
    _blocking_list_projects(application, entered, release)
    outcome: list[Any] = []
    with _serving(paths, application) as (server, thread):
        try:
            client = _call_in_thread(BrokerApplication(paths, cwd=tmp_path).list_projects, outcome)
            assert entered.wait(timeout=5)
            assert BrokerApplication(paths, cwd=tmp_path).stop() == {"stopping": True}
            # The listener closes, but the process (and its lock) stay until
            # the in-flight request has been answered.
            assert _wait_for(lambda: _listener_closed(paths))
            assert thread.is_alive()
            assert not daemon._lifetime_lock_free(paths)
        finally:
            release.set()
        client.join(timeout=10)
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert outcome == [[]]
    assert daemon._lifetime_lock_free(paths)
    del server


@requires_local_sockets
def test_the_drain_of_ordinary_requests_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon, "STOP_DRAIN_SECONDS", 0.2)
    paths = _paths(tmp_path)
    application = _application(paths, tmp_path)
    entered, release = threading.Event(), threading.Event()
    _blocking_list_projects(application, entered, release)
    outcome: list[Any] = []
    with _serving(paths, application) as (_server, thread):
        try:
            client = _call_in_thread(BrokerApplication(paths, cwd=tmp_path).list_projects, outcome)
            assert entered.wait(timeout=5)
            BrokerApplication(paths, cwd=tmp_path).stop()
            thread.join(timeout=10)
            assert not thread.is_alive()
        finally:
            release.set()
            client.join(timeout=10)


@requires_local_sockets
def test_an_in_flight_index_run_is_drained_longer_than_an_ordinary_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon, "STOP_DRAIN_SECONDS", 0.05)
    monkeypatch.setattr(daemon, "STOP_INDEX_DRAIN_SECONDS", 30)
    paths = _paths(tmp_path)
    application = _application(paths, tmp_path)
    entered, release = threading.Event(), threading.Event()

    def blocking_maintain_storage(**_params: Any) -> dict[str, Any]:
        entered.set()
        assert release.wait(timeout=20)
        return {"sentinel": True}

    application.maintain_storage = blocking_maintain_storage  # type: ignore[assignment,method-assign]
    outcome: list[Any] = []
    with _serving(paths, application) as (_server, thread):
        try:
            client = _call_in_thread(
                lambda: BrokerApplication(paths, cwd=tmp_path)._call("maintain_storage"),
                outcome,
            )
            assert entered.wait(timeout=5)
            BrokerApplication(paths, cwd=tmp_path).stop()
            assert _wait_for(lambda: _listener_closed(paths))
            # Several multiples of the ordinary bound have passed.
            time.sleep(0.3)
            assert thread.is_alive()
        finally:
            release.set()
        client.join(timeout=10)
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert outcome == [{"sentinel": True}]


@requires_local_sockets
def test_the_index_drain_is_bounded_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daemon, "STOP_DRAIN_SECONDS", 0.05)
    monkeypatch.setattr(daemon, "STOP_INDEX_DRAIN_SECONDS", 0.2)
    paths = _paths(tmp_path)
    application = _application(paths, tmp_path)
    entered, release = threading.Event(), threading.Event()

    def blocking_maintain_storage(**_params: Any) -> dict[str, Any]:
        entered.set()
        assert release.wait(timeout=20)
        return {}

    application.maintain_storage = blocking_maintain_storage  # type: ignore[assignment,method-assign]
    outcome: list[Any] = []
    with _serving(paths, application) as (_server, thread):
        try:
            client = _call_in_thread(
                lambda: BrokerApplication(paths, cwd=tmp_path)._call("maintain_storage"),
                outcome,
            )
            assert entered.wait(timeout=5)
            BrokerApplication(paths, cwd=tmp_path).stop()
            thread.join(timeout=10)
            assert not thread.is_alive()
        finally:
            release.set()
            client.join(timeout=10)


# --- DMN-7 ---------------------------------------------------------------


@requires_local_sockets
def test_ping_and_stop_give_up_quickly_on_a_wedged_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon, "DAEMON_CONTROL_TIMEOUT_SECONDS", 0.2)
    paths = _paths(tmp_path)
    _write_token(paths)
    endpoint = daemon_endpoint(paths)
    finished = threading.Event()
    accepted: list[socket.socket] = []

    def wedged() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            endpoint.unlink(missing_ok=True)
            server.bind(str(endpoint))
            server.listen()
            server.settimeout(0.05)
            while not finished.is_set():
                try:
                    accepted.append(server.accept()[0])
                except TimeoutError:
                    continue
        for connection in accepted:
            connection.close()
        endpoint.unlink(missing_ok=True)

    thread = threading.Thread(target=wedged, daemon=True)
    thread.start()
    try:
        assert _wait_for(endpoint.exists)
        started = time.monotonic()
        assert daemon.daemon_status(paths) == {"running": False}
        with pytest.raises(CodeIndexingError) as stop_failure:
            BrokerApplication(paths).stop()
        assert stop_failure.value.code is ErrorCode.DAEMON_UNAVAILABLE
        assert time.monotonic() - started < 5
    finally:
        finished.set()
        thread.join(timeout=5)


def test_ordinary_calls_keep_the_long_budget_and_control_calls_get_the_short_one() -> None:
    assert daemon._client_budget("ping") == daemon.DAEMON_CONTROL_TIMEOUT_SECONDS
    assert daemon._client_budget("stop") == daemon.DAEMON_CONTROL_TIMEOUT_SECONDS
    assert daemon._client_budget("list_projects") == daemon.DAEMON_QUERY_TIMEOUT_SECONDS
    assert daemon._client_budget("index_project") is None


# --- DMN-8 ---------------------------------------------------------------


@requires_local_sockets
def test_an_accept_error_does_not_kill_the_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon, "ACCEPT_ERROR_BACKOFF_SECONDS", 0.01)
    failures = {"remaining": 3}

    class FlakyListener(socket.socket):
        def accept(self) -> Any:  # type: ignore[override]
            if failures["remaining"] > 0:
                failures["remaining"] -= 1
                raise OSError(errno.EMFILE, "Too many open files")
            return super().accept()

    def flaky_socket() -> socket.socket:
        return FlakyListener(socket.AF_UNIX, socket.SOCK_STREAM)

    monkeypatch.setattr(daemon, "_local_socket", flaky_socket)
    paths = _paths(tmp_path)
    with _serving(paths, _application(paths, tmp_path)) as (_server, thread):
        assert _wait_for(lambda: failures["remaining"] == 0)
        assert BrokerApplication(paths, cwd=tmp_path).ping()["pid"] > 0
        assert thread.is_alive()


# --- DMN-10 / DMN-11 -----------------------------------------------------


@requires_local_sockets
def test_a_relayed_error_does_not_repeat_its_code_prefix(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    application = _application(paths, tmp_path)

    def busy() -> Any:
        raise CodeIndexingError(ErrorCode.INDEX_BUSY, "The thing is busy")

    application.list_projects = busy  # type: ignore[method-assign]
    with _serving(paths, application), pytest.raises(CodeIndexingError) as failure:
        BrokerApplication(paths, cwd=tmp_path).list_projects()
    assert str(failure.value) == "INDEX_BUSY: The thing is busy"


@pytest.mark.parametrize(
    "second_failure",
    [EOFError("closed"), ConnectionRefusedError(61, "refused"), FileNotFoundError(2, "gone")],
)
def test_the_reconnect_retry_stays_inside_the_error_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second_failure: BaseException
) -> None:
    paths = _paths(tmp_path)
    broker = BrokerApplication(paths, cwd=tmp_path)
    calls: list[str] = []

    def call_once(method: str, **_params: Any) -> Any:
        calls.append(method)
        raise FileNotFoundError(2, "gone") if len(calls) == 1 else second_failure

    monkeypatch.setattr(broker, "_call_once", call_once)
    monkeypatch.setattr(daemon, "ensure_daemon", lambda _paths: broker)
    with pytest.raises(CodeIndexingError) as failure:
        broker.list_projects()
    assert failure.value.code is ErrorCode.DAEMON_UNAVAILABLE
    assert len(calls) == 2


# --- DMN-2 ---------------------------------------------------------------


@requires_local_sockets
def test_an_index_request_waits_for_the_startup_model_warmup(tmp_path: Path) -> None:
    """DMN-2: the memory baseline must not be sampled while the warm-up load is growing RSS."""
    paths = _paths(tmp_path)
    application = _application(paths, tmp_path)
    warmup_entered, warmup_release = threading.Event(), threading.Event()
    seen: list[bool] = []

    def slow_prepare() -> None:
        warmup_entered.set()
        assert warmup_release.wait(timeout=20)

    def recording_index(**_params: Any) -> str:
        seen.append(server._model_warmup_active)
        return "indexed"

    application.prepare_model = slow_prepare  # type: ignore[method-assign]
    application.index_project = recording_index  # type: ignore[assignment,method-assign]
    outcome: list[Any] = []
    with _serving(paths, application) as (server, _thread):
        try:
            assert warmup_entered.wait(timeout=5)
            client = _call_in_thread(lambda: server._dispatch("index_project", {}), outcome)
            time.sleep(0.1)
            assert seen == []  # held back while warm-up is still loading
        finally:
            warmup_release.set()
        client.join(timeout=10)
    assert seen == [False]
    assert outcome == ["indexed"]


@requires_local_sockets
def test_the_warmup_wait_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daemon, "MODEL_WARMUP_WAIT_SECONDS", 0.1)
    paths = _paths(tmp_path)
    application = _application(paths, tmp_path)
    warmup_entered, warmup_release = threading.Event(), threading.Event()

    def slow_prepare() -> None:
        warmup_entered.set()
        assert warmup_release.wait(timeout=20)

    application.prepare_model = slow_prepare  # type: ignore[method-assign]
    application.index_project = lambda **_params: "indexed"  # type: ignore[assignment,method-assign]
    with _serving(paths, application) as (server, _thread):
        try:
            assert warmup_entered.wait(timeout=5)
            assert server._dispatch("index_project", {}) == "indexed"
        finally:
            warmup_release.set()


# --- DMN-12 --------------------------------------------------------------


def test_a_response_is_json_encoded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    real_dumps = json.dumps
    calls: list[int] = []

    def counting_dumps(*args: Any, **kwargs: Any) -> str:
        calls.append(1)
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(daemon.json, "dumps", counting_dumps)
    left, right = socket.socketpair()
    with left, right:
        daemon._send_response(left, {"id": "x", "result": {"value": [1, 2, 3]}})
        assert receive_frame(right) == {"id": "x", "result": {"value": [1, 2, 3]}}
    assert len(calls) == 1


@requires_local_sockets
def test_the_spawned_daemon_is_reaped_by_a_waiter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    application = _application(paths, tmp_path)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    process = _FakeProcess()
    thread = threading.Thread(target=server.serve, daemon=True)

    def fake_popen(*args: object, **kwargs: object) -> _FakeProcess:
        thread.start()
        return process

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)
    try:
        daemon.ensure_daemon(paths, timeout_seconds=10)
        assert process.waited.wait(timeout=5)
    finally:
        server.request_stop()
        if thread.is_alive():
            thread.join(timeout=10)


def test_the_daemon_is_spawned_from_a_module_that_does_not_import_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DMN-5: spawn re-imports the daemon's main module in every CPU embedding worker."""
    commands: list[list[str]] = []

    def fake_popen(command: list[str], **_kwargs: object) -> _FakeProcess:
        commands.append(command)
        return _FakeProcess(exit_code=1)

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)
    with pytest.raises(CodeIndexingError):
        daemon.ensure_daemon(_paths(tmp_path), timeout_seconds=1)
    assert commands[0][1:3] == ["-m", "code_indexing_mcp.daemon_entry"]


def test_the_daemon_entry_module_does_not_import_the_cli_when_spawn_reimports_it() -> None:
    script = (
        "import runpy, sys; "
        "runpy.run_module('code_indexing_mcp.daemon_entry', run_name='__mp_main__'); "
        "sys.exit(1 if 'code_indexing_mcp.cli' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode()


@requires_local_sockets
def test_request_stop_ends_serve_and_cleans_up_the_endpoint(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    with _serving(paths, _application(paths, tmp_path)) as (server, thread):
        endpoint = server.endpoint
        assert endpoint.exists()
        server.request_stop()
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert not endpoint.exists()
    assert not (paths.data / "daemon.endpoint").exists()
    assert daemon._lifetime_lock_free(paths)


@requires_local_sockets
def test_sigterm_shuts_the_daemon_down_cleanly(tmp_path: Path) -> None:
    """A SIGTERM must run serve()'s cleanup instead of leaving a stale socket."""
    runtime = Path(os.path.realpath(tmp_path)) / "rt"
    runtime.mkdir()
    data = tmp_path / "data"
    environment = {
        **os.environ,
        "CODE_INDEXING_DATA_DIR": str(data),
        "CODE_INDEXING_CACHE_DIR": str(tmp_path / "cache"),
        "CODE_INDEXING_OFFLINE": "1",
        "CODE_INDEXING_BROKER": "off",
        "XDG_RUNTIME_DIR": str(runtime),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "code_indexing_mcp.daemon_entry"],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        endpoint_record = data / "daemon.endpoint"
        assert _wait_for(lambda: endpoint_record.exists() or process.poll() is not None, timeout=60)
        assert process.poll() is None, process.stderr.read().decode() if process.stderr else ""
        endpoint = Path(endpoint_record.read_text())
        assert endpoint.exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=60) == 0
        assert not endpoint.exists()
        assert not endpoint_record.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        if process.stderr is not None:
            process.stderr.close()


@requires_local_sockets
def test_a_client_that_disconnects_without_a_request_is_not_logged_as_a_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    paths = _paths(tmp_path)
    caplog.set_level(logging.DEBUG, logger=daemon.logger.name)
    with _serving(paths, _application(paths, tmp_path)) as (server, _thread):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(server.endpoint))
        assert _wait_for(lambda: server._active_requests == 0)
        # A genuine dispatch failure still logs its traceback.
        assert BrokerApplication(paths, cwd=tmp_path).ping()["pid"] > 0
    assert not [r for r in caplog.records if "Unhandled daemon dispatch failure" in r.getMessage()]
    assert not [r for r in caplog.records if r.exc_info]


@requires_local_sockets
def test_an_empty_token_file_is_replaced_and_never_authenticates(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_token(paths, "")
    with _serving(paths, _application(paths, tmp_path)) as (server, _thread):
        assert (paths.data / "daemon.token").read_text().strip() != ""
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(server.endpoint))
            send_frame(
                connection,
                {
                    "protocol": daemon.PROTOCOL_VERSION,
                    "id": "1",
                    "token": "",
                    "method": "list_projects",
                    "params": {},
                },
            )
            response = receive_frame(connection)
        assert response["error"]["code"] == ErrorCode.INVALID_CONFIGURATION.value
        assert BrokerApplication(paths, cwd=tmp_path).list_projects() == []
    mode = (paths.data / "daemon.token").stat().st_mode & 0o777
    assert mode == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="no RLIMIT_NOFILE")
def test_the_daemon_raises_its_descriptor_soft_limit() -> None:
    import resource

    daemon.raise_open_file_limit()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    expected = 4096 if hard == resource.RLIM_INFINITY else min(hard, 4096)
    assert soft >= expected
