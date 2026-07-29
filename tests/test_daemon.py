from __future__ import annotations

import os
import socket
import stat
import threading
import time
from pathlib import Path

import pytest

from incode_mcp import daemon, embedding
from incode_mcp.application import Application, RuntimePaths
from incode_mcp.daemon import (
    BrokerApplication,
    DaemonServer,
    daemon_endpoint,
    daemon_supported,
    receive_frame,
    send_frame,
)
from incode_mcp.embedding import FastEmbedder
from incode_mcp.errors import ErrorCode, IncodeError
from incode_mcp.models import IndexReport
from incode_mcp.settings import IndexSettings

# Gate on the capability the code actually needs rather than on the platform, so
# the guard keeps tracking reality if another platform loses (or gains) AF_UNIX.
# Framing, the unsupported-platform error, and the CLI fallback stay unguarded.
requires_local_sockets = pytest.mark.skipif(
    not daemon_supported(),
    reason="the shared daemon requires Unix domain sockets",
)


class TinyEmbedder:
    model_id = "test/tiny"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, float(len(text))]


def test_length_prefixed_json_frame_round_trip() -> None:
    left, right = socket.socketpair()
    try:
        payload = {"jsonrpc": "2.0", "id": "request-1", "method": "ping"}
        send_frame(left, payload)
        assert receive_frame(right) == payload
    finally:
        left.close()
        right.close()


@requires_local_sockets
def test_broker_application_calls_one_daemon_backend(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    daemon = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    assert daemon.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=tmp_path)
    root = tmp_path / "repo"
    root.mkdir()

    project = broker.init_project(root)

    assert broker.list_projects() == [project]
    assert broker.ping()["pid"] > 0
    broker.stop()
    thread.join(timeout=2)
    assert not thread.is_alive()


@requires_local_sockets
def test_broker_restarts_daemon_after_idle_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    monkeypatch.setenv("INCODE_DATA_DIR", str(paths.data))
    monkeypatch.setenv("INCODE_CACHE_DIR", str(paths.cache))
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    first = DaemonServer(paths, application=application, idle_timeout_seconds=0.1)
    first_thread = threading.Thread(target=first.serve, daemon=True)
    first_thread.start()
    assert first.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=tmp_path)
    assert broker.ping()["pid"] > 0
    first_thread.join(timeout=2)
    assert not first_thread.is_alive()

    assert broker.list_projects() == []
    assert daemon.daemon_status(paths)["running"] is True

    broker.stop()
    for _ in range(40):
        if not daemon.daemon_status(paths)["running"]:
            break
        time.sleep(0.05)
    assert daemon.daemon_status(paths)["running"] is False


@pytest.mark.parametrize(
    "failure",
    [
        EOFError("Local daemon connection closed"),
        ConnectionResetError(104, "Connection reset by peer"),
        ConnectionRefusedError(111, "Connection refused"),
        FileNotFoundError(2, "No such file or directory"),
        IncodeError(ErrorCode.PROTOCOL_ERROR, "Local daemon frame exceeds the maximum size"),
    ],
)
def test_daemon_status_answers_the_question_however_the_ping_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """Asking whether the daemon is up has no failure answer, only False.

    The interesting case is EOFError, which is what a daemon shutting down
    between the send and the reply produces and which is not an OSError -- the
    test above polls this in a loop precisely while a daemon is going down, so a
    gap here surfaces as an unrelated-looking flake rather than as itself.
    """
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")

    class _PingFails:
        def __init__(self, _paths: RuntimePaths, **_kwargs: object) -> None:
            pass

        def _ping_once(self) -> dict[str, object]:
            raise failure

    monkeypatch.setattr(daemon, "BrokerApplication", _PingFails)

    assert daemon.daemon_status(paths) == {"running": False}


@requires_local_sockets
def test_daemon_does_not_idle_exit_while_request_is_active(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    started = threading.Event()
    release = threading.Event()
    original = application.list_projects

    def blocking_list_projects():  # type: ignore[no-untyped-def]
        started.set()
        assert release.wait(timeout=2)
        return original()

    application.list_projects = blocking_list_projects  # type: ignore[method-assign]
    daemon = DaemonServer(paths, application=application, idle_timeout_seconds=0.1)
    daemon_thread = threading.Thread(target=daemon.serve, daemon=True)
    daemon_thread.start()
    assert daemon.ready.wait(timeout=2)
    broker = BrokerApplication(paths)
    request_thread = threading.Thread(target=broker.list_projects)
    request_thread.start()
    assert started.wait(timeout=2)

    time.sleep(0.7)

    assert daemon_thread.is_alive()
    release.set()
    request_thread.join(timeout=2)
    daemon_thread.join(timeout=2)


class _CountedVector:
    """A stand-in for a FastEmbed vector, at the real model's dimension."""

    def __init__(self, text: str) -> None:
        self._text = text

    def tolist(self) -> list[float]:
        vector = [0.0] * FastEmbedder.dimension
        vector[0] = 1.0
        vector[-1] = float(len(self._text))
        return vector


@requires_local_sockets
def test_concurrent_clients_share_one_model_and_one_indexing_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The PR's headline claim, executed: N clients, one daemon, one model, one job.

    Every client currently indexes on its own; losers fail fast on the global
    lock rather than attaching to the winner's job. Genuine coalescing - all
    callers receiving the winner's report - is the daemon scheduler in
    docs/plans/2026-07-24-indexing-memory-hardening-completion.md Task 6. These
    assertions describe today's behaviour so that change is a deliberate one.
    """
    constructions: list[float] = []
    construction_lock = threading.Lock()
    embed_calls: list[list[str]] = []

    class CountedTextEmbedding:
        def __init__(self, **_: object) -> None:
            with construction_lock:
                constructions.append(time.monotonic())
            # Widen the window a real ONNX session load leaves open.
            time.sleep(0.05)

        def passage_embed(self, texts: list[str], **_: object) -> list[_CountedVector]:
            with construction_lock:
                embed_calls.append(list(texts))
            return [_CountedVector(text) for text in texts]

        def query_embed(self, text: str, **_: object) -> list[_CountedVector]:
            return [_CountedVector(text)]

    monkeypatch.setattr(embedding, "TextEmbedding", CountedTextEmbedding)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'repo'\n")
    (root / "main.py").write_text("def answer():\n    return 42\n")
    # in-process execution keeps embedding on the daemon's own FastEmbedder,
    # which is where the shared model lives; the worker path would spawn a real
    # process and load the real model.
    settings = IndexSettings.from_environment({"INCODE_INDEX_EXECUTION": "in-process"})
    application = Application(
        paths,
        embedder=FastEmbedder(paths.cache / "models", offline=True),
        cwd=root,
        settings=settings,
    )
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=5)

    clients = 8
    barrier = threading.Barrier(clients)
    outcomes: list[IndexReport | IncodeError] = []
    outcomes_lock = threading.Lock()

    def drive() -> None:
        broker = BrokerApplication(paths, cwd=root)
        try:
            barrier.wait(timeout=10)
            project = broker.init_project(root)
            outcome: IndexReport | IncodeError = broker.index_project(project.id)
        except IncodeError as exc:
            outcome = exc
        with outcomes_lock:
            outcomes.append(outcome)

    workers = [threading.Thread(target=drive) for _ in range(clients)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)

    assert not any(worker.is_alive() for worker in workers)
    assert len(outcomes) == clients
    assert len(constructions) == 1
    assert len(application.list_projects()) == 1
    project_id = application.list_projects()[0].id

    reports = [outcome for outcome in outcomes if isinstance(outcome, IndexReport)]
    errors = [outcome for outcome in outcomes if isinstance(outcome, IncodeError)]
    assert reports
    assert all(error.code is ErrorCode.INDEX_BUSY for error in errors)
    assert all(report.project_id == project_id for report in reports)
    assert all(report.errors == [] for report in reports)
    # Exactly one client did the work; any client that acquired the lock after it
    # saw an already-complete index rather than re-indexing or duplicating it.
    indexing = [report for report in reports if report.indexed_files]
    assert len(indexing) == 1
    assert indexing[0].indexed_files == 1
    assert indexing[0].embedded_chunks > 0
    assert all(report.unchanged_files == 1 for report in reports if report is not indexing[0])
    assert sum(len(call) for call in embed_calls) == indexing[0].embedded_chunks
    assert application.store.count_chunks([project_id]) == indexing[0].embedded_chunks

    with server._activity_lock:
        assert server._active_requests == 0
    BrokerApplication(paths, cwd=root).stop()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not server.endpoint.exists()


def test_require_daemon_support_explains_unsupported_platforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daemon, "daemon_supported", lambda: False)

    with pytest.raises(IncodeError) as caught:
        daemon.require_daemon_support()

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION
    assert "INCODE_BROKER=off" in str(caught.value)


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership semantics")
def test_endpoint_refuses_a_symlinked_runtime_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shared temporary root lets another user pre-plant the endpoint path."""
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / f"incode-{os.getuid()}").symlink_to(attacker)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")

    with pytest.raises(IncodeError) as caught:
        daemon_endpoint(paths)

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION
    assert not list(attacker.iterdir())


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_endpoint_directory_and_token_are_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    paths.data.mkdir(parents=True)

    endpoint = daemon_endpoint(paths)
    server = DaemonServer(paths, application=Application(paths, embedder=TinyEmbedder()))
    token = server._load_or_create_token()

    assert stat.S_IMODE(endpoint.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(server.token_path.stat().st_mode) == 0o600
    assert token and server._load_or_create_token() == token


@requires_local_sockets
def test_the_daemon_reports_the_backend_it_would_index_with(tmp_path: Path) -> None:
    """The daemon answers, because the daemon is what runs indexing."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=tmp_path)

    try:
        status = broker.model_status()
    finally:
        broker.stop()
        thread.join(timeout=2)

    assert status.embedding_model == "test/tiny"
    assert status.dimension == 4
    assert status.requested_accelerator == "auto"
    assert "CPUExecutionProvider" in status.available_providers
