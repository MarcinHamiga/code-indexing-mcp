from __future__ import annotations

import inspect
import json
import os
import socket
import stat
import threading
import time
from pathlib import Path
from typing import cast

import pytest

from code_indexing_mcp import daemon, embedding
from code_indexing_mcp.application import Application, ApplicationLike, RuntimePaths
from code_indexing_mcp.daemon import (
    BrokerApplication,
    DaemonServer,
    daemon_endpoint,
    daemon_supported,
    receive_frame,
    send_frame,
)
from code_indexing_mcp.embedding import FastEmbedder
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.models import (
    DeclarationSelector,
    ExampleSearchResponse,
    IndexReport,
    RenameOperation,
)
from code_indexing_mcp.settings import IndexSettings

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


def test_jsonable_encodes_sets_as_sorted_lists() -> None:
    """`kinds` on `find_references` is a `set[str]`; the wire only carries JSON.

    `_jsonable` used to have no branch for `set`/`frozenset`, so `json.dumps`
    raised `TypeError: Object of type set is not JSON serializable` for any
    kinds-filtered call. Sorting also keeps identical filter sets encoding
    identically regardless of the set's internal iteration order, which
    matters because the cursor embeds this same value.
    """
    encoded = daemon._jsonable({"kinds": {"call", "import"}, "limit": 100})
    assert encoded == {"kinds": ["call", "import"], "limit": 100}
    assert json.dumps(encoded)  # does not raise


def test_length_prefixed_json_frame_round_trip() -> None:
    left, right = socket.socketpair()
    try:
        payload = {"jsonrpc": "2.0", "id": "request-1", "method": "ping"}
        send_frame(left, payload)
        assert receive_frame(right) == payload
    finally:
        left.close()
        right.close()


def test_protocol_five_introduces_dead_code_report() -> None:
    assert daemon.PROTOCOL_VERSION == 5


def test_chunked_response_round_trip_is_transparent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daemon, "MAX_FRAME_BYTES", 1_024)
    monkeypatch.setattr(daemon, "MAX_RESPONSE_CHUNK_BYTES", 128)
    monkeypatch.setattr(daemon, "MAX_RESPONSE_BYTES", 8_192)
    payload = {"id": "request-1", "result": {"patch": "x" * 4_096}}
    left, right = socket.socketpair()
    sender = threading.Thread(target=daemon._send_response, args=(left, payload))
    try:
        sender.start()
        assert daemon._receive_response(right) == payload
        sender.join(timeout=2)
        assert not sender.is_alive()
    finally:
        left.close()
        right.close()


def test_response_total_size_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daemon, "MAX_RESPONSE_BYTES", 64)
    left, right = socket.socketpair()
    try:
        with pytest.raises(CodeIndexingError, match="maximum message size") as raised:
            daemon._send_response(left, {"id": "request-1", "result": "x" * 100})
    finally:
        left.close()
        right.close()

    assert raised.value.code is ErrorCode.PROTOCOL_ERROR


@requires_local_sockets
def test_broker_reassembles_a_chunked_daemon_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon, "MAX_FRAME_BYTES", 1_024)
    monkeypatch.setattr(daemon, "MAX_RESPONSE_CHUNK_BYTES", 128)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    root = tmp_path / "repo"
    root.mkdir()
    application = Application(paths, embedder=TinyEmbedder(), cwd=root)
    application.init_project(root, name="project-" + "x" * 4_096)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=root)

    try:
        projects = broker.list_projects()
    finally:
        broker.stop()
        thread.join(timeout=2)

    assert projects[0].name == "project-" + "x" * 4_096


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
def test_stopping_the_daemon_flushes_buffered_slot_touches(tmp_path: Path) -> None:
    """touch_slot is buffered in memory (D6); a stopped daemon must not lose it."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    daemon_server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=daemon_server.serve, daemon=True)
    thread.start()
    assert daemon_server.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=tmp_path)
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")

    project = broker.init_project(root)
    broker.index_project(project.id)
    broker.project_status(project.id)

    # index_project's own activation already flushed once; a later status
    # check's touch_slot still buffers, so there is something left to lose.
    assert application.store._pending_slot_touches
    version = application.store._project_slots.version

    broker.stop()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert application.store._pending_slot_touches == {}
    assert application.store._project_slots.version > version


@requires_local_sockets
def test_broker_forwards_allow_overlap_to_the_daemon(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    daemon = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    assert daemon.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=tmp_path)
    root = tmp_path / "repo"
    nested = root / "src"
    nested.mkdir(parents=True)

    try:
        parent = broker.init_project(root)
        with pytest.raises(CodeIndexingError) as raised:
            broker.init_project(nested)
        assert raised.value.code is ErrorCode.OVERLAPPING_PROJECT
        child = broker.init_project(nested, allow_overlap=True)
        assert {project.id for project in broker.list_projects()} == {parent.id, child.id}
    finally:
        broker.stop()
        thread.join(timeout=2)
        assert not thread.is_alive()


@requires_local_sockets
def test_broker_forwards_refactor_pagination_parameters(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text(
        "def answer():\n    return 42\n\ncallback = answer\n\ndef caller():\n    return answer()\n"
    )
    application = Application(paths, embedder=TinyEmbedder(), cwd=root)
    project = application.init_project(root)
    application.index_project(project.id)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=root)

    try:
        analysis = broker.analyze_refactor(
            DeclarationSelector(
                project=project.id,
                path="main.py",
                qualified_symbol="answer",
            ),
            RenameOperation(new_name="result"),
            limit=1,
        )
    finally:
        broker.stop()
        thread.join(timeout=2)

    # `cursor` alone carries the pagination signal; `completeness.state` is
    # computed from the full, unsliced result set and stays "complete" here
    # since nothing in it is a coverage gap or an unproven candidate (R4).
    assert analysis.cursor is not None
    assert analysis.completeness.state == "complete"


@requires_local_sockets
def test_search_by_example_round_trip(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    application = Application(paths, embedder=TinyEmbedder(), cwd=root)
    project = application.init_project(root)
    application.index_project(project.id)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=root)

    try:
        result = broker.search_by_example(
            "def answer():\n    return 42\n",
            projects=[project.id],
            language="python",
        )
        assert isinstance(result, ExampleSearchResponse)
        assert result.language == "python"
        assert result.segments >= 1
        assert len(result.hits) >= 1
        assert result.hits[0].path == "main.py"
        assert isinstance(result.hits[0].score, float)

        with pytest.raises(CodeIndexingError) as err_lang:
            broker.search_by_example(
                "fn main() {}",
                projects=[project.id],
                language="fortran",
            )
        assert err_lang.value.code is ErrorCode.UNSUPPORTED_LANGUAGE

        with pytest.raises(CodeIndexingError) as err_empty:
            broker.search_by_example(
                "   \n  ",
                projects=[project.id],
            )
        assert err_empty.value.code is ErrorCode.INVALID_FILTER
    finally:
        broker.stop()
        thread.join(timeout=2)


@requires_local_sockets
def test_broker_forwards_kinds_filter_for_find_references(tmp_path: Path) -> None:
    """Regression test: a kinds-filtered `find_references` used to crash the
    default daemon/broker mode with `TypeError: Object of type set is not
    JSON serializable`, because `Application.find_references` takes `kinds`
    as a `set[str]` but `_jsonable` had no branch for `set` before it reached
    `json.dumps` in `send_frame`. Only `--direct` mode, which never crosses
    the socket, was unaffected.
    """
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text(
        "def answer():\n    return 42\n\ncallback = answer\n\ndef caller():\n    return answer()\n"
    )
    application = Application(paths, embedder=TinyEmbedder(), cwd=root)
    project = application.init_project(root)
    application.index_project(project.id)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=root)

    try:
        response = broker.find_references(
            DeclarationSelector(
                project=project.id,
                path="main.py",
                qualified_symbol="answer",
            ),
            kinds={"call"},
        )
    finally:
        broker.stop()
        thread.join(timeout=2)

    assert response.hits
    assert all(hit.kind == "call" for hit in response.hits)


@requires_local_sockets
def test_broker_round_trips_impact_radius_parameters(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def base():\n    return 1\n\ndef caller():\n    return base()\n")
    application = Application(paths, embedder=TinyEmbedder(), cwd=root)
    project = application.init_project(root)
    application.index_project(project.id)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=root)

    try:
        result = broker.impact_radius(
            DeclarationSelector(
                project=project.id,
                path="main.py",
                qualified_symbol="base",
            ),
            max_depth=1,
            kinds={"call"},
            max_nodes=10,
        )
    finally:
        broker.stop()
        thread.join(timeout=2)

    assert result.layers[0].edges[0].target.qualified_symbol == "caller"
    assert result.layers[0].edges[0].kinds == ["call"]


def test_broker_round_trips_dead_code_report(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    application = Application(paths, embedder=TinyEmbedder(), cwd=root)
    project = application.init_project(root)
    application.index_project(project.id)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=root)

    try:
        result = broker.dead_code_report(project.id)
    finally:
        broker.stop()
        thread.join(timeout=2)

    assert result.project_id == project.id
    assert result.review[0].declaration.symbol == "answer"


def test_broker_freshness_uses_the_existing_status_rpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    root = tmp_path / "repo"
    root.mkdir()
    project = application.init_project(root)
    broker = BrokerApplication(paths, cwd=tmp_path)
    calls: list[tuple[str | None, list[Path] | None]] = []

    def project_status(project_name: str | None = None, *, roots=None):  # type: ignore[no-untyped-def]
        calls.append((project_name, roots))
        return application.project_status(project.id).model_copy(update={"state": "stale"})

    monkeypatch.setattr(broker, "project_status", project_status)

    assert broker.project_is_stale(project.id, roots=[root]) is True
    assert calls == [(project.id, [root])]


@requires_local_sockets
def test_broker_restarts_daemon_after_idle_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    monkeypatch.setenv("CODE_INDEXING_DATA_DIR", str(paths.data))
    monkeypatch.setenv("CODE_INDEXING_CACHE_DIR", str(paths.cache))
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
        CodeIndexingError(ErrorCode.PROTOCOL_ERROR, "Local daemon frame exceeds the maximum size"),
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


@requires_local_sockets
def test_a_silent_connection_is_dropped_after_the_receive_timeout_and_the_daemon_still_idles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3: a client that connects and never sends a request cannot hold the daemon open."""
    monkeypatch.setattr(daemon, "REQUEST_RECEIVE_TIMEOUT_SECONDS", 0.2)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=0.5)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(server.endpoint))
        # Never send anything: the receive timeout, not a client hang-up, must
        # be what ends this connection.
        frame = receive_frame(connection)
        assert frame["error"]["code"] == ErrorCode.PROTOCOL_ERROR.value
        assert connection.recv(1) == b""

    # The now-closed connection must not have pinned the idle timer open.
    thread.join(timeout=3)
    assert not thread.is_alive()


@requires_local_sockets
def test_client_call_raises_daemon_unavailable_after_the_query_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3: a wedged dispatch fails the client fast instead of hanging forever."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    started = threading.Event()
    release = threading.Event()
    original = application.list_projects

    def blocking_list_projects():  # type: ignore[no-untyped-def]
        started.set()
        assert release.wait(timeout=5)
        return original()

    application.list_projects = blocking_list_projects  # type: ignore[method-assign]
    monkeypatch.setattr(daemon, "DAEMON_QUERY_TIMEOUT_SECONDS", 0.2)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=tmp_path)

    try:
        with pytest.raises(CodeIndexingError) as raised:
            broker.list_projects()
        assert raised.value.code is ErrorCode.DAEMON_UNAVAILABLE
        assert raised.value.details["method"] == "list_projects"
        assert raised.value.details["timeout_seconds"] == 0.2
        assert started.wait(timeout=2)
    finally:
        release.set()
        BrokerApplication(paths, cwd=tmp_path).stop()
        thread.join(timeout=2)


@requires_local_sockets
def test_the_request_past_the_concurrency_cap_gets_index_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D4: a request that finds MAX_CONCURRENT_REQUESTS already dispatching is refused, not queued.

    Two permits are held directly (standing in for two requests already
    in-flight) rather than driven with real blocked dispatch threads: the
    request/response the daemon's own startup-maintenance and idle-timer
    machinery generates in the background makes the exact thread-arrival
    order the semaphore sees non-deterministic, and that ordering is not
    what this test is about -- only that the (cap + 1)th arrival is refused.
    """
    monkeypatch.setattr(daemon, "MAX_CONCURRENT_REQUESTS", 2)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)

    assert server._request_semaphore.acquire(blocking=False)
    assert server._request_semaphore.acquire(blocking=False)

    try:
        with pytest.raises(CodeIndexingError) as raised:
            BrokerApplication(paths, cwd=tmp_path).list_projects()
        assert raised.value.code is ErrorCode.INDEX_BUSY
    finally:
        server._request_semaphore.release()
        server._request_semaphore.release()
    BrokerApplication(paths, cwd=tmp_path).stop()
    thread.join(timeout=2)


@requires_local_sockets
def test_a_dispatch_failure_reaches_the_client_as_internal_error(
    tmp_path: Path,
) -> None:
    """D5: a dispatch failure is INTERNAL_ERROR, with the type in details, not the message."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    def broken_list_projects() -> None:
        raise ValueError("boom")

    application.list_projects = broken_list_projects  # type: ignore[method-assign]
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=tmp_path)

    try:
        with pytest.raises(CodeIndexingError) as raised:
            broker.list_projects()
        assert raised.value.code is ErrorCode.INTERNAL_ERROR
        assert (
            str(raised.value)
            == "INTERNAL_ERROR: The local daemon failed while handling list_projects"
        )
        assert raised.value.details == {"type": "ValueError"}
        assert "boom" not in str(raised.value)
    finally:
        broker.stop()
        thread.join(timeout=2)


@requires_local_sockets
def test_a_daemon_killed_mid_request_reaches_the_client_as_daemon_unavailable(
    tmp_path: Path,
) -> None:
    """D5: a dropped connection surfaces as DAEMON_UNAVAILABLE, not a raw EOFError."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    paths.data.mkdir(parents=True)
    (paths.data / "daemon.token").write_text("shared-token")
    endpoint = daemon_endpoint(paths)

    def dying_daemon() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(endpoint))
            server.listen()
            connection, _ = server.accept()
            with connection:
                receive_frame(connection)
                # Killed mid-request: the connection just drops, no response.
        endpoint.unlink()

    thread = threading.Thread(target=dying_daemon, daemon=True)
    thread.start()
    deadline = time.monotonic() + 2
    while not endpoint.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    broker = BrokerApplication(paths, cwd=tmp_path)
    with pytest.raises(CodeIndexingError) as raised:
        broker.list_projects()

    assert raised.value.code is ErrorCode.DAEMON_UNAVAILABLE
    assert raised.value.details["method"] == "list_projects"
    assert "log_path" in raised.value.details
    thread.join(timeout=2)


@requires_local_sockets
def test_system_exit_inside_dispatch_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D5: SystemExit is not an Exception -- it must surface as itself, not an error response."""
    # Startup maintenance also calls list_projects internally; disabled so its
    # own SystemExit (raised in a different thread, on a different schedule)
    # cannot race the one this test is actually about.
    monkeypatch.setenv("CODE_INDEXING_AUTO_MAINTENANCE", "0")
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    def exiting_list_projects() -> None:
        raise SystemExit(1)

    application.list_projects = exiting_list_projects  # type: ignore[method-assign]
    # SystemExit escapes the request thread uncaught; capture it through the
    # thread excepthook so the test asserts on it directly instead of leaving
    # it for pytest to report as an unhandled thread exception against
    # whichever test happens to be running when the thread unwinds.
    escaped: list[BaseException] = []
    monkeypatch.setattr(
        threading, "excepthook", lambda args: escaped.append(cast(BaseException, args.exc_value))
    )
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=tmp_path)

    try:
        # The client-visible effect: the connection drops with no response at
        # all, which is what proves SystemExit was never turned into an error
        # frame -- an INTERNAL_ERROR reply would mean it had been caught and
        # swallowed as if it were a plain Exception.
        with pytest.raises(CodeIndexingError) as raised:
            broker.list_projects()
        assert raised.value.code is ErrorCode.DAEMON_UNAVAILABLE
        deadline = time.monotonic() + 2
        while not escaped and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(escaped) == 1 and isinstance(escaped[0], SystemExit)
    finally:
        broker.stop()
        thread.join(timeout=2)


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
    # process and load the real model. Startup maintenance is disabled: its
    # background pass would race the barrier clients for the global lock.
    settings = IndexSettings.from_environment(
        {
            "CODE_INDEXING_INDEX_EXECUTION": "in-process",
            "CODE_INDEXING_AUTO_MAINTENANCE": "0",
        }
    )
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
    outcomes: list[IndexReport | CodeIndexingError] = []
    outcomes_lock = threading.Lock()

    def drive() -> None:
        broker = BrokerApplication(paths, cwd=root)
        try:
            barrier.wait(timeout=10)
            project = broker.init_project(root)
            outcome: IndexReport | CodeIndexingError = broker.index_project(project.id)
        except CodeIndexingError as exc:
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
    errors = [outcome for outcome in outcomes if isinstance(outcome, CodeIndexingError)]
    assert reports
    assert all(error.code is ErrorCode.INDEX_BUSY for error in errors), sorted(
        str(error) for error in errors if error.code is not ErrorCode.INDEX_BUSY
    )
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

    with pytest.raises(CodeIndexingError) as caught:
        daemon.require_daemon_support()

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION
    assert "CODE_INDEXING_BROKER=off" in str(caught.value)


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership semantics")
def test_endpoint_refuses_a_symlinked_runtime_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shared temporary root lets another user pre-plant the endpoint path."""
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / f"code-indexing-mcp-{os.getuid()}").symlink_to(attacker)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")

    with pytest.raises(CodeIndexingError) as caught:
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


def test_the_broker_reads_the_progress_the_indexing_process_publishes(tmp_path: Path) -> None:
    """Progress crosses the process boundary as a file, not as an RPC.

    The daemon thread that would have to answer an RPC is the one busy indexing,
    so a caller watching a daemon-side run reads the snapshot it publishes into
    the shared data directory.
    """

    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = application.init_project(root)
    broker = BrokerApplication(paths, cwd=tmp_path)
    seen: list[str] = []

    application.index_project(
        project.id,
        on_progress=lambda _: seen.extend(
            snapshot.phase
            for snapshot in [broker.index_progress(project.id)]
            if snapshot is not None
        ),
    )

    assert "scanning" in seen
    assert broker.index_progress(project.id) is None


@requires_local_sockets
def test_broker_application_dispatches_storage_status(tmp_path: Path) -> None:
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

    status = broker.storage_status(project.id)

    assert status.schema_version == 2
    assert status.registry.row_count == 1
    assert [entry.project.id for entry in status.projects] == [project.id]
    assert status.projects[0].consistent is True

    installation = broker.storage_status()
    assert [entry.project.id for entry in installation.projects] == [project.id]
    broker.stop()
    thread.join(timeout=2)
    assert not thread.is_alive()


@requires_local_sockets
def test_daemon_startup_runs_overdue_maintenance(tmp_path: Path) -> None:
    """The daemon owes the 24-hour maintenance cadence even before any query."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = application.init_project(root)
    application.index_project(project.id)
    daemon = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    assert daemon.ready.wait(timeout=2)
    try:
        timestamp = paths.data / "maintenance.json"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not timestamp.exists():
            time.sleep(0.05)
        assert timestamp.exists()
        payload = json.loads(timestamp.read_text())
        assert "last_maintenance_at" in payload
        # The pass ran and the project rows survived it.
        status = application.project_status(project.id)
        assert status.state == "ready"
    finally:
        broker = BrokerApplication(paths, cwd=tmp_path)
        broker.stop()
        thread.join(timeout=2)
        assert not thread.is_alive()


@requires_local_sockets
def test_daemon_startup_maintenance_respects_the_disable_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODE_INDEXING_AUTO_MAINTENANCE", "0")
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    daemon = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    assert daemon.ready.wait(timeout=2)
    try:
        time.sleep(0.3)
        assert not (paths.data / "maintenance.json").exists()
    finally:
        broker = BrokerApplication(paths, cwd=tmp_path)
        broker.stop()
        thread.join(timeout=2)
        assert not thread.is_alive()


@requires_local_sockets
def test_daemon_idle_timeout_waits_for_startup_maintenance(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    started = threading.Event()
    release = threading.Event()

    def blocking_maintenance() -> None:
        started.set()
        assert release.wait(timeout=5)

    application.maybe_run_maintenance = blocking_maintenance  # type: ignore[method-assign]
    server = DaemonServer(paths, application=application, idle_timeout_seconds=0)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)
    try:
        assert started.wait(timeout=2)
        # The listener wakes every 0.5s, comfortably beyond this idle timeout.
        time.sleep(0.7)
        assert thread.is_alive()
    finally:
        release.set()
        thread.join(timeout=3)

    assert not thread.is_alive()
    assert not server.endpoint.exists()


@requires_local_sockets
def test_daemon_startup_warms_the_query_model_once(tmp_path: Path) -> None:
    """D7: prepare_model runs once after the daemon is ready to serve."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    calls: list[None] = []

    def counted_prepare_model() -> None:
        calls.append(None)

    application.prepare_model = counted_prepare_model  # type: ignore[method-assign]
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)

    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not calls:
            time.sleep(0.02)
        assert calls == [None]
    finally:
        BrokerApplication(paths, cwd=tmp_path).stop()
        thread.join(timeout=2)


@requires_local_sockets
def test_model_warmup_failure_is_logged_not_raised(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """D7: offline mode with no cached model (or any other warm-up failure) must not crash it."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    def failing_prepare_model() -> None:
        raise CodeIndexingError(ErrorCode.MODEL_UNAVAILABLE, "no cached model in offline mode")

    application.prepare_model = failing_prepare_model  # type: ignore[method-assign]
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    with caplog.at_level("WARNING", logger="code_indexing_mcp.daemon"):
        thread.start()
        assert server.ready.wait(timeout=2)
        broker = BrokerApplication(paths, cwd=tmp_path)
        try:
            # The daemon must still be able to serve a request after a failed
            # warm-up -- the whole point of "logged, not raised".
            assert broker.list_projects() == []
        finally:
            broker.stop()
            thread.join(timeout=2)

    assert not thread.is_alive()
    messages = [record.getMessage() for record in caplog.records]
    assert any("Model warm-up after daemon startup failed" in message for message in messages)


@requires_local_sockets
def test_daemon_idle_timeout_waits_for_model_warmup(tmp_path: Path) -> None:
    """D7: a slow warm-up must not let the idle timer reap the daemon mid-load."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    started = threading.Event()
    release = threading.Event()

    def blocking_prepare_model() -> None:
        started.set()
        assert release.wait(timeout=5)

    application.prepare_model = blocking_prepare_model  # type: ignore[method-assign]
    server = DaemonServer(paths, application=application, idle_timeout_seconds=0)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)
    try:
        assert started.wait(timeout=2)
        # The listener wakes every 0.5s, comfortably beyond this idle timeout.
        time.sleep(0.7)
        assert thread.is_alive()
    finally:
        release.set()
        thread.join(timeout=3)

    assert not thread.is_alive()
    assert not server.endpoint.exists()


@requires_local_sockets
def test_broker_application_dispatches_maintain_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The daemon's startup-maintenance thread would race this test's direct
    # calls for the global lock; the scheduled pass is orthogonal here.
    monkeypatch.setenv("CODE_INDEXING_AUTO_MAINTENANCE", "0")
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    daemon = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=daemon.serve, daemon=True)
    thread.start()
    assert daemon.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=tmp_path)
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")

    project = broker.init_project(root)
    application.index_project(project.id)

    report = broker.maintain_storage(project.id, dry_run=True)

    assert report.dry_run is True
    entry = next(result for result in report.projects if result.project.id == project.id)
    assert entry.before is not None
    assert entry.after is None

    executed = broker.maintain_storage(project.id)
    assert executed.dry_run is False
    entry = next(result for result in executed.projects if result.project.id == project.id)
    assert entry.status == "ok"
    broker.stop()
    thread.join(timeout=2)
    assert not thread.is_alive()


@requires_local_sockets
def test_broker_forwards_the_index_trigger_and_serves_history(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    application = Application(paths, embedder=TinyEmbedder(), cwd=root)
    project = application.init_project(root)
    application.index_project(project.id)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=root)

    try:
        report = broker.index_project(project.id, trigger="watcher", wait_for_lock=True)
        page = broker.index_history(project.id, limit=10)
    finally:
        broker.stop()
        thread.join(timeout=2)

    assert report.trigger == "watcher"
    assert page.project is not None
    assert page.project.id == project.id
    assert any(run.run_id == report.run_id for run in page.runs)
    # Both runs are visible: the seed run (manual) and the triggered one.
    assert {run.trigger for run in page.runs} == {"manual", "watcher"}


@requires_local_sockets
def test_a_stale_daemon_is_reported_running_and_retired(tmp_path: Path) -> None:
    """A daemon left running by a previous release speaks an older protocol.

    It rejects every current-version frame (including the ping), so it must be
    reported as running with its own version -- not as absent, since it still
    holds the socket a replacement would need -- and retiring it means asking
    it to stop in the protocol version it does accept.
    """
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    paths.data.mkdir(parents=True)
    (paths.data / "daemon.token").write_text("shared-token")
    endpoint = daemon_endpoint(paths)
    old_protocol = 2

    def old_daemon() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(endpoint))
            server.listen()
            while True:
                connection, _ = server.accept()
                with connection:
                    request = receive_frame(connection)
                    if request.get("protocol") != old_protocol:
                        send_frame(
                            connection,
                            {
                                "id": request.get("id"),
                                "error": {
                                    "code": ErrorCode.INVALID_CONFIGURATION.value,
                                    "message": "Incompatible local daemon protocol",
                                    "details": {"expected": old_protocol},
                                },
                            },
                        )
                        continue
                    send_frame(connection, {"id": request.get("id"), "result": {"stopping": True}})
                    if request.get("method") == "stop":
                        break
        endpoint.unlink()

    thread = threading.Thread(target=old_daemon, daemon=True)
    thread.start()
    deadline = time.monotonic() + 2
    while not endpoint.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert daemon.daemon_status(paths) == {"running": True, "protocol": old_protocol}

    daemon._retire_stale_daemon(paths, old_protocol)
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert daemon.daemon_status(paths)["running"] is False


def test_daemon_is_current_treats_a_missing_or_mismatched_build_like_a_protocol_mismatch() -> None:
    """D1: `build` is checked exactly like `protocol` -- absent or different is stale."""
    current = {
        "running": True,
        "protocol": daemon.PROTOCOL_VERSION,
        "build": daemon.BUILD_IDENTITY,
    }
    assert daemon._daemon_is_current(current) is True
    assert daemon._daemon_is_current({**current, "build": "a-previous-build"}) is False
    # A daemon from before D1 answers ping with no `build` key at all.
    assert daemon._daemon_is_current({k: v for k, v in current.items() if k != "build"}) is False
    assert daemon._daemon_is_current({**current, "protocol": daemon.PROTOCOL_VERSION - 1}) is False
    assert daemon._daemon_is_current({**current, "running": False}) is False


def test_settings_digest_hashes_only_managed_keys_and_never_the_raw_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODE_INDEXING_EMBED_THREADS", "7")
    monkeypatch.setenv("UNRELATED_VARIABLE", "code-indexing-secret")
    digest = daemon._settings_digest(os.environ)
    assert set(digest) >= {"CODE_INDEXING_EMBED_THREADS"}
    assert "UNRELATED_VARIABLE" not in digest
    assert digest["CODE_INDEXING_EMBED_THREADS"] != "7"
    assert len(digest["CODE_INDEXING_EMBED_THREADS"]) == 64  # a sha256 hex digest
    # Deterministic: hashing the same value twice must not shuffle keys apart.
    assert digest == daemon._settings_digest(os.environ)


def test_warn_on_settings_mismatch_logs_once_and_never_the_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """D2: a differing key is named in the warning; its value never is."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    broker = BrokerApplication(paths)
    monkeypatch.setenv("CODE_INDEXING_EMBED_THREADS", "7")
    remote = daemon._settings_digest(os.environ)

    with caplog.at_level("WARNING", logger="code_indexing_mcp.daemon"):
        broker.warn_on_settings_mismatch(remote)
    assert caplog.records == []

    monkeypatch.setenv("CODE_INDEXING_EMBED_THREADS", "3")
    with caplog.at_level("WARNING", logger="code_indexing_mcp.daemon"):
        broker.warn_on_settings_mismatch(remote)
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "CODE_INDEXING_EMBED_THREADS" in message
    assert "7" not in message
    assert "3" not in message


def test_warn_on_settings_mismatch_ignores_a_non_dict_digest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A daemon from before D2 answers ping with no `settings_digest` at all."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    broker = BrokerApplication(paths)

    with caplog.at_level("WARNING", logger="code_indexing_mcp.daemon"):
        broker.warn_on_settings_mismatch(None)

    assert caplog.records == []


@requires_local_sockets
def test_ensure_daemon_warns_but_does_not_restart_on_a_settings_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """D2: differing settings between client and daemon warn once and never restart."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    monkeypatch.setenv("CODE_INDEXING_EMBED_THREADS", "7")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)

    def fail_popen(*args: object, **kwargs: object) -> None:
        raise AssertionError("a settings mismatch must warn, not restart the daemon")

    monkeypatch.setattr(daemon.subprocess, "Popen", fail_popen)
    monkeypatch.setenv("CODE_INDEXING_EMBED_THREADS", "3")

    try:
        with caplog.at_level("WARNING", logger="code_indexing_mcp.daemon"):
            daemon.ensure_daemon(paths, timeout_seconds=2)
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "CODE_INDEXING_EMBED_THREADS" in message
        assert "7" not in message
        assert "3" not in message
    finally:
        BrokerApplication(paths).stop()
        thread.join(timeout=2)
        assert not thread.is_alive()


@requires_local_sockets
def test_ensure_daemon_reuses_a_daemon_with_a_matching_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)

    def fail_popen(*args: object, **kwargs: object) -> None:
        raise AssertionError("a daemon with a matching build must be reused, not respawned")

    monkeypatch.setattr(daemon.subprocess, "Popen", fail_popen)

    try:
        broker = daemon.ensure_daemon(paths, timeout_seconds=2)
        assert broker.ping()["build"] == daemon.BUILD_IDENTITY
    finally:
        BrokerApplication(paths).stop()
        thread.join(timeout=2)
        assert not thread.is_alive()


@requires_local_sockets
def test_ensure_daemon_retires_a_running_daemon_with_a_mismatched_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D1: same protocol, different build -- retired and replaced, like a protocol mismatch.

    Mirrors ``test_a_stale_daemon_is_reported_running_and_retired``'s raw-socket
    stand-in, but on the *current* protocol: the scenario a protocol bump alone
    cannot catch, because the RPC shape has not changed, only the code or the
    schema behind it.
    """
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    paths.data.mkdir(parents=True)
    (paths.data / "daemon.token").write_text("shared-token")
    endpoint = daemon_endpoint(paths)

    def old_build_daemon() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(endpoint))
            server.listen()
            while True:
                connection, _ = server.accept()
                with connection:
                    request = receive_frame(connection)
                    if request.get("method") == "stop":
                        send_frame(
                            connection, {"id": request.get("id"), "result": {"stopping": True}}
                        )
                        break
                    send_frame(
                        connection,
                        {
                            "id": request.get("id"),
                            "result": {
                                "pid": 999999,
                                "protocol": daemon.PROTOCOL_VERSION,
                                "build": "a-previous-build",
                            },
                        },
                    )
        endpoint.unlink()

    old_thread = threading.Thread(target=old_build_daemon, daemon=True)
    old_thread.start()
    deadline = time.monotonic() + 2
    while not endpoint.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    status = daemon.daemon_status(paths)
    assert status == {
        "running": True,
        "pid": 999999,
        "protocol": daemon.PROTOCOL_VERSION,
        "build": "a-previous-build",
    }

    fresh_application = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    fresh_server = DaemonServer(paths, application=fresh_application, idle_timeout_seconds=60)
    fresh_thread: threading.Thread | None = None

    def fake_popen(*args: object, **kwargs: object) -> None:
        nonlocal fresh_thread
        fresh_thread = threading.Thread(target=fresh_server.serve, daemon=True)
        fresh_thread.start()
        return None

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)

    try:
        broker = daemon.ensure_daemon(paths, timeout_seconds=5)
        assert broker.ping()["build"] == daemon.BUILD_IDENTITY
    finally:
        old_thread.join(timeout=2)
        assert not old_thread.is_alive()
        BrokerApplication(paths).stop()
        if fresh_thread is not None:
            fresh_thread.join(timeout=2)
            assert not fresh_thread.is_alive()


@requires_local_sockets
def test_broker_forwards_scan_inspection(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    (root / "notes.md").write_text("not source\n")
    application = Application(paths, embedder=TinyEmbedder(), cwd=root)
    project = application.init_project(root)
    application.index_project(project.id)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=root)

    try:
        first = broker.inspect_scan(project.id, limit=1)
        second = broker.inspect_scan(project.id, limit=1, cursor=first.next_cursor)
        eligible = broker.inspect_scan(project.id, outcome="eligible")
    finally:
        broker.stop()
        thread.join(timeout=2)

    assert first.project is not None and first.project.id == project.id
    assert len(first.items) == 1
    assert first.next_cursor is not None
    assert len(second.items) == 1
    assert second.items[0].path != first.items[0].path
    assert {item.path.as_posix() for item in eligible.items} == {"main.py"}


@requires_local_sockets
def test_broker_round_trips_a_refactor_patch(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text(
        "def answer():\n    return 42\n\ncallback = answer\n\ndef caller():\n    return answer()\n"
    )
    application = Application(paths, embedder=TinyEmbedder(), cwd=root)
    project = application.init_project(root)
    application.index_project(project.id)
    server = DaemonServer(paths, application=application, idle_timeout_seconds=60)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert server.ready.wait(timeout=2)
    broker = BrokerApplication(paths, cwd=root)

    try:
        result = broker.emit_refactor_patch(
            DeclarationSelector(
                project=project.id,
                path="main.py",
                qualified_symbol="answer",
            ),
            RenameOperation(new_name="result"),
        )
    finally:
        broker.stop()
        thread.join(timeout=2)

    # The patch crosses the socket as a validated RefactorPatch model.
    assert result.applied == 3
    assert result.patch.startswith("diff --git a/main.py b/main.py\n")
    assert result.completeness.state == "complete"
    assert len(result.operation_digest) == 16
    assert result.slot_id


def test_broker_mirrors_application_surface() -> None:
    """D8: BrokerApplication's shared surface must track Application's, not silently drift.

    The surface checked is `ApplicationLike`'s own declared methods -- the
    query/project calls `server.py` makes polymorphically on whichever
    backend it was handed -- so a Protocol edit and this test cannot drift
    from each other; mypy separately enforces (via `create_server`'s and
    `AutoIndexingMCP.__init__`'s parameter types) that both `Application` and
    `BrokerApplication` actually satisfy it.

    `BrokerApplication` forwards most of this surface through `**params:
    Any`, which is exactly what lets it track an `Application` signature
    change automatically -- so the check below is not "identical
    signatures" (that would fail on every such method, by design) but
    "every parameter BrokerApplication *does* name explicitly still exists,
    by that name, on Application". A parameter BrokerApplication forwards
    under a name Application no longer accepts is the drift this exists to
    catch.
    """
    surface = sorted(
        name
        for name, member in vars(ApplicationLike).items()
        if not name.startswith("_") and inspect.isfunction(member)
    )
    assert surface, "ApplicationLike must declare at least one method"

    for name in surface:
        assert hasattr(Application, name), f"Application has no {name}"
        assert hasattr(BrokerApplication, name), f"BrokerApplication has no {name}"
        broker_params = {
            param.name
            for param in inspect.signature(getattr(BrokerApplication, name)).parameters.values()
            if param.name != "self" and param.kind is not inspect.Parameter.VAR_KEYWORD
        }
        app_params = {
            param.name
            for param in inspect.signature(getattr(Application, name)).parameters.values()
            if param.name != "self"
        }
        missing = broker_params - app_params
        assert not missing, (
            f"{name}: BrokerApplication names {sorted(missing)} that Application no longer accepts"
        )
