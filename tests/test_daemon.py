from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

from incode_mcp.application import Application, RuntimePaths
from incode_mcp.daemon import BrokerApplication, DaemonServer, receive_frame, send_frame


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
