"""Per-user local daemon and application-level JSON RPC client."""

from __future__ import annotations

import json
import os
import secrets
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout
from pydantic import BaseModel

from .application import Application, RuntimePaths
from .errors import ErrorCode, IncodeError
from .models import (
    CodeChunk,
    IndexReport,
    OutlineResponse,
    ProjectInfo,
    ProjectStatus,
    RemovalReport,
    SearchResponse,
    SymbolResponse,
)
from .settings import IndexSettings

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 16 * 1024**2


def daemon_endpoint(paths: RuntimePaths) -> Path:
    """Return a short, per-user socket path (macOS limits AF_UNIX paths)."""
    identity = str(os.getuid()) if hasattr(os, "getuid") else os.environ.get("USERNAME", "user")
    temporary_root = (
        Path("/private/tmp") if Path("/private/tmp").is_dir() else Path(tempfile.gettempdir())
    )
    directory = temporary_root / f"incode-{identity}"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "nt":
        directory.chmod(0o700)
    digest = sha256(str(paths.data.resolve()).encode()).hexdigest()[:16]
    return directory / f"{digest}.sock"


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("Local daemon connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(connection: socket.socket, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    if len(encoded) > MAX_FRAME_BYTES:
        raise IncodeError(
            ErrorCode.INVALID_FILTER,
            "Local daemon request exceeds the maximum frame size",
            maximum_bytes=MAX_FRAME_BYTES,
        )
    connection.sendall(struct.pack("!I", len(encoded)) + encoded)


def receive_frame(connection: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", _receive_exact(connection, 4))[0]
    if size > MAX_FRAME_BYTES:
        raise IncodeError(
            ErrorCode.INVALID_FILTER,
            "Local daemon frame exceeds the maximum size",
            maximum_bytes=MAX_FRAME_BYTES,
        )
    value = json.loads(_receive_exact(connection, size))
    if not isinstance(value, dict):
        raise ValueError("Daemon frame must contain a JSON object")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


class DaemonServer:
    def __init__(
        self,
        paths: RuntimePaths,
        *,
        application: Application | None = None,
        idle_timeout_seconds: int = 300,
    ) -> None:
        self.paths = paths
        self.application = application or Application(paths)
        self.idle_timeout_seconds = idle_timeout_seconds
        self.endpoint = daemon_endpoint(paths)
        self.token_path = paths.data / "daemon.token"
        self.ready = threading.Event()
        self._stop = threading.Event()
        self._last_activity = time.monotonic()
        self._active_requests = 0
        self._activity_lock = threading.Lock()
        self._token = ""
        self._listener: socket.socket | None = None

    def serve(self) -> None:
        self.paths.data.mkdir(parents=True, exist_ok=True)
        lock_directory = self.paths.data / "locks"
        lock_directory.mkdir(parents=True, exist_ok=True)
        lifetime_lock = FileLock(lock_directory / "daemon.lock")
        try:
            lifetime_lock.acquire(timeout=0)
        except Timeout as exc:
            raise IncodeError(
                ErrorCode.INDEX_BUSY, "The per-user indexing daemon is already running"
            ) from exc
        self._token = self._load_or_create_token()
        if self.endpoint.exists():
            self.endpoint.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener = listener
        try:
            listener.bind(str(self.endpoint))
            if os.name != "nt":
                self.endpoint.chmod(0o600)
            listener.listen(32)
            listener.settimeout(0.5)
            self.ready.set()
            while not self._stop.is_set():
                with self._activity_lock:
                    idle = (
                        self._active_requests == 0
                        and time.monotonic() - self._last_activity >= self.idle_timeout_seconds
                    )
                if idle:
                    break
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                with self._activity_lock:
                    self._last_activity = time.monotonic()
                    self._active_requests += 1
                threading.Thread(
                    target=self._handle,
                    args=(connection,),
                    name="incode-daemon-request",
                    daemon=True,
                ).start()
        finally:
            listener.close()
            self._listener = None
            if self.endpoint.exists():
                self.endpoint.unlink()
            lifetime_lock.release()
            self.ready.set()

    def _handle(self, connection: socket.socket) -> None:
        with connection:
            request_id: Any = None
            try:
                request = receive_frame(connection)
                request_id = request.get("id")
                if request.get("protocol") != PROTOCOL_VERSION:
                    raise IncodeError(
                        ErrorCode.INVALID_CONFIGURATION,
                        "Incompatible local daemon protocol",
                        expected=PROTOCOL_VERSION,
                    )
                if not secrets.compare_digest(str(request.get("token", "")), self._token):
                    raise IncodeError(
                        ErrorCode.INVALID_CONFIGURATION,
                        "Local daemon authentication failed",
                    )
                result = self._dispatch(str(request.get("method")), request.get("params") or {})
                send_frame(connection, {"id": request_id, "result": _jsonable(result)})
            except IncodeError as exc:
                send_frame(
                    connection,
                    {
                        "id": request_id,
                        "error": {
                            "code": exc.code.value,
                            "message": str(exc),
                            "details": exc.details,
                        },
                    },
                )
            except BaseException as exc:
                send_frame(
                    connection,
                    {
                        "id": request_id,
                        "error": {
                            "code": ErrorCode.EMBEDDING_WORKER_FAILED.value,
                            "message": f"{type(exc).__name__}: {exc}",
                            "details": {},
                        },
                    },
                )
            finally:
                with self._activity_lock:
                    self._last_activity = time.monotonic()
                    self._active_requests -= 1

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "ping":
            return {"pid": os.getpid(), "protocol": PROTOCOL_VERSION}
        if method == "stop":
            self._stop.set()
            return {"stopping": True}
        app = self.application
        roots = [Path(root) for root in params.pop("roots", [])]
        if method == "init_project":
            path = params.pop("path", None)
            return app.init_project(Path(path) if path is not None else None, roots=roots, **params)
        if method == "discover_project":
            return app.discover_project(Path(params["root"]))
        if method == "index_project":
            return app.index_project(roots=roots, **params)
        if method == "project_status":
            return app.project_status(roots=roots, **params)
        if method == "list_projects":
            return app.list_projects()
        if method == "remove_project":
            return app.remove_project(**params)
        if method == "resolve_project":
            return app.resolve_project(params["explicit"], roots)
        if method == "resolve_search_scope":
            return app.resolve_search_scope(params.get("projects"), params["all_projects"], roots)
        if method == "search_code":
            return app.search_code(roots=roots, **params)
        if method == "find_symbol":
            return app.find_symbol(roots=roots, **params)
        if method == "file_outline":
            return app.file_outline(roots=roots, **params)
        if method == "get_chunk":
            return app.get_chunk(**params)
        raise IncodeError(ErrorCode.INVALID_FILTER, f"Unknown daemon method: {method}")

    def _load_or_create_token(self) -> str:
        if self.token_path.exists():
            return self.token_path.read_text().strip()
        token = secrets.token_hex(32)
        self.token_path.write_text(token)
        if os.name != "nt":
            self.token_path.chmod(0o600)
        return token


class BrokerApplication:
    """Application-compatible facade backed by the per-user daemon."""

    def __init__(self, paths: RuntimePaths, *, cwd: Path | None = None) -> None:
        self.paths = paths
        self.cwd = (cwd or Path.cwd()).resolve()
        self.settings = IndexSettings.from_environment()
        self.endpoint = daemon_endpoint(paths)
        self.token_path = paths.data / "daemon.token"

    @classmethod
    def from_environment(cls, *, cwd: Path | None = None) -> BrokerApplication:
        return cls(RuntimePaths.from_environment(), cwd=cwd)

    def _call(self, method: str, **params: Any) -> Any:
        token = self.token_path.read_text().strip()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(5)
            connection.connect(str(self.endpoint))
            connection.settimeout(None)
            send_frame(
                connection,
                {
                    "protocol": PROTOCOL_VERSION,
                    "id": uuid.uuid4().hex,
                    "token": token,
                    "method": method,
                    "params": _jsonable(params),
                },
            )
            response = receive_frame(connection)
        error = response.get("error")
        if error:
            try:
                code = ErrorCode(error["code"])
            except ValueError:
                code = ErrorCode.EMBEDDING_WORKER_FAILED
            raise IncodeError(code, str(error["message"]), **error.get("details", {}))
        return response.get("result")

    def ping(self) -> dict[str, Any]:
        return dict(self._call("ping"))

    def stop(self) -> dict[str, Any]:
        return dict(self._call("stop"))

    def init_project(
        self,
        path: Path | str | None = None,
        name: str | None = None,
        force_new_id: bool = False,
        *,
        roots: list[Path] | None = None,
    ) -> ProjectInfo:
        return ProjectInfo.model_validate(
            self._call(
                "init_project",
                path=path,
                name=name,
                force_new_id=force_new_id,
                roots=roots or [],
            )
        )

    def discover_project(self, root: Path) -> ProjectInfo | None:
        value = self._call("discover_project", root=root)
        return ProjectInfo.model_validate(value) if value is not None else None

    def list_projects(self) -> list[ProjectInfo]:
        return [ProjectInfo.model_validate(value) for value in self._call("list_projects")]

    def index_project(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        force: bool = False,
        wait_for_lock: bool = False,
    ) -> IndexReport:
        return IndexReport.model_validate(
            self._call(
                "index_project",
                project=project,
                roots=roots or [],
                force=force,
                wait_for_lock=wait_for_lock,
            )
        )

    def project_status(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> ProjectStatus:
        return ProjectStatus.model_validate(
            self._call("project_status", project=project, roots=roots or [])
        )

    def remove_project(self, project: str) -> RemovalReport:
        return RemovalReport.model_validate(self._call("remove_project", project=project))

    def resolve_project(self, explicit: str | None, roots: list[Path] | None = None) -> ProjectInfo:
        return ProjectInfo.model_validate(
            self._call("resolve_project", explicit=explicit, roots=roots or [])
        )

    def resolve_search_scope(
        self,
        projects: list[str] | None,
        all_projects: bool,
        roots: list[Path] | None = None,
    ) -> list[str]:
        return list(
            self._call(
                "resolve_search_scope",
                projects=projects,
                all_projects=all_projects,
                roots=roots or [],
            )
        )

    def search_code(self, query: str, **params: Any) -> SearchResponse:
        return SearchResponse.model_validate(self._call("search_code", query=query, **params))

    def find_symbol(self, name: str, project: str | None = None, **params: Any) -> SymbolResponse:
        return SymbolResponse.model_validate(
            self._call("find_symbol", name=name, project=project, **params)
        )

    def file_outline(self, path: str, project: str | None = None, **params: Any) -> OutlineResponse:
        return OutlineResponse.model_validate(
            self._call("file_outline", path=path, project=project, **params)
        )

    def get_chunk(self, chunk_id: str) -> CodeChunk:
        return CodeChunk.model_validate(self._call("get_chunk", chunk_id=chunk_id))


def daemon_status(paths: RuntimePaths) -> dict[str, Any]:
    try:
        return {"running": True, **BrokerApplication(paths).ping()}
    except (OSError, IncodeError, FileNotFoundError):
        return {"running": False}


def ensure_daemon(paths: RuntimePaths, *, timeout_seconds: float = 10) -> BrokerApplication:
    status = daemon_status(paths)
    if status["running"]:
        return BrokerApplication(paths)
    paths.data.mkdir(parents=True, exist_ok=True)
    (paths.data / "locks").mkdir(parents=True, exist_ok=True)
    with FileLock(paths.data / "locks" / "daemon-start.lock"):
        if daemon_status(paths)["running"]:
            return BrokerApplication(paths)
        subprocess.Popen(
            [sys.executable, "-m", "incode_mcp.cli", "daemon", "run"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            broker = BrokerApplication(paths)
            try:
                broker.ping()
                return broker
            except (OSError, IncodeError, FileNotFoundError):
                time.sleep(0.05)
    raise IncodeError(
        ErrorCode.EMBEDDING_WORKER_FAILED,
        "Timed out starting the local indexing daemon",
    )
