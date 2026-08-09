"""Per-user local daemon and application-level JSON RPC client."""

from __future__ import annotations

import json
import os
import secrets
import socket
import stat
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
from pydantic import BaseModel, TypeAdapter

from .application import Application, RuntimePaths
from .errors import CodeIndexingError, ErrorCode
from .models import (
    CodeChunk,
    DeclarationSelector,
    IndexReport,
    ModelStatus,
    OutlineResponse,
    ProjectInfo,
    ProjectStatus,
    RefactorAnalysis,
    RefactorOperation,
    ReferenceResponse,
    RemovalReport,
    SearchResponse,
    SymbolResponse,
)
from .progress import IndexProgress, read_progress
from .settings import IndexSettings

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 16 * 1024**2
_REFACTOR_OPERATION: TypeAdapter[RefactorOperation] = TypeAdapter(RefactorOperation)

# Looked up dynamically because Windows' socket stubs have no AF_UNIX at all,
# so a direct reference fails type checking there even though it never runs.
# daemon_supported() gates every use.
_AF_UNIX: int | None = getattr(socket, "AF_UNIX", None)


def daemon_supported() -> bool:
    """Return whether this platform exposes Unix domain sockets."""
    return _AF_UNIX is not None


def require_daemon_support() -> None:
    if not daemon_supported():
        raise CodeIndexingError(
            ErrorCode.INVALID_CONFIGURATION,
            "The shared indexing daemon requires Unix domain sockets, which are "
            "unavailable on this platform; set CODE_INDEXING_BROKER=off or run "
            "'code-indexing-mcp serve --direct'",
            platform=sys.platform,
        )


def _local_socket() -> socket.socket:
    require_daemon_support()
    assert _AF_UNIX is not None  # require_daemon_support raises otherwise
    return socket.socket(_AF_UNIX, socket.SOCK_STREAM)


def _private_directory(directory: Path) -> Path:
    """Create *directory* as a user-private directory, refusing hostile paths.

    The endpoint may live under a shared temporary root, so an existing entry is
    only trusted when it is a real directory (not a symlink) owned by this user.
    """
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Gate on the ownership primitive rather than on the platform name, which
    # also keeps this type-checkable where os.getuid is absent from the stubs.
    if not hasattr(os, "getuid"):
        return directory
    info = os.lstat(directory)
    if not stat.S_ISDIR(info.st_mode):
        raise CodeIndexingError(
            ErrorCode.INVALID_CONFIGURATION,
            "The daemon runtime path is not a directory",
            path=str(directory),
        )
    if info.st_uid != os.getuid():
        raise CodeIndexingError(
            ErrorCode.INVALID_CONFIGURATION,
            "The daemon runtime directory is not owned by the current user",
            path=str(directory),
            owner_uid=info.st_uid,
        )
    if info.st_mode & 0o077:
        os.chmod(directory, 0o700)
    return directory


def daemon_endpoint(paths: RuntimePaths) -> Path:
    """Return a short, per-user socket path (macOS limits AF_UNIX paths to ~104 bytes)."""
    identity = str(os.getuid()) if hasattr(os, "getuid") else os.environ.get("USERNAME", "user")
    # XDG_RUNTIME_DIR is already per-user and mode 0700. Otherwise fall back to
    # the platform temporary directory, which is per-user on macOS and Windows
    # and validated by _private_directory everywhere.
    runtime_root = os.environ.get("XDG_RUNTIME_DIR")
    root = Path(runtime_root) if runtime_root else Path(tempfile.gettempdir())
    directory = _private_directory(root / f"code-indexing-mcp-{identity}")
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
        raise CodeIndexingError(
            ErrorCode.PROTOCOL_ERROR,
            "Local daemon request exceeds the maximum frame size",
            maximum_bytes=MAX_FRAME_BYTES,
        )
    connection.sendall(struct.pack("!I", len(encoded)) + encoded)


def receive_frame(connection: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", _receive_exact(connection, 4))[0]
    if size > MAX_FRAME_BYTES:
        raise CodeIndexingError(
            ErrorCode.PROTOCOL_ERROR,
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
    if isinstance(value, (set, frozenset)):
        # Sorted so identical filter sets always encode identically -- the
        # wire is JSON, which has no set type, and an unstable ordering
        # would make cursor round-trips (e.g. find_references' `kinds`)
        # spuriously mismatch on retry.
        return sorted(_jsonable(item) for item in value)
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
            raise CodeIndexingError(
                ErrorCode.INDEX_BUSY, "The per-user indexing daemon is already running"
            ) from exc
        self._token = self._load_or_create_token()
        if self.endpoint.exists():
            self.endpoint.unlink()
        listener = _local_socket()
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
                try:
                    threading.Thread(
                        target=self._handle,
                        args=(connection,),
                        name="code-indexing-mcp-daemon-request",
                        daemon=True,
                    ).start()
                except BaseException:
                    # A thread that never started will never run _handle's
                    # finally block, so release the request slot here.
                    with self._activity_lock:
                        self._active_requests -= 1
                    connection.close()
                    raise
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
                    raise CodeIndexingError(
                        ErrorCode.INVALID_CONFIGURATION,
                        "Incompatible local daemon protocol",
                        expected=PROTOCOL_VERSION,
                    )
                if not secrets.compare_digest(str(request.get("token", "")), self._token):
                    raise CodeIndexingError(
                        ErrorCode.INVALID_CONFIGURATION,
                        "Local daemon authentication failed",
                    )
                result = self._dispatch(str(request.get("method")), request.get("params") or {})
                send_frame(connection, {"id": request_id, "result": _jsonable(result)})
            except CodeIndexingError as exc:
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
                            "code": ErrorCode.PROTOCOL_ERROR.value,
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
        if method == "find_references":
            selector = DeclarationSelector.model_validate(params.pop("selector"))
            kinds = params.pop("kinds", None)
            return app.find_references(
                selector, kinds=set(kinds) if kinds is not None else None, roots=roots, **params
            )
        if method == "analyze_refactor":
            selector = DeclarationSelector.model_validate(params.pop("selector"))
            operation = _REFACTOR_OPERATION.validate_python(params.pop("operation"))
            return app.analyze_refactor(selector, operation, roots=roots, **params)
        if method == "model_status":
            # Answered by the daemon rather than the caller, because the daemon
            # is the process that will actually run indexing.
            return app.model_status()
        raise CodeIndexingError(ErrorCode.PROTOCOL_ERROR, f"Unknown daemon method: {method}")

    def _load_or_create_token(self) -> str:
        if self.token_path.exists():
            return self.token_path.read_text().strip()
        token = secrets.token_hex(32)
        # Create the file already restricted rather than widening a default-mode
        # file afterwards, which would leave the token briefly world-readable.
        try:
            descriptor = os.open(self.token_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return self.token_path.read_text().strip()
        with os.fdopen(descriptor, "w") as handle:
            handle.write(token)
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

    def _call_once(self, method: str, **params: Any) -> Any:
        token = self.token_path.read_text().strip()
        with _local_socket() as connection:
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
                code = ErrorCode.PROTOCOL_ERROR
            raise CodeIndexingError(code, str(error["message"]), **error.get("details", {}))
        return response.get("result")

    def _call(self, method: str, **params: Any) -> Any:
        try:
            return self._call_once(method, **params)
        except (FileNotFoundError, ConnectionRefusedError):
            # The daemon intentionally exits after an idle timeout, but an MCP
            # server can retain this broker for much longer. Restart only when
            # no request could have been sent; retrying broader socket failures
            # could duplicate a completed non-idempotent operation.
            ensure_daemon(self.paths)
            return self._call_once(method, **params)

    def _ping_once(self) -> dict[str, Any]:
        """Probe the daemon without starting it when the endpoint is absent."""
        return dict(self._call_once("ping"))

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

    def index_progress(self, project_id: str) -> IndexProgress | None:
        """Read the daemon's live progress snapshot straight from the shared data directory.

        Deliberately not an RPC: the daemon thread running the index is the one
        that would have to answer, and it is busy indexing.
        """

        return read_progress(self.paths.data / "progress", project_id)

    def project_status(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> ProjectStatus:
        return ProjectStatus.model_validate(
            self._call("project_status", project=project, roots=roots or [])
        )

    def project_is_stale(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> bool:
        """Use the status RPC so adding freshness does not change the daemon protocol."""
        return self.project_status(project, roots=roots).state == "stale"

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

    def find_references(self, selector: DeclarationSelector, **params: Any) -> ReferenceResponse:
        return ReferenceResponse.model_validate(
            self._call("find_references", selector=selector, **params)
        )

    def analyze_refactor(
        self, selector: DeclarationSelector, operation: RefactorOperation, **params: Any
    ) -> RefactorAnalysis:
        return RefactorAnalysis.model_validate(
            self._call("analyze_refactor", selector=selector, operation=operation, **params)
        )

    def model_status(self) -> ModelStatus:
        return ModelStatus.model_validate(self._call("model_status"))


def daemon_status(paths: RuntimePaths) -> dict[str, Any]:
    try:
        return {"running": True, **BrokerApplication(paths)._ping_once()}
    except (EOFError, OSError, CodeIndexingError):
        # EOFError is what a daemon shutting down mid-ping looks like: the
        # connect and the send both succeed, and the socket closes before the
        # reply arrives. It is not an OSError, so it used to escape a question
        # that has no failure answer -- every caller here is only asking whether
        # the daemon is up, and one that closed the connection is not.
        return {"running": False}


def ensure_daemon(paths: RuntimePaths, *, timeout_seconds: float = 10) -> BrokerApplication:
    require_daemon_support()
    status = daemon_status(paths)
    if status["running"]:
        return BrokerApplication(paths)
    paths.data.mkdir(parents=True, exist_ok=True)
    (paths.data / "locks").mkdir(parents=True, exist_ok=True)
    log_path = paths.data / "daemon.log"
    with FileLock(paths.data / "locks" / "daemon-start.lock"):
        if daemon_status(paths)["running"]:
            return BrokerApplication(paths)
        # Keep the child's stderr: a daemon that dies during startup is otherwise
        # indistinguishable from a slow one, and surfaces only as a timeout.
        with log_path.open("a") as log:
            subprocess.Popen(
                [sys.executable, "-m", "code_indexing_mcp.cli", "daemon", "run"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log,
                start_new_session=True,
            )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            broker = BrokerApplication(paths)
            try:
                broker._ping_once()
                return broker
            except (OSError, CodeIndexingError):
                time.sleep(0.05)
    raise CodeIndexingError(
        ErrorCode.DAEMON_UNAVAILABLE,
        f"Timed out starting the local indexing daemon; see {log_path}",
        log_path=str(log_path),
        timeout_seconds=timeout_seconds,
    )
