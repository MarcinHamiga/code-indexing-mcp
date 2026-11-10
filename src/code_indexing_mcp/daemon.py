"""Per-user local daemon and application-level JSON RPC client."""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import logging
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
from collections.abc import Callable, Iterator, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout
from pydantic import BaseModel, TypeAdapter

from . import __version__, update_check
from .application import Application, RuntimePaths
from .errors import CodeIndexingError, ErrorCode
from .indexing import REFERENCE_SCHEMA_VERSION
from .models import (
    ChangedSymbolsResponse,
    CodeChunk,
    DeadCodeReport,
    DeclarationSelector,
    ExampleSearchResponse,
    HistoryPage,
    ImpactRadiusResponse,
    IndexReport,
    IndexTrigger,
    MaintenanceReport,
    ModelStatus,
    OutlineResponse,
    ProjectInfo,
    ProjectStatus,
    RefactorAnalysis,
    RefactorOperation,
    RefactorPatch,
    ReferenceResponse,
    RemovalReport,
    ScanInspectionPage,
    SearchResponse,
    StorageStatus,
    SymbolResponse,
)
from .progress import IndexProgress, read_progress
from .settings import IndexSettings
from .storage import SCHEMA_VERSION

logger = logging.getLogger(__name__)

# Bumped whenever the RPC surface changes shape (version 6 added changed_symbols;
# version 5 added dead_code_report;
# version 4 added impact_radius;
# version 3 added chunked responses; version 2 added the `trigger`
# parameter to index_project): a long-lived daemon from a previous release must
# reject requests it cannot dispatch instead of failing inside them, and the
# mismatch is what tells ensure_daemon to retire it and start a current one.
PROTOCOL_VERSION = 6
# The module the daemon process is started from; see ensure_daemon (DMN-5).
DAEMON_ENTRY_MODULE = "code_indexing_mcp.daemon_entry"
MAX_FRAME_BYTES = 16 * 1024**2
MAX_RESPONSE_BYTES = 256 * 1024**2
MAX_RESPONSE_CHUNK_BYTES = 8 * 1024**2
_REFACTOR_OPERATION: TypeAdapter[RefactorOperation] = TypeAdapter(RefactorOperation)

# D3: neither side of the socket may hang forever. The server budget is how
# long it waits for a client that connected to actually send a request; the
# client budget is how long it waits for an answer once the request is sent.
# Indexing, storage maintenance, project creation, and stop are excluded from
# the client budget because they are expected to run long (a large repository
# can take minutes to embed) and are exactly the calls a caller is already
# choosing to wait out.
REQUEST_RECEIVE_TIMEOUT_SECONDS = 30
DAEMON_QUERY_TIMEOUT_SECONDS = 900
_UNBOUNDED_CLIENT_TIMEOUT_METHODS = frozenset({"index_project", "maintain_storage", "init_project"})
# DMN-7: liveness and control calls do no work, so a daemon that does not answer
# within a few seconds is wedged, not busy. ``stop`` only acknowledges: the
# caller waits for the process (lifetime lock) to go away separately, so a
# drain in progress never needs a long socket timeout.
DAEMON_CONTROL_TIMEOUT_SECONDS = 5
_CONTROL_METHODS = frozenset({"ping", "stop"})
# DMN-4: after a stop the daemon closes its listener and lets requests that are
# already running finish, up to these bounds. An index run (or other call a
# client already waits on without a budget) gets the longer one because cutting
# it off throws away minutes of embedding; both are finite so a wedged request
# cannot keep a stopping daemon, and everything waiting on its lock, forever.
STOP_DRAIN_SECONDS = 30
STOP_INDEX_DRAIN_SECONDS = 300
# How long a replacement daemon waits for the lifetime lock a stopping one
# still holds (DMN-3), and how long a caller waits for a stop to complete:
# long enough to cover an ordinary drain, short enough to report back.
LIFETIME_LOCK_WAIT_SECONDS = 15
DAEMON_STOP_WAIT_SECONDS = STOP_DRAIN_SECONDS + 5
# DMN-2: how long an index request waits for the startup model warm-up so the
# run's parent-memory baseline is not sampled while the model load is growing
# RSS. Warm-up failures end the wait too; this only bounds a stalled load.
MODEL_WARMUP_WAIT_SECONDS = 60
_NO_WARMUP_WAIT_METHODS = frozenset(
    {
        "ping",
        "stop",
        "list_projects",
        "discover_project",
        "remove_project",
        "init_project",
        "model_status",
    }
)
ACCEPT_ERROR_BACKOFF_SECONDS = 0.1
# Answered without taking a dispatch slot (DMN-6): they do no work, and a
# saturated daemon must still report itself alive and accept a stop.
_SEMAPHORE_EXEMPT_METHODS = frozenset({"ping", "stop"})
# D4: an unbounded thread-per-connection accept loop is fine for the sockets
# themselves, but every thread past this many concurrently *dispatching* a
# request is refused outright rather than left to queue invisibly behind the
# ones already running.
MAX_CONCURRENT_REQUESTS = 64


def _package_source_stamp() -> str:
    """A stamp for the installed code: stable across processes of the same
    install, different once the code changes.

    A managed install's git HEAD is the precise signal. Everywhere else (a
    development checkout, or git unavailable) the newest mtime across the
    package's own ``.py`` files stands in -- it is deterministic per installed
    tree, unlike a per-process random value, which would make every daemon
    believe every other daemon (including one it started itself) was stale.
    """
    context = update_check.install_context()
    if context is not None:
        head = update_check.checkout_head(context)
        if head is not None:
            return head
    package_directory = Path(__file__).resolve().parent
    try:
        newest = max(
            (path.stat().st_mtime_ns for path in package_directory.rglob("*.py")),
            default=0,
        )
    except OSError:
        newest = 0
    return str(newest)


def _build_identity() -> str:
    payload = f"{__version__}|{SCHEMA_VERSION}|{REFERENCE_SCHEMA_VERSION}|{_package_source_stamp()}"
    return sha256(payload.encode()).hexdigest()


# Computed once at import: a ping mismatch (daemon.py:BUILD_IDENTITY vs. the
# client's own) means a stale daemon is still serving code or a schema this
# process's checkout has moved past, and ensure_daemon retires it exactly as
# it would a protocol mismatch.
BUILD_IDENTITY = _build_identity()

# Looked up dynamically because Windows' socket stubs have no AF_UNIX at all,
# so a direct reference fails type checking there even though it never runs.
# daemon_supported() gates every use.
_AF_UNIX: int | None = getattr(socket, "AF_UNIX", None)
# Includes room for the terminating NUL on the shortest supported sockaddr_un.
UNIX_SOCKET_PATH_BYTES = 103


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
    endpoint = directory / f"{digest}.sock"
    if len(os.fsencode(endpoint)) <= UNIX_SOCKET_PATH_BYTES:
        return endpoint
    # tempfile.gettempdir() is often a long per-user path on macOS. Keep the
    # same ownership/symlink checks when choosing a short shared-root location.
    fallback_root = (
        Path("/tmp")
        if os.name != "nt"
        else Path(os.environ.get("SYSTEMROOT", "C:/Windows")) / "Temp"
    )
    fallback = (
        _private_directory(fallback_root / f"code-indexing-mcp-{identity}") / f"{digest}.sock"
    )
    if len(os.fsencode(fallback)) > UNIX_SOCKET_PATH_BYTES:
        raise CodeIndexingError(
            ErrorCode.INVALID_CONFIGURATION,
            "No usable short daemon socket path",
            path=str(fallback),
        )
    return fallback


def _published_daemon_endpoint(paths: RuntimePaths) -> Path | None:
    """Read the listener's location, shared by clients with different environments."""
    try:
        value = (paths.data / "daemon.endpoint").read_text(encoding="utf-8").strip()
    except (FileNotFoundError, UnicodeError):
        return None
    endpoint = Path(value)
    return endpoint if endpoint.is_absolute() and "\0" not in value else None


def _publish_daemon_endpoint(paths: RuntimePaths, endpoint: Path) -> None:
    # Publish only after listen(), atomically, while holding the lifetime lock.
    descriptor, name = tempfile.mkstemp(prefix=".daemon-endpoint-", dir=paths.data)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(endpoint.absolute()))
        os.replace(temporary, paths.data / "daemon.endpoint")
    finally:
        temporary.unlink(missing_ok=True)


def _daemon_endpoint_candidates(paths: RuntimePaths, previous: Path) -> Iterator[Path]:
    published = _published_daemon_endpoint(paths)
    if published is not None:
        yield published
    yield previous
    yield daemon_endpoint(paths)
    # Upgrade discovery for daemons which predate the endpoint record. In
    # particular, Codex drops XDG_RUNTIME_DIR while desktop clients retain it.
    # Only probe existing private directories; never create legacy locations.
    if hasattr(os, "getuid"):
        uid = os.getuid()
        digest = sha256(str(paths.data.resolve()).encode()).hexdigest()[:16]
        for root in (Path(f"/run/user/{uid}"), Path("/tmp")):
            directory = root / f"code-indexing-mcp-{uid}"
            try:
                info = directory.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(info.st_mode) and info.st_uid == uid and not info.st_mode & 0o077:
                endpoint = directory / f"{digest}.sock"
                if endpoint.exists():
                    yield endpoint


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
    _send_encoded_frame(connection, json.dumps(payload, separators=(",", ":")).encode())


def _send_encoded_frame(connection: socket.socket, encoded: bytes) -> None:
    if len(encoded) > MAX_FRAME_BYTES:
        raise CodeIndexingError(
            ErrorCode.PROTOCOL_ERROR,
            "Local daemon frame exceeds the maximum size",
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


def _send_response(connection: socket.socket, payload: Mapping[str, Any]) -> None:
    """Send one response, chunking its encoded JSON when it exceeds one frame."""
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise CodeIndexingError(
            ErrorCode.PROTOCOL_ERROR,
            "Local daemon response exceeds the maximum message size",
            maximum_bytes=MAX_RESPONSE_BYTES,
        )
    if len(encoded) <= MAX_FRAME_BYTES:
        _send_encoded_frame(connection, encoded)
        return
    # Base64 expands by 4/3. The additional frame headroom covers the id,
    # chunk markers, and JSON punctuation even when tests lower the frame cap.
    frame_budget = max(1, (MAX_FRAME_BYTES - 256) * 3 // 4)
    chunk_size = min(MAX_RESPONSE_CHUNK_BYTES, frame_budget)
    request_id = payload.get("id")
    for start in range(0, len(encoded), chunk_size):
        end = min(start + chunk_size, len(encoded))
        send_frame(
            connection,
            {
                "id": request_id,
                "response_chunk": base64.b64encode(encoded[start:end]).decode("ascii"),
                "more": end < len(encoded),
            },
        )


def _receive_response(connection: socket.socket) -> dict[str, Any]:
    """Receive one ordinary or transparently chunked daemon response."""
    frame = receive_frame(connection)
    if "response_chunk" not in frame:
        return frame
    request_id = frame.get("id")
    chunks: list[bytes] = []
    size = 0
    while True:
        if frame.get("id") != request_id or not isinstance(frame.get("response_chunk"), str):
            raise CodeIndexingError(ErrorCode.PROTOCOL_ERROR, "Invalid local daemon response chunk")
        try:
            chunk = base64.b64decode(frame["response_chunk"], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise CodeIndexingError(
                ErrorCode.PROTOCOL_ERROR, "Invalid local daemon response chunk encoding"
            ) from exc
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            raise CodeIndexingError(
                ErrorCode.PROTOCOL_ERROR,
                "Local daemon response exceeds the maximum message size",
                maximum_bytes=MAX_RESPONSE_BYTES,
            )
        chunks.append(chunk)
        more = frame.get("more")
        if more is False:
            break
        if more is not True:
            raise CodeIndexingError(
                ErrorCode.PROTOCOL_ERROR, "Invalid local daemon response chunk marker"
            )
        frame = receive_frame(connection)
    value = json.loads(b"".join(chunks))
    if not isinstance(value, dict):
        raise CodeIndexingError(
            ErrorCode.PROTOCOL_ERROR, "Daemon response must contain a JSON object"
        )
    return value


def _client_budget(method: str) -> float | None:
    """The socket timeout a client applies while waiting for *method*'s answer."""
    if method in _CONTROL_METHODS:
        return DAEMON_CONTROL_TIMEOUT_SECONDS
    if method in _UNBOUNDED_CLIENT_TIMEOUT_METHODS:
        return None
    return DAEMON_QUERY_TIMEOUT_SECONDS


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


def _settings_digest(environment: Mapping[str, str]) -> dict[str, str]:
    """Per-key value hashes of every ``CODE_INDEXING_*`` environment variable.

    Covers the ``IndexSettings`` fields plus ``CODE_INDEXING_OFFLINE``,
    ``DATA_DIR``, and ``CACHE_DIR`` -- every managed setting shares this
    prefix, so filtering on it is exactly "what the daemon actually
    consumes" without hand-maintaining a name list that would drift from
    ``installer.settings_spec``. A dict of *per-key* hashes, not one combined
    hash, is what lets ``BrokerApplication`` name exactly which keys differ:
    each side hashes its own value for a key and compares digests, so a raw
    setting value never has to cross the wire either way.
    """
    return {
        key: sha256(value.encode()).hexdigest()
        for key, value in environment.items()
        if key.startswith("CODE_INDEXING_")
    }


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
        self.application.cwd_scopes_requests = False
        self.idle_timeout_seconds = idle_timeout_seconds
        self.endpoint = daemon_endpoint(paths)
        self.token_path = paths.data / "daemon.token"
        # Captured once, at construction: "the environment the daemon started
        # with" (D2), which for the real entry point is the first client's
        # environment (ensure_daemon spawns the daemon subprocess inheriting
        # its own os.environ) -- not whatever a later caller happens to run
        # under.
        self._settings_digest = _settings_digest(os.environ)
        self.ready = threading.Event()
        self._stop = threading.Event()
        self._last_activity = time.monotonic()
        self._active_requests = 0
        self._maintenance_active = False
        self._model_warmup_active = False
        # Set whenever no startup warm-up is loading (initially, so a server
        # that is dispatched to directly never waits); cleared by serve().
        self._model_warmup_finished = threading.Event()
        self._model_warmup_finished.set()
        self._active_long_requests = 0
        self._activity_lock = threading.Lock()
        self._token = ""
        self._listener: socket.socket | None = None
        self._maintenance_thread: threading.Thread | None = None
        self._model_warmup_thread: threading.Thread | None = None
        # D4: bounds concurrent *dispatch*, separate from the unbounded
        # thread-per-connection accept loop below -- a request past the cap
        # is refused outright rather than left to queue invisibly.
        self._request_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)

    def request_stop(self) -> None:
        """Ask ``serve`` to stop accepting, drain, clean up and return."""
        self._stop.set()

    def serve(self) -> None:
        self.paths.ensure_private()
        lock_directory = self.paths.data / "locks"
        lock_directory.mkdir(parents=True, exist_ok=True)
        lifetime_lock = FileLock(lock_directory / "daemon.lock")
        try:
            # DMN-3: a daemon that was just asked to stop keeps the lock until
            # its drain and background threads finish, so a replacement waits
            # (bounded) instead of failing the moment it starts.
            lifetime_lock.acquire(timeout=LIFETIME_LOCK_WAIT_SECONDS)
        except Timeout as exc:
            raise CodeIndexingError(
                ErrorCode.INDEX_BUSY, "The per-user indexing daemon is already running"
            ) from exc
        self._token = self._load_or_create_token()
        # A retiring daemon closes its listener before it unlinks the path, and
        # ensure_daemon starts the replacement the moment a connect is refused,
        # so both processes can reach for the same stale path at once. Losing
        # that race must not kill the new daemon before it ever binds.
        with contextlib.suppress(FileNotFoundError):
            self.endpoint.unlink()
        listener = _local_socket()
        self._listener = listener
        try:
            listener.bind(str(self.endpoint))
            if os.name != "nt":
                self.endpoint.chmod(0o600)
            listener.listen(32)
            listener.settimeout(0.5)
            _publish_daemon_endpoint(self.paths, self.endpoint)
            # Marked before requests can arrive: an index request must see the
            # warm-up as pending, not as not-yet-started (DMN-2).
            with self._activity_lock:
                self._model_warmup_active = True
            self._model_warmup_finished.clear()
            self.ready.set()
            maintenance_thread = threading.Thread(
                target=self._run_startup_maintenance,
                name="code-indexing-mcp-daemon-maintenance",
                daemon=True,
            )
            self._maintenance_thread = maintenance_thread
            with self._activity_lock:
                self._maintenance_active = True
            try:
                maintenance_thread.start()
            except BaseException:
                with self._activity_lock:
                    self._maintenance_active = False
                self._maintenance_thread = None
                raise
            # D7: warm the query model so the first real search does not pay
            # for loading it. Its own thread, alongside maintenance's, so a
            # slow load never delays serving.
            model_warmup_thread = threading.Thread(
                target=self._warm_query_model,
                name="code-indexing-mcp-daemon-model-warmup",
                daemon=True,
            )
            self._model_warmup_thread = model_warmup_thread
            with self._activity_lock:
                self._model_warmup_active = True
            try:
                model_warmup_thread.start()
            except BaseException:
                with self._activity_lock:
                    self._model_warmup_active = False
                self._model_warmup_finished.set()
                self._model_warmup_thread = None
                raise
            while not self._stop.is_set():
                with self._activity_lock:
                    idle = (
                        self._active_requests == 0
                        and not self._maintenance_active
                        and not self._model_warmup_active
                        and time.monotonic() - self._last_activity >= self.idle_timeout_seconds
                    )
                if idle:
                    break
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                except OSError as exc:
                    # DMN-8: EMFILE/ENFILE/ECONNABORTED and friends are
                    # conditions of the moment, not reasons to take every
                    # client's daemon down. Back off briefly and keep serving.
                    if self._stop.is_set():
                        break
                    logger.warning("Daemon accept failed (%s); retrying", exc)
                    self._stop.wait(ACCEPT_ERROR_BACKOFF_SECONDS)
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
            # DMN-4: stop (or idle exit) must not pull the process out from
            # under requests that are mid-flight. The listener is already
            # closed, so nothing new arrives while they finish.
            self._drain_requests()
            if self._maintenance_thread is not None:
                self._maintenance_thread.join()
                self._maintenance_thread = None
            if self._model_warmup_thread is not None:
                self._model_warmup_thread.join()
                self._model_warmup_thread = None
            self._listener = None
            with contextlib.suppress(FileNotFoundError):
                self.endpoint.unlink()
            (self.paths.data / "daemon.endpoint").unlink(missing_ok=True)
            # Buffered slot touches (touch_slot) must not be lost when the
            # daemon exits: the next process's LRU retention decision reads
            # last_used_at from disk.
            self.application.store.close()
            lifetime_lock.release()
            self.ready.set()

    def _drain_requests(self) -> None:
        """Wait, bounded, for in-flight requests; index-class calls get longer."""
        started = time.monotonic()
        while True:
            with self._activity_lock:
                active = self._active_requests
                long_running = self._active_long_requests
            if active == 0:
                return
            budget = STOP_INDEX_DRAIN_SECONDS if long_running else STOP_DRAIN_SECONDS
            if time.monotonic() - started >= budget:
                logger.warning(
                    "Daemon stopping with %d request(s) still in flight after %.0fs",
                    active,
                    budget,
                )
                return
            time.sleep(0.02)

    def _run_startup_maintenance(self) -> None:
        """Run overdue scheduled maintenance after startup, without blocking serving.

        Maintenance needs the writer locks an index run would hold, and it is
        checked at most once per 24 hours, so a failure here only delays
        cleanup. The daemon's idle timer counts the pass as active and shutdown
        joins its thread, so neither idle reaping nor an explicit stop can cut
        maintenance off midway.
        """
        try:
            with self._activity_lock:
                self._last_activity = time.monotonic()
            self.application.maybe_run_maintenance()
        except Exception:
            logger.exception("Scheduled maintenance after daemon startup failed")
        finally:
            with self._activity_lock:
                self._maintenance_active = False
                self._last_activity = time.monotonic()

    def _warm_query_model(self) -> None:
        """D7: load the query embedding model once at startup.

        Without this, the first real query after every daemon start pays the
        model's load latency; every query after it does not. A failure here
        -- offline mode with no cached model, most plausibly -- costs nothing
        but that head start, since a query still loads the model lazily on
        its own the first time it needs it, so it is logged and moved on
        from rather than raised.
        """
        try:
            with self._activity_lock:
                self._last_activity = time.monotonic()
            self.application.prepare_model()
        except Exception as exc:
            logger.warning("Model warm-up after daemon startup failed: %s", exc)
        finally:
            with self._activity_lock:
                self._model_warmup_active = False
                self._last_activity = time.monotonic()
            self._model_warmup_finished.set()

    def _handle(self, connection: socket.socket) -> None:
        with connection:
            request_id: Any = None
            method = "<unknown>"
            long_running = False
            try:
                # D3: bounds how long a connected-but-silent client can be
                # held open; lifted once a request has actually arrived so a
                # slow reader of a large response is not cut off by the same
                # budget. Read before the D4 concurrency check below so that
                # check's own reply-and-close never races the client's still
                # in-flight send: a server that closes before a client has
                # finished writing its request can hand it a raw
                # BrokenPipeError instead of the clean error this exists to
                # guarantee.
                connection.settimeout(REQUEST_RECEIVE_TIMEOUT_SECONDS)
                try:
                    request = receive_frame(connection)
                except TimeoutError as exc:
                    raise CodeIndexingError(
                        ErrorCode.PROTOCOL_ERROR, "Timed out waiting for the request"
                    ) from exc
                finally:
                    connection.settimeout(None)
                request_id = request.get("id")
                method = str(request.get("method"))
                if method in _UNBOUNDED_CLIENT_TIMEOUT_METHODS:
                    long_running = True
                    with self._activity_lock:
                        self._active_long_requests += 1
                if request.get("protocol") != PROTOCOL_VERSION:
                    raise CodeIndexingError(
                        ErrorCode.INVALID_CONFIGURATION,
                        "Incompatible local daemon protocol",
                        expected=PROTOCOL_VERSION,
                    )
                if not self._token or not secrets.compare_digest(
                    str(request.get("token", "")), self._token
                ):
                    raise CodeIndexingError(
                        ErrorCode.INVALID_CONFIGURATION,
                        "Local daemon authentication failed",
                    )
                if method in _SEMAPHORE_EXEMPT_METHODS:
                    # Liveness and control calls cost nothing and must still
                    # work when every dispatch slot is held by queries whose
                    # clients gave up: otherwise a saturated daemon reads as
                    # "not running" and cannot be stopped.
                    result = self._dispatch(method, request.get("params") or {})
                    _send_response(connection, {"id": request_id, "result": _jsonable(result)})
                elif not self._request_semaphore.acquire(blocking=False):
                    # D4: refused outright rather than left to queue behind the
                    # MAX_CONCURRENT_REQUESTS already dispatching -- an invisible
                    # queue would just move the hang from "no response" to
                    # "a response eventually, from however deep the backlog is".
                    raise CodeIndexingError(
                        ErrorCode.INDEX_BUSY,
                        "The local daemon is at its request limit",
                        limit=MAX_CONCURRENT_REQUESTS,
                    )
                else:
                    try:
                        result = self._dispatch(method, request.get("params") or {})
                        _send_response(connection, {"id": request_id, "result": _jsonable(result)})
                    finally:
                        self._request_semaphore.release()
            except CodeIndexingError as exc:
                # A client that has already given up (D3's own query budget,
                # or a plain disconnect) leaves nothing on the other end to
                # receive this -- best-effort, since there is no one left to
                # tell that the send itself failed.
                with contextlib.suppress(OSError):
                    _send_response(
                        connection,
                        {
                            "id": request_id,
                            "error": {
                                "code": exc.code.value,
                                # The bare message: str(exc) already carries the
                                # code, and the client rebuilds the prefix (DMN-10).
                                "message": exc.args[0] if exc.args else "",
                                "details": exc.details,
                            },
                        },
                    )
            except (EOFError, ConnectionResetError, BrokenPipeError) as exc:
                # The client went away (hung up before or while we answered).
                # That is routine, not a dispatch failure worth a traceback.
                logger.debug("Client disconnected during %s: %s", method, exc)
            except Exception as exc:
                # D5: everything that reaches here is a dispatch failure, not a
                # transport one (those are CodeIndexingError above); the type
                # name is detail, not message, so it composes with
                # CodeIndexingError.for_client() the same way every other error
                # does. Deliberately narrower than the BaseException this
                # replaces: SystemExit and KeyboardInterrupt inside a request
                # thread must still surface as themselves, not be swallowed
                # into a client-facing error response.
                logger.exception("Unhandled daemon dispatch failure for method %s", method)
                with contextlib.suppress(OSError):
                    _send_response(
                        connection,
                        {
                            "id": request_id,
                            "error": {
                                "code": ErrorCode.INTERNAL_ERROR.value,
                                "message": f"The local daemon failed while handling {method}",
                                "details": {"type": type(exc).__name__},
                            },
                        },
                    )
            finally:
                with self._activity_lock:
                    self._last_activity = time.monotonic()
                    self._active_requests -= 1
                    if long_running:
                        self._active_long_requests -= 1

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "ping":
            return {
                "pid": os.getpid(),
                "protocol": PROTOCOL_VERSION,
                "build": BUILD_IDENTITY,
                "settings_digest": self._settings_digest,
            }
        if method == "stop":
            self.request_stop()
            return {"stopping": True}
        if method not in _NO_WARMUP_WAIT_METHODS:
            self._model_warmup_finished.wait(MODEL_WARMUP_WAIT_SECONDS)
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
        if method == "index_history":
            return app.index_history(roots=roots, **params)
        if method == "inspect_scan":
            return app.inspect_scan(roots=roots, **params)
        if method == "storage_status":
            return app.storage_status(roots=roots, **params)
        if method == "maintain_storage":
            return app.maintain_storage(roots=roots, **params)
        if method == "list_projects":
            return app.list_projects()
        if method == "remove_project":
            return app.remove_project(**params)
        if method == "resolve_project":
            return app.resolve_project(params["explicit"], roots)
        if method == "resolve_search_scope":
            return app.resolve_search_scope(params.get("projects"), params["all_projects"], roots)
        if method == "resolve_scope_checkouts":
            return app.resolve_scope_checkouts(
                params.get("projects"), params["all_projects"], roots
            )
        if method == "search_code":
            return app.search_code(roots=roots, **params)
        if method == "search_by_example":
            return app.search_by_example(roots=roots, **params)
        if method == "find_symbol":
            return app.find_symbol(roots=roots, **params)
        if method == "dead_code_report":
            return app.dead_code_report(roots=roots, **params)
        if method == "file_outline":
            return app.file_outline(roots=roots, **params)
        if method == "changed_symbols":
            return app.changed_symbols(roots=roots, **params)
        if method == "get_chunk":
            return app.get_chunk(roots=roots, **params)
        if method == "find_references":
            selector = DeclarationSelector.model_validate(params.pop("selector"))
            kinds = params.pop("kinds", None)
            return app.find_references(
                selector, kinds=set(kinds) if kinds is not None else None, roots=roots, **params
            )
        if method == "impact_radius":
            selector = DeclarationSelector.model_validate(params.pop("selector"))
            kinds = params.pop("kinds", None)
            return app.impact_radius(
                selector, kinds=set(kinds) if kinds is not None else None, roots=roots, **params
            )
        if method == "analyze_refactor":
            selector = DeclarationSelector.model_validate(params.pop("selector"))
            operation = _REFACTOR_OPERATION.validate_python(params.pop("operation"))
            return app.analyze_refactor(selector, operation, roots=roots, **params)
        if method == "emit_refactor_patch":
            selector = DeclarationSelector.model_validate(params.pop("selector"))
            operation = _REFACTOR_OPERATION.validate_python(params.pop("operation"))
            return app.emit_refactor_patch(selector, operation, roots=roots, **params)
        if method == "model_status":
            # Answered by the daemon rather than the caller, because the daemon
            # is the process that will actually run indexing.
            return app.model_status()
        raise CodeIndexingError(ErrorCode.PROTOCOL_ERROR, f"Unknown daemon method: {method}")

    def _load_or_create_token(self) -> str:
        if self.token_path.exists():
            existing = self.token_path.read_text().strip()
            # An empty file (a crash between create and write) would otherwise
            # authenticate an empty token; replace it instead.
            if existing:
                return existing
        token = secrets.token_hex(32)
        # Written to a private temporary file and renamed into place: the token
        # is never world-readable, and a crash cannot leave an empty file.
        temporary = self.token_path.with_name(f".daemon-token-{secrets.token_hex(4)}")
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(token)
            os.replace(temporary, self.token_path)
        finally:
            temporary.unlink(missing_ok=True)
        return token


_CWD_INDEPENDENT_METHODS = frozenset(
    {"ping", "stop", "list_projects", "remove_project", "discover_project"}
)


class BrokerApplication:
    """Application-compatible facade backed by the per-user daemon."""

    def __init__(self, paths: RuntimePaths, *, cwd: Path | None = None) -> None:
        self.paths = paths
        self.cwd = (cwd or Path.cwd()).resolve()
        self.settings = IndexSettings.from_environment()
        self.endpoint = _published_daemon_endpoint(paths) or daemon_endpoint(paths)
        self.token_path = paths.data / "daemon.token"

    @classmethod
    def from_environment(cls, *, cwd: Path | None = None) -> BrokerApplication:
        return cls(RuntimePaths.from_environment(), cwd=cwd)

    def _connect(self) -> socket.socket:
        seen: set[Path] = set()
        last_error: OSError | None = None
        for endpoint in _daemon_endpoint_candidates(self.paths, self.endpoint):
            if endpoint in seen:
                continue
            if len(os.fsencode(endpoint)) > UNIX_SOCKET_PATH_BYTES:
                last_error = FileNotFoundError("Daemon endpoint exceeds Unix socket path limit")
                continue
            seen.add(endpoint)
            connection = _local_socket()
            try:
                connection.settimeout(5)
                connection.connect(str(endpoint))
            except (FileNotFoundError, ConnectionRefusedError) as exc:
                connection.close()
                last_error = exc
                continue
            except BaseException:
                connection.close()
                raise
            self.endpoint = endpoint
            return connection
        assert last_error is not None
        raise last_error

    def _call_once(self, method: str, *, _protocol: int = PROTOCOL_VERSION, **params: Any) -> Any:
        token = self.token_path.read_text().strip()
        if method not in _CWD_INDEPENDENT_METHODS:
            # The daemon serves many clients from one process, so its own cwd
            # says nothing about this caller. Scope rootless requests to the
            # caller's cwd (the resolver walks up to the nearest marker).
            if not params.get("roots"):
                params["roots"] = [self.cwd]
            if method == "init_project" and params.get("path") is not None:
                params["path"] = (self.cwd / Path(params["path"]).expanduser()).resolve()
        # D3: unbounded only for the calls a caller is already choosing to
        # wait out (index_project's own repository can take minutes to
        # embed); every other method gets a generous but finite budget so a
        # wedged daemon fails the call instead of hanging it forever.
        budget = _client_budget(method)
        # Discovery retries only connect failures, before any request is sent.
        # Refresh the record on every call so retained brokers follow restarts.
        with self._connect() as connection:
            connection.settimeout(budget)
            try:
                send_frame(
                    connection,
                    {
                        "protocol": _protocol,
                        "id": uuid.uuid4().hex,
                        "token": token,
                        "method": method,
                        "params": _jsonable(params),
                    },
                )
                response = _receive_response(connection)
            except TimeoutError as exc:
                raise CodeIndexingError(
                    ErrorCode.DAEMON_UNAVAILABLE,
                    f"Timed out waiting for the local daemon to answer {method!r}",
                    method=method,
                    timeout_seconds=budget,
                ) from exc
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
            try:
                return self._call_once(method, **params)
            except (FileNotFoundError, ConnectionRefusedError):
                # The daemon intentionally exits after an idle timeout, but an MCP
                # server can retain this broker for much longer. Restart only when
                # no request could have been sent; retrying broader socket failures
                # could duplicate a completed non-idempotent operation.
                ensure_daemon(self.paths)
                # Inside the outer try: the retry fails like any other call (DMN-11).
                return self._call_once(method, **params)
        except (
            EOFError,
            ConnectionResetError,
            BrokenPipeError,
            TimeoutError,
            FileNotFoundError,
            ConnectionRefusedError,
        ) as exc:
            # D5: the connection died mid-request (the daemon crashed, was
            # killed, or the socket otherwise dropped) rather than at connect
            # time -- unlike the retry above, this is never retried, because
            # the daemon may already have started (or finished) a
            # non-idempotent operation such as index_project. A raw
            # socket/EOF exception must not leak past the same error contract
            # every other daemon failure here keeps. TimeoutError (a plain
            # socket.timeout) is already converted inside _call_once with a
            # more specific detail -- the query budget that expired -- so it
            # is caught here only as a backstop.
            raise CodeIndexingError(
                ErrorCode.DAEMON_UNAVAILABLE,
                f"Lost the connection to the local daemon while calling {method!r}",
                method=method,
                log_path=str(self.paths.data / "daemon.log"),
            ) from exc

    def _ping_once(self) -> dict[str, Any]:
        """Probe the daemon without starting it when the endpoint is absent."""
        return dict(self._call_once("ping"))

    def warn_on_settings_mismatch(self, remote_digest: object) -> None:
        """D2: log once if this environment disagrees with the one the daemon started with.

        Never a restart -- two live clients with different settings cannot
        both win, and racing to restart the daemon over it would be worse
        than a warning (`configure` is the durable fix, D6). Only key *names*
        are logged; values, on both sides, never leave the process that holds
        them.
        """
        if not isinstance(remote_digest, dict):
            return
        local = _settings_digest(os.environ)
        # A key present on only one side compares unequal against `.get`'s
        # `None`, so it is caught by the same comparison as a changed value.
        differing = sorted(
            key
            for key in set(remote_digest) | set(local)
            if remote_digest.get(key) != local.get(key)
        )
        if differing:
            logger.warning(
                "This client's environment differs from the running daemon's for: %s; the "
                "daemon keeps serving the settings it started with. Run `code-indexing-mcp "
                "configure` to change them durably, or `code-indexing-mcp daemon stop` to "
                "force a restart on this client's environment.",
                ", ".join(differing),
            )

    def ping(self) -> dict[str, Any]:
        return dict(self._call("ping"))

    def stop(self) -> dict[str, Any]:
        return dict(self._call("stop"))

    def init_project(
        self,
        path: Path | str | None = None,
        name: str | None = None,
        force_new_id: bool = False,
        allow_overlap: bool = False,
        *,
        roots: list[Path] | None = None,
    ) -> ProjectInfo:
        return ProjectInfo.model_validate(
            self._call(
                "init_project",
                path=path,
                name=name,
                force_new_id=force_new_id,
                allow_overlap=allow_overlap,
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
        trigger: IndexTrigger = "manual",
    ) -> IndexReport:
        return IndexReport.model_validate(
            self._call(
                "index_project",
                project=project,
                roots=roots or [],
                force=force,
                wait_for_lock=wait_for_lock,
                trigger=trigger,
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

    def index_history(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> HistoryPage:
        return HistoryPage.model_validate(
            self._call(
                "index_history",
                project=project,
                roots=roots or [],
                cursor=cursor,
                limit=limit,
            )
        )

    def inspect_scan(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        outcome: str | None = None,
        reason: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> ScanInspectionPage:
        return ScanInspectionPage.model_validate(
            self._call(
                "inspect_scan",
                project=project,
                roots=roots or [],
                outcome=outcome,
                reason=reason,
                cursor=cursor,
                limit=limit,
            )
        )

    def storage_status(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> StorageStatus:
        return StorageStatus.model_validate(
            self._call("storage_status", project=project, roots=roots or [])
        )

    def maintain_storage(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        dry_run: bool = False,
        wait_for_lock: bool = True,
    ) -> MaintenanceReport:
        return MaintenanceReport.model_validate(
            self._call(
                "maintain_storage",
                project=project,
                roots=roots or [],
                dry_run=dry_run,
                wait_for_lock=wait_for_lock,
            )
        )

    def project_is_stale(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> bool:
        """Use the status RPC so adding freshness does not change the daemon protocol."""
        return self.project_status(project, roots=roots).state in {"stale", "rebuild_required"}

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

    def resolve_scope_checkouts(
        self,
        projects: list[str] | None,
        all_projects: bool,
        roots: list[Path] | None = None,
    ) -> list[ProjectInfo]:
        return [
            ProjectInfo.model_validate(item)
            for item in self._call(
                "resolve_scope_checkouts",
                projects=projects,
                all_projects=all_projects,
                roots=roots or [],
            )
        ]

    def search_code(self, query: str, **params: Any) -> SearchResponse:
        return SearchResponse.model_validate(self._call("search_code", query=query, **params))

    def search_by_example(self, example: str, **params: Any) -> ExampleSearchResponse:
        return ExampleSearchResponse.model_validate(
            self._call("search_by_example", example=example, **params)
        )

    def find_symbol(self, name: str, project: str | None = None, **params: Any) -> SymbolResponse:
        return SymbolResponse.model_validate(
            self._call("find_symbol", name=name, project=project, **params)
        )

    def file_outline(self, path: str, project: str | None = None, **params: Any) -> OutlineResponse:
        return OutlineResponse.model_validate(
            self._call("file_outline", path=path, project=project, **params)
        )

    def changed_symbols(self, project: str | None = None, **params: Any) -> ChangedSymbolsResponse:
        return ChangedSymbolsResponse.model_validate(
            self._call("changed_symbols", project=project, **params)
        )

    def get_chunk(self, chunk_id: str, *, roots: list[Path] | None = None) -> CodeChunk:
        return CodeChunk.model_validate(
            self._call("get_chunk", chunk_id=chunk_id, roots=roots or [])
        )

    def find_references(self, selector: DeclarationSelector, **params: Any) -> ReferenceResponse:
        return ReferenceResponse.model_validate(
            self._call("find_references", selector=selector, **params)
        )

    def dead_code_report(self, project: str | None = None, **params: Any) -> DeadCodeReport:
        return DeadCodeReport.model_validate(
            self._call("dead_code_report", project=project, **params)
        )

    def impact_radius(self, selector: DeclarationSelector, **params: Any) -> ImpactRadiusResponse:
        return ImpactRadiusResponse.model_validate(
            self._call("impact_radius", selector=selector, **params)
        )

    def analyze_refactor(
        self, selector: DeclarationSelector, operation: RefactorOperation, **params: Any
    ) -> RefactorAnalysis:
        return RefactorAnalysis.model_validate(
            self._call("analyze_refactor", selector=selector, operation=operation, **params)
        )

    def emit_refactor_patch(
        self, selector: DeclarationSelector, operation: RefactorOperation, **params: Any
    ) -> RefactorPatch:
        return RefactorPatch.model_validate(
            self._call("emit_refactor_patch", selector=selector, operation=operation, **params)
        )

    def model_status(self) -> ModelStatus:
        return ModelStatus.model_validate(self._call("model_status"))


def daemon_status(paths: RuntimePaths) -> dict[str, Any]:
    try:
        return {"running": True, **BrokerApplication(paths)._ping_once()}
    except CodeIndexingError as exc:
        # A daemon from another release rejects even the ping, but its error
        # names the protocol version it speaks. Report it as running with that
        # version so ensure_daemon can retire it instead of treating it as
        # absent and then timing out against the socket it still holds.
        expected = exc.details.get("expected")
        if exc.code is ErrorCode.INVALID_CONFIGURATION and isinstance(expected, int):
            return {"running": True, "protocol": expected}
        return {"running": False}
    except (EOFError, OSError):
        # EOFError is what a daemon shutting down mid-ping looks like: the
        # connect and the send both succeed, and the socket closes before the
        # reply arrives. It is not an OSError, so it used to escape a question
        # that has no failure answer -- every caller here is only asking whether
        # the daemon is up, and one that closed the connection is not.
        return {"running": False}


def _lifetime_lock_free(paths: RuntimePaths) -> bool:
    """Whether no daemon process holds the per-user lifetime lock.

    A stopping daemon closes its socket long before it exits (it drains
    requests and joins background threads first), so "the socket refuses" does
    not mean "stopped"; the lock is held for exactly as long as the process
    is still doing daemon work.
    """
    lock_path = paths.data / "locks" / "daemon.lock"
    if not lock_path.exists():
        return True
    lock = FileLock(lock_path)
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return False
    except OSError:
        return True
    lock.release()
    return True


def raise_open_file_limit(target: int = 4096) -> None:
    """Raise this process's soft descriptor limit toward *target* (DMN-8).

    macOS starts at 256, which an index run plus several clients can exhaust.
    Best-effort: a hard limit below the target just caps it.
    """
    if sys.platform == "win32":
        return
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = target if hard == resource.RLIM_INFINITY else min(hard, target)
    if soft < wanted:
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))


def stop_daemon_and_wait(
    paths: RuntimePaths,
    *,
    timeout_seconds: float = DAEMON_STOP_WAIT_SECONDS,
    poll_interval: float = 0.05,
    status: Callable[[RuntimePaths], Mapping[str, Any]] = daemon_status,
    stop: Callable[[], object] | None = None,
    initial_running: bool | None = None,
) -> bool:
    """Stop the current daemon and report whether it exited before the deadline.

    "Exited" means the socket no longer answers *and* the lifetime lock is free:
    ``stop`` is acknowledged immediately, then the daemon drains in-flight
    requests (bounded) before the process goes away, so this waits for that
    rather than for the listener to close. A ``False`` return means the daemon
    is still draining; nothing was forced.
    """
    if not daemon_supported():
        return True
    if initial_running is None:
        initial_running = bool(status(paths).get("running"))
    if initial_running:
        stopper = stop or (lambda: BrokerApplication(paths).stop())
        stopper()
    elif _lifetime_lock_free(paths):
        return True
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not status(paths)["running"] and _lifetime_lock_free(paths):
            return True
        time.sleep(poll_interval)
    return not status(paths)["running"] and _lifetime_lock_free(paths)


def _retire_stale_daemon(paths: RuntimePaths, protocol: int) -> None:
    """Ask a daemon from another release to exit so a current one can bind.

    The stale daemon rejects every frame that does not carry its own protocol
    version, so the stop request is sent speaking it; ``stop`` has existed in
    every protocol version. Best-effort: if the daemon does not go away within
    the wait, ``ensure_daemon`` reports that the old daemon is still shutting down.
    """
    broker = BrokerApplication(paths)
    with contextlib.suppress(EOFError, OSError, CodeIndexingError):
        broker._call_once("stop", _protocol=protocol)
    _wait_until_stopped(paths, DAEMON_STOP_WAIT_SECONDS)


def _wait_until_stopped(paths: RuntimePaths, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if not daemon_status(paths)["running"] and _lifetime_lock_free(paths):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _daemon_is_current(status: Mapping[str, Any]) -> bool:
    """Whether a running daemon speaks this protocol and was built from this code.

    A `build` mismatch is handled exactly like a protocol mismatch: both mean
    the daemon cannot be trusted to answer the way this process expects. A
    daemon that predates D1 has no `build` key at all, which compares unequal
    to any real `BUILD_IDENTITY` -- the one-time "missing counts as mismatch"
    upgrade path.
    """
    return (
        bool(status["running"])
        and status.get("protocol") == PROTOCOL_VERSION
        and status.get("build") == BUILD_IDENTITY
    )


def ensure_daemon(paths: RuntimePaths, *, timeout_seconds: float = 10) -> BrokerApplication:
    require_daemon_support()
    status = daemon_status(paths)
    if _daemon_is_current(status):
        broker = BrokerApplication(paths)
        broker.warn_on_settings_mismatch(status.get("settings_digest"))
        return broker
    paths.ensure_private()
    (paths.data / "locks").mkdir(parents=True, exist_ok=True)
    log_path = paths.data / "daemon.log"
    last_error: Exception | None = None
    with FileLock(paths.data / "locks" / "daemon-start.lock"):
        status = daemon_status(paths)
        if status["running"]:
            if _daemon_is_current(status):
                broker = BrokerApplication(paths)
                broker.warn_on_settings_mismatch(status.get("settings_digest"))
                return broker
            _retire_stale_daemon(paths, int(status.get("protocol") or 0))
        # DMN-3: a daemon that was stopped keeps its lifetime lock until it has
        # drained and exited. Starting a replacement before then only makes the
        # replacement wait (or fail), so wait here, bounded, and say so plainly.
        if not _wait_until_stopped(paths, DAEMON_STOP_WAIT_SECONDS):
            status = daemon_status(paths)
            if _daemon_is_current(status):
                broker = BrokerApplication(paths)
                broker.warn_on_settings_mismatch(status.get("settings_digest"))
                return broker
            raise CodeIndexingError(
                ErrorCode.DAEMON_UNAVAILABLE,
                "The previous indexing daemon is still shutting down (it holds the daemon "
                f"lock); retry shortly; see {log_path}",
                log_path=str(log_path),
                timeout_seconds=DAEMON_STOP_WAIT_SECONDS,
            )
        # Keep the child's stderr: a daemon that dies during startup is otherwise
        # indistinguishable from a slow one, and surfaces only as a timeout.
        with log_path.open("a") as log:
            process = subprocess.Popen(
                # A tiny entry module, not ``code_indexing_mcp.cli``: spawned
                # CPU embedding workers re-import the daemon's ``__main__``,
                # and importing the whole CLI/server stack there costs ~0.7 s
                # and ~115 MB per worker (DMN-5).
                [sys.executable, "-m", DAEMON_ENTRY_MODULE],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log,
                start_new_session=True,
            )
        # Nobody else waits on the child, so an exited daemon would stay a
        # zombie of this (possibly long-lived) process.
        threading.Thread(
            target=process.wait, name="code-indexing-mcp-daemon-reaper", daemon=True
        ).start()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            broker = BrokerApplication(paths)
            # Observed before the ping: a ping that fails after the child was
            # seen exiting is a daemon that is not coming up.
            exit_code = process.poll()
            try:
                ping = broker._ping_once()
                # A freshly spawned daemon inherits this very process's
                # environment, so this is normally a no-op; it still runs so a
                # daemon someone else raced into existence between the lock
                # release above and this ping is caught the same way.
                broker.warn_on_settings_mismatch(ping.get("settings_digest"))
                return broker
            except (OSError, EOFError, CodeIndexingError) as exc:
                last_error = exc
                if exit_code is not None:
                    raise CodeIndexingError(
                        ErrorCode.DAEMON_UNAVAILABLE,
                        f"The local indexing daemon exited with status {exit_code} during "
                        f"startup; last connection error: {exc}; see {log_path}",
                        log_path=str(log_path),
                        exit_code=exit_code,
                    ) from exc
                time.sleep(0.05)
    reason = f"; last connection error: {last_error}" if last_error is not None else ""
    raise CodeIndexingError(
        ErrorCode.DAEMON_UNAVAILABLE,
        f"Timed out starting the local indexing daemon{reason}; see {log_path}",
        log_path=str(log_path),
        timeout_seconds=timeout_seconds,
        connection_error=str(last_error) if last_error is not None else None,
    ) from last_error
