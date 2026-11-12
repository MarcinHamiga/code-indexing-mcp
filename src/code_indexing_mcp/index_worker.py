"""Persistent subprocess worker that executes CPU-bound indexing off the server GIL.

Indexing parses Tree-sitter ASTs, extracts structural references, builds Arrow
batches, and drives ONNX/MLX embedding inference in Python/C extensions that
hold or frequently cycle the GIL. Running that pipeline in a dedicated child
process keeps concurrent ``search_code`` and ``find_symbol`` queries on the
server or daemon responsive at near-idle latency while preserving immediate
cross-process visibility through LanceDB's ``read_consistency_interval=0``.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import threading
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .backends import Accelerator
from .errors import CodeIndexingError, ErrorCode
from .git_state import GitState
from .models import (
    IndexProgress,
    IndexReport,
    IndexTrigger,
    ProjectInfo,
    ProjectSlot,
    ReferenceBackfillReport,
)
from .settings import IndexMode, IndexSettings
from .storage import ActiveIndexTarget
from .worker_launcher import INDEX_WORKER_TARGET, ExternalInterpreterLauncher, LaunchedWorker

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .application import RuntimePaths

logger = logging.getLogger(__name__)

_SHUTDOWN_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class IndexWorkerConfig:
    """Configuration handed to an indexing worker on startup."""

    data_directory: str
    cache_directory: str
    cwd: str
    settings: dict[str, Any]
    embedder_pickle: bytes | None = None


def settings_to_payload(settings: IndexSettings) -> dict[str, Any]:
    """Serialize ``IndexSettings`` into plain primitives for the worker pipe."""

    raw = asdict(settings)
    raw["mode"] = settings.mode.value
    raw["embedding_accelerator"] = settings.embedding_accelerator.value
    return raw


def settings_from_payload(payload: Mapping[str, Any]) -> IndexSettings:
    """Reconstruct ``IndexSettings`` from ``settings_to_payload`` output."""

    values = dict(payload)
    values["mode"] = IndexMode(values["mode"])
    values["embedding_accelerator"] = Accelerator(values["embedding_accelerator"])
    return IndexSettings(**values)


def target_to_payload(target: ActiveIndexTarget) -> dict[str, Any]:
    """Serialize an ``ActiveIndexTarget`` without extra git probes on the worker."""

    return {
        "project": target.project.model_dump(mode="json"),
        "slot": target.slot.model_dump(mode="json"),
        "partition_id": target.partition_id,
        "activation_epoch": target.activation_epoch,
        "git_state": target.git_state.model_dump(mode="json"),
    }


def target_from_payload(payload: Mapping[str, Any]) -> ActiveIndexTarget:
    """Reconstruct an ``ActiveIndexTarget`` sent from the parent process."""

    return ActiveIndexTarget(
        project=ProjectInfo.model_validate(payload["project"]),
        slot=ProjectSlot.model_validate(payload["slot"]),
        partition_id=str(payload["partition_id"]),
        activation_epoch=int(payload["activation_epoch"]),
        git_state=GitState.model_validate(payload["git_state"]),
    )


def _json_safe_details(details: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in details.items():
        if isinstance(value, Path):
            normalized[key] = str(value)
        else:
            normalized[key] = value
    return normalized


def _error_from_payload(payload: Mapping[str, Any]) -> Exception:
    if payload.get("kind") == "code_indexing":
        details = dict(payload.get("details") or {})
        return CodeIndexingError(
            ErrorCode(str(payload["code"])),
            str(payload["message"]),
            **details,
        )
    return RuntimeError(str(payload.get("message") or "Indexing worker failed"))


def _worker_pythonpath_extension(embedder: object | None) -> dict[str, str]:
    """Include the module root when tests pickle a non-installed test double."""

    if embedder is None:
        return {}
    module_name = type(embedder).__module__
    if module_name.startswith("code_indexing_mcp.") or module_name == "code_indexing_mcp":
        return {}
    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None) if module is not None else None
    if not module_file:
        return {}
    package_depth = module_name.count(".")
    import_root = Path(module_file).resolve().parents[package_depth]
    existing = os.environ.get("PYTHONPATH")
    combined = f"{import_root}{os.pathsep}{existing}" if existing else str(import_root)
    return {"PYTHONPATH": combined}


class IndexWorkerSession:
    """Manages one lazy, persistent indexing worker subprocess per ``Application``."""

    def __init__(
        self,
        paths: RuntimePaths,
        cwd: Path,
        settings: IndexSettings,
        *,
        embedder_pickle: bytes | None = None,
        extra_environment: Mapping[str, str] | None = None,
        on_worker_crash: Callable[[], None] | None = None,
    ) -> None:
        self._config = IndexWorkerConfig(
            data_directory=str(paths.data),
            cache_directory=str(paths.cache),
            cwd=str(cwd),
            settings=settings_to_payload(settings),
            embedder_pickle=embedder_pickle,
        )
        self._extra_environment = dict(extra_environment or {})
        self._on_worker_crash = on_worker_crash
        self._lock = threading.Lock()
        self._launched: LaunchedWorker | None = None
        self._closed = False
        self._active_project_id: str | None = None
        self._next_request_id = 1

    @property
    def active_project_id(self) -> str | None:
        """Return the project ID currently executing in the worker, if any."""

        return self._active_project_id

    @property
    def pid(self) -> int | None:
        """Return the child PID while the worker is alive, else ``None``."""

        launched = self._launched
        if launched is None or not launched.process.is_alive():
            return None
        return launched.process.pid

    def index(
        self,
        target: ActiveIndexTarget,
        *,
        force: bool = False,
        wait_for_lock: bool = False,
        on_progress: Callable[[IndexProgress], None] | None = None,
        trigger: IndexTrigger = "manual",
    ) -> IndexReport:
        """Execute ``Indexer.index`` in the persistent worker process."""

        payload = self._request(
            "index",
            {
                "target": target_to_payload(target),
                "force": force,
                "wait_for_lock": wait_for_lock,
                "stream_progress": on_progress is not None,
                "trigger": trigger,
            },
            wait_for_lock=wait_for_lock,
            busy_project=target.project,
            on_progress=on_progress,
        )
        return IndexReport.model_validate(payload)

    def backfill_references(
        self,
        target: ActiveIndexTarget,
        *,
        wait_for_lock: bool = False,
        lock_timeout: float | None = None,
        trigger: IndexTrigger = "reference-backfill",
    ) -> ReferenceBackfillReport:
        """Execute ``Indexer.backfill_references`` in the persistent worker."""

        payload = self._request(
            "backfill_references",
            {
                "target": target_to_payload(target),
                "wait_for_lock": wait_for_lock,
                "lock_timeout": lock_timeout,
                "trigger": trigger,
            },
            wait_for_lock=wait_for_lock,
            lock_timeout=lock_timeout,
            busy_project=target.project,
        )
        return ReferenceBackfillReport.model_validate(payload)

    def ping(self) -> int:
        """Return the worker PID, spawning the worker if not yet running."""

        return int(self._request("ping", None))

    def close(self) -> None:
        """Ask the worker to exit cleanly and reap the child process."""

        with self._lock:
            self._closed = True
            self._teardown_locked()

    def _ensure_started_locked(self) -> LaunchedWorker:
        if self._closed:
            raise RuntimeError("IndexWorkerSession has been closed")
        launched = self._launched
        if launched is not None and launched.process.is_alive():
            return launched
        self._drop_dead_locked()
        launcher = ExternalInterpreterLauncher(
            Path(sys.executable),
            environment_name="indexing worker",
            target=INDEX_WORKER_TARGET,
            extra_environment=self._extra_environment,
        )
        launched = launcher.launch(self._config)
        self._launched = launched
        return launched

    def _drop_dead_locked(self) -> None:
        launched = self._launched
        if launched is None:
            return
        self._launched = None
        with suppress(OSError):
            launched.connection.close()
        with suppress(Exception):
            launched.process.join(timeout=0.5)

    def _teardown_locked(self) -> None:
        launched = self._launched
        if launched is None:
            return
        self._launched = None
        try:
            if launched.process.is_alive():
                with suppress(OSError, EOFError, BrokenPipeError):
                    launched.connection.send((0, "shutdown", None))
                    if launched.connection.poll(_SHUTDOWN_TIMEOUT_SECONDS):
                        launched.connection.recv()
                launched.process.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            if launched.process.is_alive():
                launched.process.terminate()
                launched.process.join(timeout=2.0)
            if launched.process.is_alive():
                launched.process.kill()
                launched.process.join(timeout=1.0)
        finally:
            with suppress(OSError):
                launched.connection.close()

    def _request(
        self,
        command: str,
        payload: Any,
        *,
        wait_for_lock: bool = True,
        lock_timeout: float | None = None,
        busy_project: ProjectInfo | None = None,
        on_progress: Callable[[IndexProgress], None] | None = None,
    ) -> Any:
        if not wait_for_lock:
            acquired = self._lock.acquire(blocking=False)
        elif lock_timeout is not None:
            acquired = self._lock.acquire(blocking=True, timeout=max(0.0, lock_timeout))
        else:
            acquired = self._lock.acquire(blocking=True)
        if not acquired:
            raise CodeIndexingError(
                ErrorCode.INDEX_BUSY,
                "Another indexing job is already active: "
                f"{busy_project.name if busy_project is not None else 'indexing worker'}",
                **({"project": busy_project.id} if busy_project is not None else {}),
            )
        previous_active_project = self._active_project_id
        if command in {"index", "backfill_references"} and busy_project is not None:
            self._active_project_id = busy_project.id
        try:
            launched = self._ensure_started_locked()
            request_id = self._next_request_id
            self._next_request_id += 1
            completed = False
            try:
                launched.connection.send((request_id, command, payload))
                while True:
                    resp_id, kind, response = launched.connection.recv()
                    if resp_id != request_id:
                        raise RuntimeError(
                            f"Mismatched index worker response id {resp_id!r} "
                            f"(expected {request_id!r}, kind={kind!r})"
                        )
                    if kind == "progress":
                        if on_progress is not None:
                            on_progress(IndexProgress.model_validate(response))
                        continue
                    if kind in {"result", "pong"}:
                        completed = True
                        return response
                    if kind == "error":
                        completed = True
                        raise _error_from_payload(response)
                    raise RuntimeError(f"Unexpected index worker response {kind!r}")
            except (EOFError, BrokenPipeError, OSError) as exc:
                raise CodeIndexingError(
                    ErrorCode.INTERNAL_ERROR,
                    f"The indexing worker exited unexpectedly during {command}: {exc}",
                ) from exc
            finally:
                if not completed:
                    self._teardown_locked()
                    if self._on_worker_crash is not None:
                        self._on_worker_crash()
        finally:
            self._active_project_id = previous_active_project
            self._lock.release()


def worker_main(connection: Any, raw_config: Mapping[str, Any]) -> None:
    """Serve indexing and reference-backfill commands until shutdown or EOF."""

    from .application import Application, RuntimePaths

    paths = RuntimePaths(
        data=Path(str(raw_config["data_directory"])),
        cache=Path(str(raw_config["cache_directory"])),
    )
    settings = settings_from_payload(raw_config["settings"])
    embedder_pickle = raw_config.get("embedder_pickle")
    embedder = pickle.loads(embedder_pickle) if embedder_pickle is not None else None
    app = Application(
        paths,
        embedder=embedder,
        cwd=Path(str(raw_config["cwd"])),
        settings=settings,
        _enable_index_worker=False,
        _recover_on_start=False,
    )
    try:
        while True:
            try:
                request_id, command, payload = connection.recv()
            except EOFError:
                break
            if command == "shutdown":
                with suppress(OSError):
                    connection.send((request_id, "stopped", None))
                break
            if command == "ping":
                connection.send((request_id, "pong", os.getpid()))
                continue
            try:
                if command == "index":
                    target = target_from_payload(payload["target"])
                    progress_callback: Callable[[IndexProgress], None] | None = None
                    if payload.get("stream_progress"):

                        def emit(progress: IndexProgress, *, _req_id: int = request_id) -> None:
                            connection.send((_req_id, "progress", progress.model_dump(mode="json")))

                        progress_callback = emit
                    report = app.indexer.index(
                        target.project,
                        target=target,
                        force=bool(payload["force"]),
                        wait_for_lock=bool(payload["wait_for_lock"]),
                        on_progress=progress_callback,
                        trigger=payload["trigger"],
                    )
                    connection.send((request_id, "result", report.model_dump(mode="json")))
                elif command == "backfill_references":
                    target = target_from_payload(payload["target"])
                    backfill = app.indexer.backfill_references(
                        target.project,
                        target=target,
                        wait_for_lock=bool(payload["wait_for_lock"]),
                        lock_timeout=payload["lock_timeout"],
                        trigger=payload["trigger"],
                    )
                    connection.send((request_id, "result", backfill.model_dump(mode="json")))
                else:
                    raise ValueError(f"Unknown index worker command: {command!r}")
            except CodeIndexingError as exc:
                connection.send(
                    (
                        request_id,
                        "error",
                        {
                            "kind": "code_indexing",
                            "code": exc.code.value,
                            "message": str(exc.args[0]) if exc.args else str(exc),
                            "details": _json_safe_details(exc.details),
                        },
                    )
                )
            except Exception as exc:
                logger.exception("Indexing worker command %s failed", command)
                connection.send(
                    (
                        request_id,
                        "error",
                        {
                            "kind": "exception",
                            "message": f"{type(exc).__name__}: {exc}",
                        },
                    )
                )
    finally:
        app.close()
