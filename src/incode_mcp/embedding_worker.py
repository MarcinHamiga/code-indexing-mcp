"""Memory accounting primitives for disposable embedding workers."""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import TracebackType

import numpy as np
import psutil
from fastembed import TextEmbedding

from .embedding import DEFAULT_MODEL
from .errors import ErrorCode, IncodeError

SYSTEM_RESERVE_BYTES = 512 * 1024**2
MINIMUM_WORKER_BYTES = 1024**3
HARD_OVERSHOOT_BYTES = 128 * 1024**2


@dataclass(frozen=True)
class WorkerConfig:
    cache_directory: str
    offline: bool
    threads: int
    enable_cpu_mem_arena: bool
    dimension: int
    model_id: str = DEFAULT_MODEL


WorkerTarget = Callable[[Connection, WorkerConfig], None]


def effective_memory_ceiling(*, configured_bytes: int, available_bytes: int) -> int:
    """Return the usable indexing ceiling after preserving a system reserve."""
    return min(configured_bytes, max(0, available_bytes - SYSTEM_RESERVE_BYTES))


def _worker_main(connection: Connection, config: WorkerConfig) -> None:
    try:
        Path(config.cache_directory).mkdir(parents=True, exist_ok=True)
        model = TextEmbedding(
            model_name=config.model_id,
            cache_dir=config.cache_directory,
            local_files_only=config.offline,
            threads=config.threads,
            enable_cpu_mem_arena=config.enable_cpu_mem_arena,
        )
        while True:
            command, payload = connection.recv()
            if command == "stop":
                return
            if command != "embed":
                raise ValueError(f"Unknown worker command: {command}")
            packed = [
                np.asarray(vector, dtype="<f4").tobytes() for vector in model.passage_embed(payload)
            ]
            connection.send(("packed", packed))
    except BaseException as exc:
        with suppress(BaseException):
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


class EmbeddingWorkerSession:
    """A spawned embedding process guarded by a combined-RSS ceiling."""

    def __init__(
        self,
        config: WorkerConfig,
        *,
        configured_ceiling_bytes: int | None = None,
        effective_ceiling_bytes: int | None = None,
        target: WorkerTarget = _worker_main,
    ) -> None:
        self.config = config
        configured = configured_ceiling_bytes or 2 * 1024**3
        self.effective_ceiling_bytes = (
            effective_ceiling_bytes
            if effective_ceiling_bytes is not None
            else effective_memory_ceiling(
                configured_bytes=configured,
                available_bytes=psutil.virtual_memory().available,
            )
        )
        if self.effective_ceiling_bytes < MINIMUM_WORKER_BYTES:
            raise IncodeError(
                ErrorCode.INDEX_RESOURCE_LIMIT,
                "Insufficient available memory to load the embedding model safely",
                effective_memory_bytes=self.effective_ceiling_bytes,
                minimum_memory_bytes=MINIMUM_WORKER_BYTES,
            )
        self._target = target
        self._process: BaseProcess | None = None
        self._connection: Connection | None = None
        self.peak_combined_rss = 0

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def __enter__(self) -> EmbeddingWorkerSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self._start()
        assert self._connection is not None
        assert self._process is not None
        self._connection.send(("embed", texts))
        consecutive_over = 0
        while not self._connection.poll(0.1):
            if not self._process.is_alive():
                self.close()
                raise IncodeError(
                    ErrorCode.EMBEDDING_WORKER_FAILED,
                    "Embedding worker exited without returning a result",
                )
            rss = self._combined_rss()
            self.peak_combined_rss = max(self.peak_combined_rss, rss)
            consecutive_over = consecutive_over + 1 if rss > self.effective_ceiling_bytes else 0
            if rss > self.effective_ceiling_bytes + HARD_OVERSHOOT_BYTES or consecutive_over >= 5:
                self._terminate()
                raise IncodeError(
                    ErrorCode.INDEX_RESOURCE_LIMIT,
                    "Indexing exceeded its memory ceiling",
                    effective_memory_bytes=self.effective_ceiling_bytes,
                    peak_memory_bytes=self.peak_combined_rss,
                )
        try:
            status, payload = self._connection.recv()
        except EOFError as exc:
            self.close()
            raise IncodeError(
                ErrorCode.EMBEDDING_WORKER_FAILED,
                "Embedding worker closed its result channel",
            ) from exc
        if status == "error":
            self.close()
            raise IncodeError(ErrorCode.EMBEDDING_WORKER_FAILED, str(payload))
        if status == "packed":
            return [
                np.frombuffer(vector, dtype="<f4", count=self.config.dimension).tolist()
                for vector in payload
            ]
        return [[float(value) for value in vector] for vector in payload]

    def close(self) -> None:
        process = self._process
        connection = self._connection
        if process is None:
            return
        if process.is_alive() and connection is not None:
            with suppress(BrokenPipeError, EOFError, OSError):
                connection.send(("stop", None))
            process.join(timeout=2)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        if connection is not None:
            connection.close()
        self._process = None
        self._connection = None

    def _start(self) -> None:
        if self._process is not None:
            return
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=self._target,
            args=(child, self.config),
            name="incode-embedding-worker",
            daemon=True,
        )
        process.start()
        child.close()
        self._process = process
        self._connection = parent

    def _combined_rss(self) -> int:
        assert self._process is not None
        parent_rss = psutil.Process().memory_info().rss
        try:
            worker_rss = psutil.Process(self._process.pid).memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            worker_rss = 0
        return int(parent_rss + worker_rss)

    def _terminate(self) -> None:
        assert self._process is not None
        self._process.terminate()
        deadline = time.monotonic() + 2
        while self._process.is_alive() and time.monotonic() < deadline:
            self._process.join(timeout=0.05)
        self.close()
