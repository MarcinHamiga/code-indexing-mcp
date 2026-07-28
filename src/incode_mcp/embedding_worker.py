"""Memory accounting primitives for disposable embedding workers."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

import numpy as np
import psutil
from fastembed import TextEmbedding

from .backends import CPU_PROVIDER, Accelerator
from .embedding import (
    DEFAULT_MODEL,
    PROBE_TEXTS,
    EmbeddedSegment,
    PassageCandidate,
    SegmentPlan,
    embed_windows,
    plan_passages,
    resolve_session_providers,
    resolve_tokenizer,
    validate_probe_vectors,
)
from .errors import ErrorCode, IncodeError

SYSTEM_RESERVE_BYTES = 512 * 1024**2
MINIMUM_WORKER_BYTES = 1024**3
HARD_OVERSHOOT_BYTES = 128 * 1024**2
# Failures a smaller microbatch can plausibly survive. Model, protocol, and
# validation errors are not retried: they fail identically at any batch size.
RETRYABLE_CODES = frozenset({ErrorCode.INDEX_RESOURCE_LIMIT, ErrorCode.EMBEDDING_WORKER_FAILED})
MAX_BATCH_RETRIES = 2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerConfig:
    cache_directory: str
    offline: bool
    threads: int
    enable_cpu_mem_arena: bool
    dimension: int
    model_id: str = DEFAULT_MODEL
    # The execution providers to request, most specific first. Empty means "let
    # the runtime pick", which is the historical CPU behaviour and is kept
    # byte-identical by never passing a providers argument in that case.
    providers: tuple[str, ...] = ()
    accelerator: str = Accelerator.CPU.value

    @property
    def is_cpu(self) -> bool:
        return not self.providers or tuple(self.providers) == (CPU_PROVIDER,)


@dataclass(frozen=True)
class WorkerInfo:
    """What a worker reports about the session it actually loaded."""

    resolved_providers: tuple[str, ...]
    dimension: int


@dataclass(frozen=True)
class SessionTelemetry:
    """Per-run embedding facts an ``IndexReport`` carries back to the caller."""

    backend: str
    memory_budget_bytes: int
    peak_memory_bytes: int
    segment_count: int
    token_count: int
    retry_count: int
    fallback_count: int
    termination_reason: str | None
    tokenizer_available: bool | None
    fallback_reason: str | None = None


@runtime_checkable
class TelemetrySource(Protocol):
    """Any passage session that can describe the run it just served."""

    def telemetry(self) -> SessionTelemetry: ...


WorkerTarget = Callable[[Connection, WorkerConfig], None]


def effective_memory_ceiling(*, configured_bytes: int, available_bytes: int) -> int:
    """Return the usable indexing ceiling after preserving a system reserve."""
    return min(configured_bytes, max(0, available_bytes - SYSTEM_RESERVE_BYTES))


def indexing_memory_bytes(
    *, parent_bytes: int, worker_bytes: int, parent_baseline_bytes: int
) -> int:
    """Return the memory attributable to indexing.

    The parent may already hold a query model and open Lance datasets before any
    indexing starts, and charging that resident footprint to the indexing budget
    would trip the ceiling before the first batch runs. Only the worker plus
    parent growth since the worker started counts.
    """
    return worker_bytes + max(0, parent_bytes - parent_baseline_bytes)


def _load_model(config: WorkerConfig) -> TextEmbedding:
    """Load the model for *config*, requesting its providers when non-default.

    The CPU path deliberately passes no ``providers`` argument at all. Naming
    the CPU provider explicitly would be equivalent in principle, but the CPU
    result is the reference every accelerator is compared against, so its call
    is left exactly as it was.
    """
    Path(config.cache_directory).mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {
        "model_name": config.model_id,
        "cache_dir": config.cache_directory,
        "local_files_only": config.offline,
        "threads": config.threads,
        "enable_cpu_mem_arena": config.enable_cpu_mem_arena,
    }
    if not config.is_cpu:
        options["providers"] = list(config.providers)
    return TextEmbedding(**options)


def _worker_main(connection: Connection, config: WorkerConfig) -> None:
    try:
        model = _load_model(config)
        tokenizer = resolve_tokenizer(model)

        def embed_packed(texts: list[str]) -> list[bytes]:
            return [
                np.asarray(vector, dtype="<f4").tobytes() for vector in model.passage_embed(texts)
            ]

        while True:
            command, payload = connection.recv()
            if command == "stop":
                return
            if command == "initialize":
                # Reaching here already proves the model loaded and, for an
                # accelerator, that its provider initialised. Report what the
                # session settled on rather than what was requested.
                connection.send(
                    ("initialized", (resolve_session_providers(model), config.dimension))
                )
                continue
            if command == "memory":
                connection.send(("memory", psutil.Process().memory_info().rss))
                continue
            if command == "probe":
                connection.send(("probed", embed_packed(list(PROBE_TEXTS))))
                continue
            if command == "embed":
                connection.send(("packed", embed_packed(payload)))
                continue
            if command != "plan_and_embed":
                raise ValueError(f"Unknown worker command: {command}")
            raw_candidates, plan = payload
            candidates = [PassageCandidate(prefix, content) for prefix, content in raw_candidates]
            try:
                windows = plan_passages(
                    None if tokenizer is None else tokenizer.encode, candidates, plan
                )
            except ValueError as exc:
                # A file the planner cannot window is a bad file, not a broken
                # environment. Reported separately so the parent charges it to
                # the file instead of aborting every remaining file in the run.
                connection.send(("plan_error", str(exc)))
                continue
            planned = embed_windows(embed_packed, candidates, windows, plan)
            connection.send(
                (
                    "planned",
                    (
                        [
                            [
                                (window.start_char, window.end_char, window.token_count, vector)
                                for window, vector in segments
                            ]
                            for segments in planned
                        ],
                        tokenizer is not None,
                    ),
                )
            )
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
        # Telemetry surfaced on IndexReport so a run's shape is diagnosable
        # without re-running it under a profiler.
        self.retry_count = 0
        self.segment_count = 0
        self.token_count = 0
        self.termination_reason: str | None = None
        self.tokenizer_available: bool | None = None
        # RSS the parent already held before the worker existed. The daemon keeps
        # a query model resident in-process, and charging that to the indexing
        # budget would trip the ceiling before any indexing work happens. Only
        # parent growth during indexing counts against the budget.
        self._parent_baseline_bytes = 0

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

    def initialize(self) -> WorkerInfo:
        """Spawn the worker and load its model, reporting what it resolved.

        Separated from the first embed so a backend that cannot load, or whose
        provider fails to initialise, is diagnosed before any real content is
        handed to it -- and so the caller can terminate it and pick another.
        """
        status, payload = self._request("initialize", None)
        if status != "initialized":
            raise IncodeError(
                ErrorCode.EMBEDDING_WORKER_FAILED,
                f"Embedding worker answered initialize with {status!r}",
            )
        providers, dimension = payload
        return WorkerInfo(
            resolved_providers=tuple(str(name) for name in providers), dimension=int(dimension)
        )

    def probe(self) -> list[bytes]:
        """Run a minimum-batch inference and validate the vectors it returns.

        Raises ``IncodeError`` when the worker fails outright and ``ValueError``
        when it answers with vectors an index could not use.
        """
        status, payload = self._request("probe", None)
        if status != "probed":
            raise IncodeError(
                ErrorCode.EMBEDDING_WORKER_FAILED,
                f"Embedding worker answered probe with {status!r}",
            )
        vectors = [bytes(vector) for vector in payload]
        validate_probe_vectors(vectors, dimension=self.config.dimension, count=len(PROBE_TEXTS))
        return vectors

    def report_memory(self) -> int:
        """Return the worker's own resident set size in bytes."""
        status, payload = self._request("memory", None)
        if status != "memory":
            raise IncodeError(
                ErrorCode.EMBEDDING_WORKER_FAILED,
                f"Embedding worker answered memory with {status!r}",
            )
        return int(payload)

    def telemetry(self) -> SessionTelemetry:
        return SessionTelemetry(
            backend=self.config.accelerator,
            memory_budget_bytes=self.effective_ceiling_bytes,
            peak_memory_bytes=self.peak_combined_rss,
            segment_count=self.segment_count,
            token_count=self.token_count,
            retry_count=self.retry_count,
            fallback_count=self.retry_count,
            termination_reason=self.termination_reason,
            tokenizer_available=self.tokenizer_available,
        )

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        status, payload = self._request("embed", texts)
        if status == "packed":
            return [
                np.frombuffer(vector, dtype="<f4", count=self.config.dimension).tolist()
                for vector in payload
            ]
        return [[float(value) for value in vector] for vector in payload]

    def plan_and_embed(
        self, candidates: Sequence[PassageCandidate], plan: SegmentPlan
    ) -> list[list[EmbeddedSegment]]:
        """Window candidates by token count in the worker, retrying smaller.

        Window boundaries are a pure function of the tokenization, so a retry
        re-derives the identical segments; only the microbatch packing shrinks.
        """
        request = [(candidate.prefix, candidate.content) for candidate in candidates]
        attempt = plan
        for retry in range(MAX_BATCH_RETRIES + 1):
            try:
                status, payload = self._request("plan_and_embed", (request, attempt))
                if status == "plan_error":
                    raise ValueError(str(payload))
                break
            except IncodeError as exc:
                if (
                    exc.code not in RETRYABLE_CODES
                    or attempt.max_items <= 1
                    or retry == MAX_BATCH_RETRIES
                ):
                    # _request already recorded the specific reason; keep it
                    # rather than flattening it back to the error code.
                    raise
                attempt = replace(attempt, max_items=max(1, attempt.max_items // 2))
                self.retry_count += 1
                logger.warning(
                    "Embedding batch failed with %s; retrying with max_items=%d",
                    exc.code.value,
                    attempt.max_items,
                )
                # _request already terminated the worker, so the next attempt
                # spawns a fresh process with a fresh ONNX arena.
                self.close()

        segments_payload, tokenizer_available = payload
        self.tokenizer_available = bool(tokenizer_available)
        results: list[list[EmbeddedSegment]] = []
        for segments in segments_payload:
            decoded = [
                EmbeddedSegment(
                    start_char=start_char,
                    end_char=end_char,
                    token_count=token_count,
                    # Already packed little-endian float32 on the wire; keep it
                    # packed so staging never builds a list of Python floats.
                    vector=vector,
                )
                for start_char, end_char, token_count, vector in segments
            ]
            self.segment_count += len(decoded)
            self.token_count += sum(segment.token_count for segment in decoded)
            results.append(decoded)
        return results

    def _request(self, command: str, payload: object) -> tuple[str, Any]:
        """Send one command and wait for its reply under the memory ceiling."""
        self._start()
        assert self._connection is not None
        assert self._process is not None
        self._connection.send((command, payload))
        consecutive_over = 0
        while not self._connection.poll(0.1):
            if not self._process.is_alive():
                self.close()
                self.termination_reason = "worker_exited"
                raise IncodeError(
                    ErrorCode.EMBEDDING_WORKER_FAILED,
                    "Embedding worker exited without returning a result",
                )
            parent_rss, worker_rss = self._sample_rss()
            self.peak_combined_rss = max(self.peak_combined_rss, parent_rss + worker_rss)
            budgeted = indexing_memory_bytes(
                parent_bytes=parent_rss,
                worker_bytes=worker_rss,
                parent_baseline_bytes=self._parent_baseline_bytes,
            )
            consecutive_over = (
                consecutive_over + 1 if budgeted > self.effective_ceiling_bytes else 0
            )
            if (
                budgeted > self.effective_ceiling_bytes + HARD_OVERSHOOT_BYTES
                or consecutive_over >= 5
            ):
                self._terminate()
                self.termination_reason = "memory_ceiling"
                raise IncodeError(
                    ErrorCode.INDEX_RESOURCE_LIMIT,
                    "Indexing exceeded its memory ceiling",
                    effective_memory_bytes=self.effective_ceiling_bytes,
                    indexing_memory_bytes=budgeted,
                    peak_memory_bytes=self.peak_combined_rss,
                    parent_baseline_bytes=self._parent_baseline_bytes,
                )
        try:
            status, payload = self._connection.recv()
        except EOFError as exc:
            self.close()
            self.termination_reason = "channel_closed"
            raise IncodeError(
                ErrorCode.EMBEDDING_WORKER_FAILED,
                "Embedding worker closed its result channel",
            ) from exc
        if status == "error":
            self.close()
            self.termination_reason = "worker_error"
            raise IncodeError(ErrorCode.EMBEDDING_WORKER_FAILED, str(payload))
        return status, payload

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
        self._parent_baseline_bytes = psutil.Process().memory_info().rss
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
        # Pipe() yields PipeConnection on Windows and Connection elsewhere, and
        # typeshed does not relate the two even though both carry the send/recv/
        # poll/close surface used here. Widening on this one line keeps the
        # attribute itself typed, and keeps a cast from being redundant on POSIX.
        connection: Any = parent
        self._connection = connection

    def _sample_rss(self) -> tuple[int, int]:
        """Return the current (parent, worker) resident set sizes in bytes."""
        assert self._process is not None
        parent_rss = psutil.Process().memory_info().rss
        try:
            worker_rss = psutil.Process(self._process.pid).memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            worker_rss = 0
        return int(parent_rss), int(worker_rss)

    def _terminate(self) -> None:
        assert self._process is not None
        self._process.terminate()
        deadline = time.monotonic() + 2
        while self._process.is_alive() and time.monotonic() < deadline:
            self._process.join(timeout=0.05)
        self.close()
