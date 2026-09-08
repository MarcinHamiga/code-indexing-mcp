"""Memory accounting primitives for disposable embedding workers."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import psutil

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
from .errors import CodeIndexingError, ErrorCode
from .memory_budget import parent_memory_baseline
from .worker_launcher import SpawnLauncher, WorkerLauncher, WorkerProcess

SYSTEM_RESERVE_BYTES = 512 * 1024**2
MINIMUM_WORKER_BYTES = 1024**3
HARD_OVERSHOOT_BYTES = 128 * 1024**2
# Failures a smaller microbatch can plausibly survive. Model, protocol, and
# validation errors are not retried: they fail identically at any batch size.
RETRYABLE_CODES = frozenset({ErrorCode.INDEX_RESOURCE_LIMIT, ErrorCode.EMBEDDING_WORKER_FAILED})
MAX_BATCH_RETRIES = 2
WORKER_REQUEST_TIMEOUT_SECONDS = 120.0
WorkerStatus = Literal[
    "initialized", "memory", "probed", "packed", "planned", "plan_error", "error", "ok"
]
KNOWN_WORKER_STATUSES = frozenset(
    {"initialized", "memory", "probed", "packed", "planned", "plan_error", "error", "ok"}
)

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
    # Candidate characters this run embedded, the crossover it was decided
    # against, and why it ran where it ran. Populated by the passage session,
    # which is what makes that decision; a worker on its own has no say in it.
    character_count: int = 0
    # None when the accelerator never overtakes CPU, so there is no size to
    # report rather than a very large one.
    crossover_characters: int | None = 0
    selection_reason: str | None = None
    worker_used: bool = False


@dataclass(frozen=True)
class WorkerTelemetrySnapshot:
    """A typed snapshot of request telemetry owned by one worker session."""

    segment_count: int
    token_count: int
    retry_count: int
    peak_combined_rss: int
    termination_reason: str | None


@dataclass
class WorkerTelemetry:
    segment_count: int = 0
    token_count: int = 0
    retry_count: int = 0
    peak_combined_rss: int = 0
    termination_reason: str | None = None

    def snapshot(self) -> WorkerTelemetrySnapshot:
        return WorkerTelemetrySnapshot(
            segment_count=self.segment_count,
            token_count=self.token_count,
            retry_count=self.retry_count,
            peak_combined_rss=self.peak_combined_rss,
            termination_reason=self.termination_reason,
        )

    def restore(self, snapshot: WorkerTelemetrySnapshot) -> None:
        self.segment_count = snapshot.segment_count
        self.token_count = snapshot.token_count
        self.retry_count = snapshot.retry_count
        self.peak_combined_rss = snapshot.peak_combined_rss
        self.termination_reason = snapshot.termination_reason


@runtime_checkable
class TelemetrySource(Protocol):
    """Any passage session that can describe the run it just served."""

    def telemetry(self) -> SessionTelemetry: ...


WorkerTarget = Callable[[Connection, WorkerConfig], None]


def default_launcher() -> WorkerLauncher:
    """Return the launcher that runs a worker in this interpreter's environment.

    This is the CPU path and the fallback path, so it deliberately depends on
    nothing an installer had to prepare.
    """
    return SpawnLauncher(_worker_main)


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


def _load_model(config: WorkerConfig) -> Any:
    """Load the model for *config*, requesting its providers when non-default.

    The CPU path deliberately passes no ``providers`` argument at all. Naming
    the CPU provider explicitly would be equivalent in principle, but the CPU
    result is the reference every accelerator is compared against, so its call
    is left exactly as it was.
    """
    Path(config.cache_directory).mkdir(parents=True, exist_ok=True)
    if config.accelerator == Accelerator.MLX.value:
        from .mlx_backend import MlxEmbedding

        # No providers, threads, or arena: MLX has no ONNX session to configure,
        # and passing settings it ignores would imply they were honoured.
        return MlxEmbedding(
            cache_directory=Path(config.cache_directory),
            offline=config.offline,
            model_id=config.model_id,
        )
    if config.accelerator in {
        Accelerator.WEBGPU.value,
        Accelerator.MIGRAPHX.value,
    }:
        from .direct_onnx import DirectOnnxEmbedding

        return DirectOnnxEmbedding(
            cache_directory=Path(config.cache_directory),
            offline=config.offline,
            threads=config.threads,
            enable_cpu_mem_arena=config.enable_cpu_mem_arena,
            providers=config.providers,
            model_id=config.model_id,
            accelerator=config.accelerator,
        )
    from fastembed import TextEmbedding

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
            planned = embed_windows(
                embed_packed,
                candidates,
                windows,
                plan,
                encode=None if tokenizer is None else tokenizer.encode,
            )
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
        launcher: WorkerLauncher | None = None,
        request_timeout_seconds: float = WORKER_REQUEST_TIMEOUT_SECONDS,
        model_load_timeout_seconds: float | None = None,
        inference_timeout_seconds: float | None = None,
    ) -> None:
        self.config = config
        self.request_timeout_seconds = request_timeout_seconds
        self.model_load_timeout_seconds = model_load_timeout_seconds
        self.inference_timeout_seconds = inference_timeout_seconds
        self._initialize_before_passages = target is _worker_main
        self._initialized = False
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
            raise CodeIndexingError(
                ErrorCode.INDEX_RESOURCE_LIMIT,
                "Insufficient available memory to load the embedding model safely",
                effective_memory_bytes=self.effective_ceiling_bytes,
                minimum_memory_bytes=MINIMUM_WORKER_BYTES,
            )
        # A launcher decides which environment the worker's code runs in;
        # ``target`` names the body it runs there. The default pair is the
        # historical behaviour: this interpreter, this module's worker loop.
        self._launcher = launcher if launcher is not None else SpawnLauncher(target)
        self._process: WorkerProcess | None = None
        self._connection: Connection | None = None
        self._transfer: threading.Thread | None = None
        # How many worker processes this session has started. A batch retry
        # closes the worker and the next request silently spawns another, so a
        # caller that verified a backend needs this to notice that the process
        # it verified is not the process now serving it.
        self.spawn_count = 0
        self._telemetry = WorkerTelemetry()
        # Telemetry surfaced on IndexReport so a run's shape is diagnosable
        # without re-running it under a profiler.
        # How long spawning the worker and loading its model took, and the
        # microbatch size a retry found to be survivable. Together they are what
        # tells a later run whether starting this backend repays the wait and
        # where to start its batches; 0 means neither has been established.
        self.load_duration_ns = 0
        self.safe_max_items = 0
        self.tokenizer_available: bool | None = None
        # RSS the parent already held before the worker existed. The daemon keeps
        # a query model resident in-process, and charging that to the indexing
        # budget would trip the ceiling before any indexing work happens. Only
        # parent growth during indexing counts against the budget.
        self._parent_baseline_bytes = 0

    @property
    def peak_combined_rss(self) -> int:
        return self._telemetry.peak_combined_rss

    @peak_combined_rss.setter
    def peak_combined_rss(self, value: int) -> None:
        self._telemetry.peak_combined_rss = value

    @property
    def retry_count(self) -> int:
        return self._telemetry.retry_count

    @retry_count.setter
    def retry_count(self, value: int) -> None:
        self._telemetry.retry_count = value

    @property
    def segment_count(self) -> int:
        return self._telemetry.segment_count

    @segment_count.setter
    def segment_count(self, value: int) -> None:
        self._telemetry.segment_count = value

    @property
    def token_count(self) -> int:
        return self._telemetry.token_count

    @token_count.setter
    def token_count(self, value: int) -> None:
        self._telemetry.token_count = value

    @property
    def termination_reason(self) -> str | None:
        return self._telemetry.termination_reason

    @termination_reason.setter
    def termination_reason(self, value: str | None) -> None:
        self._telemetry.termination_reason = value

    def telemetry_snapshot(self) -> WorkerTelemetrySnapshot:
        """Return request telemetry without exposing the mutable accumulator."""
        return self._telemetry.snapshot()

    def restore_telemetry(self, snapshot: WorkerTelemetrySnapshot) -> None:
        self._telemetry.restore(snapshot)

    @contextmanager
    def measurement_scope(self) -> Iterator[None]:
        """Exclude calibration requests from the run telemetry."""
        snapshot = self.telemetry_snapshot()
        try:
            yield
        finally:
            self.restore_telemetry(snapshot)

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
        # Measured around the request rather than around the spawn alone: the
        # spawn returns as soon as the process exists, and what a run actually
        # waits for is the model being on the device and answering.
        started = time.monotonic_ns()
        status, payload = self._request("initialize", None)
        self.load_duration_ns = time.monotonic_ns() - started
        if status != "initialized":
            raise CodeIndexingError(
                ErrorCode.EMBEDDING_WORKER_FAILED,
                f"Embedding worker answered initialize with {status!r}",
            )
        self._initialized = True
        providers, dimension = payload
        return WorkerInfo(
            resolved_providers=tuple(str(name) for name in providers), dimension=int(dimension)
        )

    def probe(self) -> list[bytes]:
        """Run a minimum-batch inference and validate the vectors it returns.

        Raises ``CodeIndexingError`` when the worker fails outright and ``ValueError``
        when it answers with vectors an index could not use.
        """
        status, payload = self._request("probe", None)
        if status != "probed":
            raise CodeIndexingError(
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
            raise CodeIndexingError(
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
            worker_used=self.spawn_count > 0,
        )

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        status, payload = self._request("embed", texts)
        if status == "packed":
            return self._decode_packed_vectors(payload, count=len(texts))
        if status == "ok":
            return self._decode_float_vectors(payload, count=len(texts))
        raise self._protocol_error(f"worker answered embed with {status!r}")

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
                if status != "planned":
                    raise self._protocol_error(f"worker answered plan_and_embed with {status!r}")
                if retry:
                    # This size survived what the requested one did not. Kept so
                    # the limit is carried into the cache rather than being
                    # rediscovered by overrunning the ceiling on the next run.
                    self.safe_max_items = attempt.max_items
                break
            except CodeIndexingError as exc:
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

        segments_payload, tokenizer_available = self._decode_planned_payload(
            payload, candidate_count=len(candidates)
        )
        self.tokenizer_available = tokenizer_available
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
        if command != "initialize" and self._initialize_before_passages and not self._initialized:
            # The production protocol confirms model loading with a small request
            # before any passages can fill the channel. Injected worker targets
            # may implement their own protocol; their I/O is still supervised.
            self.initialize()
        connection = self._connection
        process = self._process
        completed = threading.Event()
        replies: list[tuple[str, Any]] = []
        failures: list[BaseException] = []

        def exchange() -> None:
            try:
                connection.send((command, payload))
                while not connection.poll(0.05):
                    if not process.is_alive():
                        raise EOFError("worker exited")
                # poll only guarantees that some bytes arrived, not a whole
                # frame. The complete read belongs inside supervision too.
                replies.append(connection.recv())
            except BaseException as exc:
                failures.append(exc)
            finally:
                completed.set()

        transfer = threading.Thread(target=exchange, daemon=True, name="embedding-worker-io")
        request_started = time.monotonic()
        timeout_seconds = self._request_timeout(command)
        deadline = request_started + timeout_seconds
        self._transfer = transfer
        transfer.start()
        consecutive_over = 0
        while True:
            reply_ready = completed.wait(0.05)
            # Sampled and enforced on every pass, including the one that found
            # the reply ready. Lazy worker imports made startup fast enough
            # that a prompt worker can answer inside the first poll every time,
            # so waiting-only enforcement would never sample it and the ceiling
            # would not exist for exactly the workers most likely to blow it.
            # A result this discards is re-embedded by the run-level fallback.
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
                raise CodeIndexingError(
                    ErrorCode.INDEX_RESOURCE_LIMIT,
                    "Indexing exceeded its memory ceiling",
                    effective_memory_bytes=self.effective_ceiling_bytes,
                    indexing_memory_bytes=budgeted,
                    peak_memory_bytes=self.peak_combined_rss,
                    parent_baseline_bytes=self._parent_baseline_bytes,
                )
            if reply_ready:
                break
            if time.monotonic() >= deadline:
                self._terminate()
                self.termination_reason = "worker_timeout"
                raise CodeIndexingError(
                    ErrorCode.EMBEDDING_WORKER_FAILED,
                    "Embedding worker exceeded its response deadline",
                    timeout_seconds=timeout_seconds,
                    command=command,
                    duration_seconds=time.monotonic() - request_started,
                )
        transfer.join()
        self._transfer = None
        if failures:
            raise self._channel_failed() from failures[0]
        reply = replies[0]
        if not isinstance(reply, tuple) or len(reply) != 2:
            raise self._protocol_error("worker returned a malformed reply envelope")
        status, payload = reply
        if not isinstance(status, str) or status not in KNOWN_WORKER_STATUSES:
            raise self._protocol_error(f"worker returned unknown reply status {status!r}")
        if status == "error":
            self.close()
            self.termination_reason = "worker_error"
            raise CodeIndexingError(ErrorCode.EMBEDDING_WORKER_FAILED, str(payload))
        return status, payload

    def _protocol_error(self, detail: str) -> CodeIndexingError:
        self.close()
        self.termination_reason = "worker_protocol_error"
        return CodeIndexingError(ErrorCode.EMBEDDING_WORKER_FAILED, detail)

    def _decode_packed_vectors(self, payload: object, *, count: int) -> list[list[float]]:
        if not isinstance(payload, (list, tuple)) or len(payload) != count:
            raise self._protocol_error(f"worker returned the wrong vector count for {count} inputs")
        decoded: list[list[float]] = []
        expected_bytes = self.config.dimension * 4
        for index, vector in enumerate(payload):
            if not isinstance(vector, (bytes, bytearray, memoryview)):
                raise self._protocol_error(f"worker vector {index} was not packed bytes")
            packed = bytes(vector)
            if len(packed) != expected_bytes:
                raise self._protocol_error(
                    f"worker vector {index} was {len(packed) // 4} wide, "
                    f"expected {self.config.dimension}"
                )
            row = np.frombuffer(packed, dtype="<f4", count=self.config.dimension)
            if not np.all(np.isfinite(row)):
                raise self._protocol_error(f"worker vector {index} contains non-finite values")
            decoded.append(row.tolist())
        return decoded

    def _decode_float_vectors(self, payload: object, *, count: int) -> list[list[float]]:
        if not isinstance(payload, (list, tuple)) or len(payload) != count:
            raise self._protocol_error(f"worker returned the wrong vector count for {count} inputs")
        decoded: list[list[float]] = []
        for index, vector in enumerate(payload):
            if not isinstance(vector, (list, tuple)) or len(vector) != self.config.dimension:
                raise self._protocol_error(
                    f"worker vector {index} was not {self.config.dimension}-dimensional"
                )
            try:
                row = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise self._protocol_error(f"worker vector {index} was not numeric") from exc
            if not np.all(np.isfinite(row)):
                raise self._protocol_error(f"worker vector {index} contains non-finite values")
            decoded.append(row)
        return decoded

    def _decode_planned_payload(
        self, payload: object, *, candidate_count: int
    ) -> tuple[list[list[tuple[int, int, int, bytes]]], bool]:
        if not isinstance(payload, tuple | list) or len(payload) != 2:
            raise self._protocol_error("worker returned a malformed planned payload")
        raw_segments, tokenizer_available = payload
        if not isinstance(tokenizer_available, bool):
            raise self._protocol_error("worker planned payload has a non-boolean tokenizer flag")
        if not isinstance(raw_segments, (list, tuple)) or len(raw_segments) != candidate_count:
            raise self._protocol_error(
                f"worker returned planned results for {candidate_count} candidates"
            )
        expected_bytes = self.config.dimension * 4
        decoded: list[list[tuple[int, int, int, bytes]]] = []
        for group in raw_segments:
            if not isinstance(group, (list, tuple)):
                raise self._protocol_error("worker planned result was not a segment list")
            decoded_group: list[tuple[int, int, int, bytes]] = []
            for segment in group:
                if not isinstance(segment, tuple | list) or len(segment) != 4:
                    raise self._protocol_error("worker returned a malformed segment")
                start, end, token_count, vector = segment
                if (
                    isinstance(start, bool)
                    or not isinstance(start, int)
                    or isinstance(end, bool)
                    or not isinstance(end, int)
                    or isinstance(token_count, bool)
                    or not isinstance(token_count, int)
                    or start < 0
                    or end < start
                    or token_count < 0
                ):
                    raise self._protocol_error("worker returned invalid segment offsets")
                if not isinstance(vector, (bytes, bytearray, memoryview)):
                    raise self._protocol_error("worker segment vector was not packed bytes")
                packed = bytes(vector)
                if len(packed) != expected_bytes:
                    raise self._protocol_error("worker segment vector has the wrong dimension")
                decoded_group.append((start, end, token_count, packed))
            decoded.append(decoded_group)
        return decoded, tokenizer_available

    def _request_timeout(self, command: str) -> float:
        configured = (
            self.model_load_timeout_seconds
            if command == "initialize"
            else self.inference_timeout_seconds
        )
        return self.request_timeout_seconds if configured is None else configured

    def _channel_failed(self) -> CodeIndexingError:
        """Report a broken command channel as this session's own failure.

        A worker that stops cleanly closes the channel, which reads as
        ``EOFError``. One that dies mid-request breaks it instead, and how that
        breakage surfaces depends on the platform and on which operation first
        touched the dead end: a spawned worker's pipe raises ``BrokenPipeError``
        from the wait, an external worker's socket raises
        ``ConnectionResetError`` from the read, and the send can raise either.
        Every one of them says the same thing -- no result is coming -- so none
        of them may reach a caller as a raw channel error. Callers degrade to
        CPU on EMBEDDING_WORKER_FAILED and can do nothing with an OSError
        escaping from inside the indexing pipeline.
        """
        self.close()
        self.termination_reason = "channel_closed"
        return CodeIndexingError(
            ErrorCode.EMBEDDING_WORKER_FAILED,
            "Embedding worker closed its result channel",
        )

    def close(self) -> None:
        process = self._process
        connection = self._connection
        if process is None:
            self._join_transfer()
            return
        # Even a tiny stop command can block behind an incomplete request.
        # A disposable worker is terminated directly so cleanup stays bounded.
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        if process.is_alive():
            # A worker in another environment is not this process's child to be
            # reaped at exit, and one that ignored SIGTERM may still be holding
            # device memory the next backend needs.
            process.kill()
            process.join(timeout=2)
        if connection is not None:
            connection.close()
        self._process = None
        self._connection = None
        self._initialized = False
        self._join_transfer()

    def _join_transfer(self) -> None:
        transfer = self._transfer
        if transfer is not None:
            transfer.join(timeout=0.2)
            if not transfer.is_alive():
                self._transfer = None

    def _start(self) -> None:
        if self._process is not None:
            return
        if self._transfer is not None and self._transfer.is_alive():
            # Do not spawn retries while a previous channel is still blocked.
            # Production pipes unblock when their worker exits and closes its end.
            raise CodeIndexingError(
                ErrorCode.EMBEDDING_WORKER_FAILED,
                "Previous embedding worker channel did not close",
            )
        self._parent_baseline_bytes = parent_memory_baseline()
        launched = self._launcher.launch(self.config)
        self._process = launched.process
        self.spawn_count += 1
        self._connection = launched.connection

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
