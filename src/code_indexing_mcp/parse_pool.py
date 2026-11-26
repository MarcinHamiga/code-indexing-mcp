"""Lean multi-process tree-sitter read/parse pool for large indexing runs.

This module intentionally avoids importing heavy storage, Arrow, LanceDB, or
MCP packages at top level so spawned worker processes start quickly.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import multiprocessing
import os
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import psutil

from .errors import CodeIndexingError, ErrorCode
from .extractor import TreeSitterExtractor
from .memory_budget import ParentMemoryBudget, _current_budget, check_parent_memory_budget
from .models import (
    REFERENCE_SCHEMA_VERSION,
    ExtractedChunk,
    ExtractedDeclarationShape,
    ExtractedReference,
    content_digest,
)
from .settings import MAX_PARSE_WORKERS
from .source_io import PinnedDirectoryCache, SourceReadError, read_source_direct

PARSE_POOL_MIN_FILES = 24
PARSE_BATCH_FILES = 8
PARSE_WORKER_MEMORY_ESTIMATE_BYTES = 160 * 1024 * 1024
PARSE_MEMORY_RESERVE_BYTES = 1024 * 1024 * 1024

ENVIRONMENT_ERROR_CODES = frozenset(
    {
        ErrorCode.MODEL_UNAVAILABLE,
        ErrorCode.INDEX_RESOURCE_LIMIT,
        ErrorCode.EMBEDDING_WORKER_FAILED,
        ErrorCode.BACKEND_UNAVAILABLE,
    }
)


class ParseTask(NamedTuple):
    """One file scheduled for read, hash, and AST/reference extraction."""

    rel_posix: str
    language: str
    prev_content_hash: str | None = None
    prev_language: str | None = None
    force: bool = False
    retrying: bool = False
    size: int = 0
    mtime_ns: int = 0
    expected_identity: tuple[int, int] | None = None
    stat_matched_previous: bool = False


class WireChunk(NamedTuple):
    """Compact pickleable tuple mirroring ``ExtractedChunk`` fields without Pydantic overhead.

    ``search_text`` is deliberately absent: it is exactly
    ``embedding_text + "\\n" + search_suffix`` and no indexing consumer reads it,
    so sending it would grow every inter-process batch by a normalized copy of
    the chunk text. :meth:`to_extracted` reconstructs it when a caller iterates
    the full model.
    """

    kind: str
    symbol: str | None
    qualified_symbol: str | None
    parent_symbol: str | None
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    content: str
    embedding_text: str
    part_index: int
    is_test: bool
    embedding_prefix: str
    search_suffix: str

    @classmethod
    def from_extracted(cls, chunk: ExtractedChunk) -> WireChunk:
        return cls(
            kind=chunk.kind,
            symbol=chunk.symbol,
            qualified_symbol=chunk.qualified_symbol,
            parent_symbol=chunk.parent_symbol,
            start_byte=chunk.start_byte,
            end_byte=chunk.end_byte,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            content=chunk.content,
            embedding_text=chunk.embedding_text,
            part_index=chunk.part_index,
            is_test=chunk.is_test,
            embedding_prefix=chunk.embedding_prefix,
            search_suffix=chunk.search_suffix,
        )

    def to_extracted(self) -> ExtractedChunk:
        return ExtractedChunk.model_construct(
            kind=self.kind,
            symbol=self.symbol,
            qualified_symbol=self.qualified_symbol,
            parent_symbol=self.parent_symbol,
            start_byte=self.start_byte,
            end_byte=self.end_byte,
            start_line=self.start_line,
            end_line=self.end_line,
            content=self.content,
            embedding_text=self.embedding_text,
            search_text=f"{self.embedding_text}\n{self.search_suffix}",
            part_index=self.part_index,
            is_test=self.is_test,
            embedding_prefix=self.embedding_prefix,
            search_suffix=self.search_suffix,
            source_start_byte=None,
            source_end_byte=None,
            original_end_byte=None,
            source_content_hash=None,
        )


ChunkTuple = WireChunk

_WORKER_EXTRACTOR: TreeSitterExtractor | None = None


def resolve_parse_workers(configured: int, memory_ceiling_bytes: int) -> int:
    """Resolve the number of parser worker processes for a run.

    Explicit values (``configured >= 1``) honor the user's setting up to
    ``MAX_PARSE_WORKERS`` without applying the automatic memory heuristic;
    automatic sizing (``configured <= 0``) caps by both CPU count and
    *memory_ceiling_bytes*.
    """
    if configured == 1:
        return 1
    if configured > 1:
        return min(configured, MAX_PARSE_WORKERS)
    cpu_cap = min(8, max(1, (os.cpu_count() or 1) - 2))
    mem_cap = max(
        1,
        (memory_ceiling_bytes - PARSE_MEMORY_RESERVE_BYTES) // PARSE_WORKER_MEMORY_ESTIMATE_BYTES,
    )
    return min(cpu_cap, mem_cap)


def per_worker_memory_ceiling(memory_ceiling_bytes: int | None, workers: int) -> int | None:
    """Divide the run ceiling across workers so their aggregate stays bounded.

    Each worker's budget measures growth from its own post-init RSS, so handing
    every worker the whole run ceiling let *N* workers grow to *N* times the
    configured bound. The automatic worker sizing already keeps
    ``PARSE_MEMORY_RESERVE_BYTES`` for the parent; subtracting it here gives the
    same total bound when workers are configured explicitly. The per-worker
    estimate is kept as a floor: an explicit high worker count should not make
    ordinary extraction trip a budget this small, only cap how far past the
    estimate each worker may grow.
    """
    if memory_ceiling_bytes is None or memory_ceiling_bytes <= 0:
        return None
    available = max(0, memory_ceiling_bytes - PARSE_MEMORY_RESERVE_BYTES)
    share = available // max(1, workers)
    return max(PARSE_WORKER_MEMORY_ESTIMATE_BYTES, share)


def content_rejection(source: bytes) -> str | None:
    """Return why *source* cannot be indexed, or None when it can."""
    if b"\x00" in source:
        return "binary"
    try:
        source.decode("utf-8-sig")
    except UnicodeDecodeError:
        return "encoding"
    return None


def build_reference_columns(
    file_id: str,
    references: Sequence[ExtractedReference],
    declarations: Sequence[ExtractedDeclarationShape],
) -> tuple[list[Any], ...]:
    """Build the 15 varying columns in ``VARYING_REFERENCE_COLUMNS`` order."""
    reference_ids: list[str] = []
    record_kinds: list[str] = []
    kinds: list[str | None] = []
    source_qualified_symbols: list[str | None] = []
    written_names: list[str | None] = []
    target_names: list[str | None] = []
    module_paths: list[str | None] = []
    imported_names: list[str | None] = []
    aliases: list[str | None] = []
    receiver_texts: list[str | None] = []
    start_bytes: list[int | None] = []
    end_bytes: list[int | None] = []
    start_lines: list[int | None] = []
    end_lines: list[int | None] = []
    shape_jsons: list[str | None] = []

    schema_str = str(REFERENCE_SCHEMA_VERSION)

    def identity(
        record_kind: str, kind: str | None, start_byte: int | None, end_byte: int | None
    ) -> str:
        # `kind` belongs in the digest because one byte range legitimately
        # carries two references: a superclass is both `inheritance` and a
        # `read`, and a decorator call is both `decorator` and `call`.
        return content_digest(
            "\0".join(
                (
                    file_id,
                    record_kind,
                    kind or "",
                    str(start_byte if start_byte is not None else -1),
                    str(end_byte if end_byte is not None else -1),
                    schema_str,
                )
            )
        )

    for reference in references:
        reference_ids.append(
            identity("reference", reference.kind, reference.start_byte, reference.end_byte)
        )
        record_kinds.append("reference")
        kinds.append(reference.kind)
        source_qualified_symbols.append(reference.source_qualified_symbol)
        written_names.append(reference.written_name)
        target_names.append(reference.target_name)
        module_paths.append(reference.module_path)
        imported_names.append(reference.imported_name)
        aliases.append(reference.alias)
        receiver_texts.append(reference.receiver_text)
        start_bytes.append(reference.start_byte)
        end_bytes.append(reference.end_byte)
        start_lines.append(reference.start_line)
        end_lines.append(reference.end_line)
        shape_jsons.append(
            reference.call_shape.model_dump_json() if reference.call_shape is not None else None
        )

    for declaration in declarations:
        reference_ids.append(
            identity("declaration", declaration.kind, declaration.start_byte, declaration.end_byte)
        )
        record_kinds.append("declaration")
        kinds.append(declaration.kind)
        source_qualified_symbols.append(declaration.qualified_symbol)
        written_names.append(declaration.symbol)
        target_names.append(declaration.symbol)
        module_paths.append(None)
        imported_names.append(None)
        aliases.append(None)
        receiver_texts.append(None)
        start_bytes.append(declaration.start_byte)
        end_bytes.append(declaration.end_byte)
        start_lines.append(declaration.start_line)
        end_lines.append(declaration.end_line)
        shape_jsons.append(
            json.dumps(
                [parameter.model_dump(mode="json") for parameter in declaration.parameters],
                separators=(",", ":"),
                sort_keys=True,
            )
        )

    # Durable proof that the current structural schema parsed this exact content.
    reference_ids.append(identity("coverage", None, None, None))
    record_kinds.append("coverage")
    kinds.append(None)
    source_qualified_symbols.append(None)
    written_names.append(None)
    target_names.append(None)
    module_paths.append(None)
    imported_names.append(None)
    aliases.append(None)
    receiver_texts.append(None)
    start_bytes.append(None)
    end_bytes.append(None)
    start_lines.append(None)
    end_lines.append(None)
    shape_jsons.append(None)

    return (
        reference_ids,
        record_kinds,
        kinds,
        source_qualified_symbols,
        written_names,
        target_names,
        module_paths,
        imported_names,
        aliases,
        receiver_texts,
        start_bytes,
        end_bytes,
        start_lines,
        end_lines,
        shape_jsons,
    )


def _worker_init(
    max_chars: int,
    max_lines: int,
    overlap_lines: int,
    memory_ceiling_bytes: int | None = None,
) -> None:
    """Initialize one parser worker's cached extractor and optional memory ceiling."""
    global _WORKER_EXTRACTOR
    _WORKER_EXTRACTOR = TreeSitterExtractor(
        max_chars=max_chars,
        max_lines=max_lines,
        overlap_lines=overlap_lines,
    )
    if memory_ceiling_bytes is not None and memory_ceiling_bytes > 0:
        _current_budget.set(
            ParentMemoryBudget(psutil.Process().memory_info().rss, memory_ceiling_bytes)
        )


def process_parse_batch(
    root: Path,
    max_file_bytes: int,
    project_id: str,
    batch: Sequence[ParseTask | tuple[Any, ...]],
    extractor: TreeSitterExtractor,
) -> list[tuple[Any, ...]]:
    """Read, validate, hash, and extract a batch of files into compact tuples."""
    results: list[tuple[Any, ...]] = []
    with PinnedDirectoryCache(root) as cache:
        for raw_task in batch:
            task = raw_task if isinstance(raw_task, ParseTask) else ParseTask(*raw_task)
            rel_posix = task.rel_posix
            language = task.language
            relative = Path(rel_posix)
            try:
                source, info = read_source_direct(
                    root,
                    relative,
                    max_file_bytes,
                    directory_cache=cache,
                    expected_identity=task.expected_identity,
                )
            except SourceReadError as exc:
                if task.stat_matched_previous:
                    results.append(
                        (
                            "extract_error",
                            rel_posix,
                            language,
                            task.size,
                            task.mtime_ns,
                            None,
                            0,
                            str(exc),
                        )
                    )
                else:
                    results.append(("read_skip", rel_posix, exc.reason, str(exc)))
                continue
            except OSError as exc:
                if task.stat_matched_previous:
                    results.append(
                        (
                            "extract_error",
                            rel_posix,
                            language,
                            task.size,
                            task.mtime_ns,
                            None,
                            0,
                            str(exc),
                        )
                    )
                else:
                    results.append(("read_skip", rel_posix, "unreadable", str(exc)))
                continue

            content_hash = content_digest(source)
            rejection = content_rejection(source)
            if rejection is not None:
                results.append(
                    (
                        "rejected",
                        rel_posix,
                        language,
                        info.st_size,
                        info.st_mtime_ns,
                        content_hash,
                        rejection,
                        len(source),
                    )
                )
                continue
            if (
                not task.force
                and not task.retrying
                and task.prev_content_hash == content_hash
                and task.prev_language == language
            ):
                results.append(
                    (
                        "unchanged_hash",
                        rel_posix,
                        language,
                        info.st_size,
                        info.st_mtime_ns,
                        content_hash,
                        len(source),
                    )
                )
                continue

            try:
                t0 = time.monotonic_ns()
                extraction = extractor.extract(relative, language, source)
                check_parent_memory_budget()
                parse_ns = time.monotonic_ns() - t0
                file_id = content_digest(f"{project_id}\0{rel_posix}")
                chunks_tuples = [WireChunk.from_extracted(chunk) for chunk in extraction.chunks]
                if not extraction.has_errors:
                    varying_columns: tuple[list[Any], ...] | None = build_reference_columns(
                        file_id, extraction.references, extraction.declarations
                    )
                    staged_ref_count = len(extraction.references)
                else:
                    varying_columns = None
                    staged_ref_count = 0
                results.append(
                    (
                        "parsed",
                        rel_posix,
                        language,
                        info.st_size,
                        info.st_mtime_ns,
                        content_hash,
                        file_id,
                        len(source),
                        extraction.has_errors,
                        chunks_tuples,
                        varying_columns,
                        staged_ref_count,
                        parse_ns,
                        extraction.reference_extraction_ns,
                    )
                )
            except Exception as exc:
                if isinstance(exc, CodeIndexingError) and exc.code in ENVIRONMENT_ERROR_CODES:
                    results.append(
                        (
                            "fatal_error",
                            exc.code.value,
                            str(exc.args[0]) if exc.args else str(exc),
                            dict(exc.details),
                        )
                    )
                    break
                results.append(
                    (
                        "extract_error",
                        rel_posix,
                        language,
                        info.st_size,
                        info.st_mtime_ns,
                        content_hash,
                        len(source),
                        str(exc),
                    )
                )
    return results


def _process_batch(
    root_str: str,
    max_file_bytes: int,
    project_id: str,
    batch: list[ParseTask],
) -> list[tuple[Any, ...]]:
    """Worker entrypoint running ``process_parse_batch`` with the worker extractor."""
    global _WORKER_EXTRACTOR
    if _WORKER_EXTRACTOR is None:
        _WORKER_EXTRACTOR = TreeSitterExtractor()
    return process_parse_batch(
        Path(root_str),
        max_file_bytes,
        project_id,
        batch,
        _WORKER_EXTRACTOR,
    )


class ParseWorkerPool:
    """Spawned process pool with bounded in-flight batches, crash detection, and ordered drain."""

    def __init__(
        self,
        workers: int,
        *,
        max_chars: int = 4_096,
        max_lines: int = 200,
        overlap_lines: int = 20,
        max_in_flight: int | None = None,
        memory_ceiling_bytes: int | None = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.workers = workers
        self.max_in_flight = (
            max(1, max_in_flight) if max_in_flight is not None else max(1, workers * 2)
        )
        self._pending: deque[Future[list[tuple[Any, ...]]]] = deque()
        context = multiprocessing.get_context("spawn")
        self._executor: ProcessPoolExecutor | None = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_worker_init,
            initargs=(
                max_chars,
                max_lines,
                overlap_lines,
                per_worker_memory_ceiling(memory_ceiling_bytes, workers),
            ),
        )

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def has_ready(self) -> bool:
        return bool(self._pending and self._pending[0].done())

    def submit(
        self,
        root: Path | str,
        max_file_bytes: int,
        project_id: str,
        batch: Sequence[ParseTask | tuple[Any, ...]],
    ) -> None:
        if not batch:
            return
        if self._executor is None:
            raise RuntimeError("ParseWorkerPool is closed")
        normalized = [task if isinstance(task, ParseTask) else ParseTask(*task) for task in batch]
        future = self._executor.submit(
            _process_batch,
            str(root),
            max_file_bytes,
            project_id,
            normalized,
        )
        self._pending.append(future)

    def try_drain_one(self) -> list[tuple[Any, ...]] | None:
        """Pop and return the oldest batch if it has already completed."""
        if not self._pending or not self._pending[0].done():
            return None
        return self._pending.popleft().result()

    def drain_one(self, stop_event: threading.Event | None = None) -> list[tuple[Any, ...]] | None:
        """Wait for the oldest pending batch, aborting promptly when *stop_event* sets.

        Raises ``BrokenProcessPool`` promptly if any worker process dies unexpectedly.
        """
        if not self._pending:
            return None
        pending = self._pending[0]
        while True:
            if stop_event is not None and stop_event.is_set():
                return None
            if pending.done():
                break
            concurrent.futures.wait((pending,), timeout=0.05)
        self._pending.popleft()
        return pending.result()

    def drain_all(
        self, stop_event: threading.Event | None = None
    ) -> Iterator[list[tuple[Any, ...]]]:
        """Yield remaining pending batches in submission order until drained or stopped."""
        while self._pending:
            batch = self.drain_one(stop_event)
            if batch is None:
                break
            yield batch

    def close(self, abort: bool = False) -> None:
        """Shut down or terminate all worker processes and wait for them to exit."""
        executor = self._executor
        if executor is None:
            return
        self._executor = None
        for future in self._pending:
            if abort:
                future.cancel()
        self._pending.clear()
        try:
            if abort:
                processes = getattr(executor, "_processes", None)
                if isinstance(processes, dict):
                    for proc in list(processes.values()):
                        with contextlib.suppress(OSError):
                            if proc.is_alive():
                                proc.terminate()
        finally:
            executor.shutdown(wait=True, cancel_futures=abort)

    def __enter__(self) -> ParseWorkerPool:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close(abort=exc_type is not None)
