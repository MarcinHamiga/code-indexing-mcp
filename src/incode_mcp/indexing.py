"""Explicit incremental indexing orchestration."""

from __future__ import annotations

import contextlib
import hashlib
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path

from filelock import FileLock, Timeout

from .embedding import (
    EmbeddedSegment,
    Embedder,
    PassageCandidate,
    PassageEmbedder,
    SegmentingEmbedder,
    SegmentPlan,
    compose_passage,
)
from .embedding_worker import EmbeddingWorkerSession
from .errors import ErrorCode, IncodeError
from .extractor import TreeSitterExtractor
from .models import ExtractedChunk, IndexIssue, IndexReport, ProjectInfo, StoredChunk, StoredFile
from .scanner import SourceScanner
from .storage import LanceStore

# Token windows overlap by at most half a budget, so windowing a file can never
# legitimately double the text its chunks already carried. Exceeding this means
# planning misbehaved; the file is rejected rather than flooding the index.
SEGMENT_TEXT_GROWTH_LIMIT = 2
# One planning request carries its candidates' text to the worker and their
# vectors back. Grouping keeps that round trip bounded on files that extract
# into thousands of chunks, and keeps a retry from re-embedding the whole file.
CANDIDATE_GROUP_CHARS = 256 * 1024
CANDIDATE_GROUP_COUNT = 256

# Failures caused by the environment rather than by a file's own content. They
# abort the run instead of being recorded against whichever file was in flight.
ENVIRONMENT_ERROR_CODES = frozenset(
    {
        ErrorCode.MODEL_UNAVAILABLE,
        ErrorCode.INDEX_RESOURCE_LIMIT,
        ErrorCode.EMBEDDING_WORKER_FAILED,
    }
)


def _candidate_groups(chunks: list[ExtractedChunk]) -> Iterator[list[ExtractedChunk]]:
    """Split a file's chunks into groups small enough for one worker round trip."""
    group: list[ExtractedChunk] = []
    characters = 0
    for chunk in chunks:
        if group and (
            len(group) >= CANDIDATE_GROUP_COUNT
            or characters + len(chunk.content) > CANDIDATE_GROUP_CHARS
        ):
            yield group
            group = []
            characters = 0
        group.append(chunk)
        characters += len(chunk.content)
    if group:
        yield group


def _digest(value: str | bytes) -> str:
    data = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


@dataclass
class _PhaseTimer:
    """Accumulate wall time per indexing phase across an interleaved run."""

    totals: dict[str, int] = field(default_factory=dict)

    @contextlib.contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        started = time.monotonic_ns()
        try:
            yield
        finally:
            self.totals[phase] = self.totals.get(phase, 0) + (time.monotonic_ns() - started)

    def milliseconds(self, phase: str) -> int:
        return self.totals.get(phase, 0) // 1_000_000


class Indexer:
    def __init__(
        self,
        *,
        store: LanceStore,
        scanner: SourceScanner,
        extractor: TreeSitterExtractor,
        embedder: Embedder,
        lock_directory: Path,
        batch_size: int = 1,
        segment_plan: SegmentPlan | None = None,
        passage_session_factory: Callable[[], AbstractContextManager[PassageEmbedder]]
        | None = None,
    ) -> None:
        self.store = store
        self.scanner = scanner
        self.extractor = extractor
        self.embedder = embedder
        self.lock_directory = lock_directory
        self.batch_size = batch_size
        self.segment_plan = segment_plan or SegmentPlan(max_items=batch_size)
        self.passage_session_factory = passage_session_factory

    def index(
        self, project: ProjectInfo, *, force: bool = False, wait_for_lock: bool = False
    ) -> IndexReport:
        started = time.monotonic_ns()
        self.lock_directory.mkdir(parents=True, exist_ok=True)
        global_lock = FileLock(self.lock_directory / "index-global.lock")
        project_lock = FileLock(self.lock_directory / f"{project.id}.lock")
        try:
            with (
                global_lock.acquire() if wait_for_lock else global_lock.acquire(timeout=0),
                project_lock.acquire() if wait_for_lock else project_lock.acquire(timeout=0),
            ):
                report = self._index_locked(project, force=force)
        except Timeout as exc:
            raise IncodeError(
                ErrorCode.INDEX_BUSY,
                f"Another indexing job is already active: {project.name}",
                project=project.id,
            ) from exc
        duration_ms = (time.monotonic_ns() - started) // 1_000_000
        return report.model_copy(update={"duration_ms": duration_ms})

    def _index_locked(self, project: ProjectInfo, *, force: bool) -> IndexReport:
        self.store.upsert_project(project, model_id=self.embedder.model_id, state="indexing")
        try:
            context = (
                self.passage_session_factory()
                if self.passage_session_factory is not None
                else contextlib.nullcontext(self.embedder)
            )
            with context as passage_embedder:
                report = self._index_scan(project, force=force, passage_embedder=passage_embedder)
            if isinstance(context, EmbeddingWorkerSession):
                report = report.model_copy(
                    update={
                        "memory_budget_bytes": context.effective_ceiling_bytes,
                        "peak_memory_bytes": context.peak_combined_rss,
                        "worker_used": True,
                        "embedded_segments": context.segment_count,
                        "embedded_tokens": context.token_count,
                        "embedding_retries": context.retry_count,
                        "worker_termination_reason": context.termination_reason,
                        "token_windowing": context.tokenizer_available,
                    }
                )
            return report
        except Exception:
            # Never leave the project stuck in "indexing" after a crash.
            with contextlib.suppress(Exception):
                self.store.upsert_project(project, model_id=self.embedder.model_id, state="error")
            raise

    def _index_scan(
        self, project: ProjectInfo, *, force: bool, passage_embedder: PassageEmbedder
    ) -> IndexReport:
        timer = _PhaseTimer()
        with timer.measure("scan"):
            existing = {record.path: record for record in self.store.list_files(project.id)}
            scan = self.scanner.scan(project, existing)
        current_paths = {item.path.as_posix() for item in scan.files}
        indexed = parsed = embedded = unchanged = metadata_only = removed = 0
        errors: list[IndexIssue] = []

        for item in scan.files:
            path = item.path.as_posix()
            previous = existing.get(path)
            if (
                not force
                and previous is not None
                and previous.size == item.size
                and previous.mtime_ns == item.mtime_ns
            ):
                unchanged += 1
                continue
            content_hash: str | None = None
            try:
                with timer.measure("scan"):
                    source = (
                        item.content
                        if item.content is not None
                        else item.absolute_path.read_bytes()
                    )
                    content_hash = _digest(source)
                if not force and previous is not None and previous.content_hash == content_hash:
                    with timer.measure("commit"):
                        self.store.upsert_file(
                            previous.model_copy(
                                update={"size": item.size, "mtime_ns": item.mtime_ns}
                            )
                        )
                    metadata_only += 1
                    continue
                with timer.measure("parse"):
                    extraction = self.extractor.extract(item.path, item.language, source)
                parsed += 1
                with timer.measure("embed"):
                    segments = self._embed_chunks(passage_embedder, extraction.chunks)
                windowed = self._windowed_chunks(extraction.chunks, segments)
                stored_file = StoredFile(
                    file_id=_digest(f"{project.id}\0{path}"),
                    project_id=project.id,
                    path=path,
                    language=item.language,
                    size=item.size,
                    mtime_ns=item.mtime_ns,
                    content_hash=content_hash,
                    has_errors=extraction.has_errors,
                    indexed_at=time.time_ns(),
                )
                chunks = [
                    self._stored_chunk(project.id, stored_file, chunk, vector)
                    for chunk, vector in windowed
                ]
                with timer.measure("commit"):
                    self.store.replace_file(stored_file, chunks)
                indexed += 1
                embedded += len(chunks)
            except Exception as exc:
                if isinstance(exc, IncodeError) and exc.code in ENVIRONMENT_ERROR_CODES:
                    # Not attributable to this file. Recording it below would
                    # stamp the file's current content hash and skip it on every
                    # later run, leaving a permanent hole in the index for what
                    # is really a transient condition.
                    raise
                errors.append(IndexIssue(path=path, message=str(exc)))
                # Record the failure so the file is not re-read, re-parsed, and
                # re-embedded on every run. It is retried only when the file
                # changes again or when force=True. Chunks from a previous
                # successful index (if any) are left untouched.
                with timer.measure("commit"):
                    self.store.upsert_file(
                        StoredFile(
                            file_id=_digest(f"{project.id}\0{path}"),
                            project_id=project.id,
                            path=path,
                            language=item.language,
                            size=item.size,
                            mtime_ns=item.mtime_ns,
                            content_hash=(
                                content_hash
                                if content_hash is not None
                                else previous.content_hash
                                if previous is not None
                                else ""
                            ),
                            has_errors=True,
                            error=str(exc),
                            indexed_at=time.time_ns(),
                        )
                    )

        with timer.measure("commit"):
            for path, record in existing.items():
                if path not in current_paths:
                    self.store.remove_file(project.id, record.file_id)
                    removed += 1

            if indexed or removed:
                self.store.ensure_indexes(project.id, compact=removed > 0)

            self.store.upsert_project(
                project,
                model_id=self.embedder.model_id,
                state="partial" if errors else "ready",
            )
        return IndexReport(
            project_id=project.id,
            discovered_files=len(scan.files),
            indexed_files=indexed,
            parsed_files=parsed,
            embedded_chunks=embedded,
            unchanged_files=unchanged,
            metadata_only_files=metadata_only,
            removed_files=removed,
            skipped_files=len(scan.skipped),
            errors=errors,
            scan_duration_ms=timer.milliseconds("scan"),
            parse_duration_ms=timer.milliseconds("parse"),
            embed_duration_ms=timer.milliseconds("embed"),
            commit_duration_ms=timer.milliseconds("commit"),
        )

    def _embed_chunks(
        self, passage_embedder: PassageEmbedder, chunks: list[ExtractedChunk]
    ) -> list[list[EmbeddedSegment]]:
        """Embed each chunk, bounded by tokens where the embedder supports it."""
        if not chunks:
            return []
        if isinstance(passage_embedder, SegmentingEmbedder):
            segments: list[list[EmbeddedSegment]] = []
            for group in _candidate_groups(chunks):
                candidates = [
                    PassageCandidate(chunk.embedding_prefix, chunk.content) for chunk in group
                ]
                segments.extend(passage_embedder.plan_and_embed(candidates, self.segment_plan))
            return segments
        # An embedder without token planning (a test double, or a future
        # backend) still indexes; its sequence length is simply unbounded.
        vectors: list[list[float]] = []
        texts = [chunk.embedding_text for chunk in chunks]
        for offset in range(0, len(texts), self.batch_size):
            vectors.extend(
                passage_embedder.embed_passages(texts[offset : offset + self.batch_size])
            )
        return [
            [EmbeddedSegment(0, len(chunk.content), 0, vector)]
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def _windowed_chunks(
        self, chunks: list[ExtractedChunk], segments: list[list[EmbeddedSegment]]
    ) -> list[tuple[ExtractedChunk, list[float]]]:
        """Pair each chunk's token windows with the vector that covers them."""
        windowed: list[tuple[ExtractedChunk, list[float]]] = []
        for chunk, planned in zip(chunks, segments, strict=True):
            if not planned:
                continue
            if (
                len(planned) == 1
                and planned[0].start_char == 0
                and planned[0].end_char >= len(chunk.content)
            ):
                # The whole chunk fit one window, which is every ordinary chunk.
                # Reuse it untouched so unwindowed files keep identical output.
                windowed.append((chunk, planned[0].vector))
                continue
            windowed.extend(
                (self._window_chunk(chunk, segment), segment.vector) for segment in planned
            )

        emitted = sum(len(chunk.content) for chunk, _ in windowed)
        source = sum(len(chunk.content) for chunk in chunks)
        if emitted > SEGMENT_TEXT_GROWTH_LIMIT * source:
            raise ValueError(
                f"Token windowing emitted {emitted} characters from {source} characters "
                f"of chunk text, above the {SEGMENT_TEXT_GROWTH_LIMIT}x limit"
            )
        return windowed

    @staticmethod
    def _window_chunk(chunk: ExtractedChunk, segment: EmbeddedSegment) -> ExtractedChunk:
        """Rebuild a chunk around one token window of its content.

        Byte and line offsets are derived from the window's character offsets
        against the chunk's own content, so they stay anchored to the source
        without re-decoding the file.
        """
        head = chunk.content[: segment.start_char]
        content = chunk.content[segment.start_char : segment.end_char]
        start_byte = chunk.start_byte + len(head.encode("utf-8"))
        start_line = chunk.start_line + head.count("\n")
        embedding_text = compose_passage(chunk.embedding_prefix, content)
        return chunk.model_copy(
            update={
                "start_byte": start_byte,
                "end_byte": start_byte + len(content.encode("utf-8")),
                "start_line": start_line,
                "end_line": start_line + content.count("\n"),
                "content": content,
                "embedding_text": embedding_text,
                "search_text": f"{embedding_text}\n{chunk.search_suffix}",
            }
        )

    @staticmethod
    def _stored_chunk(
        project_id: str, file: StoredFile, chunk: object, vector: list[float]
    ) -> StoredChunk:
        from .models import ExtractedChunk

        extracted = ExtractedChunk.model_validate(chunk)
        identity = "\0".join(
            [
                file.file_id,
                extracted.kind,
                extracted.qualified_symbol or "",
                str(extracted.start_byte),
                str(extracted.end_byte),
                str(extracted.part_index),
            ]
        )
        return StoredChunk(
            chunk_id=_digest(identity),
            file_id=file.file_id,
            project_id=project_id,
            path=file.path,
            language=file.language,
            kind=extracted.kind,
            symbol=extracted.symbol,
            qualified_symbol=extracted.qualified_symbol,
            parent_symbol=extracted.parent_symbol,
            start_byte=extracted.start_byte,
            end_byte=extracted.end_byte,
            start_line=extracted.start_line,
            end_line=extracted.end_line,
            content=extracted.content,
            embedding_text=extracted.embedding_text,
            search_text=extracted.search_text,
            content_hash=file.content_hash,
            part_index=extracted.part_index,
            vector=vector,
        )
