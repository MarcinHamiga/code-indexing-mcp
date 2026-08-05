"""Explicit incremental indexing orchestration."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path

import psutil
from filelock import FileLock, Timeout

from .embedding import (
    EmbeddedSegment,
    Embedder,
    PassageCandidate,
    PassageEmbedder,
    SegmentingEmbedder,
    SegmentPlan,
    compose_passage,
    pack_vector,
)
from .embedding_worker import TelemetrySource
from .errors import CodeIndexingError, ErrorCode
from .extractor import TreeSitterExtractor
from .models import ExtractedChunk, IndexIssue, IndexReport, ProjectInfo, SkippedFile, StoredFile
from .progress import IndexProgress, ProgressPublisher
from .scanner import SourceScanner
from .staging import ChunkRow, StagingJob
from .storage import LanceStore

logger = logging.getLogger(__name__)

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
        # Strict mode's refusal to fall back is a property of the machine and
        # the configuration, never of whichever file happened to be in flight.
        ErrorCode.BACKEND_UNAVAILABLE,
    }
)


@dataclass
class _PendingFile:
    record: StoredFile
    chunks: list[ExtractedChunk]
    source_chars: int
    error: Exception | None = None
    embedded_chunks: int = 0
    emitted_chars: int = 0


@dataclass(frozen=True)
class _PendingCandidate:
    owner: int
    chunk: ExtractedChunk


def _candidate_groups(
    candidates: list[_PendingCandidate],
) -> Iterator[list[_PendingCandidate]]:
    """Split cross-file candidates into bounded worker round trips."""
    group: list[_PendingCandidate] = []
    characters = 0
    for candidate in candidates:
        if group and (
            len(group) >= CANDIDATE_GROUP_COUNT
            or characters + len(candidate.chunk.content) > CANDIDATE_GROUP_CHARS
        ):
            yield group
            group = []
            characters = 0
        group.append(candidate)
        characters += len(candidate.chunk.content)
    if group:
        yield group


def _digest(value: str | bytes) -> str:
    data = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def _content_rejection(source: bytes) -> str | None:
    """Return why *source* cannot be indexed, or None when it can.

    Runs where the bytes are already in hand. The scanner used to do this and throw
    the bytes away, which cost every changed file a second full read.
    """
    if b"\x00" in source:
        return "binary"
    try:
        source.decode("utf-8-sig")
    except UnicodeDecodeError:
        return "encoding"
    return None


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
        staging_directory: Path | None = None,
        progress_directory: Path | None = None,
    ) -> None:
        self.store = store
        self.scanner = scanner
        self.extractor = extractor
        self.embedder = embedder
        self.lock_directory = lock_directory
        self.batch_size = batch_size
        self.segment_plan = segment_plan or SegmentPlan(max_items=batch_size)
        self.passage_session_factory = passage_session_factory
        self.staging_directory = staging_directory or lock_directory.parent / "staging"
        self.progress_directory = progress_directory or lock_directory.parent / "progress"

    def index(
        self,
        project: ProjectInfo,
        *,
        force: bool = False,
        wait_for_lock: bool = False,
        on_progress: Callable[[IndexProgress], None] | None = None,
    ) -> IndexReport:
        started = time.monotonic_ns()
        self.lock_directory.mkdir(parents=True, exist_ok=True)
        global_lock = FileLock(self.lock_directory / "index-global.lock")
        project_lock = FileLock(self.lock_directory / f"{project.id}.lock")
        progress = ProgressPublisher(
            project.id, directory=self.progress_directory, listener=on_progress
        )
        try:
            with (
                global_lock.acquire() if wait_for_lock else global_lock.acquire(timeout=0),
                project_lock.acquire() if wait_for_lock else project_lock.acquire(timeout=0),
            ):
                try:
                    report = self._index_locked(project, force=force, progress=progress)
                finally:
                    # Only the run that holds the lock owns the snapshot, so
                    # clearing here can never delete another process's progress.
                    progress.clear()
        except Timeout as exc:
            raise CodeIndexingError(
                ErrorCode.INDEX_BUSY,
                f"Another indexing job is already active: {project.name}",
                project=project.id,
            ) from exc
        duration_ms = (time.monotonic_ns() - started) // 1_000_000
        return report.model_copy(update={"duration_ms": duration_ms})

    def _index_locked(
        self, project: ProjectInfo, *, force: bool, progress: ProgressPublisher
    ) -> IndexReport:
        self.store.upsert_project(project, model_id=self.embedder.model_id, state="indexing")
        try:
            context = (
                self.passage_session_factory()
                if self.passage_session_factory is not None
                else contextlib.nullcontext(self.embedder)
            )
            with context as passage_embedder:
                report = self._index_scan(
                    project, force=force, passage_embedder=passage_embedder, progress=progress
                )
            if isinstance(context, TelemetrySource):
                # Read after the context exits, so a session that fell back from
                # an accelerator to CPU reports the backend it finished on and
                # the totals from both.
                measured = context.telemetry()
                report = report.model_copy(
                    update={
                        "embedding_backend": measured.backend,
                        "memory_budget_bytes": measured.memory_budget_bytes,
                        "peak_memory_bytes": measured.peak_memory_bytes,
                        "worker_used": True,
                        "embedded_segments": measured.segment_count,
                        "embedded_tokens": measured.token_count,
                        "embedding_retries": measured.retry_count,
                        "fallback_count": report.fallback_count + measured.fallback_count,
                        "worker_termination_reason": measured.termination_reason,
                        "token_windowing": measured.tokenizer_available,
                        "embedding_fallback_reason": measured.fallback_reason,
                        "embedded_characters": measured.character_count,
                        "embedding_crossover_characters": measured.crossover_characters or None,
                        "embedding_selection_reason": measured.selection_reason,
                    }
                )
            return report
        except Exception:
            # Never leave the project stuck in "indexing" after a crash.
            with contextlib.suppress(Exception):
                self.store.upsert_project(project, model_id=self.embedder.model_id, state="error")
            raise

    def _index_scan(
        self,
        project: ProjectInfo,
        *,
        force: bool,
        passage_embedder: PassageEmbedder,
        progress: ProgressPublisher,
    ) -> IndexReport:
        timer = _PhaseTimer()
        with timer.measure("scan"):
            existing = {record.path: record for record in self.store.list_files(project.id)}
        # The scanner streams, so the only honest total before the walk finishes
        # is what the last run saw. A first index reports a bare count instead.
        progress.update(
            phase="scanning",
            files_total=len(existing) or None,
            force=True,
        )
        current_paths: set[str] = set()
        indexed = parsed = embedded = unchanged = metadata_only = removed = skipped = 0
        files_seen = 0
        fallback_count = 0
        errors: list[IndexIssue] = []
        job: StagingJob | None = None
        pending: list[_PendingFile] = []
        pending_chunks = 0
        pending_chars = 0
        process = psutil.Process()
        peak_memory_bytes = 0

        def sample_memory() -> None:
            nonlocal peak_memory_bytes
            # Diagnostics must never turn a successful index into a failure.
            with contextlib.suppress(psutil.Error):
                peak_memory_bytes = max(peak_memory_bytes, process.memory_info().rss)

        sample_memory()

        def staging_job() -> StagingJob:
            nonlocal job
            if job is None:
                job = StagingJob(
                    self.staging_directory,
                    project.id,
                    file_schema=LanceStore.file_arrow_schema(),
                    chunk_schema=LanceStore.chunk_arrow_schema(self.store.vector_dimension),
                    reference_schema=LanceStore.reference_arrow_schema(),
                )
                job.begin()
            return job

        def stage_failure(record: StoredFile, exc: Exception) -> None:
            errors.append(IndexIssue(path=record.path, message=str(exc)))
            staging_job().stage_file(
                record.model_copy(
                    update={
                        "has_errors": True,
                        "error": str(exc),
                        "indexed_at": time.time_ns(),
                    }
                )
            )

        def flush_pending() -> None:
            nonlocal indexed, embedded, fallback_count, pending_chunks, pending_chars
            if not pending:
                return
            candidates = [
                _PendingCandidate(owner, chunk)
                for owner, staged_file in enumerate(pending)
                for chunk in staged_file.chunks
            ]
            # Announced before the work rather than after it: embedding a batch
            # is the longest thing an index does between two updates, and the
            # watcher should learn what the pause is for while it lasts.
            progress.update(phase="embedding", current_path=None, force=True)
            for group in _candidate_groups(candidates):
                active = [
                    candidate for candidate in group if pending[candidate.owner].error is None
                ]
                if not active:
                    continue
                with timer.measure("embed"):
                    try:
                        succeeded, failed, retries = self._embed_candidates(
                            passage_embedder, active
                        )
                    finally:
                        sample_memory()
                fallback_count += retries
                progress.update(phase="embedding")
                staged_rows: dict[int, list[ChunkRow]] = {}
                for candidate, segments in succeeded:
                    target = pending[candidate.owner]
                    if target.error is not None:
                        continue
                    windowed = self._windowed_chunks([candidate.chunk], [segments])
                    rows = [
                        self._chunk_row(project.id, target.record, chunk, vector)
                        for chunk, vector in windowed
                    ]
                    staged_rows.setdefault(candidate.owner, []).extend(rows)
                    target.embedded_chunks += len(rows)
                    target.emitted_chars += sum(len(chunk.content) for chunk, _ in windowed)
                for candidate, exc in failed:
                    target = pending[candidate.owner]
                    if target.error is None:
                        target.error = exc
                with timer.measure("commit"):
                    for owner in sorted(staged_rows):
                        staging_job().stage_chunks(staged_rows[owner])

            with timer.measure("commit"):
                for target in pending:
                    if (
                        target.error is None
                        and target.emitted_chars > SEGMENT_TEXT_GROWTH_LIMIT * target.source_chars
                    ):
                        target.error = ValueError(
                            f"Token windowing emitted {target.emitted_chars} characters from "
                            f"{target.source_chars} characters of chunk text, above the "
                            f"{SEGMENT_TEXT_GROWTH_LIMIT}x limit"
                        )
                    if target.error is not None:
                        stage_failure(target.record, target.error)
                        continue
                    staging_job().stage_file(target.record)
                    staging_job().mark_replaced(target.record.file_id)
                    indexed += 1
                    embedded += target.embedded_chunks
            progress.update(
                phase="embedding", files_indexed=indexed, chunks_embedded=embedded, force=True
            )
            pending.clear()
            pending_chunks = 0
            pending_chars = 0

        stream = self.scanner.iter_scan(project, existing)
        try:
            while True:
                with timer.measure("scan"):
                    item = next(stream, None)
                if item is None:
                    break
                files_seen += 1
                if isinstance(item, SkippedFile):
                    skipped += 1
                    progress.update(phase="scanning", files_seen=files_seen, current_path=None)
                    continue
                path = item.path.as_posix()
                progress.update(
                    phase="scanning",
                    files_seen=files_seen,
                    files_unchanged=unchanged,
                    current_path=path,
                )
                current_paths.add(path)
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
                record: StoredFile | None = None
                try:
                    with timer.measure("scan"):
                        source = (
                            item.content
                            if item.content is not None
                            else item.absolute_path.read_bytes()
                        )
                        rejection = _content_rejection(source)
                        content_hash = _digest(source)
                    if rejection is not None:
                        # Content rejection is a skip, not a syntax/indexing error.
                        # Stage removal so an earlier text version disappears only
                        # when the rest of this indexing transaction commits.
                        if previous is not None:
                            with timer.measure("commit"):
                                staging_job().mark_removed(previous.file_id)
                        skipped += 1
                        continue
                    if not force and previous is not None and previous.content_hash == content_hash:
                        with timer.measure("commit"):
                            staging_job().stage_file(
                                previous.model_copy(
                                    update={"size": item.size, "mtime_ns": item.mtime_ns}
                                )
                            )
                        metadata_only += 1
                        continue
                    record = StoredFile(
                        file_id=_digest(f"{project.id}\0{path}"),
                        project_id=project.id,
                        path=path,
                        language=item.language,
                        size=item.size,
                        mtime_ns=item.mtime_ns,
                        content_hash=content_hash,
                        indexed_at=time.time_ns(),
                    )
                    with timer.measure("parse"):
                        extraction = self.extractor.extract(item.path, item.language, source)
                    parsed += 1
                    source_chars = sum(len(chunk.content) for chunk in extraction.chunks)
                    if pending and (
                        len(pending) >= CANDIDATE_GROUP_COUNT
                        or pending_chunks + len(extraction.chunks) > CANDIDATE_GROUP_COUNT
                        or pending_chars + source_chars > CANDIDATE_GROUP_CHARS
                    ):
                        flush_pending()
                    pending.append(
                        _PendingFile(
                            record=record.model_copy(update={"has_errors": extraction.has_errors}),
                            chunks=extraction.chunks,
                            source_chars=source_chars,
                        )
                    )
                    pending_chunks += len(extraction.chunks)
                    pending_chars += source_chars
                except Exception as exc:
                    if isinstance(exc, CodeIndexingError) and exc.code in ENVIRONMENT_ERROR_CODES:
                        # Not attributable to this file. Recording it below would
                        # stamp the file's current content hash and skip it on every
                        # later run, leaving a permanent hole in the index for what
                        # is really a transient condition.
                        raise
                    # Record the failure so the file is not re-read, re-parsed, and
                    # re-embedded on every run. It is retried only when the file
                    # changes again or when force=True. Chunks from a previous
                    # successful index (if any) are left untouched.
                    with timer.measure("commit"):
                        stage_failure(
                            record
                            or StoredFile(
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
                                indexed_at=time.time_ns(),
                            ),
                            exc,
                        )

            flush_pending()
            progress.update(
                phase="committing",
                files_seen=files_seen,
                files_total=len(current_paths) or None,
                files_unchanged=unchanged,
                current_path=None,
                force=True,
            )
            with timer.measure("commit"):
                for path, record in existing.items():
                    if path not in current_paths:
                        staging_job().mark_removed(record.file_id)
                        removed += 1
                if job is not None:
                    self._commit_staged(project, job, errors=errors)
                else:
                    self.store.upsert_project(
                        project,
                        model_id=self.embedder.model_id,
                        state="partial" if errors else "ready",
                    )
        except BaseException:
            # A run that never reached the commit phase has not touched the
            # live tables, and one whose rollback succeeded is already undone;
            # in both cases the staged bytes are dead. discard() keeps the
            # directory only when the journal is still "committing", which
            # means the rollback failed and recovery still needs it.
            if job is not None:
                job.discard()
            raise
        return IndexReport(
            project_id=project.id,
            discovered_files=len(current_paths),
            indexed_files=indexed,
            parsed_files=parsed,
            embedded_chunks=embedded,
            unchanged_files=unchanged,
            metadata_only_files=metadata_only,
            removed_files=removed,
            skipped_files=skipped,
            errors=errors,
            scan_duration_ms=timer.milliseconds("scan"),
            parse_duration_ms=timer.milliseconds("parse"),
            embed_duration_ms=timer.milliseconds("embed"),
            commit_duration_ms=timer.milliseconds("commit"),
            embedding_backend="cpu",
            embedding_batch_size=self.batch_size,
            scan_ms=timer.milliseconds("scan"),
            parse_ms=timer.milliseconds("parse"),
            embed_ms=timer.milliseconds("embed"),
            commit_ms=timer.milliseconds("commit"),
            fallback_count=fallback_count,
            peak_memory_bytes=peak_memory_bytes,
        )

    def _commit_staged(
        self, project: ProjectInfo, job: StagingJob, *, errors: list[IndexIssue]
    ) -> None:
        """Apply a fully staged run, rolling the live tables back on any failure.

        The journal switches to ``committing`` -- with the versions to restore
        -- before the first live write, so a crash anywhere in this method is
        recoverable: the rollback here handles the live failure, and startup
        recovery handles a process death.
        """
        versions = self.store.table_versions(project.id)
        job.begin_commit(versions)
        try:
            self.store.replace_files_from_arrow(
                project.id,
                files=job.files_table(),
                chunk_groups=job.iter_chunk_groups(),
                reference_groups=job.iter_reference_groups(),
                replace_reference_file_ids=job.replace_reference_file_ids,
                removed_file_ids=job.removed_file_ids,
            )
            if job.replace_file_ids or job.replace_reference_file_ids or job.removed_file_ids:
                self.store.ensure_indexes(project.id, compact=bool(job.removed_file_ids))
            self.store.upsert_project(
                project,
                model_id=self.embedder.model_id,
                state="partial" if errors else "ready",
            )
        except BaseException:
            try:
                self.store.restore_versions(project.id, versions)
            except Exception:
                # The journal stays in "committing" and StagingJob.discard
                # deliberately keeps the directory, so the next startup's
                # recovery retries the rollback. Report the original commit
                # failure rather than this one -- it is the real cause.
                logger.exception(
                    "Could not roll back the interrupted commit for %s; keeping %s "
                    "for startup recovery",
                    project.id,
                    job.directory,
                )
            else:
                job.rolled_back()
            raise
        job.complete()

    def _embed_chunks(
        self, passage_embedder: PassageEmbedder, chunks: list[ExtractedChunk]
    ) -> list[list[EmbeddedSegment]]:
        """Embed one candidate group, bounded by tokens where supported."""
        if not chunks:
            return []
        if isinstance(passage_embedder, SegmentingEmbedder):
            candidates = [
                PassageCandidate(chunk.embedding_prefix, chunk.content) for chunk in chunks
            ]
            return passage_embedder.plan_and_embed(candidates, self.segment_plan)
        # An embedder without token planning (a test double, or a future
        # backend) still indexes; its sequence length is simply unbounded.
        vectors: list[list[float]] = []
        texts = [chunk.embedding_text for chunk in chunks]
        for offset in range(0, len(texts), self.batch_size):
            vectors.extend(
                passage_embedder.embed_passages(texts[offset : offset + self.batch_size])
            )
        return [
            [EmbeddedSegment(0, len(chunk.content), 0, pack_vector(vector))]
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def _embed_candidates(
        self,
        passage_embedder: PassageEmbedder,
        candidates: list[_PendingCandidate],
    ) -> tuple[
        list[tuple[_PendingCandidate, list[EmbeddedSegment]]],
        list[tuple[_PendingCandidate, Exception]],
        int,
    ]:
        """Embed a bounded group, bisecting only content-attributable failures."""
        try:
            segments = self._embed_chunks(
                passage_embedder, [candidate.chunk for candidate in candidates]
            )
        except Exception as exc:
            if isinstance(exc, MemoryError) or (
                isinstance(exc, CodeIndexingError) and exc.code in ENVIRONMENT_ERROR_CODES
            ):
                raise
            if len(candidates) == 1:
                return [], [(candidates[0], exc)], 0
            midpoint = len(candidates) // 2
            left_ok, left_failed, left_retries = self._embed_candidates(
                passage_embedder, candidates[:midpoint]
            )
            right_ok, right_failed, right_retries = self._embed_candidates(
                passage_embedder, candidates[midpoint:]
            )
            return (
                [*left_ok, *right_ok],
                [*left_failed, *right_failed],
                1 + left_retries + right_retries,
            )
        return list(zip(candidates, segments, strict=True)), [], 0

    def _windowed_chunks(
        self, chunks: list[ExtractedChunk], segments: list[list[EmbeddedSegment]]
    ) -> list[tuple[ExtractedChunk, bytes]]:
        """Pair each chunk's token windows with the vector that covers them."""
        windowed: list[tuple[ExtractedChunk, bytes]] = []
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
    def _chunk_row(
        project_id: str, file: StoredFile, chunk: ExtractedChunk, vector: bytes
    ) -> ChunkRow:
        identity = "\0".join(
            [
                file.file_id,
                chunk.kind,
                chunk.qualified_symbol or "",
                str(chunk.start_byte),
                str(chunk.end_byte),
                str(chunk.part_index),
            ]
        )
        return ChunkRow(
            chunk_id=_digest(identity),
            file_id=file.file_id,
            project_id=project_id,
            path=file.path,
            language=file.language,
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
            search_text=chunk.search_text,
            content_hash=file.content_hash,
            part_index=chunk.part_index,
            vector=vector,
        )
