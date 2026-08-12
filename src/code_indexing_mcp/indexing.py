"""Explicit incremental indexing orchestration."""

from __future__ import annotations

import contextlib
import hashlib
import json
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
from .models import (
    ExtractedChunk,
    ExtractedDeclarationShape,
    ExtractedReference,
    IndexIssue,
    IndexReport,
    ProjectInfo,
    ReferenceBackfillReport,
    ReferenceCoverage,
    ScannedFile,
    SkippedFile,
    StoredFile,
)
from .progress import IndexProgress, ProgressPublisher
from .scanner import SourceScanner
from .staging import ChunkRow, ReferenceRow, StagingJob
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

# Bump only when the normalized structural-row contract changes. Coverage rows
# make a new generation discoverable without coupling it to project metadata.
# Version 4 puts the reference kind in the row identity. Bumping it also
# discards any generation written by version 3, whose colliding ids are what
# made a project unindexable.
REFERENCE_SCHEMA_VERSION = 4

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
    references: list[ExtractedReference]
    declarations: list[ExtractedDeclarationShape]
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

    def backfill_references(
        self,
        project: ProjectInfo,
        *,
        wait_for_lock: bool = False,
        on_progress: Callable[[IndexProgress], None] | None = None,
    ) -> ReferenceBackfillReport:
        """Parse missing structural generations without embedding source chunks."""

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
                    return self._backfill_references_locked(project, progress=progress)
                finally:
                    progress.clear()
        except Timeout as exc:
            raise CodeIndexingError(
                ErrorCode.INDEX_BUSY,
                f"Another indexing job is already active: {project.name}",
                project=project.id,
            ) from exc

    def _backfill_references_locked(
        self, project: ProjectInfo, *, progress: ProgressPublisher
    ) -> ReferenceBackfillReport:
        # Backfill never embeds or re-parses chunks; it must not use its own
        # (always error-free) run to promote a project past whatever state a
        # prior full index earned it (S2 -- e.g. "partial" from a failed
        # index must stay "partial" until a real index run heals it).
        try:
            prior_state = self.store.project_state(project.id)
        except CodeIndexingError as exc:
            if exc.code is not ErrorCode.PROJECT_NOT_FOUND:
                raise
            # A marker-resolved project that was never registered -- e.g. it
            # has zero eligible source files, so `_project_is_stale` never
            # called `index()` to register it, or the data directory was
            # wiped while the on-disk marker survived (S11) -- has no prior
            # run to protect. `existing` below reads the same per-project
            # partition this lookup just proved absent, so it comes back
            # empty and the early "nothing missing" return fires before this
            # value would ever be read; it exists only so a future change to
            # that invariant fails safe instead of crashing.
            prior_state = "ready"
        existing = {record.path: record for record in self.store.list_files(project.id)}
        coverage_rows = self.store.reference_coverage(project.id)
        coverage = {
            row["file_id"]: ReferenceCoverage(
                file_id=row["file_id"],
                path=row["path"],
                content_hash=row["content_hash"],
                schema_version=row["schema_version"],
            )
            for row in coverage_rows
            if row["schema_version"] == REFERENCE_SCHEMA_VERSION
        }
        # Files that still carry rows from a schema version below the current
        # one. The version 4 bump was supposed to discard every generation
        # version 3 wrote (its colliding ids made a project unindexable), but
        # a file routed to `incomplete_paths` -- never re-covered -- kept its
        # old rows forever. Retire them below wherever such a file surfaces,
        # independent of whether its current content can be parsed at all.
        stale_schema_file_ids = {
            row["file_id"]
            for row in coverage_rows
            if row["schema_version"] != REFERENCE_SCHEMA_VERSION
        }
        missing = {
            record.file_id: record
            for record in existing.values()
            if (known := coverage.get(record.file_id)) is None
            or known.content_hash != record.content_hash
        }
        if not missing:
            return ReferenceBackfillReport(project_id=project.id, files_current=len(existing))

        progress.update(
            phase="extracting_references",
            files_total=len(missing),
            force=True,
        )
        job: StagingJob | None = None
        files_checked = 0
        files_backfilled = 0
        incomplete_paths: list[str] = []
        stale_paths: list[str] = []
        seen_file_ids: set[str] = set()

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

        try:
            for item in self.scanner.iter_scan(project, existing, read_contents=False):
                if not isinstance(item, ScannedFile):
                    continue
                record = existing.get(item.path.as_posix())
                if record is None or record.file_id not in missing:
                    continue
                seen_file_ids.add(record.file_id)
                files_checked += 1
                progress.update(
                    phase="extracting_references",
                    files_seen=files_checked,
                    current_path=record.path,
                )
                if record.has_errors:
                    if record.error is not None and record.error.startswith("rejected:"):
                        # Deliberate content rejection (binary/minified) will
                        # never parse -- it is not a parse failure, it is a
                        # permanent, intentional exclusion. A coverage-only
                        # row (zero references, current schema) is durable
                        # proof of that decision, so this file stops showing
                        # up as "missing" -- and being misreported as
                        # `parse_error` -- on every future backfill and query
                        # (S10).
                        staging_job().stage_references(
                            self._reference_rows(project.id, record, [], [])
                        )
                        staging_job().mark_references_replaced(record.file_id)
                        files_backfilled += 1
                        continue
                    # This file carries has_errors yet has no current coverage:
                    # it never produced a successful generation (a first-time
                    # failure, whose row keeps its own content_hash) or its
                    # coverage is otherwise absent. There is nothing trustworthy
                    # to keep -- any surviving rows would be from an unknown
                    # generation and could target unrelated text at their old
                    # byte offsets. Retire them and leave the file honestly
                    # uncovered until a successful index replaces chunks and
                    # references together. A *changed* file whose replacement
                    # failed keeps its previous content_hash and live
                    # references, so it never reaches this branch (S2).
                    staging_job().mark_references_replaced(record.file_id)
                    incomplete_paths.append(record.path)
                    continue
                try:
                    source = item.absolute_path.read_bytes()
                except OSError:
                    stale_paths.append(record.path)
                    continue
                if _content_rejection(source) is not None or _digest(source) != record.content_hash:
                    stale_paths.append(record.path)
                    continue
                try:
                    extraction = self.extractor.extract(item.path, item.language, source)
                except CodeIndexingError:
                    raise
                except Exception:
                    # A broken parser/query must not erase a prior structural
                    # generation -- it says nothing about whether this file's
                    # own content is valid. Leave this file uncovered so the
                    # next backfill retries it after the extractor is
                    # healthy, but still retire a retired-schema generation:
                    # that is wrong on its own terms, independent of whether
                    # today's extractor run succeeded.
                    if record.file_id in stale_schema_file_ids:
                        staging_job().mark_references_replaced(record.file_id)
                    incomplete_paths.append(record.path)
                    continue
                if extraction.has_errors:
                    # The bytes just read are confirmed (above) to match this
                    # file's current content_hash, so any reference rows
                    # already on file for it -- current schema or not -- are
                    # from a different generation than the one just proven
                    # invalid. Retire them: serving stale byte offsets against
                    # today's bytes is a wrong-edit hazard (S4), strictly
                    # worse than the honest "missing" this file already is.
                    staging_job().mark_references_replaced(record.file_id)
                    incomplete_paths.append(record.path)
                    continue
                staging_job().stage_references(
                    self._reference_rows(
                        project.id,
                        record,
                        extraction.references,
                        extraction.declarations,
                    )
                )
                staging_job().mark_references_replaced(record.file_id)
                files_backfilled += 1

            stale_paths.extend(
                record.path for file_id, record in missing.items() if file_id not in seen_file_ids
            )
            if stale_paths:
                # Do not publish a partial generation when the normal index
                # must first make a changed/deleted source and its embeddings
                # consistent with storage.
                if job is not None:
                    job.discard()
                return ReferenceBackfillReport(
                    project_id=project.id,
                    files_checked=files_checked,
                    files_backfilled=files_backfilled,
                    files_current=len(existing) - len(missing),
                    incomplete_paths=sorted(incomplete_paths),
                    stale_paths=sorted(set(stale_paths)),
                )
            progress.update(
                phase="committing",
                files_seen=files_checked,
                current_path=None,
                force=True,
            )
            if job is None:
                # Every file in `missing` errored or was rejected without
                # ever staging anything (a legacy pre-feature partition whose
                # only structural files fail to parse is exactly this case).
                # `has_reference_table` is only ever made true by a commit,
                # so without one it stays false forever and every future
                # `find_references`/`analyze_refactor` call is told to run
                # the exact backfill that just ran and cannot help (S8). An
                # otherwise-empty commit still creates the table.
                job = staging_job()
            self._commit_staged(project, job, errors=[], state=prior_state)
            return ReferenceBackfillReport(
                project_id=project.id,
                files_checked=files_checked,
                files_backfilled=files_backfilled,
                # `files_current` describes state *after* this report, not
                # work done *during* it, so a file this run just backfilled
                # counts the same as one that was already covered coming in
                # -- otherwise this call and the next idempotent one (which
                # sees it already covered and short-circuits before ever
                # computing `missing`) would report different totals for the
                # same converged project. Every file in `missing` reached
                # this point via either a successful backfill or
                # `incomplete_paths` (the only other outcome, `stale_paths`,
                # already returned above), so `existing` minus the latter is
                # exactly the files with current coverage now.
                files_current=len(existing) - len(incomplete_paths),
                incomplete_paths=sorted(incomplete_paths),
            )
        except BaseException:
            if job is not None:
                job.discard()
            raise

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
        # T1: this run's own reference-extraction cost and staged row count,
        # not the whole project's total (which a benchmark comparing scenarios
        # against the same project would otherwise report unchanged run to run).
        reference_extraction_ns = 0
        staged_reference_rows = 0
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
            job = staging_job()
            previous = existing.get(record.path)
            # A failed replacement keeps the previous generation live: the
            # chunks and references that remain in the tables describe the
            # previous content_hash, so the file row must keep describing them
            # too. Advancing the row to the new (failed) hash would leave the
            # row, chunks, and references on different generations -- the
            # internal divergence S1 tripped over. Only the latest observed
            # size/mtime and the error state advance.
            job.stage_file(
                record.model_copy(
                    update={
                        "content_hash": (
                            previous.content_hash if previous is not None else record.content_hash
                        ),
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
                    if target.record.has_errors:
                        # Extraction ran and produced chunks (best-effort,
                        # even with syntax errors), and those chunks --
                        # together with the file row's content_hash -- were
                        # just replaced above. The structural rows it would
                        # also produce are not trustworthy enough to stage,
                        # but leaving the *previous* generation's rows in
                        # place is worse: they would be served against bytes
                        # and a content_hash that no longer match what
                        # produced them, at whatever byte offsets the file
                        # happened to have before this edit (S4 -- a
                        # wrong-edit hazard for a caller like
                        # `analyze_refactor` that trusts those offsets).
                        # Retire them; the file heals once a later parse
                        # succeeds cleanly.
                        staging_job().mark_references_replaced(target.record.file_id)
                    else:
                        staging_job().stage_references(
                            self._reference_rows(
                                project.id,
                                target.record,
                                target.references,
                                target.declarations,
                            )
                        )
                        staging_job().mark_references_replaced(target.record.file_id)
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
                        # Content rejection is a skip, not a syntax/indexing error,
                        # but it must still leave a files row behind (S3): the
                        # scanner is path-based and never decodes content, so it
                        # keeps yielding this path on every future scan. Dropping
                        # the row entirely (as mark_removed would) makes
                        # current.keys() != existing.keys() true forever, which
                        # turns every reference query into a full re-index under
                        # the global lock. Persist a tombstone instead: a files
                        # row flagged has_errors with no chunks/references, so
                        # freshness checks see the path and (once size/mtime
                        # stop changing) treat it as unchanged.
                        rejected_record = StoredFile(
                            file_id=_digest(f"{project.id}\0{path}"),
                            project_id=project.id,
                            path=path,
                            language=item.language,
                            size=item.size,
                            mtime_ns=item.mtime_ns,
                            content_hash=content_hash,
                            has_errors=True,
                            error=f"rejected: {rejection}",
                            indexed_at=time.time_ns(),
                        )
                        with timer.measure("commit"):
                            staging_job().stage_file(rejected_record)
                            if previous is not None:
                                staging_job().mark_replaced(rejected_record.file_id)
                                staging_job().mark_references_replaced(rejected_record.file_id)
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
                    reference_extraction_ns += extraction.reference_extraction_ns
                    staged_reference_rows += len(extraction.references)
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
                            references=extraction.references,
                            declarations=extraction.declarations,
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
                        state=self._derive_index_state(project.id, errors),
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
            reference_extraction_duration_ms=reference_extraction_ns // 1_000_000,
            staged_reference_rows=staged_reference_rows,
        )

    def _derive_index_state(self, project_id: str, errors: list[IndexIssue]) -> str:
        """A project is partial while *this* run errored or any stored file row
        still records a genuine error.

        The stored rows are checked after the commit, so a file that failed
        this run is already visible there; the important case is the reverse:
        a no-op run with no fresh errors must not promote a project that still
        has stored file errors back to "ready" (it earned "partial" and only a
        real index run that heals every failed file may clear it). Rejection
        tombstones are skips, not errors, so they do not keep a project
        partial.
        """
        if errors:
            return "partial"
        if self.store.has_file_errors(project_id):
            return "partial"
        return "ready"

    def _commit_staged(
        self,
        project: ProjectInfo,
        job: StagingJob,
        *,
        errors: list[IndexIssue],
        state: str | None = None,
    ) -> None:
        """Apply a fully staged run, rolling the live tables back on any failure.

        The journal switches to ``committing`` -- with the versions to restore
        -- before the first live write, so a crash anywhere in this method is
        recoverable: the rollback here handles the live failure, and startup
        recovery handles a process death.

        ``state`` overrides the default ``"partial" if errors else "ready"``
        computation. A full index run always derives state from its own
        errors, but a reference backfill commits no chunks or embeddings of
        its own and must not overwrite a project state (e.g. ``partial`` from
        a prior failed index) that it did not itself earn (S2).
        """
        versions = self.store.table_versions(project.id)
        job.begin_commit(versions)
        try:
            self.store.replace_files_from_arrow(
                project.id,
                files=job.files_table(),
                chunk_batches=job.iter_chunk_batches(),
                reference_batches=job.iter_reference_batches(),
                removed_file_ids=job.removed_file_ids,
            )
            if job.replace_file_ids or job.replace_reference_file_ids or job.removed_file_ids:
                self.store.ensure_indexes(project.id)
            self.store.upsert_project(
                project,
                model_id=self.embedder.model_id,
                state=state if state is not None else self._derive_index_state(project.id, errors),
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
    def _reference_rows(
        project_id: str,
        file: StoredFile,
        references: list[ExtractedReference],
        declarations: list[ExtractedDeclarationShape],
    ) -> list[ReferenceRow]:
        """Build one deterministic structural generation for *file*.

        A coverage record is deliberately emitted even when the parser found
        no occurrences. It is the durable proof that the current structural
        schema parsed this exact content, rather than merely an empty query
        result from an unindexed legacy project.
        """

        rows: list[ReferenceRow] = []

        def identity(
            record_kind: str, kind: str | None, start_byte: int | None, end_byte: int | None
        ) -> str:
            # `kind` belongs in the digest because one byte range legitimately
            # carries two references: a superclass is both `inheritance` and a
            # `read`, and a decorator call is both `decorator` and `call`.
            # Omitting it gave those rows one id, and merge_insert rejects two
            # source rows matching a single target -- which permanently broke
            # every later incremental index of the project.
            return _digest(
                "\0".join(
                    (
                        file.file_id,
                        record_kind,
                        kind or "",
                        str(start_byte if start_byte is not None else -1),
                        str(end_byte if end_byte is not None else -1),
                        str(REFERENCE_SCHEMA_VERSION),
                    )
                )
            )

        for reference in references:
            rows.append(
                ReferenceRow(
                    reference_id=identity(
                        "reference", reference.kind, reference.start_byte, reference.end_byte
                    ),
                    record_kind="reference",
                    file_id=file.file_id,
                    project_id=project_id,
                    path=file.path,
                    language=file.language,
                    kind=reference.kind,
                    source_qualified_symbol=reference.source_qualified_symbol,
                    written_name=reference.written_name,
                    target_name=reference.target_name,
                    module_path=reference.module_path,
                    imported_name=reference.imported_name,
                    alias=reference.alias,
                    receiver_text=reference.receiver_text,
                    start_byte=reference.start_byte,
                    end_byte=reference.end_byte,
                    start_line=reference.start_line,
                    end_line=reference.end_line,
                    shape_json=(
                        reference.call_shape.model_dump_json()
                        if reference.call_shape is not None
                        else None
                    ),
                    content_hash=file.content_hash,
                    schema_version=REFERENCE_SCHEMA_VERSION,
                )
            )
        for declaration in declarations:
            rows.append(
                ReferenceRow(
                    reference_id=identity(
                        "declaration",
                        declaration.kind,
                        declaration.start_byte,
                        declaration.end_byte,
                    ),
                    record_kind="declaration",
                    file_id=file.file_id,
                    project_id=project_id,
                    path=file.path,
                    language=file.language,
                    kind=declaration.kind,
                    source_qualified_symbol=declaration.qualified_symbol,
                    written_name=declaration.symbol,
                    target_name=declaration.symbol,
                    module_path=None,
                    imported_name=None,
                    alias=None,
                    receiver_text=None,
                    start_byte=declaration.start_byte,
                    end_byte=declaration.end_byte,
                    start_line=declaration.start_line,
                    end_line=declaration.end_line,
                    shape_json=json.dumps(
                        [parameter.model_dump(mode="json") for parameter in declaration.parameters],
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    content_hash=file.content_hash,
                    schema_version=REFERENCE_SCHEMA_VERSION,
                )
            )
        rows.append(
            ReferenceRow(
                reference_id=identity("coverage", None, None, None),
                record_kind="coverage",
                file_id=file.file_id,
                project_id=project_id,
                path=file.path,
                language=file.language,
                kind=None,
                source_qualified_symbol=None,
                written_name=None,
                target_name=None,
                module_path=None,
                imported_name=None,
                alias=None,
                receiver_text=None,
                start_byte=None,
                end_byte=None,
                start_line=None,
                end_line=None,
                shape_json=None,
                content_hash=file.content_hash,
                schema_version=REFERENCE_SCHEMA_VERSION,
            )
        )
        return rows

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
