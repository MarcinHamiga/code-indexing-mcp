"""Explicit incremental indexing orchestration."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import psutil
from filelock import FileLock, Timeout

from . import __version__
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
from .git_state import (
    GitProbeOutcome,
    GitState,
    changed_paths_between,
    probe_git_state,
)
from .git_state import (
    slot_id as git_slot_id,
)
from .history import HistoryStore
from .models import (
    ExtractedChunk,
    ExtractedDeclarationShape,
    ExtractedReference,
    IndexIssue,
    IndexReport,
    IndexTrigger,
    ProjectInfo,
    ReferenceBackfillReport,
    ReferenceCoverage,
    RunAudit,
    ScannedFile,
    SkippedFile,
    StoredFile,
)
from .progress import IndexProgress, ProgressPublisher
from .scanner import SourceScanner
from .staging import ChunkRow, ReferenceRow, StagingJob
from .storage import SCHEMA_VERSION, ActiveIndexTarget, LanceStore, PartitionRef
from .update_check import checkout_head

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
# Version 5 adds Go to STRUCTURAL_LANGUAGES. Every language's version-bump step
# is what makes parse-only reference backfill re-extract that language's files
# (Go files already carried version-4 coverage rows with zero occurrences).
REFERENCE_SCHEMA_VERSION = 6

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
    """Return why *source* cannot be indexed, or None when it can."""
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


@dataclass(frozen=True)
class _GitGuard:
    """Git identity captured under the index lock and re-verified at commit.

    A guard of ``None`` means Git could not be probed at all; there is then no
    selector to verify, and the commit deliberately records no indexed HEAD so
    the next successful probe forces full freshness validation.
    """

    state: GitState
    slot_id: str

    @property
    def selector(self) -> str:
        return f"{self.state.selector_kind.value}:{self.state.selector_value}"


@dataclass
class _IndexScanState:
    """Mutable counters and staging state owned by one index scan."""

    current_paths: set[str] = field(default_factory=set)
    indexed: int = 0
    parsed: int = 0
    embedded: int = 0
    unchanged: int = 0
    metadata_only: int = 0
    removed: int = 0
    skipped: int = 0
    candidates_seen: int = 0
    bytes_read: int = 0
    chunks_extracted: int = 0
    chunks_staged: int = 0
    staged_bytes: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=dict)
    skipped_samples: list[str] = field(default_factory=list)
    fallback_count: int = 0
    errors: list[IndexIssue] = field(default_factory=list)
    job: StagingJob | None = None
    pending: list[_PendingFile] = field(default_factory=list)
    pending_chunks: int = 0
    pending_chars: int = 0
    reference_extraction_ns: int = 0
    staged_reference_rows: int = 0
    peak_memory_bytes: int = 0
    partition: PartitionRef | None = None

    def record_skip(self, path: str, reason: str) -> None:
        self.skipped += 1
        self.skipped_by_reason[reason] = self.skipped_by_reason.get(reason, 0) + 1
        if len(self.skipped_samples) < 20:
            self.skipped_samples.append(path)

    def add_pending(self, pending_file: _PendingFile) -> None:
        self.pending.append(pending_file)
        self.pending_chunks += len(pending_file.chunks)
        self.pending_chars += pending_file.source_chars

    def clear_pending(self) -> None:
        self.pending.clear()
        self.pending_chunks = 0
        self.pending_chars = 0

    def sample_memory(self, process: psutil.Process) -> None:
        # Diagnostics must never turn a successful index into a failure.
        with contextlib.suppress(psutil.Error):
            self.peak_memory_bytes = max(self.peak_memory_bytes, process.memory_info().rss)

    def to_report(self, project_id: str, timer: _PhaseTimer, batch_size: int) -> IndexReport:
        scan_ms = timer.milliseconds("scan")
        parse_ms = timer.milliseconds("parse")
        embed_ms = timer.milliseconds("embed")
        commit_ms = timer.milliseconds("commit")
        return IndexReport(
            project_id=project_id,
            discovered_files=len(self.current_paths),
            indexed_files=self.indexed,
            parsed_files=self.parsed,
            embedded_chunks=self.embedded,
            unchanged_files=self.unchanged,
            metadata_only_files=self.metadata_only,
            removed_files=self.removed,
            skipped_files=self.skipped,
            errors=self.errors,
            scan_duration_ms=scan_ms,
            parse_duration_ms=parse_ms,
            embed_duration_ms=embed_ms,
            commit_duration_ms=commit_ms,
            embedding_backend="cpu",
            embedding_batch_size=batch_size,
            scan_ms=scan_ms,
            parse_ms=parse_ms,
            embed_ms=embed_ms,
            commit_ms=commit_ms,
            fallback_count=self.fallback_count,
            peak_memory_bytes=self.peak_memory_bytes,
            reference_extraction_duration_ms=self.reference_extraction_ns // 1_000_000,
            staged_reference_rows=self.staged_reference_rows,
            failed_files=len(self.errors),
            skip_reasons=self.skipped_by_reason,
            skipped_samples=self.skipped_samples,
            bytes_read=self.bytes_read,
            chunks_extracted=self.chunks_extracted,
            chunks_staged=self.chunks_staged,
            staged_bytes=self.staged_bytes,
        )


class _RunRecord:
    """The audit trail of one indexing or backfill run.

    Owns the whole row lifecycle: the ``running`` insert and pre-run storage
    snapshot on ``start``, and the ``completed``/``failed`` finish with the
    post-run snapshot. A deferred record is started by the run itself once it
    knows real work is about to happen, so a no-op backfill never leaves a
    durable row behind to evict genuine runs from the bounded history window.

    Audit writes are never allowed to break an index (the same rule progress
    publishing follows): a full disk or a runs.sqlite locked past its busy
    timeout costs the audit row, not the run.
    """

    def __init__(
        self,
        *,
        history: HistoryStore | None,
        run_id: str,
        audit: Callable[[], RunAudit],
        snapshot: Callable[[], dict[str, int]],
    ) -> None:
        self._history = history
        self._run_id = run_id
        self._audit = audit
        self._snapshot = snapshot
        self._started = False
        self._storage_before: dict[str, int] = {}

    def start(self) -> None:
        if self._started or self._history is None:
            return
        self._started = True
        self._write(self._history.begin, self._audit())
        self._storage_before = self._snapshot()

    def complete(self, **counters: object) -> None:
        self._finish(state="completed", **counters)

    def fail(self) -> None:
        self._finish(state="failed")

    def _finish(self, *, state: str, **counters: object) -> None:
        if not self._started or self._history is None:
            return
        self._write(
            self._history.finish,
            self._run_id,
            state=state,
            finished_at=datetime.now(UTC).isoformat(),
            storage_before=self._storage_before,
            storage_after=self._snapshot(),
            **counters,
        )

    @staticmethod
    def _write(operation: Callable[..., None], *args: object, **kwargs: object) -> None:
        try:
            operation(*args, **kwargs)
        except (sqlite3.Error, OSError):
            logger.warning("Recording audit history failed; the run is unaffected", exc_info=True)


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
        history: HistoryStore | None = None,
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
        self.history = history

    def index(
        self,
        project: ProjectInfo,
        *,
        partition: PartitionRef | None = None,
        target: ActiveIndexTarget | None = None,
        force: bool = False,
        wait_for_lock: bool = False,
        on_progress: Callable[[IndexProgress], None] | None = None,
        trigger: IndexTrigger = "manual",
    ) -> IndexReport:
        if partition is None and target is not None:
            partition = target.partition
        elif partition is not None and target is not None and partition != target.partition:
            raise ValueError("index partition and target disagree")
        started = time.monotonic_ns()
        run_id = uuid.uuid4().hex
        self.lock_directory.mkdir(parents=True, exist_ok=True)
        global_lock = FileLock(self.lock_directory / "index-global.lock")
        project_lock = FileLock(self.lock_directory / f"{project.id}.lock")
        try:
            with (
                global_lock.acquire() if wait_for_lock else global_lock.acquire(timeout=0),
                project_lock.acquire() if wait_for_lock else project_lock.acquire(timeout=0),
            ):
                if partition is None:
                    try:
                        partition = self.store.active_partition(project.id)
                    except CodeIndexingError as exc:
                        if exc.code is not ErrorCode.PROJECT_NOT_FOUND:
                            raise
                        self.store.upsert_project(
                            project, model_id=self.embedder.model_id, state="pending"
                        )
                        partition = self.store.active_partition(project.id)
                if partition.project_id != project.id:
                    raise ValueError("index partition does not belong to project")
                # Capture the checkout identity before scanning. The guard is
                # verified again immediately before the staged commit, so a
                # branch that moved mid-scan can never publish rows into the
                # slot of a branch that no longer exists in the worktree.
                guard = self._capture_git_guard(project, partition)
                progress = ProgressPublisher(
                    project.id,
                    run_id=run_id,
                    trigger=trigger,
                    directory=self.progress_directory,
                    listener=on_progress,
                    slot_id=partition.slot_id,
                    activation_epoch=partition.activation_epoch,
                    selector=guard.selector if guard is not None else None,
                    expected_head=guard.state.head_oid if guard is not None else None,
                )
                rebuild_reason = self.store.incompatibility_reason(
                    project.id, self.embedder.model_id, partition_id=partition.partition_id
                )
                effective_trigger = trigger if rebuild_reason is None else "schema-rebuild"
                with self._recorded_run(
                    project,
                    run_id,
                    effective_trigger,
                    force=force,
                    rebuild_reason=rebuild_reason,
                    partition_id=partition.partition_id,
                ) as record:
                    if rebuild_reason is not None:
                        self._prepare_rebuild(project, rebuild_reason, partition)
                    try:
                        report = self._index_locked(
                            project,
                            partition=partition,
                            force=force,
                            progress=progress,
                            guard=guard,
                        )
                    finally:
                        # Only the run that holds the lock owns the snapshot, so
                        # clearing here can never delete another process's progress.
                        progress.clear()
                    report = report.model_copy(
                        update={
                            "run_id": run_id,
                            "trigger": effective_trigger,
                            "failed_files": len(report.errors),
                        }
                    )
                    record.complete(
                        phase_durations=self._phase_durations(report),
                        eligible_files=report.discovered_files,
                        changed_files=report.indexed_files,
                        unchanged_files=report.unchanged_files,
                        parsed_files=report.parsed_files,
                        failed_files=report.failed_files,
                        removed_files=report.removed_files,
                        skipped_total=report.skipped_files,
                        chunks_extracted=report.chunks_extracted,
                        chunks_embedded=report.embedded_chunks,
                        chunks_staged=report.chunks_staged,
                        staged_bytes=report.staged_bytes,
                        bytes_read=report.bytes_read,
                        skip_reasons=report.skip_reasons,
                        errors=report.errors[:20],
                        skipped_samples=report.skipped_samples[:20],
                        embedding_backend=report.embedding_backend,
                        embedding_fallback_reason=report.embedding_fallback_reason,
                        worker_used=report.worker_used,
                    )
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
        partition: PartitionRef | None = None,
        target: ActiveIndexTarget | None = None,
        wait_for_lock: bool = False,
        on_progress: Callable[[IndexProgress], None] | None = None,
        trigger: IndexTrigger = "reference-backfill",
    ) -> ReferenceBackfillReport:
        """Parse missing structural generations without embedding source chunks."""

        if partition is None and target is not None:
            partition = target.partition
        elif partition is not None and target is not None and partition != target.partition:
            raise ValueError("reference partition and target disagree")
        if partition is None:
            try:
                partition = self.store.active_partition(project.id)
            except CodeIndexingError as exc:
                if exc.code is ErrorCode.PROJECT_NOT_FOUND:
                    return ReferenceBackfillReport(project_id=project.id)
                raise
        if partition.project_id != project.id:
            raise ValueError("reference partition does not belong to project")
        if (
            self.store.incompatibility_reason(
                project.id, self.embedder.model_id, partition_id=partition.partition_id
            )
            is not None
        ):
            # An incompatible partition cannot be healed by a parse-only
            # backfill: its rows were written by an older model or schema.
            # Rebuild it with a full embedding run (recorded as its own
            # schema-rebuild audit row), then report every file current so
            # the caller proceeds to serve queries against the fresh index.
            self.index(
                project,
                partition=partition,
                wait_for_lock=wait_for_lock,
                on_progress=on_progress,
                trigger="schema-rebuild",
            )
            # A full rebuild can leave syntax-error files structurally
            # uncovered. Continue through normal backfill accounting so the
            # caller sees those coverage limitations instead of a false clean
            # report.

        self.lock_directory.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        global_lock = FileLock(self.lock_directory / "index-global.lock")
        project_lock = FileLock(self.lock_directory / f"{project.id}.lock")
        try:
            with (
                global_lock.acquire() if wait_for_lock else global_lock.acquire(timeout=0),
                project_lock.acquire() if wait_for_lock else project_lock.acquire(timeout=0),
                # Deferred: reference tools run a backfill on every query, and
                # the overwhelmingly common outcome is "nothing missing". Only
                # a backfill that actually starts work earns an audit row.
                self._recorded_run(
                    project,
                    run_id,
                    trigger,
                    deferred=True,
                    partition_id=partition.partition_id,
                ) as record,
            ):
                # Same contract as index(): capture the checkout identity
                # under the lock so a branch switch while waiting cannot
                # publish progress or staged rows for the wrong selector.
                guard = self._capture_git_guard(project, partition)
                progress = ProgressPublisher(
                    project.id,
                    run_id=run_id,
                    trigger=trigger,
                    directory=self.progress_directory,
                    listener=on_progress,
                    slot_id=partition.slot_id,
                    activation_epoch=partition.activation_epoch,
                    selector=guard.selector if guard is not None else None,
                    expected_head=guard.state.head_oid if guard is not None else None,
                )
                try:
                    report = self._backfill_references_locked(
                        project,
                        partition=partition,
                        progress=progress,
                        run_record=record,
                        guard=guard,
                    )
                finally:
                    progress.clear()
                record.complete(
                    eligible_files=report.files_current,
                    changed_files=report.files_backfilled,
                    failed_files=len(report.incomplete_paths),
                    # Failures live in failed_files and errors; the skip pair
                    # stays in lockstep (skipped_total == sum of the reasons).
                    skipped_total=len(report.stale_paths),
                    skip_reasons=({"stale": len(report.stale_paths)} if report.stale_paths else {}),
                    errors=[
                        IndexIssue(path=path, message="reference extraction incomplete")
                        for path in report.incomplete_paths[:20]
                    ],
                    skipped_samples=report.stale_paths[:20],
                )
                return report
        except Timeout as exc:
            raise CodeIndexingError(
                ErrorCode.INDEX_BUSY,
                f"Another indexing job is already active: {project.name}",
                project=project.id,
            ) from exc

    @contextlib.contextmanager
    def _recorded_run(
        self,
        project: ProjectInfo,
        run_id: str,
        trigger: IndexTrigger,
        *,
        force: bool = False,
        deferred: bool = False,
        rebuild_reason: str | None = None,
        partition_id: str | None = None,
    ) -> Iterator[_RunRecord]:
        """Record one run's audit trail around the body of a locked run.

        Entered after the writer locks are acquired, so a run that never got
        to start (INDEX_BUSY) records nothing. The body calls ``complete`` with
        its counters; any escaping exception marks the run ``failed`` before
        propagating. With ``deferred=True`` nothing is written until the body
        calls ``record.start()``, which it does when real work begins.
        ``rebuild_reason`` names why the partition was replaced, so the audit
        row explains a schema-rebuild run without mislabeling its trigger.
        """

        record = _RunRecord(
            history=self.history,
            run_id=run_id,
            audit=lambda: self._run_audit(
                project, run_id, trigger, force, rebuild_reason=rebuild_reason
            ),
            snapshot=lambda: self._storage_snapshot(project.id, partition_id=partition_id),
        )
        if not deferred:
            record.start()
        try:
            yield record
        except BaseException:
            record.fail()
            raise

    def _run_audit(
        self,
        project: ProjectInfo,
        run_id: str,
        trigger: IndexTrigger,
        force: bool,
        *,
        rebuild_reason: str | None = None,
    ) -> RunAudit:
        """The start-of-run audit row: identity and environment, nothing counted yet."""

        return RunAudit(
            run_id=run_id,
            project_id=project.id,
            trigger=trigger,
            server_version=__version__,
            git_revision=checkout_head(Path(__file__).resolve().parents[2]),
            model_id=self.embedder.model_id,
            schema_version=SCHEMA_VERSION,
            scan_config_hash=hashlib.sha256(project.scan.model_dump_json().encode()).hexdigest()[
                :16
            ],
            force=force,
            pid=os.getpid(),
            started_at=datetime.now(UTC).isoformat(),
            rebuild_reason=rebuild_reason,
        )

    def _prepare_rebuild(self, project: ProjectInfo, reason: str, partition: PartitionRef) -> None:
        """Delete *project*'s partition when its generation is incompatible.

        Called under the writer locks after the run's audit row is written.
        The caller records the reason before calling this method. The registry row and the
        ``.ci-mcp/project.toml`` marker survive the deletion, so a rebuild
        that fails or crashes leaves a registered, re-indexable project; the
        registry row is re-stamped to the current generation because no rows
        remain for the old generation's claim to describe.
        """
        self.store.delete_partition(
            project.id, model_id=self.embedder.model_id, partition_id=partition.partition_id
        )
        logger.warning("Rebuilding incompatible index for %s: %s", project.id, reason)

    @staticmethod
    def _phase_durations(report: IndexReport) -> dict[str, int]:
        durations: dict[str, int] = {}
        for phase in ("scan", "parse", "embed", "commit"):
            value = getattr(report, f"{phase}_ms")
            if value is not None:
                durations[phase] = value
        return durations

    def _storage_snapshot(
        self, project_id: str, *, partition_id: str | None = None
    ) -> dict[str, int]:
        """Best-effort pre/post table versions; empty when there is no partition yet."""

        with contextlib.suppress(Exception):
            physical = partition_id or project_id
            if (self.store.directory / "projects" / physical).exists():
                versions = self.store.table_versions(project_id, partition_id=partition_id)
                return {
                    "files": versions.files,
                    "chunks": versions.chunks,
                    "references": versions.references,
                }
        return {}

    def _backfill_references_locked(
        self,
        project: ProjectInfo,
        *,
        partition: PartitionRef,
        progress: ProgressPublisher,
        run_record: _RunRecord,
        guard: _GitGuard | None = None,
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
        existing = {
            record.path: record
            for record in self.store.list_files(project.id, partition_id=partition.partition_id)
        }
        coverage_rows = self.store.reference_coverage(
            project.id, partition_id=partition.partition_id
        )
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

        # Real work is about to start; only now does this run earn its audit
        # row. The storage snapshot inside start() still precedes every write.
        run_record.start()
        progress.update(
            phase="extracting_references",
            candidates_total=len(missing),
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
                    chunk_schema=LanceStore.chunk_arrow_schema(
                        self.store.vector_dimension, self.store.vector_dtype
                    ),
                    reference_schema=LanceStore.reference_arrow_schema(),
                    partition=partition,
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
                    candidates_seen=files_checked,
                    eligible_files=len(existing),
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
                candidates_seen=files_checked,
                candidates_total=files_checked,
                eligible_files=len(existing),
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
            # Structural rows describe the checkout the guard captured; a
            # reference backfill that outlived a branch switch must not pin
            # its generations into the moved-from slot.
            self._verify_git_guard(project, partition, guard)
            self._commit_staged(project, job, partition=partition, errors=[], state=prior_state)
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

    def _capture_git_guard(self, project: ProjectInfo, partition: PartitionRef) -> _GitGuard | None:
        """Snapshot the checkout identity this run will be verified against.

        Raises :class:`CodeIndexingError` with ``REPOSITORY_CHANGED`` when the
        checkout has already moved off *partition*'s slot: indexing it now
        would stage the wrong branch's rows, and the caller's retry re-resolves
        the target against the new selector.
        """
        state = probe_git_state(project.root, include_status=True)
        if state.probe is not GitProbeOutcome.GIT:
            return None
        if git_slot_id(project.id, state) != partition.slot_id:
            raise CodeIndexingError(
                ErrorCode.REPOSITORY_CHANGED,
                f"The Git checkout of {project.name} moved before indexing started",
                project=project.id,
            )
        return _GitGuard(state=state, slot_id=partition.slot_id)

    def _verify_git_guard(
        self, project: ProjectInfo, partition: PartitionRef, guard: _GitGuard | None
    ) -> None:
        """Refuse to publish rows the captured checkout no longer describes."""
        if guard is None:
            return
        state = probe_git_state(project.root)
        if state.probe is not GitProbeOutcome.GIT:
            # A transient Git failure cannot prove the checkout moved. The
            # staged rows still describe the working tree that was scanned,
            # and the commit records no indexed HEAD, so the next successful
            # probe forces full freshness validation.
            return
        if (
            git_slot_id(project.id, state) != guard.slot_id
            or state.head_oid != guard.state.head_oid
        ):
            raise CodeIndexingError(
                ErrorCode.REPOSITORY_CHANGED,
                f"The Git checkout of {project.name} changed while it was being indexed",
                project=project.id,
            )

    def _validation_plan(
        self, project: ProjectInfo, partition: PartitionRef, guard: _GitGuard | None
    ) -> tuple[frozenset[str], bool]:
        """Paths whose stored size/mtime must not be trusted this run.

        A same-slot HEAD advance validates only the paths the commits touched,
        plus whatever Git currently reports dirty or untracked. A diff that
        cannot be computed, or a slot with no recorded HEAD or clean state,
        validates every path; content hashes are still reused, so unchanged
        files pay a read and a digest, never a re-parse or re-embed.
        """
        if guard is None:
            return frozenset(), False
        state = guard.state
        verify_paths: set[str] = {*state.dirty_paths, *state.untracked_paths}
        slot = self.store.get_slot(partition.slot_id)
        if slot is None:
            return frozenset(verify_paths), False
        if slot.indexed_head is None or slot.indexed_clean is None:
            return frozenset(verify_paths), True
        if state.head_oid is not None and slot.indexed_head != state.head_oid:
            changed = changed_paths_between(
                project.root,
                slot.indexed_head,
                state.head_oid,
                project_prefix=state.project_prefix,
            )
            if changed is None:
                return frozenset(verify_paths), True
            verify_paths |= changed
        return frozenset(verify_paths), False

    def _index_locked(
        self,
        project: ProjectInfo,
        *,
        partition: PartitionRef,
        force: bool,
        progress: ProgressPublisher,
        guard: _GitGuard | None = None,
    ) -> IndexReport:
        slot_row = self.store.get_slot(partition.slot_id)
        previous_state = slot_row.state if slot_row is not None else "pending"
        verify_paths, verify_all = self._validation_plan(project, partition, guard)
        try:
            self.store.upsert_project(project, model_id=self.embedder.model_id, state="indexing")
            self.store.set_slot_state(partition, "indexing")
            context = (
                self.passage_session_factory()
                if self.passage_session_factory is not None
                else contextlib.nullcontext(self.embedder)
            )
            with context as passage_embedder:
                report = self._index_scan(
                    project,
                    partition=partition,
                    force=force,
                    passage_embedder=passage_embedder,
                    progress=progress,
                    verify_paths=verify_paths,
                    verify_all=verify_all,
                    guard=guard,
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
        except Exception as exc:
            # Never leave the project stuck in "indexing" after a crash. A
            # repository that moved under the run is retryable rather than
            # failed, so its slot returns to the pre-run state instead of
            # "error" and keeps its last-good indexed HEAD.
            fallback = (
                previous_state
                if isinstance(exc, CodeIndexingError) and exc.code is ErrorCode.REPOSITORY_CHANGED
                else "error"
            )
            with contextlib.suppress(Exception):
                self.store.upsert_project(project, model_id=self.embedder.model_id, state=fallback)
                self.store.set_slot_state(partition, fallback)
            raise

    def _staging_job(self, project: ProjectInfo, state: _IndexScanState) -> StagingJob:
        if state.job is None:
            if state.partition is None:
                raise RuntimeError("index scan has no pinned partition")
            state.job = StagingJob(
                self.staging_directory,
                project.id,
                file_schema=LanceStore.file_arrow_schema(),
                chunk_schema=LanceStore.chunk_arrow_schema(
                    self.store.vector_dimension, self.store.vector_dtype
                ),
                reference_schema=LanceStore.reference_arrow_schema(),
                partition=state.partition,
            )
            state.job.begin()
        return state.job

    def _stage_file_failure(
        self,
        project: ProjectInfo,
        existing: dict[str, StoredFile],
        state: _IndexScanState,
        record: StoredFile,
        exc: Exception,
    ) -> None:
        # Failures are reported separately from skips, whose total must equal
        # the sum of their reason counts.
        state.errors.append(IndexIssue(path=record.path, message=str(exc)))
        previous = existing.get(record.path)
        # Retained chunks and references still describe the previous content, so
        # a failed replacement must keep that generation's hash on the file row.
        self._staging_job(project, state).stage_file(
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

    def _flush_pending(
        self,
        project: ProjectInfo,
        existing: dict[str, StoredFile],
        passage_embedder: PassageEmbedder,
        progress: ProgressPublisher,
        timer: _PhaseTimer,
        process: psutil.Process,
        state: _IndexScanState,
    ) -> None:
        if not state.pending:
            return
        candidates = [
            _PendingCandidate(owner, chunk)
            for owner, pending_file in enumerate(state.pending)
            for chunk in pending_file.chunks
        ]
        # Embedding is the longest gap between progress updates, so announce it
        # before starting each pending batch.
        progress.update(phase="embedding", current_path=None, force=True)
        if state.partition is None:
            raise RuntimeError("index scan has no pinned partition")
        slot_id = state.partition.slot_id
        for group in _candidate_groups(candidates):
            active = [
                candidate for candidate in group if state.pending[candidate.owner].error is None
            ]
            if not active:
                continue
            with timer.measure("embed"):
                try:
                    succeeded, failed, retries = self._embed_candidates(passage_embedder, active)
                finally:
                    state.sample_memory(process)
            state.fallback_count += retries
            progress.update(phase="embedding")
            staged_rows: dict[int, list[ChunkRow]] = {}
            for candidate, segments in succeeded:
                target = state.pending[candidate.owner]
                if target.error is not None:
                    continue
                windowed = self._windowed_chunks([candidate.chunk], [segments])
                rows = [
                    self._chunk_row(project.id, target.record, chunk, vector, slot_id=slot_id)
                    for chunk, vector in windowed
                ]
                staged_rows.setdefault(candidate.owner, []).extend(rows)
                target.embedded_chunks += len(rows)
                target.emitted_chars += sum(len(chunk.content) for chunk, _ in windowed)
            for candidate, exc in failed:
                target = state.pending[candidate.owner]
                if target.error is None:
                    target.error = exc
            with timer.measure("commit"):
                for owner in sorted(staged_rows):
                    rows = staged_rows[owner]
                    self._staging_job(project, state).stage_chunks(rows)
                    state.chunks_staged += len(rows)
                    state.staged_bytes += sum(len(row.content.encode("utf-8")) for row in rows)

        with timer.measure("commit"):
            for target in state.pending:
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
                    self._stage_file_failure(project, existing, state, target.record, target.error)
                    continue
                job = self._staging_job(project, state)
                job.stage_file(target.record)
                job.mark_replaced(target.record.file_id)
                if target.record.has_errors:
                    # Syntax-error chunks are useful for search, but their structural
                    # offsets are unsafe. Retire any previous reference generation.
                    job.mark_references_replaced(target.record.file_id)
                else:
                    job.stage_references(
                        self._reference_rows(
                            project.id,
                            target.record,
                            target.references,
                            target.declarations,
                        )
                    )
                    job.mark_references_replaced(target.record.file_id)
                state.indexed += 1
                state.embedded += target.embedded_chunks
        progress.update(
            phase="embedding",
            changed_files=state.indexed,
            chunks_embedded=state.embedded,
            chunks_staged=state.chunks_staged,
            staged_bytes=state.staged_bytes,
            force=True,
        )
        state.clear_pending()

    def _process_scanned_file(
        self,
        project: ProjectInfo,
        item: ScannedFile,
        existing: dict[str, StoredFile],
        *,
        force: bool,
        passage_embedder: PassageEmbedder,
        progress: ProgressPublisher,
        timer: _PhaseTimer,
        process: psutil.Process,
        state: _IndexScanState,
        verify_paths: frozenset[str] = frozenset(),
        verify_all: bool = False,
    ) -> None:
        path = item.path.as_posix()
        previous = existing.get(path)
        if (
            not force
            and not verify_all
            and path not in verify_paths
            and previous is not None
            and previous.size == item.size
            and previous.mtime_ns == item.mtime_ns
        ):
            # Size and mtime alone trusted the filesystem's version of a
            # branch switch or reset: identical metadata can hide different
            # content. Paths the validation plan named -- every path after a
            # HEAD advance it could not diff, and Git's current dirty and
            # untracked paths -- fall through to the content-hash comparison
            # below instead, which reuses stored generations byte-for-byte.
            state.unchanged += 1
            return

        content_hash: str | None = None
        record: StoredFile | None = None
        try:
            with timer.measure("scan"):
                source = (
                    item.content if item.content is not None else item.absolute_path.read_bytes()
                )
                state.bytes_read += len(source)
                rejection = _content_rejection(source)
                content_hash = _digest(source)
            if rejection is not None:
                # A tombstone makes rejected content converge to unchanged on later
                # scans while carrying no chunks or structural rows.
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
                    job = self._staging_job(project, state)
                    job.stage_file(rejected_record)
                    if previous is not None:
                        job.mark_replaced(rejected_record.file_id)
                        job.mark_references_replaced(rejected_record.file_id)
                state.record_skip(path, rejection)
                return
            if not force and previous is not None and previous.content_hash == content_hash:
                with timer.measure("commit"):
                    self._staging_job(project, state).stage_file(
                        previous.model_copy(update={"size": item.size, "mtime_ns": item.mtime_ns})
                    )
                state.metadata_only += 1
                return
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
            state.parsed += 1
            state.chunks_extracted += len(extraction.chunks)
            state.reference_extraction_ns += extraction.reference_extraction_ns
            state.staged_reference_rows += len(extraction.references)
            source_chars = sum(len(chunk.content) for chunk in extraction.chunks)
        except Exception as exc:
            if isinstance(exc, CodeIndexingError) and exc.code in ENVIRONMENT_ERROR_CODES:
                # Environment failures must abort the run rather than poison the
                # file that happened to be active.
                raise
            failed_record = record or StoredFile(
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
            )
            with timer.measure("commit"):
                self._stage_file_failure(project, existing, state, failed_record, exc)
        else:
            # This block sits outside the except above on purpose. The flush
            # has already staged chunk rows for the pending files when it can
            # fail, so swallowing its failure as this file's error would leave
            # those files queued and re-stage them after other files --
            # splitting one file across batches (or duplicating its rows).
            # Flush failures must abort the run with their own error.
            if state.pending and (
                len(state.pending) >= CANDIDATE_GROUP_COUNT
                or state.pending_chunks + len(extraction.chunks) > CANDIDATE_GROUP_COUNT
                or state.pending_chars + source_chars > CANDIDATE_GROUP_CHARS
            ):
                self._flush_pending(
                    project,
                    existing,
                    passage_embedder,
                    progress,
                    timer,
                    process,
                    state,
                )
            state.add_pending(
                _PendingFile(
                    record=record.model_copy(update={"has_errors": extraction.has_errors}),
                    chunks=extraction.chunks,
                    references=extraction.references,
                    declarations=extraction.declarations,
                    source_chars=source_chars,
                )
            )

    def _index_scan(
        self,
        project: ProjectInfo,
        *,
        partition: PartitionRef,
        force: bool,
        passage_embedder: PassageEmbedder,
        progress: ProgressPublisher,
        verify_paths: frozenset[str] = frozenset(),
        verify_all: bool = False,
        guard: _GitGuard | None = None,
    ) -> IndexReport:
        timer = _PhaseTimer()
        with timer.measure("scan"):
            existing = {
                record.path: record
                for record in self.store.list_files(project.id, partition_id=partition.partition_id)
            }
        # Streaming scans do not know their total until the walk finishes.
        progress.update(phase="scanning", force=True)
        state = _IndexScanState(partition=partition)
        process = psutil.Process()
        state.sample_memory(process)
        stream = self.scanner.iter_scan(project, existing)
        try:
            while True:
                with timer.measure("scan"):
                    item = next(stream, None)
                if item is None:
                    break
                state.candidates_seen += 1
                if isinstance(item, SkippedFile):
                    state.record_skip(item.path.as_posix(), item.reason)
                    progress.update(
                        phase="scanning",
                        candidates_seen=state.candidates_seen,
                        skipped_total=state.skipped,
                        skipped_by_reason=state.skipped_by_reason,
                        current_path=None,
                    )
                    continue
                path = item.path.as_posix()
                state.current_paths.add(path)
                progress.update(
                    phase="scanning",
                    candidates_seen=state.candidates_seen,
                    eligible_files=len(state.current_paths),
                    unchanged_files=state.unchanged,
                    current_path=path,
                )
                self._process_scanned_file(
                    project,
                    item,
                    existing,
                    force=force,
                    passage_embedder=passage_embedder,
                    progress=progress,
                    timer=timer,
                    process=process,
                    state=state,
                    verify_paths=verify_paths,
                    verify_all=verify_all,
                )

            self._flush_pending(
                project,
                existing,
                passage_embedder,
                progress,
                timer,
                process,
                state,
            )
            progress.update(
                phase="committing",
                candidates_seen=state.candidates_seen,
                candidates_total=state.candidates_seen,
                eligible_files=len(state.current_paths),
                unchanged_files=state.unchanged,
                parsed_files=state.parsed,
                failed_files=len(state.errors),
                bytes_read=state.bytes_read,
                chunks_extracted=state.chunks_extracted,
                skipped_total=state.skipped,
                skipped_by_reason=state.skipped_by_reason,
                current_path=None,
                force=True,
            )
            with timer.measure("commit"):
                # The staged rows describe the checkout the guard captured
                # before scanning; publish them only if it still does.
                self._verify_git_guard(project, partition, guard)
                for path, record in existing.items():
                    if path not in state.current_paths:
                        self._staging_job(project, state).mark_removed(record.file_id)
                        state.removed += 1
                if state.job is not None:
                    self._commit_staged(
                        project, state.job, partition=partition, errors=state.errors
                    )
                else:
                    final_state = self._derive_index_state(
                        project.id, state.errors, partition_id=partition.partition_id
                    )
                    self.store.upsert_project(
                        project,
                        model_id=self.embedder.model_id,
                        state=final_state,
                    )
                    self.store.set_slot_state(partition, final_state, project=project)
        except BaseException:
            # Discard staged bytes unless a failed rollback left a committing
            # journal that startup recovery still needs.
            if state.job is not None:
                state.job.discard()
            raise
        return state.to_report(project.id, timer, self.batch_size)

    def _derive_index_state(
        self, project_id: str, errors: list[IndexIssue], *, partition_id: str | None = None
    ) -> str:
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
        if self.store.has_file_errors(project_id, partition_id=partition_id):
            return "partial"
        return "ready"

    def _commit_staged(
        self,
        project: ProjectInfo,
        job: StagingJob,
        *,
        partition: PartitionRef,
        errors: list[IndexIssue],
        state: str | None = None,
    ) -> None:
        """Apply a fully staged run, rolling the live tables back on any failure.

        The journal switches to ``committing`` -- with the versions to restore
        -- before the first live write, so a crash anywhere in this method is
        recoverable: the rollback here handles the live failure, and startup
        recovery handles a process death.

        ``state`` lets reference backfill preserve an existing project state;
        backfill commits no chunks or embeddings and therefore cannot promote a
        project whose earlier indexing failure left it partial.
        """
        if job.partition != partition:
            raise ValueError("staged job and commit partition do not match")
        versions = self.store.table_versions(project.id, partition_id=partition.partition_id)
        job.begin_commit(versions)
        try:
            with self.store.partition_access(project.id, partition_id=partition.partition_id):
                self.store.replace_files_from_arrow(
                    project.id,
                    files=job.files_table(),
                    chunk_batches=job.iter_chunk_batches(),
                    reference_batches=job.iter_reference_batches(),
                    removed_file_ids=job.removed_file_ids,
                    partition_id=partition.partition_id,
                )
                if job.replace_file_ids or job.replace_reference_file_ids or job.removed_file_ids:
                    self.store.ensure_indexes(project.id, partition_id=partition.partition_id)
                final_state = (
                    state
                    if state is not None
                    else self._derive_index_state(
                        project.id, errors, partition_id=partition.partition_id
                    )
                )
                self.store.upsert_project(
                    project,
                    model_id=self.embedder.model_id,
                    state=final_state,
                )
                self.store.set_slot_state(partition, final_state, project=project)
        except BaseException:
            try:
                with self.store.partition_access(project.id, partition_id=partition.partition_id):
                    self.store.restore_versions(
                        project.id, versions, partition_id=partition.partition_id
                    )
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
        # search_suffix (the normalized identifier terms) is deliberately not
        # recomputed per window: it derives from the path and qualified name,
        # which are the same for every window of the chunk.
        return chunk.model_copy(
            update={
                "start_byte": start_byte,
                "end_byte": start_byte + len(content.encode("utf-8")),
                "start_line": start_line,
                "end_line": start_line + content.count("\n"),
                "content": content,
                "embedding_text": embedding_text,
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
        project_id: str,
        file: StoredFile,
        chunk: ExtractedChunk,
        vector: bytes,
        *,
        slot_id: str,
    ) -> ChunkRow:
        identity = "\0".join(
            [
                slot_id,
                file.file_id,
                file.content_hash,
                chunk.kind,
                chunk.qualified_symbol or "",
                str(chunk.start_byte),
                str(chunk.end_byte),
                str(chunk.part_index),
            ]
        )
        # The logical project prefix routes get_chunk to the owning registry
        # entry; the slot participates in the digest so equal content in two
        # physical branch partitions cannot share a selector.
        return ChunkRow(
            chunk_id=f"{project_id}:{_digest(identity)}",
            file_id=file.file_id,
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
            # The normalized identifier terms are the compact search payload:
            # they replace the persisted search_text without copying the
            # source content a second time.
            identifier_terms=chunk.search_suffix,
            part_index=chunk.part_index,
            vector=vector,
            content_hash=file.content_hash,
        )
