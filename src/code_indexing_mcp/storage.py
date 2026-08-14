"""Partitioned LanceDB persistence for projects, files, chunks, and references."""

from __future__ import annotations

import gc
import itertools
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict, cast

import lancedb
import pyarrow as pa
from filelock import FileLock
from lancedb.index import FTS, BTree, HnswSq
from lancedb.query import ColumnOrdering, FullTextOperator, MultiMatchQuery
from lancedb.table import LanceTable

from .errors import CodeIndexingError, ErrorCode
from .models import (
    ChunkPreview,
    CodeChunk,
    FragmentLengthStats,
    FragmentStats,
    IndexedChunk,
    IndexStorageStats,
    ProjectInfo,
    ProjectStorageStats,
    StoredChunk,
    StoredFile,
    TableStorageStats,
)
from .projects import existing_marker_path, rooted_under, same_project_root

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 4

# Symbol lookups over-fetch because the LIKE pushdown over-matches; these bound
# how many rows are scanned before the exact filter and the caller's limit apply.
OVERFETCH_FACTOR = 10
MINIMUM_OVERFETCH = 200

# Open partitions kept resident. Each entry holds two LanceTable handles and their
# caches, and nothing evicted them before: the daemon is long-lived and get_chunk
# walks every registered project, so one call could fault in every project a user
# has ever indexed. Sixteen covers the projects one developer works across while
# keeping the ceiling independent of how many they have registered.
MAX_CACHED_PARTITIONS = 16

# Upper bound on how many partitions one hybrid query reads at once. More
# concurrency than this has not shown a latency win in benchmarks and raises
# contention on the shared partition cache, so the pool stays small and bounded.
_SEARCH_CONCURRENCY = 8

# Columns get_chunk reads. The vector and the identifier-terms column are
# excluded: nothing outside indexing and ranking can use them, and reading them
# made a single-chunk fetch an order of magnitude larger than the code it
# returned. project_id is injected from the owning partition. content_hash
# remains on the chunk row so a read cannot combine separate commit generations.
CHUNK_PAYLOAD_COLUMNS = [
    "chunk_id",
    "file_id",
    "path",
    "language",
    "kind",
    "symbol",
    "qualified_symbol",
    "parent_symbol",
    "start_byte",
    "end_byte",
    "start_line",
    "end_line",
    "content",
    "content_hash",
    "part_index",
]

# Every chunk column except the vector. list_chunks has no production caller and its
# test callers read text and offsets, so decoding vectors was pure waste.
INDEXED_CHUNK_COLUMNS = [
    "chunk_id",
    "file_id",
    "path",
    "language",
    "kind",
    "symbol",
    "qualified_symbol",
    "parent_symbol",
    "start_byte",
    "end_byte",
    "start_line",
    "end_line",
    "content",
    "identifier_terms",
    "content_hash",
    "part_index",
]


def _quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _file_ids_condition(file_ids: Iterable[str]) -> str:
    """A predicate matching every file in *file_ids* with one IN list."""
    return f"file_id IN ({', '.join(_quoted(file_id) for file_id in file_ids)})"


# The batched commit replaces each file's rows with one merge_insert whose
# ``when_not_matched_by_source_delete("file_id IN (...)")`` predicate must
# delete only the predicate's unmatched target rows, never rows of untouched
# files. Every supported lancedb release (0.34.0 and up) filters per row, but a
# regression to the older all-or-nothing gate behavior would silently delete
# every untouched file's rows on a multi-file project's second run, so the
# store refuses to commit unless a cheap scratch-table probe confirms the
# semantics. The probe is checked on the first batched commit, not on store
# construction: read-only processes (status checks, searches, metrics) never
# run the merge and must not pay for it. See
# docs/plans/2026-07-27-review-followups-index.md.
_batched_merge_semantics_cache: bool | None = None
_batched_merge_semantics_lock = threading.Lock()


def _probe_batched_merge_semantics() -> bool:
    """Return whether the installed lancedb filters source-deletes per row.

    Mirrors the real commit shape: one merge on the key with a predicate that
    also covers a sibling whose rows are absent from the source. Per-row
    semantics leave the untouched file's row and delete the predicate's
    unmatched rows; gate semantics leave only the matched replacement.
    """
    try:
        with tempfile.TemporaryDirectory() as directory:
            database = lancedb.connect(directory)
            table = database.create_table(
                "probe",
                pa.table(
                    {
                        "file_id": ["a", "b", "c"],
                        "chunk_id": ["a1", "b1", "c1"],
                        "vector": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    }
                ),
            )
            source = pa.table({"file_id": ["a"], "chunk_id": ["a2"], "vector": [[0.0, 0.0]]})
            (
                table.merge_insert("chunk_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete("file_id IN ('a', 'b')")
                .execute(source)
            )
            surviving = sorted(row["chunk_id"] for row in table.search().to_list())
            return surviving == ["a2", "c1"]
    except Exception as exc:
        logger.warning(
            "Merge-insert semantics probe failed (%s); batched commits are refused",
            exc,
        )
        logger.debug("Merge-insert semantics probe failure details", exc_info=True)
        return False


def _batched_merge_semantics_ok() -> bool:
    """Cached :func:`_probe_batched_merge_semantics`, once per process."""
    global _batched_merge_semantics_cache
    if _batched_merge_semantics_cache is None:
        with _batched_merge_semantics_lock:
            if _batched_merge_semantics_cache is None:
                _batched_merge_semantics_cache = _probe_batched_merge_semantics()
                logger.debug(
                    "Batched merge-insert semantics verified on lancedb %s",
                    getattr(lancedb, "__version__", "unknown"),
                )
    return _batched_merge_semantics_cache


def _nullable_int(value: Any) -> int | None:
    """Coerce a fragment-length statistic, tolerating None and non-finite noise."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _normalized_timestamp(value: str) -> str:
    """Normalize a Lance version timestamp string to ISO-8601."""
    try:
        return datetime.fromisoformat(value).isoformat()
    except ValueError:
        return value


# Git inspection for storage-status worktree warnings is best-effort and must
# never make the status command fail. A short timeout keeps a hung repository
# from holding the command hostage.
_GIT_TIMEOUT_SECONDS = 5.0
_GitRunner = Callable[[list[str], Path], str | None]


def _run_git_quietly(command: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _directory_bytes(directory: Path) -> int:
    """Sum the file bytes under *directory* without following symlinks.

    A symlinked directory or file inside a partition is a deliberate escape
    hatch, not storage the index owns, so its target's bytes must not be
    counted as physical index storage.
    """
    total = 0
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                total += _directory_bytes(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                total += entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
    return total


def overlap_warnings(projects: Iterable[ProjectInfo]) -> list[str]:
    """Warn about registered roots that duplicate or nest one another.

    Read-only detection: rejecting new overlaps is a later-phase registration
    concern, but status must already say when two registrations point at the
    same or a contained directory.
    """
    warnings: list[str] = []
    for left, right in itertools.combinations(list(projects), 2):
        if same_project_root(left.root, right.root):
            warnings.append(
                f"Projects {left.id!r} and {right.id!r} register the same root: {left.root}"
            )
            continue
        left_resolved = left.root.expanduser().resolve()
        right_resolved = right.root.expanduser().resolve()
        try:
            left_resolved.relative_to(right_resolved)
        except ValueError:
            try:
                right_resolved.relative_to(left_resolved)
            except ValueError:
                continue
            warnings.append(
                f"Project {left.id!r} root {left_resolved} contains the root of project "
                f"{right.id!r} ({right_resolved})"
            )
        else:
            warnings.append(
                f"Project {left.id!r} root {left_resolved} is nested inside the root of "
                f"project {right.id!r} ({right_resolved})"
            )
    return warnings


def overlapping_registration(projects: Iterable[ProjectInfo], root: Path) -> ProjectInfo | None:
    """Return an existing project whose root equals, contains, or nests in *root*.

    Every overlap kind indexes the same sources twice, so registration checks
    all three. Containment goes through rooted_under, whose samefile boundary
    check makes a differently-cased spelling of one directory count as overlap
    on case-insensitive filesystems, matching how same_project_root treats
    equality. Detection only: rejecting a new overlap is the registration
    layer's decision, and existing overlapping registrations stay valid.
    """
    root = root.expanduser().resolve()
    for project in projects:
        existing = project.root.expanduser().resolve()
        if same_project_root(existing, root):
            return project
        if rooted_under(root, existing) or rooted_under(existing, root):
            return project
    return None


def worktree_warnings(
    projects: Iterable[ProjectInfo], *, _run: _GitRunner | None = None
) -> list[str]:
    """Warn about registered roots that are checkouts of one Git repository.

    Two roots whose ``--show-toplevel`` differs but whose Git common directory
    is the same are worktrees (or a main checkout and a worktree) of one
    repository, which is a likely duplicate registration. All failures are
    swallowed: this is advisory information only.
    """
    runner = _run or _run_git_quietly
    repositories: list[tuple[ProjectInfo, Path, Path]] = []
    for project in projects:
        toplevel = runner(["git", "rev-parse", "--show-toplevel"], project.root)
        common = runner(["git", "rev-parse", "--git-common-dir"], project.root)
        if toplevel is None or common is None:
            continue
        common_path = Path(common)
        if not common_path.is_absolute():
            # --git-common-dir is relative to the query cwd, which is the
            # registered root itself, not the repository toplevel: a root that
            # is a subdirectory of a checkout reports '../.git'.
            common_path = (Path(project.root) / common_path).resolve()
        repositories.append((project, Path(toplevel), common_path))
    warnings: list[str] = []
    for (left, left_top, left_common), (right, right_top, right_common) in itertools.combinations(
        repositories, 2
    ):
        if same_project_root(left_common, right_common) and not same_project_root(
            left_top, right_top
        ):
            warnings.append(
                f"Projects {left.id!r} and {right.id!r} share Git common directory "
                f"{left_common} from different checkouts (possible worktrees of one repository)"
            )
    return warnings


def _symbol_matches(chunk: ChunkPreview, name: str, match: str) -> bool:
    """Apply exact symbol-match semantics that the SQL pre-filter cannot."""
    candidate = chunk.qualified_symbol or chunk.symbol or ""
    symbol = chunk.symbol or ""
    if match == "exact":
        return candidate == name or symbol == name
    if match == "prefix":
        return candidate.startswith(name) or symbol.startswith(name)
    return name in candidate or name in symbol


@dataclass(frozen=True)
class _ProjectTables:
    files: LanceTable
    chunks: LanceTable
    references: LanceTable | None
    generation: int


class ReferenceRecord(TypedDict):
    reference_id: str
    record_kind: str
    file_id: str
    project_id: str
    path: str
    language: str
    kind: str | None
    source_qualified_symbol: str | None
    written_name: str | None
    target_name: str | None
    module_path: str | None
    imported_name: str | None
    alias: str | None
    receiver_text: str | None
    start_byte: int | None
    end_byte: int | None
    start_line: int | None
    end_line: int | None
    shape_json: str | None
    content_hash: str
    schema_version: int


@dataclass(frozen=True)
class TableVersions:
    """A point-in-time snapshot of a project partition's three tables."""

    files: int
    chunks: int
    references: int


class LanceStore:
    def __init__(
        self,
        directory: Path,
        *,
        vector_dimension: int = 768,
        vector_index: str = "exact",
    ) -> None:
        self.directory = directory
        self.vector_dimension = vector_dimension
        self.vector_index = vector_index
        legacy_rows = self._migrate_v1(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(directory / "registry", read_consistency_interval=timedelta(0))
        self._projects = self._table(self._db, "projects", self._project_schema())
        self._partitions: OrderedDict[str, _ProjectTables] = OrderedDict()
        self._partitions_lock = threading.Lock()
        for row in legacy_rows:
            row = {
                **row,
                "vector_dimension": vector_dimension,
                "schema_version": SCHEMA_VERSION,
                "state": "pending",
                "updated_at": time.time_ns(),
            }
            self._merge(self._projects, "id", [row])

    def upsert_project(self, project: ProjectInfo, *, model_id: str, state: str = "ready") -> None:
        existing = self._rows(self._projects, f"id = {_quoted(project.id)}")
        if existing:
            registered_root = Path(str(existing[0]["root"])).resolve()
            incoming_root = project.root.resolve()
            same_root = same_project_root(registered_root, incoming_root)
            if not same_root and existing_marker_path(registered_root) is not None:
                raise CodeIndexingError(
                    ErrorCode.PROJECT_ID_CONFLICT,
                    "The project ID is already active at another path",
                    project=project.id,
                    registered_root=str(registered_root),
                    incoming_root=str(incoming_root),
                )
            if same_root:
                project = project.model_copy(update={"root": registered_root})
        if (
            existing
            and str(existing[0]["state"]) != "pending"
            and self.incompatibility_reason(project.id, model_id) is not None
        ):
            # A reconstructable generation mismatch (different embedding
            # model, vector dimension, or index schema version) marks the
            # partition for automatic rebuild instead of raising
            # INDEX_INCOMPATIBLE. The stored generation fields are left
            # untouched: they describe the rows still live in the
            # partition until the rebuild deletes and re-stamps them.
            self.mark_rebuild_required(project.id)
            return
        row = {
            "id": project.id,
            "name": project.name,
            "root": str(project.root),
            "payload": project.model_dump_json(),
            "model_id": model_id,
            "vector_dimension": self.vector_dimension,
            "schema_version": SCHEMA_VERSION,
            "state": state,
            "updated_at": time.time_ns(),
        }
        # A no-op upsert -- project discovery, a status check, a state that did
        # not change -- must not churn registry versions. The comparison
        # excludes updated_at so a real mutation still stamps it fresh. It is
        # a typed comparison, not a string coercion, so a stored null can
        # never be mistaken for the literal string "None".
        if existing and all(
            existing[0][column] == row[column] for column in row if column != "updated_at"
        ):
            return
        self._merge(self._projects, "id", [row])

    def incompatibility_reason(self, project_id: str, model_id: str) -> str | None:
        """Return why *project_id*'s stored generation must be rebuilt, or None.

        A stored embedding model, vector dimension, or index schema version
        that differs from this store's current values describes rows this
        build cannot serve coherently. Rebuilding -- deleting the partition
        and re-indexing -- recovers it, so the mismatch is returned as a
        reason rather than raised as the hard ``INDEX_INCOMPATIBLE`` failure
        it used to be. A pending registration has no live rows to describe,
        so it is never incompatible.
        """
        rows = self._rows(self._projects, f"id = {_quoted(project_id)}")
        if not rows:
            return None
        row = rows[0]
        if str(row["state"]) == "pending":
            return None
        differences: list[str] = []
        if str(row["model_id"]) != model_id:
            differences.append(f"embedding model {row['model_id']!r} -> {model_id!r}")
        if int(row["vector_dimension"]) != self.vector_dimension:
            differences.append(
                f"vector dimension {row['vector_dimension']} -> {self.vector_dimension}"
            )
        if int(row["schema_version"]) != SCHEMA_VERSION:
            differences.append(f"index schema version {row['schema_version']} -> {SCHEMA_VERSION}")
        return "; ".join(differences) if differences else None

    def mark_rebuild_required(self, project_id: str) -> None:
        """Stamp *project_id*'s registry state ``rebuild_required``.

        Read-only callers keep working -- the state surfaces through status
        and queries -- but nothing serves the partition again until an index
        run rebuilds it. The stored generation fields are not touched: they
        describe the rows still live in the partition.
        """
        rows = self._rows(self._projects, f"id = {_quoted(project_id)}")
        if not rows:
            return
        if str(rows[0]["state"]) == "rebuild_required":
            return
        row = dict(rows[0])
        row["state"] = "rebuild_required"
        row["updated_at"] = time.time_ns()
        self._merge(self._projects, "id", [row])

    def delete_partition(self, project_id: str, *, model_id: str) -> bool:
        """Delete *project_id*'s partition while preserving its registration.

        Evicts the cached table handles before removing the partition
        directory. The registry row is re-stamped to the current generation
        and ``indexing`` state: with the partition gone, no rows remain for
        the old generation's claim to describe, and the replacement is about
        to be written by the calling run. Returns False when the project is
        not registered.
        """
        with self.partition_access(project_id):
            self._advance_partition_generation(project_id)
            with self._partitions_lock:
                self._partitions.pop(project_id, None)
            partition = self.directory / "projects" / project_id
            if partition.exists():
                shutil.rmtree(partition)
            rows = self._rows(self._projects, f"id = {_quoted(project_id)}")
            if not rows:
                return False
            row = dict(rows[0])
            row["model_id"] = model_id
            row["vector_dimension"] = self.vector_dimension
            row["schema_version"] = SCHEMA_VERSION
            row["state"] = "indexing"
            row["updated_at"] = time.time_ns()
            self._merge(self._projects, "id", [row])
            return True

    def list_projects(self) -> list[ProjectInfo]:
        return [
            ProjectInfo.model_validate_json(row["payload"]) for row in self._rows(self._projects)
        ]

    def project_state(self, project_id: str) -> str:
        rows = self._rows(self._projects, f"id = {_quoted(project_id)}")
        if not rows:
            raise CodeIndexingError(ErrorCode.PROJECT_NOT_FOUND, f"Unknown project: {project_id}")
        return str(rows[0]["state"])

    def list_files(self, project_id: str) -> list[StoredFile]:
        tables = self._existing_tables(project_id)
        if tables is None:
            return []
        return [StoredFile.model_validate(row) for row in self._rows(tables.files)]

    def has_file_errors(self, project_id: str) -> bool:
        """Whether any stored file row records a genuine indexing error.

        Rejection tombstones ("rejected: ...") are deliberate, permanent skips,
        not errors, so they do not count. Reads only the error rows instead of
        materializing every file in the project.
        """
        tables = self._existing_tables(project_id)
        if tables is None:
            return False
        return any(
            not str(row["error"] or "").startswith("rejected:")
            for row in self._rows(tables.files, "has_errors = true")
        )

    def upsert_file(self, record: StoredFile) -> None:
        self._merge(
            self._tables(record.project_id).files,
            "file_id",
            [record.model_dump()],
        )

    def replace_file(self, record: StoredFile, chunks: list[StoredChunk]) -> None:
        tables = self._tables(record.project_id)
        condition = f"file_id = {_quoted(record.file_id)}"
        chunks = [
            chunk
            if chunk.content_hash
            else chunk.model_copy(update={"content_hash": record.content_hash})
            for chunk in chunks
        ]
        chunks = [
            chunk
            if chunk.content_hash
            else chunk.model_copy(update={"content_hash": record.content_hash})
            for chunk in chunks
        ]
        if chunks:
            (
                tables.chunks.merge_insert("chunk_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete(condition)
                .execute([chunk.model_dump() for chunk in chunks])
            )
        else:
            tables.chunks.delete(condition)
        self.upsert_file(record)

    def remove_file(self, project_id: str, file_id: str) -> None:
        tables = self._tables(project_id)
        condition = f"file_id = {_quoted(file_id)}"
        tables.chunks.delete(condition)
        if tables.references is None:
            raise RuntimeError("Reference table is missing from an interrupted transaction")
        tables.references.delete(condition)
        tables.files.delete(condition)

    def table_versions(self, project_id: str) -> TableVersions:
        """Snapshot every partition table's version before a commit begins."""
        tables = self._tables(project_id)
        if tables.references is None:
            raise RuntimeError("Reference table is missing from an interrupted transaction")
        return TableVersions(
            files=tables.files.version,
            chunks=tables.chunks.version,
            references=tables.references.version,
        )

    def restore_versions(
        self,
        project_id: str,
        versions: TableVersions,
        *,
        restore_references: bool = True,
    ) -> bool:
        """Return every partition table to *versions*' data.

        ``restore`` followed by ``checkout_latest`` makes the recorded version
        the live one; restoring a table that is already at that version's data
        is a no-op, so repeated recovery over the same journal is idempotent.

        Returns False when the partition no longer exists -- a project removed
        since the journal was written has nothing left to roll back. Recovery
        must not go through the create-on-write path here: materialising an
        empty partition would leave a version the journal can never name.
        """
        tables = self._existing_tables(project_id)
        if tables is None:
            return False
        tables.files.restore(versions.files)
        tables.chunks.restore(versions.chunks)
        if restore_references:
            if tables.references is None:
                raise RuntimeError("Reference table is missing from an interrupted transaction")
            tables.references.restore(versions.references)
        tables.files.checkout_latest()
        tables.chunks.checkout_latest()
        if restore_references:
            if tables.references is None:
                raise RuntimeError("Reference table is missing from an interrupted transaction")
            tables.references.checkout_latest()
        return True

    def mark_project_state(self, project_id: str, state: str) -> bool:
        """Set a registered project's state, leaving its other columns alone.

        Returns False when the project is not registered. Recovery uses this
        to flag a project whose rollback could not be completed, since it only
        has the ID from the journal rather than a full ProjectInfo.
        """
        rows = self._rows(self._projects, f"id = {_quoted(project_id)}")
        if not rows:
            return False
        if str(rows[0]["state"]) == state:
            return True
        row = dict(rows[0])
        row["state"] = state
        row["updated_at"] = time.time_ns()
        self._merge(self._projects, "id", [row])
        return True

    def replace_files_from_arrow(
        self,
        project_id: str,
        *,
        files: pa.Table,
        chunk_batches: Iterable[tuple[list[str], pa.Table]],
        reference_batches: Iterable[tuple[list[str], pa.Table]] = (),
        removed_file_ids: Iterable[str] = (),
    ) -> None:
        """Commit staged Arrow batches without materializing chunk objects.

        Each batch carries its full affected ``file_ids`` predicate: one
        ``merge_insert`` runs per non-empty batch (one batched ``delete`` when
        the whole batch has no rows), so table versions scale with O(batches)
        rather than O(files). ``when_not_matched_by_source_delete`` removes a
        file's previous rows both when its ids changed and when the file now
        extracts to nothing, since every file in the predicate is either
        present in the source or intentionally empty. Replacement ids win over
        removal ids, and removed files are deleted in one predicate per table.
        The vector columns stay fixed-size-list float32 arrays end to end.

        Refuses to commit when the installed lancedb fails the batched
        merge-insert semantics probe, because a regression would delete rows
        of untouched files rather than fail loudly.
        """
        if not _batched_merge_semantics_ok():
            raise CodeIndexingError(
                ErrorCode.UNSUPPORTED_RUNTIME,
                "The installed lancedb version does not filter "
                "when_not_matched_by_source_delete rows the way batched commits "
                "require; refusing to commit because it could delete rows of "
                "untouched files. Upgrade lancedb and retry.",
            )
        tables = self._tables(project_id)
        replacement_ids: list[str] = []
        chunk_iter = iter(chunk_batches)
        try:
            for file_ids, chunks in chunk_iter:
                replacement_ids.extend(file_ids)
                condition = _file_ids_condition(file_ids)
                if chunks.num_rows:
                    (
                        tables.chunks.merge_insert("chunk_id")
                        .when_matched_update_all()
                        .when_not_matched_insert_all()
                        .when_not_matched_by_source_delete(condition)
                        .execute(chunks)
                    )
                else:
                    tables.chunks.delete(condition)
        finally:
            close = getattr(chunk_iter, "close", None)
            if close is not None:
                close()
        if tables.references is None:
            raise RuntimeError("Reference table is missing from an interrupted transaction")
        reference_iter = iter(reference_batches)
        try:
            for file_ids, references in reference_iter:
                replacement_ids.extend(file_ids)
                condition = _file_ids_condition(file_ids)
                if references.num_rows:
                    (
                        tables.references.merge_insert("reference_id")
                        .when_matched_update_all()
                        .when_not_matched_insert_all()
                        .when_not_matched_by_source_delete(condition)
                        .execute(references)
                    )
                else:
                    tables.references.delete(condition)
        finally:
            close = getattr(reference_iter, "close", None)
            if close is not None:
                close()
        if files.num_rows:
            self._merge(tables.files, "file_id", files)
        replaced = set(replacement_ids)
        removed = [file_id for file_id in removed_file_ids if file_id not in replaced]
        if removed:
            condition = _file_ids_condition(removed)
            tables.chunks.delete(condition)
            tables.references.delete(condition)
            tables.files.delete(condition)

    def list_reference_records(
        self,
        project_id: str,
        *,
        version: int | None = None,
        schema_version: int | None = None,
        record_kinds: Iterable[str] | None = None,
    ) -> list[ReferenceRecord]:
        """Return structural rows from the requested immutable table version.

        Optional schema and record-kind filters are pushed into the table query.
        Omitting them deliberately returns every row for recovery and raw-storage
        callers. Reference classification requests reference and coverage rows;
        declarations are fetched separately through narrower methods below.
        """
        self._validate_schema_version(schema_version)
        conditions: list[str] = []
        if schema_version is not None:
            conditions.append(f"schema_version = {schema_version}")
        if record_kinds is not None:
            kinds = sorted(set(record_kinds))
            if not kinds:
                return []
            values = ", ".join(_quoted(kind) for kind in kinds)
            conditions.append(f"record_kind IN ({values})")
        condition = " AND ".join(conditions) if conditions else None
        return self._reference_rows(project_id, condition, version=version)

    def reference_coverage(
        self, project_id: str, *, version: int | None = None
    ) -> list[ReferenceRecord]:
        return self._reference_rows(project_id, "record_kind = 'coverage'", version=version)

    def reference_version(self, project_id: str) -> int:
        """Return the current structural snapshot without creating a partition."""
        tables = self._existing_tables(project_id)
        if tables is None or tables.references is None:
            return 0
        return int(tables.references.version)

    def has_reference_table(self, project_id: str) -> bool:
        """True when the references table exists for *project_id*.

        Distinguishes a legitimately empty reference index (the table
        exists, `ensure_reference_index` has run, there is simply nothing
        to report) from one that was never built at all -- a legacy
        partition indexed before this feature existed, or one whose
        `ensure_reference_index` was skipped. `_reference_rows` and
        `reference_version` collapse both cases to `[]`/`0`, so callers
        that need the distinction must ask this directly rather than trust an
        empty result.
        """
        tables = self._existing_tables(project_id)
        return tables is not None and tables.references is not None

    def coverage_for_file(
        self, project_id: str, file_id: str, schema_version: int
    ) -> list[ReferenceRecord]:
        self._validate_schema_version(schema_version)
        return self._reference_rows(
            project_id,
            "record_kind = 'coverage' "
            f"AND file_id = {_quoted(file_id)} AND schema_version = {schema_version}",
        )

    # Declarations can be narrowed by exact symbol, target name, or candidate
    # file. References cannot be narrowed the same way: aliases and renamed
    # re-exports give downstream rows arbitrary local target names, so finding a
    # conservative subset requires an iterative module-graph walk.
    def declaration_shapes(
        self,
        project_id: str,
        qualified_symbol: str,
        *,
        schema_version: int | None = None,
        version: int | None = None,
    ) -> list[ReferenceRecord]:
        self._validate_schema_version(schema_version)
        condition = (
            f"record_kind = 'declaration' AND source_qualified_symbol = {_quoted(qualified_symbol)}"
        )
        if schema_version is not None:
            condition = f"{condition} AND schema_version = {schema_version}"
        return self._reference_rows(project_id, condition, version=version)

    def target_name_candidates(
        self,
        project_id: str,
        target_name: str,
        *,
        record_kind: str | None = None,
        schema_version: int | None = None,
        version: int | None = None,
    ) -> list[ReferenceRecord]:
        self._validate_schema_version(schema_version)
        condition = f"target_name = {_quoted(target_name)}"
        if record_kind is not None:
            condition = f"record_kind = {_quoted(record_kind)} AND {condition}"
        if schema_version is not None:
            condition = f"{condition} AND schema_version = {schema_version}"
        return self._reference_rows(project_id, condition, version=version)

    def declarations_for_files(
        self,
        project_id: str,
        file_ids: Iterable[str],
        *,
        schema_version: int | None = None,
        version: int | None = None,
    ) -> list[ReferenceRecord]:
        """Return declaration rows for exactly the given files.

        `_lexical_declaration`/class-scope resolution only ever compares a
        declaration against a reference row in the *same* file, so callers
        that already narrowed to a candidate file set never need the whole
        project's declaration table.
        """
        self._validate_schema_version(schema_version)
        ids = sorted(set(file_ids))
        if not ids:
            return []
        values = ", ".join(_quoted(file_id) for file_id in ids)
        condition = f"record_kind = 'declaration' AND file_id IN ({values})"
        if schema_version is not None:
            condition = f"{condition} AND schema_version = {schema_version}"
        return self._reference_rows(project_id, condition, version=version)

    def list_chunks(self, project_ids: Iterable[str] | None = None) -> list[IndexedChunk]:
        ids = list(project_ids or [project.id for project in self.list_projects()])
        chunks: list[IndexedChunk] = []
        for project_id in ids:
            tables = self._existing_tables(project_id)
            if tables is None:
                continue
            chunks.extend(
                IndexedChunk.model_validate(row)
                for row in cast(
                    list[dict[str, Any]],
                    tables.chunks.search().select(INDEXED_CHUNK_COLUMNS).to_list(),
                )
            )
        return chunks

    def get_chunk(self, chunk_id: str) -> CodeChunk | None:
        # New-format ids carry a project-routing prefix
        # ("<project-id>:<digest>"), so the owning partition is opened directly
        # instead of scanning every registered project. A malformed id, one
        # whose prefix names no partition, or one from a pre-migration
        # generation is treated as unknown -- consistent with the existing
        # contract that chunk ids change when a file is re-indexed.
        project_id = self._chunk_project_id(chunk_id)
        if project_id is None:
            return None
        tables = self._existing_tables(project_id)
        if tables is None:
            return None
        rows = cast(
            list[dict[str, Any]],
            tables.chunks.search()
            .where(f"chunk_id = {_quoted(chunk_id)}")
            .select(CHUNK_PAYLOAD_COLUMNS)
            .to_list(),
        )
        if not rows:
            return None
        row = dict(rows[0])
        row["project_id"] = project_id
        return CodeChunk.model_validate(row)

    @staticmethod
    def _chunk_id_prefix(chunk_id: str) -> str | None:
        """The project-routing prefix of a chunk id, or None when malformed."""
        if ":" not in chunk_id:
            return None
        prefix, _, remainder = chunk_id.partition(":")
        if not prefix or not remainder:
            return None
        return prefix

    def _chunk_project_id(self, chunk_id: str) -> str | None:
        """Resolve a chunk id's routing prefix to its partition's project id.

        The prefix is the project id itself, and the partition directory is
        named by it, so resolution is a single directory probe -- no registry
        scan, and no partition is materialized for an unknown id.
        """
        prefix = self._chunk_id_prefix(chunk_id)
        if prefix is None:
            return None
        if (self.directory / "projects" / prefix).is_dir():
            return prefix
        return None

    def count_chunks(self, project_ids: Iterable[str] | None = None) -> int:
        ids = list(project_ids or [project.id for project in self.list_projects()])
        tables = (self._existing_tables(project_id) for project_id in ids)
        return sum(table.chunks.count_rows() for table in tables if table is not None)

    def hybrid_search(
        self,
        query_text: str,
        vector: list[float],
        project_ids: Iterable[str],
        condition: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        ids = list(project_ids)
        if not ids:
            return []
        # Independent partitions are read concurrently through a small bounded
        # pool, so a multi-project query costs the slowest partition plus the
        # merge rather than the sum of every partition. Results are reassembled
        # in request order so relevance-score ties break exactly as the
        # sequential implementation did.
        results: dict[int, list[dict[str, Any]]] = {}
        if len(ids) == 1:
            results[0] = self._hybrid_search_rows(ids[0], query_text, vector, condition, limit)
        else:
            workers = min(len(ids), _SEARCH_CONCURRENCY)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    index: pool.submit(
                        self._hybrid_search_rows, project_id, query_text, vector, condition, limit
                    )
                    for index, project_id in enumerate(ids)
                }
                for index, future in futures.items():
                    results[index] = future.result()
        rows = [row for index in range(len(ids)) for row in results.get(index, [])]
        rows.sort(key=lambda row: float(row.get("_relevance_score", 0.0)), reverse=True)
        return rows[:limit]

    def _hybrid_search_rows(
        self,
        project_id: str,
        query_text: str,
        vector: list[float],
        condition: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Run the hybrid query for one partition, injecting its project id."""
        tables = self._existing_tables(project_id)
        if tables is None:
            return []
        # MultiMatchQuery spans the two single-column FTS indexes (content
        # and identifier_terms); a plain string would silently search only
        # one of them whenever both exist.
        query = (
            tables.chunks.search(query_type="hybrid", vector_column_name="vector")
            .vector(vector)
            .text(
                MultiMatchQuery(
                    query_text,
                    ["content", "identifier_terms"],
                    boosts=None,
                    operator=FullTextOperator.OR,
                )
            )
        )
        if condition:
            query = query.where(condition, prefilter=True)
        query = (
            query.limit(limit)
            .select(
                [
                    "chunk_id",
                    "path",
                    "language",
                    "kind",
                    "symbol",
                    "qualified_symbol",
                    "parent_symbol",
                    "start_line",
                    "end_line",
                    "content",
                ]
            )
            .rerank()
        )
        if self.vector_index == "exact":
            query = query.bypass_vector_index()
        # project_id is not stored on chunk rows; it belongs to the
        # partition being searched, so it is injected per project.
        query_rows = cast(list[dict[str, Any]], query.to_list())
        for row in query_rows:
            row["project_id"] = project_id
        return query_rows

    def find_symbol_chunks(
        self,
        name: str,
        project_id: str,
        *,
        match: str,
        kinds: list[str] | None,
        limit: int,
    ) -> list[ChunkPreview]:
        escaped = _quoted(name)
        if match == "exact":
            symbol = f"(qualified_symbol = {escaped} OR symbol = {escaped})"
        elif match == "prefix":
            prefix = _quoted(name + "%")
            symbol = f"(qualified_symbol LIKE {prefix} OR symbol LIKE {prefix})"
        else:
            contains = _quoted("%" + name + "%")
            symbol = f"(qualified_symbol LIKE {contains} OR symbol LIKE {contains})"
        conditions = [symbol]
        if kinds:
            values = ", ".join(_quoted(kind) for kind in kinds)
            conditions.append(f"kind IN ({values})")
        tables = self._existing_tables(project_id)
        if tables is None:
            return []
        # LIKE is only a pushdown pre-filter. The query engine ignores escape
        # sequences, so `_` and `%` inside an identifier stay wildcards and the
        # predicate over-matches (`load_user` also matches `loadXuser`). It never
        # under-matches, so exact semantics are re-applied below. Over-fetch so
        # the caller's limit is applied to real matches in a stable order.
        scan_limit = max(limit * OVERFETCH_FACTOR, MINIMUM_OVERFETCH)
        rows = self._projected_chunks(
            tables.chunks,
            " AND ".join(conditions),
            limit=scan_limit,
            content=True,
            order_by=["path", "start_line", "kind"],
        )
        for row in rows:
            row["project_id"] = project_id
        if len(rows) == scan_limit:
            # The pre-filter filled the scan window, so real matches sorting
            # after it were never seen. Silent truncation is otherwise
            # indistinguishable from "no more matches exist".
            logger.debug(
                "Symbol pre-filter for %r in project %s hit the %d-row scan cap; "
                "later exact matches may be missing",
                name,
                project_id,
                scan_limit,
            )
        matches = [
            preview
            for preview in (ChunkPreview.model_validate(row) for row in rows)
            if _symbol_matches(preview, name, match)
        ]
        return matches[:limit]

    def outline_chunks(self, path: str, project_id: str) -> list[ChunkPreview]:
        tables = self._existing_tables(project_id)
        if tables is None:
            return []
        condition = (
            f"path = {_quoted(path)} AND symbol IS NOT NULL AND qualified_symbol IS NOT NULL"
        )
        rows = self._projected_chunks(
            tables.chunks,
            condition,
            limit=None,
            content=False,
        )
        for row in rows:
            row["project_id"] = project_id
        return [ChunkPreview.model_validate(row) for row in rows]

    def ensure_indexes(self, project_id: str) -> None:
        """Create missing search indexes on the write path.

        Index *existence* is a correctness requirement for search, so missing
        FTS and BTree indexes are created after a commit. Compaction, index
        optimization, and old-version cleanup are deliberately not part of this
        method: they run only in ``maintain_project`` on a schedule, so an
        incremental commit stops paying for a full optimize pass. Searches
        combine indexed rows with the unindexed tail until maintenance folds
        that tail into the indexes.
        """
        tables = self._tables(project_id)
        chunks = tables.chunks
        indices = list(chunks.list_indices())
        indexed_columns = {column for index in indices for column in index.columns}
        # Native FTS in the supported lancedb range indexes one field per
        # index, so the persisted source content and the compact normalized
        # identifier terms get one FTS index each and searches use a
        # MultiMatchQuery spanning both. That covers code-text keywords and
        # camelCase/snake_case symbol and path names without storing a second
        # full copy of the text.
        for fts_column in ("content", "identifier_terms"):
            if fts_column not in indexed_columns:
                chunks.create_index(
                    fts_column,
                    config=FTS(lower_case=True, stem=False, remove_stop_words=False),
                    replace=False,
                )
        for column in ("file_id", "language", "path", "symbol"):
            if column not in indexed_columns:
                chunks.create_index(column, config=BTree(), replace=False)
        vector_indices = [index for index in indices if "vector" in index.columns]
        if self.vector_index == "exact":
            for index in vector_indices:
                chunks.drop_index(index.name)
        elif not vector_indices and chunks.count_rows() >= 20_000:
            chunks.create_index(
                "vector",
                config=HnswSq(distance_type="cosine"),
                replace=False,
            )
        if tables.references is None:
            raise RuntimeError("Reference table is missing from an interrupted transaction")
        reference_indices = list(tables.references.list_indices())
        indexed_reference_columns = {
            column for index in reference_indices for column in index.columns
        }
        for column in (
            "file_id",
            "record_kind",
            "target_name",
            "module_path",
            "kind",
            "source_qualified_symbol",
            "schema_version",
        ):
            if column not in indexed_reference_columns:
                tables.references.create_index(column, config=BTree(), replace=False)

    def maintain_project(self, project_id: str, *, cleanup_older_than: timedelta) -> bool:
        """Compact and clean *project_id*'s partition, returning whether it ran.

        Optimizes the files, chunks, and references tables, reclaiming space
        after deletions. Versions older than ``cleanup_older_than`` are removed
        once verified; ``delete_unverified`` is never set, and a zero age must
        never be passed by an automatic path. A registered project with no
        partition is left untouched (``False``) rather than materialized.
        """
        tables = self._existing_tables(project_id)
        if tables is None:
            return False
        tables.files.optimize(cleanup_older_than=cleanup_older_than)
        tables.chunks.optimize(cleanup_older_than=cleanup_older_than)
        if tables.references is not None:
            tables.references.optimize(cleanup_older_than=cleanup_older_than)
        return True

    def maintain_registry(self, *, cleanup_older_than: timedelta) -> None:
        """Compact and clean the project registry table."""
        self._projects.optimize(cleanup_older_than=cleanup_older_than)

    def remove_project(self, project_id: str) -> bool:
        existed = bool(self._rows(self._projects, f"id = {_quoted(project_id)}"))
        self._projects.delete(f"id = {_quoted(project_id)}")
        with self._partitions_lock:
            self._partitions.pop(project_id, None)
        partition = self.directory / "projects" / project_id
        if partition.exists():
            shutil.rmtree(partition)
        return existed

    @staticmethod
    def _project_schema() -> pa.Schema:
        return pa.schema(
            [
                ("id", pa.string()),
                ("name", pa.string()),
                ("root", pa.string()),
                ("payload", pa.string()),
                ("model_id", pa.string()),
                ("vector_dimension", pa.int32()),
                ("schema_version", pa.int32()),
                ("state", pa.string()),
                ("updated_at", pa.int64()),
            ]
        )

    @staticmethod
    def _file_schema() -> pa.Schema:
        return pa.schema(
            [
                ("file_id", pa.string()),
                ("project_id", pa.string()),
                ("path", pa.string()),
                ("language", pa.string()),
                ("size", pa.int64()),
                ("mtime_ns", pa.int64()),
                ("content_hash", pa.string()),
                ("has_errors", pa.bool_()),
                ("error", pa.string()),
                ("indexed_at", pa.int64()),
            ]
        )

    @staticmethod
    def _chunk_schema(vector_dimension: int) -> pa.Schema:
        return pa.schema(
            [
                ("chunk_id", pa.string()),
                ("file_id", pa.string()),
                ("path", pa.string()),
                ("language", pa.string()),
                ("kind", pa.string()),
                ("symbol", pa.string()),
                ("qualified_symbol", pa.string()),
                ("parent_symbol", pa.string()),
                ("start_byte", pa.int64()),
                ("end_byte", pa.int64()),
                ("start_line", pa.int32()),
                ("end_line", pa.int32()),
                ("content", pa.string()),
                # Compact normalized identifier terms (camelCase and snake_case
                # split apart) that FTS searches alongside ``content``, so
                # symbol, path, and qualified-name queries match without
                # storing a second full copy of the source text.
                ("identifier_terms", pa.string()),
                ("content_hash", pa.string()),
                ("part_index", pa.int32()),
                (
                    "vector",
                    pa.list_(pa.float32(), vector_dimension),
                ),
            ]
        )

    @staticmethod
    def _reference_schema() -> pa.Schema:
        return pa.schema(
            [
                ("reference_id", pa.string()),
                ("record_kind", pa.string()),
                ("file_id", pa.string()),
                ("project_id", pa.string()),
                ("path", pa.string()),
                ("language", pa.string()),
                ("kind", pa.string()),
                ("source_qualified_symbol", pa.string()),
                ("written_name", pa.string()),
                ("target_name", pa.string()),
                ("module_path", pa.string()),
                ("imported_name", pa.string()),
                ("alias", pa.string()),
                ("receiver_text", pa.string()),
                ("start_byte", pa.int64()),
                ("end_byte", pa.int64()),
                ("start_line", pa.int32()),
                ("end_line", pa.int32()),
                ("shape_json", pa.string()),
                ("content_hash", pa.string()),
                ("schema_version", pa.int32()),
            ]
        )

    def _cached(self, project_id: str) -> _ProjectTables | None:
        """Return the cached partition for *project_id*, marking it recently used."""
        with self._partitions_lock:
            cached = self._partitions.get(project_id)
            if cached is not None:
                self._partitions.move_to_end(project_id)
            return cached

    @contextmanager
    def partition_access(self, project_id: str) -> Iterator[None]:
        """Prevent a destructive rebuild from invalidating an active query."""
        directory = self.directory.parent / "locks"
        directory.mkdir(parents=True, exist_ok=True)
        with FileLock(directory / f"partition-{project_id}.lock"):
            yield

    @contextmanager
    def partitions_access(self, project_ids: Iterable[str]) -> Iterator[None]:
        """Hold partition locks in a stable order for a multi-project query."""
        with ExitStack() as stack:
            for project_id in sorted(set(project_ids)):
                stack.enter_context(self.partition_access(project_id))
            yield

    def _remember(self, project_id: str, tables: _ProjectTables) -> _ProjectTables:
        """Cache *tables*, evicting the least recently used partition past the bound.

        Eviction only drops this dictionary's reference. A caller mid-query holds its
        own reference to the tables, so the underlying dataset stays open until that
        caller is done — the daemon serves each client on its own thread and must not
        have a table closed underneath it.
        """
        with self._partitions_lock:
            existing = self._partitions.get(project_id)
            if existing is not None:
                # Another thread opened it first; keep one instance so both callers
                # share a single set of handles.
                self._partitions.move_to_end(project_id)
                return existing
            self._partitions[project_id] = tables
            while len(self._partitions) > MAX_CACHED_PARTITIONS:
                self._partitions.popitem(last=False)
            return tables

    def _tables(self, project_id: str) -> _ProjectTables:
        """Open *project_id*'s partition, creating it. For write paths only."""
        cached = self._cached(project_id)
        if (
            cached is not None
            and cached.references is not None
            and cached.generation == self._partition_generation(project_id)
        ):
            return cached
        database = lancedb.connect(
            self.directory / "projects" / project_id,
            read_consistency_interval=timedelta(0),
        )
        tables = _ProjectTables(
            files=self._table(database, "files", self._file_schema()),
            chunks=self._table(
                database,
                "chunks",
                self._chunk_schema(self.vector_dimension),
            ),
            references=self._table(database, "references", self._reference_schema()),
            generation=self._partition_generation(project_id),
        )
        if cached is not None:
            return self._replace_cached(project_id, tables)
        return self._remember(project_id, tables)

    def _replace_cached(self, project_id: str, tables: _ProjectTables) -> _ProjectTables:
        """Replace a cached legacy partition after adding its references table."""
        with self._partitions_lock:
            self._partitions[project_id] = tables
            self._partitions.move_to_end(project_id)
        return tables

    def _existing_tables(self, project_id: str) -> _ProjectTables | None:
        """Open *project_id*'s partition without creating it, or return None.

        Reads must not materialise storage for a project they are only looking
        at. get_chunk in particular scans every registered project, so going
        through the create-on-write _tables() would leave an empty partition
        directory behind for each project that has never been indexed.
        """
        cached = self._cached(project_id)
        directory = self.directory / "projects" / project_id
        generation = self._partition_generation(project_id)
        if cached is not None:
            if directory.is_dir() and cached.generation == generation:
                return cached
            with self._partitions_lock:
                self._partitions.pop(project_id, None)
        if not directory.is_dir():
            return None
        database = lancedb.connect(directory, read_consistency_interval=timedelta(0))
        try:
            tables = _ProjectTables(
                files=cast(LanceTable, database.open_table("files")),
                chunks=cast(LanceTable, database.open_table("chunks")),
                references=self._open_optional_table(database, "references"),
                generation=generation,
            )
        except (ValueError, FileNotFoundError):
            return None
        return self._remember(project_id, tables)

    def _partition_generation(self, project_id: str) -> int:
        path = self.directory / "partition-generations" / project_id
        try:
            return int(path.read_text())
        except (OSError, ValueError):
            return 0

    def _advance_partition_generation(self, project_id: str) -> None:
        path = self.directory / "partition-generations" / project_id
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(str(self._partition_generation(project_id) + 1))
        os.replace(temporary, path)

    @staticmethod
    def _table(database: Any, name: str, schema: pa.Schema) -> LanceTable:
        return cast(
            LanceTable,
            database.create_table(name, schema=schema, exist_ok=True),
        )

    @staticmethod
    def file_arrow_schema() -> pa.Schema:
        return LanceStore._file_schema()

    @staticmethod
    def chunk_arrow_schema(vector_dimension: int) -> pa.Schema:
        return LanceStore._chunk_schema(vector_dimension)

    @staticmethod
    def reference_arrow_schema() -> pa.Schema:
        return LanceStore._reference_schema()

    @staticmethod
    def _merge(table: LanceTable, key: str, rows: list[dict[str, Any]] | pa.Table) -> None:
        (
            table.merge_insert(key)
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    @staticmethod
    def _validate_schema_version(schema_version: int | None) -> None:
        if schema_version is not None and (
            isinstance(schema_version, bool) or not isinstance(schema_version, int)
        ):
            raise ValueError("schema_version must be a non-boolean integer")

    def registry_stats(self) -> TableStorageStats:
        """Snapshot the project registry table's storage statistics."""
        return self._table_storage_stats(
            self._projects, "projects", physical_directory=self.directory / "registry"
        )

    def storage_stats(self, project_id: str) -> ProjectStorageStats:
        """Collect read-only storage statistics for one project partition.

        A registered project with no partition yields zeroed tables rather than
        materializing storage (reads never create partitions). ``consistent``
        is False when any table's version changed between the initial snapshot
        and the end of collection, meaning the observations do not form one
        atomic snapshot.
        """
        project = next((p for p in self.list_projects() if p.id == project_id), None)
        if project is None:
            raise CodeIndexingError(ErrorCode.PROJECT_NOT_FOUND, f"Unknown project: {project_id}")
        return self.storage_stats_for(project)

    def storage_stats_for(self, project: ProjectInfo) -> ProjectStorageStats:
        """Collect read-only storage statistics for an already-resolved project.

        Unlike ``storage_stats``, this does not re-scan the registry, so an
        installation-wide report can resolve every project once and then
        collect each partition without N+1 registry reads.
        """
        tables = self._existing_tables(project.id)
        partition = self.directory / "projects" / project.id
        before = self._partition_versions(tables)
        collected: list[TableStorageStats] = []
        if tables is not None:
            collected.append(
                self._table_storage_stats(
                    tables.files, "files", physical_directory=partition / "files.lance"
                )
            )
            collected.append(
                self._table_storage_stats(
                    tables.chunks, "chunks", physical_directory=partition / "chunks.lance"
                )
            )
            if tables.references is not None:
                collected.append(
                    self._table_storage_stats(
                        tables.references,
                        "references",
                        physical_directory=partition / "references.lance",
                    )
                )
        # Walked before the closing version snapshot so it falls inside the
        # consistency window: a commit landing during the walk must make the
        # report inconsistent, not yield byte counts that silently disagree
        # with the table statistics collected above.
        partition_physical_bytes = _directory_bytes(partition)
        after = self._partition_versions(tables)
        # The partition directory exists but its tables could not be opened
        # (a damaged or mid-mutation store is exactly what status is for); a
        # project that was never indexed has no directory at all. The two must
        # not be conflated: report the failure explicitly and treat the
        # snapshot as unusable.
        open_failed = tables is None and partition.is_dir()
        return ProjectStorageStats(
            project=project,
            snapshot_at=datetime.now(UTC).isoformat(),
            tables=collected,
            partition_physical_bytes=partition_physical_bytes,
            consistent=before == after and not open_failed,
            partition_open_failed=open_failed,
        )

    @staticmethod
    def _partition_versions(tables: _ProjectTables | None) -> tuple[int, int, int]:
        """Snapshot the three partition table versions, 0 for a missing references table."""
        if tables is None:
            return (0, 0, 0)
        references = 0 if tables.references is None else int(tables.references.version)
        return (int(tables.files.version), int(tables.chunks.version), references)

    @staticmethod
    def _table_storage_stats(
        table: LanceTable,
        name: str,
        *,
        physical_directory: Path,
    ) -> TableStorageStats:
        """Collect one Lance table's storage snapshot.

        Introspection is best-effort: a table whose statistics cannot be read
        (a damaged or mid-mutation store is exactly what status is for) reports
        its version and nothing else rather than failing the whole report.
        """
        try:
            stats = cast(dict[str, Any], cast(object, table.stats()))
        except (ValueError, RuntimeError):
            stats = {}
        try:
            num_rows = max(0, int(stats.get("num_rows", 0)))
            logical_bytes = max(0, int(stats.get("total_bytes", 0)))
        except (TypeError, ValueError):
            num_rows = 0
            logical_bytes = 0
        fragments: dict[str, Any] = stats.get("fragment_stats") or {}
        lengths: dict[str, Any] | None = fragments.get("lengths")
        try:
            fragment_stats = FragmentStats(
                num_fragments=max(0, int(fragments.get("num_fragments", 0))),
                num_small_fragments=max(0, int(fragments.get("num_small_fragments", 0))),
                lengths=(
                    FragmentLengthStats(
                        **{key: _nullable_int(value) for key, value in lengths.items()}
                    )
                    if lengths
                    else None
                ),
            )
        except (TypeError, ValueError):
            fragment_stats = FragmentStats()
        try:
            versions = cast(list[dict[str, Any]], table.list_versions())
        except (ValueError, RuntimeError):
            versions = []
        timestamps = [
            str(version.get("timestamp", "")) for version in versions if version.get("timestamp")
        ]
        indexes: list[IndexStorageStats] = []
        try:
            for index in table.list_indices():
                indexes.append(
                    IndexStorageStats(
                        name=index.name,
                        index_type=index.index_type,
                        columns=list(index.columns),
                        indexed_rows=max(0, int(getattr(index, "num_indexed_rows", 0) or 0)),
                        unindexed_rows=max(0, int(getattr(index, "num_unindexed_rows", 0) or 0)),
                        size_bytes=max(0, int(getattr(index, "size_bytes", 0) or 0)),
                    )
                )
        except (ValueError, RuntimeError):
            indexes = []
        return TableStorageStats(
            name=name,
            current_version=int(table.version),
            row_count=num_rows,
            logical_bytes=logical_bytes,
            physical_bytes=_directory_bytes(physical_directory),
            fragment_stats=fragment_stats,
            retained_version_count=len(versions),
            oldest_version_at=_normalized_timestamp(timestamps[0]) if timestamps else None,
            newest_version_at=_normalized_timestamp(timestamps[-1]) if timestamps else None,
            indexes=indexes,
        )

    @staticmethod
    def _rows(table: LanceTable, condition: str | None = None) -> list[dict[str, Any]]:
        query = table.search()
        if condition:
            query = query.where(condition)
        return cast(list[dict[str, Any]], query.to_list())

    def _reference_rows(
        self, project_id: str, condition: str | None, *, version: int | None = None
    ) -> list[ReferenceRecord]:
        tables = self._existing_tables(project_id)
        if tables is None or tables.references is None:
            return []
        references = tables.references
        if version is not None and version != int(references.version):
            database = lancedb.connect(
                self.directory / "projects" / project_id,
                read_consistency_interval=timedelta(0),
            )
            references = cast(LanceTable, database.open_table("references", version=version))
        query = references.search()
        if condition:
            query = query.where(condition)
        query = query.order_by(
            [
                ColumnOrdering(column_name="path"),
                ColumnOrdering(column_name="start_line"),
                ColumnOrdering(column_name="reference_id"),
            ]
        )
        return cast(list[ReferenceRecord], query.to_list())

    @staticmethod
    def _open_optional_table(database: Any, name: str) -> LanceTable | None:
        try:
            return cast(LanceTable, database.open_table(name))
        except (ValueError, FileNotFoundError):
            return None

    @staticmethod
    def _projected_chunks(
        table: LanceTable,
        condition: str,
        *,
        limit: int | None,
        content: bool,
        order_by: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        # project_id is deliberately absent: it is not stored on chunk rows
        # and belongs to the owning partition, so callers inject it.
        columns = [
            "chunk_id",
            "path",
            "language",
            "kind",
            "symbol",
            "qualified_symbol",
            "parent_symbol",
            "start_line",
            "end_line",
        ]
        if content:
            columns.append("content")
        query = table.search().where(condition).select(columns)
        if order_by is not None:
            # A stable scan order makes a truncated result set deterministic
            # rather than dependent on physical row layout.
            query = query.order_by([ColumnOrdering(column_name=name) for name in order_by])
        if limit is not None:
            query = query.limit(limit)
        return cast(list[dict[str, Any]], query.to_list())

    @classmethod
    def _migrate_v1(cls, directory: Path) -> list[dict[str, Any]]:
        legacy_projects = directory / "projects.lance"
        if not legacy_projects.exists():
            return []
        directory.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(directory.parent / f".{directory.name}-migrate.lock"):
            if not legacy_projects.exists():
                return []
            database = lancedb.connect(directory, read_consistency_interval=timedelta(0))
            table = database.open_table("projects")
            rows = cast(list[dict[str, Any]], table.search().to_list())
            del table
            del database
            gc.collect()
            backup = directory.with_name(f"{directory.name}-v1-backup-{time.time_ns()}")
            shutil.move(str(directory), str(backup))
            return rows
