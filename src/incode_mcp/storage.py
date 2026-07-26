"""Partitioned LanceDB persistence for projects, files, and chunks."""

from __future__ import annotations

import gc
import logging
import shutil
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import lancedb
import pyarrow as pa
from filelock import FileLock
from lancedb.index import FTS, BTree, HnswSq
from lancedb.query import ColumnOrdering
from lancedb.table import LanceTable

from .errors import ErrorCode, IncodeError
from .models import ChunkPreview, ProjectInfo, StoredChunk, StoredFile
from .projects import existing_marker_path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

# Symbol lookups over-fetch because the LIKE pushdown over-matches; these bound
# how many rows are scanned before the exact filter and the caller's limit apply.
OVERFETCH_FACTOR = 10
MINIMUM_OVERFETCH = 200


def _quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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


@dataclass(frozen=True)
class TableVersions:
    """A point-in-time snapshot of a project partition's two tables."""

    files: int
    chunks: int


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
        self._partitions: dict[str, _ProjectTables] = {}
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
        if (
            existing
            and str(existing[0]["state"]) != "pending"
            and (
                existing[0]["model_id"] != model_id
                or int(existing[0]["vector_dimension"]) != self.vector_dimension
                or int(existing[0]["schema_version"]) != SCHEMA_VERSION
            )
        ):
            raise IncodeError(
                ErrorCode.INDEX_INCOMPATIBLE,
                "Project index uses an incompatible schema or embedding model",
                project=project.id,
            )
        if existing:
            registered_root = Path(str(existing[0]["root"])).resolve()
            incoming_root = project.root.resolve()
            if (
                registered_root != incoming_root
                and existing_marker_path(registered_root) is not None
            ):
                raise IncodeError(
                    ErrorCode.PROJECT_ID_CONFLICT,
                    "The project ID is already active at another path",
                    project=project.id,
                    registered_root=str(registered_root),
                    incoming_root=str(incoming_root),
                )
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
        self._merge(self._projects, "id", [row])

    def list_projects(self) -> list[ProjectInfo]:
        return [
            ProjectInfo.model_validate_json(row["payload"]) for row in self._rows(self._projects)
        ]

    def project_state(self, project_id: str) -> str:
        rows = self._rows(self._projects, f"id = {_quoted(project_id)}")
        if not rows:
            raise IncodeError(ErrorCode.PROJECT_NOT_FOUND, f"Unknown project: {project_id}")
        return str(rows[0]["state"])

    def list_files(self, project_id: str) -> list[StoredFile]:
        tables = self._existing_tables(project_id)
        if tables is None:
            return []
        return [StoredFile.model_validate(row) for row in self._rows(tables.files)]

    def upsert_file(self, record: StoredFile) -> None:
        self._merge(
            self._tables(record.project_id).files,
            "file_id",
            [record.model_dump()],
        )

    def replace_file(self, record: StoredFile, chunks: list[StoredChunk]) -> None:
        tables = self._tables(record.project_id)
        condition = f"file_id = {_quoted(record.file_id)}"
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
        tables.files.delete(condition)

    def table_versions(self, project_id: str) -> TableVersions:
        """Snapshot both partition tables' versions before a commit begins."""
        tables = self._tables(project_id)
        return TableVersions(files=tables.files.version, chunks=tables.chunks.version)

    def restore_versions(self, project_id: str, versions: TableVersions) -> None:
        """Return both partition tables to *versions*' data.

        ``restore`` followed by ``checkout_latest`` makes the recorded version
        the live one; restoring a table that is already at that version's data
        is a no-op, so repeated recovery over the same journal is idempotent.
        """
        tables = self._tables(project_id)
        tables.files.restore(versions.files)
        tables.chunks.restore(versions.chunks)
        tables.files.checkout_latest()
        tables.chunks.checkout_latest()

    def replace_files_from_arrow(
        self,
        project_id: str,
        *,
        files: pa.Table,
        chunk_groups: Iterable[tuple[str, pa.Table]],
        removed_file_ids: Iterable[str] = (),
    ) -> None:
        """Commit staged Arrow batches without materializing chunk objects.

        *chunk_groups* yields one ``(file_id, table)`` pair per replaced file,
        so at most one file's chunks are live in Arrow form at a time; the
        vector columns stay fixed-size-list float32 arrays end to end. A group
        with zero rows means the file now extracts to no chunks, so its
        previous chunks are deleted.
        """
        tables = self._tables(project_id)
        for file_id, chunks in chunk_groups:
            condition = f"file_id = {_quoted(file_id)}"
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
        if files.num_rows:
            self._merge(tables.files, "file_id", files)
        for file_id in removed_file_ids:
            condition = f"file_id = {_quoted(file_id)}"
            tables.chunks.delete(condition)
            tables.files.delete(condition)

    def list_chunks(self, project_ids: Iterable[str] | None = None) -> list[StoredChunk]:
        ids = list(project_ids or [project.id for project in self.list_projects()])
        chunks: list[StoredChunk] = []
        for project_id in ids:
            tables = self._existing_tables(project_id)
            if tables is None:
                continue
            chunks.extend(StoredChunk.model_validate(row) for row in self._rows(tables.chunks))
        return chunks

    def get_chunk(self, chunk_id: str) -> StoredChunk | None:
        # chunk_id is a one-way digest of file_id, which is itself a digest of
        # the project id and path, so the owning project cannot be recovered
        # from the id. Scanning every project is inherent without an id-format
        # change and a full re-index; do not "fix" it by narrowing the loop.
        # The partitions open read-only so the scan leaves nothing behind.
        for project in self.list_projects():
            tables = self._existing_tables(project.id)
            if tables is None:
                continue
            rows = self._rows(tables.chunks, f"chunk_id = {_quoted(chunk_id)}")
            if rows:
                return StoredChunk.model_validate(rows[0])
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
        rows: list[dict[str, Any]] = []
        for project_id in project_ids:
            tables = self._existing_tables(project_id)
            if tables is None:
                continue
            query = (
                tables.chunks.search(query_type="hybrid", vector_column_name="vector")
                .vector(vector)
                .text(query_text)
            )
            if condition:
                query = query.where(condition, prefilter=True)
            query = (
                query.limit(limit)
                .select(
                    [
                        "chunk_id",
                        "project_id",
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
            rows.extend(cast(list[dict[str, Any]], query.to_list()))
        rows.sort(key=lambda row: float(row.get("_relevance_score", 0.0)), reverse=True)
        return rows[:limit]

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
        return [ChunkPreview.model_validate(row) for row in rows]

    def ensure_indexes(self, project_id: str, *, compact: bool = False) -> None:
        chunks = self._tables(project_id).chunks
        indices = list(chunks.list_indices())
        indexed_columns = {column for index in indices for column in index.columns}
        if "search_text" not in indexed_columns:
            chunks.create_index(
                "search_text",
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
        # Reclaim space after deletions, but never with delete_unverified or a
        # zero age: searches run concurrently from the daemon and from direct
        # CLI processes, so versions in active use must not be reaped.
        chunks.optimize(cleanup_older_than=timedelta(days=1) if compact else None)

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
                ("project_id", pa.string()),
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
                ("embedding_text", pa.string()),
                ("search_text", pa.string()),
                ("content_hash", pa.string()),
                ("part_index", pa.int32()),
                (
                    "vector",
                    pa.list_(pa.float32(), vector_dimension),
                ),
            ]
        )

    def _tables(self, project_id: str) -> _ProjectTables:
        """Open *project_id*'s partition, creating it. For write paths only."""
        with self._partitions_lock:
            cached = self._partitions.get(project_id)
            if cached is not None:
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
            )
            self._partitions[project_id] = tables
            return tables

    def _existing_tables(self, project_id: str) -> _ProjectTables | None:
        """Open *project_id*'s partition without creating it, or return None.

        Reads must not materialise storage for a project they are only looking
        at. get_chunk in particular scans every registered project, so going
        through the create-on-write _tables() would leave an empty partition
        directory behind for each project that has never been indexed.
        """
        with self._partitions_lock:
            cached = self._partitions.get(project_id)
            if cached is not None:
                return cached
            directory = self.directory / "projects" / project_id
            if not directory.is_dir():
                return None
            database = lancedb.connect(directory, read_consistency_interval=timedelta(0))
            try:
                tables = _ProjectTables(
                    files=cast(LanceTable, database.open_table("files")),
                    chunks=cast(LanceTable, database.open_table("chunks")),
                )
            except (ValueError, FileNotFoundError):
                return None
            self._partitions[project_id] = tables
            return tables

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
    def _merge(table: LanceTable, key: str, rows: list[dict[str, Any]] | pa.Table) -> None:
        (
            table.merge_insert(key)
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    @staticmethod
    def _rows(table: LanceTable, condition: str | None = None) -> list[dict[str, Any]]:
        query = table.search()
        if condition:
            query = query.where(condition)
        return cast(list[dict[str, Any]], query.to_list())

    @staticmethod
    def _projected_chunks(
        table: LanceTable,
        condition: str,
        *,
        limit: int | None,
        content: bool,
        order_by: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        columns = [
            "chunk_id",
            "project_id",
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
