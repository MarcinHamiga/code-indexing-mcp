"""LanceDB persistence for projects, files, and chunks."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import lancedb
import pyarrow as pa
from lancedb.index import FTS, BTree, HnswSq
from lancedb.table import LanceTable

from .errors import ErrorCode, IncodeError
from .models import ProjectInfo, StoredChunk, StoredFile
from .projects import existing_marker_path

SCHEMA_VERSION = 1


def _quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class LanceStore:
    def __init__(self, directory: Path, *, vector_dimension: int = 768) -> None:
        self.directory = directory
        self.vector_dimension = vector_dimension
        directory.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(directory)
        self._projects = self._table("projects", self._project_schema())
        self._files = self._table("files", self._file_schema())
        self._chunks = self._table("chunks", self._chunk_schema(vector_dimension))

    def upsert_project(self, project: ProjectInfo, *, model_id: str, state: str = "ready") -> None:
        existing = self._rows(self._projects, f"id = {_quoted(project.id)}")
        if existing and (
            existing[0]["model_id"] != model_id
            or int(existing[0]["vector_dimension"]) != self.vector_dimension
            or int(existing[0]["schema_version"]) != SCHEMA_VERSION
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
        rows = self._rows(self._projects)
        return [ProjectInfo.model_validate_json(row["payload"]) for row in rows]

    def project_state(self, project_id: str) -> str:
        rows = self._rows(self._projects, f"id = {_quoted(project_id)}")
        if not rows:
            raise IncodeError(ErrorCode.PROJECT_NOT_FOUND, f"Unknown project: {project_id}")
        return str(rows[0]["state"])

    def list_files(self, project_id: str) -> list[StoredFile]:
        return [
            StoredFile.model_validate(row)
            for row in self._rows(self._files, f"project_id = {_quoted(project_id)}")
        ]

    def upsert_file(self, record: StoredFile) -> None:
        self._merge(self._files, "file_id", [record.model_dump()])

    def replace_file(self, record: StoredFile, chunks: list[StoredChunk]) -> None:
        condition = (
            f"project_id = {_quoted(record.project_id)} AND file_id = {_quoted(record.file_id)}"
        )
        if chunks:
            (
                self._chunks.merge_insert("chunk_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete(condition)
                .execute([chunk.model_dump() for chunk in chunks])
            )
        else:
            self._chunks.delete(condition)
        self.upsert_file(record)

    def remove_file(self, project_id: str, file_id: str) -> None:
        condition = f"project_id = {_quoted(project_id)} AND file_id = {_quoted(file_id)}"
        self._chunks.delete(condition)
        self._files.delete(condition)

    def list_chunks(self, project_ids: Iterable[str] | None = None) -> list[StoredChunk]:
        project_ids = list(project_ids or [])
        condition = None
        if project_ids:
            values = ", ".join(_quoted(project_id) for project_id in project_ids)
            condition = f"project_id IN ({values})"
        return [StoredChunk.model_validate(row) for row in self._rows(self._chunks, condition)]

    def get_chunk(self, chunk_id: str) -> StoredChunk | None:
        rows = self._rows(self._chunks, f"chunk_id = {_quoted(chunk_id)}")
        return StoredChunk.model_validate(rows[0]) if rows else None

    def count_chunks(self, project_ids: Iterable[str] | None = None) -> int:
        project_ids = list(project_ids or [])
        condition = None
        if project_ids:
            values = ", ".join(_quoted(project_id) for project_id in project_ids)
            condition = f"project_id IN ({values})"
        return int(self._chunks.count_rows(condition))

    def hybrid_search(
        self,
        query_text: str,
        vector: list[float],
        condition: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        query = (
            self._chunks.search(query_type="hybrid", vector_column_name="vector")
            .vector(vector)
            .text(query_text)
            .where(condition, prefilter=True)
            .limit(limit)
            .rerank()
        )
        return cast(list[dict[str, Any]], query.to_list())

    def ensure_indexes(self, *, compact: bool = False) -> None:
        # NOTE: lance 8.x has no incremental FTS/BTree indexing — create_index
        # with replace=False fails once the index exists, so a rebuild
        # (replace=True) is required whenever rows changed. Callers should
        # therefore only invoke this when data actually changed.
        self._chunks.create_index(
            "search_text",
            config=FTS(lower_case=True, stem=False, remove_stop_words=False),
            replace=True,
        )
        for column in ("project_id", "file_id", "language", "path", "symbol"):
            self._chunks.create_index(column, config=BTree(), replace=True)
        if self._chunks.count_rows() >= 20_000:
            self._chunks.create_index("vector", config=HnswSq(distance_type="cosine"), replace=True)
        if compact:
            self._chunks.optimize()

    def remove_project(self, project_id: str) -> bool:
        existed = bool(self._rows(self._projects, f"id = {_quoted(project_id)}"))
        condition = f"project_id = {_quoted(project_id)}"
        self._chunks.delete(condition)
        self._files.delete(condition)
        self._projects.delete(f"id = {_quoted(project_id)}")
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
                ("vector", pa.list_(pa.float32(), vector_dimension)),
            ]
        )

    def _table(self, name: str, schema: pa.Schema) -> LanceTable:
        return self._db.create_table(name, schema=schema, exist_ok=True)

    @staticmethod
    def _merge(table: LanceTable, key: str, rows: list[dict[str, Any]]) -> None:
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
