"""Hybrid retrieval and structural lookup services."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

from .embedding import Embedder
from .errors import CodeIndexingError, ErrorCode
from .models import (
    ChunkPreview,
    CodeChunk,
    OutlineItem,
    OutlineResponse,
    SearchHit,
    SearchResponse,
    SymbolResponse,
)
from .path_filter import path_condition
from .storage import LanceStore, PartitionRef, _quoted

logger = logging.getLogger(__name__)

# Rows fetched when path patterns cannot be pushed into the scan. Ten times the
# ordinary window: enough that a moderately low-ranking match survives the Python
# filter, without materialising a whole project's chunks.
_FALLBACK_FETCH_ROWS = 500


def _as_partition_refs(value: PartitionRef | Sequence[PartitionRef]) -> list[PartitionRef]:
    """Normalize one pinned partition or a sequence of them into a list."""
    if isinstance(value, PartitionRef):
        return [value]
    return list(value)


class SearchService:
    def __init__(self, store: LanceStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def search_code(
        self,
        query: str,
        project_ids: list[str],
        *,
        languages: list[str] | None = None,
        paths: list[str] | None = None,
        kinds: list[str] | None = None,
        limit: int = 8,
        partitions: Mapping[str, PartitionRef | Sequence[PartitionRef]] | None = None,
    ) -> SearchResponse:
        """Search every given project's pinned partitions and rank globally.

        A project may pin several partitions at once -- one per requested
        checkout of a shared registration -- and their rows compete in one
        ranked result list.
        """
        query = query.strip()
        if not query or not project_ids:
            raise CodeIndexingError(
                ErrorCode.INVALID_FILTER, "Search requires a query and at least one project"
            )
        limit = max(1, min(limit, 50))
        conditions: list[str] = []
        if languages:
            conditions.append(self._in_condition("language", languages))
        if kinds:
            conditions.append(self._in_condition("kind", kinds))
        pushed_paths = path_condition(paths) if paths else None
        if pushed_paths is not None:
            conditions.append(pushed_paths)
        elif paths:
            # Without a pushdown the Python filter below runs on rows the scan
            # already truncated, so a low-ranking match can be missed. Widen the
            # window to make that less likely and say so, rather than reporting a
            # confident empty result.
            logger.debug(
                "Path patterns %r could not be pushed down; filtering %d fetched rows in "
                "Python, so low-ranking matches may be missed",
                paths,
                _FALLBACK_FETCH_ROWS,
            )
        fetch = _FALLBACK_FETCH_ROWS if paths and pushed_paths is None else max(50, limit * 5)
        if partitions is None:
            selected = {
                project_id: [self.store.active_partition(project_id)] for project_id in project_ids
            }
        else:
            selected = {
                project_id: _as_partition_refs(refs) for project_id, refs in partitions.items()
            }
        if (
            set(selected) != set(project_ids)
            or any(
                ref.project_id != project_id
                for project_id, refs in selected.items()
                for ref in refs
            )
            or any(not refs for refs in selected.values())
        ):
            raise ValueError("search partitions do not match the requested projects")
        partition_ids = {
            project_id: [ref.partition_id for ref in refs] for project_id, refs in selected.items()
        }
        with self.store.partitions_access(selected):
            rows = self.store.hybrid_search(
                query,
                self.embedder.embed_query(query),
                project_ids,
                " AND ".join(conditions) if conditions else None,
                fetch,
                partition_ids=partition_ids,
            )
        names = {project.id: project.name for project in self.store.list_projects()}
        hits: list[SearchHit] = []
        seen: set[tuple[str, str, int, int]] = set()
        for row in rows:
            chunk = ChunkPreview.model_validate(row)
            if paths and not any(PurePosixPath(chunk.path).match(pattern) for pattern in paths):
                continue
            key = (chunk.project_id, chunk.path, chunk.start_line, chunk.end_line)
            if key in seen:
                continue
            seen.add(key)
            hits.append(self._hit(chunk, names, float(row.get("_relevance_score", 0.0))))
            if len(hits) == limit:
                break
        hits.sort(key=lambda hit: (-hit.score, hit.path, hit.start_line))
        return SearchResponse(query=query, hits=hits)

    def find_symbol(
        self,
        name: str,
        project_id: str,
        *,
        match: str = "exact",
        kinds: list[str] | None = None,
        limit: int = 20,
        partition: PartitionRef | None = None,
    ) -> SymbolResponse:
        if match not in {"exact", "prefix", "contains"}:
            raise CodeIndexingError(ErrorCode.INVALID_FILTER, f"Invalid symbol match mode: {match}")
        limit = max(1, min(limit, 50))
        partition = partition or self.store.active_partition(project_id)
        if partition.project_id != project_id:
            raise ValueError("symbol partition does not belong to project")
        with self.store.partition_access(project_id, partition_id=partition.partition_id):
            candidates = self.store.find_symbol_chunks(
                name,
                project_id,
                match=match,
                kinds=kinds,
                limit=limit,
                partition_id=partition.partition_id,
            )
        names = {project.id: project.name for project in self.store.list_projects()}

        selected = sorted(
            candidates,
            key=lambda chunk: (chunk.path, chunk.start_line, chunk.kind),
        )[:limit]
        return SymbolResponse(name=name, hits=[self._hit(chunk, names, 1.0) for chunk in selected])

    def file_outline(
        self, path: str, project_id: str, *, partition: PartitionRef | None = None
    ) -> OutlineResponse:
        items: list[OutlineItem] = []
        seen: set[tuple[str, str]] = set()
        partition = partition or self.store.active_partition(project_id)
        if partition.project_id != project_id:
            raise ValueError("outline partition does not belong to project")
        with self.store.partition_access(project_id, partition_id=partition.partition_id):
            chunks = self.store.outline_chunks(
                path, project_id, partition_id=partition.partition_id
            )
        for chunk in sorted(chunks, key=lambda item: (item.path, item.start_line)):
            if chunk.path != path or not chunk.symbol or not chunk.qualified_symbol:
                continue
            key = (chunk.kind.removesuffix("_part"), chunk.qualified_symbol)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                OutlineItem(
                    kind=key[0],
                    symbol=chunk.symbol,
                    qualified_symbol=chunk.qualified_symbol,
                    parent_symbol=chunk.parent_symbol,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                )
            )
        return OutlineResponse(project_id=project_id, path=path, items=items)

    def get_chunk(self, chunk_id: str, *, partition: PartitionRef | None = None) -> CodeChunk:
        project_id = self.store._chunk_project_id(chunk_id)
        if project_id is None:
            chunk = None
        else:
            partition = partition or self.store.active_partition(project_id)
            if partition.project_id != project_id:
                raise ValueError("chunk partition does not belong to project")
            with self.store.partition_access(project_id, partition_id=partition.partition_id):
                chunk = self.store.get_chunk(chunk_id, partition_id=partition.partition_id)
        if chunk is None:
            raise CodeIndexingError(
                ErrorCode.CHUNK_NOT_FOUND,
                f"Unknown chunk: {chunk_id}; chunk ids come from search_code or find_symbol "
                "results and change when the file is re-indexed",
                chunk_id=chunk_id,
            )
        return chunk

    @staticmethod
    def _in_condition(column: str, values: list[str]) -> str:
        return f"{column} IN ({', '.join(_quoted(value) for value in values)})"

    @staticmethod
    def _hit(chunk: ChunkPreview, names: dict[str, str], score: float) -> SearchHit:
        snippet = chunk.content[:4_000]
        return SearchHit(
            chunk_id=chunk.chunk_id,
            project_id=chunk.project_id,
            project_name=names.get(chunk.project_id, chunk.project_id),
            path=chunk.path,
            language=chunk.language,
            kind=chunk.kind,
            symbol=chunk.symbol,
            qualified_symbol=chunk.qualified_symbol,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            score=score,
            snippet=snippet,
            truncated=len(chunk.content) > len(snippet),
        )
