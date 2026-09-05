"""Hybrid retrieval and structural lookup services."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from .embedding import Embedder, compose_passage
from .errors import CodeIndexingError, ErrorCode
from .extractor import TreeSitterExtractor
from .models import (
    ChunkPreview,
    CodeChunk,
    ExampleSearchResponse,
    OutlineItem,
    OutlineResponse,
    RankExplanation,
    SearchHit,
    SearchResponse,
    SymbolResponse,
)
from .path_filter import path_condition
from .storage import LanceStore, PartitionRef, RankComponents, _quoted

logger = logging.getLogger(__name__)

# Rows fetched when path patterns cannot be pushed into the scan. Ten times the
# ordinary window: enough that a moderately low-ranking match survives the Python
# filter, without materialising a whole project's chunks.
_FALLBACK_FETCH_ROWS = 500

MAX_EXAMPLE_LENGTH = 16_384

_EXAMPLE_SUFFIX: dict[str, str] = {
    "python": ".py",
    "java": ".java",
    "javascript": ".js",
    "typescript": ".ts",
    "tsx": ".tsx",
    "csharp": ".cs",
    "sql": ".sql",
    "gdscript": ".gd",
    "gdshader": ".gdshader",
    "godot_resource": ".tres",
    "yaml": ".yaml",
    "json": ".json",
    "go": ".go",
    "terraform": ".tf",
    "rust": ".rs",
    "c": ".c",
    "cpp": ".cpp",
    "lua": ".lua",
}

_DETECTION_ORDER: tuple[str, ...] = (
    "python",
    "typescript",
    "javascript",
    "go",
    "rust",
    "java",
    "csharp",
    "cpp",
    "c",
    "lua",
    "sql",
    "gdscript",
    "tsx",
    "yaml",
    "json",
    "terraform",
    "gdshader",
    "godot_resource",
)


def detect_example_language(extractor: TreeSitterExtractor, source: str) -> str | None:
    source_bytes = source.encode("utf-8")
    candidates: list[str] = []
    for language in _DETECTION_ORDER:
        suffix = _EXAMPLE_SUFFIX.get(language, "")
        try:
            result = extractor.extract(Path(f"example{suffix}"), language, source_bytes)
            if not result.has_errors and any(chunk.kind != "module" for chunk in result.chunks):
                candidates.append(language)
        except Exception:
            continue
    return candidates[0] if len(candidates) == 1 else None


def _example_passages(
    extractor: TreeSitterExtractor,
    example: str,
    language: str | None = None,
) -> tuple[str | None, list[str]]:
    if not example or not example.strip():
        raise CodeIndexingError(
            ErrorCode.INVALID_FILTER, "Example search requires a non-empty code snippet"
        )
    example_bytes = example.encode("utf-8")
    if len(example_bytes) > MAX_EXAMPLE_LENGTH:
        raise CodeIndexingError(
            ErrorCode.INVALID_FILTER,
            f"Example snippet exceeds maximum length of {MAX_EXAMPLE_LENGTH} bytes",
        )
    if language is not None:
        if language not in _EXAMPLE_SUFFIX:
            raise CodeIndexingError(
                ErrorCode.UNSUPPORTED_LANGUAGE,
                f"Unsupported language '{language}'",
            )
        suffix = _EXAMPLE_SUFFIX[language]
        result = extractor.extract(Path(f"example{suffix}"), language, example_bytes)
        passages = [
            compose_passage(chunk.embedding_prefix, chunk.content) for chunk in result.chunks
        ]
        if not passages:
            passages = [compose_passage("", example)]
        return language, passages

    detected = detect_example_language(extractor, example)
    if detected is not None:
        suffix = _EXAMPLE_SUFFIX[detected]
        result = extractor.extract(Path(f"example{suffix}"), detected, example_bytes)
        passages = [
            compose_passage(chunk.embedding_prefix, chunk.content) for chunk in result.chunks
        ]
        if not passages:
            passages = [compose_passage("", example)]
        return detected, passages
    return None, [compose_passage("", example)]


def _as_partition_refs(value: PartitionRef | Sequence[PartitionRef]) -> list[PartitionRef]:
    """Normalize one pinned partition or a sequence of them into a list."""
    if isinstance(value, PartitionRef):
        return [value]
    return list(value)


class SearchService:
    def __init__(
        self,
        store: LanceStore,
        embedder: Embedder,
        extractor: TreeSitterExtractor | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.extractor = extractor or TreeSitterExtractor()

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
        query_vector = self.embedder.embed_query(query)
        condition = " AND ".join(conditions) if conditions else None
        with self.store.partitions_access(selected):
            rows = self.store.hybrid_search(
                query,
                query_vector,
                project_ids,
                condition,
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
        if hits:
            hits = self._explain_hits(query, query_vector, hits, condition, selected)
        return SearchResponse(query=query, hits=hits)

    def search_by_example(
        self,
        example: str,
        project_ids: list[str],
        *,
        language: str | None = None,
        languages: list[str] | None = None,
        paths: list[str] | None = None,
        kinds: list[str] | None = None,
        limit: int = 8,
        partitions: Mapping[str, PartitionRef | Sequence[PartitionRef]] | None = None,
    ) -> ExampleSearchResponse:
        """Search every given project's pinned partitions by code snippet similarity."""
        if not project_ids:
            raise CodeIndexingError(
                ErrorCode.INVALID_FILTER, "Example search requires at least one project"
            )
        resolved_language, passages = _example_passages(self.extractor, example, language=language)
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
        vectors = self.embedder.embed_passages(passages)
        with self.store.partitions_access(selected):
            rows = self.store.example_search(
                vectors,
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
        return ExampleSearchResponse(
            language=resolved_language,
            segments=len(passages),
            hits=hits,
        )

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
        project_id = self.store.chunk_project_id(chunk_id)
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

    def _explain_hits(
        self,
        query: str,
        query_vector: list[float],
        hits: list[SearchHit],
        condition: str | None,
        selected: dict[str, list[PartitionRef]],
    ) -> list[SearchHit]:
        """Attach per-signal ranking explanations; never fails the search."""
        wanted: dict[str, list[str]] = {}
        for hit in hits:
            wanted.setdefault(hit.project_id, []).append(hit.chunk_id)
        partition_ids = {
            project_id: [ref.partition_id for ref in refs] for project_id, refs in selected.items()
        }
        components: dict[str, RankComponents] = {}
        try:
            with self.store.partitions_access(selected):
                components = self.store.explain_hits(
                    query, query_vector, wanted, condition, partition_ids=partition_ids
                )
        except Exception:
            logger.debug("Ranking explanation probes failed", exc_info=True)
        if not components:
            return hits
        explained: list[SearchHit] = []
        for hit in hits:
            component = components.get(hit.chunk_id)
            if component is None:
                explained.append(hit)
            else:
                explained.append(
                    hit.model_copy(update={"explanation": RankExplanation(**component)})
                )
        return explained

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
