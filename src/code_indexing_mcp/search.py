"""Hybrid retrieval and structural lookup services."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .embedding import (
    Embedder,
    PassageCandidate,
    SegmentingEmbedder,
    SegmentPlan,
    compose_passage,
)
from .errors import CodeIndexingError, ErrorCode
from .extractor import _CONTAINER_KINDS, TreeSitterExtractor
from .models import (
    SEARCH_SCOPES,
    ChunkPreview,
    CodeChunk,
    ExampleSearchResponse,
    ExtractionResult,
    OutlineItem,
    OutlineResponse,
    RankExplanation,
    SearchHit,
    SearchResponse,
    SearchScope,
    SymbolResponse,
)
from .path_filter import (
    PathPattern,
    PathPatternError,
    compile_path_patterns,
    path_condition,
    path_matches,
)
from .scanner import LANGUAGES
from .settings import IndexSettings
from .storage import LanceStore, PartitionRef, RankComponents, _quoted
from .token_batching import DEFAULT_MAX_TOKENS, MAX_WINDOWS_PER_CANDIDATE

if TYPE_CHECKING:
    from .graph_index import GraphIndexService

logger = logging.getLogger(__name__)

MAX_EXAMPLE_LENGTH = 16_384


def _example_suffixes() -> dict[str, str]:
    """One representative file suffix per language, from the scanner's table.

    The scanner's extension map is what indexing uses to decide a file's
    language, so deriving from it keeps `search_by_example(language=...)` accepting
    exactly the languages the tool schema (`LanguageName`) advertises.
    """
    suffixes: dict[str, str] = {}
    for suffix, language in LANGUAGES.items():
        suffixes.setdefault(language, suffix)
    return suffixes


_EXAMPLE_SUFFIX: dict[str, str] = _example_suffixes()

# Order matters only for the first match among equally plausible candidates:
# detection accepts a snippet only when exactly one language parses it cleanly.
# The historical order is kept, then every other indexed language follows.
_PRIMARY_DETECTION_ORDER: tuple[str, ...] = (
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
# Zig's grammar accepts a bare `name: value` line as a declaration, so prose such
# as "Note: fix this" would auto-detect as Zig and be chunked as code. It stays
# available through the explicit `language="zig"` argument.
_NOT_AUTO_DETECTED: frozenset[str] = frozenset({"zig"})
_DETECTION_ORDER: tuple[str, ...] = (
    *_PRIMARY_DETECTION_ORDER,
    *(
        language
        for language in _EXAMPLE_SUFFIX
        if language not in _PRIMARY_DETECTION_ORDER and language not in _NOT_AUTO_DETECTED
    ),
)


def _detect_example(
    extractor: TreeSitterExtractor, source: str
) -> tuple[str | None, ExtractionResult | None]:
    source_bytes = source.encode("utf-8")
    candidates: list[tuple[str, ExtractionResult]] = []
    for language in _DETECTION_ORDER:
        suffix = _EXAMPLE_SUFFIX.get(language, "")
        try:
            result = extractor.extract(Path(f"example{suffix}"), language, source_bytes)
            if not result.has_errors and any(chunk.kind != "module" for chunk in result.chunks):
                candidates.append((language, result))
        except (CodeIndexingError, ValueError) as exc:
            logger.debug("Example language detection could not parse %s: %s", language, exc)
            continue
    return candidates[0] if len(candidates) == 1 else (None, None)


def detect_example_language(extractor: TreeSitterExtractor, source: str) -> str | None:
    language, _ = _detect_example(extractor, source)
    return language


def _embed_example_passages(embedder: Embedder, passages: list[str]) -> list[list[float]]:
    """Embed example passages, windowing any that exceed the token budget.

    An ambiguous snippet comes back as one passage of up to 16 KB; embedding it
    whole would run a multi-thousand-token sequence in the long-lived process.
    A passage within the budget is one window and embeds exactly as before;
    a longer one contributes one vector per window, which is how example
    search already treats a multi-chunk snippet.
    """
    if not isinstance(embedder, SegmentingEmbedder):
        return embedder.embed_passages(passages)
    plan = SegmentPlan(max_tokens=getattr(embedder, "max_input_tokens", DEFAULT_MAX_TOKENS))
    segments = embedder.plan_and_embed([PassageCandidate("", text) for text in passages], plan)
    return [
        np.frombuffer(segment.vector, dtype="<f4").astype(float).tolist()
        for candidate_segments in segments
        for segment in candidate_segments
    ]


_EXAMPLE_REF_WEIGHTS: dict[str, float] = {
    "call": 1.0,
    "inheritance": 0.9,
    "decorator": 0.7,
    "type_use": 0.65,
}


def _collect_example_outbound_symbols(result: ExtractionResult) -> dict[str, float]:
    outbound: dict[str, float] = {}
    for ref in result.references:
        if not ref.target_name:
            continue
        weight = _EXAMPLE_REF_WEIGHTS.get(ref.kind)
        if weight is None:
            continue
        outbound[ref.target_name] = max(outbound.get(ref.target_name, 0.0), weight)
    return outbound


def _example_passages_with_refs(
    extractor: TreeSitterExtractor,
    example: str,
    language: str | None = None,
) -> tuple[str | None, list[str], dict[str, float]]:
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
        return language, passages, _collect_example_outbound_symbols(result)

    detected, detected_result = _detect_example(extractor, example)
    if detected is not None:
        assert detected_result is not None
        passages = [
            compose_passage(chunk.embedding_prefix, chunk.content)
            for chunk in detected_result.chunks
        ]
        if not passages:
            passages = [compose_passage("", example)]
        return detected, passages, _collect_example_outbound_symbols(detected_result)
    return None, [compose_passage("", example)], {}


def _example_passages(
    extractor: TreeSitterExtractor,
    example: str,
    language: str | None = None,
) -> tuple[str | None, list[str]]:
    resolved_lang, passages, _ = _example_passages_with_refs(extractor, example, language=language)
    return resolved_lang, passages


def _outline_items(
    chunks: Sequence[ChunkPreview], path: str, *, span_parts: bool = False
) -> list[OutlineItem]:
    """One outline entry per declaration, merging a split declaration's parts.

    The entry keeps the first part's line range unless *span_parts* asks for
    the whole declaration's, from its first part's start to its last's end.
    Without *span_parts* same-named declarations share the first one's entry,
    as the outline always has. With it they stay separate entries -- overloads
    or a property's getter and setter are not parts of one declaration -- so
    no merged span reaches over the declarations between them.
    """
    items: list[OutlineItem] = []
    # Each key's latest entry, whether it was built from split parts, and its
    # latest part_index (so parts separated by stripped blank lines still merge,
    # while a subsequent same-named overload restarting at part_index 0 stays
    # separate).
    latest: dict[tuple[str, str], tuple[int, bool, int]] = {}
    for chunk in sorted(chunks, key=lambda item: (item.path, item.start_line, item.part_index)):
        if chunk.path != path or not chunk.symbol or not chunk.qualified_symbol:
            continue
        key = (chunk.kind.removesuffix("_part"), chunk.qualified_symbol)
        is_part = chunk.kind.endswith("_part")
        if key in latest:
            if not span_parts:
                continue
            index, from_parts, last_part_index = latest[key]
            existing = items[index]
            # A split declaration's parts advance part_index monotonically (or,
            # for synthetic/windowed previews sharing a part_index, overlap or
            # abut); a reset to part_index 0 across a gap or a whole declaration
            # means another declaration with the same name. A container's gap
            # chunks (`class_part` text between and after its members) are the
            # exception: member chunks sit between them, so they never abut, yet
            # they belong to the container's latest entry.
            container_gap = is_part and key[0] in _CONTAINER_KINDS
            continuing_part = (
                is_part
                and from_parts
                and (
                    chunk.part_index > last_part_index
                    or (
                        chunk.part_index == last_part_index
                        and chunk.start_line <= existing.end_line + 1
                    )
                )
            )
            if container_gap or continuing_part:
                if chunk.end_line > existing.end_line:
                    items[index] = existing.model_copy(update={"end_line": chunk.end_line})
                latest[key] = (index, from_parts, chunk.part_index)
                continue
        latest[key] = (len(items), is_part, chunk.part_index)
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
    return items


def _as_partition_refs(value: PartitionRef | Sequence[PartitionRef]) -> list[PartitionRef]:
    """Normalize one pinned partition or a sequence of them into a list."""
    if isinstance(value, PartitionRef):
        return [value]
    return list(value)


@dataclass(frozen=True)
class _SearchPreparation:
    selected: dict[str, list[PartitionRef]]
    partition_ids: dict[str, list[str]]
    condition: str | None
    paths: list[PathPattern]
    limit: int
    fetch: int


class SearchService:
    def __init__(
        self,
        store: LanceStore,
        embedder: Embedder,
        extractor: TreeSitterExtractor | None = None,
        graph_service: GraphIndexService | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.extractor = extractor or TreeSitterExtractor()
        if graph_service is None:
            from .graph_index import GraphIndexService
            from .reference_service import ReferenceService

            settings = IndexSettings.from_environment()
            self.graph_service = GraphIndexService(
                store,
                ReferenceService(store),
                cache_limit=settings.graph_cache_limit,
                enabled=settings.graph_search,
            )
        else:
            self.graph_service = graph_service

    def _prepare_search(
        self,
        project_ids: list[str],
        *,
        languages: list[str] | None,
        paths: list[str] | None,
        kinds: list[str] | None,
        scope: SearchScope = "code",
        limit: int,
        partitions: Mapping[str, PartitionRef | Sequence[PartitionRef]] | None,
    ) -> _SearchPreparation:
        if scope not in SEARCH_SCOPES:
            raise CodeIndexingError(
                ErrorCode.INVALID_FILTER,
                f"Invalid search scope: {scope!r}; expected 'code', 'tests', or 'all'",
                scope=scope,
            )
        limit = max(1, min(limit, 50))
        conditions: list[str] = []
        if scope == "code":
            conditions.append("is_test = false")
        elif scope == "tests":
            conditions.append("is_test = true")
        if languages:
            conditions.append(self._in_condition("language", languages))
        if kinds:
            conditions.append(self._in_condition("kind", kinds))
        try:
            patterns = compile_path_patterns(paths or ())
        except PathPatternError as exc:
            raise CodeIndexingError(
                ErrorCode.INVALID_FILTER, f"Invalid paths filter: {exc}", paths=paths
            ) from exc
        # Every pattern compiles to a regex, so the filter is always pushed into
        # the scan; the same regex then re-checks each fetched row.
        pushed_paths = path_condition(patterns)
        if pushed_paths is not None:
            conditions.append(pushed_paths)
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
        return _SearchPreparation(
            selected=selected,
            partition_ids={
                project_id: [ref.partition_id for ref in refs]
                for project_id, refs in selected.items()
            },
            condition=" AND ".join(conditions) if conditions else None,
            paths=patterns,
            limit=limit,
            fetch=max(50, limit * 5),
        )

    def _collect_hits(
        self,
        rows: Sequence[Mapping[str, Any]],
        names: Mapping[str, str],
        preparation: _SearchPreparation,
        *,
        include_content_in_key: bool,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        seen: set[tuple[object, ...]] = set()
        for row in rows:
            chunk = ChunkPreview.model_validate(row)
            if not path_matches(chunk.path, preparation.paths):
                continue
            key: tuple[object, ...] = (
                chunk.project_id,
                chunk.path,
                chunk.start_line,
                chunk.end_line,
                chunk.content if include_content_in_key else None,
            )
            if chunk.source_start_byte is not None:
                # A window's logical result is its source chunk. Pinned
                # checkouts can differ elsewhere in the file, so the chunk's
                # own digest, not the file hash, decides whether equal source
                # collapses -- matching the content rule for unwindowed rows.
                key = (
                    chunk.project_id,
                    chunk.file_id,
                    chunk.kind,
                    chunk.qualified_symbol,
                    chunk.part_index,
                    chunk.source_start_byte,
                    chunk.source_end_byte,
                    chunk.source_content_hash if include_content_in_key else None,
                )
            if key in seen:
                continue
            seen.add(key)
            hits.append(self._hit(chunk, names, float(row.get("_relevance_score", 0.0))))
            if len(hits) == preparation.limit:
                break
        hits.sort(key=lambda hit: (-hit.score, hit.path, hit.start_line))
        return hits

    def _window_aware_hits(
        self,
        fetch_rows: Callable[[int], list[dict[str, Any]]],
        names: Mapping[str, str],
        preparation: _SearchPreparation,
        *,
        include_content_in_key: bool,
    ) -> list[SearchHit]:
        """Refill a result page when embedding windows occupied its candidate slots."""
        fetch = preparation.fetch
        # Equal source in pinned checkouts shares a logical result identity,
        # but every physical partition contributes its own window vectors.
        copies = max((len(refs) for refs in preparation.selected.values()), default=1)
        maximum = max(fetch, preparation.limit * MAX_WINDOWS_PER_CANDIDATE * copies)
        while True:
            rows = fetch_rows(fetch)
            hits = self._collect_hits(
                rows, names, preparation, include_content_in_key=include_content_in_key
            )
            if (
                len(hits) >= preparation.limit
                or len(rows) < fetch
                or fetch >= maximum
                or not any(row.get("source_start_byte") is not None for row in rows)
            ):
                return hits
            fetch = min(maximum, fetch * 2)

    @staticmethod
    def _window_source_key(row: Mapping[str, Any]) -> tuple[object, ...] | None:
        if row.get("source_start_byte") is None:
            return None
        return (
            row.get("project_id"),
            row.get("file_id"),
            row.get("kind"),
            row.get("qualified_symbol"),
            row.get("part_index"),
            row.get("source_start_byte"),
            row.get("source_end_byte"),
        )

    def _record_graph_signals(
        self,
        rows: Sequence[Mapping[str, Any]],
        out: dict[str, tuple[float, str | None]],
    ) -> None:
        by_source: dict[tuple[object, ...], tuple[float, str | None]] = {}
        for row in rows:
            if "_graph_score" not in row:
                continue
            cid = str(row["chunk_id"])
            score = float(row["_graph_score"])
            via = None if row.get("_graph_via") is None else str(row["_graph_via"])
            prev = out.get(cid)
            if prev is None or score > prev[0]:
                out[cid] = (score, via)
            skey = self._window_source_key(row)
            if skey is not None:
                sprev = by_source.get(skey)
                if sprev is None or score > sprev[0]:
                    by_source[skey] = (score, via)
        if by_source:
            for row in rows:
                skey = self._window_source_key(row)
                if skey is not None and skey in by_source:
                    cid = str(row["chunk_id"])
                    if cid not in out:
                        out[cid] = by_source[skey]

    def _record_partitions(
        self,
        rows: Sequence[Mapping[str, Any]],
        out: dict[str, str],
    ) -> None:
        """Map each row's chunk id to the physical partition it was read from."""
        by_source: dict[tuple[object, ...], str] = {}
        for row in rows:
            part_id = row.get("_partition_id")
            if part_id is None:
                continue
            out.setdefault(str(row["chunk_id"]), str(part_id))
            skey = self._window_source_key(row)
            if skey is not None:
                by_source.setdefault(skey, str(part_id))
        if by_source:
            for row in rows:
                skey = self._window_source_key(row)
                if skey is not None and skey in by_source:
                    out.setdefault(str(row["chunk_id"]), by_source[skey])

    def _attach_hit_graph_context(
        self,
        hits: list[SearchHit],
        selected: Mapping[str, Sequence[PartitionRef]],
        known_partitions: Mapping[str, str] | None = None,
    ) -> list[SearchHit]:
        if not hits or not self.graph_service.enabled:
            return hits
        known = known_partitions or {}
        updated: list[SearchHit] = []
        for hit in hits:
            refs = selected.get(hit.project_id) or []
            recorded = known.get(hit.chunk_id)
            if recorded is not None:
                candidates: list[str | None] = [recorded]
            elif refs:
                candidates = [ref.partition_id for ref in refs]
            else:
                candidates = [None]
            ctx = None
            for part_id in candidates:
                try:
                    ctx = self.graph_service.ego_context(
                        hit.project_id, hit.chunk_id, partition_id=part_id
                    )
                except Exception:
                    logger.debug("Hit graph_context failed for %s", hit.chunk_id, exc_info=True)
                    ctx = None
                if ctx is not None:
                    break
            if ctx is not None:
                updated.append(hit.model_copy(update={"graph_context": ctx}))
            else:
                updated.append(hit)
        return updated

    def search_code(
        self,
        query: str,
        project_ids: list[str],
        *,
        languages: list[str] | None = None,
        paths: list[str] | None = None,
        kinds: list[str] | None = None,
        scope: SearchScope = "code",
        limit: int = 8,
        partitions: Mapping[str, PartitionRef | Sequence[PartitionRef]] | None = None,
        include_graph_context: bool = False,
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
        preparation = self._prepare_search(
            project_ids,
            languages=languages,
            paths=paths,
            kinds=kinds,
            scope=scope,
            limit=limit,
            partitions=partitions,
        )
        query_vector = self.embedder.embed_query(query)
        names = {project.id: project.name for project in self.store.list_projects()}
        graph_expander = (
            (
                lambda pid, part_id, qtext, rows, cond, lim: self.graph_service.expand_hybrid_rows(
                    pid,
                    part_id,
                    qtext,
                    rows,
                    cond,
                    lim,
                    query_vector=query_vector,
                )
            )
            if self.graph_service.enabled
            else None
        )
        with self.store.partitions_access(preparation.selected):
            known_vector_scores: dict[str, float] = {}
            known_graph_signals: dict[str, tuple[float, str | None]] = {}
            known_partitions: dict[str, str] = {}

            def fetch_hybrid(fetch: int) -> list[dict[str, Any]]:
                known_vector_scores.clear()
                known_graph_signals.clear()
                known_partitions.clear()
                rows = self.store.hybrid_search(
                    query,
                    query_vector,
                    project_ids,
                    preparation.condition,
                    fetch,
                    partition_ids=preparation.partition_ids,
                    graph_expander=graph_expander,
                )
                for row in rows:
                    if "_distance" in row:
                        known_vector_scores[str(row["chunk_id"])] = 1.0 - float(row["_distance"])
                self._record_graph_signals(rows, known_graph_signals)
                self._record_partitions(rows, known_partitions)
                return rows

            hits = self._window_aware_hits(
                fetch_hybrid,
                names,
                preparation,
                include_content_in_key=True,
            )
            if hits:
                # Same locked view as the hits: no second lock round trip, and
                # no commit can land between a search and its explanation.
                hits = self._explain_hits(
                    query,
                    query_vector,
                    hits,
                    preparation.condition,
                    preparation.selected,
                    known_vector_scores=known_vector_scores,
                    known_graph_signals=known_graph_signals,
                )
                if include_graph_context:
                    hits = self._attach_hit_graph_context(
                        hits, preparation.selected, known_partitions
                    )
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
        scope: SearchScope = "code",
        limit: int = 8,
        partitions: Mapping[str, PartitionRef | Sequence[PartitionRef]] | None = None,
        include_graph_context: bool = False,
    ) -> ExampleSearchResponse:
        """Search every given project's pinned partitions by code snippet similarity."""
        if not project_ids:
            raise CodeIndexingError(
                ErrorCode.INVALID_FILTER, "Example search requires at least one project"
            )
        resolved_language, passages, outbound_symbols = _example_passages_with_refs(
            self.extractor, example, language=language
        )
        preparation = self._prepare_search(
            project_ids,
            languages=languages,
            paths=paths,
            kinds=kinds,
            scope=scope,
            limit=limit,
            partitions=partitions,
        )
        vectors = _embed_example_passages(self.embedder, passages)
        names = {project.id: project.name for project in self.store.list_projects()}
        with self.store.partitions_access(preparation.selected):
            known_vector_scores: dict[str, float] = {}
            known_graph_signals: dict[str, tuple[float, str | None]] = {}
            known_partitions: dict[str, str] = {}

            def fetch_example(fetch: int) -> list[dict[str, Any]]:
                known_vector_scores.clear()
                known_graph_signals.clear()
                known_partitions.clear()
                expander = (
                    (
                        lambda proj_id, part_id, part_rows, cond, lim: (
                            self.graph_service.expand_example_rows(
                                proj_id,
                                part_id,
                                outbound_symbols,
                                part_rows,
                                cond,
                                lim,
                                probe_vectors=vectors,
                            )
                        )
                    )
                    if self.graph_service.enabled
                    else None
                )
                rows = self.store.example_search(
                    vectors,
                    project_ids,
                    preparation.condition,
                    fetch,
                    partition_ids=preparation.partition_ids,
                    graph_expander=expander,
                )
                for row in rows:
                    if "_distance" in row:
                        known_vector_scores[str(row["chunk_id"])] = 1.0 - float(row["_distance"])
                self._record_graph_signals(rows, known_graph_signals)
                self._record_partitions(rows, known_partitions)
                return rows

            hits = self._window_aware_hits(
                fetch_example,
                names,
                preparation,
                include_content_in_key=False,
            )
            if hits and known_graph_signals:
                hits = self._explain_example_hits(hits, known_vector_scores, known_graph_signals)
            if hits and include_graph_context:
                hits = self._attach_hit_graph_context(hits, preparation.selected, known_partitions)
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
        scope: SearchScope = "code",
        limit: int = 20,
        partition: PartitionRef | None = None,
    ) -> SymbolResponse:
        if match not in {"exact", "prefix", "contains"}:
            raise CodeIndexingError(ErrorCode.INVALID_FILTER, f"Invalid symbol match mode: {match}")
        if scope not in SEARCH_SCOPES:
            raise CodeIndexingError(
                ErrorCode.INVALID_FILTER,
                f"Invalid search scope: {scope!r}; expected 'code', 'tests', or 'all'",
                scope=scope,
            )
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
                scope=scope,
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
        partition = partition or self.store.active_partition(project_id)
        if partition.project_id != project_id:
            raise ValueError("outline partition does not belong to project")
        with self.store.partition_access(project_id, partition_id=partition.partition_id):
            chunks = self.store.outline_chunks(
                path, project_id, partition_id=partition.partition_id
            )
        return OutlineResponse(project_id=project_id, path=path, items=_outline_items(chunks, path))

    def file_outlines(
        self, paths: Sequence[str], project_id: str, *, partition: PartitionRef
    ) -> dict[str, list[OutlineItem]]:
        """Outline several files in one batched read, keyed by path.

        Each entry spans its whole declaration, including every split part,
        so a caller can intersect it with changed line ranges.
        """
        if partition.project_id != project_id:
            raise ValueError("outline partition does not belong to project")
        with self.store.partition_access(project_id, partition_id=partition.partition_id):
            chunks = self.store.outline_chunks_for_paths(
                paths, project_id, partition_id=partition.partition_id
            )
        by_path: dict[str, list[ChunkPreview]] = {}
        for chunk in chunks:
            by_path.setdefault(chunk.path, []).append(chunk)
        return {
            path: _outline_items(by_path.get(path, []), path, span_parts=True) for path in paths
        }

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
                if chunk is not None and self.graph_service.enabled:
                    try:
                        ctx = self.graph_service.ego_context(
                            project_id, chunk.chunk_id, partition_id=partition.partition_id
                        )
                        if ctx is not None:
                            chunk = chunk.model_copy(update={"graph_context": ctx})
                    except Exception:
                        logger.debug(
                            "Chunk graph_context probe failed for %s", chunk_id, exc_info=True
                        )
        if chunk is None:
            raise CodeIndexingError(
                ErrorCode.CHUNK_NOT_FOUND,
                f"Unknown chunk: {chunk_id}; chunk ids come from search_code or find_symbol "
                "results and change when the file is re-indexed",
                chunk_id=chunk_id,
            )
        return chunk

    @staticmethod
    def _graph_diagnostic_ranks(
        hits: Sequence[SearchHit],
        known_graph_signals: Mapping[str, tuple[float, str | None]],
    ) -> dict[str, int]:
        graph_cids = [hit.chunk_id for hit in hits if hit.chunk_id in known_graph_signals]
        graph_cids.sort(key=lambda cid: (-known_graph_signals[cid][0], cid))
        ranks: dict[str, int] = {}
        rank = 0
        prev: float | None = None
        for pos, cid in enumerate(graph_cids, start=1):
            rounded = round(known_graph_signals[cid][0], 8)
            if rounded != prev:
                rank = pos
                prev = rounded
            ranks[cid] = rank
        return ranks

    def _explain_example_hits(
        self,
        hits: list[SearchHit],
        known_vector_scores: Mapping[str, float],
        known_graph_signals: Mapping[str, tuple[float, str | None]],
    ) -> list[SearchHit]:
        vec_cids = [hit.chunk_id for hit in hits if hit.chunk_id in known_vector_scores]
        vec_cids.sort(key=lambda cid: (-known_vector_scores[cid], cid))
        vec_ranks = {cid: pos for pos, cid in enumerate(vec_cids, start=1)}
        graph_ranks = self._graph_diagnostic_ranks(hits, known_graph_signals)
        explained: list[SearchHit] = []
        for hit in hits:
            v_score = known_vector_scores.get(hit.chunk_id)
            g_sig = known_graph_signals.get(hit.chunk_id)
            if v_score is None and g_sig is None:
                explained.append(hit)
                continue
            explained.append(
                hit.model_copy(
                    update={
                        "explanation": RankExplanation(
                            vector_score=v_score,
                            vector_rank=vec_ranks.get(hit.chunk_id),
                            graph_score=g_sig[0] if g_sig is not None else None,
                            graph_rank=graph_ranks.get(hit.chunk_id),
                            graph_via=g_sig[1] if g_sig is not None else None,
                        )
                    }
                )
            )
        return explained

    def _explain_hits(
        self,
        query: str,
        query_vector: list[float],
        hits: list[SearchHit],
        condition: str | None,
        selected: dict[str, list[PartitionRef]],
        *,
        known_vector_scores: Mapping[str, float] | None = None,
        known_graph_signals: Mapping[str, tuple[float, str | None]] | None = None,
    ) -> list[SearchHit]:
        """Attach per-signal ranking explanations; never fails the search.

        The caller must already hold the partitions' access (``partitions_access``).
        """
        wanted: dict[str, list[str]] = {}
        for hit in hits:
            wanted.setdefault(hit.project_id, []).append(hit.chunk_id)
        partition_ids = {
            project_id: [ref.partition_id for ref in refs] for project_id, refs in selected.items()
        }
        components: dict[str, RankComponents] = {}
        try:
            components = self.store.explain_hits(
                query,
                query_vector,
                wanted,
                condition,
                partition_ids=partition_ids,
                known_vector_scores=known_vector_scores,
            )
        except Exception:
            logger.debug("Ranking explanation probes failed", exc_info=True)
            return hits
        graph_signals = known_graph_signals or {}
        if not components and not graph_signals:
            return hits
        graph_ranks = self._graph_diagnostic_ranks(hits, graph_signals)
        explained: list[SearchHit] = []
        for hit in hits:
            component = components.get(hit.chunk_id)
            g_sig = graph_signals.get(hit.chunk_id)
            if component is None and g_sig is None:
                explained.append(hit)
            else:
                base: dict[str, Any] = dict(component) if component is not None else {}
                if g_sig is not None:
                    base["graph_score"] = round(g_sig[0], 6)
                    base["graph_rank"] = graph_ranks.get(hit.chunk_id)
                    base["graph_via"] = g_sig[1]
                explained.append(hit.model_copy(update={"explanation": RankExplanation(**base)}))
        return explained

    @staticmethod
    def _in_condition(column: str, values: list[str]) -> str:
        return f"{column} IN ({', '.join(_quoted(value) for value in values)})"

    @staticmethod
    def _hit(chunk: ChunkPreview, names: Mapping[str, str], score: float) -> SearchHit:
        snippet = chunk.content[:4_000]
        truncated = len(chunk.content) > len(snippet)
        if chunk.source_start_byte is not None and chunk.source_end_byte is not None:
            # An embedding window carries only part of its source chunk, even
            # when symbol lookup reports the whole chunk's line range; get_chunk
            # returns the rest.
            source_bytes = chunk.source_end_byte - chunk.source_start_byte
            truncated = truncated or len(snippet.encode("utf-8")) < source_bytes
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
            truncated=truncated,
            is_test=chunk.is_test,
        )
