"""Immutable domain models."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue


class _PathAsPlainString:
    """Emit a plain string schema for Path fields.

    Pydantic marks ``pathlib.Path`` with ``"format": "path"``, which is not a
    standard JSON Schema format and makes strict MCP clients warn.
    """

    def __get_pydantic_json_schema__(
        self, schema: JsonSchemaValue, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        del handler
        result = {"type": "string"}
        if "title" in schema:
            result["title"] = schema["title"]
        return result


SerializablePath = Annotated[Path, _PathAsPlainString()]

LEGACY_DEFAULT_INCLUDES_V1 = [
    "**/*.py",
    "**/*.pyi",
    "**/*.js",
    "**/*.jsx",
    "**/*.mjs",
    "**/*.cjs",
    "**/*.ts",
    "**/*.tsx",
    "**/*.mts",
    "**/*.cts",
]

DEFAULT_INCLUDES = [*LEGACY_DEFAULT_INCLUDES_V1, "**/*.java"]

# The kinds TreeSitterExtractor emits, plus the "_part" variants it produces when a
# definition is split across chunks. Closed so MCP clients get an enum instead of a
# free-text field; extend both halves together when a query file gains a capture.
ChunkKind = Literal[
    "annotation",
    "class",
    "constant",
    "constructor",
    "enum",
    "function",
    "interface",
    "method",
    "module",
    "record",
    "type",
    "annotation_part",
    "class_part",
    "constant_part",
    "constructor_part",
    "enum_part",
    "function_part",
    "interface_part",
    "method_part",
    "record_part",
    "type_part",
]

# Mirrors scanner.LANGUAGES values. Kept here rather than imported from scanner so
# models stays free of scanner imports.
LanguageName = Literal["python", "java", "javascript", "typescript", "tsx"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class ScanConfig(FrozenModel):
    include: list[str] = Field(default_factory=lambda: list(DEFAULT_INCLUDES))
    exclude: list[str] = Field(default_factory=list)
    max_file_bytes: int = Field(default=1_048_576, gt=0)


class ProjectInfo(FrozenModel):
    version: int = 1
    id: str
    name: str
    root: SerializablePath
    scan: ScanConfig = Field(default_factory=ScanConfig)


class ScannedFile(FrozenModel):
    path: SerializablePath
    absolute_path: SerializablePath
    language: str
    size: int
    mtime_ns: int
    # Streaming scans attach changed-file bytes so the indexer consumes them
    # without a second read. Collected scans deliberately leave this as None.
    content: bytes | None = None


class SkippedFile(FrozenModel):
    path: SerializablePath
    reason: str
    detail: str | None = None


class ScanResult(FrozenModel):
    files: list[ScannedFile]
    skipped: list[SkippedFile]


class ExtractedChunk(FrozenModel):
    kind: str
    symbol: str | None = None
    qualified_symbol: str | None = None
    parent_symbol: str | None = None
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    content: str
    embedding_text: str
    search_text: str
    part_index: int = 0
    # The two halves ``embedding_text`` and ``search_text`` are composed from.
    # Keeping them lets a token window be recomposed with the same context
    # header and identifier tail as the whole chunk, instead of the header being
    # windowed away from every part after the first.
    embedding_prefix: str = ""
    search_suffix: str = ""


class ExtractionResult(FrozenModel):
    chunks: list[ExtractedChunk]
    has_errors: bool = False


class StoredFile(FrozenModel):
    file_id: str
    project_id: str
    path: str
    language: str
    size: int
    mtime_ns: int
    content_hash: str
    has_errors: bool = False
    error: str | None = None
    indexed_at: int


class IndexedChunk(FrozenModel):
    """A committed chunk without its embedding vector.

    Read paths that only need chunk text and offsets use this so a whole project's
    768-float vectors are not decoded into Python lists for no consumer.
    """

    chunk_id: str
    file_id: str
    project_id: str
    path: str
    language: str
    kind: str
    symbol: str | None = None
    qualified_symbol: str | None = None
    parent_symbol: str | None = None
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    content: str
    embedding_text: str
    search_text: str
    content_hash: str
    part_index: int = 0


class StoredChunk(IndexedChunk):
    """A chunk as written to storage, vector included."""

    vector: list[float]


class ChunkPreview(FrozenModel):
    """A query result that deliberately excludes embedding and index payloads."""

    chunk_id: str
    project_id: str
    path: str
    language: str
    kind: str
    symbol: str | None = None
    qualified_symbol: str | None = None
    parent_symbol: str | None = None
    start_line: int
    end_line: int
    content: str = ""


class IndexIssue(FrozenModel):
    path: str
    message: str


class IndexReport(FrozenModel):
    project_id: str
    discovered_files: int = 0
    indexed_files: int = 0
    parsed_files: int = 0
    embedded_chunks: int = 0
    unchanged_files: int = 0
    metadata_only_files: int = 0
    removed_files: int = 0
    skipped_files: int = 0
    errors: list[IndexIssue] = Field(default_factory=list)
    duration_ms: int = 0
    # Phase breakdown of duration_ms. They do not sum to it: lock acquisition and
    # project bookkeeping sit outside, and embedding includes the one-time worker
    # spawn and model load. Optional so older clients keep validating reports.
    scan_duration_ms: int | None = None
    parse_duration_ms: int | None = None
    embed_duration_ms: int | None = None
    commit_duration_ms: int | None = None
    # Stable Phase 1 telemetry names. Keep the *_duration_ms fields above for
    # clients that adopted the earlier memory-hardening report.
    embedding_backend: str = "cpu"
    embedding_batch_size: int = 1
    scan_ms: int | None = None
    parse_ms: int | None = None
    embed_ms: int | None = None
    commit_ms: int | None = None
    fallback_count: int = 0
    memory_budget_bytes: int | None = None
    peak_memory_bytes: int | None = None
    worker_used: bool = False
    # Token-window telemetry, populated only on worker runs. embedded_segments
    # counts what the worker embedded, which includes segments from files that
    # later failed and were not committed, so it can exceed embedded_chunks.
    # token_windowing=False means no tokenizer was reachable and sequence length
    # went unbounded.
    embedded_segments: int | None = None
    embedded_tokens: int | None = None
    embedding_retries: int | None = None
    worker_termination_reason: str | None = None
    token_windowing: bool | None = None


class SearchHit(FrozenModel):
    chunk_id: str
    project_id: str
    project_name: str
    path: str
    language: str
    kind: str
    symbol: str | None = None
    qualified_symbol: str | None = None
    start_line: int
    end_line: int
    score: float
    snippet: str
    truncated: bool = False


class SearchResponse(FrozenModel):
    query: str
    hits: list[SearchHit]


class SymbolResponse(FrozenModel):
    name: str
    hits: list[SearchHit]


class OutlineItem(FrozenModel):
    kind: str
    symbol: str
    qualified_symbol: str
    parent_symbol: str | None = None
    start_line: int
    end_line: int


class OutlineResponse(FrozenModel):
    project_id: str
    path: str
    items: list[OutlineItem]


class CodeChunk(FrozenModel):
    """One indexed chunk as returned to a caller.

    Deliberately not a StoredChunk subclass. Inheriting the storage row shipped the
    768-dimension vector and both derived text columns to MCP clients: 72% of the
    response was the vector, and the code arrived three times over as content,
    embedding_text, and search_text. Adding a storage column must not silently
    widen this payload, so the fields are listed rather than inherited.
    """

    chunk_id: str
    file_id: str
    project_id: str
    path: str
    language: str
    kind: str
    symbol: str | None = None
    qualified_symbol: str | None = None
    parent_symbol: str | None = None
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    content: str
    content_hash: str
    part_index: int = 0


class ProjectStatus(FrozenModel):
    project: ProjectInfo
    state: str
    file_count: int
    chunk_count: int


class RemovalReport(FrozenModel):
    project_id: str
    removed: bool
