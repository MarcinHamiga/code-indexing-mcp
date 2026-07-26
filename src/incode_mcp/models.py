"""Immutable domain models."""

from pathlib import Path
from typing import Annotated

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


class StoredChunk(FrozenModel):
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


class CodeChunk(StoredChunk):
    pass


class ProjectStatus(FrozenModel):
    project: ProjectInfo
    state: str
    file_count: int
    chunk_count: int


class RemovalReport(FrozenModel):
    project_id: str
    removed: bool
