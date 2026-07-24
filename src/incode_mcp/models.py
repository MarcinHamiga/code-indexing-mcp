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
    memory_budget_bytes: int | None = None
    peak_memory_bytes: int | None = None
    worker_used: bool = False


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
