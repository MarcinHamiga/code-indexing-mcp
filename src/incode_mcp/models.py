"""Immutable domain models."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_INCLUDES = [
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
    root: Path
    scan: ScanConfig = Field(default_factory=ScanConfig)


class ScannedFile(FrozenModel):
    path: Path
    absolute_path: Path
    language: str
    size: int
    mtime_ns: int


class SkippedFile(FrozenModel):
    path: Path
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
