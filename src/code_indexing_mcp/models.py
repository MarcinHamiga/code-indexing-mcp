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

LEGACY_DEFAULT_INCLUDES_V2 = [*LEGACY_DEFAULT_INCLUDES_V1, "**/*.java"]

LEGACY_DEFAULT_INCLUDES_V3 = [
    *LEGACY_DEFAULT_INCLUDES_V2,
    "**/*.cs",
    "**/*.csx",
    "**/*.gd",
    "**/*.gdshader",
    "**/*.gdshaderinc",
    "**/*.tres",
    "**/*.tscn",
    "**/*.godot",
    "**/*.sql",
    "**/*.yaml",
    "**/*.yml",
    "**/*.json",
]

DEFAULT_INCLUDES = [
    *LEGACY_DEFAULT_INCLUDES_V3,
    "**/*.go",
    "**/*.tf",
    "**/*.tfvars",
    "**/*.rs",
    "**/*.c",
    "**/*.h",
    "**/*.cc",
    "**/*.cpp",
    "**/*.cxx",
    "**/*.hh",
    "**/*.hpp",
    "**/*.hxx",
    "**/*.lua",
]

# The kinds TreeSitterExtractor emits, plus the "_part" variants it produces when a
# definition is split across chunks. Closed so MCP clients get an enum instead of a
# free-text field; extend both halves together when a query file gains a capture.
ChunkKind = Literal[
    "annotation",
    "array",
    "class",
    "constant",
    "constructor",
    "enum",
    "function",
    "index",
    "interface",
    "method",
    "module",
    "object",
    "property",
    "record",
    "signal",
    "struct",
    "table",
    "trigger",
    "type",
    "view",
    "annotation_part",
    "array_part",
    "class_part",
    "constant_part",
    "constructor_part",
    "enum_part",
    "function_part",
    "index_part",
    "interface_part",
    "method_part",
    "object_part",
    "property_part",
    "record_part",
    "signal_part",
    "struct_part",
    "table_part",
    "trigger_part",
    "type_part",
    "view_part",
]

# Mirrors scanner.LANGUAGES values. Kept here rather than imported from scanner so
# models stays free of scanner imports.
LanguageName = Literal[
    "python",
    "java",
    "javascript",
    "typescript",
    "tsx",
    "csharp",
    "sql",
    "gdscript",
    "gdshader",
    "godot_resource",
    "yaml",
    "json",
    "go",
    "terraform",
    "rust",
    "c",
    "cpp",
    "lua",
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
    # Set when a run started on an accelerator and finished somewhere else.
    embedding_fallback_reason: str | None = None
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
    # Why this run embedded where it did, when the workload crossover was what
    # decided it -- so a run that stayed on CPU because it was small is
    # distinguishable from one that fell back to CPU because something broke.
    # embedded_characters is what the decision was measured against.
    embedded_characters: int | None = None
    embedding_crossover_characters: int | None = None
    embedding_selection_reason: str | None = None


class ModelStatus(FrozenModel):
    """What the embedding stack resolved to on this machine, and why.

    ``embedding_model`` rather than ``model_id`` because pydantic reserves the
    ``model_`` field prefix; the value is the model identifier either way.
    """

    embedding_model: str
    dimension: int
    requested_accelerator: str
    resolved_accelerator: str
    device: str
    execution_provider: str
    available_providers: list[str]
    stability: str
    precision: str
    runtime_version: str
    batch_size: int
    # "explicit" when configured, "measured" when calibration settled on it,
    # "reduced" when a memory-ceiling overrun forced it down from what was
    # measured, "default" when none of those applied.
    batch_calibration: str
    # "hit" or "miss" against the local probe cache; "not-applicable" on CPU,
    # which needs no probe to be trusted.
    probe_cache_state: str
    strict: bool
    # The GPU driver the accelerator environment was probed against, when one
    # was prepared. Empty on CPU, where no driver is in the picture.
    driver_version: str = ""
    # The interpreter passage embedding runs in when that is not this one, and
    # the accelerator the installer prepared, whether or not it was selected.
    accelerator_environment: str | None = None
    accelerator_prepared: str | None = None
    fallback_reason: str | None = None
    # What calibration measured on this machine, and the run size above which
    # starting the accelerator repays its model load. None means unmeasured --
    # or, for the crossover, that the accelerator never overtakes CPU at any
    # size, which is a different statement from "the threshold is large".
    cpu_characters_per_second: float | None = None
    accelerator_characters_per_second: float | None = None
    accelerator_load_ms: int | None = None
    crossover_characters: int | None = None
    # The one setting change these numbers actually argue for, when they argue
    # for one. Not advice in general -- only what the measurements support.
    recommended_override: str | None = None


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
