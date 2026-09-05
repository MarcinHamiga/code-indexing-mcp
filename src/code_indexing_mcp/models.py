"""Immutable domain models."""

import hashlib
import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, GetJsonSchemaHandler, model_validator
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

# Bump only when the normalized structural-row contract changes. Coverage rows
# make a new generation discoverable without coupling it to project metadata.
# Version 4 puts the reference kind in the row identity. Bumping it also
# discards any generation written by version 3, whose colliding ids are what
# made a project unindexable.
# Version 5 adds Go to STRUCTURAL_LANGUAGES. Every language's version-bump step
# is what makes parse-only reference backfill re-extract that language's files
# (Go files already carried version-4 coverage rows with zero occurrences).
# Version 6 adds Rust; version 7 adds Java, with on-demand-import semantics.
# Version 8 adds C#, with namespace identity carried on export rows.
#
# Lives here rather than in indexing.py (originally its home) because
# reference_service.py needs it too and importing it from indexing.py made
# that a sideways dependency between two peer services -- see D3 in
# docs/plans/2026-09-02-review-remediation-5-application-split-plan.md.
# indexing.py re-exports the name for one release so nothing importing it from
# there breaks.
REFERENCE_SCHEMA_VERSION = 8


def content_digest(value: str | bytes) -> str:
    """Return the sha256 hex digest of *value*, encoding text as UTF-8 first.

    Identifies file content and derives stable ids for chunks, structural rows,
    and reference rows. Public (moved here from indexing.py's private
    ``_digest`` per D3) because reference_service.py computes the same digest
    independently of indexing.py's own pipeline.
    """
    data = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


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

# A repository-shipped marker is trusted input up to this point: a project.toml
# committed to a repo the user opens is honoured without prompting, so its
# `max_file_bytes` cannot be allowed to raise the ceiling arbitrarily high and
# turn a single oversized file into an unbounded read and embed.
MAX_FILE_BYTES_CEILING = 16 * 1024 * 1024

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

ReferenceKind = Literal[
    "import",
    "export",
    "call",
    "type_use",
    "inheritance",
    "decorator",
    "read",
    "write",
]

ParameterKind = Literal[
    "positional_only",
    "positional",
    "keyword_only",
    "variadic",
    "keyword_variadic",
]

ResolutionLevel = Literal["exact", "likely", "unresolved"]

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


# Why an indexing run started. ``schema-rebuild`` and ``maintenance`` are
# reserved for later releases; every current run is one of the other five.
IndexTrigger = Literal[
    "manual",
    "startup",
    "watcher",
    "lazy-query",
    "reference-backfill",
    "schema-rebuild",
    "maintenance",
]


class ScanConfig(FrozenModel):
    include: list[str] = Field(default_factory=lambda: list(DEFAULT_INCLUDES))
    exclude: list[str] = Field(default_factory=list)
    max_file_bytes: int = Field(default=1_048_576, gt=0, le=MAX_FILE_BYTES_CEILING)


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


# The skip reasons the scanner itself can attach to a file. Content-level
# rejections (binary, encoding, parse, embedding) belong to the indexer and
# never appear in a scan inspection.
SCAN_SKIP_REASONS = frozenset({"unsupported", "ignored", "symlink", "oversized", "unreadable"})


class ScanInspectionItem(FrozenModel):
    """One repository-relative scan outcome, without source contents."""

    path: SerializablePath
    outcome: Literal["eligible", "skipped"]
    language: str | None = None
    reason: str | None = None
    detail: str | None = None
    size: int | None = None
    mtime_ns: int | None = None


class ScanInspectionPage(FrozenModel):
    """One page of a dry-run scan inspection.

    The page is a stat-only view of what an index run would do: it never
    embeds, mutates the index, or persists a complete manifest.
    """

    schema_version: int = 1
    project: ProjectInfo | None = None
    items: list[ScanInspectionItem] = Field(default_factory=list)
    next_cursor: str | None = None


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


class CallShape(FrozenModel):
    positional_count: int = 0
    keywords: list[str] = Field(default_factory=list)
    has_positional_spread: bool = False
    has_keyword_spread: bool = False
    type_argument_count: int | None = None
    constructor: bool = False


class ParameterShape(FrozenModel):
    name: str = Field(description="Parameter name.")
    kind: ParameterKind = Field(
        description=(
            "One of positional_only, positional, keyword_only, variadic (*args-style), "
            "or keyword_variadic (**kwargs-style)."
        )
    )
    required: bool = Field(description="Whether this parameter has no default value.")
    position: int = Field(description="Zero-based position among this signature's parameters.")
    # True when this slot is a destructured pattern (`{ a, b }` / `[a, b]`)
    # collapsed to one positional parameter (E7). `name` is a synthesized,
    # non-authoritative label in that case -- signature comparisons that
    # depend on matching it by name must route to `review` instead of
    # trusting it as a real keyword/identifier.
    destructured: bool = Field(
        default=False,
        description=(
            "Whether this slot is a destructured pattern (e.g. `{ a, b }`) collapsed to one "
            "positional parameter; if true, `name` is a synthesized label, not authoritative."
        ),
    )


class ExtractedReference(FrozenModel):
    kind: ReferenceKind
    written_name: str
    target_name: str
    source_qualified_symbol: str | None = None
    module_path: str | None = None
    imported_name: str | None = None
    alias: str | None = None
    receiver_text: str | None = None
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    call_shape: CallShape | None = None


class ExtractedDeclarationShape(FrozenModel):
    symbol: str
    qualified_symbol: str
    kind: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    parameters: list[ParameterShape] = Field(default_factory=list)


class ExtractionResult(FrozenModel):
    chunks: list[ExtractedChunk]
    references: list[ExtractedReference] = Field(default_factory=list)
    declarations: list[ExtractedDeclarationShape] = Field(default_factory=list)
    has_errors: bool = False
    # Wall time spent specifically on structural reference extraction (the
    # `_structural_records` pass), separate from parsing and chunking, so a
    # caller can report reference-extraction cost without mislabeling the
    # whole parse phase as it (T1).
    reference_extraction_ns: int = 0


class ReferenceCoverage(FrozenModel):
    """The structural extraction generation known for one indexed file."""

    file_id: str
    path: str
    content_hash: str
    schema_version: int


class ReferenceBackfillReport(FrozenModel):
    """Outcome of a parse-only structural-index catch-up run."""

    project_id: str
    files_checked: int = 0
    files_backfilled: int = 0
    files_current: int = 0
    incomplete_paths: list[str] = Field(default_factory=list)
    stale_paths: list[str] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.incomplete_paths and not self.stale_paths


class DeclarationSelector(FrozenModel):
    """One declaration identity, by chunk id or its stable source location."""

    chunk_id: str | None = Field(
        default=None,
        description=(
            "Chunk id from a search_code or find_symbol hit. Cannot be combined with "
            "project, path, or qualified_symbol -- provide this alone, or all three of them."
        ),
    )
    project: str | None = Field(
        default=None,
        description="Project id, name, or path. Required with path and qualified_symbol.",
    )
    path: str | None = Field(
        default=None,
        description=(
            "Repo-relative POSIX path to the file holding the declaration "
            "(forward slashes, relative to the project root). Required with project and "
            "qualified_symbol."
        ),
    )
    qualified_symbol: str | None = Field(
        default=None,
        description=(
            "Dotted qualified symbol name, e.g. 'Outer.method'. Required with project and path."
        ),
    )

    @model_validator(mode="after")
    def _one_selector_mode(self) -> "DeclarationSelector":
        by_chunk = self.chunk_id is not None
        by_location = all(
            value is not None for value in (self.project, self.path, self.qualified_symbol)
        )
        any_location = any(
            value is not None for value in (self.project, self.path, self.qualified_symbol)
        )
        if by_chunk and any_location:
            raise ValueError("chunk_id cannot be combined with project, path, or qualified_symbol")
        if by_chunk:
            return self
        if not by_location:
            raise ValueError(
                "Provide exactly chunk_id or project, path, and qualified_symbol together"
            )
        return self


class SelectedDeclaration(FrozenModel):
    project_id: str
    file_id: str
    path: str
    language: str
    symbol: str
    qualified_symbol: str
    kind: str
    start_line: int
    end_line: int
    chunk_id: str | None = None


class ReferenceHit(FrozenModel):
    reference_id: str
    project_id: str
    path: str
    language: str
    kind: ReferenceKind
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    snippet: str = ""
    written_name: str | None = None
    resolution: ResolutionLevel
    reason_code: str
    explanation: str


class ReferenceLimitation(FrozenModel):
    code: str
    explanation: str
    path: str | None = None


class ReferenceResponse(FrozenModel):
    selected: SelectedDeclaration
    hits: list[ReferenceHit] = Field(default_factory=list)
    limitations: list[ReferenceLimitation] = Field(default_factory=list)
    cursor: str | None = None
    snapshot_version: int = 0


class RenameOperation(FrozenModel):
    kind: Literal["rename"] = Field(default="rename", description="Discriminator; always 'rename'.")
    new_name: str = Field(description="The declaration's new name.")


class SignatureChangeOperation(FrozenModel):
    kind: Literal["signature_change"] = Field(
        default="signature_change", description="Discriminator; always 'signature_change'."
    )
    parameters: list[ParameterShape] = Field(
        description="The declaration's proposed new full parameter list, in order."
    )


RefactorOperation = Annotated[
    RenameOperation | SignatureChangeOperation, Field(discriminator="kind")
]


class RefactorFinding(ReferenceHit):
    edit_required: bool = False
    # The identifier to rewrite, which is narrower than the occurrence range:
    # the call `auth.authorize(u)` spans `auth.authorize`, but only `authorize`
    # may be replaced. Null when the identifier could not be located uniquely,
    # which means the edit has to be made by hand.
    edit_start_byte: int | None = None
    edit_end_byte: int | None = None


class CompletenessReport(FrozenModel):
    state: Literal["complete", "complete_with_dynamic_limitations", "incomplete"] = "complete"
    explanation: str = "All indexed structural candidates were considered."


class ImpactEdge(FrozenModel):
    source: SelectedDeclaration
    target: SelectedDeclaration
    kinds: list[ReferenceKind] = Field(default_factory=list)
    possible: bool = False
    tainted: bool = False
    cycle_to_depth: int | None = None


class ImpactReview(FrozenModel):
    source: SelectedDeclaration
    hit: ReferenceHit


class ImpactLayer(FrozenModel):
    depth: int
    edges: list[ImpactEdge] = Field(default_factory=list)
    review: list[ImpactReview] = Field(default_factory=list)


class ImpactBudgetExhaustion(FrozenModel):
    depth: int
    node: SelectedDeclaration
    unvisited_frontier: int


class ImpactRadiusResponse(FrozenModel):
    selected: SelectedDeclaration
    layers: list[ImpactLayer] = Field(default_factory=list)
    visited: int = 0
    budget_exhausted: bool = False
    budget_exhaustion: ImpactBudgetExhaustion | None = None
    limitations: list[ReferenceLimitation] = Field(default_factory=list)
    cursor: str | None = None
    snapshot_version: int = 0
    completeness: CompletenessReport = Field(default_factory=CompletenessReport)


class RefactorCounts(FrozenModel):
    must_change: int = 0
    likely_change: int = 0
    review: int = 0
    evidence: int = 0


class RefactorAnalysis(FrozenModel):
    selected: SelectedDeclaration
    operation: RefactorOperation
    must_change: list[RefactorFinding] = Field(default_factory=list)
    likely_change: list[RefactorFinding] = Field(default_factory=list)
    review: list[RefactorFinding] = Field(default_factory=list)
    evidence: list[RefactorFinding] = Field(default_factory=list)
    limitations: list[ReferenceLimitation] = Field(default_factory=list)
    counts: RefactorCounts = Field(default_factory=RefactorCounts)
    cursor: str | None = None
    completeness: CompletenessReport = Field(default_factory=CompletenessReport)

    @property
    def findings(self) -> list[RefactorFinding]:
        """Return every finding while preserving the caller-facing priority order."""
        return [*self.must_change, *self.likely_change, *self.review, *self.evidence]


class PatchEdit(FrozenModel):
    """One verified identifier replacement inside a patch, in raw file bytes."""

    path: str
    edit_start_byte: int
    edit_end_byte: int
    old_text: str
    new_text: str


class RefactorPatch(FrozenModel):
    """The deterministic subset of a rename analysis, rendered as a unified diff.

    `patch` is the `git apply`-able diff text and is empty when nothing was
    applied. Findings that could not be proven current stay out of the patch
    and are reported verbatim in `unapplied` (not deterministic edits) and
    `conflicted` (stale or unverifiable), so a partial patch can never read
    as a finished rename.
    """

    selected: SelectedDeclaration
    operation: RefactorOperation
    patch: str
    edits: list[PatchEdit] = Field(default_factory=list)
    applied: int = 0
    unapplied: list[RefactorFinding] = Field(default_factory=list)
    conflicted: list[RefactorFinding] = Field(default_factory=list)
    snapshot_version: int = 0
    slot_id: str | None = None
    operation_digest: str
    limitations: list[ReferenceLimitation] = Field(default_factory=list)
    completeness: CompletenessReport = Field(default_factory=CompletenessReport)


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

    Mirrors the chunk row: project_id is not stored on it because the owning
    partition knows it. content_hash stays on the row so a chunk response is
    always one coherent generation, even while a files-table update commits.
    """

    chunk_id: str
    file_id: str
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
    identifier_terms: str
    part_index: int = 0
    content_hash: str = ""


class StoredChunk(IndexedChunk):
    """A chunk as written to storage, vector included."""

    vector: list[float]


class ChunkPreview(FrozenModel):
    """A query result that deliberately excludes embedding and index payloads."""

    chunk_id: str
    project_id: str
    # Optional and unset by most producers (search/outline previews never
    # needed it); `LanceStore.find_declarations` sets it so `_select` can
    # build a `SelectedDeclaration` without a second lookup.
    file_id: str | None = None
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
    # T1: reference extraction's own timing and this run's own staged row
    # count, distinct from `parse_duration_ms` (parsing + chunking +
    # reference extraction together) and from a whole-project table read.
    reference_extraction_duration_ms: int | None = None
    staged_reference_rows: int = 0
    # Durable run identity and why it ran. Optional so older clients keep
    # validating reports and older stored reports keep validating.
    run_id: str | None = None
    trigger: IndexTrigger = "manual"
    # Counter contract for the audit record: skip reasons are broken out by
    # cause, failed files are parse/embedding failures, and the byte/chunk
    # counters track what the run actually read, extracted, and staged.
    failed_files: int = 0
    skip_reasons: dict[str, int] = Field(default_factory=dict)
    skipped_samples: list[str] = Field(default_factory=list)
    bytes_read: int = 0
    chunks_extracted: int = 0
    chunks_staged: int = 0
    staged_bytes: int = 0


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


class RankExplanation(FrozenModel):
    """Per-hit breakdown of why a hybrid search hit ranked where it did.

    Diagnostic only, and purely additive: the hybrid query fuses per-modality
    *ranks* (reciprocal rank fusion), so no exact decomposition of the final
    score into weights exists. ``fts_score`` is the best BM25-style ``_score``
    across the content and identifier-terms probes; ``vector_score`` is
    ``1 - cosine distance`` from the vector probe, the same mapping the
    vector-only path uses. Ranks are 1-based positions within the explained-hit
    set per modality. A None signal means that probe did not return the chunk.
    """

    fts_score: float | None = None
    vector_score: float | None = None
    fts_rank: int | None = None
    vector_rank: int | None = None


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
    explanation: RankExplanation | None = None


class SearchResponse(FrozenModel):
    query: str
    hits: list[SearchHit]


class ExampleSearchResponse(FrozenModel):
    language: str | None
    segments: int
    hits: list[SearchHit]


class SearchAcrossProjectsResponse(FrozenModel):
    hits: list[SearchHit]
    query: str | None = None
    language: str | None = None
    segments: int | None = None


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
    widen this payload, so the fields are listed rather than inherited. Chunk
    rows no longer persist project_id or content_hash either: both are injected
    by the read path from the owning partition and the files table.
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


# Why an indexing run started. ``schema-rebuild`` and ``maintenance`` are
# reserved for later releases; every current run is one of the other five.


class RunAudit(FrozenModel):
    """One durable audit record of an indexing or backfill run.

    Values are bounded before they are written: at most ``MAX_ERROR_SAMPLES``
    error details and ``MAX_SKIPPED_SAMPLES`` skipped-path samples are kept,
    so a pathological run cannot grow the history database without bound.
    """

    run_id: str
    project_id: str
    trigger: IndexTrigger
    server_version: str = ""
    git_revision: str | None = None
    model_id: str = ""
    schema_version: int = 0
    scan_config_hash: str = ""
    force: bool = False
    # Why a schema-rebuild run replaced its partition (model or schema
    # mismatch, for example). None for ordinary runs.
    rebuild_reason: str | None = None
    # Owning process, so startup can tell a crashed run from one another live
    # process is still executing.
    pid: int = Field(default_factory=os.getpid)
    started_at: str = ""
    finished_at: str | None = None
    # running, completed, failed, or interrupted.
    state: str = "running"
    phase_durations: dict[str, int] = Field(default_factory=dict)
    eligible_files: int = 0
    changed_files: int = 0
    unchanged_files: int = 0
    parsed_files: int = 0
    failed_files: int = 0
    removed_files: int = 0
    skipped_total: int = 0
    chunks_extracted: int = 0
    chunks_embedded: int = 0
    chunks_staged: int = 0
    staged_bytes: int = 0
    bytes_read: int = 0
    skip_reasons: dict[str, int] = Field(default_factory=dict)
    errors: list[IndexIssue] = Field(default_factory=list)
    skipped_samples: list[str] = Field(default_factory=list)
    embedding_backend: str = "cpu"
    embedding_fallback_reason: str | None = None
    worker_used: bool = False
    # Best-effort storage table versions around the run, not full partition
    # traversals: audit recording must stay inexpensive on any repository.
    storage_before: dict[str, int] = Field(default_factory=dict)
    storage_after: dict[str, int] = Field(default_factory=dict)


class RunSummary(FrozenModel):
    """Compact summary of one completed run for status surfaces."""

    run_id: str
    trigger: IndexTrigger
    state: str
    started_at: str
    finished_at: str | None = None
    duration_ms: int = 0
    eligible_files: int = 0
    changed_files: int = 0
    failed_files: int = 0
    skipped_total: int = 0
    chunks_embedded: int = 0


class HistoryPage(FrozenModel):
    """One page of a project's indexing history."""

    schema_version: int = 1
    project: ProjectInfo | None = None
    runs: list[RunAudit] = Field(default_factory=list)
    next_cursor: str | None = None


class IndexProgress(BaseModel):
    """A point-in-time snapshot of one project's indexing run.

    Every counter's name matches what it counts: ``candidates_*`` cover every
    path the scanner examined, ``eligible_files`` the files that passed the
    scan, and the ``*_files`` counters the eligible ones. Totals stay unset
    while the scanner streams and the run genuinely does not know them, and a
    candidate count is never compared with an eligible-file total.

    Defined here rather than in progress.py so progress.py stays a plain
    consumer of this module: it used to be the other way around, and the
    resulting models/progress cycle made the package import-order dependent.
    """

    project_id: str
    run_id: str = ""
    trigger: IndexTrigger = "manual"
    phase: str = "scanning"
    # Every path the scanner examined, whether it became eligible or was
    # skipped. A first index has no honest candidates_total: the scanner
    # streams, so the total is only known once the walk has finished.
    candidates_seen: int = 0
    candidates_total: int | None = None
    eligible_files: int = 0
    unchanged_files: int = 0
    changed_files: int = 0
    parsed_files: int = 0
    failed_files: int = 0
    skipped_total: int = 0
    skipped_by_reason: dict[str, int] = Field(default_factory=dict)
    bytes_read: int = 0
    chunks_extracted: int = 0
    chunks_embedded: int = 0
    chunks_staged: int = 0
    staged_bytes: int = 0
    current_path: str | None = None
    started_at: float = 0.0
    updated_at: float = 0.0
    # Monotonic anchor for phase durations, never a wall-clock timestamp:
    # it must strictly advance across phase changes on every platform.
    phase_started_at: float = 0.0
    pid: int = Field(default_factory=os.getpid)
    # Branch-slot identity of the run: which slot is being written, which
    # selector chose it, and the HEAD captured before scanning began. A
    # watcher uses these to ignore a finishing prior-branch job.
    slot_id: str | None = None
    selector: str | None = None
    expected_head: str | None = None
    activation_epoch: int | None = None

    @property
    def fraction(self) -> float | None:
        """Completion in ``[0, 1]``, or None when the total is unknown.

        Only candidate counts may feed this: comparing candidates seen with an
        eligible-file total would overstate (or understate) progress whenever
        the repository contains skipped paths.
        """

        if not self.candidates_total:
            return None
        return min(1.0, self.candidates_seen / self.candidates_total)

    def describe(self) -> str:
        """Render a one-line status suitable for a progress bar or a log line."""

        if self.phase == "committing":
            return "Committing the index"
        if self.phase == "extracting_references":
            return "Extracting structural references"
        if not self.candidates_seen:
            return "Scanning for changed files"
        if self.candidates_total:
            scanned = f"{self.candidates_seen}/~{self.candidates_total} candidates"
        else:
            scanned = f"{self.candidates_seen} candidates"
        parts = [f"{self.phase.capitalize()} {scanned}"]
        if self.eligible_files:
            parts.append(f"{self.eligible_files} eligible")
        if self.changed_files:
            parts.append(f"{self.changed_files} changed")
        if self.unchanged_files:
            parts.append(f"{self.unchanged_files} unchanged")
        if self.failed_files:
            parts.append(f"{self.failed_files} failed")
        if self.skipped_total:
            parts.append(f"{self.skipped_total} skipped")
        if self.chunks_embedded:
            parts.append(f"{self.chunks_embedded} chunks embedded")
        return ", ".join(parts)


class ProjectStatus(FrozenModel):
    project: ProjectInfo
    state: str
    file_count: int
    chunk_count: int
    # Live progress of whichever process is indexing this project right now,
    # and the compact summary of its most recent completed run. Never the
    # full history: status stays inexpensive.
    progress: IndexProgress | None = None
    last_run: RunSummary | None = None
    # Branch-aware fields, populated only for projects whose checkout resolves
    # to a Git or workspace slot. Optional so a pre-slot registry continues to
    # serialize without them.
    active_slot_id: str | None = None
    git_selector_kind: str | None = None
    git_selector_value: str | None = None
    git_head: str | None = None
    git_probe: str | None = None
    git_clean: bool | None = None
    branch_build_pending: bool | None = None
    # The checkout this status describes. Equal to project.root for the
    # canonical checkout; a linked worktree's root when the request arrived
    # through that worktree's marker.
    checkout_root: str | None = None


class RemovalReport(FrozenModel):
    project_id: str
    removed: bool


class FragmentLengthStats(FrozenModel):
    """Fragment-size distribution as reported by Lance's table statistics."""

    min: int | None = None
    max: int | None = None
    mean: float | None = None
    p25: int | None = None
    p50: int | None = None
    p75: int | None = None
    p99: int | None = None


class FragmentStats(FrozenModel):
    num_fragments: int = 0
    num_small_fragments: int = 0
    lengths: FragmentLengthStats | None = None


class IndexStorageStats(FrozenModel):
    name: str
    index_type: str
    columns: list[str]
    indexed_rows: int = 0
    unindexed_rows: int = 0
    size_bytes: int = 0


class TableStorageStats(FrozenModel):
    """One Lance table's storage snapshot, collected read-only."""

    name: str
    current_version: int = 0
    row_count: int = 0
    # Lance-reported logical bytes for the table's live data.
    logical_bytes: int = 0
    # Filesystem-reported physical bytes, measured without following symlinks.
    physical_bytes: int = 0
    fragment_stats: FragmentStats = Field(default_factory=FragmentStats)
    retained_version_count: int = 0
    # ISO-8601 timestamps of the oldest and newest retained versions.
    oldest_version_at: str | None = None
    newest_version_at: str | None = None
    indexes: list[IndexStorageStats] = Field(default_factory=list)


class ProjectSlot(FrozenModel):
    """One retained index slot of a project: a branch, commit, workspace, or legacy partition."""

    slot_id: str
    project_id: str
    partition_id: str
    selector_kind: str
    selector_value: str
    repository_identity: str | None = None
    # Informational only: the checkout whose git directory last indexed this
    # slot. It takes no part in the slot identity, so a branch checked out in
    # any worktree of the repository maps to the same slot.
    checkout_identity: str | None = None
    project_prefix: str = ""
    # The HEAD the partition was last indexed at, and whether the checkout was
    # clean then. A null HEAD or clean state forces full freshness validation.
    indexed_head: str | None = None
    indexed_clean: bool | None = None
    # The `git status` fingerprint at index time, and the JSON-encoded sorted
    # list of paths that were dirty or untracked then (or null when that list
    # exceeded MAX_PERSISTED_STATUS_PATHS). Together they let a freshness
    # check on a dirty checkout stat only the paths that could have changed
    # instead of walking the whole tree; either being null forces a full walk.
    indexed_status_fingerprint: str | None = None
    indexed_status_paths: str | None = None
    scan_config_hash: str = ""
    model_id: str = ""
    vector_dimension: int = 0
    schema_version: int = 0
    state: str = "pending"
    created_at: int = 0
    last_used_at: int = 0


class SlotStorageStats(FrozenModel):
    """One branch slot's storage snapshot inside a project."""

    slot_id: str
    partition_id: str
    selector_kind: str
    selector_value: str
    active: bool = False
    state: str = "pending"
    indexed_head: str | None = None
    indexed_clean: bool | None = None
    last_used_at: int | None = None
    physical_bytes: int = 0


class ProjectStorageStats(FrozenModel):
    """One project partition's storage snapshot."""

    project: ProjectInfo
    snapshot_at: str
    tables: list[TableStorageStats] = Field(default_factory=list)
    # Every retained branch slot of this project; the active slot's partition
    # is the one ``tables`` describes.
    slots: list[SlotStorageStats] = Field(default_factory=list)
    # Sum of the partition's table directories on disk.
    partition_physical_bytes: int = 0
    # False when a table version changed while the snapshot was collected, or
    # when the partition exists but its tables could not be opened.
    consistent: bool = True
    # True when the partition directory exists but its tables could not be
    # opened (a damaged or mid-mutation store), so no table statistics exist.
    partition_open_failed: bool = False


class StorageStatus(FrozenModel):
    """Installation-wide read-only storage statistics."""

    # 2: projects grew a per-slot ``slots`` list describing every retained
    # branch partition.
    schema_version: int = 2
    snapshot_at: str
    registry: TableStorageStats
    projects: list[ProjectStorageStats] = Field(default_factory=list)
    physical_bytes_total: int = 0
    consistent: bool = True
    overlap_warnings: list[str] = Field(default_factory=list)
    worktree_warnings: list[str] = Field(default_factory=list)


class MaintenanceProjectResult(FrozenModel):
    """One project partition's outcome of a maintenance pass.

    ``status`` is one of ``ok``, ``skipped``, or ``error``. A skipped project
    carries a ``skip_reason`` such as ``busy``, ``not-indexed``, ``dry-run``,
    or ``recovery-pending``. ``before`` and ``after`` hold full storage
    snapshots when they could be collected, so the deltas stay auditable
    rather than being reduced to counters.
    """

    project: ProjectInfo
    status: str = "skipped"
    skip_reason: str | None = None
    error: str | None = None
    before: ProjectStorageStats | None = None
    after: ProjectStorageStats | None = None
    # Retained versions removed and physical bytes reclaimed by the pass.
    versions_removed: int = 0
    bytes_reclaimed: int = 0
    # Pre-cleanup estimate of reclaimable bytes (physical minus logical): an
    # estimate, never a claim about what cleanup will actually free.
    reclaimable_bytes_estimate: int = 0


class MaintenanceReport(FrozenModel):
    """Outcome of one storage maintenance pass, manual or scheduled."""

    # 2: per-project before/after statistics grew the per-slot ``slots`` list.
    schema_version: int = 2
    # "manual" or "scheduled".
    trigger: str
    dry_run: bool
    retention_hours: int
    started_at: str
    finished_at: str
    duration_ms: int
    projects: list[MaintenanceProjectResult] = Field(default_factory=list)
    registry_before: TableStorageStats | None = None
    registry_after: TableStorageStats | None = None
    registry_status: str = "skipped"
    registry_skip_reason: str | None = None
    registry_error: str | None = None
    registry_versions_removed: int = 0
    registry_bytes_reclaimed: int = 0
    versions_removed_total: int = 0
    bytes_reclaimed_total: int = 0
    reclaimable_bytes_estimate_total: int = 0
    skipped_projects: list[str] = Field(default_factory=list)
    busy_projects: list[str] = Field(default_factory=list)
    failed_projects: list[str] = Field(default_factory=list)
