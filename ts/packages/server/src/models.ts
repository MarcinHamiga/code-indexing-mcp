/**
 * Immutable domain models.
 *
 * Two conventions hold across this file and nowhere else in the package.
 *
 * **Field names stay snake_case.** These models *are* the wire contract: the
 * MCP tool schemas are generated from them, the `.ci-mcp/project.toml` marker
 * and the progress snapshot are written from them, and the daemon frames them.
 * Renaming to camelCase would mean a hand-written mapping for every one of the
 * fifty models here, which is precisely where a migration hides its bugs.
 * Everything outside this module -- locals, helpers, settings, adapters -- is
 * camelCase as usual.
 *
 * **A schema and its type share a name.** `ProjectInfo.parse(raw)` is the
 * constructor and `ProjectInfo` is the type, so the two halves read the way
 * `ProjectInfo.model_validate(raw)` and `ProjectInfo` did.
 *
 * Pydantic's `frozen=True` has no exact analogue that survives composition
 * (`.extend()` on a frozen schema is not expressible), so immutability is a
 * `tsc` property here rather than a runtime one: nothing mutates a parsed
 * model, and `progress.ts` -- the one place that used to need a mutable copy --
 * builds a new object instead.
 *
 * Paths are plain strings. Pydantic needed `_PathAsPlainString` to stop
 * `pathlib.Path` fields emitting a non-standard `"format": "path"` that made
 * strict MCP clients warn; a TypeScript path is already a string, so the
 * wrapper has nothing left to do and the emitted schema is right by default.
 */

import { z } from "zod";

export const LEGACY_DEFAULT_INCLUDES_V1: readonly string[] = [
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
];

export const LEGACY_DEFAULT_INCLUDES_V2: readonly string[] = [
  ...LEGACY_DEFAULT_INCLUDES_V1,
  "**/*.java",
];

export const LEGACY_DEFAULT_INCLUDES_V3: readonly string[] = [
  ...LEGACY_DEFAULT_INCLUDES_V2,
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
];

export const DEFAULT_INCLUDES: readonly string[] = [
  ...LEGACY_DEFAULT_INCLUDES_V3,
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
];

/**
 * The kinds the extractor emits, plus the `_part` variants it produces when a
 * definition is split across chunks. Closed so MCP clients get an enum instead
 * of a free-text field; extend both halves together when a query file gains a
 * capture.
 */
export const CHUNK_KINDS = [
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
] as const;

export const ChunkKind = z.enum(CHUNK_KINDS);
export type ChunkKind = z.infer<typeof ChunkKind>;

export const ReferenceKind = z.enum([
  "import",
  "export",
  "call",
  "type_use",
  "inheritance",
  "decorator",
  "read",
  "write",
]);
export type ReferenceKind = z.infer<typeof ReferenceKind>;

export const ParameterKind = z.enum([
  "positional_only",
  "positional",
  "keyword_only",
  "variadic",
  "keyword_variadic",
]);
export type ParameterKind = z.infer<typeof ParameterKind>;

export const ResolutionLevel = z.enum(["exact", "likely", "unresolved"]);
export type ResolutionLevel = z.infer<typeof ResolutionLevel>;

/**
 * Mirrors the scanner's language table. Kept here rather than imported from the
 * scanner so models stays free of scanner imports.
 */
export const LanguageName = z.enum([
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
]);
export type LanguageName = z.infer<typeof LanguageName>;

/**
 * Why an indexing run started. `schema-rebuild` and `maintenance` are reserved
 * for later releases; every current run is one of the other five.
 */
export const IndexTrigger = z.enum([
  "manual",
  "startup",
  "watcher",
  "lazy-query",
  "reference-backfill",
  "schema-rebuild",
  "maintenance",
]);
export type IndexTrigger = z.infer<typeof IndexTrigger>;

export const ScanConfig = z.object({
  include: z.array(z.string()).default(() => [...DEFAULT_INCLUDES]),
  exclude: z.array(z.string()).default(() => []),
  max_file_bytes: z.int().positive().default(1_048_576),
});
export type ScanConfig = z.infer<typeof ScanConfig>;

export const ProjectInfo = z.object({
  version: z.int().default(1),
  id: z.string(),
  name: z.string(),
  root: z.string(),
  scan: ScanConfig.default(() => ScanConfig.parse({})),
});
export type ProjectInfo = z.infer<typeof ProjectInfo>;

export const ScannedFile = z.object({
  path: z.string(),
  absolute_path: z.string(),
  language: z.string(),
  size: z.int(),
  // The one field that must be a bigint. Nanosecond mtimes are around 1.7e18,
  // which is two hundred times past `Number.MAX_SAFE_INTEGER`, so a `number`
  // would round the low digits away -- and since this value is what change
  // detection compares against the stored one, rounding it would make every
  // file in a migrated index look modified exactly once. `fs.stat` hands it
  // over as a bigint already; the MCP surface in Phase 6 owes it an explicit
  // serializer, because `JSON.stringify` refuses bigints outright.
  mtime_ns: z.bigint(),
  // Streaming scans attach changed-file bytes so the indexer consumes them
  // without a second read. Collected scans deliberately leave this null.
  content: z.instanceof(Uint8Array).nullable().default(null),
});
export type ScannedFile = z.infer<typeof ScannedFile>;

export const SkippedFile = z.object({
  path: z.string(),
  reason: z.string(),
  detail: z.string().nullable().default(null),
});
export type SkippedFile = z.infer<typeof SkippedFile>;

export const ScanResult = z.object({
  files: z.array(ScannedFile),
  skipped: z.array(SkippedFile),
});
export type ScanResult = z.infer<typeof ScanResult>;

/**
 * The skip reasons the scanner itself can attach to a file. Content-level
 * rejections (binary, encoding, parse, embedding) belong to the indexer and
 * never appear in a scan inspection.
 */
export const SCAN_SKIP_REASONS: ReadonlySet<string> = new Set([
  "unsupported",
  "ignored",
  "symlink",
  "oversized",
  "unreadable",
]);

/** One repository-relative scan outcome, without source contents. */
export const ScanInspectionItem = z.object({
  path: z.string(),
  outcome: z.enum(["eligible", "skipped"]),
  language: z.string().nullable().default(null),
  reason: z.string().nullable().default(null),
  detail: z.string().nullable().default(null),
  size: z.int().nullable().default(null),
  mtime_ns: z.bigint().nullable().default(null),
});
export type ScanInspectionItem = z.infer<typeof ScanInspectionItem>;

/**
 * One page of a dry-run scan inspection.
 *
 * The page is a stat-only view of what an index run would do: it never embeds,
 * mutates the index, or persists a complete manifest.
 */
export const ScanInspectionPage = z.object({
  schema_version: z.int().default(1),
  project: ProjectInfo.nullable().default(null),
  items: z.array(ScanInspectionItem).default(() => []),
  next_cursor: z.string().nullable().default(null),
});
export type ScanInspectionPage = z.infer<typeof ScanInspectionPage>;

export const ExtractedChunk = z.object({
  kind: z.string(),
  symbol: z.string().nullable().default(null),
  qualified_symbol: z.string().nullable().default(null),
  parent_symbol: z.string().nullable().default(null),
  start_byte: z.int(),
  end_byte: z.int(),
  start_line: z.int(),
  end_line: z.int(),
  content: z.string(),
  embedding_text: z.string(),
  search_text: z.string(),
  part_index: z.int().default(0),
  // The two halves `embedding_text` and `search_text` are composed from.
  // Keeping them lets a token window be recomposed with the same context header
  // and identifier tail as the whole chunk, instead of the header being windowed
  // away from every part after the first.
  embedding_prefix: z.string().default(""),
  search_suffix: z.string().default(""),
});
export type ExtractedChunk = z.infer<typeof ExtractedChunk>;

export const CallShape = z.object({
  positional_count: z.int().default(0),
  keywords: z.array(z.string()).default(() => []),
  has_positional_spread: z.boolean().default(false),
  has_keyword_spread: z.boolean().default(false),
  type_argument_count: z.int().nullable().default(null),
  constructor: z.boolean().default(false),
});
export type CallShape = z.infer<typeof CallShape>;

export const ParameterShape = z.object({
  name: z.string().describe("Parameter name."),
  kind: ParameterKind.describe(
    "One of positional_only, positional, keyword_only, variadic (*args-style), " +
      "or keyword_variadic (**kwargs-style).",
  ),
  required: z.boolean().describe("Whether this parameter has no default value."),
  position: z.int().describe("Zero-based position among this signature's parameters."),
  // True when this slot is a destructured pattern (`{ a, b }` / `[a, b]`)
  // collapsed to one positional parameter (E7). `name` is a synthesized,
  // non-authoritative label in that case -- signature comparisons that depend on
  // matching it by name must route to `review` instead of trusting it as a real
  // keyword/identifier.
  destructured: z
    .boolean()
    .default(false)
    .describe(
      "Whether this slot is a destructured pattern (e.g. `{ a, b }`) collapsed to one " +
        "positional parameter; if true, `name` is a synthesized label, not authoritative.",
    ),
});
export type ParameterShape = z.infer<typeof ParameterShape>;

export const ExtractedReference = z.object({
  kind: ReferenceKind,
  written_name: z.string(),
  target_name: z.string(),
  source_qualified_symbol: z.string().nullable().default(null),
  module_path: z.string().nullable().default(null),
  imported_name: z.string().nullable().default(null),
  alias: z.string().nullable().default(null),
  receiver_text: z.string().nullable().default(null),
  start_byte: z.int(),
  end_byte: z.int(),
  start_line: z.int(),
  end_line: z.int(),
  call_shape: CallShape.nullable().default(null),
});
export type ExtractedReference = z.infer<typeof ExtractedReference>;

export const ExtractedDeclarationShape = z.object({
  symbol: z.string(),
  qualified_symbol: z.string(),
  kind: z.string(),
  start_byte: z.int(),
  end_byte: z.int(),
  start_line: z.int(),
  end_line: z.int(),
  parameters: z.array(ParameterShape).default(() => []),
});
export type ExtractedDeclarationShape = z.infer<typeof ExtractedDeclarationShape>;

export const ExtractionResult = z.object({
  chunks: z.array(ExtractedChunk),
  references: z.array(ExtractedReference).default(() => []),
  declarations: z.array(ExtractedDeclarationShape).default(() => []),
  has_errors: z.boolean().default(false),
  // Wall time spent specifically on structural reference extraction, separate
  // from parsing and chunking, so a caller can report reference-extraction cost
  // without mislabeling the whole parse phase as it (T1). A duration, not a
  // timestamp, so it stays inside the safe-integer range and needs no bigint.
  reference_extraction_ns: z.int().default(0),
});
export type ExtractionResult = z.infer<typeof ExtractionResult>;

/** The structural extraction generation known for one indexed file. */
export const ReferenceCoverage = z.object({
  file_id: z.string(),
  path: z.string(),
  content_hash: z.string(),
  schema_version: z.int(),
});
export type ReferenceCoverage = z.infer<typeof ReferenceCoverage>;

/** Outcome of a parse-only structural-index catch-up run. */
export const ReferenceBackfillReport = z.object({
  project_id: z.string(),
  files_checked: z.int().default(0),
  files_backfilled: z.int().default(0),
  files_current: z.int().default(0),
  incomplete_paths: z.array(z.string()).default(() => []),
  stale_paths: z.array(z.string()).default(() => []),
});
export type ReferenceBackfillReport = z.infer<typeof ReferenceBackfillReport>;

/** Whether a backfill run left nothing behind. */
export function isBackfillComplete(report: ReferenceBackfillReport): boolean {
  return report.incomplete_paths.length === 0 && report.stale_paths.length === 0;
}

/** One declaration identity, by chunk id or its stable source location. */
export const DeclarationSelector = z
  .object({
    chunk_id: z
      .string()
      .nullable()
      .default(null)
      .describe(
        "Chunk id from a search_code or find_symbol hit. Cannot be combined with " +
          "project, path, or qualified_symbol -- provide this alone, or all three of them.",
      ),
    project: z
      .string()
      .nullable()
      .default(null)
      .describe("Project id, name, or path. Required with path and qualified_symbol."),
    path: z
      .string()
      .nullable()
      .default(null)
      .describe(
        "Repo-relative POSIX path to the file holding the declaration " +
          "(forward slashes, relative to the project root). Required with project and " +
          "qualified_symbol.",
      ),
    qualified_symbol: z
      .string()
      .nullable()
      .default(null)
      .describe(
        "Dotted qualified symbol name, e.g. 'Outer.method'. Required with project and path.",
      ),
  })
  .superRefine((selector, context) => {
    const byChunk = selector.chunk_id !== null;
    const location = [selector.project, selector.path, selector.qualified_symbol];
    const byLocation = location.every((value) => value !== null);
    const anyLocation = location.some((value) => value !== null);
    if (byChunk && anyLocation) {
      context.addIssue({
        code: "custom",
        message: "chunk_id cannot be combined with project, path, or qualified_symbol",
      });
      return;
    }
    if (!byChunk && !byLocation) {
      context.addIssue({
        code: "custom",
        message: "Provide exactly chunk_id or project, path, and qualified_symbol together",
      });
    }
  });
export type DeclarationSelector = z.infer<typeof DeclarationSelector>;

export const SelectedDeclaration = z.object({
  project_id: z.string(),
  file_id: z.string(),
  path: z.string(),
  language: z.string(),
  symbol: z.string(),
  qualified_symbol: z.string(),
  kind: z.string(),
  start_line: z.int(),
  end_line: z.int(),
  chunk_id: z.string().nullable().default(null),
});
export type SelectedDeclaration = z.infer<typeof SelectedDeclaration>;

export const ReferenceHit = z.object({
  reference_id: z.string(),
  project_id: z.string(),
  path: z.string(),
  language: z.string(),
  kind: ReferenceKind,
  start_line: z.int(),
  end_line: z.int(),
  start_byte: z.int(),
  end_byte: z.int(),
  snippet: z.string().default(""),
  written_name: z.string().nullable().default(null),
  resolution: ResolutionLevel,
  reason_code: z.string(),
  explanation: z.string(),
});
export type ReferenceHit = z.infer<typeof ReferenceHit>;

export const ReferenceLimitation = z.object({
  code: z.string(),
  explanation: z.string(),
  path: z.string().nullable().default(null),
});
export type ReferenceLimitation = z.infer<typeof ReferenceLimitation>;

export const ReferenceResponse = z.object({
  selected: SelectedDeclaration,
  hits: z.array(ReferenceHit).default(() => []),
  limitations: z.array(ReferenceLimitation).default(() => []),
  cursor: z.string().nullable().default(null),
  snapshot_version: z.int().default(0),
});
export type ReferenceResponse = z.infer<typeof ReferenceResponse>;

export const RenameOperation = z.object({
  kind: z.literal("rename").default("rename").describe("Discriminator; always 'rename'."),
  new_name: z.string().describe("The declaration's new name."),
});
export type RenameOperation = z.infer<typeof RenameOperation>;

export const SignatureChangeOperation = z.object({
  kind: z
    .literal("signature_change")
    .default("signature_change")
    .describe("Discriminator; always 'signature_change'."),
  parameters: z
    .array(ParameterShape)
    .describe("The declaration's proposed new full parameter list, in order."),
});
export type SignatureChangeOperation = z.infer<typeof SignatureChangeOperation>;

export const RefactorOperation = z.discriminatedUnion("kind", [
  RenameOperation,
  SignatureChangeOperation,
]);
export type RefactorOperation = z.infer<typeof RefactorOperation>;

export const RefactorFinding = ReferenceHit.extend({
  edit_required: z.boolean().default(false),
  // The identifier to rewrite, which is narrower than the occurrence range: the
  // call `auth.authorize(u)` spans `auth.authorize`, but only `authorize` may be
  // replaced. Null when the identifier could not be located uniquely, which
  // means the edit has to be made by hand.
  edit_start_byte: z.int().nullable().default(null),
  edit_end_byte: z.int().nullable().default(null),
});
export type RefactorFinding = z.infer<typeof RefactorFinding>;

export const CompletenessReport = z.object({
  state: z
    .enum(["complete", "complete_with_dynamic_limitations", "incomplete"])
    .default("complete"),
  explanation: z.string().default("All indexed structural candidates were considered."),
});
export type CompletenessReport = z.infer<typeof CompletenessReport>;

export const RefactorCounts = z.object({
  must_change: z.int().default(0),
  likely_change: z.int().default(0),
  review: z.int().default(0),
  evidence: z.int().default(0),
});
export type RefactorCounts = z.infer<typeof RefactorCounts>;

export const RefactorAnalysis = z.object({
  selected: SelectedDeclaration,
  operation: RefactorOperation,
  must_change: z.array(RefactorFinding).default(() => []),
  likely_change: z.array(RefactorFinding).default(() => []),
  review: z.array(RefactorFinding).default(() => []),
  evidence: z.array(RefactorFinding).default(() => []),
  limitations: z.array(ReferenceLimitation).default(() => []),
  counts: RefactorCounts.default(() => RefactorCounts.parse({})),
  cursor: z.string().nullable().default(null),
  completeness: CompletenessReport.default(() => CompletenessReport.parse({})),
});
export type RefactorAnalysis = z.infer<typeof RefactorAnalysis>;

/** Every finding, in the caller-facing priority order. */
export function refactorFindings(analysis: RefactorAnalysis): RefactorFinding[] {
  return [
    ...analysis.must_change,
    ...analysis.likely_change,
    ...analysis.review,
    ...analysis.evidence,
  ];
}

export const StoredFile = z.object({
  file_id: z.string(),
  project_id: z.string(),
  path: z.string(),
  language: z.string(),
  size: z.int(),
  mtime_ns: z.bigint(),
  content_hash: z.string(),
  has_errors: z.boolean().default(false),
  error: z.string().nullable().default(null),
  indexed_at: z.int(),
});
export type StoredFile = z.infer<typeof StoredFile>;

/**
 * A committed chunk without its embedding vector.
 *
 * Read paths that only need chunk text and offsets use this so a whole
 * project's 768-float vectors are not decoded for no consumer.
 *
 * Mirrors the chunk row: project_id is not stored on it because the owning
 * partition knows it. content_hash stays on the row so a chunk response is
 * always one coherent generation, even while a files-table update commits.
 */
export const IndexedChunk = z.object({
  chunk_id: z.string(),
  file_id: z.string(),
  path: z.string(),
  language: z.string(),
  kind: z.string(),
  symbol: z.string().nullable().default(null),
  qualified_symbol: z.string().nullable().default(null),
  parent_symbol: z.string().nullable().default(null),
  start_byte: z.int(),
  end_byte: z.int(),
  start_line: z.int(),
  end_line: z.int(),
  content: z.string(),
  identifier_terms: z.string(),
  part_index: z.int().default(0),
  content_hash: z.string().default(""),
});
export type IndexedChunk = z.infer<typeof IndexedChunk>;

/** A chunk as written to storage, vector included. */
export const StoredChunk = IndexedChunk.extend({
  vector: z.array(z.number()),
});
export type StoredChunk = z.infer<typeof StoredChunk>;

/** A query result that deliberately excludes embedding and index payloads. */
export const ChunkPreview = z.object({
  chunk_id: z.string(),
  project_id: z.string(),
  path: z.string(),
  language: z.string(),
  kind: z.string(),
  symbol: z.string().nullable().default(null),
  qualified_symbol: z.string().nullable().default(null),
  parent_symbol: z.string().nullable().default(null),
  start_line: z.int(),
  end_line: z.int(),
  content: z.string().default(""),
});
export type ChunkPreview = z.infer<typeof ChunkPreview>;

export const IndexIssue = z.object({
  path: z.string(),
  message: z.string(),
});
export type IndexIssue = z.infer<typeof IndexIssue>;

export const IndexReport = z.object({
  project_id: z.string(),
  discovered_files: z.int().default(0),
  indexed_files: z.int().default(0),
  parsed_files: z.int().default(0),
  embedded_chunks: z.int().default(0),
  unchanged_files: z.int().default(0),
  metadata_only_files: z.int().default(0),
  removed_files: z.int().default(0),
  skipped_files: z.int().default(0),
  errors: z.array(IndexIssue).default(() => []),
  duration_ms: z.int().default(0),
  // Phase breakdown of duration_ms. They do not sum to it: lock acquisition and
  // project bookkeeping sit outside, and embedding includes the one-time worker
  // spawn and model load. Optional so older clients keep validating reports.
  scan_duration_ms: z.int().nullable().default(null),
  parse_duration_ms: z.int().nullable().default(null),
  embed_duration_ms: z.int().nullable().default(null),
  commit_duration_ms: z.int().nullable().default(null),
  // Stable Phase 1 telemetry names. Keep the *_duration_ms fields above for
  // clients that adopted the earlier memory-hardening report.
  embedding_backend: z.string().default("cpu"),
  embedding_batch_size: z.int().default(1),
  // Set when a run started on an accelerator and finished somewhere else.
  embedding_fallback_reason: z.string().nullable().default(null),
  scan_ms: z.int().nullable().default(null),
  parse_ms: z.int().nullable().default(null),
  embed_ms: z.int().nullable().default(null),
  commit_ms: z.int().nullable().default(null),
  fallback_count: z.int().default(0),
  memory_budget_bytes: z.int().nullable().default(null),
  peak_memory_bytes: z.int().nullable().default(null),
  worker_used: z.boolean().default(false),
  // Token-window telemetry, populated only on worker runs. embedded_segments
  // counts what the worker embedded, which includes segments from files that
  // later failed and were not committed, so it can exceed embedded_chunks.
  // token_windowing=false means no tokenizer was reachable and sequence length
  // went unbounded.
  embedded_segments: z.int().nullable().default(null),
  embedded_tokens: z.int().nullable().default(null),
  embedding_retries: z.int().nullable().default(null),
  worker_termination_reason: z.string().nullable().default(null),
  token_windowing: z.boolean().nullable().default(null),
  // Why this run embedded where it did, when the workload crossover was what
  // decided it -- so a run that stayed on CPU because it was small is
  // distinguishable from one that fell back to CPU because something broke.
  // embedded_characters is what the decision was measured against.
  embedded_characters: z.int().nullable().default(null),
  embedding_crossover_characters: z.int().nullable().default(null),
  embedding_selection_reason: z.string().nullable().default(null),
  // T1: reference extraction's own timing and this run's own staged row count,
  // distinct from `parse_duration_ms` (parsing + chunking + reference extraction
  // together) and from a whole-project table read.
  reference_extraction_duration_ms: z.int().nullable().default(null),
  staged_reference_rows: z.int().default(0),
  // Durable run identity and why it ran. Optional so older clients keep
  // validating reports and older stored reports keep validating.
  run_id: z.string().nullable().default(null),
  trigger: IndexTrigger.default("manual"),
  // Counter contract for the audit record: skip reasons are broken out by
  // cause, failed files are parse/embedding failures, and the byte/chunk
  // counters track what the run actually read, extracted, and staged.
  failed_files: z.int().default(0),
  skip_reasons: z.record(z.string(), z.int()).default(() => ({})),
  skipped_samples: z.array(z.string()).default(() => []),
  bytes_read: z.int().default(0),
  chunks_extracted: z.int().default(0),
  chunks_staged: z.int().default(0),
  staged_bytes: z.int().default(0),
});
export type IndexReport = z.infer<typeof IndexReport>;

/**
 * What the embedding stack resolved to on this machine, and why.
 *
 * `embedding_model` rather than `model_id` because pydantic reserved the
 * `model_` field prefix; the name is part of the wire contract now, so it stays
 * even though zod has no such rule.
 */
export const ModelStatus = z.object({
  embedding_model: z.string(),
  dimension: z.int(),
  requested_accelerator: z.string(),
  resolved_accelerator: z.string(),
  device: z.string(),
  execution_provider: z.string(),
  available_providers: z.array(z.string()),
  stability: z.string(),
  precision: z.string(),
  runtime_version: z.string(),
  batch_size: z.int(),
  // "explicit" when configured, "measured" when calibration settled on it,
  // "reduced" when a memory-ceiling overrun forced it down from what was
  // measured, "default" when none of those applied.
  batch_calibration: z.string(),
  // "hit" or "miss" against the local probe cache; "not-applicable" on CPU,
  // which needs no probe to be trusted.
  probe_cache_state: z.string(),
  strict: z.boolean(),
  // The GPU driver the accelerator environment was probed against, when one was
  // prepared. Empty on CPU, where no driver is in the picture.
  driver_version: z.string().default(""),
  // The interpreter passage embedding runs in when that is not this one, and
  // the accelerator the installer prepared, whether or not it was selected.
  accelerator_environment: z.string().nullable().default(null),
  accelerator_prepared: z.string().nullable().default(null),
  fallback_reason: z.string().nullable().default(null),
  // What calibration measured on this machine, and the run size above which
  // starting the accelerator repays its model load. Null means unmeasured -- or,
  // for the crossover, that the accelerator never overtakes CPU at any size,
  // which is a different statement from "the threshold is large".
  cpu_characters_per_second: z.number().nullable().default(null),
  accelerator_characters_per_second: z.number().nullable().default(null),
  accelerator_load_ms: z.int().nullable().default(null),
  crossover_characters: z.int().nullable().default(null),
  // The one setting change these numbers actually argue for, when they argue for
  // one. Not advice in general -- only what the measurements support.
  recommended_override: z.string().nullable().default(null),
});
export type ModelStatus = z.infer<typeof ModelStatus>;

export const SearchHit = z.object({
  chunk_id: z.string(),
  project_id: z.string(),
  project_name: z.string(),
  path: z.string(),
  language: z.string(),
  kind: z.string(),
  symbol: z.string().nullable().default(null),
  qualified_symbol: z.string().nullable().default(null),
  start_line: z.int(),
  end_line: z.int(),
  score: z.number(),
  snippet: z.string(),
  truncated: z.boolean().default(false),
});
export type SearchHit = z.infer<typeof SearchHit>;

export const SearchResponse = z.object({
  query: z.string(),
  hits: z.array(SearchHit),
});
export type SearchResponse = z.infer<typeof SearchResponse>;

export const SymbolResponse = z.object({
  name: z.string(),
  hits: z.array(SearchHit),
});
export type SymbolResponse = z.infer<typeof SymbolResponse>;

export const OutlineItem = z.object({
  kind: z.string(),
  symbol: z.string(),
  qualified_symbol: z.string(),
  parent_symbol: z.string().nullable().default(null),
  start_line: z.int(),
  end_line: z.int(),
});
export type OutlineItem = z.infer<typeof OutlineItem>;

export const OutlineResponse = z.object({
  project_id: z.string(),
  path: z.string(),
  items: z.array(OutlineItem),
});
export type OutlineResponse = z.infer<typeof OutlineResponse>;

/**
 * One indexed chunk as returned to a caller.
 *
 * Deliberately not an extension of StoredChunk. Inheriting the storage row
 * shipped the 768-dimension vector and both derived text columns to MCP
 * clients: 72% of the response was the vector, and the code arrived three times
 * over as content, embedding_text, and search_text. Adding a storage column must
 * not silently widen this payload, so the fields are listed rather than
 * inherited. Chunk rows no longer persist project_id or content_hash either:
 * both are injected by the read path from the owning partition and the files
 * table.
 */
export const CodeChunk = z.object({
  chunk_id: z.string(),
  file_id: z.string(),
  project_id: z.string(),
  path: z.string(),
  language: z.string(),
  kind: z.string(),
  symbol: z.string().nullable().default(null),
  qualified_symbol: z.string().nullable().default(null),
  parent_symbol: z.string().nullable().default(null),
  start_byte: z.int(),
  end_byte: z.int(),
  start_line: z.int(),
  end_line: z.int(),
  content: z.string(),
  content_hash: z.string(),
  part_index: z.int().default(0),
});
export type CodeChunk = z.infer<typeof CodeChunk>;

/**
 * One durable audit record of an indexing or backfill run.
 *
 * Values are bounded before they are written: at most `MAX_ERROR_SAMPLES` error
 * details and `MAX_SKIPPED_SAMPLES` skipped-path samples are kept, so a
 * pathological run cannot grow the history database without bound.
 */
export const RunAudit = z.object({
  run_id: z.string(),
  project_id: z.string(),
  trigger: IndexTrigger,
  server_version: z.string().default(""),
  git_revision: z.string().nullable().default(null),
  model_id: z.string().default(""),
  schema_version: z.int().default(0),
  scan_config_hash: z.string().default(""),
  force: z.boolean().default(false),
  // Why a schema-rebuild run replaced its partition (model or schema mismatch,
  // for example). Null for ordinary runs.
  rebuild_reason: z.string().nullable().default(null),
  // Owning process, so startup can tell a crashed run from one another live
  // process is still executing.
  pid: z.int().default(() => process.pid),
  started_at: z.string().default(""),
  finished_at: z.string().nullable().default(null),
  // running, completed, failed, or interrupted.
  state: z.string().default("running"),
  phase_durations: z.record(z.string(), z.int()).default(() => ({})),
  eligible_files: z.int().default(0),
  changed_files: z.int().default(0),
  unchanged_files: z.int().default(0),
  parsed_files: z.int().default(0),
  failed_files: z.int().default(0),
  removed_files: z.int().default(0),
  skipped_total: z.int().default(0),
  chunks_extracted: z.int().default(0),
  chunks_embedded: z.int().default(0),
  chunks_staged: z.int().default(0),
  staged_bytes: z.int().default(0),
  bytes_read: z.int().default(0),
  skip_reasons: z.record(z.string(), z.int()).default(() => ({})),
  errors: z.array(IndexIssue).default(() => []),
  skipped_samples: z.array(z.string()).default(() => []),
  embedding_backend: z.string().default("cpu"),
  embedding_fallback_reason: z.string().nullable().default(null),
  worker_used: z.boolean().default(false),
  // Best-effort storage table versions around the run, not full partition
  // traversals: audit recording must stay inexpensive on any repository.
  storage_before: z.record(z.string(), z.int()).default(() => ({})),
  storage_after: z.record(z.string(), z.int()).default(() => ({})),
});
export type RunAudit = z.infer<typeof RunAudit>;

/** Compact summary of one completed run for status surfaces. */
export const RunSummary = z.object({
  run_id: z.string(),
  trigger: IndexTrigger,
  state: z.string(),
  started_at: z.string(),
  finished_at: z.string().nullable().default(null),
  duration_ms: z.int().default(0),
  eligible_files: z.int().default(0),
  changed_files: z.int().default(0),
  failed_files: z.int().default(0),
  skipped_total: z.int().default(0),
  chunks_embedded: z.int().default(0),
});
export type RunSummary = z.infer<typeof RunSummary>;

/** One page of a project's indexing history. */
export const HistoryPage = z.object({
  schema_version: z.int().default(1),
  project: ProjectInfo.nullable().default(null),
  runs: z.array(RunAudit).default(() => []),
  next_cursor: z.string().nullable().default(null),
});
export type HistoryPage = z.infer<typeof HistoryPage>;

/**
 * A point-in-time snapshot of one project's indexing run.
 *
 * Every counter's name matches what it counts: `candidates_*` cover every path
 * the scanner examined, `eligible_files` the files that passed the scan, and the
 * `*_files` counters the eligible ones. Totals stay unset while the scanner
 * streams and the run genuinely does not know them, and a candidate count is
 * never compared with an eligible-file total.
 *
 * Defined here rather than in progress.ts so progress.ts stays a plain consumer
 * of this module: it used to be the other way around, and the resulting
 * models/progress cycle made the package import-order dependent.
 */
export const IndexProgress = z.object({
  project_id: z.string(),
  run_id: z.string().default(""),
  trigger: IndexTrigger.default("manual"),
  phase: z.string().default("scanning"),
  // Every path the scanner examined, whether it became eligible or was skipped.
  // A first index has no honest candidates_total: the scanner streams, so the
  // total is only known once the walk has finished.
  candidates_seen: z.int().default(0),
  candidates_total: z.int().nullable().default(null),
  eligible_files: z.int().default(0),
  unchanged_files: z.int().default(0),
  changed_files: z.int().default(0),
  parsed_files: z.int().default(0),
  failed_files: z.int().default(0),
  skipped_total: z.int().default(0),
  skipped_by_reason: z.record(z.string(), z.int()).default(() => ({})),
  bytes_read: z.int().default(0),
  chunks_extracted: z.int().default(0),
  chunks_embedded: z.int().default(0),
  chunks_staged: z.int().default(0),
  staged_bytes: z.int().default(0),
  current_path: z.string().nullable().default(null),
  started_at: z.number().default(0),
  updated_at: z.number().default(0),
  // Monotonic anchor for phase durations, never a wall-clock timestamp: it must
  // strictly advance across phase changes on every platform.
  phase_started_at: z.number().default(0),
  pid: z.int().default(() => process.pid),
});
export type IndexProgress = z.infer<typeof IndexProgress>;

/**
 * Completion in `[0, 1]`, or null when the total is unknown.
 *
 * Only candidate counts may feed this: comparing candidates seen with an
 * eligible-file total would overstate (or understate) progress whenever the
 * repository contains skipped paths.
 */
export function progressFraction(progress: IndexProgress): number | null {
  if (!progress.candidates_total) return null;
  return Math.min(1, progress.candidates_seen / progress.candidates_total);
}

/** Render a one-line status suitable for a progress bar or a log line. */
export function describeProgress(progress: IndexProgress): string {
  if (progress.phase === "committing") return "Committing the index";
  if (progress.phase === "extracting_references") return "Extracting structural references";
  if (!progress.candidates_seen) return "Scanning for changed files";
  const scanned = progress.candidates_total
    ? `${progress.candidates_seen}/~${progress.candidates_total} candidates`
    : `${progress.candidates_seen} candidates`;
  const parts = [`${capitalize(progress.phase)} ${scanned}`];
  if (progress.eligible_files) parts.push(`${progress.eligible_files} eligible`);
  if (progress.changed_files) parts.push(`${progress.changed_files} changed`);
  if (progress.unchanged_files) parts.push(`${progress.unchanged_files} unchanged`);
  if (progress.failed_files) parts.push(`${progress.failed_files} failed`);
  if (progress.skipped_total) parts.push(`${progress.skipped_total} skipped`);
  if (progress.chunks_embedded) parts.push(`${progress.chunks_embedded} chunks embedded`);
  return parts.join(", ");
}

/** `str.capitalize()`: the first character upper, every other one lower. */
function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1).toLowerCase();
}

export const ProjectStatus = z.object({
  project: ProjectInfo,
  state: z.string(),
  file_count: z.int(),
  chunk_count: z.int(),
  // Live progress of whichever process is indexing this project right now, and
  // the compact summary of its most recent completed run. Never the full
  // history: status stays inexpensive.
  progress: IndexProgress.nullable().default(null),
  last_run: RunSummary.nullable().default(null),
});
export type ProjectStatus = z.infer<typeof ProjectStatus>;

export const RemovalReport = z.object({
  project_id: z.string(),
  removed: z.boolean(),
});
export type RemovalReport = z.infer<typeof RemovalReport>;

/** Fragment-size distribution as reported by Lance's table statistics. */
export const FragmentLengthStats = z.object({
  min: z.int().nullable().default(null),
  max: z.int().nullable().default(null),
  mean: z.number().nullable().default(null),
  p25: z.int().nullable().default(null),
  p50: z.int().nullable().default(null),
  p75: z.int().nullable().default(null),
  p99: z.int().nullable().default(null),
});
export type FragmentLengthStats = z.infer<typeof FragmentLengthStats>;

export const FragmentStats = z.object({
  num_fragments: z.int().default(0),
  num_small_fragments: z.int().default(0),
  lengths: FragmentLengthStats.nullable().default(null),
});
export type FragmentStats = z.infer<typeof FragmentStats>;

export const IndexStorageStats = z.object({
  name: z.string(),
  index_type: z.string(),
  columns: z.array(z.string()),
  indexed_rows: z.int().default(0),
  unindexed_rows: z.int().default(0),
  size_bytes: z.int().default(0),
});
export type IndexStorageStats = z.infer<typeof IndexStorageStats>;

/** One Lance table's storage snapshot, collected read-only. */
export const TableStorageStats = z.object({
  name: z.string(),
  current_version: z.int().default(0),
  row_count: z.int().default(0),
  // Lance-reported logical bytes for the table's live data.
  logical_bytes: z.int().default(0),
  // Filesystem-reported physical bytes, measured without following symlinks.
  physical_bytes: z.int().default(0),
  fragment_stats: FragmentStats.default(() => FragmentStats.parse({})),
  retained_version_count: z.int().default(0),
  // ISO-8601 timestamps of the oldest and newest retained versions.
  oldest_version_at: z.string().nullable().default(null),
  newest_version_at: z.string().nullable().default(null),
  indexes: z.array(IndexStorageStats).default(() => []),
});
export type TableStorageStats = z.infer<typeof TableStorageStats>;

/** One project partition's storage snapshot. */
export const ProjectStorageStats = z.object({
  project: ProjectInfo,
  snapshot_at: z.string(),
  tables: z.array(TableStorageStats).default(() => []),
  // Sum of the partition's table directories on disk.
  partition_physical_bytes: z.int().default(0),
  // False when a table version changed while the snapshot was collected, or when
  // the partition exists but its tables could not be opened.
  consistent: z.boolean().default(true),
  // True when the partition directory exists but its tables could not be opened
  // (a damaged or mid-mutation store), so no table statistics exist.
  partition_open_failed: z.boolean().default(false),
});
export type ProjectStorageStats = z.infer<typeof ProjectStorageStats>;

/** Installation-wide read-only storage statistics. */
export const StorageStatus = z.object({
  schema_version: z.int().default(1),
  snapshot_at: z.string(),
  registry: TableStorageStats,
  projects: z.array(ProjectStorageStats).default(() => []),
  physical_bytes_total: z.int().default(0),
  consistent: z.boolean().default(true),
  overlap_warnings: z.array(z.string()).default(() => []),
  worktree_warnings: z.array(z.string()).default(() => []),
});
export type StorageStatus = z.infer<typeof StorageStatus>;

/**
 * One project partition's outcome of a maintenance pass.
 *
 * `status` is one of `ok`, `skipped`, or `error`. A skipped project carries a
 * `skip_reason` such as `busy`, `not-indexed`, `dry-run`, or
 * `recovery-pending`. `before` and `after` hold full storage snapshots when they
 * could be collected, so the deltas stay auditable rather than being reduced to
 * counters.
 */
export const MaintenanceProjectResult = z.object({
  project: ProjectInfo,
  status: z.string().default("skipped"),
  skip_reason: z.string().nullable().default(null),
  error: z.string().nullable().default(null),
  before: ProjectStorageStats.nullable().default(null),
  after: ProjectStorageStats.nullable().default(null),
  // Retained versions removed and physical bytes reclaimed by the pass.
  versions_removed: z.int().default(0),
  bytes_reclaimed: z.int().default(0),
  // Pre-cleanup estimate of reclaimable bytes (physical minus logical): an
  // estimate, never a claim about what cleanup will actually free.
  reclaimable_bytes_estimate: z.int().default(0),
});
export type MaintenanceProjectResult = z.infer<typeof MaintenanceProjectResult>;

/** Outcome of one storage maintenance pass, manual or scheduled. */
export const MaintenanceReport = z.object({
  schema_version: z.int().default(1),
  // "manual" or "scheduled".
  trigger: z.string(),
  dry_run: z.boolean(),
  retention_hours: z.int(),
  started_at: z.string(),
  finished_at: z.string(),
  duration_ms: z.int(),
  projects: z.array(MaintenanceProjectResult).default(() => []),
  registry_before: TableStorageStats.nullable().default(null),
  registry_after: TableStorageStats.nullable().default(null),
  registry_status: z.string().default("skipped"),
  registry_skip_reason: z.string().nullable().default(null),
  registry_error: z.string().nullable().default(null),
  registry_versions_removed: z.int().default(0),
  registry_bytes_reclaimed: z.int().default(0),
  versions_removed_total: z.int().default(0),
  bytes_reclaimed_total: z.int().default(0),
  reclaimable_bytes_estimate_total: z.int().default(0),
  skipped_projects: z.array(z.string()).default(() => []),
  busy_projects: z.array(z.string()).default(() => []),
  failed_projects: z.array(z.string()).default(() => []),
});
export type MaintenanceReport = z.infer<typeof MaintenanceReport>;
