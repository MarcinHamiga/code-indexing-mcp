/**
 * The slice of storage that reference resolution depends on.
 *
 * `reference_service.py` takes a concrete `LanceStore`. Here it takes this
 * interface instead, for two reasons. The migration order (§7) lands the
 * resolver in Phase 2 and LanceDB in Phase 3, so the dependency has to point at
 * something that exists yet; and every method below is a *narrowed*, pushed-down
 * query rather than a table scan, which is a contract worth writing down
 * explicitly -- the reason `listReferenceRecords` takes `recordKinds`, and the
 * reason declarations are fetched through three targeted lookups instead of
 * being pulled in with the reference rows, is that both were once full-table
 * materializations per page.
 *
 * Phase 3's `LanceStore` implements this; the suite drives it with an in-memory
 * store built from real extractor output, so the resolver is tested against the
 * rows the pipeline actually produces rather than against hand-written ones.
 *
 * Every method is async because LanceDB's JS bindings are: there is no
 * synchronous query API to hide behind an adapter, and pretending otherwise
 * would mean blocking the loop a stdio MCP server shares with everything else.
 */

import type { CodeChunk, IndexedChunk, ProjectInfo } from "./models.ts";

/**
 * The structural-row schema generation.
 *
 * Rows written under an earlier generation carry a since-discarded id scheme,
 * so every query pushes this into its `WHERE` clause rather than filtering
 * after materializing. It lives here rather than with the indexer because it is
 * a property of the stored rows, and both the writer and every reader need it.
 */
export const REFERENCE_SCHEMA_VERSION = 4;

/** One row of the structural index: a reference, a declaration, or a coverage marker. */
export interface ReferenceRecord {
  reference_id: string;
  record_kind: string;
  file_id: string;
  project_id: string;
  path: string;
  language: string;
  kind: string | null;
  source_qualified_symbol: string | null;
  written_name: string | null;
  target_name: string | null;
  module_path: string | null;
  imported_name: string | null;
  alias: string | null;
  receiver_text: string | null;
  start_byte: number | null;
  end_byte: number | null;
  start_line: number | null;
  end_line: number | null;
  shape_json: string | null;
  content_hash: string;
  schema_version: number;
}

export interface ReferenceStore {
  /**
   * Whether the references table exists for a project.
   *
   * Distinguishes a legitimately empty reference index (the table exists,
   * there is simply nothing to report) from one that was never built at all --
   * a partition indexed before the feature existed, or one whose
   * `ensureReferenceIndex` was skipped. Both collapse to `[]`/`0` through the
   * other methods, so a caller that needs the distinction must ask directly
   * rather than trust an empty result.
   */
  hasReferenceTable(projectId: string): Promise<boolean>;

  /** The current structural snapshot version, without creating a partition. */
  referenceVersion(projectId: string): Promise<number>;

  /** Structural rows from one immutable table version, filtered in the query. */
  listReferenceRecords(
    projectId: string,
    options: {
      version?: number;
      schemaVersion?: number;
      recordKinds?: readonly string[];
    },
  ): Promise<ReferenceRecord[]>;

  /** Declaration rows for exactly the given files. */
  declarationsForFiles(
    projectId: string,
    fileIds: Iterable<string>,
    options: { version?: number; schemaVersion?: number },
  ): Promise<ReferenceRecord[]>;

  /** Rows whose own `target_name` is the given name, optionally one record kind. */
  targetNameCandidates(
    projectId: string,
    targetName: string,
    options: { recordKind?: string; version?: number; schemaVersion?: number },
  ): Promise<ReferenceRecord[]>;

  /** Declaration rows for one exact qualified symbol. */
  declarationShapes(
    projectId: string,
    qualifiedSymbol: string,
    options: { version?: number; schemaVersion?: number },
  ): Promise<ReferenceRecord[]>;

  getChunk(chunkId: string): Promise<CodeChunk | null>;

  listChunks(projectIds: readonly string[]): Promise<IndexedChunk[]>;

  listProjects(): Promise<ProjectInfo[]>;
}
