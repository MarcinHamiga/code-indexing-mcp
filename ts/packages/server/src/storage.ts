/** Partitioned LanceDB persistence compatible with the Python store's layout. */

import { execFileSync } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import type { Dirent } from "node:fs";
import path from "node:path";
import {
  connect,
  Index,
  MultiMatchQuery,
  Operator,
  rerankers,
  type Connection,
  type Table,
} from "@lancedb/lancedb";
import {
  Bool,
  type DataType,
  Field,
  FixedSizeList,
  Float16,
  Float32,
  Int32,
  Int64,
  Schema,
  Utf8,
} from "apache-arrow";
import {
  ChunkPreview,
  CodeChunk,
  IndexedChunk,
  ProjectInfo,
  type ProjectStorageStats,
  StoredFile,
  type StorageStatus,
  type StoredChunk,
  type TableStorageStats,
} from "./models.ts";
import {
  ReferenceSnapshotExpiredError,
  type ReferenceRecord,
  type ReferenceStore,
} from "./reference-store.ts";
import { CodeIndexingError } from "./errors.ts";
import { isRelativeTo, resolvePath } from "./paths.ts";
import { existingMarkerPath, rootedUnder, sameProjectRoot } from "./projects.ts";

export const SCHEMA_VERSION = 5;
export const MAX_CACHED_PARTITIONS = 16;
const SEARCH_CONCURRENCY = 8;
const GIT_TIMEOUT_MS = 5000;
const MISSING_REFERENCE_TABLE = "Reference table is missing from an interrupted transaction";

const CHUNK_COLUMNS = [
  "chunk_id",
  "file_id",
  "path",
  "language",
  "kind",
  "symbol",
  "qualified_symbol",
  "parent_symbol",
  "start_byte",
  "end_byte",
  "start_line",
  "end_line",
  "content",
  "identifier_terms",
  "content_hash",
  "part_index",
] as const;

const CHUNK_PAYLOAD_COLUMNS = CHUNK_COLUMNS.filter((column) => column !== "identifier_terms");

export interface TableVersions {
  files: number;
  chunks: number;
  references: number;
}

export interface StorageOptions {
  vectorDimension?: number;
  vectorIndex?: "exact" | "hnsw";
  vectorStorage?: "float16" | "float32";
}

export interface ReplacementBatch<T> {
  fileIds: readonly string[];
  rows: readonly T[];
}

export type GitRunner = (command: readonly string[], cwd: string) => string | null;

interface ProjectTables {
  files: Table;
  chunks: Table;
  references: Table | null;
  generation: number;
}

type StoredProject = {
  id: string;
  name: string;
  root: string;
  payload: string;
  model_id: string;
  vector_dimension: number;
  schema_version: number;
  state: string;
  updated_at: bigint;
};

/**
 * The asynchronous LanceDB backing store.
 *
 * The constructor is intentionally usable immediately: initialization is held by
 * an internal promise so callers do not need a separate connection lifecycle.
 */
export class LanceStore implements ReferenceStore {
  readonly directory: string;
  readonly vectorDimension: number;
  readonly vectorIndex: "exact" | "hnsw";
  readonly vectorStorage: "float16" | "float32";

  #registry: Promise<{ connection: Connection; projects: Table }>;
  #partitions = new Map<string, ProjectTables>();
  #locks = new Map<string, Promise<unknown>>();

  constructor(directory: string, options: StorageOptions = {}) {
    const storage = options.vectorStorage ?? "float16";
    if (storage !== "float16" && storage !== "float32") {
      throw new Error(`vector_storage must be float32 or float16, got '${String(storage)}'`);
    }
    this.directory = directory;
    this.vectorDimension = options.vectorDimension ?? 768;
    this.vectorIndex = options.vectorIndex ?? "exact";
    this.vectorStorage = storage;
    this.#registry = this.#openRegistry();
  }

  static projectArrowSchema(): Schema {
    return schema([
      field("id"),
      field("name"),
      field("root"),
      field("payload"),
      field("model_id"),
      field("vector_dimension", new Int32()),
      field("schema_version", new Int32()),
      field("state"),
      field("updated_at", new Int64()),
    ]);
  }

  static fileArrowSchema(): Schema {
    return schema([
      field("file_id"),
      field("project_id"),
      field("path"),
      field("language"),
      field("size", new Int64()),
      field("mtime_ns", new Int64()),
      field("content_hash"),
      field("has_errors", new Bool()),
      field("error", new Utf8(), true),
      field("indexed_at", new Int64()),
    ]);
  }

  static chunkArrowSchema(
    vectorDimension: number,
    vectorStorage: "float16" | "float32" = "float16",
  ): Schema {
    return schema([
      field("chunk_id"),
      field("file_id"),
      field("path"),
      field("language"),
      field("kind"),
      field("symbol", new Utf8(), true),
      field("qualified_symbol", new Utf8(), true),
      field("parent_symbol", new Utf8(), true),
      field("start_byte", new Int64()),
      field("end_byte", new Int64()),
      field("start_line", new Int32()),
      field("end_line", new Int32()),
      field("content"),
      field("identifier_terms"),
      field("content_hash"),
      field("part_index", new Int32()),
      field(
        "vector",
        new FixedSizeList(
          vectorDimension,
          new Field("item", vectorStorage === "float16" ? new Float16() : new Float32()),
        ),
      ),
    ]);
  }

  static referenceArrowSchema(): Schema {
    return schema([
      field("reference_id"),
      field("record_kind"),
      field("file_id"),
      field("project_id"),
      field("path"),
      field("language"),
      field("kind", new Utf8(), true),
      field("source_qualified_symbol", new Utf8(), true),
      field("written_name", new Utf8(), true),
      field("target_name", new Utf8(), true),
      field("module_path", new Utf8(), true),
      field("imported_name", new Utf8(), true),
      field("alias", new Utf8(), true),
      field("receiver_text", new Utf8(), true),
      field("start_byte", new Int64(), true),
      field("end_byte", new Int64(), true),
      field("start_line", new Int32(), true),
      field("end_line", new Int32(), true),
      field("shape_json", new Utf8(), true),
      field("content_hash"),
      field("schema_version", new Int32()),
    ]);
  }

  async upsertProject(
    project: ProjectInfo,
    options: { modelId: string; state?: string },
  ): Promise<void> {
    const projects = await this.#projects();
    const existing = (await rows<StoredProject>(projects, equals("id", project.id)))[0];
    if (existing !== undefined) {
      const registeredRoot = resolvePath(existing.root);
      const incomingRoot = resolvePath(project.root);
      if (
        !sameProjectRoot(registeredRoot, incomingRoot) &&
        existingMarkerPath(registeredRoot) !== null
      ) {
        throw new CodeIndexingError(
          "PROJECT_ID_CONFLICT",
          "The project ID is already active at another path",
          {
            project: project.id,
            registered_root: registeredRoot,
            incoming_root: incomingRoot,
          },
        );
      }
      if (sameProjectRoot(registeredRoot, incomingRoot)) {
        project = { ...project, root: registeredRoot };
      }
      if (
        existing.state !== "pending" &&
        (await this.incompatibilityReason(project.id, options.modelId)) !== null
      ) {
        // The live partition still describes the old generation. Preserve its
        // model/schema/dimension until the rebuild deletes it and re-stamps it.
        await this.markRebuildRequired(project.id);
        return;
      }
    }
    const state = options.state ?? "ready";
    const record: StoredProject = {
      id: project.id,
      name: project.name,
      root: resolvePath(project.root),
      payload: JSON.stringify(project),
      model_id: options.modelId,
      vector_dimension: this.vectorDimension,
      schema_version: SCHEMA_VERSION,
      state,
      updated_at: BigInt(Date.now()) * 1_000_000n,
    };
    if (existing !== undefined && sameProject(existing, record)) return;
    await merge(projects, "id", [record]);
  }

  async incompatibilityReason(projectId: string, modelId: string): Promise<string | null> {
    const projects = await this.#projects();
    const current = (await rows<StoredProject>(projects, equals("id", projectId)))[0];
    if (current === undefined || current.state === "pending") return null;
    const reasons: string[] = [];
    if (current.model_id !== modelId)
      reasons.push(
        `embedding model ${JSON.stringify(current.model_id)} -> ${JSON.stringify(modelId)}`,
      );
    if (current.vector_dimension !== this.vectorDimension)
      reasons.push(`vector dimension ${current.vector_dimension} -> ${this.vectorDimension}`);
    if (current.schema_version !== SCHEMA_VERSION)
      reasons.push(`index schema version ${current.schema_version} -> ${SCHEMA_VERSION}`);
    const tables = await this.#existingTables(projectId);
    if (tables !== null) {
      const vector = (await tables.chunks.schema()).fields.find((item) => item.name === "vector");
      const expected = this.vectorStorage === "float16" ? "Float16" : "Float32";
      if (vector?.type.children[0]?.type.constructor.name !== expected) {
        reasons.push(
          `vector storage ${vector?.type.children[0]?.type.constructor.name ?? "missing"} -> ${expected}`,
        );
      }
    }
    return reasons.length === 0 ? null : reasons.join("; ");
  }

  async markRebuildRequired(projectId: string): Promise<boolean> {
    return this.markProjectState(projectId, "rebuild_required");
  }

  async deletePartition(projectId: string, modelId: string): Promise<boolean> {
    return this.withPartitionAccess(projectId, async () => {
      await this.#advancePartitionGeneration(projectId);
      const projects = await this.#projects();
      const record = (await rows<StoredProject>(projects, equals("id", projectId)))[0];
      if (record === undefined) return false;
      this.#closePartition(projectId);
      await fs.rm(this.#partitionPath(projectId), { recursive: true, force: true });
      await merge(projects, "id", [
        {
          ...record,
          model_id: modelId,
          vector_dimension: this.vectorDimension,
          schema_version: SCHEMA_VERSION,
          state: "indexing",
          updated_at: BigInt(Date.now()) * 1_000_000n,
        },
      ]);
      return true;
    });
  }

  async listProjects(): Promise<ProjectInfo[]> {
    const projects = await this.#projects();
    return (await rows<StoredProject>(projects))
      .map((row) => ProjectInfo.parse(JSON.parse(row.payload)))
      .sort((left, right) => left.name.localeCompare(right.name));
  }

  async projectState(projectId: string): Promise<string> {
    const projects = await this.#projects();
    const state = (await rows<StoredProject>(projects, equals("id", projectId)))[0]?.state;
    if (state === undefined) {
      throw new CodeIndexingError("PROJECT_NOT_FOUND", `Unknown project: ${projectId}`);
    }
    return state;
  }

  async listFiles(projectId: string): Promise<StoredFile[]> {
    const tables = await this.#existingTables(projectId);
    if (tables === null) return [];
    return (await rows<Record<string, unknown>>(tables.files)).map(parseStoredFile);
  }

  async hasFileErrors(projectId: string): Promise<boolean> {
    const tables = await this.#existingTables(projectId);
    if (tables === null) return false;
    const failures = await rows<{ error: string | null }>(tables.files, "has_errors = true");
    return failures.some((row) => !(row.error ?? "").startsWith("rejected:"));
  }

  async upsertFile(record: StoredFile): Promise<void> {
    const tables = await this.#tables(record.project_id);
    await merge(tables.files, "file_id", [record]);
  }

  async replaceFile(
    record: StoredFile,
    chunks: readonly StoredChunk[],
    references: readonly ReferenceRecord[] = [],
  ): Promise<void> {
    await this.replaceFilesFromArrow(record.project_id, {
      files: [record],
      chunkBatches: [{ fileIds: [record.file_id], rows: chunks }],
      referenceBatches: [{ fileIds: [record.file_id], rows: references }],
    });
  }

  async replaceFilesFromArrow(
    projectId: string,
    input: {
      files: readonly StoredFile[];
      chunkBatches: Iterable<ReplacementBatch<StoredChunk>>;
      referenceBatches?: Iterable<ReplacementBatch<ReferenceRecord>>;
      removedFileIds?: Iterable<string>;
    },
  ): Promise<void> {
    if (!(await batchedMergeSemanticsOk())) {
      throw new CodeIndexingError(
        "UNSUPPORTED_RUNTIME",
        "The installed lancedb version does not filter " +
          "when_not_matched_by_source_delete rows the way batched commits " +
          "require; refusing to commit because it could delete rows of " +
          "untouched files. Upgrade lancedb and retry.",
      );
    }
    const tables = await this.#tables(projectId);
    const replaced = new Set<string>();
    for (const batch of input.chunkBatches) {
      batch.fileIds.forEach((id) => {
        replaced.add(id);
      });
      await replaceRows(tables.chunks, "chunk_id", batch.fileIds, batch.rows, (row) => ({
        ...row,
        content_hash: row.content_hash || fileHash(input.files, row.file_id),
      }));
    }
    for (const batch of input.referenceBatches ?? []) {
      batch.fileIds.forEach((id) => {
        replaced.add(id);
      });
      if (tables.references === null) throw new Error(MISSING_REFERENCE_TABLE);
      await replaceRows(tables.references, "reference_id", batch.fileIds, batch.rows);
    }
    if (input.files.length > 0) await merge(tables.files, "file_id", input.files);
    const removed = [...new Set(input.removedFileIds ?? [])].filter((id) => !replaced.has(id));
    if (removed.length > 0) {
      const condition = idsCondition(removed);
      await Promise.all([tables.chunks.delete(condition), tables.files.delete(condition)]);
      if (tables.references !== null) await tables.references.delete(condition);
    }
  }

  async removeFile(projectId: string, fileId: string): Promise<void> {
    const tables = await this.#tables(projectId);
    if (tables.references === null) throw new Error(MISSING_REFERENCE_TABLE);
    const condition = equals("file_id", fileId);
    await tables.chunks.delete(condition);
    await tables.references.delete(condition);
    await tables.files.delete(condition);
  }

  async tableVersions(projectId: string): Promise<TableVersions> {
    const tables = await this.#tables(projectId);
    if (tables.references === null) throw new Error(MISSING_REFERENCE_TABLE);
    const [files, chunks, references] = await Promise.all([
      tables.files.version(),
      tables.chunks.version(),
      tables.references.version(),
    ]);
    return { files, chunks, references };
  }

  async restoreVersions(
    projectId: string,
    versions: TableVersions,
    options: { restoreReferences?: boolean } = {},
  ): Promise<boolean> {
    const tables = await this.#existingTables(projectId);
    if (tables === null) return false;
    await restore(tables.files, versions.files);
    await restore(tables.chunks, versions.chunks);
    if (options.restoreReferences ?? true) {
      if (tables.references === null) throw new Error(MISSING_REFERENCE_TABLE);
      await restore(tables.references, versions.references);
    }
    return true;
  }

  async markProjectState(projectId: string, state: string): Promise<boolean> {
    const projects = await this.#projects();
    const record = (await rows<StoredProject>(projects, equals("id", projectId)))[0];
    if (record === undefined) return false;
    if (record.state !== state) {
      await merge(projects, "id", [
        { ...record, state, updated_at: BigInt(Date.now()) * 1_000_000n },
      ]);
    }
    return true;
  }

  async hasReferenceTable(projectId: string): Promise<boolean> {
    const tables = await this.#existingTables(projectId);
    return tables !== null && tables.references !== null;
  }

  async referenceVersion(projectId: string): Promise<number> {
    const references = (await this.#existingTables(projectId))?.references;
    return references === null || references === undefined ? 0 : references.version();
  }

  async listReferenceRecords(
    projectId: string,
    options: { version?: number; schemaVersion?: number; recordKinds?: readonly string[] },
  ): Promise<ReferenceRecord[]> {
    validateSchemaVersion(options.schemaVersion);
    if (options.recordKinds?.length === 0) return [];
    return this.#referenceRows(projectId, referenceCondition(options), options.version);
  }

  async declarationsForFiles(
    projectId: string,
    fileIds: Iterable<string>,
    options: { version?: number; schemaVersion?: number },
  ): Promise<ReferenceRecord[]> {
    validateSchemaVersion(options.schemaVersion);
    const ids = [...new Set(fileIds)];
    if (ids.length === 0) return [];
    return this.#referenceRows(
      projectId,
      and("record_kind = 'declaration'", idsCondition(ids), schemaCondition(options.schemaVersion)),
      options.version,
    );
  }

  async targetNameCandidates(
    projectId: string,
    targetName: string,
    options: { recordKind?: string; version?: number; schemaVersion?: number },
  ): Promise<ReferenceRecord[]> {
    validateSchemaVersion(options.schemaVersion);
    return this.#referenceRows(
      projectId,
      and(
        options.recordKind === undefined ? null : equals("record_kind", options.recordKind),
        equals("target_name", targetName),
        schemaCondition(options.schemaVersion),
      ),
      options.version,
    );
  }

  async referenceCoverage(
    projectId: string,
    options: { version?: number } = {},
  ): Promise<ReferenceRecord[]> {
    return this.#referenceRows(projectId, "record_kind = 'coverage'", options.version);
  }

  async coverageForFile(
    projectId: string,
    fileId: string,
    schemaVersion: number,
  ): Promise<ReferenceRecord[]> {
    validateSchemaVersion(schemaVersion);
    return this.#referenceRows(
      projectId,
      and("record_kind = 'coverage'", equals("file_id", fileId), schemaCondition(schemaVersion)),
    );
  }

  async declarationShapes(
    projectId: string,
    qualifiedSymbol: string,
    options: { version?: number; schemaVersion?: number },
  ): Promise<ReferenceRecord[]> {
    validateSchemaVersion(options.schemaVersion);
    return this.#referenceRows(
      projectId,
      and(
        "record_kind = 'declaration'",
        equals("source_qualified_symbol", qualifiedSymbol),
        schemaCondition(options.schemaVersion),
      ),
      options.version,
    );
  }

  async getChunk(chunkId: string): Promise<CodeChunk | null> {
    const projectId = chunkIdPrefix(chunkId);
    if (projectId === null || !(await exists(this.#partitionPath(projectId)))) return null;
    const tables = await this.#existingTables(projectId);
    if (tables === null) return null;
    const row = (
      await rows<Record<string, unknown>>(tables.chunks, equals("chunk_id", chunkId), [
        ...CHUNK_PAYLOAD_COLUMNS,
      ])
    )[0];
    return row === undefined
      ? null
      : CodeChunk.parse({ ...numberFields(row), project_id: projectId });
  }

  async listChunks(projectIds: readonly string[]): Promise<IndexedChunk[]> {
    const chunks = await Promise.all(
      projectIds.map(async (projectId) => {
        const tables = await this.#existingTables(projectId);
        return tables === null
          ? []
          : (await rows<Record<string, unknown>>(tables.chunks, undefined, [...CHUNK_COLUMNS])).map(
              (row) => IndexedChunk.parse(numberFields(row)),
            );
      }),
    );
    return chunks.flat();
  }

  async countChunks(projectIds: readonly string[]): Promise<number> {
    const counts = await Promise.all(
      projectIds.map(
        async (projectId) => (await this.#existingTables(projectId))?.chunks.countRows() ?? 0,
      ),
    );
    return counts.reduce((total, count) => total + count, 0);
  }

  async findSymbolChunks(
    name: string,
    projectId: string,
    options: { match: "exact" | "prefix" | "contains"; kinds?: readonly string[]; limit: number },
  ): Promise<ChunkPreview[]> {
    const tables = await this.#existingTables(projectId);
    if (tables === null) return [];
    const wildcard = options.match === "exact" ? name : `%${name}%`;
    const comparison = options.match === "exact" ? "=" : "LIKE";
    const filter = and(
      `(qualified_symbol ${comparison} ${quote(wildcard)} OR symbol ${comparison} ${quote(wildcard)})`,
      options.kinds === undefined || options.kinds.length === 0
        ? null
        : `kind IN (${options.kinds.map(quote).join(", ")})`,
    );
    const result = await tables.chunks
      .query()
      .where(filter)
      .select([
        "chunk_id",
        "path",
        "language",
        "kind",
        "symbol",
        "qualified_symbol",
        "parent_symbol",
        "start_line",
        "end_line",
        "content",
      ])
      .orderBy([{ columnName: "path" }, { columnName: "start_line" }, { columnName: "kind" }])
      .limit(Math.max(options.limit * 10, 200))
      .toArray();
    return result
      .map((row) => ChunkPreview.parse({ ...numberFields(row), project_id: projectId }))
      .filter((row) => symbolMatches(row, name, options.match))
      .slice(0, options.limit);
  }

  async outlineChunks(projectId: string, sourcePath: string): Promise<ChunkPreview[]> {
    const tables = await this.#existingTables(projectId);
    if (tables === null) return [];
    const result = await tables.chunks
      .query()
      .where(and(equals("path", sourcePath), "symbol IS NOT NULL", "qualified_symbol IS NOT NULL"))
      .select([
        "chunk_id",
        "path",
        "language",
        "kind",
        "symbol",
        "qualified_symbol",
        "parent_symbol",
        "start_line",
        "end_line",
      ])
      .toArray();
    return result.map((row) => ChunkPreview.parse({ ...numberFields(row), project_id: projectId }));
  }

  async hybridSearch(
    queryText: string,
    vector: readonly number[],
    projectIds: readonly string[],
    options: { condition?: string; limit: number },
  ): Promise<Record<string, unknown>[]> {
    const results = await mapPool(projectIds, SEARCH_CONCURRENCY, async (projectId) => {
      const tables = await this.#existingTables(projectId);
      if (tables === null) return [];
      // S1 proved this is the JS equivalent of Python's MultiMatch hybrid
      // query: text must be attached before vector because VectorQuery does
      // not expose nearestToText.
      const query = tables.chunks
        .query()
        .nearestToText(
          new MultiMatchQuery(queryText, ["content", "identifier_terms"], {
            operator: Operator.Or,
          }),
        )
        .nearestTo([...vector]);
      if (options.condition !== undefined) query.where(options.condition);
      if (this.vectorIndex === "exact") query.bypassVectorIndex();
      const reranker = await rerankers.RRFReranker.create();
      const queryRows = await query
        .rerank(reranker)
        .select([
          "chunk_id",
          "path",
          "language",
          "kind",
          "symbol",
          "qualified_symbol",
          "parent_symbol",
          "start_line",
          "end_line",
          "content",
        ])
        .limit(options.limit)
        .toArray();
      return queryRows.map((row) => ({ ...numberFields(row), project_id: projectId }));
    });
    return results.flat().sort(relevanceSort).slice(0, options.limit);
  }

  async ensureIndexes(projectId: string): Promise<void> {
    const tables = await this.#tables(projectId);
    const chunkIndexes = await tables.chunks.listIndices();
    const indexed = new Set(chunkIndexes.flatMap((index) => index.columns));
    for (const column of ["content", "identifier_terms"]) {
      if (!indexed.has(column)) {
        await tables.chunks.createIndex(column, {
          config: Index.fts({ lowercase: true, stem: false, removeStopWords: false }),
          replace: false,
        });
      }
    }
    for (const column of ["file_id", "language", "path", "symbol"]) {
      if (!indexed.has(column))
        await tables.chunks.createIndex(column, { config: Index.btree(), replace: false });
    }
    const vectorIndices = chunkIndexes.filter((index) => index.columns.includes("vector"));
    if (this.vectorIndex === "exact") {
      for (const index of vectorIndices) await tables.chunks.dropIndex(index.name);
    } else if (vectorIndices.length === 0 && (await tables.chunks.countRows()) >= 20_000) {
      await tables.chunks.createIndex("vector", {
        config: Index.hnswSq({ distanceType: "cosine" }),
        replace: false,
      });
    }
    if (tables.references === null) throw new Error(MISSING_REFERENCE_TABLE);
    const referenceIndexed = new Set(
      (await tables.references.listIndices()).flatMap((index) => index.columns),
    );
    for (const column of [
      "file_id",
      "record_kind",
      "target_name",
      "module_path",
      "kind",
      "source_qualified_symbol",
      "schema_version",
    ]) {
      if (!referenceIndexed.has(column)) {
        await tables.references.createIndex(column, { config: Index.btree(), replace: false });
      }
    }
  }

  async maintainProject(projectId: string, cleanupOlderThan: Date): Promise<boolean> {
    const tables = await this.#existingTables(projectId);
    if (tables === null) return false;
    const options = { cleanupOlderThan, deleteUnverified: false };
    await tables.files.optimize(options);
    await tables.chunks.optimize(options);
    if (tables.references !== null) await tables.references.optimize(options);
    return true;
  }

  async maintainRegistry(cleanupOlderThan: Date): Promise<void> {
    const projects = await this.#projects();
    await projects.optimize({ cleanupOlderThan, deleteUnverified: false });
  }

  async removeProject(projectId: string): Promise<boolean> {
    const projects = await this.#projects();
    const existed = (await rows<StoredProject>(projects, equals("id", projectId))).length > 0;
    if (existed) await projects.delete(equals("id", projectId));
    this.#closePartition(projectId);
    await fs.rm(this.#partitionPath(projectId), { recursive: true, force: true });
    return existed;
  }

  async registryStats(): Promise<TableStorageStats> {
    return this.#tableStats(
      await this.#projects(),
      "projects",
      path.join(this.directory, "registry"),
    );
  }

  async storageStats(projectId: string): Promise<ProjectStorageStats | null> {
    const project = (await this.listProjects()).find((item) => item.id === projectId);
    return project === undefined ? null : this.storageStatsFor(project);
  }

  async storageStatsFor(project: ProjectInfo): Promise<ProjectStorageStats> {
    const tables = await this.#existingTables(project.id);
    const partition = this.#partitionPath(project.id);
    const before = await partitionVersions(tables);
    const tableStats =
      tables === null
        ? []
        : await Promise.all([
            this.#tableStats(tables.files, "files", path.join(partition, "files.lance")),
            this.#tableStats(tables.chunks, "chunks", path.join(partition, "chunks.lance")),
            ...(tables.references === null
              ? []
              : [
                  this.#tableStats(
                    tables.references,
                    "references",
                    path.join(partition, "references.lance"),
                  ),
                ]),
          ]);
    const partitionExists = await exists(partition);
    const [partitionPhysicalBytes, after] = await Promise.all([
      directoryBytes(partition),
      partitionVersions(tables),
    ]);
    return {
      project,
      snapshot_at: new Date().toISOString(),
      tables: tableStats,
      partition_physical_bytes: partitionPhysicalBytes,
      consistent: equalVersions(before, after) && !(tables === null && partitionExists),
      partition_open_failed: tables === null && partitionExists,
    };
  }

  async storageStatus(): Promise<StorageStatus> {
    const [registry, projects] = await Promise.all([this.registryStats(), this.listProjects()]);
    const partitions = await Promise.all(projects.map((project) => this.storageStatsFor(project)));
    return {
      schema_version: 1,
      snapshot_at: new Date().toISOString(),
      registry,
      projects: partitions,
      physical_bytes_total: await directoryBytes(this.directory),
      consistent: partitions.every((item) => item.consistent),
      overlap_warnings: overlapWarnings(projects),
      worktree_warnings: worktreeWarnings(projects),
    };
  }

  async withPartitionAccess<T>(projectId: string, fn: () => Promise<T>): Promise<T> {
    const previous = this.#locks.get(projectId) ?? Promise.resolve();
    let release!: () => void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.#locks.set(
      projectId,
      previous.then(() => held),
    );
    await previous;
    try {
      return await fn();
    } finally {
      release();
      if (this.#locks.get(projectId) === held) this.#locks.delete(projectId);
    }
  }

  cachedPartitionIds(): string[] {
    return [...this.#partitions.keys()];
  }

  async openTables(projectId: string): Promise<void> {
    await this.#tables(projectId);
  }

  async close(): Promise<void> {
    for (const [projectId] of this.#partitions) this.#closePartition(projectId);
    (await this.#registry).connection.close();
  }

  async #openRegistry(): Promise<{ connection: Connection; projects: Table }> {
    const legacy = await migrateV1(this.directory);
    await fs.mkdir(this.directory, { recursive: true });
    const connection = await connect(path.join(this.directory, "registry"));
    const projects = await table(connection, "projects", LanceStore.projectArrowSchema());
    for (const row of legacy) {
      await merge(projects, "id", [
        {
          ...row,
          vector_dimension: this.vectorDimension,
          schema_version: SCHEMA_VERSION,
          state: "pending",
          updated_at: BigInt(Date.now()) * 1_000_000n,
        },
      ]);
    }
    return { connection, projects };
  }

  async #projects(): Promise<Table> {
    return (await this.#registry).projects;
  }

  #partitionPath(projectId: string): string {
    return path.join(this.directory, "projects", projectId);
  }

  #closePartition(projectId: string): void {
    const tables = this.#partitions.get(projectId);
    if (tables === undefined) return;
    tables.files.close();
    tables.chunks.close();
    tables.references?.close();
    this.#partitions.delete(projectId);
  }

  async #tables(projectId: string): Promise<ProjectTables> {
    const generation = await this.#partitionGeneration(projectId);
    const cached = this.#partitions.get(projectId);
    if (cached !== undefined && cached.references !== null && cached.generation === generation) {
      this.#partitions.delete(projectId);
      this.#partitions.set(projectId, cached);
      return cached;
    }
    await fs.mkdir(this.#partitionPath(projectId), { recursive: true });
    const connection = await connect(this.#partitionPath(projectId));
    const tables = {
      files: await table(connection, "files", LanceStore.fileArrowSchema()),
      chunks: await table(
        connection,
        "chunks",
        LanceStore.chunkArrowSchema(this.vectorDimension, this.vectorStorage),
      ),
      references: await table(connection, "references", LanceStore.referenceArrowSchema()),
      generation,
    };
    this.#remember(projectId, tables);
    return tables;
  }

  async #existingTables(projectId: string): Promise<ProjectTables | null> {
    const generation = await this.#partitionGeneration(projectId);
    const cached = this.#partitions.get(projectId);
    if (
      cached !== undefined &&
      cached.generation === generation &&
      (await exists(this.#partitionPath(projectId)))
    ) {
      this.#partitions.delete(projectId);
      this.#partitions.set(projectId, cached);
      return cached;
    }
    if (cached !== undefined) this.#closePartition(projectId);
    if (!(await exists(this.#partitionPath(projectId)))) return null;
    const connection = await connect(this.#partitionPath(projectId));
    try {
      const names = await connection.tableNames();
      if (!names.includes("files") || !names.includes("chunks")) {
        connection.close();
        return null;
      }
      const tables = {
        files: await connection.openTable("files"),
        chunks: await connection.openTable("chunks"),
        references: names.includes("references") ? await connection.openTable("references") : null,
        generation,
      };
      this.#remember(projectId, tables);
      return tables;
    } catch {
      connection.close();
      return null;
    }
  }

  async #partitionGeneration(projectId: string): Promise<number> {
    try {
      return Number.parseInt(
        await fs.readFile(path.join(this.directory, "partition-generations", projectId), "utf8"),
        10,
      );
    } catch {
      return 0;
    }
  }

  async #advancePartitionGeneration(projectId: string): Promise<void> {
    const directory = path.join(this.directory, "partition-generations");
    await fs.mkdir(directory, { recursive: true });
    const target = path.join(directory, projectId);
    const next = `${(await this.#partitionGeneration(projectId)) + 1}`;
    const temporary = `${target}.tmp`;
    await fs.writeFile(temporary, next);
    await fs.rename(temporary, target);
  }

  #remember(projectId: string, tables: ProjectTables): void {
    this.#closePartition(projectId);
    this.#partitions.set(projectId, tables);
    while (this.#partitions.size > MAX_CACHED_PARTITIONS) {
      const oldest = this.#partitions.keys().next().value;
      if (oldest === undefined) break;
      this.#closePartition(oldest);
    }
  }

  async #referenceRows(
    projectId: string,
    condition: string | null,
    version?: number,
  ): Promise<ReferenceRecord[]> {
    const tables = await this.#existingTables(projectId);
    if (tables?.references === null || tables === null) return [];
    let references = tables.references;
    if (version !== undefined && version !== (await references.version())) {
      try {
        const connection = await connect(this.#partitionPath(projectId));
        references = await connection.openTable("references", [], { version });
      } catch (error) {
        throw new ReferenceSnapshotExpiredError(`no such reference table version ${version}`, {
          cause: error,
        });
      }
    }
    const query = references
      .query()
      .orderBy([
        { columnName: "path" },
        { columnName: "start_line" },
        { columnName: "reference_id" },
      ]);
    if (condition !== null) query.where(condition);
    return (await query.toArray()).map(
      (row) => ({ ...numberFields(row), project_id: projectId }) as ReferenceRecord,
    );
  }

  async #tableStats(
    tableHandle: Table,
    name: string,
    physicalPath: string,
  ): Promise<TableStorageStats> {
    const [currentVersion, statistics, versions, indices, physicalBytes] = await Promise.all([
      tableHandle.version(),
      tableHandle.stats().catch(() => null),
      tableHandle.listVersions().catch(() => []),
      tableHandle.listIndices().catch(() => []),
      directoryBytes(physicalPath),
    ]);
    return {
      name,
      current_version: currentVersion,
      row_count: statistics?.numRows ?? 0,
      logical_bytes: statistics?.totalBytes ?? 0,
      physical_bytes: physicalBytes,
      fragment_stats: {
        num_fragments: statistics?.fragmentStats.numFragments ?? 0,
        num_small_fragments: statistics?.fragmentStats.numSmallFragments ?? 0,
        lengths: null,
      },
      retained_version_count: versions.length,
      oldest_version_at: versions[0]?.timestamp.toISOString() ?? null,
      newest_version_at: versions.at(-1)?.timestamp.toISOString() ?? null,
      indexes: indices.map((index) => ({
        name: index.name,
        index_type: index.indexType,
        columns: index.columns,
        indexed_rows: index.numIndexedRows ?? 0,
        unindexed_rows: index.numUnindexedRows ?? 0,
        size_bytes: index.sizeBytes ?? 0,
      })),
    };
  }
}

function field(name: string, type: DataType = new Utf8(), nullable = false): Field {
  return new Field(name, type, nullable);
}

function schema(fields: Field[]): Schema {
  return new Schema(fields);
}

async function table(connection: Connection, name: string, schema: Schema): Promise<Table> {
  if ((await connection.tableNames()).includes(name)) return connection.openTable(name);
  return connection.createEmptyTable(name, schema);
}

async function merge(
  tableHandle: Table,
  key: string,
  records: readonly Record<string, unknown>[],
): Promise<void> {
  if (records.length === 0) return;
  await tableHandle
    .mergeInsert(key)
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .execute([...records]);
}

async function replaceRows<T extends object>(
  tableHandle: Table,
  key: string,
  fileIds: readonly string[],
  source: readonly T[],
  map: (row: T) => Record<string, unknown> = (row) => row as Record<string, unknown>,
): Promise<void> {
  if (fileIds.length === 0) return;
  if (source.length === 0) {
    await tableHandle.delete(idsCondition(fileIds));
    return;
  }
  await tableHandle
    .mergeInsert(key)
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .whenNotMatchedBySourceDelete({ where: idsCondition(fileIds) })
    .execute(source.map(map));
}

async function restore(tableHandle: Table, version: number): Promise<void> {
  if ((await tableHandle.version()) === version) return;
  await tableHandle.checkout(version);
  await tableHandle.restore();
  await tableHandle.checkoutLatest();
}

async function rows<T>(
  tableHandle: Table,
  condition?: string,
  columns?: readonly string[],
): Promise<T[]> {
  const query = tableHandle.query();
  if (condition !== undefined) query.where(condition);
  if (columns !== undefined) query.select([...columns]);
  return (await query.toArray()) as T[];
}

function quote(value: string): string {
  return `'${value.replaceAll("'", "''")}'`;
}

function equals(column: string, value: string): string {
  return `${column} = ${quote(value)}`;
}

function idsCondition(ids: readonly string[]): string {
  return `file_id IN (${ids.map(quote).join(", ")})`;
}

function and(...conditions: (string | null)[]): string {
  return conditions.filter((condition): condition is string => condition !== null).join(" AND ");
}

function schemaCondition(version: number | undefined): string | null {
  return version === undefined ? null : `schema_version = ${version}`;
}

function validateSchemaVersion(version: number | undefined): void {
  if (version !== undefined && (typeof version !== "number" || !Number.isInteger(version))) {
    throw new TypeError("schema_version must be a non-boolean integer");
  }
}

function chunkIdPrefix(chunkId: string): string | null {
  const index = chunkId.indexOf(":");
  if (index <= 0 || index === chunkId.length - 1) return null;
  return chunkId.slice(0, index);
}

export function overlapWarnings(projects: readonly ProjectInfo[]): string[] {
  const warnings: string[] = [];
  for (const [left, right] of pairs(projects)) {
    if (sameProjectRoot(left.root, right.root)) {
      warnings.push(`Projects '${left.id}' and '${right.id}' register the same root: ${left.root}`);
      continue;
    }
    const leftResolved = resolvePath(left.root);
    const rightResolved = resolvePath(right.root);
    if (isRelativeTo(leftResolved, rightResolved) && leftResolved !== rightResolved) {
      warnings.push(
        `Project '${left.id}' root ${leftResolved} is nested inside the root of project '${right.id}' (${rightResolved})`,
      );
    } else if (isRelativeTo(rightResolved, leftResolved) && leftResolved !== rightResolved) {
      warnings.push(
        `Project '${left.id}' root ${leftResolved} contains the root of project '${right.id}' (${rightResolved})`,
      );
    }
  }
  return warnings;
}

export function overlappingRegistration(
  projects: readonly ProjectInfo[],
  root: string,
): ProjectInfo | null {
  const resolved = resolvePath(root);
  for (const project of projects) {
    const existing = resolvePath(project.root);
    if (sameProjectRoot(existing, resolved)) return project;
    if (rootedUnder(existing, resolved) || rootedUnder(resolved, existing)) return project;
  }
  return null;
}

export function worktreeWarnings(
  projects: readonly ProjectInfo[],
  run: GitRunner = runGitQuietly,
): string[] {
  const repositories: Array<{ project: ProjectInfo; toplevel: string; common: string }> = [];
  for (const project of projects) {
    const toplevel = run(["git", "rev-parse", "--show-toplevel"], project.root);
    const common = run(["git", "rev-parse", "--git-common-dir"], project.root);
    if (toplevel === null || common === null) continue;
    repositories.push({
      project,
      toplevel,
      common: path.isAbsolute(common) ? common : resolvePath(path.join(project.root, common)),
    });
  }
  const warnings: string[] = [];
  for (const [left, right] of pairs(repositories)) {
    if (
      sameProjectRoot(left.common, right.common) &&
      !sameProjectRoot(left.toplevel, right.toplevel)
    ) {
      warnings.push(
        `Projects '${left.project.id}' and '${right.project.id}' share Git common directory ${left.common} from different checkouts (possible worktrees of one repository)`,
      );
    }
  }
  return warnings;
}

function runGitQuietly(command: readonly string[], cwd: string): string | null {
  try {
    return execFileSync(command[0] as string, command.slice(1), {
      cwd,
      encoding: "utf8",
      timeout: GIT_TIMEOUT_MS,
      stdio: ["ignore", "pipe", "pipe"],
    }).trim();
  } catch {
    return null;
  }
}

function* pairs<T>(items: readonly T[]): Generator<[T, T]> {
  for (let left = 0; left < items.length; left += 1) {
    for (let right = left + 1; right < items.length; right += 1) {
      yield [items[left] as T, items[right] as T];
    }
  }
}

let mergeSemantics: boolean | undefined;

export async function probeBatchedMergeSemantics(): Promise<boolean> {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "ci-mcp-merge-probe-"));
  try {
    const connection = await connect(directory);
    const tableHandle = await connection.createTable("probe", [
      { file_id: "a", chunk_id: "a1", vector: [0, 0] },
      { file_id: "b", chunk_id: "b1", vector: [0, 0] },
      { file_id: "c", chunk_id: "c1", vector: [0, 0] },
    ]);
    await tableHandle
      .mergeInsert("chunk_id")
      .whenMatchedUpdateAll()
      .whenNotMatchedInsertAll()
      .whenNotMatchedBySourceDelete({ where: "file_id IN ('a', 'b')" })
      .execute([{ file_id: "a", chunk_id: "a2", vector: [0, 0] }]);
    const surviving = (await tableHandle.query().select(["chunk_id"]).toArray())
      .map((row) => String(row.chunk_id))
      .sort();
    connection.close();
    return surviving.join() === "a2,c1";
  } catch {
    return false;
  } finally {
    await fs.rm(directory, { recursive: true, force: true });
  }
}

export async function batchedMergeSemanticsOk(): Promise<boolean> {
  mergeSemantics ??= await probeBatchedMergeSemantics();
  return mergeSemantics;
}

export function setBatchedMergeSemanticsOk(value: boolean | undefined): void {
  mergeSemantics = value;
}

async function migrateV1(directory: string): Promise<StoredProject[]> {
  if (!(await exists(path.join(directory, "projects.lance")))) return [];
  const connection = await connect(directory);
  try {
    const projects = await connection.openTable("projects");
    const records = await rows<StoredProject>(projects);
    connection.close();
    const backup = `${directory}-v1-backup-${process.hrtime.bigint()}`;
    await fs.rename(directory, backup);
    return records;
  } catch {
    connection.close();
    return [];
  }
}

async function mapPool<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const index = next;
      next += 1;
      results[index] = await fn(items[index] as T);
    }
  });
  await Promise.all(workers);
  return results;
}

function referenceCondition(options: {
  schemaVersion?: number;
  recordKinds?: readonly string[];
}): string | null {
  return (
    and(
      schemaCondition(options.schemaVersion),
      options.recordKinds === undefined
        ? null
        : `record_kind IN (${options.recordKinds.map(quote).join(", ")})`,
    ) || null
  );
}

function fileHash(files: readonly StoredFile[], fileId: string): string {
  return files.find((file) => file.file_id === fileId)?.content_hash ?? "";
}

function parseStoredFile(row: Record<string, unknown>): StoredFile {
  return StoredFile.parse({ ...numberFields(row), mtime_ns: BigInt(String(row.mtime_ns)) });
}

/** Arrow returns int64 values as bigint; public API offsets remain JSON-safe numbers. */
function numberFields(row: Record<string, unknown>): Record<string, unknown> {
  const result = { ...row };
  for (const name of ["size", "indexed_at", "start_byte", "end_byte", "start_line", "end_line"]) {
    if (typeof result[name] === "bigint") result[name] = Number(result[name]);
  }
  return result;
}

function sameProject(left: StoredProject, right: StoredProject): boolean {
  return (
    left.id === right.id &&
    left.name === right.name &&
    left.root === right.root &&
    left.payload === right.payload &&
    left.model_id === right.model_id &&
    left.vector_dimension === right.vector_dimension &&
    left.schema_version === right.schema_version &&
    left.state === right.state
  );
}

function symbolMatches(
  chunk: { symbol: string | null; qualified_symbol: string | null },
  name: string,
  match: "exact" | "prefix" | "contains",
): boolean {
  return [chunk.symbol, chunk.qualified_symbol].some((value) => {
    if (value === null) return false;
    return match === "exact"
      ? value === name
      : match === "prefix"
        ? value.startsWith(name)
        : value.includes(name);
  });
}

function relevanceSort(left: Record<string, unknown>, right: Record<string, unknown>): number {
  return Number(right._relevance_score ?? 0) - Number(left._relevance_score ?? 0);
}

async function partitionVersions(tables: ProjectTables | null): Promise<TableVersions> {
  if (tables === null) return { files: 0, chunks: 0, references: 0 };
  const [files, chunks, references] = await Promise.all([
    tables.files.version(),
    tables.chunks.version(),
    tables.references?.version() ?? 0,
  ]);
  return { files, chunks, references };
}

function equalVersions(left: TableVersions, right: TableVersions): boolean {
  return (
    left.files === right.files &&
    left.chunks === right.chunks &&
    left.references === right.references
  );
}

async function exists(target: string): Promise<boolean> {
  return fs
    .stat(target)
    .then(() => true)
    .catch(() => false);
}

async function directoryBytes(directory: string): Promise<number> {
  let total = 0;
  let entries: Dirent<string>[];
  try {
    entries = await fs.readdir(directory, { withFileTypes: true });
  } catch {
    return 0;
  }
  for (const entry of entries) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) total += await directoryBytes(target);
    else if (entry.isFile()) total += Number((await fs.stat(target)).size);
  }
  return total;
}
