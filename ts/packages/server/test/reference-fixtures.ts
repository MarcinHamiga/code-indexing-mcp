/**
 * An in-memory {@link ReferenceStore} built by running the real extractor.
 *
 * The Python resolver suites index a temporary repository through the whole
 * pipeline (`Indexer` + `LanceStore`) and query the result. Neither exists yet
 * -- storage is Phase 3 and the indexer is Phase 5 -- so this stands in for the
 * part that matters: it writes the files to disk, runs the *real*
 * `TreeSitterExtractor` over them, and assembles rows with the *real*
 * `referenceRows`. What it replaces is only the persistence, so the resolver is
 * still exercised against the rows the pipeline actually produces rather than
 * against rows hand-written to make it pass.
 *
 * When Phase 5 lands, these suites should be re-pointed at a real indexed
 * store; the fixtures and assertions carry over unchanged, and any behaviour
 * that only holds against this store will show up then.
 */

import fs from "node:fs";
import path from "node:path";
import { TreeSitterExtractor } from "../src/extractor.ts";
import type { CodeChunk, IndexedChunk, ProjectInfo } from "../src/models.ts";
import { digest, fileId, referenceRows } from "../src/reference-records.ts";
import {
  REFERENCE_SCHEMA_VERSION,
  ReferenceSnapshotExpiredError,
  type ReferenceRecord,
  type ReferenceStore,
} from "../src/reference-store.ts";
import { languageForExtension } from "../src/scanner.ts";

export class InMemoryReferenceStore implements ReferenceStore {
  readonly project: ProjectInfo;
  readonly records: ReferenceRecord[] = [];
  readonly chunks: IndexedChunk[] = [];
  /** Set false to model a partition whose reference index was never built. */
  referenceTableExists = true;
  version = 1;

  constructor(project: ProjectInfo) {
    this.project = project;
  }

  async hasReferenceTable(projectId: string): Promise<boolean> {
    return this.referenceTableExists && projectId === this.project.id;
  }

  async referenceVersion(): Promise<number> {
    return this.version;
  }

  /**
   * The shared filter every query below applies.
   *
   * Deliberately *not* implemented by delegating to `listReferenceRecords`: the
   * real store issues each of these as its own pushed-down query, so a suite
   * that counts calls to prove the pushdown happened would otherwise see one
   * fake method's implementation detail rather than the resolver's behaviour.
   */
  #rows(
    projectId: string,
    options: { version?: number; schemaVersion?: number; recordKinds?: readonly string[] },
  ): ReferenceRecord[] {
    if (projectId !== this.project.id) return [];
    if (options.version !== undefined && options.version !== this.version) {
      // A pinned snapshot that no longer exists is what the real store reports
      // as a vanished table version, and the service maps to STALE_CURSOR.
      throw new ReferenceSnapshotExpiredError(`no such table version ${options.version}`);
    }
    const kinds = options.recordKinds === undefined ? null : new Set(options.recordKinds);
    return this.records.filter(
      (row) =>
        (options.schemaVersion === undefined || row.schema_version === options.schemaVersion) &&
        (kinds === null || kinds.has(row.record_kind)),
    );
  }

  async listReferenceRecords(
    projectId: string,
    options: { version?: number; schemaVersion?: number; recordKinds?: readonly string[] },
  ): Promise<ReferenceRecord[]> {
    return this.#rows(projectId, options);
  }

  async declarationsForFiles(
    projectId: string,
    fileIds: Iterable<string>,
    options: { version?: number; schemaVersion?: number },
  ): Promise<ReferenceRecord[]> {
    const wanted = new Set(fileIds);
    if (wanted.size === 0) return [];
    return this.#rows(projectId, { ...options, recordKinds: ["declaration"] }).filter((row) =>
      wanted.has(row.file_id),
    );
  }

  async targetNameCandidates(
    projectId: string,
    targetName: string,
    options: { recordKind?: string; version?: number; schemaVersion?: number },
  ): Promise<ReferenceRecord[]> {
    const narrowed =
      options.recordKind === undefined
        ? options
        : { ...options, recordKinds: [options.recordKind] };
    return this.#rows(projectId, narrowed).filter((row) => row.target_name === targetName);
  }

  async declarationShapes(
    projectId: string,
    qualifiedSymbol: string,
    options: { version?: number; schemaVersion?: number },
  ): Promise<ReferenceRecord[]> {
    return this.#rows(projectId, { ...options, recordKinds: ["declaration"] }).filter(
      (row) => row.source_qualified_symbol === qualifiedSymbol,
    );
  }

  async getChunk(chunkId: string): Promise<CodeChunk | null> {
    const chunk = this.chunks.find((item) => item.chunk_id === chunkId);
    if (chunk === undefined) return null;
    const { identifier_terms: _identifierTerms, ...rest } = chunk;
    return { ...rest, project_id: this.project.id };
  }

  async listChunks(projectIds: readonly string[]): Promise<IndexedChunk[]> {
    return projectIds.includes(this.project.id) ? this.chunks : [];
  }

  async listProjects(): Promise<ProjectInfo[]> {
    return [this.project];
  }
}

/**
 * Write `files` under a fresh repository root, extract them, and return a store.
 *
 * Mirrors `_indexed_service` in the Python suites: every file the scanner would
 * classify is extracted, and every extracted file gets its structural
 * generation, including the coverage row that proves a language with no
 * reference query was still looked at.
 */
export function indexedStore(
  root: string,
  files: Readonly<Record<string, string>>,
  options: { projectId?: string } = {},
): InMemoryReferenceStore {
  const projectId = options.projectId ?? "test-project";
  fs.mkdirSync(root, { recursive: true });
  for (const [relative, source] of Object.entries(files)) {
    const destination = path.join(root, ...relative.split("/"));
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.writeFileSync(destination, source);
  }
  const store = new InMemoryReferenceStore({
    version: 1,
    id: projectId,
    name: path.basename(root),
    root,
    scan: { include: [], exclude: [], max_file_bytes: 1_048_576 },
  });
  const extractor = new TreeSitterExtractor();
  for (const relative of Object.keys(files).sort()) {
    const language = languageForExtension(path.extname(relative));
    if (language === undefined) continue;
    const bytes = new Uint8Array(fs.readFileSync(path.join(root, ...relative.split("/"))));
    const result = extractor.extract(relative, language, bytes);
    const file = {
      file_id: fileId(projectId, relative),
      project_id: projectId,
      path: relative,
      language,
      size: bytes.length,
      mtime_ns: 0n,
      content_hash: digest(bytes),
      has_errors: result.has_errors,
      error: null,
      indexed_at: 0,
    };
    store.records.push(...referenceRows(projectId, file, result.references, result.declarations));
    for (const chunk of result.chunks) {
      store.chunks.push({
        chunk_id: `${projectId}:${digest([file.file_id, file.content_hash, chunk.kind, chunk.qualified_symbol ?? "", String(chunk.start_byte), String(chunk.end_byte), String(chunk.part_index)].join("\0"))}`,
        file_id: file.file_id,
        path: relative,
        language,
        kind: chunk.kind,
        symbol: chunk.symbol,
        qualified_symbol: chunk.qualified_symbol,
        parent_symbol: chunk.parent_symbol,
        start_byte: chunk.start_byte,
        end_byte: chunk.end_byte,
        start_line: chunk.start_line,
        end_line: chunk.end_line,
        content: chunk.content,
        identifier_terms: chunk.search_suffix,
        part_index: chunk.part_index,
        content_hash: file.content_hash,
      });
    }
  }
  return store;
}

/** The schema version every fixture writes, so a test can pin a stale one. */
export const FIXTURE_SCHEMA_VERSION = REFERENCE_SCHEMA_VERSION;
