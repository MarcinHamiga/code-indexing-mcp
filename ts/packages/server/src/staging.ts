/**
 * Journalled Arrow staging for crash-recoverable index commits.
 *
 * A run writes its file, chunk, and reference rows beneath
 * `<data>/staging/<project-id>/<job-id>/` before mutating Lance.  The journal
 * moves to `committing` only after the Arrow IPC payloads are durable and the
 * live table versions have been recorded, leaving startup recovery enough
 * information to restore a partially applied commit.
 */

import { randomUUID } from "node:crypto";
import type { Dirent, WriteStream } from "node:fs";
import { createWriteStream } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { finished } from "node:stream/promises";
import {
  makeData,
  Precision,
  RecordBatch,
  RecordBatchFileWriter,
  RecordBatchReader,
  type Schema,
  Struct,
  Table,
  tableToIPC,
  vectorFromArray,
} from "apache-arrow";
import type { StoredChunk, StoredFile } from "./models.ts";
import type { ReferenceRecord } from "./reference-store.ts";
import type { LanceStore, ReplacementBatch, TableVersions } from "./storage.ts";

export const JOURNAL_NAME = "journal.json";
export const FILES_NAME = "files.arrow";
export const CHUNKS_NAME = "chunks.arrow";
export const REFERENCES_NAME = "references.arrow";

export const PHASE_STAGING = "staging";
export const PHASE_COMMITTING = "committing";
export const PHASE_COMPLETE = "complete";
export const PHASE_ROLLED_BACK = "rolled_back";

export const LEGACY_JOURNAL_FORMAT_VERSION = 1;
export const JOURNAL_FORMAT_VERSION = 2;
export const MAX_RECOVERY_ATTEMPTS = 3;

export const COMMIT_BATCH_MAX_FILES = 64;
export const COMMIT_BATCH_MAX_ROWS = 20_000;
export const COMMIT_BATCH_MAX_BYTES = 64 * 1024 * 1024;

/** The TypeScript model keeps vectors unpacked; Arrow owns their IPC encoding. */
export type ChunkRow = StoredChunk;
export type ReferenceRow = ReferenceRecord;

type JournalPhase =
  | typeof PHASE_STAGING
  | typeof PHASE_COMMITTING
  | typeof PHASE_COMPLETE
  | typeof PHASE_ROLLED_BACK;

interface StagingJournal {
  version: number;
  job_id: string;
  project_id: string;
  phase: JournalPhase | string;
  created_at_ns: number;
  files_version?: number;
  chunks_version?: number;
  references_version?: number;
  replace_file_ids?: string[];
  replace_reference_file_ids?: string[];
  removed_file_ids?: string[];
  recovery_attempts?: number;
}

interface PayloadWriter {
  schema: Schema;
  writer: RecordBatchFileWriter;
  output: WriteStream;
  temporary: string;
  wrote: boolean;
}

export type StagingStore = Pick<
  LanceStore,
  "restoreVersions" | "markProjectState" | "replaceFilesFromArrow" | "tableVersions"
>;

export interface StagingJobOptions {
  fileSchema: Schema;
  chunkSchema: Schema;
  referenceSchema: Schema;
  jobId?: string;
}

/** Write a whole small control file, then atomically publish and sync it. */
async function writeAtomically(target: string, payload: string): Promise<void> {
  const temporary = `${target}.tmp`;
  let handle: fs.FileHandle | null = null;
  try {
    handle = await fs.open(temporary, "wx");
    await handle.writeFile(payload, "utf8");
    await handle.sync();
    await handle.close();
    handle = null;
    await fs.rename(temporary, target);
    await syncDirectory(path.dirname(target));
  } catch (error) {
    if (handle !== null) await handle.close().catch(() => undefined);
    await fs.rm(temporary, { force: true }).catch(() => undefined);
    throw error;
  }
}

/** Directory fsync is unavailable on Windows and some filesystems. */
async function syncDirectory(directory: string): Promise<void> {
  let handle: fs.FileHandle | null = null;
  try {
    handle = await fs.open(directory, "r");
    await handle.sync();
  } catch {
    // The preceding file fsync and rename still provide the strongest guarantee
    // supported by platforms that cannot open a directory as a file.
  } finally {
    await handle?.close().catch(() => undefined);
  }
}

/** Whether maintenance must retain table versions for a possibly interrupted commit. */
export async function hasPendingRecovery(stagingRoot: string, projectId: string): Promise<boolean> {
  const projectDirectory = path.join(stagingRoot, projectId);
  let jobs: Dirent[];
  try {
    jobs = await fs.readdir(projectDirectory, { withFileTypes: true });
  } catch (error) {
    return !isMissing(error);
  }
  const known = new Set<string>([
    PHASE_STAGING,
    PHASE_COMMITTING,
    PHASE_COMPLETE,
    PHASE_ROLLED_BACK,
  ]);
  for (const job of jobs) {
    if (!job.isDirectory()) continue;
    try {
      const journal = parseJournal(
        await fs.readFile(path.join(projectDirectory, job.name, JOURNAL_NAME), "utf8"),
      );
      if (journal.phase === PHASE_COMMITTING || !known.has(journal.phase)) return true;
    } catch (error) {
      if (isMissing(error)) continue;
      return true;
    }
  }
  return false;
}

/** One index run's staged Arrow payloads and recovery journal. */
export class StagingJob {
  readonly stagingRoot: string;
  readonly projectId: string;
  readonly jobId: string;
  readonly replaceFileIds: string[] = [];
  readonly replaceReferenceFileIds: string[] = [];
  readonly removedFileIds: string[] = [];

  #journal: StagingJournal | null = null;
  #files: PayloadWriter | null = null;
  #chunks: PayloadWriter | null = null;
  #references: PayloadWriter | null = null;
  #chunkFileIds = new Set<string>();
  #referenceFileIds = new Set<string>();
  #lastChunkFileId: string | null = null;
  #lastReferenceFileId: string | null = null;
  #fileSchema: Schema;
  #chunkSchema: Schema;
  #referenceSchema: Schema;

  constructor(stagingRoot: string, projectId: string, options: StagingJobOptions) {
    this.stagingRoot = stagingRoot;
    this.projectId = projectId;
    this.jobId = options.jobId ?? `${Date.now().toString(16)}-${randomUUID()}`;
    this.#fileSchema = options.fileSchema;
    this.#chunkSchema = options.chunkSchema;
    this.#referenceSchema = options.referenceSchema;
  }

  get directory(): string {
    return path.join(this.stagingRoot, this.projectId, this.jobId);
  }

  async begin(): Promise<void> {
    if (this.#journal !== null) throw new Error("staging job has already begun");
    await fs.mkdir(this.directory, { recursive: true });
    this.#journal = {
      version: JOURNAL_FORMAT_VERSION,
      job_id: this.jobId,
      project_id: this.projectId,
      phase: PHASE_STAGING,
      created_at_ns: Date.now() * 1_000_000,
    };
    await this.#writeJournal();
    this.#files = this.#openWriter(FILES_NAME, this.#fileSchema);
    this.#chunks = this.#openWriter(CHUNKS_NAME, this.#chunkSchema);
    this.#references = this.#openWriter(REFERENCES_NAME, this.#referenceSchema);
  }

  async stageFile(record: StoredFile): Promise<void> {
    await this.#writeRows(this.#requireWriter(this.#files), [record]);
  }

  async stageChunks(rows: readonly ChunkRow[]): Promise<void> {
    const first = rows[0];
    if (first === undefined) return;
    const fileId = first.file_id;
    if (rows.some((row) => row.file_id !== fileId)) {
      throw new Error("a staged chunk batch must contain rows from one file");
    }
    if (this.#chunkFileIds.has(fileId) && this.#lastChunkFileId !== fileId) {
      throw new Error("staged chunk batches for a file must be contiguous");
    }
    this.#chunkFileIds.add(fileId);
    this.#lastChunkFileId = fileId;
    await this.#writeRows(this.#requireWriter(this.#chunks), rows);
  }

  async stageReferences(rows: readonly ReferenceRow[]): Promise<void> {
    const first = rows[0];
    if (first === undefined) return;
    const fileId = first.file_id;
    if (rows.some((row) => row.file_id !== fileId)) {
      throw new Error("a staged reference batch must contain rows from one file");
    }
    if (this.#referenceFileIds.has(fileId) && this.#lastReferenceFileId !== fileId) {
      throw new Error("staged reference batches for a file must be contiguous");
    }
    this.#referenceFileIds.add(fileId);
    this.#lastReferenceFileId = fileId;
    await this.#writeRows(this.#requireWriter(this.#references), rows);
  }

  markReplaced(fileId: string): void {
    addOnce(this.replaceFileIds, fileId);
  }

  markReferencesReplaced(fileId: string): void {
    addOnce(this.replaceReferenceFileIds, fileId);
  }

  markRemoved(fileId: string): void {
    addOnce(this.removedFileIds, fileId);
  }

  async beginCommit(versions: TableVersions): Promise<void> {
    this.#requireJournal();
    await this.#closeWriters();
    this.#journal = {
      ...this.#requireJournal(),
      phase: PHASE_COMMITTING,
      files_version: versions.files,
      chunks_version: versions.chunks,
      references_version: versions.references,
      replace_file_ids: [...this.replaceFileIds],
      replace_reference_file_ids: [...this.replaceReferenceFileIds],
      removed_file_ids: [...this.removedFileIds],
    };
    await this.#writeJournal();
  }

  /** The storage API accepts decoded rows; the durable source is Arrow IPC. */
  async filesTable(): Promise<StoredFile[]> {
    return this.#readRows<StoredFile>(FILES_NAME);
  }

  async iterChunkBatches(options: BatchOptions = {}): Promise<ReplacementBatch<StoredChunk>[]> {
    return this.#batches<StoredChunk>(CHUNKS_NAME, this.replaceFileIds, options);
  }

  async iterReferenceBatches(
    options: BatchOptions = {},
  ): Promise<ReplacementBatch<ReferenceRecord>[]> {
    return this.#batches<ReferenceRecord>(REFERENCES_NAME, this.replaceReferenceFileIds, options);
  }

  async complete(): Promise<void> {
    this.#setTerminalPhase(PHASE_COMPLETE);
    await this.#writeJournal();
    await fs.rm(this.directory, { recursive: true, force: true });
  }

  async rolledBack(): Promise<void> {
    this.#setTerminalPhase(PHASE_ROLLED_BACK);
    await this.#writeJournal();
    await fs.rm(this.directory, { recursive: true, force: true });
  }

  async discard(): Promise<void> {
    if (this.#journal?.phase === PHASE_COMMITTING) return;
    await this.#closeWriters(false).catch(() => undefined);
    await fs.rm(this.directory, { recursive: true, force: true });
  }

  #openWriter(name: string, schema: Schema): PayloadWriter {
    const temporary = path.join(this.directory, `${name}.tmp`);
    const output = createWriteStream(temporary, { flags: "wx" });
    const writer = new RecordBatchFileWriter();
    writer.toNodeStream().pipe(output);
    return { schema, writer, output, temporary, wrote: false };
  }

  async #writeRows<T extends object>(payload: PayloadWriter, rows: readonly T[]): Promise<void> {
    // `RecordBatchFileWriter.write(Table)` finishes an auto-destroy writer.
    // Feed its batches directly so one file writer remains open for the run.
    for (const batch of tableFromRows(payload.schema, rows).batches) payload.writer.write(batch);
    payload.wrote = true;
  }

  async #closeWriters(finalize = true): Promise<void> {
    const writers = [this.#files, this.#chunks, this.#references].filter(
      (writer): writer is PayloadWriter => writer !== null,
    );
    this.#files = this.#chunks = this.#references = null;
    for (const payload of writers) {
      if (finalize && !payload.wrote) await this.#writeRows(payload, []);
      if (finalize) payload.writer.close();
      else payload.writer.abort();
      if (finalize) {
        await finished(payload.output);
        const handle = await fs.open(payload.temporary, "r");
        try {
          await handle.sync();
        } finally {
          await handle.close();
        }
        await fs.rename(payload.temporary, payload.temporary.slice(0, -4));
      } else {
        payload.output.destroy();
      }
    }
    if (finalize) await syncDirectory(this.directory);
  }

  async #readRows<T>(name: string): Promise<T[]> {
    const rows: T[] = [];
    for await (const batch of this.#recordBatches(name)) {
      rows.push(...batch.toArray().map((row) => normalizeRow(row, batch.schema) as T));
    }
    return rows;
  }

  async *#recordBatches(name: string): AsyncGenerator<RecordBatch> {
    const handle = await fs.open(path.join(this.directory, name), "r");
    try {
      const reader = await RecordBatchReader.from(handle);
      for await (const batch of reader) yield batch;
    } finally {
      await handle.close().catch(() => undefined);
    }
  }

  async #batches<T extends { file_id: string }>(
    name: string,
    wantedIds: readonly string[],
    options: BatchOptions,
  ): Promise<ReplacementBatch<T>[]> {
    const maxFiles = options.maxFiles ?? COMMIT_BATCH_MAX_FILES;
    const maxRows = options.maxRows ?? COMMIT_BATCH_MAX_ROWS;
    const maxBytes = options.maxBytes ?? COMMIT_BATCH_MAX_BYTES;
    for (const [name2, value] of Object.entries({ maxFiles, maxRows, maxBytes })) {
      if (!Number.isInteger(value) || value < 1)
        throw new Error(`${name2} must be a positive integer`);
    }
    const wanted = new Set(wantedIds);
    const seen = new Set<string>();
    const groups: Array<{ fileId: string; rows: T[] }> = [];
    let currentFileId: string | null = null;
    let currentRows: T[] = [];
    for await (const batch of this.#recordBatches(name)) {
      for (const item of batch.toArray()) {
        const row = normalizeRow(item, batch.schema) as T;
        if (row.file_id !== currentFileId) {
          if (currentFileId !== null && wanted.has(currentFileId)) {
            groups.push({ fileId: currentFileId, rows: currentRows });
          }
          currentFileId = row.file_id;
          currentRows = [];
        }
        currentRows.push(row);
      }
    }
    if (currentFileId !== null && wanted.has(currentFileId)) {
      groups.push({ fileId: currentFileId, rows: currentRows });
    }
    for (const group of groups) seen.add(group.fileId);
    for (const fileId of wantedIds) {
      if (!seen.has(fileId)) groups.push({ fileId, rows: [] });
    }

    const result: ReplacementBatch<T>[] = [];
    let fileIds: string[] = [];
    let rows: T[] = [];
    let bytes = 0;
    const release = (): void => {
      if (fileIds.length > 0) result.push({ fileIds, rows });
      fileIds = [];
      rows = [];
      bytes = 0;
    };
    for (const group of groups) {
      const groupBytes =
        group.rows.length === 0
          ? 0
          : tableToBytes(
              name === CHUNKS_NAME ? this.#chunkSchema : this.#referenceSchema,
              group.rows,
            );
      if (
        fileIds.length > 0 &&
        (fileIds.length >= maxFiles ||
          rows.length + group.rows.length > maxRows ||
          bytes + groupBytes > maxBytes)
      ) {
        release();
      }
      fileIds.push(group.fileId);
      rows.push(...group.rows);
      bytes += groupBytes;
    }
    release();
    return result;
  }

  #requireWriter(writer: PayloadWriter | null): PayloadWriter {
    if (writer === null) throw new Error("staging payload is not open");
    return writer;
  }

  #requireJournal(): StagingJournal {
    if (this.#journal === null) throw new Error("staging job has not begun");
    return this.#journal;
  }

  #setTerminalPhase(phase: typeof PHASE_COMPLETE | typeof PHASE_ROLLED_BACK): void {
    this.#journal = { ...this.#requireJournal(), phase };
  }

  async #writeJournal(): Promise<void> {
    await writeAtomically(
      path.join(this.directory, JOURNAL_NAME),
      JSON.stringify(this.#requireJournal()),
    );
  }
}

export interface BatchOptions {
  maxFiles?: number;
  maxRows?: number;
  maxBytes?: number;
}

/** Roll back every interrupted commit before the store accepts queries. */
export async function recoverStagedCommits(
  stagingRoot: string,
  store: StagingStore,
): Promise<number> {
  let projectEntries: Dirent[];
  try {
    projectEntries = await fs.readdir(stagingRoot, { withFileTypes: true });
  } catch (error) {
    if (isMissing(error)) return 0;
    throw error;
  }
  let recovered = 0;
  for (const project of projectEntries.sort((left, right) => left.name.localeCompare(right.name))) {
    if (!project.isDirectory()) continue;
    const projectDirectory = path.join(stagingRoot, project.name);
    const jobs = await fs
      .readdir(projectDirectory, { withFileTypes: true })
      .catch(() => [] as Dirent[]);
    for (const job of jobs.sort((left, right) => left.name.localeCompare(right.name))) {
      if (!job.isDirectory()) continue;
      const directory = path.join(projectDirectory, job.name);
      const journalPath = path.join(directory, JOURNAL_NAME);
      let journal: StagingJournal;
      try {
        journal = parseJournal(await fs.readFile(journalPath, "utf8"));
      } catch {
        // An unreadable journal may be repaired manually; removing it could
        // discard the only record of the versions a crash recovery needs.
        continue;
      }
      if (journal.phase !== PHASE_COMMITTING) {
        await fs.rm(directory, { recursive: true, force: true });
        continue;
      }
      try {
        const version = Number(journal.version ?? LEGACY_JOURNAL_FORMAT_VERSION);
        const legacy = version === LEGACY_JOURNAL_FORMAT_VERSION;
        if (!legacy && version !== JOURNAL_FORMAT_VERSION) {
          throw new Error(`unsupported staging journal version: ${version}`);
        }
        const versions = {
          files: journalNumber(journal, "files_version"),
          chunks: journalNumber(journal, "chunks_version"),
          references: legacy ? 0 : journalNumber(journal, "references_version"),
        };
        const restored = await store.restoreVersions(journal.project_id, versions, {
          restoreReferences: !legacy,
        });
        if (!restored) {
          await fs.rm(directory, { recursive: true, force: true });
          continue;
        }
        await writeAtomically(
          journalPath,
          JSON.stringify({ ...journal, phase: PHASE_ROLLED_BACK }),
        );
        await fs.rm(directory, { recursive: true, force: true });
        recovered += 1;
      } catch {
        const attempts = recoveryAttempts(journal) + 1;
        if (attempts < MAX_RECOVERY_ATTEMPTS) {
          await writeAtomically(
            journalPath,
            JSON.stringify({ ...journal, recovery_attempts: attempts }),
          );
          continue;
        }
        await store.markProjectState(journal.project_id, "error").catch(() => undefined);
        await fs.rm(directory, { recursive: true, force: true });
      }
    }
  }
  return recovered;
}

function tableFromRows<T extends object>(schema: Schema, rows: readonly T[]): Table {
  const columns = schema.fields.map((field) =>
    vectorFromArray(
      rows.map((row) =>
        arrowValue(field.type.constructor.name, (row as Record<string, unknown>)[field.name]),
      ),
      field.type,
    ),
  );
  const data = makeData({
    type: new Struct(schema.fields),
    length: rows.length,
    nullCount: 0,
    children: columns.map((column) => column.data[0]),
  } as never);
  return new Table(new RecordBatch(schema, data as never));
}

function arrowValue(typeName: string, value: unknown): unknown {
  if (typeName === "Int64" && value !== null && value !== undefined)
    return BigInt(value as number | bigint);
  return value;
}

function tableToBytes<T extends object>(schema: Schema, rows: readonly T[]): number {
  // IPC size is a stable upper bound for the in-flight Arrow representation.
  return tableToIPC(tableFromRows(schema, rows), "file").byteLength;
}

function normalizeRow(row: unknown, schema?: Schema): Record<string, unknown> {
  const result = { ...(row as Record<string, unknown>) };
  for (const name of ["size", "indexed_at", "start_byte", "end_byte", "start_line", "end_line"]) {
    if (typeof result[name] === "bigint") result[name] = Number(result[name]);
  }
  const vectorField = schema?.fields.find((field) => field.name === "vector");
  const float16 = vectorField?.type.children[0]?.type.precision === Precision.HALF;
  let vector: number[] | null = null;
  if (ArrayBuffer.isView(result.vector)) {
    vector = Array.from(result.vector as unknown as ArrayLike<number>);
  } else if (
    typeof result.vector === "object" &&
    result.vector !== null &&
    "toArray" in result.vector &&
    typeof result.vector.toArray === "function"
  ) {
    vector = Array.from(result.vector.toArray() as ArrayLike<number>);
  }
  if (vector !== null) result.vector = float16 ? vector.map(float16ToNumber) : vector;
  return result;
}

function float16ToNumber(value: number): number {
  const sign = value & 0x8000 ? -1 : 1;
  const exponent = (value >>> 10) & 0x1f;
  const fraction = value & 0x3ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + fraction / 0x400) * 2 ** (exponent - 15);
}

function parseJournal(value: string): StagingJournal {
  const parsed: unknown = JSON.parse(value);
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("staging journal is not an object");
  }
  const journal = parsed as Partial<StagingJournal>;
  if (typeof journal.project_id !== "string" || typeof journal.phase !== "string") {
    throw new Error("staging journal is missing required fields");
  }
  return journal as StagingJournal;
}

function journalNumber(
  journal: StagingJournal,
  key: "files_version" | "chunks_version" | "references_version",
): number {
  const value = journal[key];
  if (!Number.isInteger(value) || (value as number) < 0)
    throw new Error(`invalid ${key} in staging journal`);
  return value as number;
}

function recoveryAttempts(journal: StagingJournal): number {
  return Number.isInteger(journal.recovery_attempts) && (journal.recovery_attempts ?? 0) >= 0
    ? (journal.recovery_attempts as number)
    : 0;
}

function addOnce(values: string[], value: string): void {
  if (!values.includes(value)) values.push(value);
}

function isMissing(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT";
}
