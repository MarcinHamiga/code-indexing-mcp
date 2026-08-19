/**
 * Explicit incremental indexing orchestration.
 */

import { createHash, randomUUID } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import lockfile from "proper-lockfile";
import {
  composePassage,
  embeddedSegment,
  type Embedder,
  type EmbeddedSegment,
  type PassageEmbedder,
  passageCandidate,
  packVector,
  type SegmentingEmbedder,
  type SegmentPlan,
  segmentPlan,
} from "./embedding.ts";
import type { SessionTelemetry } from "./embedding-worker.ts";
import { CodeIndexingError, isCodeIndexingError } from "./errors.ts";
import type { TreeSitterExtractor } from "./extractor.ts";
import type { FinishUpdates, HistoryStore } from "./history.ts";
import {
  type ExtractedChunk,
  type ExtractedDeclarationShape,
  type ExtractedReference,
  type IndexIssue,
  IndexReport,
  type IndexTrigger,
  type ProjectInfo,
  ReferenceBackfillReport,
  type ReferenceCoverage,
  RunAudit,
  type ScannedFile,
  type StoredFile,
} from "./models.ts";
import { ProgressPublisher } from "./progress.ts";
import { pythonJsonDumps } from "./python-compat.ts";
import { REFERENCE_SCHEMA_VERSION, type ReferenceRecord } from "./reference-store.ts";
import type { SourceScanner } from "./scanner.ts";
import { type ChunkRow, type ReferenceRow, StagingJob } from "./staging.ts";
import { LanceStore, SCHEMA_VERSION } from "./storage.ts";
import { checkoutHead } from "./update-check.ts";

export const SEGMENT_TEXT_GROWTH_LIMIT = 2;
export const CANDIDATE_GROUP_CHARS = 256 * 1024;
export const CANDIDATE_GROUP_COUNT = 256;
export { REFERENCE_SCHEMA_VERSION };
export const SERVER_VERSION = "0.0.0";

const ENVIRONMENT_ERROR_CODES = new Set([
  "MODEL_UNAVAILABLE",
  "INDEX_RESOURCE_LIMIT",
  "EMBEDDING_WORKER_FAILED",
  "BACKEND_UNAVAILABLE",
]);

export type PassageSessionFactory = () => PassageSession;

export interface PassageSession extends PassageEmbedder {
  enter?(): Promise<unknown> | unknown;
  exit?(error?: unknown): Promise<void> | void;
  close?(): Promise<void> | void;
  telemetry?(): SessionTelemetry;
}

interface PendingFile {
  record: StoredFile;
  chunks: ExtractedChunk[];
  references: ExtractedReference[];
  declarations: ExtractedDeclarationShape[];
  sourceChars: number;
  error: unknown;
  embeddedChunks: number;
  emittedChars: number;
}

interface PendingCandidate {
  owner: number;
  chunk: ExtractedChunk;
}

function* candidateGroups(candidates: readonly PendingCandidate[]): Generator<PendingCandidate[]> {
  let group: PendingCandidate[] = [];
  let characters = 0;
  for (const candidate of candidates) {
    if (
      group.length > 0 &&
      (group.length >= CANDIDATE_GROUP_COUNT ||
        characters + candidate.chunk.content.length > CANDIDATE_GROUP_CHARS)
    ) {
      yield group;
      group = [];
      characters = 0;
    }
    group.push(candidate);
    characters += candidate.chunk.content.length;
  }
  if (group.length > 0) yield group;
}

export function digest(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

export function contentRejection(source: Uint8Array): string | null {
  if (source.includes(0)) return "binary";
  try {
    new TextDecoder("utf-8", { fatal: true }).decode(source);
  } catch {
    return "encoding";
  }
  return null;
}

class PhaseTimer {
  readonly totals = new Map<string, number>();

  async measure<T>(phase: string, body: () => Promise<T> | T): Promise<T> {
    const started = process.hrtime.bigint();
    try {
      return await body();
    } finally {
      const elapsed = Number(process.hrtime.bigint() - started);
      this.totals.set(phase, (this.totals.get(phase) ?? 0) + elapsed);
    }
  }

  milliseconds(phase: string): number {
    return Math.floor((this.totals.get(phase) ?? 0) / 1_000_000);
  }
}

class IndexScanState {
  currentPaths = new Set<string>();
  indexed = 0;
  parsed = 0;
  embedded = 0;
  unchanged = 0;
  metadataOnly = 0;
  removed = 0;
  skipped = 0;
  candidatesSeen = 0;
  bytesRead = 0;
  chunksExtracted = 0;
  chunksStaged = 0;
  stagedBytes = 0;
  skippedByReason: Record<string, number> = {};
  skippedSamples: string[] = [];
  fallbackCount = 0;
  errors: IndexIssue[] = [];
  job: StagingJob | null = null;
  pending: PendingFile[] = [];
  pendingChunks = 0;
  pendingChars = 0;
  referenceExtractionNs = 0;
  stagedReferenceRows = 0;
  peakMemoryBytes = 0;

  recordSkip(filePath: string, reason: string): void {
    this.skipped += 1;
    this.skippedByReason[reason] = (this.skippedByReason[reason] ?? 0) + 1;
    if (this.skippedSamples.length < 20) this.skippedSamples.push(filePath);
  }

  addPending(pendingFile: PendingFile): void {
    this.pending.push(pendingFile);
    this.pendingChunks += pendingFile.chunks.length;
    this.pendingChars += pendingFile.sourceChars;
  }

  clearPending(): void {
    this.pending = [];
    this.pendingChunks = 0;
    this.pendingChars = 0;
  }

  sampleMemory(): void {
    try {
      this.peakMemoryBytes = Math.max(this.peakMemoryBytes, process.memoryUsage().rss);
    } catch {
      // Diagnostics must never turn a successful index into a failure.
    }
  }

  toReport(projectId: string, timer: PhaseTimer, batchSize: number): IndexReport {
    const scanMs = timer.milliseconds("scan");
    const parseMs = timer.milliseconds("parse");
    const embedMs = timer.milliseconds("embed");
    const commitMs = timer.milliseconds("commit");
    return IndexReport.parse({
      project_id: projectId,
      discovered_files: this.currentPaths.size,
      indexed_files: this.indexed,
      parsed_files: this.parsed,
      embedded_chunks: this.embedded,
      unchanged_files: this.unchanged,
      metadata_only_files: this.metadataOnly,
      removed_files: this.removed,
      skipped_files: this.skipped,
      errors: this.errors,
      scan_duration_ms: scanMs,
      parse_duration_ms: parseMs,
      embed_duration_ms: embedMs,
      commit_duration_ms: commitMs,
      embedding_backend: "cpu",
      embedding_batch_size: batchSize,
      scan_ms: scanMs,
      parse_ms: parseMs,
      embed_ms: embedMs,
      commit_ms: commitMs,
      fallback_count: this.fallbackCount,
      peak_memory_bytes: this.peakMemoryBytes,
      reference_extraction_duration_ms: Math.floor(this.referenceExtractionNs / 1_000_000),
      staged_reference_rows: this.stagedReferenceRows,
      failed_files: this.errors.length,
      skip_reasons: this.skippedByReason,
      skipped_samples: this.skippedSamples,
      bytes_read: this.bytesRead,
      chunks_extracted: this.chunksExtracted,
      chunks_staged: this.chunksStaged,
      staged_bytes: this.stagedBytes,
    });
  }
}

class RunRecord {
  readonly #history: HistoryStore | undefined;
  readonly #runId: string;
  readonly #audit: () => RunAudit;
  readonly #snapshot: () => Promise<Record<string, number>>;
  #started = false;
  #storageBefore: Record<string, number> = {};

  constructor(options: {
    history: HistoryStore | undefined;
    runId: string;
    audit: () => RunAudit;
    snapshot: () => Promise<Record<string, number>>;
  }) {
    this.#history = options.history;
    this.#runId = options.runId;
    this.#audit = options.audit;
    this.#snapshot = options.snapshot;
  }

  async start(): Promise<void> {
    if (this.#started || this.#history === undefined) return;
    this.#started = true;
    writeAudit(() => this.#history?.begin(this.#audit()));
    this.#storageBefore = await this.#snapshot();
  }

  async complete(counters: FinishUpdates): Promise<void> {
    await this.#finish("completed", counters);
  }

  async fail(): Promise<void> {
    await this.#finish("failed", {});
  }

  async #finish(state: string, counters: FinishUpdates): Promise<void> {
    if (!this.#started || this.#history === undefined) return;
    const after = await this.#snapshot();
    writeAudit(() =>
      this.#history?.finish(this.#runId, {
        state,
        finished_at: new Date().toISOString(),
        storage_before: this.#storageBefore,
        storage_after: after,
        ...counters,
      }),
    );
  }
}

function writeAudit(operation: () => void): void {
  try {
    operation();
  } catch {
    // A full disk or a locked history database costs the audit row, not the run.
  }
}

export interface IndexerOptions {
  store: LanceStore;
  scanner: SourceScanner;
  extractor: TreeSitterExtractor;
  embedder: Embedder;
  lockDirectory: string;
  batchSize?: number;
  segmentPlan?: SegmentPlan;
  passageSessionFactory?: PassageSessionFactory;
  stagingDirectory?: string;
  progressDirectory?: string;
  history?: HistoryStore;
}

export class Indexer {
  readonly store: LanceStore;
  readonly scanner: SourceScanner;
  readonly extractor: TreeSitterExtractor;
  readonly embedder: Embedder;
  readonly lockDirectory: string;
  readonly batchSize: number;
  readonly segmentPlan: SegmentPlan;
  readonly passageSessionFactory: PassageSessionFactory | undefined;
  readonly stagingDirectory: string;
  readonly progressDirectory: string;
  readonly history: HistoryStore | undefined;

  constructor(options: IndexerOptions) {
    this.store = options.store;
    this.scanner = options.scanner;
    this.extractor = options.extractor;
    this.embedder = options.embedder;
    this.lockDirectory = options.lockDirectory;
    this.batchSize = options.batchSize ?? 1;
    this.segmentPlan = options.segmentPlan ?? segmentPlan({ maxItems: this.batchSize });
    this.passageSessionFactory = options.passageSessionFactory;
    this.stagingDirectory =
      options.stagingDirectory ?? path.join(path.dirname(this.lockDirectory), "staging");
    this.progressDirectory =
      options.progressDirectory ?? path.join(path.dirname(this.lockDirectory), "progress");
    this.history = options.history;
  }

  async index(
    project: ProjectInfo,
    {
      force = false,
      waitForLock = false,
      onProgress,
      trigger = "manual",
    }: {
      force?: boolean;
      waitForLock?: boolean;
      onProgress?: (progress: import("./models.ts").IndexProgress) => void;
      trigger?: IndexTrigger;
    } = {},
  ): Promise<IndexReport> {
    const started = process.hrtime.bigint();
    const runId = randomUUID().replaceAll("-", "");
    fs.mkdirSync(this.lockDirectory, { recursive: true });
    const progress = new ProgressPublisher(project.id, {
      runId,
      trigger,
      directory: this.progressDirectory,
      listener: onProgress,
    });
    let report: IndexReport | undefined;
    try {
      await withIndexLocks(this.lockDirectory, project.id, waitForLock, async () => {
        const rebuildReason = await this.store.incompatibilityReason(
          project.id,
          this.embedder.modelId,
        );
        const effectiveTrigger = rebuildReason === null ? trigger : "schema-rebuild";
        const record = this.#recordedRun(project, runId, effectiveTrigger, {
          force,
          rebuildReason,
        });
        await record.start();
        try {
          if (rebuildReason !== null) await this.#prepareRebuild(project, rebuildReason);
          try {
            report = await this.#indexLocked(project, { force, progress });
          } finally {
            progress.clear();
          }
          report = IndexReport.parse({
            ...report,
            run_id: runId,
            trigger: effectiveTrigger,
            failed_files: report.errors.length,
          });
          await record.complete({
            phase_durations: phaseDurations(report),
            eligible_files: report.discovered_files,
            changed_files: report.indexed_files,
            unchanged_files: report.unchanged_files,
            parsed_files: report.parsed_files,
            failed_files: report.failed_files,
            removed_files: report.removed_files,
            skipped_total: report.skipped_files,
            chunks_extracted: report.chunks_extracted,
            chunks_embedded: report.embedded_chunks,
            chunks_staged: report.chunks_staged,
            staged_bytes: report.staged_bytes,
            bytes_read: report.bytes_read,
            skip_reasons: report.skip_reasons,
            errors: report.errors.slice(0, 20),
            skipped_samples: report.skipped_samples.slice(0, 20),
            embedding_backend: report.embedding_backend,
            embedding_fallback_reason: report.embedding_fallback_reason,
            worker_used: report.worker_used,
          });
        } catch (error) {
          await record.fail();
          throw error;
        }
      });
    } catch (error) {
      throw busyOrRethrow(error, project);
    }
    const durationMs = Number(process.hrtime.bigint() - started) / 1_000_000;
    if (report === undefined) throw new Error("index run produced no report");
    return IndexReport.parse({ ...report, duration_ms: Math.floor(durationMs) });
  }

  async backfillReferences(
    project: ProjectInfo,
    {
      waitForLock = false,
      onProgress,
      trigger = "reference-backfill",
    }: {
      waitForLock?: boolean;
      onProgress?: (progress: import("./models.ts").IndexProgress) => void;
      trigger?: IndexTrigger;
    } = {},
  ): Promise<ReferenceBackfillReport> {
    if ((await this.store.incompatibilityReason(project.id, this.embedder.modelId)) !== null) {
      await this.index(project, {
        waitForLock,
        trigger: "schema-rebuild",
        ...(onProgress === undefined ? {} : { onProgress }),
      });
    }
    fs.mkdirSync(this.lockDirectory, { recursive: true });
    const runId = randomUUID().replaceAll("-", "");
    const progress = new ProgressPublisher(project.id, {
      runId,
      trigger,
      directory: this.progressDirectory,
      listener: onProgress,
    });
    try {
      return await withIndexLocks(this.lockDirectory, project.id, waitForLock, async () => {
        const record = this.#recordedRun(project, runId, trigger, { deferred: true });
        try {
          let report: ReferenceBackfillReport;
          try {
            report = await this.#backfillReferencesLocked(project, progress, record);
          } finally {
            progress.clear();
          }
          await record.complete({
            eligible_files: report.files_current,
            changed_files: report.files_backfilled,
            failed_files: report.incomplete_paths.length,
            skipped_total: report.stale_paths.length,
            skip_reasons: report.stale_paths.length > 0 ? { stale: report.stale_paths.length } : {},
            errors: report.incomplete_paths
              .slice(0, 20)
              .map((filePath) => ({ path: filePath, message: "reference extraction incomplete" })),
            skipped_samples: report.stale_paths.slice(0, 20),
          });
          return report;
        } catch (error) {
          await record.fail();
          throw error;
        }
      });
    } catch (error) {
      throw busyOrRethrow(error, project);
    }
  }

  #recordedRun(
    project: ProjectInfo,
    runId: string,
    trigger: IndexTrigger,
    {
      force = false,
      rebuildReason = null,
    }: { force?: boolean; deferred?: boolean; rebuildReason?: string | null } = {},
  ): RunRecord {
    return new RunRecord({
      history: this.history,
      runId,
      audit: () =>
        RunAudit.parse({
          run_id: runId,
          project_id: project.id,
          trigger,
          server_version: SERVER_VERSION,
          git_revision: checkoutHead(repositoryRoot()),
          model_id: this.embedder.modelId,
          schema_version: SCHEMA_VERSION,
          scan_config_hash: digest(pythonJsonDumps(project.scan)).slice(0, 16),
          force,
          pid: process.pid,
          started_at: new Date().toISOString(),
          rebuild_reason: rebuildReason,
        }),
      snapshot: () => this.#storageSnapshot(project.id),
    });
  }

  async #prepareRebuild(project: ProjectInfo, reason: string): Promise<void> {
    await this.store.deletePartition(project.id, this.embedder.modelId);
    void reason;
  }

  async #storageSnapshot(projectId: string): Promise<Record<string, number>> {
    try {
      if (fs.existsSync(path.join(this.store.directory, "projects", projectId))) {
        const versions = await this.store.tableVersions(projectId);
        return {
          files: versions.files,
          chunks: versions.chunks,
          references: versions.references,
        };
      }
    } catch {
      // Best-effort; empty when there is no partition yet.
    }
    return {};
  }

  async #backfillReferencesLocked(
    project: ProjectInfo,
    progress: ProgressPublisher,
    runRecord: RunRecord,
  ): Promise<ReferenceBackfillReport> {
    let priorState: string;
    try {
      priorState = await this.store.projectState(project.id);
    } catch (error) {
      if (!isCodeIndexingError(error) || error.code !== "PROJECT_NOT_FOUND") throw error;
      priorState = "ready";
    }
    const existing = new Map(
      (await this.store.listFiles(project.id)).map((record) => [record.path, record]),
    );
    const coverageRows = await this.store.referenceCoverage(project.id);
    const coverage = new Map<string, ReferenceCoverage>();
    const staleSchemaFileIds = new Set<string>();
    for (const row of coverageRows) {
      if (row.schema_version === REFERENCE_SCHEMA_VERSION) {
        coverage.set(row.file_id, {
          file_id: row.file_id,
          path: row.path,
          content_hash: row.content_hash,
          schema_version: row.schema_version,
        });
      } else {
        staleSchemaFileIds.add(row.file_id);
      }
    }
    const missing = new Map<string, StoredFile>();
    for (const record of existing.values()) {
      const known = coverage.get(record.file_id);
      if (known === undefined || known.content_hash !== record.content_hash) {
        missing.set(record.file_id, record);
      }
    }
    if (missing.size === 0) {
      return ReferenceBackfillReport.parse({
        project_id: project.id,
        files_current: existing.size,
      });
    }

    await runRecord.start();
    progress.update(
      { phase: "extracting_references", candidates_total: missing.size },
      { force: true },
    );
    let staged: StagingJob | undefined;
    let filesChecked = 0;
    let filesBackfilled = 0;
    const incompletePaths: string[] = [];
    const stalePaths: string[] = [];
    const seenFileIds = new Set<string>();

    const stagingJob = async (): Promise<StagingJob> => {
      if (staged === undefined) {
        staged = new StagingJob(this.stagingDirectory, project.id, this.#schemas());
        await staged.begin();
      }
      return staged;
    };

    try {
      for await (const item of this.scanner.iterScan(project, existing, { readContents: false })) {
        if (!isScannedFile(item)) continue;
        const record = existing.get(item.path);
        if (record === undefined || !missing.has(record.file_id)) continue;
        seenFileIds.add(record.file_id);
        filesChecked += 1;
        progress.update({
          phase: "extracting_references",
          candidates_seen: filesChecked,
          eligible_files: existing.size,
          current_path: record.path,
        });
        if (record.has_errors) {
          if (record.error?.startsWith("rejected:")) {
            const active = await stagingJob();
            await active.stageReferences(referenceRows(project.id, record, [], []));
            active.markReferencesReplaced(record.file_id);
            filesBackfilled += 1;
            continue;
          }
          (await stagingJob()).markReferencesReplaced(record.file_id);
          incompletePaths.push(record.path);
          continue;
        }
        let source: Uint8Array;
        try {
          source = await fs.promises.readFile(item.absolute_path);
        } catch {
          stalePaths.push(record.path);
          continue;
        }
        if (contentRejection(source) !== null || digest(source) !== record.content_hash) {
          stalePaths.push(record.path);
          continue;
        }
        let extraction: ReturnType<TreeSitterExtractor["extract"]>;
        try {
          extraction = this.extractor.extract(item.path, item.language, source);
        } catch (error) {
          if (isCodeIndexingError(error)) throw error;
          if (staleSchemaFileIds.has(record.file_id)) {
            (await stagingJob()).markReferencesReplaced(record.file_id);
          }
          incompletePaths.push(record.path);
          continue;
        }
        if (extraction.has_errors) {
          (await stagingJob()).markReferencesReplaced(record.file_id);
          incompletePaths.push(record.path);
          continue;
        }
        const active = await stagingJob();
        await active.stageReferences(
          referenceRows(project.id, record, extraction.references, extraction.declarations),
        );
        active.markReferencesReplaced(record.file_id);
        filesBackfilled += 1;
      }

      for (const [fileId, record] of missing) {
        if (!seenFileIds.has(fileId)) stalePaths.push(record.path);
      }
      if (stalePaths.length > 0) {
        await staged?.discard();
        return ReferenceBackfillReport.parse({
          project_id: project.id,
          files_checked: filesChecked,
          files_backfilled: filesBackfilled,
          files_current: existing.size - missing.size,
          incomplete_paths: [...incompletePaths].sort(),
          stale_paths: [...new Set(stalePaths)].sort(),
        });
      }
      progress.update(
        {
          phase: "committing",
          candidates_seen: filesChecked,
          candidates_total: filesChecked,
          eligible_files: existing.size,
          current_path: null,
        },
        { force: true },
      );
      await this.#commitStaged(project, await stagingJob(), { errors: [], state: priorState });
      return ReferenceBackfillReport.parse({
        project_id: project.id,
        files_checked: filesChecked,
        files_backfilled: filesBackfilled,
        files_current: existing.size - incompletePaths.length,
        incomplete_paths: [...incompletePaths].sort(),
      });
    } catch (error) {
      await staged?.discard();
      throw error;
    }
  }

  async #indexLocked(
    project: ProjectInfo,
    { force, progress }: { force: boolean; progress: ProgressPublisher },
  ): Promise<IndexReport> {
    await this.store.upsertProject(project, {
      modelId: this.embedder.modelId,
      state: "indexing",
    });
    try {
      const session = this.passageSessionFactory?.();
      try {
        if (session?.enter !== undefined) await session.enter();
        const passageEmbedder = session ?? this.embedder;
        let report = await this.#indexScan(project, {
          force,
          passageEmbedder,
          progress,
        });
        if (session?.telemetry !== undefined) {
          const measured = session.telemetry();
          report = IndexReport.parse({
            ...report,
            embedding_backend: measured.backend,
            memory_budget_bytes: measured.memoryBudgetBytes,
            peak_memory_bytes: measured.peakMemoryBytes,
            worker_used: true,
            embedded_segments: measured.segmentCount,
            embedded_tokens: measured.tokenCount,
            embedding_retries: measured.retryCount,
            fallback_count: report.fallback_count + measured.fallbackCount,
            worker_termination_reason: measured.terminationReason,
            token_windowing: measured.tokenizerAvailable,
            embedding_fallback_reason: measured.fallbackReason ?? null,
            embedded_characters: measured.characterCount ?? null,
            embedding_crossover_characters: measured.crossoverCharacters ?? null,
            embedding_selection_reason: measured.selectionReason ?? null,
          });
        }
        return report;
      } finally {
        if (session?.exit !== undefined) await session.exit();
        else if (session?.close !== undefined) await session.close();
      }
    } catch (error) {
      try {
        await this.store.upsertProject(project, {
          modelId: this.embedder.modelId,
          state: "error",
        });
      } catch {
        // Never leave the project stuck in "indexing" after a crash.
      }
      throw error;
    }
  }

  #schemas() {
    return {
      fileSchema: LanceStore.fileArrowSchema(),
      chunkSchema: LanceStore.chunkArrowSchema(
        this.store.vectorDimension,
        this.store.vectorStorage,
      ),
      referenceSchema: LanceStore.referenceArrowSchema(),
    };
  }

  async #stagingJob(project: ProjectInfo, state: IndexScanState): Promise<StagingJob> {
    if (state.job === null) {
      state.job = new StagingJob(this.stagingDirectory, project.id, this.#schemas());
      await state.job.begin();
    }
    return state.job;
  }

  async #stageFileFailure(
    project: ProjectInfo,
    existing: Map<string, StoredFile>,
    state: IndexScanState,
    record: StoredFile,
    error: unknown,
  ): Promise<void> {
    state.errors.push({
      path: record.path,
      message: String(error instanceof Error ? error.message : error),
    });
    const previous = existing.get(record.path);
    await (await this.#stagingJob(project, state)).stageFile({
      ...record,
      content_hash: previous !== undefined ? previous.content_hash : record.content_hash,
      has_errors: true,
      error: String(error instanceof Error ? error.message : error),
      indexed_at: wallNanos(),
    });
  }

  async #flushPending(
    project: ProjectInfo,
    existing: Map<string, StoredFile>,
    passageEmbedder: PassageEmbedder,
    progress: ProgressPublisher,
    timer: PhaseTimer,
    state: IndexScanState,
  ): Promise<void> {
    if (state.pending.length === 0) return;
    const candidates: PendingCandidate[] = state.pending.flatMap((pendingFile, owner) =>
      pendingFile.chunks.map((chunk) => ({ owner, chunk })),
    );
    progress.update({ phase: "embedding", current_path: null }, { force: true });
    for (const group of candidateGroups(candidates)) {
      const active = group.filter((candidate) => state.pending[candidate.owner]?.error === null);
      if (active.length === 0) continue;
      let succeeded: [PendingCandidate, EmbeddedSegment[]][] = [];
      let failed: [PendingCandidate, unknown][] = [];
      let retries = 0;
      await timer.measure("embed", async () => {
        try {
          [succeeded, failed, retries] = await this.#embedCandidates(passageEmbedder, active);
        } finally {
          state.sampleMemory();
        }
      });
      state.fallbackCount += retries;
      progress.update({ phase: "embedding" });
      const stagedRows = new Map<number, ChunkRow[]>();
      for (const [candidate, segments] of succeeded) {
        const target = state.pending[candidate.owner];
        if (target === undefined || target.error !== null) continue;
        const windowed = windowedChunks([candidate.chunk], [segments]);
        const rows = windowed.map(([chunk, vector]) =>
          chunkRow(project.id, target.record, chunk, vector),
        );
        const existingRows = stagedRows.get(candidate.owner) ?? [];
        existingRows.push(...rows);
        stagedRows.set(candidate.owner, existingRows);
        target.embeddedChunks += rows.length;
        target.emittedChars += windowed.reduce((sum, [chunk]) => sum + chunk.content.length, 0);
      }
      for (const [candidate, error] of failed) {
        const target = state.pending[candidate.owner];
        if (target !== undefined && target.error === null) target.error = error;
      }
      await timer.measure("commit", async () => {
        for (const owner of [...stagedRows.keys()].sort((left, right) => left - right)) {
          const rows = stagedRows.get(owner) ?? [];
          await (await this.#stagingJob(project, state)).stageChunks(rows);
          state.chunksStaged += rows.length;
          state.stagedBytes += rows.reduce(
            (sum, row) => sum + Buffer.byteLength(row.content, "utf8"),
            0,
          );
        }
      });
    }

    await timer.measure("commit", async () => {
      for (const target of state.pending) {
        if (
          target.error === null &&
          target.emittedChars > SEGMENT_TEXT_GROWTH_LIMIT * target.sourceChars
        ) {
          target.error = new Error(
            `Token windowing emitted ${target.emittedChars} characters from ` +
              `${target.sourceChars} characters of chunk text, above the ` +
              `${SEGMENT_TEXT_GROWTH_LIMIT}x limit`,
          );
        }
        if (target.error !== null) {
          await this.#stageFileFailure(project, existing, state, target.record, target.error);
          continue;
        }
        const job = await this.#stagingJob(project, state);
        await job.stageFile(target.record);
        job.markReplaced(target.record.file_id);
        if (target.record.has_errors) {
          job.markReferencesReplaced(target.record.file_id);
        } else {
          await job.stageReferences(
            referenceRows(project.id, target.record, target.references, target.declarations),
          );
          job.markReferencesReplaced(target.record.file_id);
        }
        state.indexed += 1;
        state.embedded += target.embeddedChunks;
      }
    });
    progress.update(
      {
        phase: "embedding",
        changed_files: state.indexed,
        chunks_embedded: state.embedded,
        chunks_staged: state.chunksStaged,
        staged_bytes: state.stagedBytes,
      },
      { force: true },
    );
    state.clearPending();
  }

  async #processScannedFile(
    project: ProjectInfo,
    item: ScannedFile,
    existing: Map<string, StoredFile>,
    {
      force,
      passageEmbedder,
      progress,
      timer,
      state,
    }: {
      force: boolean;
      passageEmbedder: PassageEmbedder;
      progress: ProgressPublisher;
      timer: PhaseTimer;
      state: IndexScanState;
    },
  ): Promise<void> {
    const filePath = item.path;
    const previous = existing.get(filePath);
    if (
      !force &&
      previous !== undefined &&
      previous.size === item.size &&
      previous.mtime_ns === item.mtime_ns
    ) {
      state.unchanged += 1;
      return;
    }

    let contentHash: string | undefined;
    let record: StoredFile | null = null;
    try {
      let source: Uint8Array = new Uint8Array();
      let rejection: string | null = null;
      await timer.measure("scan", async () => {
        source = item.content ?? (await fs.promises.readFile(item.absolute_path));
        state.bytesRead += source.byteLength;
        rejection = contentRejection(source);
        contentHash = digest(source);
      });
      const hashed = contentHash ?? digest(source);
      if (rejection !== null) {
        const rejectedRecord = storedFileRecord(project, item, filePath, hashed, {
          has_errors: true,
          error: `rejected: ${rejection}`,
        });
        await timer.measure("commit", async () => {
          const job = await this.#stagingJob(project, state);
          await job.stageFile(rejectedRecord);
          if (previous !== undefined) {
            job.markReplaced(rejectedRecord.file_id);
            job.markReferencesReplaced(rejectedRecord.file_id);
          }
        });
        state.recordSkip(filePath, rejection);
        return;
      }
      if (!force && previous !== undefined && previous.content_hash === hashed) {
        await timer.measure("commit", async () => {
          await (await this.#stagingJob(project, state)).stageFile({
            ...previous,
            size: item.size,
            mtime_ns: item.mtime_ns,
          });
        });
        state.metadataOnly += 1;
        return;
      }
      record = storedFileRecord(project, item, filePath, hashed);
      const extraction = await timer.measure("parse", () =>
        this.extractor.extract(item.path, item.language, source),
      );
      state.parsed += 1;
      state.chunksExtracted += extraction.chunks.length;
      state.referenceExtractionNs += extraction.reference_extraction_ns;
      state.stagedReferenceRows += extraction.references.length;
      const sourceChars = extraction.chunks.reduce((sum, chunk) => sum + chunk.content.length, 0);
      if (
        state.pending.length > 0 &&
        (state.pending.length >= CANDIDATE_GROUP_COUNT ||
          state.pendingChunks + extraction.chunks.length > CANDIDATE_GROUP_COUNT ||
          state.pendingChars + sourceChars > CANDIDATE_GROUP_CHARS)
      ) {
        await this.#flushPending(project, existing, passageEmbedder, progress, timer, state);
      }
      state.addPending({
        record: { ...record, has_errors: extraction.has_errors },
        chunks: extraction.chunks,
        references: extraction.references,
        declarations: extraction.declarations,
        sourceChars,
        error: null,
        embeddedChunks: 0,
        emittedChars: 0,
      });
    } catch (error) {
      if (isCodeIndexingError(error) && ENVIRONMENT_ERROR_CODES.has(error.code)) throw error;
      const failedRecord =
        record ??
        storedFileRecord(
          project,
          item,
          filePath,
          contentHash !== undefined ? contentHash : (previous?.content_hash ?? ""),
        );
      await timer.measure("commit", async () => {
        await this.#stageFileFailure(project, existing, state, failedRecord, error);
      });
    }
  }

  async #indexScan(
    project: ProjectInfo,
    {
      force,
      passageEmbedder,
      progress,
    }: {
      force: boolean;
      passageEmbedder: PassageEmbedder;
      progress: ProgressPublisher;
    },
  ): Promise<IndexReport> {
    const timer = new PhaseTimer();
    const existing = await timer.measure(
      "scan",
      async () =>
        new Map((await this.store.listFiles(project.id)).map((record) => [record.path, record])),
    );
    progress.update({ phase: "scanning" }, { force: true });
    const state = new IndexScanState();
    state.sampleMemory();
    try {
      for await (const item of this.scanner.iterScan(project, existing)) {
        await timer.measure("scan", async () => undefined);
        state.candidatesSeen += 1;
        if (!isScannedFile(item)) {
          state.recordSkip(item.path, item.reason);
          progress.update({
            phase: "scanning",
            candidates_seen: state.candidatesSeen,
            skipped_total: state.skipped,
            skipped_by_reason: state.skippedByReason,
            current_path: null,
          });
          continue;
        }
        const filePath = item.path;
        state.currentPaths.add(filePath);
        progress.update({
          phase: "scanning",
          candidates_seen: state.candidatesSeen,
          eligible_files: state.currentPaths.size,
          unchanged_files: state.unchanged,
          current_path: filePath,
        });
        await this.#processScannedFile(project, item, existing, {
          force,
          passageEmbedder,
          progress,
          timer,
          state,
        });
      }

      await this.#flushPending(project, existing, passageEmbedder, progress, timer, state);
      progress.update(
        {
          phase: "committing",
          candidates_seen: state.candidatesSeen,
          candidates_total: state.candidatesSeen,
          eligible_files: state.currentPaths.size,
          unchanged_files: state.unchanged,
          parsed_files: state.parsed,
          failed_files: state.errors.length,
          bytes_read: state.bytesRead,
          chunks_extracted: state.chunksExtracted,
          skipped_total: state.skipped,
          skipped_by_reason: state.skippedByReason,
          current_path: null,
        },
        { force: true },
      );
      await timer.measure("commit", async () => {
        for (const [filePath, record] of existing) {
          if (!state.currentPaths.has(filePath)) {
            (await this.#stagingJob(project, state)).markRemoved(record.file_id);
            state.removed += 1;
          }
        }
        if (state.job !== null) {
          await this.#commitStaged(project, state.job, { errors: state.errors });
        } else {
          await this.store.upsertProject(project, {
            modelId: this.embedder.modelId,
            state: await this.#deriveIndexState(project.id, state.errors),
          });
        }
      });
    } catch (error) {
      if (state.job !== null) await state.job.discard();
      throw error;
    }
    return state.toReport(project.id, timer, this.batchSize);
  }

  async #deriveIndexState(projectId: string, errors: readonly IndexIssue[]): Promise<string> {
    if (errors.length > 0) return "partial";
    if (await this.store.hasFileErrors(projectId)) return "partial";
    return "ready";
  }

  async #commitStaged(
    project: ProjectInfo,
    job: StagingJob,
    { errors, state }: { errors: IndexIssue[]; state?: string },
  ): Promise<void> {
    const versions = await this.store.tableVersions(project.id);
    await job.beginCommit(versions);
    try {
      await this.store.replaceFilesFromArrow(project.id, {
        files: await job.filesTable(),
        chunkBatches: await job.iterChunkBatches(),
        referenceBatches: await job.iterReferenceBatches(),
        removedFileIds: job.removedFileIds,
      });
      if (
        job.replaceFileIds.length > 0 ||
        job.replaceReferenceFileIds.length > 0 ||
        job.removedFileIds.length > 0
      ) {
        await this.store.ensureIndexes(project.id);
      }
      await this.store.upsertProject(project, {
        modelId: this.embedder.modelId,
        state: state ?? (await this.#deriveIndexState(project.id, errors)),
      });
    } catch (error) {
      try {
        await this.store.restoreVersions(project.id, versions);
      } catch {
        await job.discard();
        throw error;
      }
      await job.rolledBack();
      throw error;
    }
    await job.complete();
  }

  async #embedChunks(
    passageEmbedder: PassageEmbedder,
    chunks: readonly ExtractedChunk[],
  ): Promise<EmbeddedSegment[][]> {
    if (chunks.length === 0) return [];
    if (isSegmenting(passageEmbedder)) {
      const candidates = chunks.map((chunk) =>
        passageCandidate(chunk.embedding_prefix, chunk.content),
      );
      return await passageEmbedder.planAndEmbed(candidates, this.segmentPlan);
    }
    const vectors: number[][] = [];
    const texts = chunks.map((chunk) => chunk.embedding_text);
    for (let offset = 0; offset < texts.length; offset += this.batchSize) {
      const batch = await passageEmbedder.embedPassages(
        texts.slice(offset, offset + this.batchSize),
      );
      vectors.push(...batch);
    }
    return chunks.map((chunk, index) => [
      embeddedSegment(0, chunk.content.length, 0, packVector(vectors[index] ?? [])),
    ]);
  }

  async #embedCandidates(
    passageEmbedder: PassageEmbedder,
    candidates: readonly PendingCandidate[],
  ): Promise<[[PendingCandidate, EmbeddedSegment[]][], [PendingCandidate, unknown][], number]> {
    try {
      const segments = await this.#embedChunks(
        passageEmbedder,
        candidates.map((candidate) => candidate.chunk),
      );
      return [candidates.map((candidate, index) => [candidate, segments[index] ?? []]), [], 0];
    } catch (error) {
      if (isCodeIndexingError(error) && ENVIRONMENT_ERROR_CODES.has(error.code)) throw error;
      const first = candidates[0];
      if (candidates.length === 1 && first !== undefined) return [[], [[first, error]], 0];
      const midpoint = Math.floor(candidates.length / 2);
      const [leftOk, leftFailed, leftRetries] = await this.#embedCandidates(
        passageEmbedder,
        candidates.slice(0, midpoint),
      );
      const [rightOk, rightFailed, rightRetries] = await this.#embedCandidates(
        passageEmbedder,
        candidates.slice(midpoint),
      );
      return [
        [...leftOk, ...rightOk],
        [...leftFailed, ...rightFailed],
        1 + leftRetries + rightRetries,
      ];
    }
  }
}

function isSegmenting(embedder: PassageEmbedder): embedder is PassageEmbedder & SegmentingEmbedder {
  return "planAndEmbed" in embedder && typeof embedder.planAndEmbed === "function";
}

function isScannedFile(item: ScannedFile | { path: string; reason: string }): item is ScannedFile {
  return "language" in item;
}

function storedFileRecord(
  project: ProjectInfo,
  item: ScannedFile,
  filePath: string,
  contentHash: string,
  extra: Partial<StoredFile> = {},
): StoredFile {
  return {
    file_id: digest(`${project.id}\0${filePath}`),
    project_id: project.id,
    path: filePath,
    language: item.language,
    size: item.size,
    mtime_ns: item.mtime_ns,
    content_hash: contentHash,
    has_errors: extra.has_errors ?? false,
    error: extra.error ?? null,
    indexed_at: extra.indexed_at ?? wallNanos(),
  };
}

function wallNanos(): number {
  return Date.now();
}

function phaseDurations(report: IndexReport): Record<string, number> {
  const durations: Record<string, number> = {};
  for (const phase of ["scan", "parse", "embed", "commit"] as const) {
    const value = report[`${phase}_ms`];
    if (value !== null) durations[phase] = value;
  }
  return durations;
}

function windowedChunks(
  chunks: readonly ExtractedChunk[],
  segments: readonly EmbeddedSegment[][],
): [ExtractedChunk, Uint8Array][] {
  const windowed: [ExtractedChunk, Uint8Array][] = [];
  for (let index = 0; index < chunks.length; index += 1) {
    const chunk = chunks[index];
    const planned = segments[index] ?? [];
    if (chunk === undefined || planned.length === 0) continue;
    const only = planned[0];
    if (
      planned.length === 1 &&
      only !== undefined &&
      only.startChar === 0 &&
      only.endChar >= chunk.content.length
    ) {
      windowed.push([chunk, only.vector]);
      continue;
    }
    for (const segment of planned) windowed.push([windowChunk(chunk, segment), segment.vector]);
  }
  return windowed;
}

function windowChunk(chunk: ExtractedChunk, segment: EmbeddedSegment): ExtractedChunk {
  const head = chunk.content.slice(0, segment.startChar);
  const content = chunk.content.slice(segment.startChar, segment.endChar);
  const startByte = chunk.start_byte + Buffer.byteLength(head, "utf8");
  const startLine = chunk.start_line + (head.match(/\n/g)?.length ?? 0);
  return {
    ...chunk,
    start_byte: startByte,
    end_byte: startByte + Buffer.byteLength(content, "utf8"),
    start_line: startLine,
    end_line: startLine + (content.match(/\n/g)?.length ?? 0),
    content,
    embedding_text: composePassage(chunk.embedding_prefix, content),
  };
}

export function referenceRows(
  projectId: string,
  file: StoredFile,
  references: readonly ExtractedReference[],
  declarations: readonly ExtractedDeclarationShape[],
): ReferenceRow[] {
  const identity = (
    recordKind: string,
    kind: string | null,
    startByte: number | null,
    endByte: number | null,
  ): string =>
    digest(
      [
        file.file_id,
        recordKind,
        kind ?? "",
        String(startByte ?? -1),
        String(endByte ?? -1),
        String(REFERENCE_SCHEMA_VERSION),
      ].join("\0"),
    );

  const rows: ReferenceRow[] = [];
  for (const reference of references) {
    rows.push(
      referenceRecord(projectId, file, {
        reference_id: identity(
          "reference",
          reference.kind,
          reference.start_byte,
          reference.end_byte,
        ),
        record_kind: "reference",
        kind: reference.kind,
        source_qualified_symbol: reference.source_qualified_symbol,
        written_name: reference.written_name,
        target_name: reference.target_name,
        module_path: reference.module_path,
        imported_name: reference.imported_name,
        alias: reference.alias,
        receiver_text: reference.receiver_text,
        start_byte: reference.start_byte,
        end_byte: reference.end_byte,
        start_line: reference.start_line,
        end_line: reference.end_line,
        shape_json: reference.call_shape === null ? null : pythonJsonDumps(reference.call_shape),
      }),
    );
  }
  for (const declaration of declarations) {
    rows.push(
      referenceRecord(projectId, file, {
        reference_id: identity(
          "declaration",
          declaration.kind,
          declaration.start_byte,
          declaration.end_byte,
        ),
        record_kind: "declaration",
        kind: declaration.kind,
        source_qualified_symbol: declaration.qualified_symbol,
        written_name: declaration.symbol,
        target_name: declaration.symbol,
        module_path: null,
        imported_name: null,
        alias: null,
        receiver_text: null,
        start_byte: declaration.start_byte,
        end_byte: declaration.end_byte,
        start_line: declaration.start_line,
        end_line: declaration.end_line,
        shape_json: pythonJsonDumps(declaration.parameters),
      }),
    );
  }
  rows.push(
    referenceRecord(projectId, file, {
      reference_id: identity("coverage", null, null, null),
      record_kind: "coverage",
      kind: null,
      source_qualified_symbol: null,
      written_name: null,
      target_name: null,
      module_path: null,
      imported_name: null,
      alias: null,
      receiver_text: null,
      start_byte: null,
      end_byte: null,
      start_line: null,
      end_line: null,
      shape_json: null,
    }),
  );
  return rows;
}

function referenceRecord(
  projectId: string,
  file: StoredFile,
  fields: Omit<
    ReferenceRecord,
    "file_id" | "project_id" | "path" | "language" | "content_hash" | "schema_version"
  >,
): ReferenceRow {
  return {
    ...fields,
    file_id: file.file_id,
    project_id: projectId,
    path: file.path,
    language: file.language,
    content_hash: file.content_hash,
    schema_version: REFERENCE_SCHEMA_VERSION,
  };
}

export function chunkRow(
  projectId: string,
  file: StoredFile,
  chunk: ExtractedChunk,
  vector: Uint8Array,
): ChunkRow {
  const identity = [
    file.file_id,
    file.content_hash,
    chunk.kind,
    chunk.qualified_symbol ?? "",
    String(chunk.start_byte),
    String(chunk.end_byte),
    String(chunk.part_index),
  ].join("\0");
  return {
    chunk_id: `${projectId}:${digest(identity)}`,
    file_id: file.file_id,
    path: file.path,
    language: file.language,
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
    vector: unpackFloat32(vector),
    content_hash: file.content_hash,
  };
}

export function unpackFloat32(bytes: Uint8Array): number[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values: number[] = [];
  for (let offset = 0; offset + 4 <= bytes.byteLength; offset += 4) {
    values.push(view.getFloat32(offset, true));
  }
  return values;
}

function repositoryRoot(): string {
  let directory = path.dirname(fileURLToPath(import.meta.url));
  for (;;) {
    if (fs.existsSync(path.join(directory, ".git"))) return directory;
    const parent = path.dirname(directory);
    if (parent === directory) return directory;
    directory = parent;
  }
}

async function withIndexLocks<T>(
  lockDirectory: string,
  projectId: string,
  wait: boolean,
  body: () => Promise<T>,
): Promise<T> {
  const globalPath = path.join(lockDirectory, "index-global.lock");
  const projectPath = path.join(lockDirectory, `${projectId}.lock`);
  const releaseGlobal = await acquireLock(globalPath, wait);
  try {
    const releaseProject = await acquireLock(projectPath, wait);
    try {
      return await body();
    } finally {
      await releaseProject();
    }
  } finally {
    await releaseGlobal();
  }
}

export async function acquireLock(
  lockPath: string,
  wait: boolean,
  timeoutSeconds?: number,
): Promise<() => Promise<void>> {
  fs.mkdirSync(path.dirname(lockPath), { recursive: true });
  if (!fs.existsSync(lockPath)) fs.writeFileSync(lockPath, "");
  try {
    return await lockfile.lock(lockPath, lockOptions(wait, timeoutSeconds));
  } catch (error) {
    const code =
      error !== null && typeof error === "object" && "code" in error
        ? String(error.code)
        : undefined;
    if (code === "ENOENT") {
      fs.mkdirSync(path.dirname(lockPath), { recursive: true });
      fs.writeFileSync(lockPath, "");
      try {
        return await lockfile.lock(lockPath, lockOptions(wait, timeoutSeconds));
      } catch (retryError) {
        throw lockBusy(retryError);
      }
    }
    throw lockBusy(error);
  }
}

function lockOptions(wait: boolean, timeoutSeconds?: number) {
  return {
    realpath: false as const,
    retries: wait
      ? {
          retries:
            timeoutSeconds === undefined ? 10_000 : Math.max(1, Math.ceil(timeoutSeconds * 20)),
          minTimeout: 50,
          maxTimeout: 200,
          factor: 1.2,
        }
      : 0,
    stale: 30 * 60 * 1000,
  };
}

function lockBusy(error: unknown): Error {
  const busy = new Error("INDEX_BUSY");
  busy.cause = error;
  return busy;
}

function busyOrRethrow(error: unknown, project: ProjectInfo): never {
  if (error instanceof Error && error.message === "INDEX_BUSY") {
    throw new CodeIndexingError(
      "INDEX_BUSY",
      `Another indexing job is already active: ${project.name}`,
      { project: project.id },
      { cause: error },
    );
  }
  throw error;
}
