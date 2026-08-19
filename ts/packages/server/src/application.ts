/**
 * Application services shared by MCP and CLI adapters.
 */

import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { applyEnvironment, loadEnvironment } from "./accelerator-env.ts";
import {
  type BackendDescriptor,
  type BackendSelection,
  CPU_BACKEND,
  availableExecutionProviders,
  backendFor,
  describeEnvironment,
  platformFingerprint,
  runtimeVersion,
  selectBackend,
} from "./backends.ts";
import { LIMITED_BY_MEMORY, crossoverCharacters } from "./calibration.ts";
import { type Embedder, OnnxEmbedder, segmentPlan } from "./embedding.ts";
import {
  EmbeddingWorkerSession,
  defaultLauncher,
  type WorkerConfig,
  workerConfig,
} from "./embedding-worker.ts";
import { CodeIndexingError } from "./errors.ts";
import { TreeSitterExtractor } from "./extractor.ts";
import { HistoryStore } from "./history.ts";
import { acquireLock, Indexer } from "./indexing.ts";
import {
  type CodeChunk,
  type DeclarationSelector,
  type HistoryPage,
  type IndexProgress,
  type IndexReport,
  type IndexTrigger,
  type MaintenanceProjectResult,
  MaintenanceReport,
  ModelStatus,
  type OutlineResponse,
  type ProjectInfo,
  type ProjectStatus,
  type ProjectStorageStats,
  type RefactorAnalysis,
  type RefactorOperation,
  type ReferenceBackfillReport,
  type ReferenceResponse,
  RemovalReport,
  SCAN_SKIP_REASONS,
  ScanConfig,
  ScanInspectionItem,
  ScanInspectionPage,
  type SearchResponse,
  type StorageStatus,
  type StoredFile,
  type SymbolResponse,
  type TableStorageStats,
} from "./models.ts";
import { PassageBackendSession } from "./passage-backend.ts";
import {
  type ProbeKey,
  type ProbeRecord,
  ProbeCache,
  modelArtifactFingerprint,
  probeKey,
} from "./probe-cache.ts";
import { readProgress } from "./progress.ts";
import {
  ProjectResolver,
  existingMarkerPath,
  findProjectRoot,
  initializeProject,
  projectRootIdentity,
  readProjectMarker,
  sameProjectRoot,
} from "./projects.ts";
import { pythonJsonDumps } from "./python-compat.ts";
import { ReferenceService } from "./reference-service.ts";
import { SourceScanner } from "./scanner.ts";
import { SearchService } from "./search.ts";
import { type IndexSettings, indexSettingsFromEnvironment } from "./settings.ts";
import { hasPendingRecovery, recoverStagedCommits } from "./staging.ts";
import {
  LanceStore,
  chunkIdPrefix,
  overlapWarnings,
  overlappingRegistration,
  worktreeWarnings,
} from "./storage.ts";
import { maxTokenProductFor } from "./token-batching.ts";
import { resolvePath } from "./paths.ts";

export const RECOVERY_LOCK_TIMEOUT_SECONDS = 5;
export const MAINTENANCE_CHECK_INTERVAL_MS = 24 * 60 * 60 * 1000;
export const FRESHNESS_CACHE_SECONDS = 5;
export const SCAN_INSPECTION_MAX_LIMIT = 200;
export const MAINTENANCE_TIMESTAMP_FILE = "maintenance.json";
export const MAINTENANCE_LOCK_FILE = "maintenance-schedule.lock";

const PROJECT_SHAPE_MARKERS = [
  ".git",
  "pyproject.toml",
  "setup.py",
  "setup.cfg",
  "package.json",
  "tsconfig.json",
  "jsconfig.json",
];

export interface RuntimePaths {
  readonly data: string;
  readonly cache: string;
}

export function runtimePathsFromEnvironment(
  environment: NodeJS.ProcessEnv = process.env,
): RuntimePaths {
  const data =
    environment.CODE_INDEXING_DATA_DIR ?? path.join(defaultDataHome(), "code-indexing-mcp");
  const cache =
    environment.CODE_INDEXING_CACHE_DIR ?? path.join(defaultCacheHome(), "code-indexing-mcp");
  return { data: resolvePath(data), cache: resolvePath(cache) };
}

function defaultDataHome(): string {
  if (process.platform === "darwin") {
    return path.join(os.homedir(), "Library", "Application Support");
  }
  if (process.platform === "win32") {
    return process.env.APPDATA ?? path.join(os.homedir(), "AppData", "Roaming");
  }
  return process.env.XDG_DATA_HOME ?? path.join(os.homedir(), ".local", "share");
}

function defaultCacheHome(): string {
  if (process.platform === "darwin") return path.join(os.homedir(), "Library", "Caches");
  if (process.platform === "win32") {
    return process.env.LOCALAPPDATA ?? path.join(os.homedir(), "AppData", "Local");
  }
  return process.env.XDG_CACHE_HOME ?? path.join(os.homedir(), ".cache");
}

function rate(record: ProbeRecord | undefined): number | null {
  if (record === undefined || record.charactersPerSecond <= 0) return null;
  return record.charactersPerSecond;
}

function estimateReclaimable(stats: ProjectStorageStats): number {
  return stats.tables.reduce(
    (total, table) => total + Math.max(0, table.physical_bytes - table.logical_bytes),
    0,
  );
}

function versionsRemoved(before: ProjectStorageStats, after: ProjectStorageStats): number {
  const afterByName = new Map(after.tables.map((table) => [table.name, table]));
  let removed = 0;
  for (const table of before.tables) {
    const later = afterByName.get(table.name);
    if (later !== undefined) {
      removed += Math.max(0, table.retained_version_count - later.retained_version_count);
    }
  }
  return removed;
}

function readMaintenanceTimestamp(filePath: string): Date | null {
  try {
    const payload = JSON.parse(fs.readFileSync(filePath, "utf8")) as {
      last_maintenance_at?: string;
    };
    if (typeof payload.last_maintenance_at !== "string") return null;
    const timestamp = new Date(payload.last_maintenance_at);
    if (
      Number.isNaN(timestamp.getTime()) ||
      !/[zZ]|[+-]\d\d:\d\d$/.test(payload.last_maintenance_at)
    ) {
      return null;
    }
    return timestamp;
  } catch {
    return null;
  }
}

function writeMaintenanceTimestamp(filePath: string): void {
  const payload = `${pythonJsonDumps({
    last_maintenance_at: new Date().toISOString(),
    schema_version: 1,
  })}\n`;
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, payload, "utf8");
  fs.renameSync(temporary, filePath);
}

export class Application {
  readonly paths: RuntimePaths;
  readonly cwd: string;
  readonly settings: IndexSettings;
  readonly embedder: Embedder;
  readonly store: LanceStore;
  readonly history: HistoryStore;
  readonly servingProviders: readonly string[];
  readonly acceleratorEnvironment: ReturnType<typeof loadEnvironment>;
  readonly backendSelection: BackendSelection;
  readonly probeCache: ProbeCache;
  readonly indexer: Indexer;
  readonly search: SearchService;
  readonly references: ReferenceService;
  embeddingBatchSize: number;
  batchCalibration: string;
  #probeKey: ProbeKey | undefined;
  #runtimeFallback: BackendSelection | undefined;
  #cleanFreshnessUntil = new Map<string, [number, string]>();
  readonly #ready: Promise<void>;

  constructor(
    paths: RuntimePaths,
    {
      embedder,
      cwd,
      settings,
    }: { embedder?: Embedder; cwd?: string; settings?: IndexSettings } = {},
  ) {
    this.paths = paths;
    this.cwd = resolvePath(cwd ?? process.cwd());
    this.settings = settings ?? indexSettingsFromEnvironment();
    const offline = ["1", "true", "yes"].includes(
      (process.env.CODE_INDEXING_OFFLINE ?? "").toLowerCase(),
    );
    this.embedder =
      embedder ??
      new OnnxEmbedder(path.join(paths.cache, "models"), {
        offline,
        threads: this.settings.embeddingThreads,
        enableCpuMemArena: this.settings.embeddingCpuArena,
      });
    this.store = new LanceStore(path.join(paths.data, "lancedb"), {
      vectorDimension: this.embedder.dimension,
      vectorIndex: this.settings.vectorIndex,
      vectorStorage: this.settings.vectorStorage,
    });
    this.history = new HistoryStore(path.join(paths.data, "history"));
    this.history.markInterrupted();
    const lockDirectory = path.join(paths.data, "locks");
    fs.mkdirSync(lockDirectory, { recursive: true });
    const ready = this.#recover(lockDirectory);
    // An application that is dropped before any method awaits recovery (a CLI
    // one-shot, a fast test) must not leave a floating rejection behind; the
    // first caller to await `#ready` still observes a failure.
    ready.catch(() => undefined);
    this.#ready = ready;
    this.servingProviders = availableExecutionProviders();
    this.acceleratorEnvironment = loadEnvironment(paths.data);
    this.backendSelection = this.#selectBackend();
    this.probeCache = new ProbeCache(path.join(paths.cache, "backend-probes.json"));
    this.embeddingBatchSize = this.settings.embeddingBatchSize;
    this.batchCalibration = "explicit";
    if (this.settings.embeddingBatchAuto) {
      this.batchCalibration = "default";
      if (this.backendSelection.usesAccelerator) {
        this.#probeKey = this.#buildProbeKey(this.embedder);
        const cached = this.probeCache.load(this.#probeKey);
        if (cached !== undefined && cached.batchSize > 0) {
          this.embeddingBatchSize = cached.batchSize;
          this.batchCalibration = cached.limitedBy ? "reduced" : "measured";
        }
      }
    }
    const passageSessionFactory =
      this.embedder instanceof OnnxEmbedder && this.settings.indexExecution === "worker"
        ? this.#passageSessionFactory(this.embedder)
        : undefined;
    this.indexer = new Indexer({
      store: this.store,
      scanner: new SourceScanner(),
      extractor: new TreeSitterExtractor(),
      embedder: this.embedder,
      lockDirectory,
      batchSize: this.embeddingBatchSize,
      segmentPlan: segmentPlan({
        maxTokens: this.settings.embeddingMaxTokens,
        overlapTokens: this.settings.embeddingOverlapTokens,
        maxItems: this.embeddingBatchSize,
        maxTokenProduct: maxTokenProductFor(this.settings.indexMemoryBytes, {
          maxTokens: this.settings.embeddingMaxTokens,
        }),
      }),
      ...(passageSessionFactory === undefined ? {} : { passageSessionFactory }),
      stagingDirectory: path.join(paths.data, "staging"),
      progressDirectory: path.join(paths.data, "progress"),
      history: this.history,
    });
    this.search = new SearchService(this.store, this.embedder);
    this.references = new ReferenceService(this.store);
  }

  static fromEnvironment({ cwd }: { cwd?: string } = {}): Application {
    return new Application(runtimePathsFromEnvironment(), {
      ...(cwd === undefined ? {} : { cwd }),
    });
  }

  async #recover(lockDirectory: string): Promise<void> {
    try {
      const release = await acquireLock(
        path.join(lockDirectory, "index-global.lock"),
        true,
        RECOVERY_LOCK_TIMEOUT_SECONDS,
      );
      try {
        await recoverStagedCommits(path.join(this.paths.data, "staging"), this.store);
      } finally {
        await release();
      }
    } catch {
      // A run in flight owns the lock; anything left by an older crash is
      // picked up by the next start that finds it free.
    }
  }

  #selectBackend(): BackendSelection {
    const record = this.acceleratorEnvironment.environment;
    const providers = [...this.servingProviders];
    if (record !== null) {
      const prepared = backendFor(record.accelerator);
      if (prepared !== undefined && record.providers.includes(prepared.provider)) {
        providers.push(prepared.provider);
      }
    }
    let selection = selectBackend(this.settings.embeddingAccelerator, {
      availableProviders: providers,
    });
    if (record !== null && selection.usesAccelerator) {
      selection = selection.describedAs(applyEnvironment(selection.descriptor, record));
    }
    const rejection = this.acceleratorEnvironment.reason;
    if (rejection !== null && !selection.usesAccelerator) {
      selection = selection.diagnosed(rejection);
    }
    return selection;
  }

  #runsExternally(_descriptor: BackendDescriptor): boolean {
    // Node ONNX providers share the server's runtime; workers are isolated for
    // memory accounting, not because they need a second dependency environment.
    return false;
  }

  get effectiveBackendSelection(): BackendSelection {
    return this.#runtimeFallback ?? this.backendSelection;
  }

  #rememberFallback(degraded: BackendSelection): void {
    this.#runtimeFallback = degraded;
  }

  #buildProbeKey(embedder: Embedder): ProbeKey {
    const descriptor = this.backendSelection.descriptor;
    const cacheDirectory =
      "cacheDirectory" in embedder && typeof embedder.cacheDirectory === "string"
        ? embedder.cacheDirectory
        : path.join(this.paths.cache, "models");
    return probeKey({
      modelId: embedder.modelId,
      modelArtifact: modelArtifactFingerprint(cacheDirectory, embedder.modelId),
      accelerator: descriptor.accelerator,
      provider: descriptor.provider,
      runtimeVersion: descriptor.runtimeVersion || runtimeVersion(descriptor.runtime),
      platform: platformFingerprint(),
      device: descriptor.device,
      driverVersion: descriptor.driverVersion,
    });
  }

  #cpuProbeKey(): ProbeKey {
    const cacheDirectory =
      "cacheDirectory" in this.embedder && typeof this.embedder.cacheDirectory === "string"
        ? this.embedder.cacheDirectory
        : path.join(this.paths.cache, "models");
    return probeKey({
      modelId: this.embedder.modelId,
      modelArtifact: modelArtifactFingerprint(cacheDirectory, this.embedder.modelId),
      accelerator: CPU_BACKEND.accelerator,
      provider: CPU_BACKEND.provider,
      runtimeVersion: runtimeVersion(CPU_BACKEND.runtime),
      platform: platformFingerprint(),
      device: CPU_BACKEND.device,
    });
  }

  #cpuMaxItems(): number {
    if (!this.settings.embeddingBatchAuto) return 0;
    return this.probeCache.load(this.#cpuProbeKey())?.batchSize ?? 0;
  }

  #measurements(): [ProbeRecord | undefined, ProbeRecord | undefined] {
    const selection = this.effectiveBackendSelection;
    const cpu = this.probeCache.load(this.#cpuProbeKey());
    if (!selection.usesAccelerator) return [cpu, undefined];
    const key = this.#probeKey ?? this.#buildProbeKey(this.embedder);
    return [cpu, this.probeCache.load(key)];
  }

  crossoverCharacters(): number | null {
    if (this.settings.embeddingStrict) return 0;
    if (!this.settings.embeddingCrossoverAuto) return this.settings.embeddingCrossoverCharacters;
    const [cpu, accelerator] = this.#measurements();
    if (cpu === undefined || accelerator === undefined) return 0;
    return this.#measuredCrossover();
  }

  #measuredCrossover(): number | null {
    const [cpu, accelerator] = this.#measurements();
    if (cpu === undefined || accelerator === undefined) return null;
    return crossoverCharacters({
      acceleratorLoadNs: accelerator.loadNs,
      cpuLoadNs: cpu.loadNs,
      cpuCharactersPerSecond: cpu.charactersPerSecond,
      acceleratorCharactersPerSecond: accelerator.charactersPerSecond,
    });
  }

  #recommendedOverride(
    cpu: ProbeRecord | undefined,
    accelerator: ProbeRecord | undefined,
  ): string | null {
    if (accelerator !== undefined && accelerator.limitedBy === LIMITED_BY_MEMORY) {
      return (
        "CODE_INDEXING_EMBED_MEMORY_MB (a batch overran the ceiling and was reduced to " +
        `${accelerator.batchSize})`
      );
    }
    if (
      cpu !== undefined &&
      accelerator !== undefined &&
      cpu.charactersPerSecond > 0 &&
      accelerator.charactersPerSecond > 0 &&
      accelerator.charactersPerSecond <= cpu.charactersPerSecond
    ) {
      return (
        "CODE_INDEXING_EMBED_ACCELERATOR=cpu (the accelerator measured no faster than CPU " +
        "on this machine)"
      );
    }
    return null;
  }

  #passageSessionFactory(embedder: OnnxEmbedder): () => PassageBackendSession {
    const ceilingBytes = this.settings.indexMemoryBytes;
    const strict = this.settings.embeddingStrict;
    const key = this.#probeKey;
    const makeConfig = (providers: readonly string[], accelerator: string): WorkerConfig =>
      workerConfig({
        cacheDirectory: embedder.cacheDirectory,
        offline: embedder.offline,
        threads: this.settings.embeddingThreads,
        enableCpuMemArena: this.settings.embeddingCpuArena,
        dimension: embedder.dimension,
        modelId: embedder.modelId,
        providers,
        accelerator,
      });
    const descriptor = this.backendSelection.descriptor;
    const acceleratorConfig = makeConfig(descriptor.providers, descriptor.accelerator);
    const cpuConfig = makeConfig(CPU_BACKEND.providers, CPU_BACKEND.accelerator);
    return () =>
      new PassageBackendSession(this.effectiveBackendSelection, {
        acceleratorFactory: () =>
          new EmbeddingWorkerSession(acceleratorConfig, {
            configuredCeilingBytes: ceilingBytes,
            launcher: defaultLauncher(),
          }),
        cpuFactory: () =>
          new EmbeddingWorkerSession(cpuConfig, {
            configuredCeilingBytes: ceilingBytes,
            launcher: defaultLauncher(),
          }),
        strict,
        probeCache: this.probeCache,
        ...(key === undefined ? {} : { probeKey: key }),
        cpuProbeKey: this.#cpuProbeKey(),
        calibratedBatchSize: 0,
        dimension: embedder.dimension,
        onDegrade: (degraded) => this.#rememberFallback(degraded),
        crossoverCharacters: this.crossoverCharacters(),
        ...(this.settings.embeddingCalibrate ? { calibrationPlan: this.indexer.segmentPlan } : {}),
        cpuMaxItems: this.#cpuMaxItems(),
      });
  }

  async modelStatus(): Promise<ModelStatus> {
    await this.#ready;
    const selection = this.effectiveBackendSelection;
    const descriptor = describeEnvironment(selection.descriptor);
    const probeState = selection.usesAccelerator
      ? this.probeCache.state(this.#probeKey ?? this.#buildProbeKey(this.embedder))
      : "not-applicable";
    const record = this.acceleratorEnvironment.environment;
    const external = selection.usesAccelerator && this.#runsExternally(descriptor);
    const [cpu, accelerator] = this.#measurements();
    return ModelStatus.parse({
      embedding_model: this.embedder.modelId,
      dimension: this.embedder.dimension,
      requested_accelerator: selection.requested,
      resolved_accelerator: descriptor.accelerator,
      device: descriptor.device,
      execution_provider: descriptor.provider,
      available_providers: [...selection.availableProviders],
      stability: descriptor.stability,
      precision: descriptor.precision,
      runtime_version: descriptor.runtimeVersion,
      driver_version: descriptor.driverVersion,
      accelerator_environment: external && record !== null ? record.interpreter : null,
      accelerator_prepared: record === null ? null : record.accelerator,
      batch_size: this.embeddingBatchSize,
      batch_calibration: this.batchCalibration,
      probe_cache_state: probeState,
      strict: this.settings.embeddingStrict,
      fallback_reason: selection.fallbackReason,
      cpu_characters_per_second: rate(cpu),
      accelerator_characters_per_second: rate(accelerator),
      accelerator_load_ms:
        accelerator === undefined ? null : Math.floor(accelerator.loadNs / 1_000_000),
      crossover_characters: this.#measuredCrossover(),
      recommended_override: this.#recommendedOverride(cpu, accelerator),
    });
  }

  async initProject(
    projectPath?: string | null,
    {
      name,
      forceNewId = false,
      allowOverlap = false,
      roots,
    }: {
      name?: string;
      forceNewId?: boolean;
      allowOverlap?: boolean;
      roots?: readonly string[];
    } = {},
  ): Promise<ProjectInfo> {
    await this.#ready;
    let target = projectPath;
    if (target === undefined && roots !== undefined && roots.length > 0) {
      const unique: string[] = [];
      for (const root of roots) {
        if (!unique.some((existing) => sameProjectRoot(root, existing))) unique.push(root);
      }
      if (unique.length > 1) {
        throw new CodeIndexingError(
          "AMBIGUOUS_PROJECT",
          "Multiple MCP roots are available; provide an explicit path",
        );
      }
      target = unique[0];
    }
    const root = target ?? this.cwd;
    const releaseRoot = await this.#rootLock(root);
    try {
      const releaseRegistration = await this.#registrationLock();
      try {
        const resolved = resolvePath(root);
        if (!allowOverlap && !forceNewId) {
          const existing = overlappingRegistration(await this.store.listProjects(), resolved);
          if (existing !== null) {
            const marker =
              existingMarkerPath(resolved) !== null ? readProjectMarker(resolved) : null;
            if (marker === null || marker.id !== existing.id) {
              throw new CodeIndexingError(
                "OVERLAPPING_PROJECT",
                `Project root ${resolved} overlaps the registered root ` +
                  `${existing.root} of project ${JSON.stringify(existing.id)}; pass ` +
                  "allow_overlap=true to register it anyway",
                {
                  existing_project: existing.id,
                  new_project: marker === null ? null : marker.id,
                },
              );
            }
          }
        }
        const project = initializeProject(root, { name, forceNewId });
        await this.#registerProject(project);
        this.invalidateFreshness(project.id);
        return project;
      } finally {
        await releaseRegistration();
      }
    } finally {
      await releaseRoot();
    }
  }

  async discoverProject(root: string): Promise<ProjectInfo | null> {
    await this.#ready;
    const resolved = resolvePath(root);
    if (!fs.existsSync(resolved) || !fs.statSync(resolved).isDirectory()) return null;
    const release = await this.#rootLock(resolved);
    try {
      const markerRoot = findProjectRoot(resolved);
      let project: ProjectInfo;
      if (markerRoot !== null) {
        project = readProjectMarker(markerRoot);
      } else {
        if (!(await this.#isProjectShaped(resolved))) return null;
        project = initializeProject(resolved);
      }
      await this.#registerProject(project);
      this.invalidateFreshness(project.id);
      return project;
    } finally {
      await release();
    }
  }

  async indexProject(
    project?: string | null,
    {
      roots,
      force = false,
      waitForLock = false,
      onProgress,
      trigger = "manual",
    }: {
      roots?: readonly string[];
      force?: boolean;
      waitForLock?: boolean;
      onProgress?: (progress: IndexProgress) => void;
      trigger?: IndexTrigger;
    } = {},
  ): Promise<IndexReport> {
    await this.#ready;
    const resolved = await this.#resolve(project, roots);
    try {
      return await this.indexer.index(resolved, {
        force,
        waitForLock,
        trigger,
        ...(onProgress === undefined ? {} : { onProgress }),
      });
    } finally {
      this.invalidateFreshness(resolved.id);
    }
  }

  indexProgress(projectId: string): IndexProgress | null {
    return readProgress(path.join(this.paths.data, "progress"), projectId);
  }

  async ensureReferenceIndex(
    project?: string | null,
    { roots }: { roots?: readonly string[] } = {},
  ): Promise<ReferenceBackfillReport> {
    await this.#ready;
    const resolved = await this.#resolve(project, roots);
    if (await this.#projectIsStale(resolved)) {
      await this.indexer.index(resolved, { waitForLock: true, trigger: "lazy-query" });
    }
    let report = await this.indexer.backfillReferences(resolved, {
      waitForLock: true,
      trigger: "reference-backfill",
    });
    if (report.stale_paths.length > 0) {
      await this.indexer.index(resolved, { waitForLock: true, trigger: "lazy-query" });
      report = await this.indexer.backfillReferences(resolved, {
        waitForLock: true,
        trigger: "reference-backfill",
      });
    }
    this.invalidateFreshness(resolved.id);
    return report;
  }

  async projectStatus(
    project?: string | null,
    { roots }: { roots?: readonly string[] } = {},
  ): Promise<ProjectStatus> {
    await this.#ready;
    const resolved = await this.#resolve(project, roots);
    const files = await this.store.listFiles(resolved.id);
    let state = await this.store.projectState(resolved.id);
    if (state === "ready" || state === "partial") {
      const fingerprint = pythonJsonDumps(resolved.scan);
      const cached = this.#cleanFreshnessUntil.get(resolved.id);
      if (cached !== undefined && cached[1] === fingerprint && cached[0] > nowSeconds()) {
        // recent clean answer
      } else if (
        await this.#projectIsStale(resolved, new Map(files.map((record) => [record.path, record])))
      ) {
        this.#cleanFreshnessUntil.delete(resolved.id);
        state = "stale";
      } else {
        this.#cleanFreshnessUntil.set(resolved.id, [
          nowSeconds() + FRESHNESS_CACHE_SECONDS,
          fingerprint,
        ]);
      }
    }
    return {
      project: resolved,
      state,
      file_count: files.length,
      chunk_count: await this.store.countChunks([resolved.id]),
      progress: this.indexProgress(resolved.id),
      last_run: this.history.recent(resolved.id),
    };
  }

  invalidateFreshness(projectId: string): void {
    this.#cleanFreshnessUntil.delete(projectId);
  }

  async indexHistory(
    project?: string | null,
    {
      roots,
      cursor = null,
      limit = 20,
    }: { roots?: readonly string[]; cursor?: string | null; limit?: number } = {},
  ): Promise<HistoryPage> {
    await this.#ready;
    const resolved = await this.#resolve(project, roots);
    if (limit < 1) {
      throw new CodeIndexingError("INVALID_FILTER", "history limit must be at least 1");
    }
    try {
      return this.history.listRuns(resolved.id, { cursor, limit, project: resolved });
    } catch (error) {
      if (error instanceof Error && error.message.includes("cursor")) {
        throw new CodeIndexingError(
          "INVALID_CURSOR",
          "invalid history cursor",
          {},
          { cause: error },
        );
      }
      throw error;
    }
  }

  async storageStatus(
    project?: string | null,
    { roots }: { roots?: readonly string[] } = {},
  ): Promise<StorageStatus> {
    await this.#ready;
    const snapshotAt = new Date().toISOString();
    const registryBefore = await this.store.registryStats();
    const registered = await this.listProjects();
    const projects =
      project !== undefined && project !== null
        ? [await this.store.storageStatsFor(await this.#resolve(project, roots))]
        : await Promise.all(registered.map((item) => this.store.storageStatsFor(item)));
    const registryAfter = await this.store.registryStats();
    return {
      schema_version: 1,
      snapshot_at: snapshotAt,
      registry: registryAfter,
      projects,
      physical_bytes_total:
        registryAfter.physical_bytes +
        projects.reduce((total, stats) => total + stats.partition_physical_bytes, 0),
      consistent:
        registryBefore.current_version === registryAfter.current_version &&
        projects.every((stats) => stats.consistent),
      overlap_warnings: overlapWarnings(registered),
      worktree_warnings: worktreeWarnings(registered),
    };
  }

  async maintainStorage(
    project?: string | null,
    {
      roots,
      dryRun = false,
      waitForLock = false,
      trigger = "manual",
    }: {
      roots?: readonly string[];
      dryRun?: boolean;
      waitForLock?: boolean;
      trigger?: string;
    } = {},
  ): Promise<MaintenanceReport> {
    await this.#ready;
    const started = process.hrtime.bigint();
    const startedAt = new Date().toISOString();
    const retention = new Date(Date.now() - this.settings.versionRetentionHours * 60 * 60 * 1000);
    const registered = await this.listProjects();
    const scope =
      project !== undefined && project !== null
        ? [await this.#resolve(project, roots)]
        : registered;
    const lockDirectory = path.join(this.paths.data, "locks");
    fs.mkdirSync(lockDirectory, { recursive: true });
    const results: MaintenanceProjectResult[] = [];
    for (const registeredProject of scope) {
      let before: ProjectStorageStats | null = null;
      let estimate = 0;
      if (dryRun) {
        try {
          before = await this.store.storageStatsFor(registeredProject);
          estimate = estimateReclaimable(before);
          results.push(
            before.partition_open_failed
              ? {
                  project: registeredProject,
                  before,
                  status: "error",
                  skip_reason: null,
                  error: "Partition exists but its tables could not be opened",
                  after: null,
                  versions_removed: 0,
                  bytes_reclaimed: 0,
                  reclaimable_bytes_estimate: estimate,
                }
              : {
                  project: registeredProject,
                  before,
                  status: "skipped",
                  skip_reason: before.tables.length === 0 ? "not-indexed" : "dry-run",
                  error: null,
                  after: null,
                  versions_removed: 0,
                  bytes_reclaimed: 0,
                  reclaimable_bytes_estimate: estimate,
                },
          );
        } catch (error) {
          results.push({
            project: registeredProject,
            status: "error",
            skip_reason: null,
            error: formatError(error),
            before: null,
            after: null,
            versions_removed: 0,
            bytes_reclaimed: 0,
            reclaimable_bytes_estimate: 0,
          });
        }
        continue;
      }
      let releaseGlobal: (() => Promise<void>) | undefined;
      let releaseProject: (() => Promise<void>) | undefined;
      try {
        try {
          releaseGlobal = await acquireLock(
            path.join(lockDirectory, "index-global.lock"),
            waitForLock,
          );
        } catch {
          results.push(skipped(registeredProject, "busy"));
          continue;
        }
        try {
          releaseProject = await acquireLock(
            path.join(lockDirectory, `${registeredProject.id}.lock`),
            waitForLock,
          );
        } catch {
          results.push(skipped(registeredProject, "busy"));
          continue;
        }
        if (await hasPendingRecovery(path.join(this.paths.data, "staging"), registeredProject.id)) {
          results.push(skipped(registeredProject, "recovery-pending"));
          continue;
        }
        before = await this.store.storageStatsFor(registeredProject);
        estimate = estimateReclaimable(before);
        if (before.partition_open_failed) {
          results.push({
            project: registeredProject,
            before,
            status: "error",
            skip_reason: null,
            error: "Partition exists but its tables could not be opened",
            after: null,
            versions_removed: 0,
            bytes_reclaimed: 0,
            reclaimable_bytes_estimate: estimate,
          });
          continue;
        }
        if (before.tables.length === 0) {
          results.push({
            ...skipped(registeredProject, "not-indexed"),
            before,
            reclaimable_bytes_estimate: estimate,
          });
          continue;
        }
        await this.store.maintainProject(registeredProject.id, retention);
        const after = await this.store.storageStatsFor(registeredProject);
        if (after.partition_open_failed || after.tables.length === 0) {
          throw new Error("Partition became unreadable during maintenance");
        }
        results.push({
          project: registeredProject,
          before,
          after,
          status: "ok",
          skip_reason: null,
          error: null,
          versions_removed: versionsRemoved(before, after),
          bytes_reclaimed: Math.max(
            0,
            before.partition_physical_bytes - after.partition_physical_bytes,
          ),
          reclaimable_bytes_estimate: estimate,
        });
      } catch (error) {
        results.push({
          project: registeredProject,
          before,
          after: null,
          status: "error",
          skip_reason: null,
          error: formatError(error),
          versions_removed: 0,
          bytes_reclaimed: 0,
          reclaimable_bytes_estimate: estimate,
        });
      } finally {
        if (releaseProject !== undefined) await releaseProject();
        if (releaseGlobal !== undefined) await releaseGlobal();
      }
    }

    let registryBefore: TableStorageStats | null = null;
    let registryAfter: TableStorageStats | null = null;
    let registryStatus = "skipped";
    let registrySkipReason: string | null = dryRun ? "dry-run" : null;
    let registryError: string | null = null;
    let registryVersionsRemoved = 0;
    let registryBytesReclaimed = 0;
    if (dryRun) {
      registryBefore = await this.store.registryStats();
    } else {
      let releaseGlobal: (() => Promise<void>) | undefined;
      try {
        try {
          releaseGlobal = await acquireLock(
            path.join(lockDirectory, "index-global.lock"),
            waitForLock,
          );
        } catch {
          registrySkipReason = "busy";
        }
        if (releaseGlobal !== undefined) {
          registryBefore = await this.store.registryStats();
          await this.store.maintainRegistry(retention);
          registryAfter = await this.store.registryStats();
          registryStatus = "ok";
          registryVersionsRemoved = Math.max(
            0,
            registryBefore.retained_version_count - registryAfter.retained_version_count,
          );
          registryBytesReclaimed = Math.max(
            0,
            registryBefore.physical_bytes - registryAfter.physical_bytes,
          );
        }
      } catch (error) {
        registryStatus = "error";
        registryError = formatError(error);
      } finally {
        if (releaseGlobal !== undefined) await releaseGlobal();
      }
    }
    return MaintenanceReport.parse({
      trigger,
      dry_run: dryRun,
      retention_hours: this.settings.versionRetentionHours,
      started_at: startedAt,
      finished_at: new Date().toISOString(),
      duration_ms: Math.floor(Number(process.hrtime.bigint() - started) / 1_000_000),
      projects: results,
      registry_before: registryBefore,
      registry_after: registryAfter,
      registry_status: registryStatus,
      registry_skip_reason: registrySkipReason,
      registry_error: registryError,
      registry_versions_removed: registryVersionsRemoved,
      registry_bytes_reclaimed: registryBytesReclaimed,
      versions_removed_total: results.reduce((total, result) => total + result.versions_removed, 0),
      bytes_reclaimed_total: results.reduce((total, result) => total + result.bytes_reclaimed, 0),
      reclaimable_bytes_estimate_total: results.reduce(
        (total, result) => total + result.reclaimable_bytes_estimate,
        0,
      ),
      skipped_projects: results
        .filter((result) => result.status === "skipped" && result.skip_reason !== "busy")
        .map((result) => result.project.id),
      busy_projects: results
        .filter((result) => result.status === "skipped" && result.skip_reason === "busy")
        .map((result) => result.project.id),
      failed_projects: results
        .filter((result) => result.status === "error")
        .map((result) => result.project.id),
    });
  }

  async maybeRunMaintenance(): Promise<MaintenanceReport | null> {
    await this.#ready;
    if (!this.settings.autoMaintenance) return null;
    const timestampPath = path.join(this.paths.data, MAINTENANCE_TIMESTAMP_FILE);
    const lockDirectory = path.join(this.paths.data, "locks");
    fs.mkdirSync(lockDirectory, { recursive: true });
    let release: (() => Promise<void>) | undefined;
    try {
      release = await acquireLock(path.join(lockDirectory, MAINTENANCE_LOCK_FILE), false);
    } catch {
      return null;
    }
    try {
      const last = readMaintenanceTimestamp(timestampPath);
      if (last !== null && Date.now() - last.getTime() < MAINTENANCE_CHECK_INTERVAL_MS) {
        return null;
      }
      const report = await this.maintainStorage(undefined, {
        dryRun: false,
        waitForLock: false,
        trigger: "scheduled",
      });
      const projectsComplete = report.projects.every(
        (result) =>
          result.status === "ok" ||
          (result.status === "skipped" && result.skip_reason === "not-indexed"),
      );
      if (report.registry_status === "ok" && projectsComplete) {
        writeMaintenanceTimestamp(timestampPath);
      }
      return report;
    } finally {
      if (release !== undefined) await release();
    }
  }

  async projectIsStale(
    project?: string | null,
    { roots }: { roots?: readonly string[] } = {},
  ): Promise<boolean> {
    await this.#ready;
    return this.#projectIsStale(await this.#resolve(project, roots));
  }

  async #projectIsStale(
    project: ProjectInfo,
    existing?: Map<string, StoredFile>,
  ): Promise<boolean> {
    const known =
      existing ??
      new Map((await this.store.listFiles(project.id)).map((record) => [record.path, record]));
    const current = new Map<string, { size: number; mtime_ns: bigint; language: string }>();
    for await (const item of this.indexer.scanner.iterScan(project, known, {
      readContents: false,
    })) {
      if ("language" in item) current.set(item.path, item);
    }
    if (current.size !== known.size) return true;
    for (const key of known.keys()) {
      if (!current.has(key)) return true;
    }
    for (const [filePath, item] of current) {
      const stored = known.get(filePath);
      if (
        stored === undefined ||
        item.size !== stored.size ||
        item.mtime_ns !== stored.mtime_ns ||
        item.language !== stored.language
      ) {
        return true;
      }
    }
    return false;
  }

  async inspectScan(
    project?: string | null,
    {
      roots,
      outcome = null,
      reason = null,
      cursor = null,
      limit = 50,
    }: {
      roots?: readonly string[];
      outcome?: string | null;
      reason?: string | null;
      cursor?: string | null;
      limit?: number;
    } = {},
  ): Promise<ScanInspectionPage> {
    await this.#ready;
    const resolved = await this.#resolve(project, roots);
    if (limit < 1 || limit > SCAN_INSPECTION_MAX_LIMIT) {
      throw new CodeIndexingError(
        "INVALID_FILTER",
        `scan limit must be between 1 and ${SCAN_INSPECTION_MAX_LIMIT}`,
      );
    }
    if (outcome !== null && outcome !== "eligible" && outcome !== "skipped") {
      throw new CodeIndexingError("INVALID_FILTER", "scan outcome must be 'eligible' or 'skipped'");
    }
    if (reason !== null && !SCAN_SKIP_REASONS.has(reason)) {
      throw new CodeIndexingError("INVALID_FILTER", `unknown scan skip reason: ${reason}`);
    }
    let skip = 0;
    if (cursor !== null) {
      skip = Number(cursor);
      if (!Number.isInteger(skip) || skip < 0) {
        throw new CodeIndexingError("INVALID_CURSOR", "invalid scan cursor");
      }
    }
    const items: ScanInspectionItem[] = [];
    let nextCursor: string | null = null;
    let matched = 0;
    for await (const item of this.indexer.scanner.iterScan(resolved, undefined, {
      readContents: false,
    })) {
      const scanned = "language" in item;
      if (outcome === "eligible" && !scanned) continue;
      if (outcome === "skipped" && scanned) continue;
      if (reason !== null && (scanned || item.reason !== reason)) continue;
      matched += 1;
      if (matched <= skip) continue;
      if (items.length >= limit) {
        nextCursor = String(skip + items.length);
        break;
      }
      items.push(
        scanned
          ? ScanInspectionItem.parse({
              path: item.path,
              outcome: "eligible",
              language: item.language,
              size: item.size,
              mtime_ns: item.mtime_ns,
            })
          : ScanInspectionItem.parse({
              path: item.path,
              outcome: "skipped",
              reason: item.reason,
              detail: item.detail,
            }),
      );
    }
    return ScanInspectionPage.parse({ project: resolved, items, next_cursor: nextCursor });
  }

  async listProjects(): Promise<ProjectInfo[]> {
    await this.#ready;
    return [...(await this.store.listProjects())].sort((left, right) => {
      if (left.name !== right.name) return left.name < right.name ? -1 : 1;
      return left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
    });
  }

  async removeProject(project: string): Promise<RemovalReport> {
    await this.#ready;
    const resolved = await this.#resolve(project, []);
    this.#cleanFreshnessUntil.delete(resolved.id);
    return RemovalReport.parse({
      project_id: resolved.id,
      removed: await this.store.removeProject(resolved.id),
    });
  }

  async searchCode(
    query: string,
    {
      projects,
      allProjects = false,
      languages,
      paths,
      kinds,
      limit = 8,
      roots,
    }: {
      projects?: readonly string[];
      allProjects?: boolean;
      languages?: readonly string[];
      paths?: readonly string[];
      kinds?: readonly string[];
      limit?: number;
      roots?: readonly string[];
    } = {},
  ): Promise<SearchResponse> {
    await this.#ready;
    const projectIds = await this.resolveSearchScope(projects, allProjects, roots);
    await this.#ensureQueryGenerations(projectIds, roots);
    return this.search.searchCode(query, projectIds, {
      limit,
      ...(languages === undefined ? {} : { languages }),
      ...(paths === undefined ? {} : { paths }),
      ...(kinds === undefined ? {} : { kinds }),
    });
  }

  async findSymbol(
    name: string,
    project?: string | null,
    {
      match = "exact",
      kinds,
      limit = 20,
      roots,
    }: {
      match?: "exact" | "prefix" | "contains";
      kinds?: readonly string[];
      limit?: number;
      roots?: readonly string[];
    } = {},
  ): Promise<SymbolResponse> {
    await this.#ready;
    const resolved = await this.resolveProject(project, roots);
    await this.#ensureQueryGenerations([resolved.id], roots);
    return this.search.findSymbol(name, resolved.id, {
      match,
      limit,
      ...(kinds === undefined ? {} : { kinds }),
    });
  }

  async fileOutline(
    sourcePath: string,
    project?: string | null,
    { roots }: { roots?: readonly string[] } = {},
  ): Promise<OutlineResponse> {
    await this.#ready;
    const resolved = await this.resolveProject(project, roots);
    await this.#ensureQueryGenerations([resolved.id], roots);
    return this.search.fileOutline(sourcePath, resolved.id);
  }

  async getChunk(chunkId: string): Promise<CodeChunk> {
    await this.#ready;
    const projectId = chunkIdPrefix(chunkId);
    if (projectId !== null) await this.#ensureQueryGenerations([projectId], undefined);
    return this.search.getChunk(chunkId);
  }

  async #prepareReferenceQuery(
    selector: DeclarationSelector,
    roots: readonly string[] | undefined,
  ): Promise<[DeclarationSelector, ReferenceBackfillReport]> {
    let projectId: string;
    let prepared = selector;
    if (selector.project !== null) {
      const resolved = await this.#resolve(selector.project, roots);
      prepared = { ...selector, project: resolved.id };
      projectId = resolved.id;
    } else {
      const chunkProjectId = chunkIdPrefix(selector.chunk_id ?? "");
      if (chunkProjectId === null) {
        await this.search.getChunk(selector.chunk_id ?? "");
        throw new Error("get_chunk unexpectedly returned without a project id");
      }
      projectId = chunkProjectId;
    }
    await this.#ensureQueryGenerations([projectId], roots);
    return [
      prepared,
      await this.ensureReferenceIndex(projectId, roots === undefined ? {} : { roots }),
    ];
  }

  async #ensureQueryGenerations(
    projectIds: readonly string[],
    roots: readonly string[] | undefined,
  ): Promise<void> {
    for (const projectId of projectIds) {
      if ((await this.store.incompatibilityReason(projectId, this.embedder.modelId)) !== null) {
        await this.indexProject(projectId, {
          waitForLock: true,
          trigger: "lazy-query",
          ...(roots === undefined ? {} : { roots }),
        });
      }
    }
  }

  async findReferences(
    selector: DeclarationSelector,
    {
      kinds,
      limit = 100,
      cursor,
      roots,
    }: {
      kinds?: ReadonlySet<string> | null;
      limit?: number;
      cursor?: string | null;
      roots?: readonly string[];
    } = {},
  ): Promise<ReferenceResponse> {
    await this.#ready;
    const [prepared, report] = await this.#prepareReferenceQuery(selector, roots);
    return this.store.withPartitionAccess(report.project_id, () =>
      this.references.findReferences(prepared, {
        limit,
        backfill: report,
        ...(kinds === undefined ? {} : { kinds }),
        ...(cursor === undefined ? {} : { cursor }),
      }),
    );
  }

  async analyzeRefactor(
    selector: DeclarationSelector,
    operation: RefactorOperation,
    {
      limit = 500,
      cursor,
      roots,
    }: { limit?: number; cursor?: string | null; roots?: readonly string[] } = {},
  ): Promise<RefactorAnalysis> {
    await this.#ready;
    const [prepared, report] = await this.#prepareReferenceQuery(selector, roots);
    return this.store.withPartitionAccess(report.project_id, () =>
      this.references.analyzeRefactor(prepared, operation, {
        limit,
        backfill: report,
        ...(cursor === undefined ? {} : { cursor }),
      }),
    );
  }

  async prepareModel(): Promise<void> {
    if (this.embedder instanceof OnnxEmbedder) await this.embedder.prepare();
  }

  async #rootLock(root: string): Promise<() => Promise<void>> {
    const directory = path.join(this.paths.data, "locks");
    fs.mkdirSync(directory, { recursive: true });
    const digest = createHash("sha256").update(projectRootIdentity(root)).digest("hex");
    return acquireLock(path.join(directory, `discover-${digest}.lock`), true);
  }

  async #registrationLock(): Promise<() => Promise<void>> {
    const directory = path.join(this.paths.data, "locks");
    fs.mkdirSync(directory, { recursive: true });
    return acquireLock(path.join(directory, "registration.lock"), true);
  }

  async #registerProject(project: ProjectInfo): Promise<void> {
    const known = new Set((await this.store.listProjects()).map((item) => item.id));
    const state = known.has(project.id) ? await this.store.projectState(project.id) : "pending";
    await this.store.upsertProject(project, { modelId: this.embedder.modelId, state });
  }

  async #isProjectShaped(root: string): Promise<boolean> {
    return (
      PROJECT_SHAPE_MARKERS.some((marker) => fs.existsSync(path.join(root, marker))) &&
      (await this.indexer.scanner.hasSupportedSource(root, ScanConfig.parse({})))
    );
  }

  async resolveProject(explicit?: string | null, roots?: readonly string[]): Promise<ProjectInfo> {
    await this.#ready;
    return this.#resolve(explicit, roots);
  }

  async resolveSearchScope(
    projects: readonly string[] | undefined,
    allProjects: boolean,
    roots?: readonly string[],
  ): Promise<string[]> {
    await this.#ready;
    return this.#searchScope(projects, allProjects, roots);
  }

  async #resolve(
    explicit: string | null | undefined,
    roots: readonly string[] | undefined,
  ): Promise<ProjectInfo> {
    return new ProjectResolver(await this.store.listProjects()).resolve({
      ...(explicit === null || explicit === undefined ? {} : { explicit }),
      roots: roots ?? [],
      cwd: this.cwd,
    });
  }

  async #searchScope(
    projects: readonly string[] | undefined,
    allProjects: boolean,
    roots: readonly string[] | undefined,
  ): Promise<string[]> {
    if (projects !== undefined && projects.length > 0 && allProjects) {
      throw new CodeIndexingError(
        "INVALID_FILTER",
        "projects and all_projects cannot be used together",
      );
    }
    let projectIds: string[];
    if (allProjects) {
      projectIds = (await this.listProjects()).map((project) => project.id);
    } else if (projects !== undefined && projects.length > 0) {
      projectIds = [];
      for (const project of projects) projectIds.push((await this.#resolve(project, roots)).id);
    } else {
      projectIds = [(await this.#resolve(undefined, roots)).id];
    }
    if (projectIds.length === 0) {
      throw new CodeIndexingError(
        "PROJECT_NOT_FOUND",
        "No indexed projects are available; init_project registers one and " +
          "index_project builds its index",
      );
    }
    return projectIds;
  }
}

function nowSeconds(): number {
  return performance.now() / 1000;
}

function formatError(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  return String(error);
}

function skipped(project: ProjectInfo, reason: string): MaintenanceProjectResult {
  return {
    project,
    status: "skipped",
    skip_reason: reason,
    error: null,
    before: null,
    after: null,
    versions_removed: 0,
    bytes_reclaimed: 0,
    reclaimable_bytes_estimate: 0,
  };
}
