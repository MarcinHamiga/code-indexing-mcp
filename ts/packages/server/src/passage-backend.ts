/**
 * The passage-embedding session that survives its own accelerator failing.
 */

import { type BackendSelection, CPU_BACKEND } from "./backends.ts";
import {
  type CalibrationResult,
  calibrate,
  calibrationResult,
  LIMITED_BY_MEMORY,
} from "./calibration.ts";
import {
  type EmbeddedSegment,
  type PassageCandidate,
  type SegmentPlan,
  segmentPlan,
} from "./embedding.ts";
import type { EmbeddingWorkerSession, SessionTelemetry } from "./embedding-worker.ts";
import { CodeIndexingError, type ErrorCode, isCodeIndexingError } from "./errors.ts";
import type { ProbeCache, ProbeKey, ProbeRecord } from "./probe-cache.ts";

const BACKEND_FAILURE_CODES = new Set<ErrorCode>([
  "EMBEDDING_WORKER_FAILED",
  "INDEX_RESOURCE_LIMIT",
  "MODEL_UNAVAILABLE",
]);

export type SessionFactory = () => EmbeddingWorkerSession;

function recordedCalibration(record: ProbeRecord): CalibrationResult | undefined {
  if (record.charactersPerSecond <= 0) return undefined;
  return calibrationResult({
    maxItems: record.batchSize,
    charactersPerSecond: record.charactersPerSecond,
    loadNs: record.loadNs,
    limitedBy: record.limitedBy,
  });
}

interface RunCounters {
  segmentCount: number;
  tokenCount: number;
  retryCount: number;
  peakCombinedRss: number;
  safeMaxItems: number;
  terminationReason: string | null;
}

function snapshotCounters(session: EmbeddingWorkerSession): RunCounters {
  return {
    segmentCount: session.segmentCount,
    tokenCount: session.tokenCount,
    retryCount: session.retryCount,
    peakCombinedRss: session.peakCombinedRss,
    safeMaxItems: session.safeMaxItems,
    terminationReason: session.terminationReason,
  };
}

function restoreCounters(session: EmbeddingWorkerSession, counters: RunCounters): void {
  session.segmentCount = counters.segmentCount;
  session.tokenCount = counters.tokenCount;
  session.retryCount = counters.retryCount;
  session.peakCombinedRss = counters.peakCombinedRss;
  session.safeMaxItems = counters.safeMaxItems;
  session.terminationReason = counters.terminationReason;
}

function reasonOf(error: unknown): string {
  return isCodeIndexingError(error) ? error.forClient() : String(error);
}

export class PassageBackendSession {
  selection: BackendSelection;
  readonly strict: boolean;
  charactersEmbedded = 0;
  crossoverCharacters: number | null;
  backendUsed: string;
  fallbackCount = 0;
  fallbackReason: string | null;
  probeState = "unprobed";
  calibration: CalibrationResult | undefined;
  private readonly acceleratorFactory: SessionFactory;
  private readonly cpuFactory: SessionFactory;
  private readonly probeCache: ProbeCache | undefined;
  private readonly probeKey: ProbeKey | undefined;
  private readonly cpuProbeKey: ProbeKey | undefined;
  private calibratedBatchSize: number;
  private sessionMaxItems = 0;
  private readonly calibrationPlan: SegmentPlan | undefined;
  private readonly cpuMaxItems: number;
  private onProvisionalCpu = false;
  private readonly dimension: number;
  private readonly onDegrade: ((selection: BackendSelection) => void) | undefined;
  private session: EmbeddingWorkerSession | undefined;
  private onCpu: boolean;
  private verifiedSpawn = 0;
  private readonly retired: SessionTelemetry[] = [];

  constructor(
    selection: BackendSelection,
    {
      acceleratorFactory,
      cpuFactory,
      strict = false,
      probeCache,
      probeKey,
      cpuProbeKey,
      calibratedBatchSize = 0,
      dimension = 0,
      onDegrade,
      crossoverCharacters = 0,
      calibrationPlan,
      cpuMaxItems = 0,
    }: {
      acceleratorFactory: SessionFactory;
      cpuFactory: SessionFactory;
      strict?: boolean;
      probeCache?: ProbeCache;
      probeKey?: ProbeKey;
      cpuProbeKey?: ProbeKey;
      calibratedBatchSize?: number;
      dimension?: number;
      onDegrade?: (selection: BackendSelection) => void;
      crossoverCharacters?: number | null;
      calibrationPlan?: SegmentPlan;
      cpuMaxItems?: number;
    },
  ) {
    this.selection = selection;
    this.strict = strict;
    this.acceleratorFactory = acceleratorFactory;
    this.cpuFactory = cpuFactory;
    this.probeCache = probeCache;
    this.probeKey = probeKey;
    this.cpuProbeKey = cpuProbeKey;
    this.calibratedBatchSize = calibratedBatchSize;
    this.calibrationPlan = calibrationPlan;
    this.cpuMaxItems = cpuMaxItems;
    this.crossoverCharacters = crossoverCharacters;
    this.dimension = dimension;
    this.onDegrade = onDegrade;
    this.onCpu = !selection.usesAccelerator;
    this.backendUsed = this.onCpu ? CPU_BACKEND.accelerator : selection.accelerator;
    this.fallbackReason = selection.honored ? null : selection.fallbackReason;
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.close();
    await this.measureReference();
  }

  async enter(): Promise<this> {
    if (this.strict) this.selection.requireHonored();
    return this;
  }

  async exit(error?: unknown): Promise<void> {
    await this.close();
    if (error === undefined) await this.measureReference();
  }

  async close(): Promise<void> {
    const session = this.session;
    if (session === undefined) return;
    this.retired.push(session.telemetry());
    this.session = undefined;
    await session.close();
  }

  async planAndEmbed(
    candidates: readonly PassageCandidate[],
    plan: SegmentPlan,
  ): Promise<EmbeddedSegment[][]> {
    const pending = candidates.reduce(
      (sum, candidate) => sum + candidate.prefix.length + candidate.content.length,
      0,
    );
    return await this.attempt(
      (session) => session.planAndEmbed(candidates, this.planFor(session, plan)),
      pending,
    );
  }

  private planFor(session: EmbeddingWorkerSession, plan: SegmentPlan): SegmentPlan {
    if (session.config.accelerator === CPU_BACKEND.accelerator) {
      return this.cpuMaxItems ? segmentPlan({ ...plan, maxItems: this.cpuMaxItems }) : plan;
    }
    if (this.sessionMaxItems && this.sessionMaxItems !== plan.maxItems) {
      return segmentPlan({ ...plan, maxItems: this.sessionMaxItems });
    }
    return plan;
  }

  async embedPassages(texts: string[]): Promise<number[][]> {
    return await this.attempt(
      (session) => session.embedPassages(texts),
      texts.reduce((sum, text) => sum + text.length, 0),
    );
  }

  private async attempt<Result>(
    call: (session: EmbeddingWorkerSession) => Promise<Result>,
    pending = 0,
  ): Promise<Result> {
    let result: Result;
    try {
      result = await call(await this.active(pending));
    } catch (error) {
      if (
        this.onCpu ||
        this.onProvisionalCpu ||
        !isCodeIndexingError(error) ||
        !BACKEND_FAILURE_CODES.has(error.code)
      ) {
        throw error;
      }
      await this.degrade(reasonOf(error));
      result = await call(await this.active(pending));
    }
    this.charactersEmbedded += pending;
    this.adoptReducedBatchSize();
    return result;
  }

  private async active(pending = 0): Promise<EmbeddingWorkerSession> {
    if (this.onCpu) {
      if (this.session === undefined) this.session = this.startCpu();
      return this.session;
    }
    const crossover = this.crossoverCharacters;
    if (crossover === null || crossover > this.charactersEmbedded + pending) {
      if (this.session === undefined) {
        this.onProvisionalCpu = true;
        this.session = this.startCpu();
      }
      return this.session;
    }
    if (this.onProvisionalCpu) {
      this.onProvisionalCpu = false;
      await this.close();
    }
    let session = this.session;
    if (session !== undefined && session.spawnCount === this.verifiedSpawn) return session;
    if (session === undefined) session = this.acceleratorFactory();
    try {
      await this.verify(session);
    } catch (error) {
      this.session = session;
      this.probeState = "failed";
      await this.degrade(reasonOf(error));
      this.session = this.startCpu();
      return this.session;
    }
    this.session = session;
    this.backendUsed = this.selection.descriptor.accelerator;
    return session;
  }

  private startCpu(): EmbeddingWorkerSession {
    this.backendUsed = CPU_BACKEND.accelerator;
    return this.cpuFactory();
  }

  private async verify(session: EmbeddingWorkerSession): Promise<void> {
    const info = await session.initialize();
    const descriptor = this.selection.descriptor;
    if (
      info.resolvedProviders.length > 0 &&
      !info.resolvedProviders.includes(descriptor.provider)
    ) {
      throw new CodeIndexingError(
        "BACKEND_UNAVAILABLE",
        `${descriptor.provider} was requested but the session runs on ${info.resolvedProviders.join(", ")}`,
        { requested: descriptor.provider, resolved: [...info.resolvedProviders] },
      );
    }
    if (info.dimension !== session.config.dimension) {
      throw new CodeIndexingError(
        "BACKEND_UNAVAILABLE",
        `${descriptor.accelerator} reported dimension ${info.dimension}, expected ${session.config.dimension}`,
      );
    }
    const cached = this.cachedProbe();
    if (cached !== undefined) {
      this.probeState = "cached";
      this.verifiedSpawn = session.spawnCount;
      this.calibration = recordedCalibration(cached);
      return;
    }
    await session.probe();
    this.probeState = "verified";
    this.verifiedSpawn = session.spawnCount;
    await this.measure(session);
    this.recordProbe();
  }

  private async measure(session: EmbeddingWorkerSession): Promise<void> {
    if (this.calibrationPlan === undefined) return;
    const before = snapshotCounters(session);
    try {
      this.calibration = await calibrate(session, this.calibrationPlan, {
        loadNs: session.loadDurationNs,
      });
    } finally {
      restoreCounters(session, before);
      this.verifiedSpawn = session.spawnCount;
    }
    if (this.calibration !== undefined) {
      this.calibratedBatchSize = this.calibration.maxItems;
      this.sessionMaxItems = this.calibration.maxItems;
    }
  }

  private async measureReference(): Promise<void> {
    if (this.probeState !== "verified") return;
    if (this.calibrationPlan === undefined || this.probeCache === undefined) return;
    const key = this.cpuProbeKey;
    if (key === undefined || this.probeCache.load(key) !== undefined) return;
    const session = this.cpuFactory();
    let measured: CalibrationResult | undefined;
    try {
      await session.initialize();
      measured = await calibrate(session, this.calibrationPlan, { loadNs: session.loadDurationNs });
    } catch {
      return;
    } finally {
      await session.close();
    }
    if (measured === undefined) return;
    this.probeCache.store(key, {
      batchSize: measured.maxItems,
      dimension: this.dimension,
      detail: CPU_BACKEND.provider,
      charactersPerSecond: measured.charactersPerSecond,
      loadNs: measured.loadNs,
      limitedBy: measured.limitedBy,
    });
  }

  private adoptReducedBatchSize(): void {
    const session = this.session;
    if (
      session === undefined ||
      this.onCpu ||
      this.onProvisionalCpu ||
      !session.safeMaxItems ||
      session.safeMaxItems === this.calibratedBatchSize
    ) {
      return;
    }
    this.calibratedBatchSize = session.safeMaxItems;
    this.sessionMaxItems = session.safeMaxItems;
    if (this.calibration !== undefined) {
      this.calibration = {
        ...this.calibration,
        maxItems: session.safeMaxItems,
        limitedBy: LIMITED_BY_MEMORY,
      };
    }
    this.recordProbe(LIMITED_BY_MEMORY);
  }

  private cachedProbe(): ProbeRecord | undefined {
    if (this.probeCache === undefined || this.probeKey === undefined) return undefined;
    return this.probeCache.load(this.probeKey);
  }

  private recordProbe(limitedBy = ""): void {
    if (this.probeCache === undefined || this.probeKey === undefined) return;
    const measured = this.calibration;
    this.probeCache.store(this.probeKey, {
      batchSize: this.calibratedBatchSize,
      dimension: this.dimension,
      detail: this.selection.descriptor.provider,
      charactersPerSecond: measured === undefined ? 0 : measured.charactersPerSecond,
      loadNs: measured === undefined ? 0 : measured.loadNs,
      limitedBy: limitedBy || (measured === undefined ? "" : measured.limitedBy),
    });
  }

  private async degrade(reason: string): Promise<void> {
    if (this.strict) {
      throw new CodeIndexingError(
        "BACKEND_UNAVAILABLE",
        `Embedding accelerator ${this.selection.accelerator} failed and ` +
          `CODE_INDEXING_EMBED_STRICT forbids the CPU fallback: ${reason}`,
        {
          requested: this.selection.requested,
          accelerator: this.selection.accelerator,
          reason,
        },
      );
    }
    await this.close();
    this.selection = this.selection.fellBackTo(CPU_BACKEND, reason);
    this.fallbackReason = reason;
    this.fallbackCount += 1;
    this.onCpu = true;
    this.onProvisionalCpu = false;
    this.verifiedSpawn = 0;
    this.onDegrade?.(this.selection);
  }

  telemetry(): SessionTelemetry {
    const entries = this.allTelemetry();
    const termination =
      [...entries].reverse().find((entry) => entry.terminationReason)?.terminationReason ?? null;
    const tokenizer =
      entries.find((entry) => entry.tokenizerAvailable !== null)?.tokenizerAvailable ?? null;
    return {
      backend: this.backendUsed,
      memoryBudgetBytes: Math.max(0, ...entries.map((entry) => entry.memoryBudgetBytes)),
      peakMemoryBytes: Math.max(0, ...entries.map((entry) => entry.peakMemoryBytes)),
      segmentCount: entries.reduce((sum, entry) => sum + entry.segmentCount, 0),
      tokenCount: entries.reduce((sum, entry) => sum + entry.tokenCount, 0),
      retryCount: entries.reduce((sum, entry) => sum + entry.retryCount, 0),
      fallbackCount: entries.reduce((sum, entry) => sum + entry.retryCount, 0) + this.fallbackCount,
      terminationReason: termination,
      tokenizerAvailable: tokenizer,
      fallbackReason: this.fallbackReason,
      characterCount: this.charactersEmbedded,
      crossoverCharacters: this.crossoverCharacters,
      selectionReason: this.selectionReason(),
    };
  }

  private selectionReason(): string | null {
    const accelerator = this.selection.accelerator;
    if (this.onCpu) return null;
    if (this.crossoverCharacters === null) {
      return `${accelerator} measured no faster than CPU on this machine`;
    }
    if (!this.crossoverCharacters) return null;
    if (this.onProvisionalCpu || this.backendUsed === CPU_BACKEND.accelerator) {
      return (
        `embedded ${this.charactersEmbedded} characters, below the ` +
        `${this.crossoverCharacters}-character crossover for ${accelerator}`
      );
    }
    return `passed the ${this.crossoverCharacters}-character crossover for ${accelerator}`;
  }

  private allTelemetry(): SessionTelemetry[] {
    const live = this.session === undefined ? [] : [this.session.telemetry()];
    return [...this.retired, ...live];
  }
}
