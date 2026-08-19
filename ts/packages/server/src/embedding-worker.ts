/**
 * Memory accounting primitives for disposable embedding workers.
 */

import os from "node:os";
import pidusage from "pidusage";
import { CPU_PROVIDER } from "./backends.ts";
import {
  DEFAULT_MODEL,
  type EmbeddedSegment,
  embeddedSegment,
  embedWindows,
  type PassageCandidate,
  PROBE_TEXTS,
  passageCandidate,
  planPassages,
  resolveSessionProviders,
  resolveTokenizer,
  type SegmentPlan,
  segmentPlan,
  validateProbeVectors,
} from "./embedding.ts";
import { CodeIndexingError, type ErrorCode } from "./errors.ts";
import { decodeBytes, type WorkerConnection } from "./worker-channel.ts";
import {
  FunctionLauncher,
  registerWorkerTarget,
  SpawnLauncher,
  type WorkerLauncher,
  type WorkerProcess,
  type WorkerTarget,
} from "./worker-launcher.ts";

export const SYSTEM_RESERVE_BYTES = 512 * 1024 ** 2;
export const MINIMUM_WORKER_BYTES = 1024 ** 3;
export const HARD_OVERSHOOT_BYTES = 128 * 1024 ** 2;
export const MAX_BATCH_RETRIES = 2;

const RETRYABLE_CODES = new Set<ErrorCode>(["INDEX_RESOURCE_LIMIT", "EMBEDDING_WORKER_FAILED"]);

export interface WorkerConfig {
  readonly cacheDirectory: string;
  readonly offline: boolean;
  readonly threads: number;
  readonly enableCpuMemArena: boolean;
  readonly dimension: number;
  readonly modelId: string;
  readonly providers: readonly string[];
  readonly accelerator: string;
}

export function workerConfig(fields: {
  cacheDirectory: string;
  offline: boolean;
  threads: number;
  enableCpuMemArena: boolean;
  dimension: number;
  modelId?: string;
  providers?: readonly string[];
  accelerator?: string;
}): WorkerConfig {
  return {
    cacheDirectory: fields.cacheDirectory,
    offline: fields.offline,
    threads: fields.threads,
    enableCpuMemArena: fields.enableCpuMemArena,
    dimension: fields.dimension,
    modelId: fields.modelId ?? DEFAULT_MODEL,
    providers: fields.providers ?? [],
    accelerator: fields.accelerator ?? "cpu",
  };
}

export function workerConfigIsCpu(config: WorkerConfig): boolean {
  return (
    config.providers.length === 0 ||
    (config.providers.length === 1 && config.providers[0] === CPU_PROVIDER)
  );
}

export interface WorkerInfo {
  readonly resolvedProviders: readonly string[];
  readonly dimension: number;
}

export interface SessionTelemetry {
  readonly backend: string;
  readonly memoryBudgetBytes: number;
  readonly peakMemoryBytes: number;
  readonly segmentCount: number;
  readonly tokenCount: number;
  readonly retryCount: number;
  readonly fallbackCount: number;
  readonly terminationReason: string | null;
  readonly tokenizerAvailable: boolean | null;
  readonly fallbackReason?: string | null;
  readonly characterCount?: number;
  readonly crossoverCharacters?: number | null;
  readonly selectionReason?: string | null;
}

export function defaultLauncher(): WorkerLauncher {
  return new SpawnLauncher(workerMain);
}

export function effectiveMemoryCeiling({
  configuredBytes,
  availableBytes,
}: {
  configuredBytes: number;
  availableBytes: number;
}): number {
  return Math.min(configuredBytes, Math.max(0, availableBytes - SYSTEM_RESERVE_BYTES));
}

export function indexingMemoryBytes({
  parentBytes,
  workerBytes,
  parentBaselineBytes,
}: {
  parentBytes: number;
  workerBytes: number;
  parentBaselineBytes: number;
}): number {
  return workerBytes + Math.max(0, parentBytes - parentBaselineBytes);
}

export type ModelLoader = (
  config: WorkerConfig,
) => Promise<PassageWorkerModel> | PassageWorkerModel;

export interface PassageWorkerModel {
  passageEmbed(
    texts: string[],
  ):
    | Iterable<ArrayLike<number>>
    | ArrayLike<number>[]
    | Promise<Iterable<ArrayLike<number>> | ArrayLike<number>[]>;
  readonly resolvedProviders?: readonly string[];
  readonly tokenizer?: { encode: (text: string) => unknown };
}

export let loadModel: ModelLoader = defaultLoadModel;

export function setLoadModel(loader: ModelLoader): void {
  loadModel = loader;
}

async function defaultLoadModel(config: WorkerConfig): Promise<PassageWorkerModel> {
  const { DirectOnnxEmbedding } = await import("./direct-onnx.ts");
  return DirectOnnxEmbedding.create(config.cacheDirectory, {
    offline: config.offline,
    threads: config.threads,
    enableCpuMemArena: config.enableCpuMemArena,
    providers: config.providers.length > 0 ? config.providers : [CPU_PROVIDER],
    modelId: config.modelId,
    accelerator: config.accelerator,
  });
}

export async function workerMain(
  connection: WorkerConnection,
  config: WorkerConfig,
): Promise<void> {
  try {
    const model = await loadModel(config);
    const tokenizer = resolveTokenizer(model);
    const embedPacked = async (texts: string[]): Promise<Uint8Array[]> =>
      [...(await model.passageEmbed(texts))].map((vector) => packLittleEndian(vector));

    for (;;) {
      const message = (await connection.recv()) as [string, unknown];
      const [command, payload] = message;
      if (command === "stop") return;
      if (command === "initialize") {
        connection.send(["initialized", [resolveSessionProviders(model), config.dimension]]);
        continue;
      }
      if (command === "memory") {
        connection.send(["memory", process.memoryUsage().rss]);
        continue;
      }
      if (command === "probe") {
        connection.send(["probed", await embedPacked([...PROBE_TEXTS])]);
        continue;
      }
      if (command === "embed") {
        connection.send(["packed", await embedPacked(payload as string[])]);
        continue;
      }
      if (command !== "plan_and_embed") {
        throw new Error(`Unknown worker command: ${command}`);
      }
      const [rawCandidates, plan] = payload as [
        Array<[string, string]>,
        SegmentPlan & Record<string, number>,
      ];
      const candidates = rawCandidates.map(([prefix, content]) =>
        passageCandidate(prefix, content),
      );
      const resolvedPlan = segmentPlan({
        maxTokens: plan.maxTokens ?? plan.max_tokens,
        overlapTokens: plan.overlapTokens ?? plan.overlap_tokens,
        maxItems: plan.maxItems ?? plan.max_items,
        maxTokenProduct: plan.maxTokenProduct ?? plan.max_token_product,
        maxWindows: plan.maxWindows ?? plan.max_windows,
      });
      try {
        const encode =
          tokenizer === undefined
            ? undefined
            : (text: string) => {
                const encoded = tokenizer.encode(text);
                if (encoded !== null && typeof encoded === "object" && "offsets" in encoded) {
                  return encoded as { offsets: readonly (readonly [number, number])[] };
                }
                return { offsets: [] };
              };
        const windows = planPassages(encode, candidates, resolvedPlan);
        const planned = await embedWindows(embedPacked, candidates, windows, resolvedPlan);
        connection.send([
          "planned",
          [
            planned.map((segments) =>
              segments.map(([window, vector]) => [
                window.startChar,
                window.endChar,
                window.tokenCount,
                vector,
              ]),
            ),
            tokenizer !== undefined,
          ],
        ]);
      } catch (error) {
        if (error instanceof Error && !(error instanceof CodeIndexingError)) {
          connection.send(["plan_error", error.message]);
          continue;
        }
        throw error;
      }
    }
  } catch (error) {
    try {
      const name = error instanceof Error ? error.constructor.name : "Error";
      const message = error instanceof Error ? error.message : String(error);
      connection.send(["error", `${name}: ${message}`]);
    } catch {
      return;
    }
  } finally {
    connection.close();
  }
}

registerWorkerTarget("embedding-worker:workerMain", workerMain);

function packLittleEndian(vector: ArrayLike<number>): Uint8Array {
  const out = Buffer.allocUnsafe(vector.length * 4);
  for (let index = 0; index < vector.length; index++) {
    out.writeFloatLE(vector[index] ?? 0, index * 4);
  }
  return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
}

export type SampleRss = () => [number, number] | Promise<[number, number]>;

export class EmbeddingWorkerSession {
  readonly config: WorkerConfig;
  readonly effectiveCeilingBytes: number;
  spawnCount = 0;
  peakCombinedRss = 0;
  retryCount = 0;
  loadDurationNs = 0;
  safeMaxItems = 0;
  segmentCount = 0;
  tokenCount = 0;
  terminationReason: string | null = null;
  tokenizerAvailable: boolean | null = null;
  sampleRss: SampleRss;
  private readonly launcher: WorkerLauncher;
  private processHandle: WorkerProcess | undefined;
  private connection: WorkerConnection | undefined;
  parentBaselineBytes = 0;

  constructor(
    config: WorkerConfig,
    {
      configuredCeilingBytes,
      effectiveCeilingBytes,
      target = workerMain,
      launcher,
      sampleRss,
    }: {
      configuredCeilingBytes?: number;
      effectiveCeilingBytes?: number;
      target?: WorkerTarget;
      launcher?: WorkerLauncher;
      sampleRss?: SampleRss;
    } = {},
  ) {
    this.config = config;
    const configured = configuredCeilingBytes ?? 2 * 1024 ** 3;
    this.effectiveCeilingBytes =
      effectiveCeilingBytes ??
      effectiveMemoryCeiling({
        configuredBytes: configured,
        availableBytes: os.freemem(),
      });
    if (this.effectiveCeilingBytes < MINIMUM_WORKER_BYTES) {
      throw new CodeIndexingError(
        "INDEX_RESOURCE_LIMIT",
        "Insufficient available memory to load the embedding model safely",
        {
          effective_memory_bytes: this.effectiveCeilingBytes,
          minimum_memory_bytes: MINIMUM_WORKER_BYTES,
        },
      );
    }
    this.launcher = launcher ?? new SpawnLauncher(target);
    this.sampleRss = sampleRss ?? (() => this.readRss());
  }

  get pid(): number | undefined {
    return this.processHandle?.pid;
  }

  async initialize(): Promise<WorkerInfo> {
    const started = Number(process.hrtime.bigint());
    const [status, payload] = await this.request("initialize", null);
    this.loadDurationNs = Number(process.hrtime.bigint()) - started;
    if (status !== "initialized") {
      throw new CodeIndexingError(
        "EMBEDDING_WORKER_FAILED",
        `Embedding worker answered initialize with ${JSON.stringify(status)}`,
      );
    }
    const [providers, dimension] = payload as [unknown[], number];
    return {
      resolvedProviders: providers.map((name) => String(name)),
      dimension: Number(dimension),
    };
  }

  async probe(): Promise<Uint8Array[]> {
    const [status, payload] = await this.request("probe", null);
    if (status !== "probed") {
      throw new CodeIndexingError(
        "EMBEDDING_WORKER_FAILED",
        `Embedding worker answered probe with ${JSON.stringify(status)}`,
      );
    }
    const vectors = (payload as unknown[]).map((vector) => decodeBytes(vector));
    validateProbeVectors(vectors, { dimension: this.config.dimension, count: PROBE_TEXTS.length });
    return vectors;
  }

  async reportMemory(): Promise<number> {
    const [status, payload] = await this.request("memory", null);
    if (status !== "memory") {
      throw new CodeIndexingError(
        "EMBEDDING_WORKER_FAILED",
        `Embedding worker answered memory with ${JSON.stringify(status)}`,
      );
    }
    return Number(payload);
  }

  telemetry(): SessionTelemetry {
    return {
      backend: this.config.accelerator,
      memoryBudgetBytes: this.effectiveCeilingBytes,
      peakMemoryBytes: this.peakCombinedRss,
      segmentCount: this.segmentCount,
      tokenCount: this.tokenCount,
      retryCount: this.retryCount,
      fallbackCount: this.retryCount,
      terminationReason: this.terminationReason,
      tokenizerAvailable: this.tokenizerAvailable,
    };
  }

  async embedPassages(texts: string[]): Promise<number[][]> {
    const [status, payload] = await this.request("embed", texts);
    if (status === "packed") {
      return (payload as unknown[]).map((vector) => {
        const packed = decodeBytes(vector);
        const row = new Float32Array(packed.buffer, packed.byteOffset, this.config.dimension);
        return Array.from(row);
      });
    }
    return (payload as number[][]).map((vector) => vector.map((value) => Number(value)));
  }

  async planAndEmbed(
    candidates: readonly PassageCandidate[],
    plan: SegmentPlan,
  ): Promise<EmbeddedSegment[][]> {
    const request = candidates.map((candidate) => [candidate.prefix, candidate.content]);
    let attempt = plan;
    let payload: unknown;
    for (let retry = 0; retry <= MAX_BATCH_RETRIES; retry++) {
      try {
        const [status, body] = await this.request("plan_and_embed", [request, attempt]);
        if (status === "plan_error") throw new Error(String(body));
        if (retry > 0) this.safeMaxItems = attempt.maxItems;
        payload = body;
        break;
      } catch (error) {
        if (!isRetryable(error) || attempt.maxItems <= 1 || retry === MAX_BATCH_RETRIES) {
          throw error;
        }
        attempt = segmentPlan({
          ...attempt,
          maxItems: Math.max(1, Math.trunc(attempt.maxItems / 2)),
        });
        this.retryCount += 1;
        await this.close();
      }
    }
    const [segmentsPayload, tokenizerAvailable] = payload as [unknown[][], unknown];
    this.tokenizerAvailable = Boolean(tokenizerAvailable);
    const results: EmbeddedSegment[][] = [];
    for (const segments of segmentsPayload) {
      const decoded = segments.map((entry) => {
        const [startChar, endChar, tokenCount, vector] = entry as [number, number, number, unknown];
        return embeddedSegment(startChar, endChar, tokenCount, decodeBytes(vector));
      });
      this.segmentCount += decoded.length;
      this.tokenCount += decoded.reduce((sum, segment) => sum + segment.tokenCount, 0);
      results.push(decoded);
    }
    return results;
  }

  async close(): Promise<void> {
    const handle = this.processHandle;
    const connection = this.connection;
    if (handle === undefined) return;
    if (handle.isAlive() && connection !== undefined) {
      try {
        connection.send(["stop", null]);
      } catch {
        /* already gone */
      }
      await handle.join(2);
    }
    if (handle.isAlive()) {
      handle.terminate();
      await handle.join(2);
    }
    if (handle.isAlive()) {
      handle.kill();
      await handle.join(2);
    }
    connection?.close();
    this.processHandle = undefined;
    this.connection = undefined;
  }

  [Symbol.asyncDispose](): Promise<void> {
    return this.close();
  }

  private async request(command: string, payload: unknown): Promise<[string, unknown]> {
    await this.start();
    const connection = this.connection;
    const handle = this.processHandle;
    if (connection === undefined || handle === undefined) {
      throw this.channelFailed();
    }
    try {
      connection.send([command, payload]);
    } catch (error) {
      throw this.channelFailed(error);
    }
    let consecutiveOver = 0;
    for (;;) {
      let replyReady: boolean;
      try {
        replyReady = await connection.poll(0.1);
      } catch (error) {
        throw this.channelFailed(error);
      }
      if (!replyReady && !handle.isAlive()) {
        await this.close();
        this.terminationReason = "worker_exited";
        throw new CodeIndexingError(
          "EMBEDDING_WORKER_FAILED",
          "Embedding worker exited without returning a result",
        );
      }
      const [parentRss, workerRss] = await this.sampleRss();
      this.peakCombinedRss = Math.max(this.peakCombinedRss, parentRss + workerRss);
      const budgeted = indexingMemoryBytes({
        parentBytes: parentRss,
        workerBytes: workerRss,
        parentBaselineBytes: this.parentBaselineBytes,
      });
      consecutiveOver = budgeted > this.effectiveCeilingBytes ? consecutiveOver + 1 : 0;
      if (budgeted > this.effectiveCeilingBytes + HARD_OVERSHOOT_BYTES || consecutiveOver >= 5) {
        await this.terminate();
        this.terminationReason = "memory_ceiling";
        throw new CodeIndexingError(
          "INDEX_RESOURCE_LIMIT",
          "Indexing exceeded its memory ceiling",
          {
            effective_memory_bytes: this.effectiveCeilingBytes,
            indexing_memory_bytes: budgeted,
            peak_memory_bytes: this.peakCombinedRss,
            parent_baseline_bytes: this.parentBaselineBytes,
          },
        );
      }
      if (replyReady) break;
    }
    let status: string;
    let body: unknown;
    try {
      const reply = (await connection.recv()) as [string, unknown];
      status = reply[0];
      body = reply[1];
    } catch (error) {
      throw this.channelFailed(error);
    }
    if (status === "error") {
      await this.close();
      this.terminationReason = "worker_error";
      throw new CodeIndexingError("EMBEDDING_WORKER_FAILED", String(body));
    }
    return [status, body];
  }

  private channelFailed(_cause?: unknown): CodeIndexingError {
    void this.close();
    this.terminationReason = "channel_closed";
    return new CodeIndexingError(
      "EMBEDDING_WORKER_FAILED",
      "Embedding worker closed its result channel",
    );
  }

  private async start(): Promise<void> {
    if (this.processHandle !== undefined) return;
    if (this.parentBaselineBytes === 0) {
      this.parentBaselineBytes = process.memoryUsage().rss;
    }
    const launched = await this.launcher.launch(this.config);
    this.processHandle = launched.process;
    this.spawnCount += 1;
    this.connection = launched.connection;
  }

  private async readRss(): Promise<[number, number]> {
    const parentRss = process.memoryUsage().rss;
    const pid = this.processHandle?.pid;
    if (pid === undefined) return [parentRss, 0];
    try {
      return [parentRss, await childRss(pid)];
    } catch {
      return [parentRss, 0];
    }
  }

  private async terminate(): Promise<void> {
    this.processHandle?.terminate();
    const deadline = Date.now() + 2000;
    while (this.processHandle?.isAlive() && Date.now() < deadline) {
      await this.processHandle.join(0.05);
    }
    await this.close();
  }
}

function isRetryable(error: unknown): error is CodeIndexingError {
  return error instanceof CodeIndexingError && RETRYABLE_CODES.has(error.code);
}

export async function childRss(pid: number): Promise<number> {
  return (await pidusage(pid)).memory;
}

export { FunctionLauncher };
