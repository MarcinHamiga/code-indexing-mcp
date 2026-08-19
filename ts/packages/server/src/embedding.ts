/**
 * Local embedding protocol and in-process CPU embedder.
 *
 * FastEmbed is not ported: the direct ONNX path is the only path, per the
 * migration plan §5.1. Query and passage embedding share that path so they
 * stay in one vector space.
 */

import { CodeIndexingError } from "./errors.ts";
import {
  DEFAULT_MAX_TOKEN_PRODUCT,
  DEFAULT_MAX_TOKENS,
  DEFAULT_OVERLAP_TOKENS,
  MAX_WINDOWS_PER_CANDIDATE,
  planCandidateWindows,
  planMicrobatches,
  type TokenEncoding,
  type TokenWindow,
} from "./token-batching.ts";

export const DEFAULT_MODEL = "jinaai/jina-embeddings-v2-base-code";
export const DEFAULT_DIMENSION = 768;

export interface PassageCandidate {
  readonly prefix: string;
  readonly content: string;
}

export function passageCandidate(prefix: string, content: string): PassageCandidate {
  return { prefix, content };
}

export interface EmbeddedSegment {
  readonly startChar: number;
  readonly endChar: number;
  readonly tokenCount: number;
  readonly vector: Uint8Array;
}

export function embeddedSegment(
  startChar: number,
  endChar: number,
  tokenCount: number,
  vector: Uint8Array,
): EmbeddedSegment {
  return { startChar, endChar, tokenCount, vector };
}

export interface SegmentPlan {
  readonly maxTokens: number;
  readonly overlapTokens: number;
  readonly maxItems: number;
  readonly maxTokenProduct: number;
  readonly maxWindows: number;
}

export function segmentPlan(overrides: Partial<SegmentPlan> = {}): SegmentPlan {
  return {
    maxTokens: DEFAULT_MAX_TOKENS,
    overlapTokens: DEFAULT_OVERLAP_TOKENS,
    maxItems: 1,
    maxTokenProduct: DEFAULT_MAX_TOKEN_PRODUCT,
    maxWindows: MAX_WINDOWS_PER_CANDIDATE,
    ...overrides,
  };
}

export function composePassage(prefix: string, content: string): string {
  return prefix ? `${prefix}\n${content}` : content;
}

export function packVector(
  vector:
    | ArrayLike<number>
    | { tolist: () => ArrayLike<number> }
    | { toList: () => ArrayLike<number> },
): Uint8Array {
  let values: ArrayLike<number>;
  if (!isArrayLike(vector) && "tolist" in vector && typeof vector.tolist === "function") {
    values = vector.tolist();
  } else if (!isArrayLike(vector) && "toList" in vector && typeof vector.toList === "function") {
    values = vector.toList();
  } else {
    values = vector as ArrayLike<number>;
  }
  const out = Buffer.allocUnsafe(values.length * 4);
  for (let index = 0; index < values.length; index++) {
    out.writeFloatLE(values[index] ?? 0, index * 4);
  }
  return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
}

function isArrayLike(value: object): value is ArrayLike<number> {
  return "length" in value && typeof (value as ArrayLike<number>).length === "number";
}

export function resolveTokenizer(model: object): { encode: (text: string) => unknown } | undefined {
  for (const path of [["model", "tokenizer"], ["tokenizer"]] as const) {
    let probe: unknown = model;
    for (const attribute of path) {
      if (probe === null || probe === undefined || typeof probe !== "object") {
        probe = undefined;
        break;
      }
      probe = (probe as Record<string, unknown>)[attribute];
    }
    if (probe !== null && probe !== undefined && typeof probe === "object" && "encode" in probe) {
      return probe as { encode: (text: string) => unknown };
    }
  }
  return undefined;
}

export function resolveSessionProviders(model: object): readonly string[] {
  const direct = (model as { resolvedProviders?: unknown }).resolvedProviders;
  if (direct !== undefined && direct !== null && isIterable(direct)) {
    return [...direct].map((name) => String(name));
  }
  for (const path of [["model", "model"], ["model"]] as const) {
    let probe: unknown = model;
    for (const attribute of path) {
      if (probe === null || probe === undefined || typeof probe !== "object") {
        probe = undefined;
        break;
      }
      probe = (probe as Record<string, unknown>)[attribute];
    }
    const getter =
      probe !== null && probe !== undefined && typeof probe === "object"
        ? (probe as { getProviders?: unknown }).getProviders
        : undefined;
    if (typeof getter === "function") {
      try {
        const names = getter.call(probe) as unknown;
        if (isIterable(names)) return [...names].map((name) => String(name));
      } catch {
        return [];
      }
    }
  }
  return [];
}

function isIterable(value: unknown): value is Iterable<unknown> {
  return (
    value !== null && value !== undefined && typeof value === "object" && Symbol.iterator in value
  );
}

export const PROBE_TEXTS: readonly string[] = [
  "def probe() -> int:\n    return 0\n",
  "class Probe:\n    pass\n",
];

export function validateProbeVectors(
  vectors: readonly Uint8Array[],
  { dimension, count }: { dimension: number; count: number },
): void {
  if (vectors.length !== count) {
    throw new Error(`probe returned ${vectors.length} vectors for ${count} inputs`);
  }
  const expectedBytes = dimension * 4;
  for (const [index, packed] of vectors.entries()) {
    if (packed.byteLength !== expectedBytes) {
      throw new Error(
        `probe vector ${index} is ${Math.floor(packed.byteLength / 4)} wide, expected ${dimension}`,
      );
    }
    const row = new Float32Array(packed.buffer, packed.byteOffset, dimension);
    let sumSquares = 0;
    for (const value of row) {
      if (!Number.isFinite(value)) {
        throw new Error(`probe vector ${index} contains non-finite values`);
      }
      sumSquares += value * value;
    }
    const norm = Math.sqrt(sumSquares);
    if (norm < 0.9 || norm > 1.1) {
      throw new Error(`probe vector ${index} has norm ${norm.toFixed(4)}, expected ~1.0`);
    }
  }
}

export type EncodeFn = (text: string) => TokenEncoding;

export function planPassages(
  encode: EncodeFn | undefined,
  candidates: readonly PassageCandidate[],
  plan: SegmentPlan,
): TokenWindow[][] {
  if (encode === undefined) {
    return candidates.map((candidate) => [
      { startChar: 0, endChar: candidate.content.length, tokenCount: 0 },
    ]);
  }
  return planCandidateWindows(
    encode,
    candidates.map((candidate) => ({ prefix: candidate.prefix, content: candidate.content })),
    {
      maxTokens: plan.maxTokens,
      overlapTokens: plan.overlapTokens,
      maxWindows: plan.maxWindows,
    },
  );
}

export function embedWindows<Vector>(
  embed: (texts: string[]) => Vector[],
  candidates: readonly PassageCandidate[],
  windowsPerCandidate: readonly (readonly TokenWindow[])[],
  plan: SegmentPlan,
): [TokenWindow, Vector][][] {
  if (candidates.length === 0) return [];
  const owners: number[] = [];
  const windows: TokenWindow[] = [];
  const texts: string[] = [];
  for (const [index, candidate] of candidates.entries()) {
    const planned = windowsPerCandidate[index] ?? [];
    for (const window of planned) {
      owners.push(index);
      windows.push(window);
      texts.push(
        composePassage(candidate.prefix, candidate.content.slice(window.startChar, window.endChar)),
      );
    }
  }

  const results: [TokenWindow, Vector][][] = candidates.map(() => []);
  for (const batch of planMicrobatches(
    windows.map((window) => window.tokenCount),
    { maxItems: plan.maxItems, maxTokenProduct: plan.maxTokenProduct },
  )) {
    const vectors = embed(batch.map((position) => texts[position] ?? ""));
    for (const [offset, position] of batch.entries()) {
      const owner = owners[position];
      const window = windows[position];
      const vector = vectors[offset];
      if (owner === undefined || window === undefined || vector === undefined) continue;
      results[owner]?.push([window, vector]);
    }
  }
  for (const candidateResults of results) {
    candidateResults.sort(
      (left, right) => left[0].startChar - right[0].startChar || left[0].endChar - right[0].endChar,
    );
  }
  return results;
}

export function embedPlannedSegments<Vector>(
  encode: EncodeFn | undefined,
  embed: (texts: string[]) => Vector[],
  candidates: readonly PassageCandidate[],
  plan: SegmentPlan,
): [TokenWindow, Vector][][] {
  if (candidates.length === 0) return [];
  return embedWindows(embed, candidates, planPassages(encode, candidates, plan), plan);
}

export interface ModelIdentity {
  readonly modelId: string;
  readonly dimension: number;
}

export interface PassageEmbedder {
  embedPassages(texts: string[]): Promise<number[][]> | number[][];
}

export interface QueryEmbedder extends ModelIdentity {
  embedQuery(text: string): Promise<number[]> | number[];
}

export interface SegmentingEmbedder {
  planAndEmbed(
    candidates: readonly PassageCandidate[],
    plan: SegmentPlan,
  ): Promise<EmbeddedSegment[][]> | EmbeddedSegment[][];
}

export type Embedder = QueryEmbedder & PassageEmbedder;

export interface PassageModel {
  passageEmbed(texts: string[]): Iterable<ArrayLike<number>> | ArrayLike<number>[];
  readonly resolvedProviders?: readonly string[];
  readonly tokenizer?: { encode: (text: string) => unknown };
}

export interface ModelLoadOptions {
  cacheDirectory: string;
  offline: boolean;
  threads: number | undefined;
  enableCpuMemArena: boolean;
  modelId: string;
}

export type ModelFactory = (options: ModelLoadOptions) => PassageModel | Promise<PassageModel>;

export let modelFactory: ModelFactory | undefined;

export function setModelFactory(factory: ModelFactory | undefined): void {
  modelFactory = factory;
}

export class OnnxEmbedder implements Embedder, SegmentingEmbedder {
  readonly modelId = DEFAULT_MODEL;
  readonly dimension = DEFAULT_DIMENSION;
  readonly cacheDirectory: string;
  readonly offline: boolean;
  readonly threads: number | undefined;
  readonly enableCpuMemArena: boolean;
  private model: PassageModel | undefined;
  private loading: Promise<PassageModel> | undefined;

  constructor(
    cacheDirectory: string,
    {
      offline = false,
      threads,
      enableCpuMemArena = false,
    }: { offline?: boolean; threads?: number; enableCpuMemArena?: boolean } = {},
  ) {
    this.cacheDirectory = cacheDirectory;
    this.offline = offline;
    this.threads = threads;
    this.enableCpuMemArena = enableCpuMemArena;
  }

  async prepare(): Promise<void> {
    await this.getModel();
  }

  async embedPassages(texts: string[]): Promise<number[][]> {
    return vectorsFrom(this.embedRaw(await this.getModel(), texts));
  }

  async planAndEmbed(
    candidates: readonly PassageCandidate[],
    plan: SegmentPlan,
  ): Promise<EmbeddedSegment[][]> {
    const model = await this.getModel();
    const tokenizer = resolveTokenizer(model);
    const encode =
      tokenizer === undefined ? undefined : asEncodeFn(tokenizer.encode.bind(tokenizer));
    const planned = embedPlannedSegments(
      encode,
      (texts) => this.embedRaw(model, texts).map((row) => packVector(row)),
      candidates,
      plan,
    );
    return planned.map((segments) =>
      segments.map(([window, vector]) =>
        embeddedSegment(window.startChar, window.endChar, window.tokenCount, vector),
      ),
    );
  }

  async embedQuery(text: string): Promise<number[]> {
    const rows = await this.embedPassages([text]);
    const first = rows[0];
    if (first === undefined) throw new Error("query embedding returned no vectors");
    return first;
  }

  private embedRaw(model: PassageModel, texts: string[]): ArrayLike<number>[] {
    return [...model.passageEmbed(texts)];
  }

  private async getModel(): Promise<PassageModel> {
    if (this.model !== undefined) return this.model;
    if (this.loading !== undefined) return this.loading;
    this.loading = this.loadModel();
    try {
      this.model = await this.loading;
      return this.model;
    } finally {
      this.loading = undefined;
    }
  }

  private async loadModel(): Promise<PassageModel> {
    try {
      const factory = modelFactory ?? defaultModelFactory;
      return await factory({
        cacheDirectory: this.cacheDirectory,
        offline: this.offline,
        threads: this.threads,
        enableCpuMemArena: this.enableCpuMemArena,
        modelId: this.modelId,
      });
    } catch (error) {
      throw new CodeIndexingError(
        "MODEL_UNAVAILABLE",
        `Embedding model is unavailable: ${this.modelId}`,
        { model: this.modelId, offline: this.offline },
        { cause: error },
      );
    }
  }
}

async function defaultModelFactory(options: ModelLoadOptions): Promise<PassageModel> {
  const { DirectOnnxEmbedding } = await import("./direct-onnx.ts");
  return new DirectOnnxEmbedding(options.cacheDirectory, {
    offline: options.offline,
    threads: options.threads,
    enableCpuMemArena: options.enableCpuMemArena,
    providers: ["CPUExecutionProvider"],
    modelId: options.modelId,
  });
}

function asEncodeFn(encode: (text: string) => unknown): EncodeFn {
  return (text) => {
    const encoded = encode(text);
    if (encoded !== null && typeof encoded === "object" && "offsets" in encoded) {
      return encoded as TokenEncoding;
    }
    return { offsets: [] };
  };
}

function vectorsFrom(rows: Iterable<ArrayLike<number>>): number[][] {
  return [...rows].map((row) => Array.from(row));
}
