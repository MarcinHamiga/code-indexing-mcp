/**
 * Direct ONNX passage embedding.
 *
 * The implementation is deliberately model-specific. Index compatibility depends
 * on more than an ONNX file: the tokenizer, pooling, normalization, dimension,
 * and artifact revision all have to retain the semantics of the CPU query model.
 */

import fs from "node:fs";
import path from "node:path";
import { DEFAULT_MODEL } from "./embedding.ts";

export const DEFAULT_MODEL_ARTIFACT = "onnx/model.onnx";

const MODEL_FILES = [
  "config.json",
  "tokenizer.json",
  "tokenizer_config.json",
  "special_tokens_map.json",
  DEFAULT_MODEL_ARTIFACT,
];

export interface OnnxEncoding {
  ids: readonly number[];
  attentionMask: readonly number[];
  offsets?: readonly (readonly [number, number])[];
  specialTokensMask?: readonly number[];
}

export interface OnnxTokenizer {
  encodeBatch(documents: string[]): OnnxEncoding[];
  encode(text: string): OnnxEncoding;
}

export interface OnnxSessionInput {
  name: string;
}

export interface OnnxSession {
  getInputs(): readonly OnnxSessionInput[];
  getProviders(): readonly string[];
  run(
    outputNames: null,
    inputFeed: Record<string, ArrayLike<number | bigint>>,
    dimensions?: readonly [batch: number, sequence: number],
  ): unknown | Promise<unknown>;
}

export type SessionFactory = (
  modelPath: string,
  options: {
    providers: readonly string[];
    threads: number | undefined;
    enableCpuMemArena: boolean;
  },
) => OnnxSession | Promise<OnnxSession>;

export type SnapshotDownload = (options: {
  repoId: string;
  allowPatterns: string[];
  cacheDir: string;
  localFilesOnly: boolean;
}) => string | Promise<string>;

export type TokenizerBindings = {
  Tokenizer: { fromFile: (filePath: string) => LoadedTokenizer };
  AddedToken: new (values: Record<string, unknown>) => unknown;
};

export interface LoadedTokenizer {
  padding?: unknown;
  enableTruncation(options: { maxLength: number }): void;
  enablePadding(options: { padId: number; padToken: string }): void;
  addSpecialTokens(tokens: unknown[]): void;
  encodeBatch(documents: string[]): OnnxEncoding[];
  encode(text: string): OnnxEncoding;
}

export interface DirectOnnxOptions {
  offline: boolean;
  threads: number | undefined;
  enableCpuMemArena: boolean;
  providers: readonly string[];
  modelId?: string;
  accelerator?: string;
}

export let snapshotDownload: SnapshotDownload | undefined;
export let tokenizerBindings: TokenizerBindings | undefined;
export let resolveModelSnapshot: (
  cacheDirectory: string,
  options: { modelId?: string; offline: boolean },
) => string | Promise<string> = defaultResolveModelSnapshot;
export let loadTokenizer: typeof defaultLoadTokenizer = defaultLoadTokenizer;
export let createSession: SessionFactory = defaultCreateSession;
export let createWebgpuSession: typeof defaultCreateWebgpuSession = defaultCreateWebgpuSession;

export function configureDirectOnnx(overrides: {
  snapshotDownload?: SnapshotDownload | undefined;
  resolveModelSnapshot?: (
    cacheDirectory: string,
    options: { modelId?: string; offline: boolean },
  ) => string | Promise<string>;
  loadTokenizer?: typeof defaultLoadTokenizer;
  createSession?: SessionFactory;
  createWebgpuSession?: typeof defaultCreateWebgpuSession;
  onnxRuntimeBindings?: WebgpuOrtBindings | undefined;
  webgpuPluginBindings?: WebgpuPlugin | undefined;
}): void {
  if ("snapshotDownload" in overrides) snapshotDownload = overrides.snapshotDownload;
  if (overrides.resolveModelSnapshot) resolveModelSnapshot = overrides.resolveModelSnapshot;
  if (overrides.loadTokenizer) loadTokenizer = overrides.loadTokenizer;
  if (overrides.createSession) createSession = overrides.createSession;
  if (overrides.createWebgpuSession) createWebgpuSession = overrides.createWebgpuSession;
  if ("onnxRuntimeBindings" in overrides) onnxRuntimeBindings = overrides.onnxRuntimeBindings;
  if ("webgpuPluginBindings" in overrides) webgpuPluginBindings = overrides.webgpuPluginBindings;
}

export interface WebgpuOrtBindings {
  GraphOptimizationLevel: { ORT_ENABLE_ALL: unknown };
  SessionOptions: new () => WebgpuSessionOptions;
  registerExecutionProviderLibrary: (name: string, libraryPath: string) => void;
  getEpDevices: () => ReadonlyArray<{ ep_name: string; epName?: string }>;
  InferenceSession: new (
    modelPath: string,
    options: { sess_options: WebgpuSessionOptions },
  ) => OnnxSession;
}

export interface WebgpuSessionOptions {
  graphOptimizationLevel: unknown;
  enableCpuMemArena: boolean;
  intraOpNumThreads: number;
  interOpNumThreads: number;
  addProviderForDevices(devices: unknown[], options: Record<string, string>): void;
}

export interface WebgpuPlugin {
  getLibraryPath(): string;
  getEpName(): string;
}

export let onnxRuntimeBindings: WebgpuOrtBindings | undefined;
export let webgpuPluginBindings: WebgpuPlugin | undefined;

export async function defaultResolveModelSnapshot(
  cacheDirectory: string,
  { modelId = DEFAULT_MODEL, offline }: { modelId?: string; offline: boolean },
): Promise<string> {
  if (modelId !== DEFAULT_MODEL) {
    throw new Error(
      `The direct ONNX backend only supports the index model ${DEFAULT_MODEL}; got ${modelId}`,
    );
  }
  const download = snapshotDownload ?? huggingfaceSnapshotDownload;
  const resolved = await download({
    repoId: modelId,
    allowPatterns: [...MODEL_FILES],
    cacheDir: cacheDirectory,
    localFilesOnly: offline,
  });
  return resolved;
}

async function huggingfaceSnapshotDownload(options: {
  repoId: string;
  allowPatterns: string[];
  cacheDir: string;
  localFilesOnly: boolean;
}): Promise<string> {
  const { downloadFileToCacheDir } = await import("@huggingface/hub");
  const destRoot = path.join(
    options.cacheDir,
    `models--${options.repoId.replaceAll("/", "--")}`,
    "snapshots",
    "local",
  );
  const repositoryRoot = path.dirname(path.dirname(destRoot));
  const candidates = [destRoot];
  try {
    const revision = fs.readFileSync(path.join(repositoryRoot, "refs", "main"), "utf8").trim();
    if (revision !== "") candidates.push(path.join(repositoryRoot, "snapshots", revision));
  } catch {
    // A first download has no main ref yet.
  }
  const cached = candidates.find((root) =>
    options.allowPatterns.every((relative) => fs.existsSync(path.join(root, relative))),
  );
  if (cached !== undefined) {
    return cached;
  }
  if (options.localFilesOnly) {
    const missing = options.allowPatterns.find(
      (relative) => !fs.existsSync(path.join(destRoot, relative)),
    );
    throw new Error(`Offline snapshot is missing ${missing}`);
  }
  let snapshotRoot: string | undefined;
  for (const relative of options.allowPatterns) {
    const downloaded = await downloadFileToCacheDir({
      repo: options.repoId,
      path: relative,
      cacheDir: options.cacheDir,
    });
    let root = downloaded;
    for (const _segment of relative.split("/")) root = path.dirname(root);
    if (snapshotRoot !== undefined && snapshotRoot !== root) {
      throw new Error("Model files resolved to different Hugging Face snapshots");
    }
    snapshotRoot = root;
  }
  if (snapshotRoot === undefined) throw new Error("Model snapshot contains no requested files");
  return snapshotRoot;
}

function jsonObject(filePath: string): Record<string, unknown> {
  let value: unknown;
  try {
    value = JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch (error) {
    throw new Error(`Could not read model configuration at ${filePath}: ${error}`);
  }
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Model configuration at ${filePath} is not a JSON object`);
  }
  return value as Record<string, unknown>;
}

export function defaultLoadTokenizer(modelDirectory: string): OnnxTokenizer {
  const config = jsonObject(path.join(modelDirectory, "config.json"));
  const tokenizerConfig = jsonObject(path.join(modelDirectory, "tokenizer_config.json"));
  const specialTokens = jsonObject(path.join(modelDirectory, "special_tokens_map.json"));

  const lengths = [tokenizerConfig.model_max_length, tokenizerConfig.max_length]
    .filter((value): value is number => typeof value === "number" && value > 0)
    .map((value) => Math.trunc(value));
  if (lengths.length === 0) {
    throw new Error("Tokenizer config has no positive model_max_length or max_length");
  }
  const maxContext = Math.min(...lengths);

  if (tokenizerBindings === undefined) {
    return loadHuggingFaceTokenizer(modelDirectory, tokenizerConfig, maxContext);
  }

  const tokenizer = tokenizerBindings.Tokenizer.fromFile(
    path.join(modelDirectory, "tokenizer.json"),
  );
  tokenizer.enableTruncation({ maxLength: maxContext });
  if (!tokenizer.padding) {
    const padToken = tokenizerConfig.pad_token;
    if (typeof padToken !== "string") {
      throw new Error("Tokenizer config has no pad_token");
    }
    const padId = typeof config.pad_token_id === "number" ? Math.trunc(config.pad_token_id) : 0;
    tokenizer.enablePadding({ padId, padToken });
  }

  for (const rawToken of Object.values(specialTokens)) {
    if (typeof rawToken === "string") {
      tokenizer.addSpecialTokens([rawToken]);
    } else if (rawToken !== null && typeof rawToken === "object" && !Array.isArray(rawToken)) {
      tokenizer.addSpecialTokens([
        new tokenizerBindings.AddedToken(rawToken as Record<string, unknown>),
      ]);
    }
  }
  return tokenizer;
}

function loadHuggingFaceTokenizer(
  modelDirectory: string,
  tokenizerConfig: Record<string, unknown>,
  maxContext: number,
): OnnxTokenizer {
  interface Encoding {
    ids: number[];
    tokens: string[];
    attention_mask: number[];
  }
  interface Tokenizer {
    encode(text: string): Encoding;
    token_to_id(token: string): number | undefined;
    decoder: { convert_tokens_to_string(tokens: string[]): string } | null;
  }
  const loaded = import.meta.require("@huggingface/tokenizers") as {
    Tokenizer: new (tokenizer: object, config: object) => Tokenizer;
  };
  const tokenizer = new loaded.Tokenizer(
    jsonObject(path.join(modelDirectory, "tokenizer.json")),
    tokenizerConfig,
  );
  const padToken = tokenizerConfig.pad_token;
  if (typeof padToken !== "string") throw new Error("Tokenizer config has no pad_token");
  const padId = tokenizer.token_to_id(padToken) ?? 0;
  const specialIds = new Set(tokenizer.encode("").ids);

  const encode = (text: string): OnnxEncoding => {
    const raw = tokenizer.encode(text);
    const ids = raw.ids.slice(0, maxContext);
    const attentionMask = raw.attention_mask.slice(0, maxContext);
    const offsets: Array<readonly [number, number]> = [];
    const specialTokensMask: number[] = [];
    let cursor = 0;
    for (const [index, id] of ids.entries()) {
      const special = specialIds.has(id);
      specialTokensMask.push(special ? 1 : 0);
      if (special) {
        offsets.push([0, 0]);
        continue;
      }
      const fragment = tokenizer.decoder?.convert_tokens_to_string([raw.tokens[index] ?? ""]) ?? "";
      const start = cursor;
      cursor = Math.min(text.length, cursor + fragment.length);
      offsets.push([start, cursor]);
    }
    return { ids, attentionMask, offsets, specialTokensMask };
  };

  return {
    encode,
    encodeBatch: (documents) => {
      const rows = documents.map(encode);
      const sequence = Math.max(0, ...rows.map((row) => row.ids.length));
      return rows.map((row) => ({
        ...row,
        ids: [...row.ids, ...Array(sequence - row.ids.length).fill(padId)],
        attentionMask: [
          ...row.attentionMask,
          ...Array(sequence - row.attentionMask.length).fill(0),
        ],
      }));
    },
  };
}

export function meanPoolAndNormalize(
  modelOutput: number[][][],
  attentionMask: number[][],
): number[][] {
  const batch = modelOutput.length;
  const pooled: number[][] = [];
  for (let row = 0; row < batch; row++) {
    const tokens = modelOutput[row] ?? [];
    const mask = attentionMask[row] ?? [];
    const dim = tokens[0]?.length ?? 0;
    const summed = new Float32Array(dim);
    let count = 0;
    for (let token = 0; token < tokens.length; token++) {
      const weight = mask[token] ?? 0;
      count += weight;
      const values = tokens[token] ?? [];
      for (let axis = 0; axis < dim; axis++) {
        summed[axis] = (summed[axis] ?? 0) + (values[axis] ?? 0) * weight;
      }
    }
    const denom = Math.max(count, 1e-9);
    const rowValues = new Array<number>(dim);
    let sumSquares = 0;
    for (let axis = 0; axis < dim; axis++) {
      const value = (summed[axis] ?? 0) / denom;
      rowValues[axis] = value;
      sumSquares += value * value;
    }
    const norm = Math.max(Math.sqrt(sumSquares), 1e-12);
    pooled.push(rowValues.map((value) => value / norm));
  }
  return pooled;
}

export async function defaultCreateSession(
  modelPath: string,
  {
    providers,
    threads,
    enableCpuMemArena,
  }: { providers: readonly string[]; threads: number | undefined; enableCpuMemArena: boolean },
): Promise<OnnxSession> {
  const ort = import.meta.require("onnxruntime-node") as {
    InferenceSession: {
      create: (
        model: string,
        options: Record<string, unknown>,
      ) => Promise<{
        inputNames: string[];
        handler?: { getProviders?: () => string[] };
        run: (
          feeds: Record<string, unknown>,
        ) => Promise<Record<string, { data: Float32Array; dims: number[] }>>;
      }>;
    };
    Tensor: new (type: string, data: BigInt64Array | Int32Array, dims: number[]) => unknown;
  };
  const created = await ort.InferenceSession.create(modelPath, {
    executionProviders: providers.map(nodeExecutionProvider),
    graphOptimizationLevel: "all",
    enableCpuMemArena,
    ...(threads === undefined ? {} : { intraOpNumThreads: threads, interOpNumThreads: threads }),
  });
  return wrapOrtSession(created, ort.Tensor, providers);
}

function wrapOrtSession(
  session: {
    inputNames: string[];
    handler?: { getProviders?: () => string[] };
    run: (
      feeds: Record<string, unknown>,
    ) => Promise<Record<string, { data: Float32Array; dims: number[] }>>;
  },
  Tensor: new (type: string, data: BigInt64Array | Int32Array, dims: number[]) => unknown,
  requestedProviders: readonly string[],
): OnnxSession {
  return {
    getInputs: () => session.inputNames.map((name) => ({ name })),
    getProviders: () => session.handler?.getProviders?.() ?? [...requestedProviders],
    run: async (_outputNames, inputFeed, dimensions) => {
      const feeds: Record<string, unknown> = {};
      for (const [name, data] of Object.entries(inputFeed)) {
        const values = data instanceof BigInt64Array ? data : bigintFromInt32(toInt32(data));
        const batch = dimensions?.[0] ?? 1;
        const seq = dimensions?.[1] ?? data.length / batch;
        feeds[name] = new Tensor("int64", values, [batch, seq]);
      }
      const outputs = await session.run(feeds);
      const first = Object.values(outputs)[0];
      if (first === undefined) return [];
      return [reshape3(first.data, first.dims)];
    },
  };
}

function nodeExecutionProvider(provider: string): string {
  const names: Record<string, string> = {
    CPUExecutionProvider: "cpu",
    CUDAExecutionProvider: "cuda",
    DmlExecutionProvider: "dml",
    CoreMLExecutionProvider: "coreml",
    MIGraphXExecutionProvider: "migraphx",
    WebGpuExecutionProvider: "webgpu",
  };
  return names[provider] ?? provider;
}

function toInt32(data: ArrayLike<number | bigint>): Int32Array {
  if (data instanceof Int32Array) return data;
  const out = new Int32Array(data.length);
  for (let index = 0; index < data.length; index++) out[index] = Number(data[index] ?? 0);
  return out;
}

function bigintFromInt32(data: Int32Array): BigInt64Array {
  const out = new BigInt64Array(data.length);
  for (let index = 0; index < data.length; index++) out[index] = BigInt(data[index] ?? 0);
  return out;
}

function reshape3(data: Float32Array, dims: number[]): number[][][] {
  const batch = dims[0] ?? 1;
  const seq = dims[1] ?? 1;
  const hidden = dims[2] ?? data.length / (batch * seq);
  const out: number[][][] = [];
  let offset = 0;
  for (let row = 0; row < batch; row++) {
    const tokens: number[][] = [];
    for (let token = 0; token < seq; token++) {
      tokens.push(Array.from(data.subarray(offset, offset + hidden)));
      offset += hidden;
    }
    out.push(tokens);
  }
  return out;
}

export async function defaultCreateWebgpuSession(
  modelPath: string,
  { threads, enableCpuMemArena }: { threads: number | undefined; enableCpuMemArena: boolean },
): Promise<[OnnxSession, string]> {
  const ort = onnxRuntimeBindings;
  const plugin = webgpuPluginBindings;
  if (ort === undefined || plugin === undefined) {
    const session = await defaultCreateSession(modelPath, {
      providers: ["WebGpuExecutionProvider", "CPUExecutionProvider"],
      threads,
      enableCpuMemArena,
    });
    return [session, "WebGpuExecutionProvider"];
  }
  const provider = String(plugin.getEpName());
  ort.registerExecutionProviderLibrary("code-indexing-mcp_webgpu_ep", plugin.getLibraryPath());
  const devices = ort
    .getEpDevices()
    .filter((device) => String(device.ep_name ?? device.epName) === provider);
  if (devices.length === 0) {
    throw new Error("The WebGPU plugin registered but exposed no WebGPU device");
  }
  const options = new ort.SessionOptions();
  options.graphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_ALL;
  options.enableCpuMemArena = enableCpuMemArena;
  if (threads !== undefined) {
    options.intraOpNumThreads = threads;
    options.interOpNumThreads = threads;
  }
  options.addProviderForDevices([...devices], {});
  const session = new ort.InferenceSession(modelPath, { sess_options: options });
  return [session, provider];
}

export class DirectOnnxEmbedding {
  readonly tokenizer: OnnxTokenizer;
  readonly model: OnnxSession;
  readonly resolvedProviders: readonly string[];

  static async create(
    cacheDirectory: string,
    options: DirectOnnxOptions,
  ): Promise<DirectOnnxEmbedding> {
    const modelDirectory = await resolveModelSnapshot(cacheDirectory, {
      offline: options.offline,
      ...(options.modelId === undefined ? {} : { modelId: options.modelId }),
    });
    const modelPath = path.join(modelDirectory, DEFAULT_MODEL_ARTIFACT);
    if (!fs.existsSync(modelPath) || !fs.statSync(modelPath).isFile()) {
      throw new Error(`The model snapshot has no ONNX artifact at ${modelPath}`);
    }
    const tokenizer = loadTokenizer(modelDirectory);
    let model: OnnxSession;
    if (options.accelerator === "webgpu") {
      const [session, pluginProvider] = await createWebgpuSession(modelPath, {
        threads: options.threads,
        enableCpuMemArena: options.enableCpuMemArena,
      });
      model = session;
      if (options.providers.length > 0 && pluginProvider !== options.providers[0]) {
        throw new Error(
          `The WebGPU plugin registered ${pluginProvider}, expected ${options.providers[0]}`,
        );
      }
    } else {
      model = await createSession(modelPath, {
        providers: options.providers,
        threads: options.threads,
        enableCpuMemArena: options.enableCpuMemArena,
      });
    }
    return new DirectOnnxEmbedding(tokenizer, model);
  }

  private constructor(tokenizer: OnnxTokenizer, model: OnnxSession) {
    this.tokenizer = tokenizer;
    this.model = model;
    this.resolvedProviders = [...new Set(this.model.getProviders().map((name) => String(name)))];
  }

  async passageEmbed(documents: string | Iterable<string>): Promise<number[][]> {
    const texts = typeof documents === "string" ? [documents] : [...documents];
    if (texts.length === 0) return [];
    const encoded = this.tokenizer.encodeBatch(texts);
    const seq = encoded[0]?.ids.length ?? 0;
    const inputIds = flattenInt64(
      encoded.map((row) => row.ids),
      seq,
    );
    const attentionMask = flattenInt64(
      encoded.map((row) => row.attentionMask),
      seq,
    );
    const inputNames = new Set(this.model.getInputs().map((item) => item.name));
    const inputs: Record<string, BigInt64Array> = { input_ids: inputIds };
    if (inputNames.has("attention_mask")) inputs.attention_mask = attentionMask;
    if (inputNames.has("token_type_ids")) {
      inputs.token_type_ids = new BigInt64Array(inputIds.length);
    }
    const outputs = await this.model.run(null, inputs, [texts.length, seq]);
    if (!Array.isArray(outputs) || outputs.length === 0) {
      throw new Error("The ONNX model returned no outputs");
    }
    const hidden = tensorFromRun(outputs[0]);
    return meanPoolAndNormalize(
      hidden,
      encoded.map((row) => [...row.attentionMask]),
    );
  }
}

function flattenInt64(rows: readonly (readonly number[])[], seq: number): BigInt64Array {
  const out = new BigInt64Array(rows.length * seq);
  for (const [row, values] of rows.entries()) {
    for (let index = 0; index < seq; index++) {
      out[row * seq + index] = BigInt(values[index] ?? 0);
    }
  }
  return out;
}

function tensorFromRun(output: unknown): number[][][] {
  if (Array.isArray(output) && Array.isArray(output[0]) && Array.isArray(output[0][0])) {
    return output as number[][][];
  }
  return [];
}
