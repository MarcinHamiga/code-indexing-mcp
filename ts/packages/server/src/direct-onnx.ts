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
  run(outputNames: null, inputFeed: Record<string, ArrayLike<number | bigint>>): unknown;
}

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

export let snapshotDownload: SnapshotDownload | undefined;
export let tokenizerBindings: TokenizerBindings | undefined;
export let resolveModelSnapshot: (
  cacheDirectory: string,
  options: { modelId?: string; offline: boolean },
) => string | Promise<string> = defaultResolveModelSnapshot;
export let loadTokenizer: typeof defaultLoadTokenizer = defaultLoadTokenizer;
export let createSession: typeof defaultCreateSession = defaultCreateSession;
export let createWebgpuSession: typeof defaultCreateWebgpuSession = defaultCreateWebgpuSession;

export function configureDirectOnnx(overrides: {
  snapshotDownload?: SnapshotDownload | undefined;
  resolveModelSnapshot?: (
    cacheDirectory: string,
    options: { modelId?: string; offline: boolean },
  ) => string | Promise<string>;
  loadTokenizer?: typeof defaultLoadTokenizer;
  createSession?: typeof defaultCreateSession;
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
  const { downloadFile, listFiles } = await import("@huggingface/hub");
  const destRoot = path.join(
    options.cacheDir,
    `models--${options.repoId.replaceAll("/", "--")}`,
    "snapshots",
    "local",
  );
  fs.mkdirSync(destRoot, { recursive: true });
  const files = options.localFilesOnly
    ? options.allowPatterns
    : await collectHubFiles(listFiles, options.repoId, options.allowPatterns);
  for (const relative of files) {
    const dest = path.join(destRoot, relative);
    if (options.localFilesOnly && fs.existsSync(dest)) continue;
    if (options.localFilesOnly) {
      throw new Error(`Offline snapshot is missing ${relative}`);
    }
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    const blob = await downloadFile({ repo: options.repoId, path: relative });
    if (blob === null) continue;
    const buffer = Buffer.from(await blob.arrayBuffer());
    fs.writeFileSync(dest, buffer);
  }
  return destRoot;
}

async function collectHubFiles(
  listFiles: (options: { repo: string }) => AsyncGenerator<{ path: string }>,
  repoId: string,
  allowPatterns: string[],
): Promise<string[]> {
  const allowed = new Set(allowPatterns);
  const found: string[] = [];
  for await (const file of listFiles({ repo: repoId })) {
    if (allowed.has(file.path)) found.push(file.path);
  }
  return found.length > 0 ? found : allowPatterns;
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
  const bindings = tokenizerBindings ?? nativeTokenizerBindings();
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

  const tokenizer = bindings.Tokenizer.fromFile(path.join(modelDirectory, "tokenizer.json"));
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
      tokenizer.addSpecialTokens([new bindings.AddedToken(rawToken as Record<string, unknown>)]);
    }
  }
  return tokenizer;
}

function nativeTokenizerBindings(): TokenizerBindings {
  const loaded = import.meta.require("@huggingface/tokenizers") as {
    Tokenizer: { fromFile: (filePath: string) => LoadedTokenizer };
    AddedToken: new (values: Record<string, unknown>) => unknown;
  };
  return { Tokenizer: loaded.Tokenizer, AddedToken: loaded.AddedToken };
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

export function defaultCreateSession(
  modelPath: string,
  {
    providers,
    threads,
    enableCpuMemArena,
  }: { providers: readonly string[]; threads: number | undefined; enableCpuMemArena: boolean },
): OnnxSession {
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
  const created = waitFor(
    ort.InferenceSession.create(modelPath, {
      executionProviders: [...providers],
      graphOptimizationLevel: "all",
      enableCpuMemArena,
      ...(threads === undefined ? {} : { intraOpNumThreads: threads, interOpNumThreads: threads }),
    }),
  );
  return wrapOrtSession(created, ort.Tensor);
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
): OnnxSession {
  return {
    getInputs: () => session.inputNames.map((name) => ({ name })),
    getProviders: () => session.handler?.getProviders?.() ?? [],
    run: (_outputNames, inputFeed) => {
      const feeds: Record<string, unknown> = {};
      for (const [name, data] of Object.entries(inputFeed)) {
        const values = data instanceof BigInt64Array ? data : bigintFromInt32(toInt32(data));
        const batch = inferBatch(toInt32(data));
        const seq = data.length / batch;
        feeds[name] = new Tensor("int64", values, [batch, seq]);
      }
      const outputs = waitFor(session.run(feeds));
      const first = Object.values(outputs)[0];
      if (first === undefined) return [];
      return [reshape3(first.data, first.dims)];
    },
  };
}

function inferBatch(_data: Int32Array | BigInt64Array): number {
  return 1;
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

function waitFor<T>(value: T | Promise<T>): T {
  if (value !== null && typeof value === "object" && "then" in (value as object)) {
    throw new Error("ONNX session construction must be injected in tests; live create is async");
  }
  return value as T;
}

export function defaultCreateWebgpuSession(
  modelPath: string,
  { threads, enableCpuMemArena }: { threads: number | undefined; enableCpuMemArena: boolean },
): [OnnxSession, string] {
  const ort = onnxRuntimeBindings;
  const plugin = webgpuPluginBindings;
  if (ort === undefined || plugin === undefined) {
    throw new Error("The WebGPU plugin is not available in this Phase 4 CPU build");
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

  constructor(
    cacheDirectory: string,
    {
      offline,
      threads,
      enableCpuMemArena,
      providers,
      modelId = DEFAULT_MODEL,
      accelerator = "",
    }: {
      offline: boolean;
      threads: number | undefined;
      enableCpuMemArena: boolean;
      providers: readonly string[];
      modelId?: string;
      accelerator?: string;
    },
  ) {
    const modelDirectory = mustString(
      resolveModelSnapshot(cacheDirectory, { modelId, offline }),
      "resolveModelSnapshot",
    );
    const modelPath = path.join(modelDirectory, DEFAULT_MODEL_ARTIFACT);
    if (!fs.existsSync(modelPath) || !fs.statSync(modelPath).isFile()) {
      throw new Error(`The model snapshot has no ONNX artifact at ${modelPath}`);
    }
    this.tokenizer = loadTokenizer(modelDirectory);
    if (accelerator === "webgpu") {
      const [session, pluginProvider] = createWebgpuSession(modelPath, {
        threads,
        enableCpuMemArena,
      });
      this.model = session;
      if (providers.length > 0 && pluginProvider !== providers[0]) {
        throw new Error(`The WebGPU plugin registered ${pluginProvider}, expected ${providers[0]}`);
      }
    } else {
      this.model = createSession(modelPath, { providers, threads, enableCpuMemArena });
    }
    this.resolvedProviders = [...new Set(this.model.getProviders().map((name) => String(name)))];
  }

  passageEmbed(documents: string | Iterable<string>): number[][] {
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
    const outputs = this.model.run(null, inputs);
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

function mustString(value: string | Promise<string>, name: string): string {
  if (typeof value === "string") return value;
  throw new Error(`${name} returned a Promise; inject a sync implementation`);
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
