/** Metal passage embedding through MLX for Apple Silicon. */

import fs from "node:fs";
import path from "node:path";
import { onnx } from "onnx-proto";
import { MLX_PROVIDER } from "./backends.ts";
import {
  DEFAULT_MODEL_ARTIFACT,
  type OnnxTokenizer,
  loadTokenizer,
  resolveModelSnapshot,
} from "./direct-onnx.ts";
import { DEFAULT_MODEL } from "./embedding.ts";

const WEIGHT_LAYOUT_VERSION = 1;
const MASK_FILL = -1e9;
const ALIBI_SLOPE_NODE = "/encoder/Mul_1";

type Mx = typeof import("@frost-beta/mlx").mx;
type MxArray = InstanceType<Mx["array"]>;
type Weights = Record<string, MxArray>;
type ProtoDimension = number | { toNumber(): number };
interface ProtoTensor {
  name: string;
  dataType: number;
  dataLocation: number;
  dims: ProtoDimension[];
  rawData: Uint8Array;
}
interface ProtoNode {
  name: string;
  input: string[];
  output: string[];
  attribute: Array<{ t?: ProtoTensor }>;
}
interface ProtoModel {
  graph: { initializer: ProtoTensor[]; node: ProtoNode[] };
}

export interface MlxModelConfig {
  hiddenSize: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  intermediateSize: number;
  layerNormEps: number;
}

export function readMlxModelConfig(modelDirectory: string): MlxModelConfig {
  const configPath = path.join(modelDirectory, "config.json");
  let document: Record<string, unknown>;
  try {
    document = JSON.parse(fs.readFileSync(configPath, "utf8")) as Record<string, unknown>;
  } catch (error) {
    throw new Error(`Could not read model configuration at ${configPath}: ${error}`);
  }
  const architecture: Record<string, string> = {
    model_type: "bert",
    position_embedding_type: "alibi",
    feed_forward_type: "geglu",
    hidden_act: "gelu",
    emb_pooler: "mean",
  };
  for (const [field, expected] of Object.entries(architecture)) {
    if (document[field] !== expected) {
      throw new Error(
        `The MLX backend reproduces ${field}=${JSON.stringify(expected)}; this model declares ${JSON.stringify(document[field])}`,
      );
    }
  }
  const positive = (field: string): number => {
    const value = document[field];
    if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
      throw new Error(`Model configuration has no positive ${field}`);
    }
    return value;
  };
  const hiddenSize = positive("hidden_size");
  const numAttentionHeads = positive("num_attention_heads");
  if (hiddenSize % numAttentionHeads !== 0) {
    throw new Error(
      `hidden_size ${hiddenSize} does not divide evenly across ${numAttentionHeads} attention heads`,
    );
  }
  const layerNormEps = document.layer_norm_eps;
  if (typeof layerNormEps !== "number" || layerNormEps <= 0) {
    throw new Error("Model configuration has no positive layer_norm_eps");
  }
  return {
    hiddenSize,
    numHiddenLayers: positive("num_hidden_layers"),
    numAttentionHeads,
    intermediateSize: positive("intermediate_size"),
    layerNormEps,
  };
}

function convertedWeightsPath(cacheDirectory: string, modelDirectory: string): string {
  return path.join(
    cacheDirectory,
    "mlx",
    `${path.basename(modelDirectory) || "unknown"}-jina-v${WEIGHT_LAYOUT_VERSION}-f32.safetensors`,
  );
}

function tensorValues(
  tensor: ProtoTensor | undefined,
  shape: readonly number[],
  description: string,
): Float32Array {
  if (tensor === undefined) throw new Error(`The ONNX artifact has no ${description}`);
  if (tensor.dataType !== 1) throw new Error(`The ${description} is not a FLOAT tensor`);
  if (tensor.dataLocation !== 0) throw new Error(`The ${description} is stored outside model.onnx`);
  const actual = tensor.dims.map((value) => (typeof value === "number" ? value : value.toNumber()));
  if (actual.length !== shape.length || actual.some((value, index) => value !== shape[index])) {
    throw new Error(`The ${description} is ${actual.join("x")}, expected ${shape.join("x")}`);
  }
  const raw = tensor.rawData as Uint8Array;
  if (raw.byteLength !== shape.reduce((total, value) => total * value, 1) * 4) {
    throw new Error(`The ${description} has an invalid FLOAT payload`);
  }
  return new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
}

function layerSources(index: number, config: MlxModelConfig): Record<string, [string, number[]]> {
  const hidden = config.hiddenSize;
  const parameter = `encoder.layer.${index}`;
  const prefix = `/encoder/layer.${index}/attention`;
  return {
    "query.bias": [`${parameter}.attention.self.query.bias`, [hidden]],
    "key.bias": [`${parameter}.attention.self.key.bias`, [hidden]],
    "value.bias": [`${parameter}.attention.self.value.bias`, [hidden]],
    "attention_output.bias": [`${parameter}.attention.output.dense.bias`, [hidden]],
    "down.bias": [`${parameter}.mlp.down_layer.bias`, [hidden]],
    "norm_q.weight": [`${parameter}.attention.self.layer_norm_q.weight`, [hidden]],
    "norm_q.bias": [`${parameter}.attention.self.layer_norm_q.bias`, [hidden]],
    "norm_k.weight": [`${parameter}.attention.self.layer_norm_k.weight`, [hidden]],
    "norm_k.bias": [`${parameter}.attention.self.layer_norm_k.bias`, [hidden]],
    "attention_norm.weight": [`${parameter}.attention.output.LayerNorm.weight`, [hidden]],
    "attention_norm.bias": [`${parameter}.attention.output.LayerNorm.bias`, [hidden]],
    "norm_1.weight": [`${parameter}.layer_norm_1.weight`, [hidden]],
    "norm_1.bias": [`${parameter}.layer_norm_1.bias`, [hidden]],
    "norm_2.weight": [`${parameter}.layer_norm_2.weight`, [hidden]],
    "norm_2.bias": [`${parameter}.layer_norm_2.bias`, [hidden]],
    "query.weight": [`${prefix}/self/query/MatMul`, [hidden, hidden]],
    "key.weight": [`${prefix}/self/key/MatMul`, [hidden, hidden]],
    "value.weight": [`${prefix}/self/value/MatMul`, [hidden, hidden]],
    "attention_output.weight": [`${prefix}/output/dense/MatMul`, [hidden, hidden]],
    "up_gated.weight": [
      `${prefix}/mlp/up_gated_layer/MatMul`,
      [hidden, 2 * config.intermediateSize],
    ],
    "down.weight": [`${prefix}/mlp/down_layer/MatMul`, [config.intermediateSize, hidden]],
  };
}

function extractWeights(modelPath: string, config: MlxModelConfig): Record<string, Float32Array> {
  const model = onnx.ModelProto.decode(fs.readFileSync(modelPath)) as unknown as ProtoModel;
  const tensors = new Map<string, ProtoTensor>(
    model.graph.initializer.map((tensor) => [tensor.name, tensor]),
  );
  const nodes = new Map<string, ProtoNode>(model.graph.node.map((node) => [node.name, node]));
  const outputs = new Map<string, ProtoNode>();
  for (const node of model.graph.node) for (const output of node.output) outputs.set(output, node);
  const take = (name: string | undefined, shape: number[], description: string) =>
    tensorValues(
      name === undefined ? undefined : tensors.get(name),
      shape,
      `${description} tensor ${JSON.stringify(name)}`,
    );
  const byNode = (nodeName: string, shape: number[], description: string) => {
    const node = nodes.get(nodeName);
    return take(node?.input?.[1], shape, description);
  };
  const word = tensors.get("embeddings.word_embeddings.weight");
  const tokenType = tensors.get("embeddings.token_type_embeddings.weight");
  const dimensions = (tensor: ProtoTensor | undefined) =>
    tensor?.dims.map((value) => (typeof value === "number" ? value : value.toNumber())) ?? [];
  const wordShape = dimensions(word);
  const tokenTypeShape = dimensions(tokenType);
  const vocabularySize = wordShape[0];
  const tokenTypeCount = tokenTypeShape[0];
  if (vocabularySize === undefined || tokenTypeCount === undefined) {
    throw new Error("The ONNX artifact has no two-dimensional embedding tables");
  }
  const values: Record<string, Float32Array> = {
    "embeddings.word_embeddings": take(
      "embeddings.word_embeddings.weight",
      [vocabularySize, config.hiddenSize],
      "word embedding",
    ),
    "embeddings.token_type": take(
      "embeddings.token_type_embeddings.weight",
      [tokenTypeCount, config.hiddenSize],
      "token type embedding",
    ),
    "embeddings.norm.weight": take(
      "embeddings.LayerNorm.weight",
      [config.hiddenSize],
      "embedding normalization",
    ),
    "embeddings.norm.bias": take(
      "embeddings.LayerNorm.bias",
      [config.hiddenSize],
      "embedding normalization",
    ),
  };
  const alibi = nodes.get(ALIBI_SLOPE_NODE);
  const constantOutput = alibi?.input[0];
  const constant = constantOutput === undefined ? undefined : outputs.get(constantOutput);
  values.alibi_slopes = tensorValues(
    constant?.attribute?.[0]?.t,
    [config.numAttentionHeads, 1, 1],
    "ALiBi slopes",
  );
  for (let index = 0; index < config.numHiddenLayers; index++) {
    for (const [name, [source, shape]] of Object.entries(layerSources(index, config))) {
      values[`layers.${index}.${name}`] = source.includes("/MatMul")
        ? byNode(source, shape, `layer ${index} ${name}`)
        : take(source, shape, `layer ${index} ${name}`);
    }
  }
  return values;
}

async function loadMlx(): Promise<Mx> {
  try {
    const module = await import("@frost-beta/mlx");
    if (!module.mx.metal.isAvailable()) throw new Error("Metal is unavailable");
    return module.mx;
  } catch (error) {
    throw new Error(`MLX Metal is unavailable: ${error}`);
  }
}

async function ensureConvertedWeights(
  modelDirectory: string,
  cacheDirectory: string,
  config: MlxModelConfig,
): Promise<string> {
  const target = convertedWeightsPath(cacheDirectory, modelDirectory);
  if (fs.existsSync(target)) return target;
  fs.mkdirSync(path.dirname(target), { recursive: true });
  const mx = await loadMlx();
  const extracted = extractWeights(path.join(modelDirectory, DEFAULT_MODEL_ARTIFACT), config);
  const weights: Record<string, MxArray> = {};
  for (const [name, values] of Object.entries(extracted))
    weights[name] = mx.array(values, mx.float32);
  const temporary = `${target}.${process.pid}.tmp.safetensors`;
  try {
    mx.saveSafetensors(temporary, weights);
    fs.renameSync(temporary, target);
  } finally {
    fs.rmSync(temporary, { force: true });
  }
  return target;
}

function weight(weights: Weights, name: string): MxArray {
  const value = weights[name];
  if (value === undefined) throw new Error(`The converted MLX weights have no ${name}`);
  return value;
}

function forward(
  mx: Mx,
  config: MlxModelConfig,
  weights: Weights,
  inputIds: MxArray,
  attentionMask: MxArray,
): MxArray {
  const batch = inputIds.shape[0];
  const sequence = inputIds.shape[1];
  if (batch === undefined || sequence === undefined) {
    throw new Error("MLX input ids must have batch and sequence dimensions");
  }
  const heads = config.numAttentionHeads;
  const headDim = config.hiddenSize / heads;
  const normalize = (value: MxArray, name: string) =>
    mx.fast.layerNorm(
      value,
      weight(weights, `${name}.weight`),
      weight(weights, `${name}.bias`),
      config.layerNormEps,
    );
  let hidden = mx.add(
    mx
      .take(weight(weights, "embeddings.word_embeddings"), inputIds.reshape(-1), 0)
      .reshape(batch, sequence, config.hiddenSize),
    weight(weights, "embeddings.token_type").index(0),
  );
  hidden = normalize(hidden, "embeddings.norm");
  const positions = mx.arange(sequence);
  const distance = mx
    .abs(mx.subtract(mx.expandDims(positions, 1), mx.expandDims(positions, 0)))
    .astype(mx.float32);
  const bias = mx.add(
    mx.multiply(weight(weights, "alibi_slopes"), distance),
    mx.multiply(
      mx.subtract(1, attentionMask.astype(mx.float32)).reshape(batch, 1, 1, sequence),
      MASK_FILL,
    ),
  );
  const project = (name: string, value: MxArray) =>
    mx.add(mx.matmul(value, weight(weights, `${name}.weight`)), weight(weights, `${name}.bias`));
  const splitHeads = (value: MxArray) =>
    value.reshape(batch, sequence, heads, headDim).transpose(0, 2, 1, 3);
  for (let index = 0; index < config.numHiddenLayers; index++) {
    const layer = `layers.${index}`;
    const query = normalize(project(`${layer}.query`, hidden), `${layer}.norm_q`);
    const key = normalize(project(`${layer}.key`, hidden), `${layer}.norm_k`);
    const value = project(`${layer}.value`, hidden);
    const context = mx.fast
      .scaledDotProductAttention(
        splitHeads(query),
        splitHeads(key),
        splitHeads(value),
        1 / Math.sqrt(headDim),
        bias,
      )
      .transpose(0, 2, 1, 3)
      .reshape(batch, sequence, config.hiddenSize);
    const attention = normalize(
      mx.add(project(`${layer}.attention_output`, context), hidden),
      `${layer}.attention_norm`,
    );
    const residual = normalize(mx.add(hidden, attention), `${layer}.norm_1`);
    const gated = mx.matmul(residual, weight(weights, `${layer}.up_gated.weight`));
    const up = gated.index("...", mx.Slice(0, config.intermediateSize));
    const gate = gated.index("...", mx.Slice(config.intermediateSize));
    const gelu = mx.multiply(
      gate,
      mx.multiply(0.5, mx.add(1, mx.erf(mx.divide(gate, Math.sqrt(2))))),
    );
    hidden = normalize(
      mx.add(
        residual,
        mx.add(
          mx.matmul(mx.multiply(up, gelu), weight(weights, `${layer}.down.weight`)),
          weight(weights, `${layer}.down.bias`),
        ),
      ),
      `${layer}.norm_2`,
    );
  }
  return hidden;
}

export class MlxEmbedding {
  readonly tokenizer: OnnxTokenizer;
  readonly resolvedProviders = [MLX_PROVIDER];
  private readonly mx: Mx;
  private readonly config: MlxModelConfig;
  private readonly weights: Weights;

  private constructor(mx: Mx, config: MlxModelConfig, weights: Weights, tokenizer: OnnxTokenizer) {
    this.mx = mx;
    this.config = config;
    this.weights = weights;
    this.tokenizer = tokenizer;
  }

  static async create(
    cacheDirectory: string,
    { offline, modelId = DEFAULT_MODEL }: { offline: boolean; modelId?: string },
  ): Promise<MlxEmbedding> {
    if (modelId !== DEFAULT_MODEL)
      throw new Error(
        `The MLX backend only supports the index model ${DEFAULT_MODEL}; got ${modelId}`,
      );
    const modelDirectory = await resolveModelSnapshot(cacheDirectory, { offline, modelId });
    const config = readMlxModelConfig(modelDirectory);
    const weightsPath = await ensureConvertedWeights(modelDirectory, cacheDirectory, config);
    const mx = await loadMlx();
    const weights = mx.load(weightsPath) as unknown as Weights;
    const embedding = new MlxEmbedding(mx, config, weights, loadTokenizer(modelDirectory));
    mx.eval(forward(mx, config, weights, mx.zeros([1, 2], mx.int64), mx.ones([1, 2], mx.int64)));
    return embedding;
  }

  async passageEmbed(documents: string | Iterable<string>): Promise<number[][]> {
    const texts = typeof documents === "string" ? [documents] : [...documents];
    if (texts.length === 0) return [];
    const encoded = this.tokenizer.encodeBatch(texts);
    const ids = this.mx.array(
      encoded.map((row) => [...row.ids]),
      this.mx.int64,
    );
    const mask = this.mx.array(
      encoded.map((row) => [...row.attentionMask]),
      this.mx.int64,
    );
    const hidden = forward(this.mx, this.config, this.weights, ids, mask);
    const expandedMask = mask
      .astype(this.mx.float32)
      .reshape(mask.shape[0] ?? 0, mask.shape[1] ?? 0, 1);
    const pooled = this.mx.divide(
      this.mx.sum(this.mx.multiply(hidden, expandedMask), 1),
      this.mx.maximum(this.mx.sum(expandedMask, 1), 1e-9),
    );
    const normalized = this.mx.divide(
      pooled,
      this.mx.maximum(this.mx.linalg.norm(pooled, 2, 1, true), 1e-12),
    );
    this.mx.eval(normalized);
    const rows = normalized.tolist() as number[][];
    this.mx.clearCache();
    return rows;
  }
}
