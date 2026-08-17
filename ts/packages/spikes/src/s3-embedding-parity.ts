/**
 * S3 -- Embedding parity.
 *
 * Embed the probe corpus through onnxruntime-node plus HF tokenizers, and
 * score it with the ported acceptance metrics against Python-produced vectors.
 * This decides §5.2 (and therefore decision D1): if parity holds, migrated
 * installs keep their indexes; if it does not, the index model revision is
 * bumped and the existing staleness machinery rebuilds on first use.
 *
 * The spike is in two halves so the cheap one always runs:
 *
 *  1. Metrics parity -- does the TypeScript port of `acceptance.py` compute the
 *     same numbers as the Python original on a committed fixture? A parity
 *     verdict from a miscomputed metric is worse than no verdict, so this is
 *     checked first and needs nothing but the repository.
 *  2. Vector parity -- the real comparison, which needs the ~640 MB model
 *     artifact and a reference vector file produced by the Python build. Both
 *     are opt-in through the environment, and the check skips (loudly) without
 *     them rather than silently passing.
 *
 * To run the second half:
 *   .venv/bin/python scripts/write_python_vectors.py /tmp/reference.json
 *   S3_REFERENCE_VECTORS=/tmp/reference.json \
 *   S3_MODEL_DIR=~/.cache/code-indexing-mcp/models/jina-v2-base-code \
 *     bun run src/s3-embedding-parity.ts
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { cosineRows, meanPoolAndNormalize, topKOverlap } from "./acceptance.ts";
import { Spike, repoRoot } from "./harness.ts";

/** The promotion gate `test_accelerator_acceptance.py` already enforces. */
const MIN_COSINE = 0.999;
const MIN_TOP_K_OVERLAP = 0.99;
const TOP_K = 5;

/** `embedding.py`: the model the index is built with, and its dimension. */
const MODEL_ID = "jinaai/jina-embeddings-v2-base-code";
const DIMENSION = 768;
/** `direct_onnx.py::DEFAULT_MODEL_ARTIFACT`. */
const MODEL_ARTIFACT = join("onnx", "model.onnx");

const spike = new Spike("s3", "Embedding parity");
spike.header();

await spike.check("the ported acceptance metrics match the Python originals", () => {
  const fixture = JSON.parse(
    readFileSync(
      join(repoRoot(), "ts", "packages", "spikes", "fixtures", "acceptance-parity.json"),
      "utf8",
    ),
  ) as {
    reference: number[][];
    candidate: number[][];
    queries: number[][];
    expected: {
      cosineRows: number[];
      topKOverlap5: number;
      topKOverlap1: number;
    };
  };

  const cosine = cosineRows(fixture.reference, fixture.candidate);
  // float32 in numpy against float64 here, so compare at float32 resolution
  // rather than demanding bit equality of two different precisions.
  const tolerance = 1e-6;
  const worst = Math.max(
    ...cosine.map((value, index) => Math.abs(value - (fixture.expected.cosineRows[index] ?? 0))),
  );
  if (worst > tolerance) {
    throw new Error(`cosineRows diverges from Python by ${worst.toExponential(2)}`);
  }

  const overlap5 = topKOverlap(fixture.queries, fixture.reference, fixture.candidate, 5);
  const overlap1 = topKOverlap(fixture.queries, fixture.reference, fixture.candidate, 1);
  if (overlap5 !== fixture.expected.topKOverlap5 || overlap1 !== fixture.expected.topKOverlap1) {
    throw new Error(
      `topKOverlap diverges: got ${overlap5}/${overlap1}, ` +
        `Python produced ${fixture.expected.topKOverlap5}/${fixture.expected.topKOverlap1}`,
    );
  }
  return `cosineRows within ${worst.toExponential(2)}; topKOverlap identical at k=1 and k=5`;
});

interface ReferenceVectors {
  readonly modelId: string;
  readonly dimension: number;
  readonly documents: readonly string[];
  readonly queries: readonly string[];
  readonly documentVectors: readonly (readonly number[])[];
  readonly queryVectors: readonly (readonly number[])[];
}

await spike.check("TypeScript vectors match Python vectors on the probe corpus", async () => {
  const referencePath = process.env.S3_REFERENCE_VECTORS;
  const modelDirectory = process.env.S3_MODEL_DIR;
  if (referencePath === undefined || !existsSync(referencePath)) {
    return {
      skip: "set S3_REFERENCE_VECTORS to the output of scripts/write_python_vectors.py",
    };
  }
  if (modelDirectory === undefined || !existsSync(join(modelDirectory, MODEL_ARTIFACT))) {
    return {
      skip: `set S3_MODEL_DIR to a snapshot of ${MODEL_ID} containing ${MODEL_ARTIFACT}`,
    };
  }

  const reference = JSON.parse(readFileSync(referencePath, "utf8")) as ReferenceVectors;
  if (reference.modelId !== MODEL_ID) {
    throw new Error(`reference vectors are from ${reference.modelId}, not ${MODEL_ID}`);
  }
  if (reference.dimension !== DIMENSION) {
    throw new Error(`reference vectors are ${reference.dimension}-wide, expected ${DIMENSION}`);
  }

  const { Tokenizer } = await import("@huggingface/tokenizers");
  const ort = await import("onnxruntime-node");

  const tokenizer = new Tokenizer(
    JSON.parse(readFileSync(join(modelDirectory, "tokenizer.json"), "utf8")) as object,
    JSON.parse(readFileSync(join(modelDirectory, "tokenizer_config.json"), "utf8")) as object,
  );

  const session = await ort.InferenceSession.create(join(modelDirectory, MODEL_ARTIFACT));

  const embed = async (texts: readonly string[]): Promise<number[][]> => {
    const encodings = texts.map((text) => tokenizer.encode(text));
    const width = Math.max(...encodings.map((encoding) => encoding.ids.length));

    const ids: bigint[] = [];
    const mask: bigint[] = [];
    const types: bigint[] = [];
    const maskRows: number[][] = [];
    for (const encoding of encodings) {
      const row: number[] = [];
      for (let index = 0; index < width; index += 1) {
        const present = index < encoding.ids.length;
        ids.push(BigInt(present ? (encoding.ids[index] ?? 0) : 0));
        mask.push(present ? BigInt(encoding.attention_mask[index] ?? 1) : 0n);
        types.push(0n);
        row.push(present ? 1 : 0);
      }
      maskRows.push(row);
    }

    const dims = [texts.length, width];
    const feeds: Record<string, unknown> = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(ids), dims),
      attention_mask: new ort.Tensor("int64", BigInt64Array.from(mask), dims),
    };
    // JinaBERT declares token_type_ids; feeding an input the graph does not
    // declare is an error, so only supply what it asks for.
    if (session.inputNames.includes("token_type_ids")) {
      feeds.token_type_ids = new ort.Tensor("int64", BigInt64Array.from(types), dims);
    }

    const outputs = await session.run(feeds as never);
    const firstName = session.outputNames[0];
    if (firstName === undefined) throw new Error("the model declares no outputs");
    const hidden = outputs[firstName];
    if (hidden === undefined) throw new Error(`missing output ${firstName}`);
    return meanPoolAndNormalize(hidden.data as Float32Array, maskRows, [
      texts.length,
      width,
      DIMENSION,
    ]);
  };

  const documents = await embed(reference.documents);
  const queries = await embed(reference.queries);

  const cosine = cosineRows(reference.documentVectors, documents);
  const minimum = Math.min(...cosine);
  const overlap = topKOverlap(queries, reference.documentVectors, documents, TOP_K);

  const verdict =
    `min cosine ${minimum.toPrecision(9)} (gate >= ${MIN_COSINE}), ` +
    `top-${TOP_K} overlap ${overlap.toFixed(4)} (gate >= ${MIN_TOP_K_OVERLAP}) ` +
    `over ${reference.documents.length} documents`;

  if (minimum < MIN_COSINE || overlap < MIN_TOP_K_OVERLAP) {
    // A failure here is not a blocker -- it resolves D1 the other way, and the
    // rebuild path already exists and is exercised. Say so in the record.
    throw new Error(`${verdict} -- parity FAILS, D1 resolves to "rebuild indexes on migrate"`);
  }
  return `${verdict} -- parity holds, D1 resolves to "keep indexes"`;
});

spike.finish();
