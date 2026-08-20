/** MLX model configuration guards that run without an Apple-Silicon addon. */

import { expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { readMlxModelConfig } from "../src/mlx-backend.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

function writeConfig(directory: string, overrides: Record<string, unknown> = {}): void {
  fs.writeFileSync(
    path.join(directory, "config.json"),
    JSON.stringify({
      model_type: "bert",
      position_embedding_type: "alibi",
      feed_forward_type: "geglu",
      hidden_act: "gelu",
      emb_pooler: "mean",
      hidden_size: 8,
      num_hidden_layers: 2,
      num_attention_heads: 2,
      intermediate_size: 16,
      layer_norm_eps: 1e-12,
      ...overrides,
    }),
  );
}

test("MLX accepts the pinned JinaBERT configuration", () => {
  const directory = temporaryDirectory();
  try {
    writeConfig(directory);
    expect(readMlxModelConfig(directory)).toEqual({
      hiddenSize: 8,
      numHiddenLayers: 2,
      numAttentionHeads: 2,
      intermediateSize: 16,
      layerNormEps: 1e-12,
    });
  } finally {
    removeDirectory(directory);
  }
});

test("MLX refuses a model with a different architecture", () => {
  const directory = temporaryDirectory();
  try {
    writeConfig(directory, { feed_forward_type: "gelu" });
    expect(() => readMlxModelConfig(directory)).toThrow('feed_forward_type="geglu"');
  } finally {
    removeDirectory(directory);
  }
});

test("MLX refuses a hidden size that cannot split into attention heads", () => {
  const directory = temporaryDirectory();
  try {
    writeConfig(directory, { hidden_size: 7 });
    expect(() => readMlxModelConfig(directory)).toThrow("does not divide evenly");
  } finally {
    removeDirectory(directory);
  }
});
