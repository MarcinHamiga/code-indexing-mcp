/** Direct ONNX pooling, snapshot gating, and injected session wiring. */

import { expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import {
  configureDirectOnnx,
  createSession,
  createWebgpuSession,
  DEFAULT_MODEL_ARTIFACT,
  DirectOnnxEmbedding,
  loadTokenizer,
  meanPoolAndNormalize,
  resolveModelSnapshot,
} from "../src/direct-onnx.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

test("only the index compatible model is accepted", async () => {
  const directory = temporaryDirectory();
  try {
    let caught: unknown;
    try {
      await resolveModelSnapshot(directory, { modelId: "another/model", offline: true });
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(Error);
    expect(String(caught)).toContain("only supports");
  } finally {
    removeDirectory(directory);
  }
});

test("mean pooling ignores padding and normalizes float32 rows", () => {
  const output = [
    [
      [1.0, 0.0],
      [3.0, 0.0],
      [100.0, 100.0],
    ],
    [
      [0.0, 0.0],
      [0.0, 0.0],
      [0.0, 0.0],
    ],
  ];
  const attention = [
    [1, 1, 0],
    [0, 0, 0],
  ];

  const pooled = meanPoolAndNormalize(output, attention);

  expect(pooled[0]?.[0]).toBeCloseTo(1.0);
  expect(pooled[0]?.[1]).toBeCloseTo(0.0);
  expect(pooled[1]?.[0]).toBeCloseTo(0.0);
  expect(pooled[1]?.[1]).toBeCloseTo(0.0);
});

test("model snapshot uses the shared cache and offline flag", async () => {
  const directory = temporaryDirectory();
  try {
    const snapshot = path.join(directory, "snapshot");
    const calls: Record<string, unknown>[] = [];
    configureDirectOnnx({
      snapshotDownload: (options) => {
        calls.push(options);
        return snapshot;
      },
    });
    try {
      const resolved = await resolveModelSnapshot(path.join(directory, "models"), {
        modelId: "jinaai/jina-embeddings-v2-base-code",
        offline: true,
      });
      expect(resolved).toBe(snapshot);
      expect(calls).toEqual([
        {
          repoId: "jinaai/jina-embeddings-v2-base-code",
          allowPatterns: [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            DEFAULT_MODEL_ARTIFACT,
          ],
          cacheDir: path.join(directory, "models"),
          localFilesOnly: true,
        },
      ]);
    } finally {
      configureDirectOnnx({ snapshotDownload: undefined });
    }
  } finally {
    removeDirectory(directory);
  }
});

test("direct model builds exact onnx inputs and reports resolved providers", async () => {
  const directory = temporaryDirectory();
  try {
    const modelDirectory = path.join(directory, "snapshot");
    fs.mkdirSync(path.join(modelDirectory, "onnx"), { recursive: true });
    fs.writeFileSync(path.join(modelDirectory, DEFAULT_MODEL_ARTIFACT), "onnx");

    const documents: string[] = [];
    const lastInputs: Record<string, BigInt64Array> = {};
    const originalResolve = resolveModelSnapshot;
    const originalLoad = loadTokenizer;
    const originalCreate = createSession;
    configureDirectOnnx({
      resolveModelSnapshot: () => modelDirectory,
      loadTokenizer: () => ({
        encodeBatch: (texts: string[]) => {
          documents.push(...texts);
          return [
            { ids: [11, 12, 0], attentionMask: [1, 1, 0] },
            { ids: [21, 22, 23], attentionMask: [1, 1, 1] },
          ];
        },
        encode: () => ({ ids: [1], attentionMask: [1] }),
      }),
      createSession: (modelPath, options) => {
        expect(modelPath).toBe(path.join(modelDirectory, DEFAULT_MODEL_ARTIFACT));
        expect(options.providers).toEqual(["MIGraphXExecutionProvider", "CPUExecutionProvider"]);
        expect(options.threads).toBe(3);
        expect(options.enableCpuMemArena).toBe(false);
        return {
          getInputs: () => [
            { name: "input_ids" },
            { name: "attention_mask" },
            { name: "token_type_ids" },
          ],
          getProviders: () => ["MIGraphXExecutionProvider", "CPUExecutionProvider"],
          run: (_names, inputs) => {
            Object.assign(lastInputs, inputs);
            return [
              [
                [
                  [1.0, 0.0],
                  [3.0, 0.0],
                  [100.0, 100.0],
                ],
                [
                  [0.0, 2.0],
                  [0.0, 4.0],
                  [0.0, 6.0],
                ],
              ],
            ];
          },
        };
      },
    });
    try {
      const model = await DirectOnnxEmbedding.create(path.join(directory, "models"), {
        offline: true,
        threads: 3,
        enableCpuMemArena: false,
        providers: ["MIGraphXExecutionProvider", "CPUExecutionProvider"],
      });
      const vectors = await model.passageEmbed(["short", "longer"]);
      expect(documents).toEqual(["short", "longer"]);
      expect(vectors[0]?.[0]).toBeCloseTo(1.0);
      expect(vectors[1]?.[1]).toBeCloseTo(1.0);
      expect(model.resolvedProviders).toEqual([
        "MIGraphXExecutionProvider",
        "CPUExecutionProvider",
      ]);
      expect(lastInputs.token_type_ids?.every((value) => value === 0n)).toBe(true);
    } finally {
      configureDirectOnnx({
        resolveModelSnapshot: originalResolve,
        loadTokenizer: originalLoad,
        createSession: originalCreate,
      });
    }
  } finally {
    removeDirectory(directory);
  }
});

test("webgpu session refuses a registered plugin with no device", async () => {
  configureDirectOnnx({
    onnxRuntimeBindings: {
      GraphOptimizationLevel: { ORT_ENABLE_ALL: "all" },
      SessionOptions: class {
        graphOptimizationLevel: unknown;
        enableCpuMemArena = true;
        intraOpNumThreads = 0;
        interOpNumThreads = 0;
        addProviderForDevices(): void {}
      },
      registerExecutionProviderLibrary: () => undefined,
      getEpDevices: () => [{ ep_name: "CPUExecutionProvider" }],
      InferenceSession: class {
        getInputs(): [] {
          return [];
        }
        getProviders(): [] {
          return [];
        }
        run(): [] {
          return [];
        }
      },
    },
    webgpuPluginBindings: {
      getLibraryPath: () => "/runtime/webgpu.dylib",
      getEpName: () => "WebGpuExecutionProvider",
    },
  });
  try {
    await expect(
      createWebgpuSession("/tmp/model.onnx", { threads: 2, enableCpuMemArena: false }),
    ).rejects.toThrow("no WebGPU device");
  } finally {
    configureDirectOnnx({ onnxRuntimeBindings: undefined, webgpuPluginBindings: undefined });
  }
});
