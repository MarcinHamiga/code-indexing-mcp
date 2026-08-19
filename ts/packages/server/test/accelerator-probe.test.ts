/** Real-inference accelerator probe behavior with a deterministic model seam. */

import { afterEach, expect, test } from "bun:test";
import { probeAccelerator } from "../src/accelerator-probe.ts";
import { CPU_PROVIDER } from "../src/backends.ts";
import { DEFAULT_MODEL, PROBE_TEXTS } from "../src/embedding.ts";
import {
  loadPassageModel,
  setLoadModel,
  type PassageWorkerModel,
} from "../src/embedding-worker.ts";

function model(providers: readonly string[]): PassageWorkerModel {
  return {
    resolvedProviders: providers,
    passageEmbed: (texts) => {
      expect(texts).toEqual([...PROBE_TEXTS]);
      return texts.map(() => [1, 0, 0, 0]);
    },
  };
}

afterEach(() => {
  setLoadModel(loadPassageModel);
});

test("a plugin provider is verified by the session it created", async () => {
  setLoadModel(() => model(["WebGpuExecutionProvider", CPU_PROVIDER]));

  const report = await probeAccelerator("webgpu", {
    cacheDirectory: "/tmp/models",
    offline: true,
    modelId: DEFAULT_MODEL,
    dimension: 4,
  });

  expect(report.ok).toBe(true);
  expect(report.resolvedProviders).toEqual(["WebGpuExecutionProvider", CPU_PROVIDER]);
  expect(report.providers).toContain("WebGpuExecutionProvider");
  expect(report.dimension).toBe(4);
});

test("a session that silently falls back to CPU cannot be promoted", async () => {
  setLoadModel(() => model([CPU_PROVIDER]));

  await expect(
    probeAccelerator("webgpu", {
      cacheDirectory: "/tmp/models",
      offline: true,
      dimension: 4,
    }),
  ).rejects.toThrow(
    "WebGpuExecutionProvider was requested but the session runs on CPUExecutionProvider",
  );
});

test("a direct session that cannot report its provider cannot be promoted", async () => {
  setLoadModel(() => model([]));

  await expect(
    probeAccelerator("webgpu", {
      cacheDirectory: "/tmp/models",
      offline: true,
      dimension: 4,
    }),
  ).rejects.toThrow("WebGpuExecutionProvider was requested but the session runs on no providers");
});

test("a non-ONNX backend is probed without an ONNX provider list", async () => {
  setLoadModel(() => model(["MlxMetalBackend"]));

  const report = await probeAccelerator("mlx", {
    cacheDirectory: "/tmp/models",
    offline: true,
    dimension: 4,
  });

  expect(report.ok).toBe(true);
  expect(report.resolvedProviders).toEqual(["MlxMetalBackend"]);
  expect(report.device).toBe("metal");
  // An MLX environment has no ONNX Runtime, so it publishes no ONNX providers
  // -- a CPU execution provider nothing there could run must not be recorded.
  expect(report.providers).toEqual(["MlxMetalBackend"]);
});

test("an mlx session reporting no backend fails the probe", async () => {
  setLoadModel(() => model([]));

  await expect(
    probeAccelerator("mlx", {
      cacheDirectory: "/tmp/models",
      offline: true,
      dimension: 4,
    }),
  ).rejects.toThrow("MlxMetalBackend was requested but the session runs on no providers");
});
