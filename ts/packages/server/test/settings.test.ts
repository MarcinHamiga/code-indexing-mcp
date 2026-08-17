/** Environment parsing for the memory-safe indexing settings. */

import { describe, expect, test } from "bun:test";
import { type CodeIndexingError, isCodeIndexingError } from "../src/errors.ts";
import { indexSettingsFromEnvironment } from "../src/settings.ts";

/** Assert the call is refused, and hand back the error for further checks. */
function rejects(environment: Record<string, string>): CodeIndexingError {
  let caught: unknown;
  try {
    indexSettingsFromEnvironment(environment);
  } catch (error) {
    caught = error;
  }
  if (!isCodeIndexingError(caught)) {
    throw new Error(`expected a CodeIndexingError, got ${String(caught)}`);
  }
  expect(caught.code).toBe("INVALID_CONFIGURATION");
  return caught;
}

test("indexing defaults to lazy", () => {
  const settings = indexSettingsFromEnvironment({});

  expect(settings.mode).toBe("lazy");
  expect(settings.embeddingBatchSize).toBe(1);
  expect(settings.embeddingThreads).toBeGreaterThanOrEqual(1);
  expect(settings.embeddingCpuArena).toBe(false);
  expect(settings.vectorIndex).toBe("exact");
  expect(settings.vectorStorage).toBe("float16");
  expect(settings.indexWaitSeconds).toBe(300);
  expect(settings.indexExecution).toBe("worker");
  expect(settings.brokerMode).toBe("auto");
});

test("vector storage is parsed and validated", () => {
  expect(
    indexSettingsFromEnvironment({ CODE_INDEXING_VECTOR_STORAGE: "FLOAT32" }).vectorStorage,
  ).toBe("float32");

  rejects({ CODE_INDEXING_VECTOR_STORAGE: "int8" });
});

test("index wait seconds is validated", () => {
  expect(
    indexSettingsFromEnvironment({ CODE_INDEXING_INDEX_WAIT_SECONDS: "0" }).indexWaitSeconds,
  ).toBe(0);

  rejects({ CODE_INDEXING_INDEX_WAIT_SECONDS: "-1" });
});

describe("the legacy auto-index flag", () => {
  test.each([
    ["1", "eager"],
    ["true", "eager"],
    ["0", "manual"],
  ] as const)("%s maps to %s", (legacy, expected) => {
    expect(indexSettingsFromEnvironment({ CODE_INDEXING_AUTO_INDEX: legacy }).mode).toBe(expected);
  });

  test("the explicit mode takes precedence", () => {
    expect(
      indexSettingsFromEnvironment({
        CODE_INDEXING_INDEX_MODE: "manual",
        CODE_INDEXING_AUTO_INDEX: "1",
      }).mode,
    ).toBe("manual");
  });

  test("an unknown mode is a configuration error", () => {
    rejects({ CODE_INDEXING_INDEX_MODE: "whenever" });
  });
});

test("invalid index settings raise a stable error carrying the setting name", () => {
  const error = rejects({ CODE_INDEXING_EMBED_BATCH_SIZE: "0" });

  expect(error.details.setting).toBe("CODE_INDEXING_EMBED_BATCH_SIZE");
  expect(error.details.value).toBe("0");
});

test("an integer setting accepts exactly what Python's int() accepts", () => {
  // Underscore grouping is valid; a trailing suffix is a typo, not a value.
  expect(
    indexSettingsFromEnvironment({ CODE_INDEXING_INDEX_WAIT_SECONDS: "1_200" }).indexWaitSeconds,
  ).toBe(1_200);
  expect(
    indexSettingsFromEnvironment({ CODE_INDEXING_INDEX_WAIT_SECONDS: " 60 " }).indexWaitSeconds,
  ).toBe(60);
  rejects({ CODE_INDEXING_INDEX_WAIT_SECONDS: "60s" });
  rejects({ CODE_INDEXING_INDEX_WAIT_SECONDS: "6e2" });
  rejects({ CODE_INDEXING_INDEX_WAIT_SECONDS: "60.0" });
  rejects({ CODE_INDEXING_INDEX_WAIT_SECONDS: "" });
});

test("the memory budget override keeps the worker default", () => {
  const settings = indexSettingsFromEnvironment({ CODE_INDEXING_INDEX_MEMORY_MB: "1536" });

  expect(settings.indexMemoryBytes).toBe(1536 * 1024 * 1024);
  expect(settings.indexExecution).toBe("worker");
});

test("token window settings default to the measured budget", () => {
  const settings = indexSettingsFromEnvironment({});

  expect(settings.embeddingMaxTokens).toBe(1024);
  expect(settings.embeddingOverlapTokens).toBe(64);
});

test("token window settings are configurable", () => {
  const settings = indexSettingsFromEnvironment({
    CODE_INDEXING_EMBED_MAX_TOKENS: "512",
    CODE_INDEXING_EMBED_OVERLAP_TOKENS: "32",
  });

  expect(settings.embeddingMaxTokens).toBe(512);
  expect(settings.embeddingOverlapTokens).toBe(32);
});

test("a token budget above the model limit is rejected", () => {
  rejects({ CODE_INDEXING_EMBED_MAX_TOKENS: "16384" });
});

test("the accelerator defaults to automatic selection", () => {
  const settings = indexSettingsFromEnvironment({});

  expect(settings.embeddingAccelerator).toBe("auto");
  expect(settings.embeddingStrict).toBe(false);
  expect(settings.embeddingBatchAuto).toBe(true);
});

test.each([
  ["cpu", "cpu"],
  ["CUDA", "cuda"],
  ["webgpu", "webgpu"],
  ["migraphx", "migraphx"],
  ["coreml", "coreml"],
  ["mlx", "mlx"],
] as const)("the accelerator is configurable: %s", (value, expected) => {
  expect(
    indexSettingsFromEnvironment({ CODE_INDEXING_EMBED_ACCELERATOR: value }).embeddingAccelerator,
  ).toBe(expected);
});

test("an unknown accelerator is a configuration error", () => {
  const error = rejects({ CODE_INDEXING_EMBED_ACCELERATOR: "tpu" });

  expect(error.message).toContain("auto, cpu, cuda, mlx, webgpu, migraphx, coreml");
});

test("strict mode is configurable", () => {
  expect(indexSettingsFromEnvironment({ CODE_INDEXING_EMBED_STRICT: "1" }).embeddingStrict).toBe(
    true,
  );
  expect(indexSettingsFromEnvironment({ CODE_INDEXING_EMBED_STRICT: "off" }).embeddingStrict).toBe(
    false,
  );
  rejects({ CODE_INDEXING_EMBED_STRICT: "maybe" });
});

test("an automatic batch size keeps the CPU default", () => {
  const settings = indexSettingsFromEnvironment({ CODE_INDEXING_EMBED_BATCH_SIZE: "auto" });

  expect(settings.embeddingBatchSize).toBe(1);
  expect(settings.embeddingBatchAuto).toBe(true);
});

test("an explicit batch size is marked as not calibratable", () => {
  const settings = indexSettingsFromEnvironment({ CODE_INDEXING_EMBED_BATCH_SIZE: "64" });

  expect(settings.embeddingBatchSize).toBe(64);
  expect(settings.embeddingBatchAuto).toBe(false);
});

test("the batch size range reaches the documented maximum", () => {
  expect(
    indexSettingsFromEnvironment({ CODE_INDEXING_EMBED_BATCH_SIZE: "256" }).embeddingBatchSize,
  ).toBe(256);

  rejects({ CODE_INDEXING_EMBED_BATCH_SIZE: "257" });
});

describe("the accelerator crossover", () => {
  test("is measured by default", () => {
    const settings = indexSettingsFromEnvironment({});

    expect(settings.embeddingCrossoverAuto).toBe(true);
    expect(settings.embeddingCrossoverCharacters).toBe(0);
    expect(settings.embeddingCalibrate).toBe(true);
  });

  test("can be turned off entirely", () => {
    // "off" means the accelerator starts on the first chunk, which is what every
    // run did before anything measured whether that paid.
    const settings = indexSettingsFromEnvironment({ CODE_INDEXING_EMBED_CROSSOVER: "off" });

    expect(settings.embeddingCrossoverAuto).toBe(false);
    expect(settings.embeddingCrossoverCharacters).toBe(0);
  });

  test("an explicit size overrides the measured one", () => {
    const settings = indexSettingsFromEnvironment({ CODE_INDEXING_EMBED_CROSSOVER: "250000" });

    expect(settings.embeddingCrossoverAuto).toBe(false);
    expect(settings.embeddingCrossoverCharacters).toBe(250_000);
  });

  test("a value that is neither a mode nor a size is rejected", () => {
    rejects({ CODE_INDEXING_EMBED_CROSSOVER: "sometimes" });
    rejects({ CODE_INDEXING_EMBED_CROSSOVER: String(1024 ** 3 + 1) });
  });

  test("calibration can be declined", () => {
    expect(
      indexSettingsFromEnvironment({ CODE_INDEXING_EMBED_CALIBRATE: "0" }).embeddingCalibrate,
    ).toBe(false);
  });
});

describe("the memory ceiling variables", () => {
  test("the documented name is accepted", () => {
    expect(
      indexSettingsFromEnvironment({ CODE_INDEXING_EMBED_MEMORY_MB: "2048" }).indexMemoryBytes,
    ).toBe(2048 * 1024 * 1024);
  });

  test("the newer name wins over the legacy one", () => {
    expect(
      indexSettingsFromEnvironment({
        CODE_INDEXING_EMBED_MEMORY_MB: "2048",
        CODE_INDEXING_INDEX_MEMORY_MB: "1024",
      }).indexMemoryBytes,
    ).toBe(2048 * 1024 * 1024);
  });

  test("an exported but empty variable does not shadow the legacy name", () => {
    // An empty export is a shell saying "unset", not a value of zero length.
    expect(
      indexSettingsFromEnvironment({
        CODE_INDEXING_EMBED_MEMORY_MB: "",
        CODE_INDEXING_INDEX_MEMORY_MB: "1536",
      }).indexMemoryBytes,
    ).toBe(1536 * 1024 * 1024);
  });

  test("the default sits between one and two gigabytes", () => {
    const settings = indexSettingsFromEnvironment({});

    expect(settings.indexMemoryBytes).toBeGreaterThanOrEqual(1024 * 1024 * 1024);
    expect(settings.indexMemoryBytes).toBeLessThanOrEqual(2048 * 1024 * 1024);
  });
});

describe("automatic maintenance", () => {
  test("defaults to enabled with 24h retention", () => {
    const settings = indexSettingsFromEnvironment({});

    expect(settings.autoMaintenance).toBe(true);
    expect(settings.versionRetentionHours).toBe(24);
  });

  test("is configurable", () => {
    const settings = indexSettingsFromEnvironment({
      CODE_INDEXING_AUTO_MAINTENANCE: "off",
      CODE_INDEXING_VERSION_RETENTION_HOURS: "48",
    });

    expect(settings.autoMaintenance).toBe(false);
    expect(settings.versionRetentionHours).toBe(48);
  });

  test("retention never reaches zero hours", () => {
    // Zero-hour automatic retention would reap versions concurrent readers use.
    rejects({ CODE_INDEXING_VERSION_RETENTION_HOURS: "0" });
    rejects({ CODE_INDEXING_VERSION_RETENTION_HOURS: "-1" });
  });

  test("retention has a bounded maximum", () => {
    rejects({ CODE_INDEXING_VERSION_RETENTION_HOURS: "100000" });
  });
});
