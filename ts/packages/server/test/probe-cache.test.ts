/** Probe cache fingerprints, schema versioning, and bounded JSON records. */

import { expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import {
  CACHE_SCHEMA_VERSION,
  MAX_RECORDS,
  modelArtifactFingerprint,
  ProbeCache,
  type ProbeKey,
  probeFingerprint,
  probeKey,
} from "../src/probe-cache.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

function key(overrides: Partial<ProbeKey> = {}): ProbeKey {
  return probeKey({
    modelId: "jinaai/jina-embeddings-v2-base-code",
    modelArtifact: "artifact-a",
    accelerator: "cuda",
    provider: "CUDAExecutionProvider",
    runtimeVersion: "1.20.0",
    platform: "darwin-arm64-25.5.0",
    device: "cuda:0",
    driverVersion: "550.54",
    ...overrides,
  });
}

test("a stored probe is found again", () => {
  const directory = temporaryDirectory();
  try {
    const cache = new ProbeCache(path.join(directory, "probes.json"));
    const stored = key();

    expect(cache.state(stored)).toBe("miss");
    cache.store(stored, { batchSize: 16, dimension: 768, detail: "CUDAExecutionProvider" });

    const record = cache.load(stored);
    expect(record).toBeDefined();
    expect(record?.batchSize).toBe(16);
    expect(record?.dimension).toBe(768);
    expect(cache.state(stored)).toBe("hit");
  } finally {
    removeDirectory(directory);
  }
});

test("a stored calibration survives the round trip", () => {
  const directory = temporaryDirectory();
  try {
    const cache = new ProbeCache(path.join(directory, "probes.json"));
    cache.store(key(), {
      batchSize: 8,
      dimension: 768,
      charactersPerSecond: 12_345.5,
      loadNs: 2_500_000_000,
      limitedBy: "memory",
    });

    const record = cache.load(key());
    expect(record?.charactersPerSecond).toBe(12_345.5);
    expect(record?.loadNs).toBe(2_500_000_000);
    expect(record?.limitedBy).toBe("memory");
  } finally {
    removeDirectory(directory);
  }
});

test("a record written before calibration is not read as uncalibrated", () => {
  const directory = temporaryDirectory();
  try {
    const filePath = path.join(directory, "probes.json");
    fs.writeFileSync(
      filePath,
      JSON.stringify({
        schema_version: 1,
        records: [
          {
            fingerprint: probeFingerprint(key()),
            batch_size: 8,
            dimension: 768,
            recorded_at_ns: 1,
          },
        ],
      }),
    );

    expect(new ProbeCache(filePath).load(key())).toBeUndefined();
  } finally {
    removeDirectory(directory);
  }
});

test.each([
  "modelId",
  "modelArtifact",
  "accelerator",
  "provider",
  "runtimeVersion",
  "platform",
  "device",
  "driverVersion",
] as const)("every key component invalidates the record (%s)", (field) => {
  const directory = temporaryDirectory();
  try {
    const cache = new ProbeCache(path.join(directory, "probes.json"));
    cache.store(key(), { batchSize: 16, dimension: 768 });

    expect(cache.load(key({ [field]: "changed" }))).toBeUndefined();
  } finally {
    removeDirectory(directory);
  }
});

test("restoring the original configuration finds the record again", () => {
  const directory = temporaryDirectory();
  try {
    const cache = new ProbeCache(path.join(directory, "probes.json"));
    cache.store(key(), { batchSize: 8, dimension: 768 });
    cache.store(key({ runtimeVersion: "1.21.0" }), { batchSize: 4, dimension: 768 });

    const original = cache.load(key());
    expect(original?.batchSize).toBe(8);
  } finally {
    removeDirectory(directory);
  }
});

test("storing the same key twice replaces rather than duplicates", () => {
  const directory = temporaryDirectory();
  try {
    const filePath = path.join(directory, "probes.json");
    const cache = new ProbeCache(filePath);
    cache.store(key(), { batchSize: 8, dimension: 768 });
    cache.store(key(), { batchSize: 32, dimension: 768 });

    expect(cache.load(key())?.batchSize).toBe(32);
    expect(
      (JSON.parse(fs.readFileSync(filePath, "utf8")) as { records: unknown[] }).records.length,
    ).toBe(1);
  } finally {
    removeDirectory(directory);
  }
});

test("a corrupt cache reads as empty rather than raising", () => {
  const directory = temporaryDirectory();
  try {
    const filePath = path.join(directory, "probes.json");
    fs.writeFileSync(filePath, "{ not json");
    expect(new ProbeCache(filePath).load(key())).toBeUndefined();
  } finally {
    removeDirectory(directory);
  }
});

test("a cache from another schema version is ignored", () => {
  const directory = temporaryDirectory();
  try {
    const filePath = path.join(directory, "probes.json");
    const cache = new ProbeCache(filePath);
    cache.store(key(), { batchSize: 8, dimension: 768 });
    const payload = JSON.parse(fs.readFileSync(filePath, "utf8")) as { schema_version: number };
    payload.schema_version = CACHE_SCHEMA_VERSION + 1;
    fs.writeFileSync(filePath, JSON.stringify(payload));

    expect(cache.load(key())).toBeUndefined();
  } finally {
    removeDirectory(directory);
  }
});

test("a partial record is dropped without taking the others with it", () => {
  const directory = temporaryDirectory();
  try {
    const filePath = path.join(directory, "probes.json");
    const cache = new ProbeCache(filePath);
    cache.store(key(), { batchSize: 8, dimension: 768 });
    const payload = JSON.parse(fs.readFileSync(filePath, "utf8")) as { records: unknown[] };
    payload.records.unshift({ fingerprint: "orphan" });
    fs.writeFileSync(filePath, JSON.stringify(payload));

    expect(cache.load(key())?.batchSize).toBe(8);
  } finally {
    removeDirectory(directory);
  }
});

test("the cache is trimmed to its bound", () => {
  const directory = temporaryDirectory();
  try {
    const filePath = path.join(directory, "probes.json");
    const cache = new ProbeCache(filePath);
    for (let index = 0; index < MAX_RECORDS + 5; index++) {
      cache.store(key({ device: `cuda:${index}` }), { batchSize: index + 1, dimension: 768 });
    }

    const stored = (JSON.parse(fs.readFileSync(filePath, "utf8")) as { records: unknown[] })
      .records;
    expect(stored.length).toBe(MAX_RECORDS);
    expect(cache.load(key({ device: "cuda:0" }))).toBeUndefined();
    expect(cache.load(key({ device: `cuda:${MAX_RECORDS + 4}` }))).toBeDefined();
  } finally {
    removeDirectory(directory);
  }
});

test("a missing cache file is simply a miss", () => {
  const directory = temporaryDirectory();
  try {
    expect(new ProbeCache(path.join(directory, "absent", "probes.json")).state(key())).toBe("miss");
  } finally {
    removeDirectory(directory);
  }
});

test("an unwritable cache directory does not fail the run", () => {
  const directory = temporaryDirectory();
  try {
    const blocked = path.join(directory, "file");
    fs.writeFileSync(blocked, "not a directory");
    new ProbeCache(path.join(blocked, "probes.json")).store(key(), {
      batchSize: 8,
      dimension: 768,
    });
  } finally {
    removeDirectory(directory);
  }
});

test("the artifact fingerprint notices a changed model file", () => {
  const directory = temporaryDirectory();
  try {
    const models = path.join(directory, "models");
    fs.mkdirSync(path.join(models, "jina"), { recursive: true });
    const artifact = path.join(models, "jina", "model.onnx");
    fs.writeFileSync(artifact, Buffer.alloc(128, 120));
    const before = modelArtifactFingerprint(models, "jina");
    fs.writeFileSync(artifact, Buffer.alloc(256, 120));
    expect(modelArtifactFingerprint(models, "jina")).not.toBe(before);
  } finally {
    removeDirectory(directory);
  }
});

test("the artifact fingerprint is stable across calls", () => {
  const directory = temporaryDirectory();
  try {
    const models = path.join(directory, "models");
    fs.mkdirSync(models);
    fs.writeFileSync(path.join(models, "model.onnx"), Buffer.alloc(64, 120));
    expect(modelArtifactFingerprint(models, "jina")).toBe(modelArtifactFingerprint(models, "jina"));
  } finally {
    removeDirectory(directory);
  }
});

test("the artifact fingerprint survives a missing cache directory", () => {
  const directory = temporaryDirectory();
  try {
    expect(modelArtifactFingerprint(path.join(directory, "absent"), "jina")).toBeTruthy();
  } finally {
    removeDirectory(directory);
  }
});

function fastembedLayout(cache: string, modelId: string): string {
  const artifact = path.join(
    cache,
    `models--${modelId.replaceAll("/", "--")}`,
    "blobs",
    "model.onnx",
  );
  fs.mkdirSync(path.dirname(artifact), { recursive: true });
  fs.writeFileSync(artifact, Buffer.alloc(128, 120));
  return artifact;
}

test("the fingerprint ignores the rest of a shared model cache", () => {
  const directory = temporaryDirectory();
  try {
    const cache = path.join(directory, "models");
    fastembedLayout(cache, "jinaai/jina-embeddings-v2-base-code");
    const before = modelArtifactFingerprint(cache, "jinaai/jina-embeddings-v2-base-code");

    fs.mkdirSync(path.join(cache, ".locks", "models--jinaai--jina-embeddings-v2-base-code"), {
      recursive: true,
    });
    fs.writeFileSync(
      path.join(cache, ".locks", "models--jinaai--jina-embeddings-v2-base-code", "a.lock"),
      "1",
    );
    fastembedLayout(cache, "someone/another-model");
    fs.writeFileSync(path.join(cache, "CACHEDIR.TAG"), "Signature: 8a477f597d28d172");

    expect(modelArtifactFingerprint(cache, "jinaai/jina-embeddings-v2-base-code")).toBe(before);
  } finally {
    removeDirectory(directory);
  }
});

test("the fingerprint still notices the models own artifact changing", () => {
  const directory = temporaryDirectory();
  try {
    const cache = path.join(directory, "models");
    const artifact = fastembedLayout(cache, "jinaai/jina-embeddings-v2-base-code");
    const before = modelArtifactFingerprint(cache, "jinaai/jina-embeddings-v2-base-code");
    fs.writeFileSync(artifact, Buffer.alloc(256, 120));
    expect(modelArtifactFingerprint(cache, "jinaai/jina-embeddings-v2-base-code")).not.toBe(before);
  } finally {
    removeDirectory(directory);
  }
});

test("two models in one cache do not share a fingerprint", () => {
  const directory = temporaryDirectory();
  try {
    const cache = path.join(directory, "models");
    fastembedLayout(cache, "jinaai/jina-embeddings-v2-base-code");
    fastembedLayout(cache, "someone/another-model");
    expect(modelArtifactFingerprint(cache, "jinaai/jina-embeddings-v2-base-code")).not.toBe(
      modelArtifactFingerprint(cache, "someone/another-model"),
    );
  } finally {
    removeDirectory(directory);
  }
});

test("an unrecognised layout falls back to the whole cache", () => {
  const directory = temporaryDirectory();
  try {
    const cache = path.join(directory, "models");
    fs.mkdirSync(cache);
    fs.writeFileSync(path.join(cache, "model.onnx"), Buffer.alloc(64, 120));
    const before = modelArtifactFingerprint(cache, "jina");
    fs.writeFileSync(path.join(cache, "model.onnx"), Buffer.alloc(65, 120));
    expect(modelArtifactFingerprint(cache, "jina")).not.toBe(before);
  } finally {
    removeDirectory(directory);
  }
});
