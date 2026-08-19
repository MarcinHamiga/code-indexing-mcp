import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { BackendDescriptor, BackendSelection, CPU_BACKEND, CPU_PROVIDER } from "../src/backends.ts";
import { PROBE_TEXTS, segmentPlan } from "../src/embedding.ts";
import { EmbeddingWorkerSession, workerConfig } from "../src/embedding-worker.ts";
import { isCodeIndexingError } from "../src/errors.ts";
import { TreeSitterExtractor } from "../src/extractor.ts";
import { Indexer } from "../src/indexing.ts";
import { PassageBackendSession } from "../src/passage-backend.ts";
import { initializeProject } from "../src/projects.ts";
import { SourceScanner } from "../src/scanner.ts";
import { LanceStore } from "../src/storage.ts";
import type { WorkerConnection } from "../src/worker-channel.ts";
import { FunctionLauncher, type WorkerTarget } from "../src/worker-launcher.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

const DIMENSION = 4;

class RecordingEmbedder {
  readonly modelId = "test/code";
  readonly dimension = DIMENSION;

  embedPassages(texts: string[]): number[][] {
    return texts.map((text) => [text.length % 7, 1, 2, 3]);
  }

  embedQuery(text: string): number[] {
    return [text.length % 7, 1, 2, 3];
  }
}

function unitVector(dimension = DIMENSION): Uint8Array {
  const out = Buffer.allocUnsafe(dimension * 4);
  out.writeFloatLE(1, 0);
  return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
}

async function healthyWorker(
  connection: WorkerConnection,
  config: { dimension: number; accelerator: string },
): Promise<void> {
  for (;;) {
    const [command, payload] = (await connection.recv()) as [string, unknown];
    if (command === "stop") return;
    if (command === "initialize") {
      connection.send([
        "initialized",
        [
          config.accelerator === "cpu" ? [CPU_PROVIDER] : ["CUDAExecutionProvider", CPU_PROVIDER],
          config.dimension,
        ],
      ]);
      continue;
    }
    if (command === "probe") {
      connection.send(["probed", PROBE_TEXTS.map(() => unitVector(config.dimension))]);
      continue;
    }
    if (command === "plan_and_embed") {
      const candidates = (payload as { candidates?: unknown[] }).candidates ?? [null];
      connection.send([
        "planned",
        [candidates.map(() => [[0, 1, 1, unitVector(config.dimension)]]), true],
      ]);
    }
  }
}

async function crashingWorker(connection: WorkerConnection): Promise<void> {
  await connection.recv();
  throw new Error("accelerator crashed");
}

function selection(): BackendSelection {
  return new BackendSelection({
    requested: "cuda",
    descriptor: new BackendDescriptor({
      accelerator: "cuda",
      provider: "CUDAExecutionProvider",
      device: "cuda:0",
      stability: "automatic",
      precision: "float32",
    }),
    availableProviders: ["CUDAExecutionProvider", CPU_PROVIDER],
  });
}

function config(directory: string, accelerator: string) {
  return workerConfig({
    cacheDirectory: directory,
    offline: true,
    threads: 1,
    enableCpuMemArena: false,
    dimension: DIMENSION,
    providers:
      accelerator === "cpu" ? CPU_BACKEND.providers : ["CUDAExecutionProvider", CPU_PROVIDER],
    accelerator,
  });
}

function sessionFactory(
  directory: string,
  acceleratorTarget: WorkerTarget,
  {
    strict = false,
    crossoverCharacters = 0,
  }: { strict?: boolean; crossoverCharacters?: number } = {},
): () => PassageBackendSession {
  return () =>
    new PassageBackendSession(selection(), {
      acceleratorFactory: () =>
        new EmbeddingWorkerSession(config(directory, "cuda"), {
          effectiveCeilingBytes: 2 * 1024 ** 3,
          launcher: new FunctionLauncher(acceleratorTarget),
        }),
      cpuFactory: () =>
        new EmbeddingWorkerSession(config(directory, "cpu"), {
          effectiveCeilingBytes: 2 * 1024 ** 3,
          launcher: new FunctionLauncher(healthyWorker),
        }),
      strict,
      dimension: DIMENSION,
      crossoverCharacters,
    });
}

function makeIndexer(factory: (() => PassageBackendSession) | undefined) {
  const store = new LanceStore(path.join(temporary, "data"), { vectorDimension: DIMENSION });
  return {
    store,
    indexer: new Indexer({
      store,
      scanner: new SourceScanner(),
      extractor: new TreeSitterExtractor(),
      embedder: new RecordingEmbedder(),
      lockDirectory: path.join(temporary, "locks"),
      segmentPlan: segmentPlan({ maxTokens: 64, maxItems: 4 }),
      ...(factory === undefined ? {} : { passageSessionFactory: factory }),
      stagingDirectory: path.join(temporary, "staging"),
    }),
  };
}

function repository(files = 3): string {
  const root = path.join(temporary, "repo");
  fs.mkdirSync(root, { recursive: true });
  for (let index = 0; index < files; index += 1) {
    fs.writeFileSync(
      path.join(root, `module_${index}.py`),
      `def function_${index}(value):\n    return value + ${index}\n`,
    );
  }
  return root;
}

let temporary: string;

beforeEach(() => {
  temporary = temporaryDirectory();
});

afterEach(() => {
  removeDirectory(temporary);
});

describe("Indexer passage backend", () => {
  test("a working accelerator indexes and is named in the report", async () => {
    const project = initializeProject(repository());
    const { indexer, store } = makeIndexer(sessionFactory(temporary, healthyWorker));
    const report = await indexer.index(project);
    expect(report.errors).toEqual([]);
    expect(report.worker_used).toBe(true);
    expect(["cuda", "cpu"]).toContain(report.embedding_backend);
    expect(await store.countChunks([project.id])).toBeGreaterThan(0);
  });

  test("a failing accelerator still produces a complete index", async () => {
    const project = initializeProject(repository());
    const { indexer, store } = makeIndexer(sessionFactory(temporary, crashingWorker));
    const report = await indexer.index(project);
    expect(report.errors).toEqual([]);
    expect(report.embedding_backend).toBe("cpu");
    expect(report.fallback_count).toBeGreaterThan(0);
    expect(await store.countChunks([project.id])).toBeGreaterThan(0);
  });

  test("strict mode fails the run instead of falling back", async () => {
    const project = initializeProject(repository());
    const { indexer } = makeIndexer(sessionFactory(temporary, crashingWorker, { strict: true }));
    try {
      await indexer.index(project);
      throw new Error("expected backend unavailable");
    } catch (error) {
      expect(isCodeIndexingError(error) && error.code === "BACKEND_UNAVAILABLE").toBe(true);
    }
  });
});
