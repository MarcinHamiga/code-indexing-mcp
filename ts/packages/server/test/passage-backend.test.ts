/** Accelerator failover, crossover deferral, and probe caching. */

import { expect, test } from "bun:test";
import path from "node:path";
import { BackendDescriptor, BackendSelection, CPU_BACKEND, CPU_PROVIDER } from "../src/backends.ts";
import { PROBE_TEXTS, passageCandidate, segmentPlan } from "../src/embedding.ts";
import { EmbeddingWorkerSession, workerConfig } from "../src/embedding-worker.ts";
import { isCodeIndexingError } from "../src/errors.ts";
import { PassageBackendSession } from "../src/passage-backend.ts";
import { ProbeCache, probeKey } from "../src/probe-cache.ts";
import type { WorkerConnection } from "../src/worker-channel.ts";
import { encodeBytes } from "../src/worker-channel.ts";
import { FunctionLauncher, type WorkerTarget } from "../src/worker-launcher.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

function unitVector(dimension = 4): Uint8Array {
  const values = Array.from({ length: dimension }, (_, index) => (index === 0 ? 1 : 0));
  const out = Buffer.allocUnsafe(dimension * 4);
  for (const [index, value] of values.entries()) out.writeFloatLE(value, index * 4);
  return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
}

function acceleratorSelection(): BackendSelection {
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

async function healthyWorker(
  connection: WorkerConnection,
  config: { dimension: number; accelerator: string },
): Promise<void> {
  for (;;) {
    const [command] = (await connection.recv()) as [string, unknown];
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
      connection.send(["planned", [[[[0, 1, 1, unitVector(config.dimension)]]], true]]);
    }
  }
}

function config(directory: string, accelerator: string) {
  return workerConfig({
    cacheDirectory: directory,
    offline: true,
    threads: 1,
    enableCpuMemArena: false,
    dimension: 4,
    providers: accelerator === "cpu" ? [] : ["CUDAExecutionProvider", CPU_PROVIDER],
    accelerator,
  });
}

function workerSession(
  directory: string,
  accelerator: string,
  target: WorkerTarget,
): EmbeddingWorkerSession {
  return new EmbeddingWorkerSession(config(directory, accelerator), {
    effectiveCeilingBytes: 2 * 1024 ** 3,
    launcher: new FunctionLauncher(target),
  });
}

function backend(
  directory: string,
  {
    selection = acceleratorSelection(),
    acceleratorTarget = healthyWorker as WorkerTarget,
    cpuTarget = healthyWorker as WorkerTarget,
    strict = false,
    probeCache,
    probeKey: key,
    crossoverCharacters = 0,
    calibrationPlan,
    cpuMaxItems = 0,
  }: {
    selection?: BackendSelection;
    acceleratorTarget?: WorkerTarget;
    cpuTarget?: WorkerTarget;
    strict?: boolean;
    probeCache?: ProbeCache;
    probeKey?: ReturnType<typeof probeKey>;
    crossoverCharacters?: number | null;
    calibrationPlan?: ReturnType<typeof segmentPlan>;
    cpuMaxItems?: number;
  } = {},
): PassageBackendSession {
  return new PassageBackendSession(selection, {
    acceleratorFactory: () => workerSession(directory, "cuda", acceleratorTarget),
    cpuFactory: () => workerSession(directory, "cpu", cpuTarget),
    strict,
    ...(probeCache === undefined ? {} : { probeCache }),
    ...(key === undefined ? {} : { probeKey: key }),
    crossoverCharacters,
    ...(calibrationPlan === undefined ? {} : { calibrationPlan }),
    cpuMaxItems,
    dimension: 4,
  });
}

test("a working accelerator is used and reported", async () => {
  const directory = temporaryDirectory();
  try {
    const session = backend(directory);
    await session.enter();
    await session.planAndEmbed([passageCandidate("", "x")], segmentPlan());
    expect(session.backendUsed).toBe("cuda");
    expect(session.fallbackCount).toBe(0);
    await session.exit();
  } finally {
    removeDirectory(directory);
  }
});

test("a cpu selection never starts an accelerator", async () => {
  const directory = temporaryDirectory();
  try {
    let started = 0;
    const session = backend(directory, {
      selection: new BackendSelection({
        requested: "cpu",
        descriptor: CPU_BACKEND,
        availableProviders: [CPU_PROVIDER],
      }),
      acceleratorTarget: async () => {
        started += 1;
        throw new Error("accelerator must not start");
      },
    });
    await session.enter();
    await session.planAndEmbed([passageCandidate("", "x")], segmentPlan());
    expect(session.backendUsed).toBe("cpu");
    expect(started).toBe(0);
    await session.exit();
  } finally {
    removeDirectory(directory);
  }
});

test("a backend that fails verification is replaced by cpu", async () => {
  const directory = temporaryDirectory();
  try {
    const session = backend(directory, {
      acceleratorTarget: async (connection) => {
        const [command] = (await connection.recv()) as [string, unknown];
        if (command === "initialize") connection.send(["error", "load failed"]);
      },
    });
    await session.enter();
    await session.planAndEmbed([passageCandidate("", "x")], segmentPlan());
    expect(session.backendUsed).toBe("cpu");
    expect(session.fallbackCount).toBe(1);
    await session.exit();
  } finally {
    removeDirectory(directory);
  }
});

test("strict mode refuses the fallback when verification fails", async () => {
  const directory = temporaryDirectory();
  try {
    const session = backend(directory, {
      strict: true,
      acceleratorTarget: async (connection) => {
        const [command] = (await connection.recv()) as [string, unknown];
        if (command === "initialize") connection.send(["error", "load failed"]);
      },
    });
    await session.enter();
    let caught: unknown;
    try {
      await session.planAndEmbed([passageCandidate("", "x")], segmentPlan());
    } catch (error) {
      caught = error;
    }
    expect(isCodeIndexingError(caught)).toBe(true);
    if (!isCodeIndexingError(caught)) return;
    expect(caught.code).toBe("BACKEND_UNAVAILABLE");
    await session.exit(caught);
  } finally {
    removeDirectory(directory);
  }
});

test("a successful probe is cached and then reused", async () => {
  const directory = temporaryDirectory();
  try {
    const cache = new ProbeCache(path.join(directory, "probes.json"));
    const key = probeKey({
      modelId: "jina",
      modelArtifact: "a",
      accelerator: "cuda",
      provider: "CUDAExecutionProvider",
      runtimeVersion: "1",
      platform: "test",
      device: "cuda:0",
    });
    let probes = 0;
    const target: WorkerTarget = async (connection, config) => {
      for (;;) {
        const [command] = (await connection.recv()) as [string, unknown];
        if (command === "stop") return;
        if (command === "initialize") {
          connection.send(["initialized", [["CUDAExecutionProvider"], config.dimension]]);
          continue;
        }
        if (command === "probe") {
          probes += 1;
          connection.send(["probed", PROBE_TEXTS.map(() => unitVector(config.dimension))]);
          continue;
        }
        if (command === "plan_and_embed") {
          connection.send(["planned", [[[[0, 1, 1, unitVector(config.dimension)]]], true]]);
        }
      }
    };
    const first = backend(directory, {
      probeCache: cache,
      probeKey: key,
      acceleratorTarget: target,
    });
    await first.enter();
    await first.planAndEmbed([passageCandidate("", "x")], segmentPlan());
    await first.exit();
    const second = backend(directory, {
      probeCache: cache,
      probeKey: key,
      acceleratorTarget: target,
    });
    await second.enter();
    await second.planAndEmbed([passageCandidate("", "x")], segmentPlan());
    await second.exit();
    expect(probes).toBe(1);
    expect(second.probeState).toBe("cached");
  } finally {
    removeDirectory(directory);
  }
});

test("a run below the crossover never starts the accelerator", async () => {
  const directory = temporaryDirectory();
  try {
    let started = 0;
    const session = backend(directory, {
      crossoverCharacters: 10_000,
      acceleratorTarget: async () => {
        started += 1;
      },
    });
    await session.enter();
    await session.planAndEmbed([passageCandidate("", "x")], segmentPlan());
    expect(session.backendUsed).toBe("cpu");
    expect(started).toBe(0);
    expect(session.fallbackCount).toBe(0);
    await session.exit();
  } finally {
    removeDirectory(directory);
  }
});

test("a deferred start is not a fallback", async () => {
  const directory = temporaryDirectory();
  try {
    const session = backend(directory, { crossoverCharacters: 10_000 });
    await session.enter();
    await session.planAndEmbed([passageCandidate("", "x")], segmentPlan());
    expect(session.fallbackReason).toBeNull();
    expect(session.telemetry().selectionReason ?? "").toContain("below the");
    await session.exit();
  } finally {
    removeDirectory(directory);
  }
});

void encodeBytes;
