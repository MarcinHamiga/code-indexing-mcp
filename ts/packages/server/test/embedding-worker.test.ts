/** Memory ceiling, batch retry, and worker protocol. */

import { expect, test } from "bun:test";
import { spawn } from "node:child_process";
import {
  PROBE_TEXTS,
  packVector as packFromEmbedding,
  passageCandidate,
  segmentPlan,
} from "../src/embedding.ts";
import {
  defaultLauncher,
  EmbeddingWorkerSession,
  childRss,
  effectiveMemoryCeiling,
  indexingMemoryBytes,
  MINIMUM_WORKER_BYTES,
  SYSTEM_RESERVE_BYTES,
  workerConfig,
} from "../src/embedding-worker.ts";
import { isCodeIndexingError } from "../src/errors.ts";
import type { WorkerConnection } from "../src/worker-channel.ts";
import { encodeBytes } from "../src/worker-channel.ts";
import { FunctionLauncher, type WorkerTarget } from "../src/worker-launcher.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

function unitRow(dimension = 4): Uint8Array {
  const values = Array.from({ length: dimension }, (_, index) => (index === 0 ? 1 : 0));
  return packFromEmbedding(values);
}

async function fakeWorker(
  connection: WorkerConnection,
  config: { dimension: number },
): Promise<void> {
  for (;;) {
    const [command] = (await connection.recv()) as [string, unknown];
    if (command === "stop") return;
    if (command === "initialize") {
      connection.send(["initialized", [["CPUExecutionProvider"], config.dimension]]);
      continue;
    }
    if (command === "memory") {
      connection.send(["memory", 1]);
      continue;
    }
    if (command === "probe") {
      connection.send(["probed", PROBE_TEXTS.map(() => unitRow(config.dimension))]);
      continue;
    }
    if (command === "embed") {
      connection.send(["packed", [unitRow(config.dimension)]]);
    }
  }
}

function session(
  directory: string,
  {
    target = fakeWorker as WorkerTarget,
    effectiveCeilingBytes = 2 * 1024 ** 3,
    sampleRss,
    dimension = 4,
  }: {
    target?: WorkerTarget;
    effectiveCeilingBytes?: number;
    sampleRss?: () => [number, number];
    dimension?: number;
  } = {},
): EmbeddingWorkerSession {
  return new EmbeddingWorkerSession(
    workerConfig({
      cacheDirectory: directory,
      offline: true,
      threads: 1,
      enableCpuMemArena: false,
      dimension,
    }),
    {
      effectiveCeilingBytes,
      launcher: new FunctionLauncher(target),
      ...(sampleRss === undefined ? {} : { sampleRss }),
    },
  );
}

test("pack vector accepts rows that only expose tolist", () => {
  const packed = packFromEmbedding({ tolist: () => [1, 0, 0, 0] });
  expect(packed.byteLength).toBe(16);
  expect(new Float32Array(packed.buffer, packed.byteOffset, 4)[0]).toBe(1);
});

test("effective memory ceiling reserves system memory", () => {
  expect(
    effectiveMemoryCeiling({ configuredBytes: 8 * 1024 ** 3, availableBytes: 2 * 1024 ** 3 }),
  ).toBe(2 * 1024 ** 3 - SYSTEM_RESERVE_BYTES);
});

test("effective memory ceiling uses configured limit when ram is available", () => {
  expect(
    effectiveMemoryCeiling({ configuredBytes: 2 * 1024 ** 3, availableBytes: 16 * 1024 ** 3 }),
  ).toBe(2 * 1024 ** 3);
});

test("embedding worker round trips vectors and stops", async () => {
  const directory = temporaryDirectory();
  try {
    const worker = session(directory);
    const info = await worker.initialize();
    expect(info.dimension).toBe(4);
    expect(await worker.reportMemory()).toBe(1);
    const vectors = await worker.embedPassages(["hello"]);
    expect(vectors[0]?.[0]).toBe(1);
    await worker.close();
    expect(worker.pid).toBeUndefined();
  } finally {
    removeDirectory(directory);
  }
});

test("the default launcher starts a distinct operating-system process", async () => {
  const directory = temporaryDirectory();
  try {
    const launched = await defaultLauncher().launch(
      workerConfig({
        cacheDirectory: directory,
        offline: true,
        threads: 1,
        enableCpuMemArena: false,
        dimension: 4,
      }),
    );
    expect(launched.process.pid).toBeDefined();
    expect(launched.process.pid).not.toBe(process.pid);
    launched.process.terminate();
    await launched.process.join(2);
    launched.connection.close();
  } finally {
    removeDirectory(directory);
  }
});

test("embedding worker refuses unsafe effective budget", () => {
  const directory = temporaryDirectory();
  try {
    let caught: unknown;
    try {
      session(directory, { effectiveCeilingBytes: MINIMUM_WORKER_BYTES - 1 });
    } catch (error) {
      caught = error;
    }
    expect(isCodeIndexingError(caught)).toBe(true);
    if (!isCodeIndexingError(caught)) return;
    expect(caught.code).toBe("INDEX_RESOURCE_LIMIT");
  } finally {
    removeDirectory(directory);
  }
});

test("indexing memory excludes the parent baseline", () => {
  expect(
    indexingMemoryBytes({ parentBytes: 2_000, workerBytes: 100, parentBaselineBytes: 1_900 }),
  ).toBe(200);
});

test("indexing memory counts parent growth during indexing", () => {
  expect(
    indexingMemoryBytes({ parentBytes: 3_000, workerBytes: 100, parentBaselineBytes: 1_000 }),
  ).toBe(2_100);
});

test("indexing memory never goes negative when the parent shrinks", () => {
  expect(
    indexingMemoryBytes({ parentBytes: 500, workerBytes: 100, parentBaselineBytes: 1_000 }),
  ).toBe(100);
});

test("worker RSS is sampled from the requested process", async () => {
  const child = spawn(process.execPath, ["-e", "setTimeout(() => {}, 10_000)"], {
    stdio: "ignore",
  });
  try {
    expect(child.pid).toBeDefined();
    expect(await childRss(child.pid as number)).toBeGreaterThan(0);
  } finally {
    child.kill();
  }
});

test("resident parent memory does not trip the ceiling", async () => {
  const directory = temporaryDirectory();
  try {
    const resident = MINIMUM_WORKER_BYTES + SYSTEM_RESERVE_BYTES;
    const worker = session(directory, {
      effectiveCeilingBytes: MINIMUM_WORKER_BYTES,
      sampleRss: () => [resident, 1],
    });
    worker.parentBaselineBytes = resident;
    await worker.initialize();
    await worker.close();
  } finally {
    removeDirectory(directory);
  }
});

test("worker growth still trips the ceiling", async () => {
  const directory = temporaryDirectory();
  try {
    const worker = session(directory, {
      effectiveCeilingBytes: MINIMUM_WORKER_BYTES,
      sampleRss: () => [100, MINIMUM_WORKER_BYTES + HARD_OVERSHOOT()],
    });
    let caught: unknown;
    try {
      await worker.initialize();
    } catch (error) {
      caught = error;
    }
    expect(isCodeIndexingError(caught)).toBe(true);
    if (!isCodeIndexingError(caught)) return;
    expect(caught.code).toBe("INDEX_RESOURCE_LIMIT");
    expect(worker.terminationReason).toBe("memory_ceiling");
  } finally {
    removeDirectory(directory);
  }
});

function HARD_OVERSHOOT(): number {
  return 128 * 1024 ** 2 + 1;
}

test("a probe that returns usable vectors is accepted", async () => {
  const directory = temporaryDirectory();
  try {
    const worker = session(directory);
    const vectors = await worker.probe();
    expect(vectors.length).toBe(PROBE_TEXTS.length);
    await worker.close();
  } finally {
    removeDirectory(directory);
  }
});

test("a probe whose vectors could not search an index is rejected", async () => {
  const directory = temporaryDirectory();
  try {
    const worker = session(directory, {
      target: async (connection) => {
        for (;;) {
          const [command] = (await connection.recv()) as [string, unknown];
          if (command === "stop") return;
          if (command === "probe") {
            connection.send([
              "probed",
              [encodeBytes(Buffer.alloc(8)), encodeBytes(Buffer.alloc(8))],
            ]);
          }
        }
      },
    });
    let caught: unknown;
    try {
      await worker.probe();
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(Error);
    await worker.close();
  } finally {
    removeDirectory(directory);
  }
});

test("the default worker config requests no providers", () => {
  const config = workerConfig({
    cacheDirectory: "/tmp",
    offline: true,
    threads: 1,
    enableCpuMemArena: false,
    dimension: 768,
  });
  expect(config.providers).toEqual([]);
  expect(config.accelerator).toBe("cpu");
});

test("plan and embed returns a segment per token window", async () => {
  const directory = temporaryDirectory();
  try {
    const worker = new EmbeddingWorkerSession(
      workerConfig({
        cacheDirectory: directory,
        offline: true,
        threads: 1,
        enableCpuMemArena: false,
        dimension: 4,
      }),
      {
        effectiveCeilingBytes: 2 * 1024 ** 3,
        launcher: new FunctionLauncher(async (connection) => {
          for (;;) {
            const [command, payload] = (await connection.recv()) as [string, unknown];
            if (command === "stop") return;
            if (command === "plan_and_embed") {
              const [candidates] = payload as [Array<[string, string]>, unknown];
              connection.send([
                "planned",
                [candidates.map(([_, content]) => [[0, content.length, 1, unitRow()]]), false],
              ]);
            }
          }
        }),
      },
    );
    const result = await worker.planAndEmbed(
      [passageCandidate("kind: module", "value = 1")],
      segmentPlan(),
    );
    expect(result[0]?.length).toBe(1);
    expect(result[0]?.[0]?.startChar).toBe(0);
    expect(worker.segmentCount).toBe(1);
    await worker.close();
  } finally {
    removeDirectory(directory);
  }
});

test("an unplannable candidate raises a plain error the file absorbs", async () => {
  const directory = temporaryDirectory();
  try {
    const worker = new EmbeddingWorkerSession(
      workerConfig({
        cacheDirectory: directory,
        offline: true,
        threads: 1,
        enableCpuMemArena: false,
        dimension: 4,
      }),
      {
        effectiveCeilingBytes: 2 * 1024 ** 3,
        launcher: new FunctionLauncher(async (connection) => {
          for (;;) {
            const [command] = (await connection.recv()) as [string, unknown];
            if (command === "stop") return;
            if (command === "plan_and_embed") connection.send(["plan_error", "too many windows"]);
          }
        }),
      },
    );
    let caught: unknown;
    try {
      await worker.planAndEmbed([passageCandidate("", "x")], segmentPlan());
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(Error);
    expect(isCodeIndexingError(caught)).toBe(false);
    await worker.close();
  } finally {
    removeDirectory(directory);
  }
});

test("a failed batch is retried at a halved microbatch size", async () => {
  const directory = temporaryDirectory();
  try {
    const worker = new EmbeddingWorkerSession(
      workerConfig({
        cacheDirectory: directory,
        offline: true,
        threads: 1,
        enableCpuMemArena: false,
        dimension: 4,
      }),
      {
        effectiveCeilingBytes: 2 * 1024 ** 3,
        launcher: new FunctionLauncher(async (connection) => {
          for (;;) {
            const [command, payload] = (await connection.recv()) as [string, unknown];
            if (command === "stop") return;
            if (command === "plan_and_embed") {
              const plan = (payload as [unknown, { maxItems: number }])[1];
              if (plan.maxItems > 1) {
                connection.send(["error", "boom"]);
                continue;
              }
              connection.send(["planned", [[[[0, 1, 1, unitRow()]]], true]]);
            }
          }
        }),
      },
    );
    const result = await worker.planAndEmbed(
      [passageCandidate("", "x")],
      segmentPlan({ maxItems: 4 }),
    );
    expect(result[0]?.length).toBe(1);
    expect(worker.retryCount).toBeGreaterThan(0);
    expect(worker.safeMaxItems).toBe(1);
    await worker.close();
  } finally {
    removeDirectory(directory);
  }
});

test("telemetry names the backend the worker ran on", async () => {
  const directory = temporaryDirectory();
  try {
    const worker = session(directory);
    await worker.initialize();
    expect(worker.telemetry().backend).toBe("cpu");
    await worker.close();
  } finally {
    removeDirectory(directory);
  }
});
