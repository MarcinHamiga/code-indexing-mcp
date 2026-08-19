/** The installer's probe invocation contract: one snake_case JSON line, exit codes. */

import { afterEach, expect, test } from "bun:test";
import { runProbeCli } from "../src/accelerator-probe-cli.ts";
import { CPU_PROVIDER } from "../src/backends.ts";
import { DEFAULT_MODEL, PROBE_TEXTS } from "../src/embedding.ts";
import {
  loadPassageModel,
  setLoadModel,
  type PassageWorkerModel,
} from "../src/embedding-worker.ts";

interface BufferSink {
  write(chunk: string): void;
  buffer: string;
}

interface Harness {
  stdout(): string;
  stderr(): string;
  run(argv: readonly string[]): Promise<number>;
}

function recordingSink(): BufferSink {
  const sink: BufferSink = {
    write(chunk: string): void {
      sink.buffer += chunk;
    },
    buffer: "",
  };
  return sink;
}

function harness(): Harness {
  const stdout = recordingSink();
  const stderr = recordingSink();
  return {
    stdout: () => stdout.buffer,
    stderr: () => stderr.buffer,
    run: (argv) => runProbeCli(argv, { stdout, stderr }),
  };
}

function mockModel(providers: readonly string[]): PassageWorkerModel {
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

test("a successful probe prints the installer record contract and exits zero", async () => {
  setLoadModel(() => mockModel(["WebGpuExecutionProvider", CPU_PROVIDER]));
  const run = harness();

  const status = await run.run(["--accelerator", "webgpu", "--dimension", "4", "--offline"]);

  expect(status).toBe(0);
  expect(run.stderr()).toBe("");
  const payload = JSON.parse(run.stdout()) as Record<string, unknown>;
  expect(payload.ok).toBe(true);
  expect(payload.accelerator).toBe("webgpu");
  expect(payload.interpreter).toBe(process.execPath);
  expect(payload.providers).toContain("WebGpuExecutionProvider");
  expect(payload.providers).toContain(CPU_PROVIDER);
  expect(payload.resolved_providers).toEqual(["WebGpuExecutionProvider", CPU_PROVIDER]);
  expect(payload.runtime_version).toBeTypeOf("string");
  expect(payload.python_version).toBeTypeOf("string");
  expect(payload.model_id).toBe(DEFAULT_MODEL);
  expect(payload.dimension).toBe(4);
});

test("a session that silently became CPU fails the probe with ok false", async () => {
  setLoadModel(() => mockModel([CPU_PROVIDER]));
  const run = harness();

  const status = await run.run(["--accelerator", "webgpu", "--dimension", "4"]);

  expect(status).toBe(1);
  expect(run.stderr()).toBe("");
  const payload = JSON.parse(run.stdout()) as Record<string, unknown>;
  expect(payload.ok).toBe(false);
  expect(payload.error).toBeTypeOf("string");
  expect(payload.error).toContain("session runs on");
});

test("missing --accelerator prints usage to stderr and exits two", async () => {
  const run = harness();

  const status = await run.run(["--dimension", "4"]);

  expect(status).toBe(2);
  expect(run.stdout()).toBe("");
  expect(run.stderr()).toContain("usage:");
  expect(run.stderr()).toContain("--accelerator");
});

test("an unknown accelerator is reported as a probe failure, not an option error", async () => {
  const run = harness();

  const status = await run.run(["--accelerator", "nonsense", "--dimension", "4"]);

  expect(status).toBe(1);
  expect(run.stderr()).toBe("");
  const payload = JSON.parse(run.stdout()) as Record<string, unknown>;
  expect(payload.ok).toBe(false);
  expect(payload.error).toContain("Unknown embedding accelerator");
});
