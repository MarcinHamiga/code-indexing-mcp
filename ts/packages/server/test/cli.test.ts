import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { Application, type RuntimePaths } from "../src/application.ts";
import {
  applicationFactory,
  benchmarkCommands,
  installerCommands,
  main,
  ProgressPrinter,
  serverFactory,
  VERSION,
} from "../src/cli.ts";
import type { Embedder } from "../src/embedding.ts";
import { IndexProgress } from "../src/models.ts";
import { runtimeRootHolder, writeCache } from "../src/update-check.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

class TinyEmbedder implements Embedder {
  readonly modelId = "test/tiny";
  readonly dimension = 4;
  embedPassages(texts: string[]): number[][] {
    return texts.map((text) => [1, 0, 0, text.length]);
  }
  embedQuery(text: string): number[] {
    return [1, 0, 0, text.length];
  }
}

let temporary: string;
let previousData: string | undefined;
let previousCache: string | undefined;
const restored: Array<() => void> = [];

beforeEach(() => {
  temporary = temporaryDirectory();
  previousData = process.env.CODE_INDEXING_DATA_DIR;
  previousCache = process.env.CODE_INDEXING_CACHE_DIR;
  process.env.CODE_INDEXING_DATA_DIR = path.join(temporary, "data");
  process.env.CODE_INDEXING_CACHE_DIR = path.join(temporary, "cache");
  process.env.CODE_INDEXING_BROKER = "off";
  process.env.CODE_INDEXING_OFFLINE = "1";
});

afterEach(() => {
  if (previousData === undefined) delete process.env.CODE_INDEXING_DATA_DIR;
  else process.env.CODE_INDEXING_DATA_DIR = previousData;
  if (previousCache === undefined) delete process.env.CODE_INDEXING_CACHE_DIR;
  else process.env.CODE_INDEXING_CACHE_DIR = previousCache;
  delete process.env.CODE_INDEXING_BROKER;
  delete process.env.CODE_INDEXING_OFFLINE;
  delete process.env.CODE_INDEXING_EMBED_ACCELERATOR;
  delete process.env.CODE_INDEXING_MCP_INSTALL_DIR;
  delete process.env.CODE_INDEXING_UPDATE_CHECK;
  while (restored.length > 0) restored.pop()?.();
  removeDirectory(temporary);
});

function useTinyEmbedder(): void {
  const original = applicationFactory.current;
  applicationFactory.current = (paths: RuntimePaths, options: { cwd?: string } = {}) =>
    new Application(paths, { embedder: new TinyEmbedder(), ...options });
  restored.push(() => {
    applicationFactory.current = original;
  });
}

function withPlatform(platform: string): void {
  const original = process.platform;
  Object.defineProperty(process, "platform", { value: platform, configurable: true });
  restored.push(() => {
    Object.defineProperty(process, "platform", { value: original, configurable: true });
  });
}

/** Make update-check believe this process runs from a managed install. */
function fakeManagedInstall(remoteSha: string): string {
  const installDirectory = path.join(temporary, "install");
  fs.mkdirSync(path.join(installDirectory, ".git", "refs", "heads"), { recursive: true });
  fs.writeFileSync(path.join(installDirectory, ".git", "HEAD"), "ref: refs/heads/main\n");
  fs.writeFileSync(
    path.join(installDirectory, ".git", "refs", "heads", "main"),
    `${"a".repeat(40)}\n`,
  );
  process.env.CODE_INDEXING_MCP_INSTALL_DIR = installDirectory;
  runtimeRootHolder.current = path.join(installDirectory, "app");
  fs.mkdirSync(path.join(installDirectory, "app"), { recursive: true });
  restored.push(() => {
    runtimeRootHolder.current = null;
  });
  const cache = path.join(temporary, "cache");
  writeCache(cache, {
    checkedAt: Date.now() / 1000,
    localSha: "a".repeat(40),
    remoteSha,
  });
  return installDirectory;
}

describe("CLI", () => {
  test("version flag prints the version and exits", async () => {
    const stdout = captureStdout();
    expect(await main(["--version"])).toBe(0);
    expect(stdout()).toMatch(new RegExp(`^code-indexing-mcp ${VERSION} `));
  });

  test("initializes and lists projects", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const stdout = captureStdout();
    expect(await main(["init", root])).toBe(0);
    const initialized = JSON.parse(stdout()) as { id: string; name: string };
    expect(initialized.name).toBe("repo");
    const listed = captureStdout();
    expect(await main(["projects", "list"])).toBe(0);
    const payload = JSON.parse(listed()) as Array<{ id: string }>;
    expect(payload.map((project) => project.id)).toEqual([initialized.id]);
  });

  test("init rejects a nested root without allow-overlap", async () => {
    const root = path.join(temporary, "repo");
    const nested = path.join(root, "src");
    fs.mkdirSync(nested, { recursive: true });
    expect(await main(["init", root])).toBe(0);
    const stderr = captureStderr();
    expect(await main(["init", nested])).toBe(2);
    expect(stderr()).toContain("OVERLAPPING_PROJECT");
    const stdout = captureStdout();
    expect(await main(["init", nested, "--allow-overlap"])).toBe(0);
    expect((JSON.parse(stdout()) as { name: string }).name).toBe("src");
  });

  test("installer commands are delegated without loading installer code", async () => {
    const calls: readonly string[][] = [];
    const original = installerCommands.run;
    installerCommands.run = (argv) => {
      (calls as string[][]).push([...argv]);
      return 7;
    };
    restored.push(() => {
      installerCommands.run = original;
    });

    expect(await main(["configure", "--no-tui", "--set", "CODE_INDEXING_OFFLINE=1"])).toBe(7);
    expect(calls).toEqual([["configure", "--no-tui", "--set", "CODE_INDEXING_OFFLINE=1"]]);
  });

  test("benchmark search passes options through and prints machine-readable output", async () => {
    const received: Record<string, unknown> = {};
    const original = benchmarkCommands.search;
    benchmarkCommands.search = async (_paths, options) => {
      Object.assign(received, options);
      return { schema_version: 1, scopes: { "1": {} } } as never;
    };
    restored.push(() => {
      benchmarkCommands.search = original;
    });
    const workDir = path.join(temporary, "benchmark");
    const stdout = captureStdout();
    expect(
      await main([
        "benchmark",
        "search",
        "--projects",
        "50",
        "--iterations",
        "3",
        "--work-dir",
        workDir,
      ]),
    ).toBe(0);
    expect((JSON.parse(stdout()) as { schema_version: number }).schema_version).toBe(1);
    expect(received.projects).toBe(50);
    expect(received.iterations).toBe(3);
    expect(received.workDir).toBe(workDir);
  });

  test("benchmark index passes options through and prints machine-readable output", async () => {
    const received: Record<string, unknown> = {};
    const original = benchmarkCommands.index;
    benchmarkCommands.index = async (_paths, options) => {
      Object.assign(received, options);
      return { schema_version: 1, scenarios: { cold_start: {} } } as never;
    };
    restored.push(() => {
      benchmarkCommands.index = original;
    });
    const workDir = path.join(temporary, "benchmark");
    const stdout = captureStdout();
    expect(
      await main([
        "benchmark",
        "index",
        "--files",
        "3",
        "--functions-per-file",
        "2",
        "--batch-size",
        "8",
        "--work-dir",
        workDir,
      ]),
    ).toBe(0);
    expect((JSON.parse(stdout()) as { schema_version: number }).schema_version).toBe(1);
    expect(received.files).toBe(3);
    expect(received.functionsPerFile).toBe(2);
    expect(received.batchSize).toBe(8);
    expect(received.workDir).toBe(workDir);
  });

  test("benchmark precision passes options through and prints machine-readable output", async () => {
    const received: Record<string, unknown> = {};
    const original = benchmarkCommands.precision;
    benchmarkCommands.precision = async (_paths, options) => {
      Object.assign(received, options);
      return { schema_version: 1, variants: {} } as never;
    };
    restored.push(() => {
      benchmarkCommands.precision = original;
    });
    const workDir = path.join(temporary, "benchmark");
    const stdout = captureStdout();
    expect(
      await main([
        "benchmark",
        "precision",
        "--passages",
        "40",
        "--iterations",
        "2",
        "--recall-floor",
        "0.99",
        "--rank-floor",
        "0.95",
        "--work-dir",
        workDir,
      ]),
    ).toBe(0);
    expect((JSON.parse(stdout()) as { schema_version: number }).schema_version).toBe(1);
    expect(received.passages).toBe(40);
    expect(received.iterations).toBe(2);
    expect(received.recallFloor).toBe(0.99);
    expect(received.rankFloor).toBe(0.95);
    expect(received.workDir).toBe(workDir);
  });

  test("serve falls back to direct when local sockets are unavailable", async () => {
    delete process.env.CODE_INDEXING_BROKER;
    withPlatform("win32");
    const served: Record<string, unknown> = {};
    const originalServer = serverFactory.current;
    serverFactory.current = (app) => {
      served.app = app;
      return {
        run: async () => {
          served.ran = true;
        },
      };
    };
    restored.push(() => {
      serverFactory.current = originalServer;
    });
    expect(await main(["serve"])).toBe(0);
    expect(served.ran).toBe(true);
    expect(served.app instanceof Application).toBe(true);
  });

  test("serve refuses an explicit broker opt-in without local sockets", async () => {
    process.env.CODE_INDEXING_BROKER = "on";
    withPlatform("win32");
    const stderr = captureStderr();
    expect(await main(["serve"])).toBe(2);
    expect(stderr()).toContain("CODE_INDEXING_BROKER=off");
  });

  test("reports the resolved embedding backend", async () => {
    process.env.CODE_INDEXING_EMBED_ACCELERATOR = "cpu";
    const stdout = captureStdout();
    expect(await main(["model", "status"])).toBe(0);
    const status = JSON.parse(stdout()) as Record<string, unknown>;
    expect(status.requested_accelerator).toBe("cpu");
    expect(status.resolved_accelerator).toBe("cpu");
    expect(status.execution_provider).toBe("CPUExecutionProvider");
    expect(status.probe_cache_state).toBe("not-applicable");
    expect(status.fallback_reason).toBe(null);
    expect(status.strict).toBe(false);
  });

  test("model status explains an accelerator it cannot honour", async () => {
    process.env.CODE_INDEXING_EMBED_ACCELERATOR = "cuda";
    const stdout = captureStdout();
    expect(await main(["model", "status"])).toBe(0);
    const status = JSON.parse(stdout()) as Record<string, unknown>;
    if ((status.available_providers as string[]).includes("CUDAExecutionProvider")) {
      return; // honoured on a CUDA host; nothing unhonourable to explain
    }
    expect(status.resolved_accelerator).toBe("cpu");
    expect(String(status.fallback_reason)).toContain("CUDAExecutionProvider");
  });

  test("projects list reports an available update on stderr", async () => {
    fakeManagedInstall("b".repeat(40));
    const stdout = captureStdout();
    const stderr = captureStderr();
    expect(await main(["projects", "list"])).toBe(0);
    expect(JSON.parse(stdout())).toEqual([]);
    expect(stderr()).toContain("code-indexing-mcp update");
  });

  test("update notice is silent when disabled", async () => {
    fakeManagedInstall("b".repeat(40));
    process.env.CODE_INDEXING_UPDATE_CHECK = "off";
    const stderr = captureStderr();
    expect(await main(["projects", "list"])).toBe(0);
    expect(stderr()).not.toContain("update is available");
  });

  test("index narrates its progress on stderr and keeps stdout JSON", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    useTinyEmbedder();
    expect(await main(["init", root])).toBe(0);
    const stdout = captureStdout();
    const stderr = captureStderr();
    expect(await main(["index", root])).toBe(0);
    expect((JSON.parse(stdout()) as { indexed_files: number }).indexed_files).toBe(1);
    const narration = stderr();
    expect(narration).toContain("Scanning for changed files");
    expect(narration).toContain("Embedding");
    expect(narration).toContain("Committing the index");
  }, 20_000);

  test("a terminal gets one status line that is cleaned up afterwards", () => {
    const written: string[] = [];
    const stream = {
      isTTY: true,
      write: (chunk: string | Uint8Array): boolean => {
        written.push(typeof chunk === "string" ? chunk : Buffer.from(chunk).toString());
        return true;
      },
    } as unknown as NodeJS.WriteStream;
    const printer = new ProgressPrinter(stream);
    printer.call(
      IndexProgress.parse({
        project_id: "abc",
        candidates_seen: 1,
        phase: "scanning",
      }),
    );
    printer.call(
      IndexProgress.parse({
        project_id: "abc",
        candidates_seen: 2,
        phase: "scanning",
      }),
    );
    printer.clear();
    const output = written.join("");
    expect(output.includes("\n")).toBe(false);
    expect(output).toContain("Scanning 2 candidates");
    expect(output.replace(/\r+$/, "").endsWith(" ".repeat("Scanning 2 candidates".length))).toBe(
      true,
    );
  });

  test("reports storage status as JSON", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    useTinyEmbedder();
    expect(await main(["init", root])).toBe(0);
    expect(await main(["index", root])).toBe(0);
    const stdout = captureStdout();
    expect(await main(["storage", "status", root])).toBe(0);
    const payload = JSON.parse(stdout()) as Record<string, any>;
    expect(payload.schema_version).toBe(1);
    expect(payload.registry.row_count).toBe(1);
    expect(payload.projects).toHaveLength(1);
    expect(payload.projects[0].consistent).toBe(true);
    expect(payload.projects[0].partition_physical_bytes).toBeGreaterThan(0);
    expect(payload.physical_bytes_total).toBeGreaterThan(0);
  }, 20_000);

  test("storage status defaults to the whole installation", async () => {
    useTinyEmbedder();
    const stdout = captureStdout();
    expect(await main(["storage", "status"])).toBe(0);
    const payload = JSON.parse(stdout()) as Record<string, any>;
    expect(payload.registry.row_count).toBe(0);
    expect(payload.projects).toEqual([]);
  });

  test("storage vacuum is dry-run by default", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    useTinyEmbedder();
    expect(await main(["init", root])).toBe(0);
    expect(await main(["index", root])).toBe(0);
    const stdout = captureStdout();
    expect(await main(["storage", "vacuum", root])).toBe(0);
    const payload = JSON.parse(stdout()) as Record<string, any>;
    expect(payload.schema_version).toBe(1);
    expect(payload.dry_run).toBe(true);
    expect(payload.trigger).toBe("manual");
    const entry = payload.projects[0] as Record<string, any>;
    expect(entry.status).toBe("skipped");
    expect(entry.before.partition_physical_bytes).toBeGreaterThan(0);
    expect(entry.after).toBe(null);
  }, 20_000);

  test("storage vacuum requires the execute flag to mutate", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    useTinyEmbedder();
    expect(await main(["init", root])).toBe(0);
    expect(await main(["index", root])).toBe(0);
    const stdout = captureStdout();
    expect(await main(["storage", "vacuum", root, "--execute"])).toBe(0);
    const payload = JSON.parse(stdout()) as Record<string, any>;
    expect(payload.dry_run).toBe(false);
    const entry = payload.projects[0] as Record<string, any>;
    expect(entry.status).toBe("ok");
    expect(entry.after).not.toBe(null);
    expect(payload.registry_after).not.toBe(null);
  }, 20_000);

  test("reports indexing history as JSON", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    useTinyEmbedder();
    expect(await main(["init", root])).toBe(0);
    expect(await main(["index", root])).toBe(0);
    const stdout = captureStdout();
    expect(await main(["history", root])).toBe(0);
    const payload = JSON.parse(stdout()) as Record<string, any>;
    expect(payload.schema_version).toBe(1);
    expect(payload.project.id).toBeTruthy();
    expect(payload.runs).toHaveLength(1);
    expect(payload.runs[0].trigger).toBe("manual");
    expect(payload.runs[0].state).toBe("completed");
    expect(payload.runs[0].chunks_embedded).toBeGreaterThanOrEqual(1);
  }, 20_000);

  test("reports scan inspection as JSON", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    fs.writeFileSync(path.join(root, "notes.md"), "not source\n");
    useTinyEmbedder();
    expect(await main(["init", root])).toBe(0);
    expect(await main(["index", root])).toBe(0);
    const stdout = captureStdout();
    expect(await main(["scan", root, "--outcome", "eligible"])).toBe(0);
    const payload = JSON.parse(stdout()) as Record<string, any>;
    expect(payload.schema_version).toBe(1);
    expect(payload.project.id).toBeTruthy();
    expect(payload.items.map((item: Record<string, unknown>) => item.path)).toEqual(["main.py"]);
    expect(payload.items[0].language).toBe("python");
    expect(payload.items[0].outcome).toBe("eligible");
    expect(payload.next_cursor).toBe(null);
  }, 20_000);
});

function captureStdout(): () => string {
  const writes: string[] = [];
  const original = process.stdout.write.bind(process.stdout);
  process.stdout.write = ((chunk: string | Uint8Array, ...rest: unknown[]) => {
    writes.push(typeof chunk === "string" ? chunk : Buffer.from(chunk).toString());
    return original(chunk, ...(rest as []));
  }) as typeof process.stdout.write;
  return () => {
    process.stdout.write = original;
    return writes.join("");
  };
}

function captureStderr(): () => string {
  const writes: string[] = [];
  const original = process.stderr.write.bind(process.stderr);
  process.stderr.write = ((chunk: string | Uint8Array, ...rest: unknown[]) => {
    writes.push(typeof chunk === "string" ? chunk : Buffer.from(chunk).toString());
    return original(chunk, ...(rest as []));
  }) as typeof process.stderr.write;
  return () => {
    process.stderr.write = original;
    return writes.join("");
  };
}
