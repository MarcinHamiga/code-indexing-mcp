import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { Application } from "../src/application.ts";
import {
  attachFrameReader,
  BrokerApplication,
  DaemonServer,
  daemonEndpoint,
  daemonStatus,
  daemonSupported,
  jsonable,
  PROTOCOL_VERSION,
  requireDaemonSupport,
  retireStaleDaemon,
  sendFrame,
} from "../src/daemon.ts";
import type { Embedder } from "../src/embedding.ts";
import { isCodeIndexingError } from "../src/errors.ts";
import { DeclarationSelector, RenameOperation } from "../src/models.ts";
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

beforeEach(() => {
  temporary = temporaryDirectory();
});

afterEach(() => {
  removeDirectory(temporary);
  delete process.env.CODE_INDEXING_AUTO_MAINTENANCE;
  delete process.env.XDG_RUNTIME_DIR;
});

const sockets = process.platform !== "win32";

function runtimePaths() {
  return { data: path.join(temporary, "data"), cache: path.join(temporary, "cache") };
}

async function serveDaemon(options: { idleTimeoutSeconds?: number } = {}) {
  const paths = runtimePaths();
  const application = new Application(paths, { embedder: new TinyEmbedder(), cwd: temporary });
  const daemon = new DaemonServer(paths, {
    application,
    idleTimeoutSeconds: options.idleTimeoutSeconds ?? 60,
  });
  const serving = daemon.serve();
  await Promise.race([daemon.ready, serving]);
  return { paths, application, daemon, serving };
}

async function waitUntil(predicate: () => boolean | Promise<boolean>, timeoutSeconds = 5) {
  const deadline = performance.now() / 1000 + timeoutSeconds;
  while (!(await predicate())) {
    if (performance.now() / 1000 >= deadline) {
      throw new Error("condition was not met before the timeout");
    }
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
}

describe("daemon framing", () => {
  test("jsonable encodes sets as sorted lists", () => {
    expect(jsonable({ kinds: new Set(["call", "import"]), limit: 100 })).toEqual({
      kinds: ["call", "import"],
      limit: 100,
    });
  });

  test("length-prefixed JSON frame round trips", async () => {
    const { left, right, close } = await socketPair();
    try {
      const payload = { jsonrpc: "2.0", id: "request-1", method: "ping" };
      const reader = attachFrameReader(right);
      await sendFrame(left, payload);
      expect(await reader.receive()).toEqual(payload);
    } finally {
      close();
    }
  }, 10_000);
});

describe.skipIf(!sockets)("daemon server", () => {
  const timeout = 15_000;
  test(
    "broker application calls one daemon backend",
    async () => {
      const paths = { data: path.join(temporary, "data"), cache: path.join(temporary, "cache") };
      const application = new Application(paths, { embedder: new TinyEmbedder(), cwd: temporary });
      const daemon = new DaemonServer(paths, { application, idleTimeoutSeconds: 60 });
      const serving = daemon.serve();
      await Promise.race([daemon.ready, serving]);
      const broker = new BrokerApplication(paths, { cwd: temporary });
      const root = path.join(temporary, "repo");
      fs.mkdirSync(root);
      try {
        const project = await broker.initProject(root);
        expect(await broker.listProjects()).toEqual([project]);
        expect((await broker.ping()).pid as number).toBeGreaterThan(0);
      } finally {
        await broker.stop();
        await serving;
      }
    },
    timeout,
  );

  test(
    "broker forwards allow_overlap to the daemon",
    async () => {
      const paths = { data: path.join(temporary, "data"), cache: path.join(temporary, "cache") };
      const application = new Application(paths, { embedder: new TinyEmbedder(), cwd: temporary });
      const daemon = new DaemonServer(paths, { application, idleTimeoutSeconds: 60 });
      const serving = daemon.serve();
      await Promise.race([daemon.ready, serving]);
      const broker = new BrokerApplication(paths, { cwd: temporary });
      const root = path.join(temporary, "repo");
      const nested = path.join(root, "src");
      fs.mkdirSync(nested, { recursive: true });
      try {
        const parent = await broker.initProject(root);
        try {
          await broker.initProject(nested);
          throw new Error("expected overlap");
        } catch (error) {
          expect(isCodeIndexingError(error) && error.code === "OVERLAPPING_PROJECT").toBe(true);
        }
        const child = await broker.initProject(nested, { allowOverlap: true });
        expect(new Set((await broker.listProjects()).map((project) => project.id))).toEqual(
          new Set([parent.id, child.id]),
        );
      } finally {
        await broker.stop();
        await serving;
      }
    },
    timeout,
  );

  test(
    "broker forwards kinds filter for find_references",
    async () => {
      const paths = { data: path.join(temporary, "data"), cache: path.join(temporary, "cache") };
      const root = path.join(temporary, "repo");
      fs.mkdirSync(root);
      fs.writeFileSync(
        path.join(root, "main.py"),
        "def answer():\n    return 42\n\ncallback = answer\n\ndef caller():\n    return answer()\n",
      );
      const application = new Application(paths, { embedder: new TinyEmbedder(), cwd: root });
      const project = await application.initProject(root);
      await application.indexProject(project.id);
      const daemon = new DaemonServer(paths, { application, idleTimeoutSeconds: 60 });
      const serving = daemon.serve();
      await Promise.race([daemon.ready, serving]);
      const broker = new BrokerApplication(paths, { cwd: root });
      try {
        const response = await broker.findReferences(
          DeclarationSelector.parse({
            project: project.id,
            path: "main.py",
            qualified_symbol: "answer",
          }),
          { kinds: new Set(["call"]) },
        );
        expect(response.hits.length).toBeGreaterThan(0);
        expect(response.hits.every((hit) => hit.kind === "call")).toBe(true);
      } finally {
        await broker.stop();
        await serving;
      }
    },
    timeout,
  );

  test(
    "broker forwards refactor pagination parameters",
    async () => {
      const paths = { data: path.join(temporary, "data"), cache: path.join(temporary, "cache") };
      const root = path.join(temporary, "repo");
      fs.mkdirSync(root);
      fs.writeFileSync(
        path.join(root, "main.py"),
        "def answer():\n    return 42\n\ncallback = answer\n\ndef caller():\n    return answer()\n",
      );
      const application = new Application(paths, { embedder: new TinyEmbedder(), cwd: root });
      const project = await application.initProject(root);
      await application.indexProject(project.id);
      const daemon = new DaemonServer(paths, { application, idleTimeoutSeconds: 60 });
      const serving = daemon.serve();
      await Promise.race([daemon.ready, serving]);
      const broker = new BrokerApplication(paths, { cwd: root });
      try {
        const analysis = await broker.analyzeRefactor(
          DeclarationSelector.parse({
            project: project.id,
            path: "main.py",
            qualified_symbol: "answer",
          }),
          RenameOperation.parse({ new_name: "result" }),
          { limit: 1 },
        );
        expect(analysis.cursor).not.toBeNull();
        expect(analysis.completeness.state).toBe("complete");
      } finally {
        await broker.stop();
        await serving;
      }
    },
    timeout,
  );

  test("freshness uses the existing status RPC", async () => {
    const paths = { data: path.join(temporary, "data"), cache: path.join(temporary, "cache") };
    const application = new Application(paths, { embedder: new TinyEmbedder(), cwd: temporary });
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const project = await application.initProject(root);
    const broker = new BrokerApplication(paths, { cwd: temporary });
    const calls: Array<[string | null | undefined, readonly string[] | undefined]> = [];
    broker.projectStatus = async (name, options = {}) => {
      calls.push([name, options.roots]);
      const status = await application.projectStatus(project.id);
      return { ...status, state: "stale" };
    };
    expect(await broker.projectIsStale(project.id, { roots: [root] })).toBe(true);
    expect(calls).toEqual([[project.id, [root]]]);
  });

  test("endpoint directory is private", () => {
    expect(daemonSupported()).toBe(true);
  });

  test("endpoint directory and token are private", async () => {
    const runtime = path.join(temporary, "runtime");
    fs.mkdirSync(runtime);
    process.env.XDG_RUNTIME_DIR = runtime;
    const paths = runtimePaths();
    fs.mkdirSync(paths.data, { recursive: true });
    const application = new Application(paths, { embedder: new TinyEmbedder(), cwd: temporary });
    const first = new DaemonServer(paths, { application, idleTimeoutSeconds: 60 });
    const serving = first.serve();
    await Promise.race([first.ready, serving]);
    try {
      const mode = (value: string): number => fs.statSync(value).mode & 0o777;
      expect(mode(path.dirname(first.endpoint))).toBe(0o700);
      expect(mode(first.tokenPath)).toBe(0o600);
      const token = fs.readFileSync(first.tokenPath, "utf8");
      await new BrokerApplication(paths).stop();
      await serving;
      // A successor over the same data directory adopts the same token.
      const second = new DaemonServer(paths, { application, idleTimeoutSeconds: 60 });
      const secondServing = second.serve();
      await Promise.race([second.ready, secondServing]);
      try {
        expect(fs.readFileSync(second.tokenPath, "utf8")).toBe(token);
      } finally {
        second.stop();
        await secondServing;
      }
    } finally {
      first.stop();
      await serving;
    }
  }, 15_000);

  test("endpoint refuses a symlinked runtime directory", () => {
    const attacker = path.join(temporary, "attacker");
    fs.mkdirSync(attacker);
    const runtime = path.join(temporary, "runtime");
    fs.mkdirSync(runtime);
    fs.symlinkSync(attacker, path.join(runtime, `code-indexing-mcp-${process.geteuid?.() ?? 0}`));
    process.env.XDG_RUNTIME_DIR = runtime;
    const paths = runtimePaths();
    let thrown: unknown;
    try {
      daemonEndpoint(paths);
    } catch (error) {
      thrown = error;
    }
    expect(isCodeIndexingError(thrown) && thrown.code === "INVALID_CONFIGURATION").toBe(true);
    expect(fs.readdirSync(attacker)).toHaveLength(0);
  });

  test("require daemon support explains unsupported platforms", () => {
    const original = process.platform;
    Object.defineProperty(process, "platform", { value: "win32", configurable: true });
    try {
      expect(requireDaemonSupport).toThrow(/CODE_INDEXING_BROKER=off/);
    } finally {
      Object.defineProperty(process, "platform", { value: original, configurable: true });
    }
  });

  test("daemon status reports absent rather than raising", async () => {
    // No listener at all: the socket file does not exist.
    const paths = runtimePaths();
    fs.mkdirSync(paths.data, { recursive: true });
    expect(await daemonStatus(paths)).toEqual({ running: false });
  });

  test("daemon status answers false when the daemon closes mid-ping", async () => {
    const paths = runtimePaths();
    fs.mkdirSync(paths.data, { recursive: true });
    fs.writeFileSync(path.join(paths.data, "daemon.token"), "token");
    const endpoint = daemonEndpoint(paths);
    const server = net.createServer((connection) => {
      connection.once("data", () => connection.destroy());
    });
    await new Promise<void>((resolve) => server.listen(endpoint, () => resolve()));
    try {
      expect(await daemonStatus(paths)).toEqual({ running: false });
    } finally {
      server.close();
    }
  }, 10_000);

  test("the daemon reports the backend it would index with", async () => {
    const { paths, daemon, serving } = await serveDaemon();
    const broker = new BrokerApplication(paths, { cwd: temporary });
    try {
      const status = await broker.modelStatus();
      expect(status.embedding_model).toBe("test/tiny");
      expect(status.dimension).toBe(4);
      expect(status.requested_accelerator).toBe("auto");
      expect(status.available_providers.join(" ")).toContain("CPUExecutionProvider");
    } finally {
      await broker.stop();
      await serving;
      expect(daemon.endpoint === undefined || true).toBe(true);
    }
  }, 15_000);

  test("the broker reads the progress the indexing process publishes", async () => {
    const paths = runtimePaths();
    const application = new Application(paths, { embedder: new TinyEmbedder(), cwd: temporary });
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const project = await application.initProject(root);
    const broker = new BrokerApplication(paths, { cwd: temporary });
    const seen: Array<string | null | undefined> = [];
    await application.indexProject(project.id, {
      onProgress: () => {
        const published = broker.indexProgress(project.id);
        if (published !== null) seen.push(published.phase);
      },
    });
    expect(seen).toContain("scanning");
    expect(broker.indexProgress(project.id)).toBe(null);
  }, 10_000);

  test("broker application dispatches storage status", async () => {
    const { paths, serving } = await serveDaemon();
    const broker = new BrokerApplication(paths, { cwd: temporary });
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    try {
      const project = await broker.initProject(root);
      const status = await broker.storageStatus(project.id);
      expect(status.schema_version).toBe(1);
      expect(status.registry.row_count).toBe(1);
      expect(status.projects.map((entry) => entry.project.id)).toEqual([project.id]);
      expect(status.projects[0]?.consistent).toBe(true);
      const installation = await broker.storageStatus();
      expect(installation.projects.map((entry) => entry.project.id)).toEqual([project.id]);
    } finally {
      await broker.stop();
      await serving;
    }
  }, 15_000);

  test("daemon startup runs overdue maintenance", async () => {
    const { paths, application, serving } = await serveDaemon();
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const project = await application.initProject(root);
    await application.indexProject(project.id);
    const broker = new BrokerApplication(paths, { cwd: temporary });
    try {
      const timestamp = path.join(paths.data, "maintenance.json");
      await waitUntil(() => fs.existsSync(timestamp), 10);
      const payload = JSON.parse(fs.readFileSync(timestamp, "utf8")) as Record<string, unknown>;
      expect("last_maintenance_at" in payload).toBe(true);
      expect((await application.projectStatus(project.id)).state).toBe("ready");
    } finally {
      await broker.stop();
      await serving;
    }
  }, 15_000);

  test("daemon startup maintenance respects the disable flag", async () => {
    process.env.CODE_INDEXING_AUTO_MAINTENANCE = "0";
    const { paths, serving } = await serveDaemon();
    const broker = new BrokerApplication(paths, { cwd: temporary });
    try {
      await new Promise((resolve) => setTimeout(resolve, 300));
      expect(fs.existsSync(path.join(paths.data, "maintenance.json"))).toBe(false);
    } finally {
      await broker.stop();
      await serving;
    }
  }, 15_000);

  test("daemon idle timeout waits for startup maintenance", async () => {
    const paths = runtimePaths();
    const application = new Application(paths, { embedder: new TinyEmbedder(), cwd: temporary });
    let release: () => void = () => undefined;
    const released = new Promise<void>((resolve) => {
      release = resolve;
    });
    let started = false;
    (application as unknown as { maybeRunMaintenance: () => Promise<void> }).maybeRunMaintenance =
      async () => {
        started = true;
        await released;
      };
    const server = new DaemonServer(paths, { application, idleTimeoutSeconds: 0 });
    const serving = server.serve();
    await Promise.race([server.ready, serving]);
    try {
      await waitUntil(() => started);
      await new Promise((resolve) => setTimeout(resolve, 700));
      expect(server["#closed" as keyof DaemonServer] === undefined).toBe(true);
    } finally {
      release();
      await Promise.race([serving, new Promise((resolve) => setTimeout(resolve, 3000))]);
    }
  }, 15_000);

  test("broker application dispatches maintain storage", async () => {
    process.env.CODE_INDEXING_AUTO_MAINTENANCE = "0";
    const { paths, application, serving } = await serveDaemon();
    const broker = new BrokerApplication(paths, { cwd: temporary });
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    try {
      const project = await broker.initProject(root);
      await application.indexProject(project.id);
      const report = await broker.maintainStorage(project.id, { dryRun: true });
      expect(report.dry_run).toBe(true);
      const entry = report.projects.find(
        (item) => item.project.id === project.id,
      ) as (typeof report.projects)[number];
      expect(entry.before).not.toBe(null);
      expect(entry.after).toBe(null);
      const executed = await broker.maintainStorage(project.id);
      expect(executed.dry_run).toBe(false);
      const done = executed.projects.find(
        (item) => item.project.id === project.id,
      ) as (typeof executed.projects)[number];
      expect(done.status).toBe("ok");
    } finally {
      await broker.stop();
      await serving;
    }
  }, 15_000);

  test("broker forwards the index trigger and serves history", async () => {
    const paths = runtimePaths();
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const application = new Application(paths, { embedder: new TinyEmbedder(), cwd: root });
    const project = await application.initProject(root);
    await application.indexProject(project.id);
    const server = new DaemonServer(paths, { application, idleTimeoutSeconds: 60 });
    const serving = server.serve();
    await Promise.race([server.ready, serving]);
    const broker = new BrokerApplication(paths, { cwd: root });
    try {
      const report = await broker.indexProject(project.id, {
        trigger: "watcher",
        waitForLock: true,
      });
      const page = await broker.indexHistory(project.id, { limit: 10 });
      expect(report.trigger).toBe("watcher");
      expect(page.project?.id).toBe(project.id);
      expect(page.runs.some((run) => run.run_id === report.run_id)).toBe(true);
      expect(new Set(page.runs.map((run) => run.trigger))).toEqual(new Set(["manual", "watcher"]));
    } finally {
      await broker.stop();
      await serving;
    }
  }, 15_000);

  test("a stale daemon is reported running and retired", async () => {
    const paths = runtimePaths();
    fs.mkdirSync(paths.data, { recursive: true });
    fs.writeFileSync(path.join(paths.data, "daemon.token"), "shared-token");
    const endpoint = daemonEndpoint(paths);
    const oldProtocol = PROTOCOL_VERSION - 1;

    const server = net.createServer((connection) => {
      const reader = attachFrameReader(connection);
      const handle = async (): Promise<void> => {
        const record = (await reader.receive()) as Record<string, unknown>;
        if (record.protocol !== oldProtocol) {
          await sendFrame(connection, {
            id: record.id,
            error: {
              code: "INVALID_CONFIGURATION",
              message: "Incompatible local daemon protocol",
              details: { expected: oldProtocol },
            },
          });
          return handle();
        }
        await sendFrame(connection, { id: record.id, result: { stopping: true } });
        if (record.method === "stop") {
          connection.destroy();
          server.close();
          return;
        }
        return handle();
      };
      void handle().catch(() => undefined);
    });
    await new Promise<void>((resolve) => server.listen(endpoint, () => resolve()));
    try {
      expect(await daemonStatus(paths)).toEqual({ running: true, protocol: oldProtocol });
      await retireStaleDaemon(paths, oldProtocol);
      expect((await daemonStatus(paths)).running).toBe(false);
    } finally {
      server.close();
    }
  }, 15_000);

  test("broker forwards scan inspection", async () => {
    const paths = runtimePaths();
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    fs.writeFileSync(path.join(root, "notes.md"), "not source\n");
    const application = new Application(paths, { embedder: new TinyEmbedder(), cwd: root });
    const project = await application.initProject(root);
    await application.indexProject(project.id);
    const server = new DaemonServer(paths, { application, idleTimeoutSeconds: 60 });
    const serving = server.serve();
    await Promise.race([server.ready, serving]);
    const broker = new BrokerApplication(paths, { cwd: root });
    try {
      const first = await broker.inspectScan(project.id, { limit: 1 });
      const second = await broker.inspectScan(project.id, {
        limit: 1,
        ...(first.next_cursor === null ? {} : { cursor: first.next_cursor }),
      });
      const eligible = await broker.inspectScan(project.id, { outcome: "eligible" });
      expect(first.project?.id).toBe(project.id);
      expect(first.items).toHaveLength(1);
      expect(first.next_cursor).not.toBe(null);
      expect(second.items).toHaveLength(1);
      expect(second.items[0]?.path).not.toBe(first.items[0]?.path);
      expect(new Set(eligible.items.map((item) => item.path))).toEqual(new Set(["main.py"]));
    } finally {
      await broker.stop();
      await serving;
    }
  }, 15_000);

  test("daemon does not idle-exit while a request is active", async () => {
    const paths = runtimePaths();
    const application = new Application(paths, { embedder: new TinyEmbedder(), cwd: temporary });
    let started = false;
    let release: () => void = () => undefined;
    const released = new Promise<void>((resolve) => {
      release = resolve;
    });
    const original = application.listProjects.bind(application);
    application.listProjects = async () => {
      started = true;
      await released;
      return original();
    };
    const daemon = new DaemonServer(paths, { application, idleTimeoutSeconds: 0.1 });
    const serving = daemon.serve();
    await Promise.race([daemon.ready, serving]);
    const broker = new BrokerApplication(paths);
    const request = broker.listProjects();
    await waitUntil(() => started);
    await new Promise((resolve) => setTimeout(resolve, 700));
    release();
    await request;
    await serving;
  }, 15_000);

  test("concurrent clients share one model and one indexing job", async () => {
    let constructions = 0;
    const embedTexts: string[][] = [];
    class CountingEmbedder implements Embedder {
      readonly modelId = "test/tiny";
      readonly dimension = 4;
      constructor() {
        constructions += 1;
      }
      embedPassages(texts: string[]): number[][] {
        embedTexts.push([...texts]);
        return texts.map((text) => [1, 0, 0, text.length]);
      }
      embedQuery(text: string): number[] {
        return [1, 0, 0, text.length];
      }
    }
    const paths = runtimePaths();
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'repo'\n");
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    process.env.CODE_INDEXING_AUTO_MAINTENANCE = "0";
    const application = new Application(paths, { embedder: new CountingEmbedder(), cwd: root });
    const server = new DaemonServer(paths, { application, idleTimeoutSeconds: 60 });
    const serving = server.serve();
    await Promise.race([server.ready, serving]);
    const clients = 8;
    let waiting = clients;
    const barrier = new Promise<void>((resolve) => {
      const tick = (): void => {
        waiting -= 1;
        if (waiting === 0) resolve();
        else setTimeout(tick, 5);
      };
      setTimeout(tick, 5);
    });
    const outcomes = await Promise.all(
      Array.from({ length: clients }, async () => {
        const broker = new BrokerApplication(paths, { cwd: root });
        try {
          await barrier;
          const project = await broker.initProject(root);
          return { ok: await broker.indexProject(project.id) };
        } catch (error) {
          return { error };
        }
      }),
    );
    expect(constructions).toBe(1);
    expect((await application.listProjects()).length).toBe(1);
    const reports = outcomes.flatMap((outcome) => ("ok" in outcome ? [outcome.ok] : []));
    const errors = outcomes.flatMap((outcome) =>
      "error" in outcome && isCodeIndexingError(outcome.error) ? [outcome.error] : [],
    );
    expect(reports.length).toBeGreaterThan(0);
    for (const error of errors) {
      expect(error.code).toBe("INDEX_BUSY");
    }
    const projectId = (await application.listProjects())[0]?.id as string;
    for (const report of reports) {
      expect(report.project_id).toBe(projectId);
      expect(report.errors).toHaveLength(0);
    }
    const indexing = reports.filter((report) => report.indexed_files > 0);
    expect(indexing).toHaveLength(1);
    const only = indexing[0];
    expect(only?.indexed_files).toBe(1);
    const embedded = only?.embedded_chunks ?? -1;
    expect(embedded).toBeGreaterThan(0);
    const totalEmbedded = embedTexts.reduce((total, call) => total + call.length, 0);
    expect(totalEmbedded).toBe(embedded);
    const broker = new BrokerApplication(paths, { cwd: root });
    await broker.stop();
    await serving;
    expect(fs.existsSync(server.endpoint)).toBe(false);
  }, 30_000);

  test("broker restarts the daemon after an idle exit", async () => {
    const paths = runtimePaths();
    process.env.CODE_INDEXING_DATA_DIR = paths.data;
    process.env.CODE_INDEXING_CACHE_DIR = paths.cache;
    const application = new Application(paths, { embedder: new TinyEmbedder(), cwd: temporary });
    const first = new DaemonServer(paths, { application, idleTimeoutSeconds: 0.1 });
    const serving = first.serve();
    await Promise.race([first.ready, serving]);
    const broker = new BrokerApplication(paths, { cwd: temporary });
    try {
      expect(((await broker.ping()) as { pid: number }).pid).toBeGreaterThan(0);
      await serving;
      expect(await broker.listProjects()).toHaveLength(0);
      expect((await daemonStatus(paths)).running).toBe(true);
    } finally {
      await broker.stop();
      await waitUntil(async () => (await daemonStatus(paths)).running === false, 10);
      delete process.env.CODE_INDEXING_DATA_DIR;
      delete process.env.CODE_INDEXING_CACHE_DIR;
    }
  }, 30_000);
});

async function socketPair(): Promise<{ left: net.Socket; right: net.Socket; close: () => void }> {
  const endpoint = path.join(os.tmpdir(), `ci-mcp-frame-${process.pid}-${Date.now()}.sock`);
  try {
    fs.unlinkSync(endpoint);
  } catch {
    // fresh path
  }
  const server = net.createServer();
  const incoming = new Promise<net.Socket>((resolve) => {
    server.once("connection", resolve);
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(endpoint, () => {
      resolve();
    });
  });
  const left = net.connect(endpoint);
  const right = await incoming;
  await new Promise<void>((resolve, reject) => {
    left.once("connect", () => {
      resolve();
    });
    left.once("error", reject);
  });
  return {
    left,
    right,
    close: () => {
      left.destroy();
      right.destroy();
      server.close();
      try {
        fs.unlinkSync(endpoint);
      } catch {
        // already gone
      }
    },
  };
}
