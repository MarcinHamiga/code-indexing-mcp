import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { Application } from "../src/application.ts";
import type { Embedder } from "../src/embedding.ts";
import { CodeIndexingError } from "../src/errors.ts";
import { acquireLock } from "../src/indexing.ts";
import {
  createServer,
  EAGER_RECONCILE_SECONDS,
  PROGRESS_POLL_SECONDS,
  WATCH_RETRY_INITIAL_SECONDS,
  WATCH_RETRY_MAXIMUM_SECONDS,
  type WatchRoot,
  watchRoot,
} from "../src/server.ts";
import { caseInsensitiveAlias, removeDirectory, temporaryDirectory } from "./helpers.ts";

function tinyVectors(texts: string[]): number[][] {
  return texts.map((text) => [1, 0, 0, text.length]);
}

class TinyEmbedder implements Embedder {
  readonly modelId = "test/tiny";
  readonly dimension = 4;
  embedPassages(texts: string[]): number[][] {
    return tinyVectors(texts);
  }
  embedQuery(text: string): number[] {
    return [1, 0, 0, text.length];
  }
}

/** Blocks in embedPassages until released, like the Python BlockingEmbedder. */
class BlockingEmbedder implements Embedder {
  readonly modelId = "test/tiny";
  readonly dimension = 4;
  started = false;
  release: Promise<void>;
  #letGo: () => void = () => undefined;

  constructor() {
    this.release = new Promise<void>((resolve) => {
      this.#letGo = resolve;
    });
  }

  letGo(): void {
    this.#letGo();
  }

  async embedPassages(texts: string[]): Promise<number[][]> {
    this.started = true;
    await this.release;
    return tinyVectors(texts);
  }

  embedQuery(text: string): number[] {
    return [1, 0, 0, text.length];
  }
}

/** Blocks only while `block` is set, like the Python SwitchableBlockingEmbedder. */
class SwitchableBlockingEmbedder implements Embedder {
  readonly modelId = "test/tiny";
  readonly dimension = 4;
  block = false;
  started = false;
  release: Promise<void>;
  #letGo: () => void = () => undefined;

  constructor() {
    this.release = new Promise<void>((resolve) => {
      this.#letGo = resolve;
    });
  }

  letGo(): void {
    this.#letGo();
  }

  async embedPassages(texts: string[]): Promise<number[][]> {
    if (this.block) {
      this.started = true;
      await this.release;
    }
    return tinyVectors(texts);
  }

  embedQuery(text: string): number[] {
    return [1, 0, 0, text.length];
  }
}

/** Fails the first embedPassages call with MODEL_UNAVAILABLE, then behaves. */
class FlakyEmbedder implements Embedder {
  readonly modelId = "test/tiny";
  readonly dimension = 4;
  calls = 0;

  async embedPassages(texts: string[]): Promise<number[][]> {
    this.calls += 1;
    if (this.calls === 1) {
      throw new CodeIndexingError("MODEL_UNAVAILABLE", "embedding backend unavailable");
    }
    return tinyVectors(texts);
  }

  embedQuery(text: string): number[] {
    return [1, 0, 0, text.length];
  }
}

/** Fails every embedPassages call with MODEL_UNAVAILABLE. */
class FailingEmbedder implements Embedder {
  readonly modelId = "test/tiny";
  readonly dimension = 4;
  calls = 0;

  async embedPassages(): Promise<number[][]> {
    this.calls += 1;
    throw new CodeIndexingError("MODEL_UNAVAILABLE", "embedding backend unavailable");
  }

  embedQuery(text: string): number[] {
    return [1, 0, 0, text.length];
  }
}

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve: () => void = () => undefined;
  const promise = new Promise<void>((settled) => {
    resolve = settled;
  });
  return { promise, resolve };
}

async function waitUntil(
  predicate: () => boolean | Promise<boolean>,
  timeoutSeconds = 5,
): Promise<void> {
  const deadline = performance.now() / 1000 + timeoutSeconds;
  while (!(await predicate())) {
    if (performance.now() / 1000 >= deadline) {
      throw new Error("condition was not met before the timeout");
    }
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
}

/** A watch backend whose yields are driven by the test, standing in for watchfiles. */
function gatedWatch(...gates: Array<Promise<void>>): WatchRoot {
  return async function* () {
    for (const gate of gates) {
      await gate;
      yield;
    }
    await new Promise(() => undefined);
  };
}

/** Write with a visibly later timestamp so polling backends see the change. */
function writeWithLaterMtime(file: string, content: string): void {
  const previous = fs.statSync(file).mtimeMs;
  fs.writeFileSync(file, content);
  fs.utimesSync(file, new Date(), new Date(previous + 2000));
}

let temporary: string;
const environment = new Map<string, string | undefined>();
const timings: Array<[typeof EAGER_RECONCILE_SECONDS, number]> = [
  [EAGER_RECONCILE_SECONDS, EAGER_RECONCILE_SECONDS.value],
  [WATCH_RETRY_INITIAL_SECONDS, WATCH_RETRY_INITIAL_SECONDS.value],
  [WATCH_RETRY_MAXIMUM_SECONDS, WATCH_RETRY_MAXIMUM_SECONDS.value],
  [PROGRESS_POLL_SECONDS, PROGRESS_POLL_SECONDS.value],
];
const originalWatchRoot = watchRoot.current;

function setEnv(name: string, value: string): void {
  environment.set(name, process.env[name]);
  process.env[name] = value;
}

beforeEach(() => {
  temporary = temporaryDirectory();
  delete process.env.CODE_INDEXING_MODE;
});

afterEach(() => {
  removeDirectory(temporary);
  for (const [name, value] of environment) {
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }
  environment.clear();
  for (const [holder, value] of timings) holder.value = value;
  watchRoot.current = originalWatchRoot;
});

interface StorageStatusPayload {
  schema_version: number;
  registry: { name: string; row_count: number; logical_bytes: number };
  physical_bytes_total: number;
  projects: Array<{
    project: { id: string };
    consistent: boolean;
    partition_open_failed: boolean;
    partition_physical_bytes: number;
    tables: Array<{ name: string }>;
  }>;
}

interface MaintenancePayload {
  schema_version: number;
  dry_run: boolean;
  trigger: string;
  retention_hours: number;
  projects: Array<{
    status: string;
    after: unknown;
    before: { partition_physical_bytes: number };
  }>;
  registry_after: unknown;
}

interface HistoryPayload {
  schema_version: number;
  project: { id: string };
  runs: Array<{
    state: string;
    trigger: string;
    run_id: string;
    chunks_embedded: number;
  }>;
}

interface ProjectStatusPayload {
  last_run: { state: string; trigger: string; eligible_files: number };
  progress: unknown;
}

interface ScanPagePayload {
  schema_version: number;
  project: { id: string };
  items: Array<{ path: string; outcome: string; language: string; size: number | null }>;
  next_cursor: string | null;
}

function tinyApp(): Application {
  return new Application(
    { data: path.join(temporary, "data"), cache: path.join(temporary, "cache") },
    { embedder: new TinyEmbedder(), cwd: temporary },
  );
}

function paths(): { data: string; cache: string } {
  return { data: path.join(temporary, "data"), cache: path.join(temporary, "cache") };
}

/**
 * Bun abandons timed-out tests rather than cancelling them, so a test that
 * blocks mid-flight must drain its background work before returning;
 * otherwise the abandoned job waits on files the next test's cleanup removes.
 */
async function settle(server: { coordinator: { tasks: Promise<void>[] } | null }): Promise<void> {
  const tasks = server.coordinator?.tasks ?? [];
  await Promise.race([
    Promise.all([...tasks]),
    new Promise((resolve) => setTimeout(resolve, 5000)),
  ]);
}

const READ_ONLY_TOOLS = new Set(["list_projects", "get_chunk"]);
const AUTO_REGISTERING_TOOLS = new Set([
  "project_status",
  "index_history",
  "inspect_scan",
  "index_storage_status",
  "search_code",
  "search_across_projects",
  "find_symbol",
  "find_references",
  "analyze_refactor",
  "file_outline",
]);
const WRITE_TOOLS = new Set(["init_project", "index_project", "remove_project"]);

describe("MCP server", () => {
  test("registers the focused tool suite", async () => {
    const tools = await createServer(tinyApp(), { autoIndex: false }).listTools();
    expect(new Set(tools.map((tool) => tool.name))).toEqual(
      new Set([
        "init_project",
        "index_project",
        "project_status",
        "index_history",
        "inspect_scan",
        "index_storage_status",
        "index_storage_maintenance",
        "list_projects",
        "remove_project",
        "search_code",
        "search_across_projects",
        "find_symbol",
        "find_references",
        "analyze_refactor",
        "file_outline",
        "get_chunk",
      ]),
    );
    expect(tools).toHaveLength(16);
    expect(tools.every((tool) => !("ctx" in (tool.inputSchema.properties ?? {})))).toBe(true);
  });

  test("every tool declares description, title, and annotations", async () => {
    const tools = await createServer(tinyApp(), { autoIndex: false }).listTools();
    expect(new Set(tools.map((tool) => tool.name))).toEqual(
      new Set([
        ...READ_ONLY_TOOLS,
        ...AUTO_REGISTERING_TOOLS,
        ...WRITE_TOOLS,
        "index_storage_maintenance",
      ]),
    );
    for (const tool of tools) {
      expect(tool.description && tool.description.length > 60).toBe(true);
      expect(tool.title).toBeTruthy();
      expect(tool.annotations?.openWorldHint).toBe(false);
    }
  });

  test("read and write tools are annotated distinctly", async () => {
    const tools = await createServer(tinyApp(), { autoIndex: false }).listTools();
    const annotations = Object.fromEntries(tools.map((tool) => [tool.name, tool.annotations]));
    for (const name of READ_ONLY_TOOLS) {
      expect(annotations[name]?.readOnlyHint).toBe(true);
      expect(annotations[name]?.destructiveHint).toBe(false);
    }
    for (const name of AUTO_REGISTERING_TOOLS) {
      expect(annotations[name]?.readOnlyHint).toBe(false);
      expect(annotations[name]?.destructiveHint).toBe(false);
      expect(annotations[name]?.idempotentHint).toBe(true);
    }
    expect(annotations.remove_project?.destructiveHint).toBe(true);
    expect(annotations.remove_project?.idempotentHint).toBe(true);
    expect(annotations.index_project?.destructiveHint).toBe(false);
    expect(annotations.index_project?.idempotentHint).toBe(true);
    expect(annotations.init_project?.destructiveHint).toBe(true);
    expect(annotations.init_project?.idempotentHint).toBe(false);
  });

  test("every tool parameter is documented and bounded", async () => {
    const tools = Object.fromEntries(
      (await createServer(tinyApp(), { autoIndex: false }).listTools()).map((tool) => [
        tool.name,
        tool,
      ]),
    );
    for (const [name, tool] of Object.entries(tools)) {
      for (const [parameter, spec] of Object.entries(tool.inputSchema.properties ?? {})) {
        expect("description" in (spec as object), `${name}.${parameter}`).toBe(true);
      }
    }
    const searchCode = tools.search_code;
    const limit = searchCode?.inputSchema.properties?.limit as { minimum: number; maximum: number };
    expect([limit.minimum, limit.maximum]).toEqual([1, 50]);
    const cross = tools.search_across_projects?.inputSchema;
    expect(new Set(Object.keys(cross?.properties ?? {}))).toEqual(
      new Set(["query", "projects", "languages", "paths", "kinds", "limit"]),
    );
    expect(new Set(cross?.required ?? [])).toEqual(new Set(["query", "projects"]));
    const projectsSpec = cross?.properties?.projects as { minItems: number };
    expect(projectsSpec.minItems).toBe(2);
    const matchSpec = tools.find_symbol?.inputSchema.properties?.match as { enum: string[] };
    expect(matchSpec.enum).toEqual(["exact", "prefix", "contains"]);
  });

  test("analyze_refactor description credits signature-change evidence", async () => {
    const tools = Object.fromEntries(
      (await createServer(tinyApp(), { autoIndex: false }).listTools()).map((tool) => [
        tool.name,
        tool,
      ]),
    );
    expect(tools.analyze_refactor?.description?.toLowerCase()).toContain("call sites");
  });

  test("search_across_projects schema rejects one project", async () => {
    const server = createServer(tinyApp(), { autoIndex: false });
    expect(
      server.callTool("search_across_projects", { query: "answer", projects: ["only-one"] }),
    ).rejects.toThrow();
  });

  test("tool error carries code and details", async () => {
    const server = createServer(tinyApp(), { autoIndex: false });
    expect(server.callTool("get_chunk", { chunk_id: "missing" })).rejects.toThrow(
      /CHUNK_NOT_FOUND/,
    );
  });

  test("default server defers indexing until the first code query", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    fs.writeFileSync(path.join(root, "main.py"), "def locate_feature():\n    return True\n");
    const app = tinyApp();
    const server = createServer(app);
    server.listRoots = async () => [root];
    server.startCoordinator();
    await server.callTool("project_status", {});
    expect((await app.listProjects())[0]?.id).toBeDefined();
    const before = await app.projectStatus(undefined, { roots: [root] });
    expect(before.file_count).toBe(0);
    await server.callTool("find_symbol", { name: "locate_feature" });
    const after = await app.projectStatus(undefined, { roots: [root] });
    expect(after.file_count).toBeGreaterThan(0);
    server.close();
  });

  test("project_status registers an unmarked root", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const app = tinyApp();
    const server = createServer(app);
    server.startCoordinator();
    server.listRoots = async () => [root];
    expect(fs.existsSync(path.join(root, ".ci-mcp"))).toBe(false);
    await server.callTool("project_status", {});
    expect(fs.existsSync(path.join(root, ".ci-mcp", "project.toml"))).toBe(true);
    expect((await app.listProjects()).map((project) => project.root)).toEqual([path.resolve(root)]);
    server.close();
  });

  test("init_project rejects overlap unless allow_overlap is set", async () => {
    const root = path.join(temporary, "repo");
    const nested = path.join(root, "src");
    fs.mkdirSync(nested, { recursive: true });
    const app = tinyApp();
    const server = createServer(app, { autoIndex: false });
    server.listRoots = async () => [root];
    await server.callTool("init_project", { path: root });
    expect(server.callTool("init_project", { path: nested })).rejects.toThrow(
      /OVERLAPPING_PROJECT/,
    );
    await server.callTool("init_project", { path: nested, allow_overlap: true });
    expect(await app.listProjects()).toHaveLength(2);
  });

  test("lazy query refreshes a modified source", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    const source = path.join(root, "main.py");
    fs.writeFileSync(source, "def before_change():\n    return 1\n");
    const app = tinyApp();
    const server = createServer(app);
    server.listRoots = async () => [root];
    server.startCoordinator();
    try {
      await server.callTool("find_symbol", { name: "before_change" });
      fs.writeFileSync(source, "def after_change():\n    return 2\n");
      await server.callTool("find_symbol", { name: "after_change" });
      const project = (await app.listProjects())[0] as { id: string };
      expect((await app.findSymbol("after_change", project.id)).hits.length).toBeGreaterThan(0);
      expect((await app.findSymbol("before_change", project.id)).hits).toHaveLength(0);
    } finally {
      server.close();
    }
  });

  test("lazy query refreshes created and deleted sources", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    const removed = path.join(root, "removed.py");
    fs.writeFileSync(removed, "def removed_symbol():\n    return True\n");
    const app = tinyApp();
    const server = createServer(app);
    server.listRoots = async () => [root];
    server.startCoordinator();
    try {
      await server.callTool("find_symbol", { name: "removed_symbol" });
      fs.unlinkSync(removed);
      fs.writeFileSync(path.join(root, "added.py"), "def added_symbol():\n    return True\n");
      await server.callTool("find_symbol", { name: "added_symbol" });
      const project = (await app.listProjects())[0] as { id: string };
      expect((await app.findSymbol("added_symbol", project.id)).hits.length).toBeGreaterThan(0);
      expect((await app.findSymbol("removed_symbol", project.id)).hits).toHaveLength(0);
    } finally {
      server.close();
    }
  });

  test("lazy query refreshes an explicit project outside the active roots", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    const source = path.join(root, "main.py");
    fs.writeFileSync(source, "def before_change():\n    return 1\n");
    const app = tinyApp();
    const project = await app.initProject(root);
    await app.indexProject(project.id);
    const server = createServer(app);
    server.listRoots = async () => [];
    server.startCoordinator();
    fs.writeFileSync(source, "def after_change():\n    return 2\n");
    try {
      await server.callTool("find_symbol", { name: "after_change", project: project.id });
      expect((await app.findSymbol("after_change", project.id)).hits.length).toBeGreaterThan(0);
      expect((await app.findSymbol("before_change", project.id)).hits).toHaveLength(0);
    } finally {
      server.close();
    }
  });

  test("manual mode does not refresh a changed source", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    const source = path.join(root, "main.py");
    fs.writeFileSync(source, "def before_change():\n    return 1\n");
    const app = tinyApp();
    const project = await app.initProject(root);
    await app.indexProject(project.id);
    const server = createServer(app, { autoIndex: false });
    server.listRoots = async () => [root];
    server.startCoordinator();
    fs.writeFileSync(source, "def after_change():\n    return 2\n");
    try {
      await server.callTool("find_symbol", { name: "after_change" });
      expect((await app.findSymbol("after_change", project.id)).hits).toHaveLength(0);
      expect((await app.findSymbol("before_change", project.id)).hits.length).toBeGreaterThan(0);
    } finally {
      server.close();
    }
  });

  test("search_across_projects returns filtered globally limited hits", async () => {
    const app = tinyApp();
    const roots = [path.join(temporary, "alpha"), path.join(temporary, "beta")];
    for (const root of roots) fs.mkdirSync(path.join(root, "src"), { recursive: true });
    fs.writeFileSync(
      path.join(roots[0] as string, "src", "feature.py"),
      "def shared_feature_alpha():\n    return 'alpha'\n",
    );
    fs.writeFileSync(
      path.join(roots[1] as string, "src", "feature.ts"),
      "export function sharedFeatureBeta() { return 'beta'; }\n",
    );
    const alpha = await app.initProject(roots[0], { name: "alpha-service" });
    const beta = await app.initProject(roots[1], { name: "beta-service" });
    await app.indexProject(alpha.id);
    await app.indexProject(beta.id);
    const server = createServer(app, { autoIndex: false });
    server.listRoots = async () => [];
    const filters = {
      languages: ["python", "typescript"],
      paths: ["src/*"],
      kinds: ["function"],
    };
    const hits = (
      (await server.callTool("search_across_projects", {
        query: "shared feature",
        projects: [alpha.id, roots[1]],
        ...filters,
        limit: 2,
      })) as { hits: Array<Record<string, unknown>> }
    ).hits;
    expect(hits).toHaveLength(2);
    expect(new Set(hits.map((hit) => hit.project_id))).toEqual(new Set([alpha.id, beta.id]));
    expect(new Set(hits.map((hit) => hit.project_name))).toEqual(new Set([alpha.name, beta.name]));
    for (const hit of hits) {
      expect(String(hit.path).startsWith("src/")).toBe(true);
      expect(hit.kind).toBe("function");
    }
    const limited = (
      (await server.callTool("search_across_projects", {
        query: "shared feature",
        projects: [alpha.name, beta.id],
        ...filters,
        limit: 1,
      })) as { hits: unknown[] }
    ).hits;
    expect(limited).toHaveLength(1);
    const pythonOnly = (
      (await server.callTool("search_across_projects", {
        query: "shared feature",
        projects: [alpha.id, beta.name],
        languages: ["python"],
      })) as { hits: Array<Record<string, unknown>> }
    ).hits;
    expect(new Set(pythonOnly.map((hit) => hit.project_id))).toEqual(new Set([alpha.id]));
  });

  test("search_across_projects rejects duplicate project aliases", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const app = tinyApp();
    const project = await app.initProject(root, { name: "service" });
    const server = createServer(app, { autoIndex: false });
    server.listRoots = async () => [];
    await expect(
      server.callTool("search_across_projects", {
        query: "answer",
        projects: [project.id, project.name],
      }),
    ).rejects.toThrow(/INVALID_FILTER/);
  });

  test("search_across_projects preserves the missing-selector error", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    const app = tinyApp();
    const project = await app.initProject(root, { name: "service" });
    const server = createServer(app, { autoIndex: false });
    server.listRoots = async () => [];
    await expect(
      server.callTool("search_across_projects", {
        query: "answer",
        projects: [project.id, "missing-project"],
      }),
    ).rejects.toThrow(/PROJECT_NOT_FOUND/);
  });

  test("search_across_projects preserves the ambiguous-selector error", async () => {
    const app = tinyApp();
    const roots = ["one", "two", "three"].map((name) => path.join(temporary, name));
    for (const root of roots) fs.mkdirSync(root);
    await app.initProject(roots[0] as string, { name: "shared-service" });
    await app.initProject(roots[1] as string, { name: "shared-service" });
    const unique = await app.initProject(roots[2] as string, { name: "unique-service" });
    const server = createServer(app, { autoIndex: false });
    server.listRoots = async () => [];
    await expect(
      server.callTool("search_across_projects", {
        query: "answer",
        projects: ["shared-service", unique.id],
      }),
    ).rejects.toThrow(/AMBIGUOUS_PROJECT/);
  });

  test("lazy search_across_projects refreshes projects outside the active roots", async () => {
    const app = tinyApp();
    const projects = [];
    for (const name of ["one", "two"]) {
      const root = path.join(temporary, name);
      fs.mkdirSync(root);
      fs.writeFileSync(path.join(root, "main.py"), `def before_change_${name}():\n    return 1\n`);
      const project = await app.initProject(root, { name: `service-${name}` });
      await app.indexProject(project.id);
      projects.push(project);
      fs.writeFileSync(path.join(root, "main.py"), `def after_change_${name}():\n    return 2\n`);
    }
    const server = createServer(app);
    server.listRoots = async () => [];
    server.startCoordinator();
    try {
      await server.callTool("search_across_projects", {
        query: "after change",
        projects: projects.map((project) => project.id),
      });
      for (const [index, project] of projects.entries()) {
        const name = index === 0 ? "one" : "two";
        expect(
          (await app.findSymbol(`after_change_${name}`, project.id)).hits.length,
        ).toBeGreaterThan(0);
        expect((await app.findSymbol(`before_change_${name}`, project.id)).hits).toHaveLength(0);
      }
    } finally {
      server.close();
    }
  });

  test("manual search_across_projects does not refresh changed sources", async () => {
    const app = tinyApp();
    const projects = [];
    for (const name of ["one", "two"]) {
      const root = path.join(temporary, name);
      fs.mkdirSync(root);
      fs.writeFileSync(path.join(root, "main.py"), `def before_change_${name}():\n    return 1\n`);
      const project = await app.initProject(root, { name: `service-${name}` });
      await app.indexProject(project.id);
      projects.push(project);
      fs.writeFileSync(path.join(root, "main.py"), `def after_change_${name}():\n    return 2\n`);
    }
    const server = createServer(app, { autoIndex: false });
    server.listRoots = async () => [];
    server.startCoordinator();
    try {
      await server.callTool("search_across_projects", {
        query: "after change",
        projects: projects.map((project) => project.id),
      });
      for (const [index, project] of projects.entries()) {
        const name = index === 0 ? "one" : "two";
        expect((await app.findSymbol(`after_change_${name}`, project.id)).hits).toHaveLength(0);
        expect(
          (await app.findSymbol(`before_change_${name}`, project.id)).hits.length,
        ).toBeGreaterThan(0);
      }
    } finally {
      server.close();
    }
  });

  test("eager monitor refreshes created and deleted sources", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    const removed = path.join(root, "removed.py");
    fs.writeFileSync(removed, "def removed_symbol():\n    return True\n");
    const app = tinyApp();
    const firstChange = deferred();
    const secondChange = deferred();
    watchRoot.current = gatedWatch(firstChange.promise, secondChange.promise);
    const server = createServer(app, { autoIndex: true });
    server.listRoots = async () => [root];
    server.startCoordinator();
    try {
      await server.listTools();
      await server.callTool("find_symbol", { name: "removed_symbol" });
      const project = (await app.listProjects())[0] as { id: string };
      fs.unlinkSync(removed);
      fs.writeFileSync(path.join(root, "added.py"), "def added_symbol():\n    return True\n");
      firstChange.resolve();
      await waitUntil(
        async () =>
          (await app.findSymbol("added_symbol", project.id)).hits.length > 0 &&
          (await app.findSymbol("removed_symbol", project.id)).hits.length === 0,
      );
    } finally {
      secondChange.resolve();
      server.close();
    }
  });

  test("eager monitor repeats when a source changes during the refresh", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    const source = path.join(root, "main.py");
    fs.writeFileSync(source, "def initial_symbol():\n    return 0\n");
    const embedder = new SwitchableBlockingEmbedder();
    const app = new Application(paths(), { embedder, cwd: temporary });
    const firstChange = deferred();
    const secondChange = deferred();
    watchRoot.current = gatedWatch(firstChange.promise, secondChange.promise);
    const server = createServer(app, { autoIndex: true });
    server.listRoots = async () => [root];
    server.startCoordinator();
    try {
      await server.listTools();
      await server.callTool("find_symbol", { name: "initial_symbol" });
      const project = (await app.listProjects())[0] as { id: string };
      embedder.block = true;
      writeWithLaterMtime(source, "def first_change():\n    return 1\n");
      expect(await app.projectIsStale(project.id)).toBe(true);
      firstChange.resolve();
      await waitUntil(() => embedder.started);
      // The change lands while the first refresh is still embedding; the
      // event-driven second pass must reconcile it.
      writeWithLaterMtime(source, "def final_change():\n    return 2\n");
      secondChange.resolve();
      embedder.letGo();
      await waitUntil(
        async () => (await app.findSymbol("final_change", project.id)).hits.length > 0,
      );
      expect((await app.findSymbol("first_change", project.id)).hits).toHaveLength(0);
      expect((await app.findSymbol("initial_symbol", project.id)).hits).toHaveLength(0);
    } finally {
      embedder.letGo();
      server.close();
    }
  });

  test("eager reconciliation detects git-exclusion changes without an event", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    const { execFileSync } = await import("node:child_process");
    execFileSync("git", ["init", "-q", root]);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    fs.writeFileSync(path.join(root, "local_only.py"), "def local_symbol():\n    return True\n");
    const app = tinyApp();
    const never = deferred();
    watchRoot.current = gatedWatch(never.promise);
    EAGER_RECONCILE_SECONDS.value = 0.05;
    const server = createServer(app, { autoIndex: true });
    server.listRoots = async () => [root];
    server.startCoordinator();
    try {
      await server.listTools();
      await server.callTool("find_symbol", { name: "local_symbol" });
      const project = (await app.listProjects())[0] as { id: string };
      fs.mkdirSync(path.join(root, ".git", "info"), { recursive: true });
      fs.writeFileSync(path.join(root, ".git", "info", "exclude"), "local_only.py\n");
      await waitUntil(
        async () => (await app.findSymbol("local_symbol", project.id)).hits.length === 0,
      );
    } finally {
      never.resolve();
      server.close();
    }
  }, 15_000);

  test("eager monitor restarts after a watcher failure", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    const source = path.join(root, "main.py");
    fs.writeFileSync(source, "def before_failure():\n    return 1\n");
    const app = tinyApp();
    let watchCalls = 0;
    const change = deferred();
    watchRoot.current = async function* () {
      watchCalls += 1;
      if (watchCalls === 1) throw new Error("simulated watcher failure");
      await change.promise;
      yield;
      await new Promise(() => undefined);
    };
    WATCH_RETRY_INITIAL_SECONDS.value = 0.01;
    WATCH_RETRY_MAXIMUM_SECONDS.value = 0.02;
    const server = createServer(app, { autoIndex: true });
    server.listRoots = async () => [root];
    server.startCoordinator();
    try {
      await server.listTools();
      await server.callTool("find_symbol", { name: "before_failure" });
      await waitUntil(() => watchCalls >= 2);
      const project = (await app.listProjects())[0] as { id: string };
      fs.writeFileSync(source, "def after_recovery():\n    return 2\n");
      change.resolve();
      await waitUntil(
        async () => (await app.findSymbol("after_recovery", project.id)).hits.length > 0,
      );
      expect((await app.findSymbol("before_failure", project.id)).hits).toHaveLength(0);
    } finally {
      server.close();
    }
  }, 15_000);

  test("first automatic index materializes the project tree once", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const app = tinyApp();
    const scanner = app.indexer.scanner;
    const original = scanner.iterScan.bind(scanner);
    let scans = 0;
    scanner.iterScan = async function* (...args: Parameters<typeof original>) {
      scans += 1;
      yield* original(...args);
    };
    const server = createServer(app);
    server.listRoots = async () => [root];
    server.startCoordinator();
    try {
      await server.listTools();
      await server.callTool("search_code", { query: "answer" });
      expect(scans).toBe(1);
    } finally {
      server.close();
    }
  });

  test("startup index is not dropped when the server closes mid-run", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const embedder = new BlockingEmbedder();
    const app = new Application(paths(), { embedder, cwd: temporary });
    const server = createServer(app, { autoIndex: true });
    server.listRoots = async () => [root];
    server.startCoordinator();
    await server.listTools();
    await waitUntil(() => embedder.started);
    server.close();
    embedder.letGo();
    await waitUntil(
      async () => (await app.projectStatus(undefined, { roots: [root] })).state === "ready",
      10,
    );
  }, 15_000);

  test("closing the server cancels a startup job waiting for the index lock", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const app = tinyApp();
    const project = await app.initProject(root);
    let attempts = 0;
    const original = app.indexProject.bind(app);
    app.indexProject = async (...args: Parameters<typeof original>) => {
      attempts += 1;
      return original(...args);
    };
    const server = createServer(app, { autoIndex: true });
    server.listRoots = async () => [root];
    server.startCoordinator();
    const release = await acquireLock(
      path.join(app.paths.data, "locks", "index-global.lock"),
      true,
    );
    try {
      await server.listTools();
      await waitUntil(() => attempts > 0);
    } finally {
      server.close();
      await release();
      await settle(server);
    }
    for (let index = 0; index < 50; index += 1) {
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    expect((await app.projectStatus(project.id)).state).toBe("pending");
  }, 15_000);

  test("code query waits for the startup index to finish", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "package.json"), '{"name": "project"}\n');
    fs.writeFileSync(path.join(root, "main.js"), "export function answer() { return 42; }\n");
    const embedder = new BlockingEmbedder();
    const app = new Application(paths(), { embedder, cwd: temporary });
    const server = createServer(app, { autoIndex: true });
    server.listRoots = async () => [root];
    server.startCoordinator();
    try {
      await server.listTools();
      await waitUntil(() => embedder.started);
      let settled = false;
      const query = server.callTool("search_code", { query: "answer" }).then((value) => {
        settled = true;
        return value;
      });
      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(settled).toBe(false);
      embedder.letGo();
      expect(await query).toBeTruthy();
    } finally {
      embedder.letGo();
      server.close();
    }
  }, 15_000);

  test("explicit code query ignores an unrelated startup index", async () => {
    const readyRoot = path.join(temporary, "ready");
    fs.mkdirSync(readyRoot);
    fs.writeFileSync(path.join(readyRoot, "main.py"), "def answer():\n    return 42\n");
    const startupRoot = path.join(temporary, "startup");
    fs.mkdirSync(startupRoot);
    fs.writeFileSync(path.join(startupRoot, "pyproject.toml"), "[project]\nname = 'startup'\n");
    fs.writeFileSync(path.join(startupRoot, "slow.py"), "def slow():\n    return True\n");
    const embedder = new SwitchableBlockingEmbedder();
    const app = new Application(paths(), { embedder, cwd: temporary });
    const readyProject = await app.initProject(readyRoot);
    await app.indexProject(readyProject.id);
    embedder.block = true;
    const server = createServer(app, { autoIndex: true });
    server.listRoots = async () => [startupRoot];
    server.startCoordinator();
    try {
      await server.listTools();
      await waitUntil(() => embedder.started);
      const result = await Promise.race([
        server.callTool("search_code", { query: "answer", projects: [readyProject.id] }),
        new Promise((_resolve, reject) => setTimeout(() => reject(new Error("timed out")), 500)),
      ]);
      expect(result).toBeTruthy();
    } finally {
      embedder.letGo();
      server.close();
    }
  }, 15_000);

  test("startup maintenance defers to startup indexing", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const embedder = new BlockingEmbedder();
    const app = new Application(paths(), { embedder, cwd: temporary });
    const server = createServer(app, { autoIndex: true });
    server.listRoots = async () => [root];
    server.startCoordinator();
    const timestampPath = path.join(app.paths.data, "maintenance.json");
    const maintenance = server.runStartupMaintenance();
    try {
      await server.listTools();
      await waitUntil(() => embedder.started);
      await new Promise((resolve) => setTimeout(resolve, 300));
      expect(fs.existsSync(timestampPath)).toBe(false);
      embedder.letGo();
      await maintenance;
      expect(fs.existsSync(timestampPath)).toBe(true);
    } finally {
      embedder.letGo();
      server.close();
    }
  }, 15_000);

  test("lazy server runs startup maintenance before root scheduling", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const app = tinyApp();
    const project = await app.initProject(root);
    await app.indexProject(project.id);
    const server = createServer(app);
    server.listRoots = async () => [root];
    server.startCoordinator();
    try {
      await server.runStartupMaintenance();
      expect(fs.existsSync(path.join(app.paths.data, "maintenance.json"))).toBe(true);
    } finally {
      server.close();
    }
  });

  test("explicit code query ignores an unrelated startup failure", async () => {
    const readyRoot = path.join(temporary, "ready");
    fs.mkdirSync(readyRoot);
    fs.writeFileSync(path.join(readyRoot, "main.py"), "def answer():\n    return 42\n");
    const failingRoot = path.join(temporary, "failing");
    fs.mkdirSync(failingRoot);
    fs.writeFileSync(path.join(failingRoot, "pyproject.toml"), "[project]\nname = 'failing'\n");
    fs.writeFileSync(path.join(failingRoot, "broken.py"), "def broken():\n    return True\n");
    const runtimePaths = paths();
    const setupApp = new Application(runtimePaths, {
      embedder: new TinyEmbedder(),
      cwd: temporary,
    });
    const readyProject = await setupApp.initProject(readyRoot);
    await setupApp.indexProject(readyProject.id);
    const embedder = new FailingEmbedder();
    const app = new Application(runtimePaths, { embedder, cwd: temporary });
    const server = createServer(app, { autoIndex: true });
    server.listRoots = async () => [failingRoot];
    server.startCoordinator();
    try {
      await server.listTools();
      await waitUntil(() => embedder.calls >= 1);
      const result = await server.callTool("search_code", {
        query: "answer",
        projects: [readyProject.id],
      });
      expect(result).toBeTruthy();
    } finally {
      server.close();
    }
  }, 15_000);

  test("auto index can be disabled", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    fs.writeFileSync(path.join(root, "main.py"), "value = 1\n");
    const app = tinyApp();
    setEnv("CODE_INDEXING_AUTO_INDEX", "0");
    const server = createServer(app);
    server.listRoots = async () => [root];
    server.startCoordinator();
    try {
      await server.listTools();
      await expect(server.callTool("search_code", { query: "value" })).rejects.toThrow(
        /PROJECT_NOT_FOUND/,
      );
      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(fs.existsSync(path.join(root, ".ci-mcp"))).toBe(false);
      expect(await app.listProjects()).toHaveLength(0);
    } finally {
      server.close();
    }
  });

  test("failed startup index is retried on the next tool call", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const embedder = new FlakyEmbedder();
    const app = new Application(paths(), { embedder, cwd: temporary });
    const server = createServer(app);
    server.listRoots = async () => [root];
    server.startCoordinator();
    try {
      await expect(server.callTool("search_code", { query: "answer" })).rejects.toThrow(
        /MODEL_UNAVAILABLE/,
      );
      await server.callTool("search_code", { query: "answer" });
    } finally {
      server.close();
    }
  }, 15_000);

  test("index_project tool recovers after a startup failure", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const embedder = new FlakyEmbedder();
    const app = new Application(paths(), { embedder, cwd: temporary });
    const server = createServer(app);
    server.listRoots = async () => [root];
    server.startCoordinator();
    try {
      await expect(server.callTool("search_code", { query: "answer" })).rejects.toThrow(
        /MODEL_UNAVAILABLE/,
      );
      // A coordinator with no memory of the failure resolves the project by
      // name and indexes it directly.
      const projectName = (await app.listProjects())[0]?.name as string;
      server.coordinator = null;
      server.listRoots = async () => [];
      server.startCoordinator();
      await server.callTool("index_project", { project: projectName });
    } finally {
      server.close();
    }
  }, 15_000);

  test("discovery is not blocked by concurrent indexing", async () => {
    const rootA = path.join(temporary, "project_a");
    fs.mkdirSync(rootA);
    fs.writeFileSync(path.join(rootA, "pyproject.toml"), "[project]\nname = 'project-a'\n");
    fs.writeFileSync(path.join(rootA, "main.py"), "def a():\n    return 1\n");
    const rootB = path.join(temporary, "project_b");
    fs.mkdirSync(rootB);
    fs.writeFileSync(path.join(rootB, "pyproject.toml"), "[project]\nname = 'project-b'\n");
    fs.writeFileSync(path.join(rootB, "main.py"), "def b():\n    return 2\n");
    const embedder = new BlockingEmbedder();
    const app = new Application(paths(), { embedder, cwd: temporary });
    const server = createServer(app, { autoIndex: true });
    server.listRoots = async () => [rootA, rootB];
    server.startCoordinator();
    try {
      await server.listTools();
      await waitUntil(() => embedder.started);
      // Both roots are discovered and registered even though one is stuck
      // indexing: discovery does not share the indexing slot.
      await waitUntil(
        async () =>
          fs.existsSync(path.join(rootA, ".ci-mcp", "project.toml")) &&
          fs.existsSync(path.join(rootB, ".ci-mcp", "project.toml")) &&
          (await app.listProjects()).length === 2,
      );
      const blocked =
        (await app.projectStatus(undefined, { roots: [rootA] })).state === "indexing"
          ? rootA
          : rootB;
      const other = blocked === rootA ? rootB : rootA;
      const result = await Promise.race([
        server.callTool("project_status", { project: path.basename(other) }),
        new Promise((_resolve, reject) => setTimeout(() => reject(new Error("timed out")), 5000)),
      ]);
      expect(result).toBeTruthy();
    } finally {
      embedder.letGo();
      server.close();
    }
  }, 20_000);

  test("first query reports progress while the initial index runs", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const embedder = new BlockingEmbedder();
    const app = new Application(paths(), { embedder, cwd: temporary });
    const server = createServer(app);
    server.listRoots = async () => [root];
    server.startCoordinator();
    const messages: Array<string | null | undefined> = [];
    try {
      const query = server.callTool(
        "search_code",
        { query: "answer" },
        {
          onProgress: (_progress, _total, message) => {
            messages.push(message);
          },
        },
      );
      await waitUntil(() => embedder.started);
      embedder.letGo();
      await query;
      expect(messages).toContain("Building the initial index");
    } finally {
      embedder.letGo();
      server.close();
    }
  }, 15_000);

  test("first query fails when a competing index holds the lock past the deadline", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const app = tinyApp();
    setEnv("CODE_INDEXING_INDEX_WAIT_SECONDS", "1");
    const server = createServer(app);
    server.listRoots = async () => [root];
    server.startCoordinator();
    const release = await acquireLock(
      path.join(app.paths.data, "locks", "index-global.lock"),
      true,
    );
    try {
      await expect(server.callTool("search_code", { query: "answer" })).rejects.toThrow(
        /INDEX_BUSY/,
      );
    } finally {
      await release();
      await settle(server);
      server.close();
    }
  }, 20_000);

  test("a second root gives up instead of queueing behind the first", async () => {
    const rootA = path.join(temporary, "project_a");
    fs.mkdirSync(rootA);
    fs.writeFileSync(path.join(rootA, "pyproject.toml"), "[project]\nname = 'project-a'\n");
    fs.writeFileSync(path.join(rootA, "main.py"), "def a():\n    return 1\n");
    const rootB = path.join(temporary, "project_b");
    fs.mkdirSync(rootB);
    fs.writeFileSync(path.join(rootB, "pyproject.toml"), "[project]\nname = 'project-b'\n");
    fs.writeFileSync(path.join(rootB, "main.py"), "def b():\n    return 2\n");
    const embedder = new BlockingEmbedder();
    const app = new Application(paths(), { embedder, cwd: temporary });
    setEnv("CODE_INDEXING_INDEX_WAIT_SECONDS", "0");
    const server = createServer(app, { autoIndex: true });
    server.listRoots = async () => [rootA, rootB];
    server.startCoordinator();
    try {
      await server.listTools();
      await waitUntil(() => embedder.started);
      await waitUntil(async () => (await app.listProjects()).length === 2);
      const blocked =
        (await app.projectStatus(undefined, { roots: [rootA] })).state === "indexing"
          ? rootA
          : rootB;
      const waiting = blocked === rootA ? rootB : rootA;
      const waitingId = (await app.projectStatus(undefined, { roots: [waiting] })).project.id;
      await expect(
        server.callTool("search_code", { query: "value", projects: [waitingId] }),
      ).rejects.toThrow(/INDEX_BUSY/);
    } finally {
      embedder.letGo();
      await settle(server);
      server.close();
    }
  }, 20_000);

  test("first query succeeds when the index lock frees before the deadline", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const app = tinyApp();
    setEnv("CODE_INDEXING_INDEX_WAIT_SECONDS", "60");
    const server = createServer(app);
    server.listRoots = async () => [root];
    server.startCoordinator();
    const release = await acquireLock(
      path.join(app.paths.data, "locks", "index-global.lock"),
      true,
    );
    try {
      const query = server.callTool("search_code", { query: "answer" });
      await new Promise((resolve) => setTimeout(resolve, 300));
      await release();
      await query;
      expect((await app.projectStatus(undefined, { roots: [root] })).state).toBe("ready");
    } finally {
      await settle(server);
      server.close();
    }
  }, 20_000);

  test("a ready project does not report indexing progress", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const app = tinyApp();
    const server = createServer(app);
    server.listRoots = async () => [root];
    server.startCoordinator();
    const messages: Array<string | null | undefined> = [];
    try {
      await server.callTool("search_code", { query: "answer" });
      await server.callTool(
        "search_code",
        { query: "answer" },
        {
          onProgress: (_p, _t, message) => {
            messages.push(message);
          },
        },
      );
      expect(messages).toHaveLength(0);
    } finally {
      server.close();
    }
  }, 15_000);

  test("server instructions guide index-first usage", () => {
    const app = tinyApp();
    const server = createServer(app);
    const instructions = server.instructions;
    for (const tool of [
      "search_code",
      "search_across_projects",
      "find_symbol",
      "find_references",
      "analyze_refactor",
      "file_outline",
      "get_chunk",
      "project_status",
      "index_project",
    ]) {
      expect(instructions.includes(tool), tool).toBe(true);
    }
  });

  test("project_status deduplicates case-insensitive client roots", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const alias = caseInsensitiveAlias(root);
    if (alias === null) return; // filesystem is case-sensitive; nothing to prove
    const app = tinyApp();
    const server = createServer(app);
    server.listRoots = async () => [root, alias];
    server.startCoordinator();
    try {
      await server.callTool("project_status", {});
      expect(await app.listProjects()).toHaveLength(1);
    } finally {
      server.close();
    }
  });

  test("index_project reports file counts while it runs", async () => {
    class SlowEmbedder implements Embedder {
      readonly modelId = "test/tiny";
      readonly dimension = 4;
      async embedPassages(texts: string[]): Promise<number[][]> {
        await new Promise((resolve) => setTimeout(resolve, 50));
        return tinyVectors(texts);
      }
      embedQuery(text: string): number[] {
        return [1, 0, 0, text.length];
      }
    }
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    for (let number = 0; number < 8; number += 1) {
      fs.writeFileSync(
        path.join(root, `module_${number}.py`),
        `def answer_${number}():\n    return ${number}\n`,
      );
    }
    PROGRESS_POLL_SECONDS.value = 0.02;
    const app = new Application(paths(), { embedder: new SlowEmbedder(), cwd: temporary });
    const server = createServer(app);
    server.listRoots = async () => [root];
    server.startCoordinator();
    const reports: Array<{ progress: number; message: string | null | undefined }> = [];
    try {
      await server.callTool(
        "index_project",
        { project: root },
        {
          onProgress: (progress, _total, message) => {
            reports.push({ progress, message });
          },
        },
      );
      expect(reports[0]?.message).toBe("Indexing project");
      expect(
        reports.slice(1, -1).some((report) => /files|candidates/.test(report.message ?? "")),
      ).toBe(true);
      const values = reports.map((report) => report.progress);
      expect(values).toEqual([...values].sort((left, right) => left - right));
      expect(String(reports[reports.length - 1]?.message)).toContain("chunks embedded");
    } finally {
      server.close();
    }
  }, 15_000);

  test("index_storage_status tool reports installation statistics", async () => {
    const { prepare } = await import("./storage-status-setup.ts");
    const { app, project, root, server } = await prepare(temporary);
    try {
      const scoped = (await server.callTool("index_storage_status", {
        project: root,
      })) as StorageStatusPayload;
      const installation = (await server.callTool(
        "index_storage_status",
        {},
      )) as StorageStatusPayload;
      for (const payload of [scoped, installation]) {
        expect(payload.schema_version).toBe(1);
        expect(payload.registry.name).toBe("projects");
        expect(payload.registry.row_count).toBe(1);
        expect(payload.registry.logical_bytes).toBeGreaterThan(0);
        expect(payload.physical_bytes_total).toBeGreaterThan(0);
        expect(payload.projects.map((entry) => entry.project.id)).toEqual([project.id]);
        const entry = payload.projects[0] as (typeof payload.projects)[number];
        expect(entry.consistent).toBe(true);
        expect(entry.partition_open_failed).toBe(false);
        expect(entry.partition_physical_bytes).toBeGreaterThan(0);
        expect(new Set(entry.tables.map((table) => table.name))).toEqual(
          new Set(["files", "chunks", "references"]),
        );
      }
      expect(app).toBeTruthy();
    } finally {
      server.close();
    }
  });

  test("index_storage_maintenance tool defaults to dry run", async () => {
    const { prepare } = await import("./storage-status-setup.ts");
    const { project, root, server } = await prepare(temporary);
    try {
      const scoped = (await server.callTool("index_storage_maintenance", {
        project: root,
      })) as MaintenancePayload;
      const installation = (await server.callTool(
        "index_storage_maintenance",
        {},
      )) as MaintenancePayload;
      for (const payload of [scoped, installation]) {
        expect(payload.schema_version).toBe(1);
        expect(payload.dry_run).toBe(true);
        expect(payload.trigger).toBe("manual");
        expect(payload.retention_hours).toBe(24);
        const entry = payload.projects[0] as (typeof payload.projects)[number];
        expect(entry.status).toBe("skipped");
        expect(entry.after).toBe(null);
        expect(entry.before.partition_physical_bytes).toBeGreaterThan(0);
      }
      expect(project).toBeTruthy();
    } finally {
      server.close();
    }
  });

  test("index_storage_maintenance tool can execute cleanup", async () => {
    const { prepare } = await import("./storage-status-setup.ts");
    const { app, project, root, server } = await prepare(temporary);
    try {
      const payload = (await server.callTool("index_storage_maintenance", {
        project: root,
        dry_run: false,
      })) as MaintenancePayload;
      expect(payload.dry_run).toBe(false);
      const entry = payload.projects[0] as (typeof payload.projects)[number];
      expect(entry.status).toBe("ok");
      expect(entry.after).not.toBe(null);
      expect(payload.registry_after).not.toBe(null);
      expect((await app.projectStatus(project.id)).state).toBe("ready");
    } finally {
      server.close();
    }
  });

  test("index_history tool reports paginated runs", async () => {
    const { prepare } = await import("./storage-status-setup.ts");
    const { project, root, server } = await prepare(temporary);
    try {
      const payload = (await server.callTool("index_history", {
        project: root,
        limit: 1,
      })) as HistoryPayload;
      expect(payload.schema_version).toBe(1);
      expect(payload.project.id).toBe(project.id);
      expect(payload.runs).toHaveLength(1);
      expect(payload.runs[0]?.state).toBe("completed");
      expect(payload.runs[0]?.trigger).toBe("manual");
      expect(payload.runs[0]?.run_id).toBeTruthy();
      expect(payload.runs[0]?.chunks_embedded).toBeGreaterThanOrEqual(1);
    } finally {
      server.close();
    }
  });

  test("project_status includes progress and the last run", async () => {
    const { prepare } = await import("./storage-status-setup.ts");
    const { project, root, server } = await prepare(temporary);
    try {
      const payload = (await server.callTool("project_status", {
        project: root,
      })) as ProjectStatusPayload;
      expect(payload.last_run.state).toBe("completed");
      expect(payload.last_run.trigger).toBe("manual");
      expect(payload.last_run.eligible_files).toBe(1);
      expect(payload.progress).toBe(null);
      expect(project).toBeTruthy();
    } finally {
      server.close();
    }
  });

  test("inspect_scan tool returns paginated filtered results", async () => {
    const { prepare } = await import("./storage-status-setup.ts");
    const { project, root, server } = await prepare(temporary);
    try {
      const first = (await server.callTool("inspect_scan", {
        project: root,
        limit: 1,
      })) as ScanPagePayload;
      expect(first.schema_version).toBe(1);
      expect(first.project.id).toBe(project.id);
      expect(first.items).toHaveLength(1);
      expect(first.next_cursor).toBeTruthy();
      const second = (await server.callTool("inspect_scan", {
        project: root,
        limit: 1,
        cursor: first.next_cursor as string,
      })) as ScanPagePayload;
      expect(second.items).toHaveLength(1);
      expect(second.items[0]?.path).not.toBe(first.items[0]?.path);
      const eligible = (await server.callTool("inspect_scan", {
        project: root,
        outcome: "eligible",
      })) as ScanPagePayload;
      expect(eligible.items.map((item) => item.path)).toEqual(["main.py"]);
      expect(eligible.items[0]?.outcome).toBe("eligible");
      expect(eligible.items[0]?.language).toBe("python");
      expect(eligible.items[0]?.size).not.toBe(null);
    } finally {
      server.close();
    }
  });

  test("eager watcher marks the root dirty without a freshness walk", async () => {
    const root = path.join(temporary, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
    const source = path.join(root, "main.py");
    fs.writeFileSync(source, "def before_change():\n    return 1\n");
    const app = tinyApp();
    let staleChecks = 0;
    const original = app.projectIsStale.bind(app);
    app.projectIsStale = async (...args: Parameters<typeof original>) => {
      staleChecks += 1;
      return original(...args);
    };
    const seedChecked = deferred();
    const countingOriginal = app.projectIsStale.bind(app);
    app.projectIsStale = async (...args: Parameters<typeof countingOriginal>) => {
      const result = await countingOriginal(...args);
      seedChecked.resolve();
      return result;
    };
    const change = deferred();
    watchRoot.current = gatedWatch(change.promise);
    EAGER_RECONCILE_SECONDS.value = 3600;
    const server = createServer(app, { autoIndex: true });
    server.listRoots = async () => [root];
    server.startCoordinator();
    try {
      await server.listTools();
      await server.callTool("find_symbol", { name: "before_change" });
      const project = (await app.listProjects())[0] as { id: string };
      await seedChecked.promise;
      const baseline = staleChecks;
      writeWithLaterMtime(source, "def after_change():\n    return 2\n");
      change.resolve();
      await waitUntil(
        async () => (await app.findSymbol("after_change", project.id)).hits.length > 0,
      );
      expect(staleChecks).toBe(baseline);
    } finally {
      server.close();
    }
  }, 15_000);
});
