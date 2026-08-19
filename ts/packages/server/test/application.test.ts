import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { Application } from "../src/application.ts";
import type { Embedder } from "../src/embedding.ts";
import { isCodeIndexingError } from "../src/errors.ts";
import { DeclarationSelector, isBackfillComplete } from "../src/models.ts";
import { existingMarkerPath } from "../src/projects.ts";
import { indexSettingsFromEnvironment } from "../src/settings.ts";
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

class OtherModelTinyEmbedder implements Embedder {
  readonly modelId = "test/other-tiny";
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
});

function app(cwd: string, embedder: Embedder = new TinyEmbedder()): Application {
  return new Application(
    { data: path.join(temporary, "data"), cache: path.join(temporary, "cache") },
    { embedder, cwd },
  );
}

describe("Application", () => {
  test("orchestrates the default project lifecycle", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def locate_feature():\n    return True\n");
    const application = app(root);
    const project = await application.initProject(root);
    expect((await application.projectStatus(undefined, { roots: [root] })).state).toBe("pending");
    const report = await application.indexProject(undefined, { roots: [root] });
    const status = await application.projectStatus(undefined, { roots: [root] });
    expect(status.state).toBe("ready");
    const search = await application.searchCode("locate feature", { roots: [root] });
    const removal = await application.removeProject(project.id);
    expect(report.project_id).toBe(project.id);
    expect(status.file_count).toBe(1);
    expect(status.chunk_count).toBeGreaterThanOrEqual(1);
    expect(search.hits[0]?.symbol).toBe("locate_feature");
    expect(removal.removed).toBe(true);
    expect(await application.listProjects()).toEqual([]);
    expect(fs.existsSync(path.join(root, ".ci-mcp", "project.toml"))).toBe(true);
  });

  test("can ensure the structural index without a semantic search", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const application = app(root);
    const project = await application.initProject(root);
    await application.indexProject(project.id);
    const report = await application.ensureReferenceIndex(project.id);
    expect(isBackfillComplete(report)).toBe(true);
    expect(report.files_current).toBe(1);
  });

  test("modified source marks an index stale", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const source = path.join(root, "main.py");
    fs.writeFileSync(source, "value = 1\n");
    const application = app(root);
    const project = await application.initProject(root);
    await application.indexProject(project.id);
    expect(await application.projectIsStale(project.id)).toBe(false);
    fs.writeFileSync(source, "value = 200\n");
    expect(await application.projectIsStale(project.id)).toBe(true);
    expect((await application.projectStatus(project.id)).state).toBe("stale");
  });

  test("created and deleted source mark an index stale", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const source = path.join(root, "main.py");
    fs.writeFileSync(source, "value = 1\n");
    const application = app(root);
    const project = await application.initProject(root);
    await application.indexProject(project.id);
    fs.writeFileSync(path.join(root, "added.py"), "added = True\n");
    expect(await application.projectIsStale(project.id)).toBe(true);
    fs.unlinkSync(path.join(root, "added.py"));
    fs.unlinkSync(source);
    expect(await application.projectIsStale(project.id)).toBe(true);
  });

  test("a rejected file does not make the project permanently stale", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    fs.writeFileSync(path.join(root, "garbage.py"), Buffer.from("def broken(\x00):\n    pass\n"));
    const application = app(root);
    const project = await application.initProject(root);
    await application.indexProject(project.id);
    expect(await application.projectIsStale(project.id)).toBe(false);
    expect(["ready", "partial"]).toContain((await application.projectStatus(project.id)).state);
  });

  test("init_project defaults to the single client root", async () => {
    const root = path.join(temporary, "client-root");
    fs.mkdirSync(root);
    const application = app(temporary);
    const project = await application.initProject(undefined, { roots: [root] });
    expect(project.root).toBe(path.resolve(root));
  });

  test("init_project rejects a root nested inside a registered project", async () => {
    const root = path.join(temporary, "repo");
    const nested = path.join(root, "src");
    fs.mkdirSync(nested, { recursive: true });
    const application = app(temporary);
    const parent = await application.initProject(root);
    try {
      await application.initProject(nested);
      throw new Error("expected overlap rejection");
    } catch (error) {
      expect(isCodeIndexingError(error) && error.code === "OVERLAPPING_PROJECT").toBe(true);
    }
    expect(await application.listProjects()).toEqual([parent]);
  });

  test("init_project rejecting an overlap writes no marker", async () => {
    const root = path.join(temporary, "repo");
    const nested = path.join(root, "src");
    fs.mkdirSync(nested, { recursive: true });
    const application = app(temporary);
    await application.initProject(root);
    await expect(application.initProject(nested)).rejects.toBeDefined();
    expect(existingMarkerPath(nested)).toBeNull();
    const child = await application.initProject(nested, { allowOverlap: true });
    expect(existingMarkerPath(nested)).not.toBeNull();
    expect(child.root).toBe(path.resolve(nested));
  });

  test("reinitializing the same root keeps one registration", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const application = app(temporary);
    const first = await application.initProject(root);
    const second = await application.initProject(root);
    expect(second.id).toBe(first.id);
    expect(await application.listProjects()).toHaveLength(1);
  });

  test("force_new_id replaces a registration", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const application = app(temporary);
    const first = await application.initProject(root);
    const second = await application.initProject(root, { forceNewId: true });
    expect(second.id).not.toBe(first.id);
  });

  test("discover_project requires marker and supported source", async () => {
    const empty = path.join(temporary, "empty");
    fs.mkdirSync(empty);
    const application = app(temporary);
    expect(await application.discoverProject(empty)).toBeNull();
    fs.writeFileSync(path.join(empty, "package.json"), "{}\n");
    fs.writeFileSync(path.join(empty, "main.py"), "x = 1\n");
    expect((await application.discoverProject(empty))?.root).toBe(path.resolve(empty));
  });

  test("explicit cross-project search", async () => {
    const left = path.join(temporary, "left");
    const right = path.join(temporary, "right");
    fs.mkdirSync(left);
    fs.mkdirSync(right);
    fs.writeFileSync(path.join(left, "a.py"), "def alpha():\n    return 1\n");
    fs.writeFileSync(path.join(right, "b.py"), "def beta():\n    return 2\n");
    const application = app(temporary);
    const first = await application.initProject(left);
    const second = await application.initProject(right);
    await application.indexProject(first.id);
    await application.indexProject(second.id);
    const hits = await application.searchCode("alpha", {
      projects: [first.id, second.id],
    });
    expect(hits.hits.some((hit) => hit.symbol === "alpha")).toBe(true);
  });

  test("query rebuilds an incompatible project before serving results", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def locate_feature():\n    return True\n");
    const first = app(root);
    const project = await first.initProject(root);
    await first.indexProject(project.id);
    const second = app(root, new OtherModelTinyEmbedder());
    const search = await second.searchCode("locate feature", { roots: [root] });
    expect(search.hits[0]?.symbol).toBe("locate_feature");
  });

  test("resolves a backend and reports it", async () => {
    const application = app(temporary);
    const status = await application.modelStatus();
    expect(status.resolved_accelerator).toBe("cpu");
    expect(status.probe_cache_state).toBe("not-applicable");
  });

  test("storage status reports registry, project, and totals", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 1\n");
    const application = app(root);
    const project = await application.initProject(root);
    await application.indexProject(project.id);
    const status = await application.storageStatus();
    expect(status.projects).toHaveLength(1);
    expect(status.registry.row_count).toBeGreaterThanOrEqual(1);
  });

  test("maintenance dry run mutates nothing", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 1\n");
    const application = app(root);
    const project = await application.initProject(root);
    await application.indexProject(project.id);
    const before = await application.storageStatus(project.id);
    const report = await application.maintainStorage(project.id, { dryRun: true });
    expect(report.dry_run).toBe(true);
    expect(report.projects[0]?.after).toBeNull();
    const after = await application.storageStatus(project.id);
    expect(after.registry.current_version).toBe(before.registry.current_version);
  });

  test("inspect_scan paginates and filters", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "a.py"), "x = 1\n");
    fs.writeFileSync(path.join(root, "b.py"), "y = 2\n");
    fs.writeFileSync(path.join(root, "notes.txt"), "skip\n");
    const application = app(root);
    await application.initProject(root);
    const first = await application.inspectScan(undefined, { limit: 1, outcome: "eligible" });
    expect(first.items).toHaveLength(1);
    expect(first.next_cursor).not.toBeNull();
    const second = await application.inspectScan(undefined, {
      limit: 1,
      outcome: "eligible",
      cursor: first.next_cursor,
    });
    expect(second.items).toHaveLength(1);
    expect(second.items[0]?.path).not.toBe(first.items[0]?.path);
  });

  test("inspect_scan rejects unknown filters", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const application = app(root);
    await application.initProject(root);
    try {
      await application.inspectScan(undefined, { outcome: "nope" });
      throw new Error("expected invalid filter");
    } catch (error) {
      expect(isCodeIndexingError(error) && error.code === "INVALID_FILTER").toBe(true);
    }
  });

  test("index history is paginated", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 1\n");
    const application = app(root);
    const project = await application.initProject(root);
    await application.indexProject(project.id);
    await application.indexProject(project.id, { force: true });
    const page = await application.indexHistory(project.id, { limit: 1 });
    expect(page.runs).toHaveLength(1);
    expect(page.next_cursor).not.toBeNull();
  });

  test("reference tools prepare selectors", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
    const application = app(root);
    const project = await application.initProject(root);
    await application.indexProject(project.id);
    const response = await application.findReferences(
      DeclarationSelector.parse({
        project: project.id,
        path: "main.py",
        qualified_symbol: "answer",
      }),
    );
    expect(response.selected.qualified_symbol).toBe("answer");
  });

  test("maybe_run_maintenance is disabled by configuration", async () => {
    const settings = {
      ...indexSettingsFromEnvironment(),
      autoMaintenance: false,
    };
    const application = new Application(
      { data: path.join(temporary, "data"), cache: path.join(temporary, "cache") },
      { embedder: new TinyEmbedder(), cwd: temporary, settings },
    );
    expect(await application.maybeRunMaintenance()).toBeNull();
  });
});
