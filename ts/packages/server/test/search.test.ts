import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import type { Embedder } from "../src/embedding.ts";
import { TreeSitterExtractor } from "../src/extractor.ts";
import { Indexer } from "../src/indexing.ts";
import { initializeProject } from "../src/projects.ts";
import { SourceScanner } from "../src/scanner.ts";
import { SearchService } from "../src/search.ts";
import { LanceStore } from "../src/storage.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

class SemanticEmbedder implements Embedder {
  readonly modelId = "test/semantic-code";
  readonly dimension = 4;

  static vector(text: string): number[] {
    const lowered = text.toLowerCase();
    if (lowered.includes("auth") || lowered.includes("permission")) return [1, 0, 0, 0];
    if (lowered.includes("invoice") || lowered.includes("billing")) return [0, 1, 0, 0];
    return [0, 0, 1, 0];
  }

  embedPassages(texts: string[]): number[][] {
    return texts.map((text) => SemanticEmbedder.vector(text));
  }

  embedQuery(text: string): number[] {
    return SemanticEmbedder.vector(text);
  }
}

let temporary: string;

beforeEach(() => {
  temporary = temporaryDirectory();
});

afterEach(() => {
  removeDirectory(temporary);
});

async function indexedProjects() {
  const embedder = new SemanticEmbedder();
  const store = new LanceStore(path.join(temporary, "data"), {
    vectorDimension: embedder.dimension,
  });
  const indexer = new Indexer({
    store,
    scanner: new SourceScanner(),
    extractor: new TreeSitterExtractor(),
    embedder,
    lockDirectory: path.join(temporary, "locks"),
  });
  const projectIds: string[] = [];
  const sources = {
    auth: "def enforce_permissions(user):\n    return user.is_admin\n",
    billing: "def create_invoice(order):\n    return order.total\n",
  };
  for (const [name, source] of Object.entries(sources)) {
    const root = path.join(temporary, name);
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, `${name}.py`), source);
    const project = initializeProject(root);
    projectIds.push(project.id);
    await indexer.index(project);
  }
  return { store, search: new SearchService(store, embedder), projectIds };
}

async function indexedTree(files: Record<string, string>) {
  const embedder = new SemanticEmbedder();
  const store = new LanceStore(path.join(temporary, "data"), {
    vectorDimension: embedder.dimension,
  });
  const indexer = new Indexer({
    store,
    scanner: new SourceScanner(),
    extractor: new TreeSitterExtractor(),
    embedder,
    lockDirectory: path.join(temporary, "locks"),
  });
  const root = path.join(temporary, "repo");
  fs.mkdirSync(root, { recursive: true });
  for (const [relative, content] of Object.entries(files)) {
    const target = path.join(root, relative);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, content);
  }
  const project = initializeProject(root);
  await indexer.index(project);
  return { search: new SearchService(store, embedder), project: project.id, store };
}

describe("SearchService", () => {
  test("hybrid search respects project scope and filters", async () => {
    const { search, projectIds } = await indexedProjects();
    const auth = await search.searchCode("where are permissions enforced", [projectIds[0] ?? ""]);
    const billing = await search.searchCode("create billing invoice", [projectIds[1] ?? ""], {
      languages: ["python"],
    });
    expect(auth.hits[0]?.symbol).toBe("enforce_permissions");
    expect(new Set(auth.hits.map((hit) => hit.project_id))).toEqual(
      new Set(projectIds[0] === undefined ? [] : [projectIds[0]]),
    );
    expect(billing.hits[0]?.symbol).toBe("create_invoice");
    expect(
      (await search.searchCode("invoice", [projectIds[0] ?? ""], { paths: ["billing/**"] })).hits,
    ).toEqual([]);
  });

  test("symbol lookup and outline use indexed metadata", async () => {
    const { search, projectIds } = await indexedProjects();
    const symbols = await search.findSymbol("enforce_permissions", projectIds[0] ?? "");
    expect(symbols.hits[0]?.symbol).toBe("enforce_permissions");
    const outline = await search.fileOutline("auth.py", projectIds[0] ?? "");
    expect(outline.items.some((item) => item.symbol === "enforce_permissions")).toBe(true);
  });

  test("search truncates snippet and get_chunk returns full content", async () => {
    const body = "x = 1\n".repeat(2000);
    const { search, project, store } = await indexedTree({
      "long.py": `def huge():\n    """doc"""\n${body}`,
    });
    const result = await search.searchCode("huge", [project], { limit: 20 });
    const hit = result.hits[0];
    expect(hit).toBeDefined();
    expect(hit?.snippet.length).toBeLessThanOrEqual(4000);
    const chunk = await search.getChunk(hit?.chunk_id ?? "");
    expect(chunk.content.includes("x = 1") || hit?.snippet.includes("huge")).toBe(true);
    const stored = await store.getChunk(hit?.chunk_id ?? "");
    expect(stored?.content).toBe(chunk.content);
  });

  test("path filter finds matches below the fetch window", async () => {
    const sources: Record<string, string> = {};
    for (let index = 0; index < 120; index += 1) {
      sources[`noise/m${index}.py`] =
        `def enforce_permissions_${index}(user):\n` +
        "    'permission permission permission check'\n" +
        "    return user.permission\n";
    }
    sources["rare/needle.py"] =
      "def audit_gate(user):\n    'permission'\n    return user.allowed\n";
    const { search, project } = await indexedTree(sources);
    const unfiltered = await search.searchCode("permission check", [project], { limit: 8 });
    const filtered = await search.searchCode("permission check", [project], {
      paths: ["rare/*"],
      limit: 8,
    });
    expect(new Set(unfiltered.hits.map((hit) => hit.path.split("/")[0]))).toEqual(
      new Set(["noise"]),
    );
    expect(filtered.hits.map((hit) => hit.path)).toEqual(["rare/needle.py"]);
  });

  test("untranslatable path pattern still filters after fetch", async () => {
    const { search, project } = await indexedTree({
      "src/a.py": "def alpha_two():\n    return 2\n",
      "tests/c.py": "def alpha_three():\n    return 3\n",
    });
    expect((await search.searchCode("alpha", [project], { paths: ["/src/a.py"] })).hits).toEqual(
      [],
    );
  });

  test("literal empty character class falls back without a regex error", async () => {
    const { search, project } = await indexedTree({
      "literal/a[].py": "def alpha_literal():\n    return 1\n",
    });
    expect(
      (await search.searchCode("alpha", [project], { paths: ["*[]*.py"] })).hits.map(
        (hit) => hit.path,
      ),
    ).toEqual(["literal/a[].py"]);
  });

  test("get_chunk excludes embedding and duplicated text", async () => {
    const { search, project } = await indexedTree({
      "main.py": "def answer():\n    return 42\n",
    });
    const hit = (await search.searchCode("answer", [project])).hits[0];
    const chunk = await search.getChunk(hit?.chunk_id ?? "");
    expect("vector" in chunk).toBe(false);
    expect("embedding_text" in chunk).toBe(false);
    expect("search_text" in chunk).toBe(false);
  });

  test("symbol match does not treat underscores as wildcards", async () => {
    const { search, project } = await indexedTree({
      "main.py": "def load_user():\n    return 1\n\ndef loadXuser():\n    return 2\n",
    });
    const hits = await search.findSymbol("load_user", project);
    expect(hits.hits.map((hit) => hit.symbol)).toEqual(["load_user"]);
  });
});
