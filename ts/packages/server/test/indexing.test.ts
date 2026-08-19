import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import {
  type Embedder,
  type EmbeddedSegment,
  embeddedSegment,
  packVector,
  type PassageCandidate,
  type SegmentingEmbedder,
  segmentPlan,
} from "../src/embedding.ts";
import { CodeIndexingError, isCodeIndexingError } from "../src/errors.ts";
import { TreeSitterExtractor } from "../src/extractor.ts";
import { HistoryStore } from "../src/history.ts";
import { Indexer, REFERENCE_SCHEMA_VERSION } from "../src/indexing.ts";
import { initializeProject } from "../src/projects.ts";
import { SourceScanner } from "../src/scanner.ts";
import { LanceStore } from "../src/storage.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

class RecordingEmbedder implements Embedder {
  readonly modelId = "test/code";
  readonly dimension = 4;
  passageBatches: string[][] = [];

  embedPassages(texts: string[]): number[][] {
    if (texts.some((text) => text.includes("RAISE_EMBEDDING"))) {
      throw new Error("embedding failed");
    }
    this.passageBatches.push(texts);
    return texts.map((text) => [text.length % 7, 1, 2, 3]);
  }

  embedQuery(text: string): number[] {
    return [text.length % 7, 1, 2, 3];
  }
}

class OtherModelEmbedder implements Embedder {
  readonly modelId = "test/other";
  readonly dimension = 4;

  embedPassages(texts: string[]): number[][] {
    return texts.map((text) => [text.length % 7, 1, 2, 3]);
  }

  embedQuery(text: string): number[] {
    return [text.length % 7, 1, 2, 3];
  }
}

class ResourceLimitEmbedder extends RecordingEmbedder {
  override embedPassages(): number[][] {
    throw new CodeIndexingError("INDEX_RESOURCE_LIMIT", "out of memory");
  }
}

class WindowingEmbedder extends RecordingEmbedder implements SegmentingEmbedder {
  readonly windowChars: number;

  constructor(windowChars = 8) {
    super();
    this.windowChars = windowChars;
  }

  planAndEmbed(candidates: readonly PassageCandidate[]): EmbeddedSegment[][] {
    return candidates.map((candidate) => {
      const segments: EmbeddedSegment[] = [];
      for (let start = 0; start < candidate.content.length; start += this.windowChars) {
        const end = Math.min(start + this.windowChars, candidate.content.length);
        segments.push(embeddedSegment(start, end, 1, packVector([end - start, 1, 2, 3])));
      }
      return segments;
    });
  }
}

let temporary: string;

beforeEach(() => {
  temporary = temporaryDirectory();
});

afterEach(() => {
  removeDirectory(temporary);
});

function writeRepo(name: string, files: Record<string, string>): string {
  const root = path.join(temporary, name);
  fs.mkdirSync(root, { recursive: true });
  for (const [relative, content] of Object.entries(files)) {
    const target = path.join(root, relative);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, content);
  }
  return root;
}

function makeIndexer(
  embedder: Embedder = new RecordingEmbedder(),
  {
    batchSize = 1,
    history,
    plan,
    factory,
  }: {
    batchSize?: number;
    history?: HistoryStore;
    plan?: ReturnType<typeof segmentPlan>;
    factory?: () => import("../src/indexing.ts").PassageSession;
  } = {},
) {
  const store = new LanceStore(path.join(temporary, "data"), {
    vectorDimension: embedder.dimension,
  });
  const indexer = new Indexer({
    store,
    scanner: new SourceScanner(),
    extractor: new TreeSitterExtractor(),
    embedder,
    lockDirectory: path.join(temporary, "locks"),
    batchSize,
    ...(history === undefined ? {} : { history }),
    ...(plan === undefined ? {} : { segmentPlan: plan }),
    ...(factory === undefined ? {} : { passageSessionFactory: factory }),
    stagingDirectory: path.join(temporary, "staging"),
  });
  return { indexer, store };
}

describe("Indexer", () => {
  test("skips unchanged and metadata-only files", async () => {
    const root = writeRepo("repo", { "main.py": "def answer():\n    return 42\n" });
    const project = initializeProject(root);
    const embedder = new RecordingEmbedder();
    const { indexer, store } = makeIndexer(embedder);
    await indexer.index(project);
    const first = embedder.passageBatches.length;

    await indexer.index(project);
    expect(embedder.passageBatches.length).toBe(first);

    const source = path.join(root, "main.py");
    const now = new Date();
    fs.utimesSync(source, now, now);
    await indexer.index(project);
    expect(embedder.passageBatches.length).toBe(first);
    expect((await store.listFiles(project.id))[0]?.size).toBe(fs.statSync(source).size);
  });

  test("stages references, declarations, and coverage", async () => {
    const root = writeRepo("repo", {
      "main.py": "from helper import answer\n\ndef run():\n    return answer()\n",
      "helper.py": "def answer():\n    return 42\n",
    });
    const project = initializeProject(root);
    const { indexer, store } = makeIndexer();
    await indexer.index(project);
    const rows = await store.listReferenceRecords(project.id, {});
    expect(rows.some((row) => row.record_kind === "reference")).toBe(true);
    expect(rows.some((row) => row.record_kind === "declaration")).toBe(true);
    expect(rows.some((row) => row.record_kind === "coverage")).toBe(true);
    expect(rows.every((row) => row.schema_version === REFERENCE_SCHEMA_VERSION)).toBe(true);
  });

  test("a file with no structural occurrences still gets coverage", async () => {
    const root = writeRepo("repo", { "empty.py": "VALUE = 1\n" });
    const project = initializeProject(root);
    const { indexer, store } = makeIndexer();
    await indexer.index(project);
    const rows = await store.listReferenceRecords(project.id, {});
    expect(rows.some((row) => row.record_kind === "coverage")).toBe(true);
  });

  test("replaces changed files and removes deleted files", async () => {
    const root = writeRepo("repo", {
      "keep.py": "def keep():\n    return 1\n",
      "gone.py": "def gone():\n    return 2\n",
    });
    const project = initializeProject(root);
    const { indexer, store } = makeIndexer();
    await indexer.index(project);
    fs.writeFileSync(path.join(root, "keep.py"), "def keep():\n    return 3\n");
    fs.unlinkSync(path.join(root, "gone.py"));
    await indexer.index(project);
    const files = await store.listFiles(project.id);
    expect(files.map((file) => file.path).sort()).toEqual(["keep.py"]);
    const chunks = await store.listChunks([project.id]);
    expect(chunks.every((chunk) => chunk.path !== "gone.py")).toBe(true);
  });

  test("reindexing changed content retires the previous chunk id", async () => {
    const root = writeRepo("repo", { "main.py": "def answer():\n    return 42\n" });
    const project = initializeProject(root);
    const { indexer, store } = makeIndexer();
    await indexer.index(project);
    const previous = (await store.listChunks([project.id]))[0];
    expect(previous).toBeDefined();
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 43\n");
    await indexer.index(project);
    const current = (await store.listChunks([project.id]))[0];
    expect(current?.chunk_id).not.toBe(previous?.chunk_id);
    expect(await store.getChunk(previous?.chunk_id ?? "")).toBeNull();
    expect((await store.getChunk(current?.chunk_id ?? ""))?.content.endsWith("43")).toBe(true);
  });

  test("failed changed file preserves the previous generation", async () => {
    const root = writeRepo("repo", { "main.py": "def answer():\n    return 42\n" });
    const project = initializeProject(root);
    const embedder = new RecordingEmbedder();
    const { indexer, store } = makeIndexer(embedder);
    await indexer.index(project);
    const original = (await store.listFiles(project.id))[0];
    const originalChunks = await store.listChunks([project.id]);
    fs.writeFileSync(path.join(root, "main.py"), "def RAISE_EMBEDDING():\n    return 43\n");
    const failed = await indexer.index(project);
    expect(failed.errors).toHaveLength(1);
    const record = (await store.listFiles(project.id))[0];
    expect(record?.has_errors).toBe(true);
    expect(record?.content_hash).toBe(original?.content_hash);
    expect(await store.listChunks([project.id])).toEqual(originalChunks);
  });

  test("reference backfill parses unchanged files without embedding", async () => {
    const root = writeRepo("repo", { "main.py": "def answer():\n    return 42\n" });
    const project = initializeProject(root);
    const embedder = new RecordingEmbedder();
    const { indexer, store } = makeIndexer(embedder);
    await indexer.index(project);
    const batches = embedder.passageBatches.length;
    const file = (await store.listFiles(project.id))[0];
    expect(file).toBeDefined();
    await store.replaceFilesFromArrow(project.id, {
      files: [],
      chunkBatches: [],
      referenceBatches: [{ fileIds: [file?.file_id ?? ""], rows: [] }],
    });
    const report = await indexer.backfillReferences(project);
    expect(report.files_backfilled).toBe(1);
    expect(embedder.passageBatches.length).toBe(batches);
    expect(
      (await store.listReferenceRecords(project.id, {})).some(
        (row) => row.record_kind === "coverage",
      ),
    ).toBe(true);
  });

  test("force reindexes a previously failed file", async () => {
    const root = writeRepo("repo", { "main.py": "def RAISE_EMBEDDING():\n    return 1\n" });
    const project = initializeProject(root);
    const embedder = new RecordingEmbedder();
    const { indexer, store } = makeIndexer(embedder);
    await indexer.index(project);
    fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 2\n");
    await indexer.index(project, { force: true });
    const record = (await store.listFiles(project.id))[0];
    expect(record?.has_errors).toBe(false);
    expect((await store.listChunks([project.id])).length).toBeGreaterThan(0);
  });

  test("a noop run cannot promote a partial project with stored errors", async () => {
    const root = writeRepo("repo", { "main.py": "def RAISE_EMBEDDING():\n    return 1\n" });
    const project = initializeProject(root);
    const { indexer, store } = makeIndexer();
    await indexer.index(project);
    expect(await store.projectState(project.id)).toBe("partial");
    await indexer.index(project);
    expect(await store.projectState(project.id)).toBe("partial");
  });

  test("resource limit aborts instead of poisoning the file record", async () => {
    const root = writeRepo("repo", { "main.py": "def answer():\n    return 42\n" });
    const project = initializeProject(root);
    const { indexer, store } = makeIndexer(new ResourceLimitEmbedder());
    try {
      await indexer.index(project);
      throw new Error("expected resource limit");
    } catch (error) {
      expect(isCodeIndexingError(error) && error.code === "INDEX_RESOURCE_LIMIT").toBe(true);
    }
    expect(await store.listFiles(project.id)).toEqual([]);
  });

  test("binary files are skipped", async () => {
    const root = writeRepo("repo", { "ok.py": "def answer():\n    return 1\n" });
    fs.writeFileSync(path.join(root, "bad.py"), Buffer.from("def broken(\x00):\n    pass\n"));
    const project = initializeProject(root);
    const { indexer, store } = makeIndexer();
    const report = await indexer.index(project);
    expect(report.skipped_files).toBeGreaterThanOrEqual(1);
    expect(report.skip_reasons.binary).toBe(1);
    const files = await store.listFiles(project.id);
    expect(files.some((file) => file.path === "bad.py" && file.has_errors)).toBe(true);
  });

  test("a token-dense chunk is split into several stored chunks", async () => {
    const root = writeRepo("repo", {
      "main.py": "def long_name():\n    return 'abcdefghijklmnopqrstuvwxyz'\n",
    });
    const project = initializeProject(root);
    const { indexer, store } = makeIndexer(new WindowingEmbedder(8), {
      plan: segmentPlan({ maxTokens: 4, maxItems: 4 }),
    });
    await indexer.index(project);
    const chunks = (await store.listChunks([project.id])).filter(
      (chunk) => chunk.kind === "function" || chunk.kind === "function_part",
    );
    expect(chunks.length).toBeGreaterThan(1);
  });

  test("records a completed run in the audit history", async () => {
    const root = writeRepo("repo", { "main.py": "def answer():\n    return 42\n" });
    const project = initializeProject(root);
    const history = new HistoryStore(path.join(temporary, "history"));
    const { indexer } = makeIndexer(new RecordingEmbedder(), { history });
    const report = await indexer.index(project);
    const page = history.listRuns(project.id);
    expect(page.runs[0]?.state).toBe("completed");
    expect(page.runs[0]?.run_id).toBe(report.run_id ?? "");
    expect(page.runs[0]?.eligible_files).toBeGreaterThan(0);
  });

  test("rebuilds a partition written by an incompatible model", async () => {
    const root = writeRepo("repo", { "main.py": "def answer():\n    return 42\n" });
    const project = initializeProject(root);
    const first = makeIndexer(new RecordingEmbedder());
    await first.indexer.index(project);
    const previous = (await first.store.listChunks([project.id]))[0];
    const second = makeIndexer(new OtherModelEmbedder());
    const report = await second.indexer.index(project);
    expect(report.trigger).toBe("schema-rebuild");
    const current = (await second.store.listChunks([project.id]))[0];
    expect(current).toBeDefined();
    expect(previous).toBeDefined();
  });

  test("two references over one range survive a reindex", async () => {
    const root = writeRepo("repo", {
      "main.py": "class Child(Parent):\n    pass\n",
    });
    const project = initializeProject(root);
    const { indexer, store } = makeIndexer();
    await indexer.index(project);
    const first = (await store.listReferenceRecords(project.id, {})).filter(
      (row) => row.record_kind === "reference",
    );
    await indexer.index(project, { force: true });
    const second = (await store.listReferenceRecords(project.id, {})).filter(
      (row) => row.record_kind === "reference",
    );
    expect(second.length).toBe(first.length);
    expect(second.length).toBeGreaterThan(1);
  });
});
