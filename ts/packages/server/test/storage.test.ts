import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { isCodeIndexingError } from "../src/errors.ts";
import {
  REFERENCE_SCHEMA_VERSION,
  type ReferenceRecord,
  type ReferenceStore,
} from "../src/reference-store.ts";
import {
  LanceStore,
  MAX_CACHED_PARTITIONS,
  overlapWarnings,
  overlappingRegistration,
  probeBatchedMergeSemantics,
  setBatchedMergeSemanticsOk,
  worktreeWarnings,
} from "../src/storage.ts";
import type { ProjectInfo, StoredChunk, StoredFile } from "../src/models.ts";
import { caseInsensitiveAlias, removeDirectory, temporaryDirectory } from "./helpers.ts";

let temporary: string;

beforeEach(() => {
  temporary = temporaryDirectory();
});

afterEach(() => {
  removeDirectory(temporary);
});

function project(id = "project-a", root = `/tmp/${id}`): ProjectInfo {
  return {
    version: 1,
    id,
    name: id,
    root,
    scan: { include: [], exclude: [], max_file_bytes: 1 },
  };
}

function file(projectId: string, fileId = "file-a"): StoredFile {
  return {
    file_id: fileId,
    project_id: projectId,
    path: "src/example.ts",
    language: "typescript",
    size: 42,
    mtime_ns: 1n,
    content_hash: "hash",
    has_errors: false,
    error: null,
    indexed_at: 1,
  };
}

function chunk(projectId: string, fileId = "file-a", suffix = "one"): StoredChunk {
  return {
    chunk_id: `${projectId}:${suffix}`,
    file_id: fileId,
    path: "src/example.ts",
    language: "typescript",
    kind: "function",
    symbol: "answer",
    qualified_symbol: "answer",
    parent_symbol: null,
    start_byte: 0,
    end_byte: 20,
    start_line: 1,
    end_line: 2,
    content: "export function answer() { return 42; }",
    identifier_terms: "answer example",
    content_hash: "hash",
    part_index: 0,
    vector: [1, 0],
  };
}

function reference(projectId: string, fileId = "file-a"): ReferenceRecord {
  return {
    reference_id: "ref-a",
    record_kind: "declaration",
    file_id: fileId,
    project_id: projectId,
    path: "src/example.ts",
    language: "typescript",
    kind: "function",
    source_qualified_symbol: "answer",
    written_name: "answer",
    target_name: "answer",
    module_path: null,
    imported_name: null,
    alias: null,
    receiver_text: null,
    start_byte: 0,
    end_byte: 6,
    start_line: 1,
    end_line: 1,
    shape_json: "{}",
    content_hash: "hash",
    schema_version: REFERENCE_SCHEMA_VERSION,
  };
}

async function store(): Promise<LanceStore> {
  const instance = new LanceStore(temporary, { vectorDimension: 2 });
  await instance.upsertProject(project(), { modelId: "model", state: "ready" });
  return instance;
}

describe("LanceStore", () => {
  test("stores Python-compatible partition rows and projected chunks", async () => {
    const instance = await store();
    await instance.replaceFile(file("project-a"), [chunk("project-a")], [reference("project-a")]);

    const contract: ReferenceStore = instance;
    expect(await contract.hasReferenceTable("project-a")).toBe(true);
    expect(await contract.getChunk("project-a:one")).toMatchObject({
      project_id: "project-a",
      chunk_id: "project-a:one",
    });
    expect(await instance.outlineChunks("project-a", "src/example.ts")).toHaveLength(1);
    expect(
      await contract.declarationShapes("project-a", "answer", {
        schemaVersion: REFERENCE_SCHEMA_VERSION,
      }),
    ).toHaveLength(1);
    expect(
      await contract.targetNameCandidates("project-a", "answer", { recordKind: "declaration" }),
    ).toHaveLength(1);
    await instance.close();
  });

  test("replaces each file's chunks and references without deleting siblings", async () => {
    const instance = await store();
    await instance.replaceFilesFromArrow("project-a", {
      files: [file("project-a", "a"), file("project-a", "b")],
      chunkBatches: [
        { fileIds: ["a"], rows: [chunk("project-a", "a", "a1")] },
        { fileIds: ["b"], rows: [chunk("project-a", "b", "b1")] },
      ],
      referenceBatches: [
        { fileIds: ["a"], rows: [{ ...reference("project-a", "a"), reference_id: "ra" }] },
        { fileIds: ["b"], rows: [{ ...reference("project-a", "b"), reference_id: "rb" }] },
      ],
    });
    await instance.replaceFilesFromArrow("project-a", {
      files: [file("project-a", "a")],
      chunkBatches: [{ fileIds: ["a"], rows: [chunk("project-a", "a", "a2")] }],
      referenceBatches: [{ fileIds: ["a"], rows: [] }],
    });

    expect((await instance.listChunks(["project-a"])).map((item) => item.chunk_id).sort()).toEqual([
      "project-a:a2",
      "project-a:b1",
    ]);
    expect(await instance.listReferenceRecords("project-a", {})).toMatchObject([
      { reference_id: "rb" },
    ]);
    await instance.close();
  });

  test("captures and restores all partition table versions", async () => {
    const instance = await store();
    await instance.replaceFile(file("project-a"), [chunk("project-a")], [reference("project-a")]);
    const versions = await instance.tableVersions("project-a");
    await instance.removeFile("project-a", "file-a");
    expect(await instance.getChunk("project-a:one")).toBeNull();
    expect(await instance.restoreVersions("project-a", versions)).toBe(true);
    expect(await instance.getChunk("project-a:one")).not.toBeNull();
    await instance.close();
  });

  test("creates search indexes and searches independently partitioned rows", async () => {
    const instance = await store();
    const identifierMatch = {
      ...chunk("project-a", "file-a", "identifier"),
      path: "src/allowed.ts",
      content: "export const unrelated = true;",
      identifier_terms: "identifierOnly",
      vector: [0, 1],
    };
    const vectorOnly = {
      ...chunk("project-a", "file-a", "vector"),
      path: "test/excluded.ts",
      content: "export const distractor = true;",
      identifier_terms: "distractor",
      vector: [1, 0],
    };
    await instance.replaceFile(
      file("project-a"),
      [identifierMatch, vectorOnly],
      [reference("project-a")],
    );
    await instance.ensureIndexes("project-a");

    const hits = await instance.hybridSearch("identifierOnly", [1, 0], ["project-a"], { limit: 5 });
    expect(hits[0]).toMatchObject({ chunk_id: "project-a:identifier", project_id: "project-a" });
    const filtered = await instance.hybridSearch("identifierOnly", [1, 0], ["project-a"], {
      condition: "path = 'src/allowed.ts'",
      limit: 5,
    });
    expect(filtered).toHaveLength(1);
    expect(filtered[0]).toMatchObject({ chunk_id: "project-a:identifier" });
    expect((await instance.storageStatus()).projects[0]?.tables).toHaveLength(3);
    await instance.close();
  });

  test("preserves an incompatible live generation and rejects active ID conflicts", async () => {
    const registeredRoot = path.join(temporary, "registered");
    const incomingRoot = path.join(temporary, "incoming");
    fs.mkdirSync(path.join(registeredRoot, ".ci-mcp"), { recursive: true });
    fs.writeFileSync(path.join(registeredRoot, ".ci-mcp", "project.toml"), "version = 1\n");
    fs.mkdirSync(incomingRoot, { recursive: true });

    const first = new LanceStore(temporary, { vectorDimension: 2 });
    await first.upsertProject(project("shared", registeredRoot), { modelId: "model-v1" });
    let conflict: unknown = null;
    try {
      await first.upsertProject(project("shared", incomingRoot), { modelId: "model-v1" });
    } catch (error) {
      conflict = error;
    }
    expect(isCodeIndexingError(conflict) && conflict.code).toBe("PROJECT_ID_CONFLICT");
    await first.close();

    const changed = new LanceStore(temporary, { vectorDimension: 3 });
    await changed.upsertProject(project("shared", path.join(registeredRoot, ".")), {
      modelId: "model-v2",
    });
    expect(await changed.projectState("shared")).toBe("rebuild_required");
    await changed.close();

    const original = new LanceStore(temporary, { vectorDimension: 2 });
    expect(await original.incompatibilityReason("shared", "model-v1")).toBeNull();
    expect((await original.listProjects())[0]?.root).toBe(registeredRoot);
    await original.close();
  });

  test("reads never materialize a partition", async () => {
    const instance = new LanceStore(temporary, { vectorDimension: 2 });
    await instance.upsertProject(project(), { modelId: "model", state: "pending" });
    expect(await instance.getChunk("no-such-chunk")).toBeNull();
    expect(await instance.listFiles("project-a")).toEqual([]);
    expect(await instance.listChunks(["project-a"])).toEqual([]);
    expect(await instance.countChunks(["project-a"])).toBe(0);
    expect(await instance.outlineChunks("project-a", "module.py")).toEqual([]);
    expect(
      await instance.findSymbolChunks("answer", "project-a", { match: "exact", limit: 5 }),
    ).toEqual([]);
    expect(await instance.listReferenceRecords("project-a", {})).toEqual([]);
    expect(await instance.referenceCoverage("project-a")).toEqual([]);
    expect(await instance.declarationShapes("project-a", "answer", {})).toEqual([]);
    expect(await instance.targetNameCandidates("project-a", "answer", {})).toEqual([]);
    expect(await instance.declarationsForFiles("project-a", ["file-1"], {})).toEqual([]);
    expect(fs.existsSync(path.join(temporary, "projects"))).toBe(false);
    await instance.close();
  });

  test("uses one partition per project", async () => {
    const instance = await store();
    await instance.upsertFile(file("project-a"));
    expect(fs.existsSync(path.join(temporary, "registry", "projects.lance"))).toBe(true);
    expect(fs.existsSync(path.join(temporary, "projects", "project-a", "files.lance"))).toBe(true);
    expect(fs.existsSync(path.join(temporary, "projects", "project-a", "chunks.lance"))).toBe(true);
    expect(fs.existsSync(path.join(temporary, "projects", "project-a", "references.lance"))).toBe(
      true,
    );
    await instance.close();
  });

  test("applies exact structural filters and schema-version pushdown", async () => {
    const instance = await store();
    const coverage = {
      ...reference("project-a"),
      reference_id: "coverage",
      record_kind: "coverage",
      kind: null,
      source_qualified_symbol: null,
      written_name: null,
      target_name: null,
      shape_json: null,
    };
    const declaration = {
      ...reference("project-a"),
      reference_id: "declaration",
      record_kind: "declaration",
      source_qualified_symbol: "package.answer",
      target_name: "answer",
      shape_json: "[]",
    };
    const imported = {
      ...reference("project-a"),
      reference_id: "import",
      record_kind: "reference",
      kind: "import",
      module_path: "package",
      target_name: "answer",
    };
    const stale = { ...imported, reference_id: "stale", schema_version: 3 };
    await instance.replaceFilesFromArrow("project-a", {
      files: [file("project-a")],
      chunkBatches: [],
      referenceBatches: [{ fileIds: ["file-a"], rows: [coverage, declaration, imported, stale] }],
    });

    expect(await instance.coverageForFile("project-a", "file-a", REFERENCE_SCHEMA_VERSION)).toEqual(
      [coverage],
    );
    expect(await instance.declarationShapes("project-a", "package.answer", {})).toMatchObject([
      { reference_id: "declaration" },
    ]);
    expect(
      (await instance.targetNameCandidates("project-a", "answer", {})).map(
        (row) => row.reference_id,
      ),
    ).toEqual(["declaration", "import", "stale"]);
    expect(
      await instance.targetNameCandidates("project-a", "answer", { recordKind: "declaration" }),
    ).toMatchObject([{ reference_id: "declaration" }]);
    expect(await instance.declarationsForFiles("project-a", ["file-a"], {})).toMatchObject([
      { reference_id: "declaration" },
    ]);
    expect(await instance.declarationsForFiles("project-a", [], {})).toEqual([]);
    expect(
      await instance.listReferenceRecords("project-a", { recordKinds: ["declaration"] }),
    ).toMatchObject([{ reference_id: "declaration" }]);
    expect(
      (await instance.listReferenceRecords("project-a", { schemaVersion: 4 })).map(
        (row) => row.reference_id,
      ),
    ).toEqual(["coverage", "declaration", "import"]);
    expect(await instance.listReferenceRecords("project-a", { schemaVersion: 3 })).toMatchObject([
      { reference_id: "stale" },
    ]);
    expect(() =>
      instance.listReferenceRecords("project-a", { schemaVersion: true as never }),
    ).toThrow("schema_version");
    expect(() => instance.coverageForFile("project-a", "file-a", "1" as never)).toThrow(
      "schema_version",
    );
    await instance.close();
  });

  test("indexes every exact reference filter column", async () => {
    const instance = await store();
    await instance.ensureIndexes("project-a");
    await instance.ensureIndexes("project-a");
    const stats = await instance.storageStats("project-a");
    const columns = new Set(
      stats?.tables
        .find((table) => table.name === "references")
        ?.indexes.flatMap((index) => index.columns),
    );
    expect(columns).toEqual(
      new Set([
        "file_id",
        "record_kind",
        "target_name",
        "module_path",
        "kind",
        "source_qualified_symbol",
        "schema_version",
      ]),
    );
    await instance.close();
  });

  test("migrates a v1 store and registers it for lazy rebuild", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const info = project("legacy", root);
    const legacy = await import("@lancedb/lancedb");
    const connection = await legacy.connect(temporary);
    const projects = await connection.createEmptyTable("projects", LanceStore.projectArrowSchema());
    await projects.add([
      {
        id: info.id,
        name: info.name,
        root: info.root,
        payload: JSON.stringify(info),
        model_id: "test/model",
        vector_dimension: 2,
        schema_version: 1,
        state: "ready",
        updated_at: 1n,
      },
    ]);
    connection.close();

    const instance = new LanceStore(temporary, { vectorDimension: 2 });
    expect(await instance.listProjects()).toEqual([info]);
    expect(await instance.projectState("legacy")).toBe("pending");
    expect(
      fs
        .readdirSync(path.dirname(temporary))
        .some((name) => name.includes("lancedb-v1-backup-") || name.includes("-v1-backup-")),
    ).toBe(true);
    await instance.close();
  });

  test("has_file_errors ignores rejection tombstones", async () => {
    const instance = await store();
    expect(await instance.hasFileErrors("project-a")).toBe(false);
    await instance.upsertFile({
      ...file("project-a"),
      has_errors: true,
      error: "rejected: binary",
    });
    expect(await instance.hasFileErrors("project-a")).toBe(false);
    await instance.upsertFile({
      ...file("project-a", "broken"),
      has_errors: true,
      error: "parse failed",
    });
    expect(await instance.hasFileErrors("project-a")).toBe(true);
    await instance.close();
  });

  test("distinguishes a missing reference table from an empty one", async () => {
    const instance = await store();
    expect(await instance.hasReferenceTable("missing")).toBe(false);
    await instance.upsertFile(file("project-a"));
    expect(await instance.hasReferenceTable("project-a")).toBe(true);
    expect(await instance.listReferenceRecords("project-a", {})).toEqual([]);
    await instance.close();
  });

  test("get_chunk rejects malformed, unknown, and pre-migration ids", async () => {
    const instance = await store();
    await instance.replaceFile(file("project-a"), [chunk("project-a")], [reference("project-a")]);
    expect(await instance.getChunk("not-a-chunk")).toBeNull();
    expect(await instance.getChunk(":missing")).toBeNull();
    expect(await instance.getChunk("unknown:digest")).toBeNull();
    expect(await instance.getChunk("project-a:")).toBeNull();
    await instance.close();
  });

  test("removing a file deletes its reference rows", async () => {
    const instance = await store();
    await instance.replaceFile(file("project-a"), [chunk("project-a")], [reference("project-a")]);
    expect(await instance.countChunks(["project-a"])).toBe(1);
    await instance.removeFile("project-a", "file-a");
    expect(await instance.listFiles("project-a")).toEqual([]);
    expect(await instance.countChunks(["project-a"])).toBe(0);
    expect(await instance.listReferenceRecords("project-a", {})).toEqual([]);
    await instance.close();
  });

  test("an empty batch deletes that file's previous chunks", async () => {
    const instance = await store();
    await instance.replaceFilesFromArrow("project-a", {
      files: [file("project-a")],
      chunkBatches: [
        { fileIds: ["file-a"], rows: [chunk("project-a"), chunk("project-a", "file-a", "two")] },
      ],
    });
    expect(await instance.countChunks(["project-a"])).toBe(2);
    await instance.replaceFilesFromArrow("project-a", {
      files: [file("project-a")],
      chunkBatches: [{ fileIds: ["file-a"], rows: [] }],
    });
    expect(await instance.countChunks(["project-a"])).toBe(0);
    await instance.close();
  });

  test("replacement ids win over removal ids", async () => {
    const instance = await store();
    await instance.replaceFilesFromArrow("project-a", {
      files: [file("project-a"), file("project-a", "gone")],
      chunkBatches: [
        { fileIds: ["file-a"], rows: [chunk("project-a", "file-a", "old-a")] },
        { fileIds: ["gone"], rows: [chunk("project-a", "gone", "old")] },
      ],
    });
    const kept = chunk("project-a", "file-a", "kept");
    await instance.replaceFilesFromArrow("project-a", {
      files: [file("project-a")],
      chunkBatches: [{ fileIds: ["file-a"], rows: [kept] }],
      removedFileIds: ["file-a", "gone"],
    });
    expect((await instance.listChunks(["project-a"])).map((item) => item.chunk_id)).toEqual([
      "project-a:kept",
    ]);
    await instance.close();
  });

  test("upsert_project skips a noop registry write", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const info = project("project-a", root);
    const instance = new LanceStore(temporary, { vectorDimension: 2 });
    await instance.upsertProject(info, { modelId: "model", state: "ready" });
    const before = await instance.registryStats();
    await instance.upsertProject(info, { modelId: "model", state: "ready" });
    expect((await instance.registryStats()).current_version).toBe(before.current_version);
    await instance.upsertProject(info, { modelId: "model", state: "partial" });
    expect((await instance.registryStats()).current_version).toBeGreaterThan(
      before.current_version,
    );
    await instance.close();
  });

  test("delete_partition preserves registration and re-stamps the generation", async () => {
    const instance = await store();
    await instance.replaceFile(file("project-a"), [chunk("project-a")], [reference("project-a")]);
    expect(await instance.deletePartition("project-a", "model")).toBe(true);
    expect(await instance.projectState("project-a")).toBe("indexing");
    expect(await instance.getChunk("project-a:one")).toBeNull();
    expect((await instance.listProjects())[0]?.id).toBe("project-a");
    await instance.close();
  });

  test("evicts least-recently-used partitions and reopens their data", async () => {
    const instance = new LanceStore(temporary, { vectorDimension: 2 });
    await instance.upsertProject(project(), { modelId: "model" });
    await instance.replaceFile(file("project-a"), [chunk("project-a")], [reference("project-a")]);
    for (let index = 0; index < MAX_CACHED_PARTITIONS + 1; index += 1) {
      const id = `filler-${index.toString().padStart(2, "0")}`;
      await instance.upsertProject(project(id, path.join(temporary, id)), { modelId: "model" });
      await instance.openTables(id);
    }
    expect(instance.cachedPartitionIds()).not.toContain("project-a");
    expect(instance.cachedPartitionIds()).toHaveLength(MAX_CACHED_PARTITIONS);
    expect(await instance.countChunks(["project-a"])).toBe(1);
    expect(await instance.getChunk("project-a:one")).not.toBeNull();
    await instance.close();
  });

  test("keeps a recently used partition when the cache overflows", async () => {
    const instance = new LanceStore(temporary, { vectorDimension: 2 });
    const ids: string[] = [];
    for (let index = 0; index < MAX_CACHED_PARTITIONS; index += 1) {
      const id = `id-${index.toString().padStart(2, "0")}`;
      await instance.upsertProject(project(id, path.join(temporary, id)), { modelId: "model" });
      await instance.openTables(id);
      ids.push(id);
    }
    await instance.openTables(ids[0] as string);
    await instance.upsertProject(project("overflow", path.join(temporary, "overflow")), {
      modelId: "model",
    });
    await instance.openTables("overflow");
    expect(instance.cachedPartitionIds()).toContain(ids[0] as string);
    expect(instance.cachedPartitionIds()).not.toContain(ids[1] as string);
    await instance.close();
  });

  test("storage stats work without a partition and flag a damaged one", async () => {
    const instance = new LanceStore(temporary, { vectorDimension: 2 });
    await instance.upsertProject(project(), { modelId: "model", state: "pending" });
    const missing = await instance.storageStats("project-a");
    expect(missing?.tables).toEqual([]);
    expect(missing?.partition_open_failed).toBe(false);
    fs.mkdirSync(path.join(temporary, "projects", "project-a"), { recursive: true });
    const damaged = await instance.storageStats("project-a");
    expect(damaged?.partition_open_failed).toBe(true);
    expect(damaged?.consistent).toBe(false);
    await instance.close();
  });

  test("does not follow symlinks when counting physical bytes", async () => {
    const instance = await store();
    await instance.upsertFile(file("project-a"));
    const outside = path.join(temporary, "outside.bin");
    fs.writeFileSync(outside, "x".repeat(1_000_000));
    fs.symlinkSync(outside, path.join(temporary, "projects", "project-a", "escape"));
    const stats = await instance.storageStats("project-a");
    expect(stats?.partition_physical_bytes ?? 0).toBeLessThan(1_000_000);
    await instance.close();
  });

  test("stores chunk vectors as float16 by default and float32 on opt-out", async () => {
    const narrowed = await store();
    await narrowed.replaceFile(file("project-a"), [chunk("project-a")], [reference("project-a")]);
    const hits = await narrowed.hybridSearch("answer", [1, 0], ["project-a"], { limit: 5 });
    expect(hits[0]?.chunk_id).toBe("project-a:one");
    await narrowed.close();

    const wide = new LanceStore(path.join(temporary, "f32"), {
      vectorDimension: 2,
      vectorStorage: "float32",
    });
    await wide.upsertProject(project("wide", path.join(temporary, "wide")), { modelId: "model" });
    await wide.replaceFile(file("wide"), [chunk("wide")], [reference("wide")]);
    await wide.close();
    expect(() => new LanceStore(temporary, { vectorStorage: "int8" as never })).toThrow(
      "vector_storage",
    );
  });

  test("a vector-storage flip marks rebuild required", async () => {
    const first = await store();
    await first.replaceFile(file("project-a"), [chunk("project-a")], [reference("project-a")]);
    await first.close();
    const flipped = new LanceStore(temporary, { vectorDimension: 2, vectorStorage: "float32" });
    expect(await flipped.incompatibilityReason("project-a", "model")).toContain("vector storage");
    await flipped.upsertProject(project(), { modelId: "model" });
    expect(await flipped.projectState("project-a")).toBe("rebuild_required");
    await flipped.close();
  });

  test("the merge-semantics probe passes and a failed probe refuses the commit", async () => {
    expect(await probeBatchedMergeSemantics()).toBe(true);
    const instance = await store();
    setBatchedMergeSemanticsOk(false);
    try {
      await expect(
        instance.replaceFilesFromArrow("project-a", { files: [], chunkBatches: [] }),
      ).rejects.toMatchObject({ code: "UNSUPPORTED_RUNTIME" });
    } finally {
      setBatchedMergeSemanticsOk(undefined);
    }
    expect(await instance.listChunks(["project-a"])).toEqual([]);
    await instance.close();
  });

  test("maintenance skips a registered project without a partition", async () => {
    const instance = new LanceStore(temporary, { vectorDimension: 2 });
    await instance.upsertProject(project(), { modelId: "model", state: "pending" });
    expect(await instance.maintainProject("project-a", new Date(0))).toBe(false);
    await instance.close();
  });
});

describe("overlap and worktree warnings", () => {
  test("detect duplicate and nested roots", () => {
    const root = path.join(temporary, "repo");
    const nested = path.join(root, "src");
    const other = path.join(temporary, "other");
    fs.mkdirSync(nested, { recursive: true });
    fs.mkdirSync(other);
    const info = (id: string, rootPath: string): ProjectInfo => project(id, rootPath);
    expect(overlapWarnings([info("a", root), info("b", root)])).toEqual([
      `Projects 'a' and 'b' register the same root: ${root}`,
    ]);
    const nestedWarnings = overlapWarnings([info("a", root), info("b", nested)]);
    expect(nestedWarnings).toHaveLength(1);
    expect(nestedWarnings[0]).toMatch(/nested inside|contains the root/);
    expect(overlapWarnings([info("a", root), info("b", other)])).toEqual([]);
  });

  test("overlapping registration detects exact, nested, and parent roots", () => {
    const root = path.join(temporary, "repo");
    const nested = path.join(root, "src");
    const sibling = path.join(temporary, "sibling");
    fs.mkdirSync(nested, { recursive: true });
    fs.mkdirSync(sibling);
    const parent = project("parent", root);
    const child = project("child", nested);
    expect(overlappingRegistration([parent], root)).toEqual(parent);
    expect(overlappingRegistration([parent], nested)).toEqual(parent);
    expect(overlappingRegistration([child], root)).toEqual(child);
    expect(overlappingRegistration([parent, child], sibling)).toBeNull();
  });

  test("overlapping registration matches case-insensitive aliases", () => {
    const root = path.join(temporary, "repo");
    const nested = path.join(root, "src");
    fs.mkdirSync(nested, { recursive: true });
    const alias = caseInsensitiveAlias(root);
    if (alias === null) return;
    const registered = project("parent", alias);
    expect(overlappingRegistration([registered], nested)).toEqual(registered);
    expect(overlappingRegistration([project("child", nested)], alias)).not.toBeNull();
  });

  test("worktree warnings share a git common directory", () => {
    const first = path.join(temporary, "first");
    const second = path.join(temporary, "second");
    fs.mkdirSync(first);
    fs.mkdirSync(second);
    const common = path.join(temporary, ".git");
    const projects = [project("a", first), project("b", second)];
    const warnings = worktreeWarnings(projects, (command, cwd) => {
      if (command.at(-1) === "--show-toplevel") return cwd;
      return common;
    });
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain("common directory");
    expect(
      worktreeWarnings(projects, (command, cwd) =>
        command.at(-1) === "--show-toplevel" ? cwd : path.join(cwd, ".git"),
      ),
    ).toEqual([]);
    expect(worktreeWarnings(projects, () => null)).toEqual([]);
  });

  test("relative git common directories resolve against the registered root", () => {
    const mainRoot = path.join(temporary, "repo");
    const worktreeRoot = path.join(temporary, "worktree");
    fs.mkdirSync(mainRoot);
    fs.mkdirSync(worktreeRoot);
    const common = path.join(mainRoot, ".git");
    const warnings = worktreeWarnings(
      [project("a", mainRoot), project("b", worktreeRoot)],
      (command, cwd) => {
        if (command.at(-1) === "--show-toplevel") return cwd;
        return cwd === mainRoot ? ".git" : common;
      },
    );
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain(common);
  });
});
