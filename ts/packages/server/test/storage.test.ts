import { afterEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { isCodeIndexingError } from "../src/errors.ts";
import {
  REFERENCE_SCHEMA_VERSION,
  type ReferenceRecord,
  type ReferenceStore,
} from "../src/reference-store.ts";
import { LanceStore } from "../src/storage.ts";
import type { ProjectInfo, StoredChunk, StoredFile } from "../src/models.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

const temporary = temporaryDirectory();

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
});
