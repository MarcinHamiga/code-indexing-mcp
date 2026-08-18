/** Journalled Arrow staging and bounded crash recovery. */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs/promises";
import path from "node:path";
import { Float16, tableFromIPC, type Schema } from "apache-arrow";
import type { StoredChunk, StoredFile } from "../src/models.ts";
import type { ReferenceRecord } from "../src/reference-store.ts";
import { LanceStore, type TableVersions } from "../src/storage.ts";
import type { StagingStore } from "../src/staging.ts";
import {
  CHUNKS_NAME,
  FILES_NAME,
  JOURNAL_NAME,
  MAX_RECOVERY_ATTEMPTS,
  PHASE_COMMITTING,
  StagingJob,
  hasPendingRecovery,
  recoverStagedCommits,
} from "../src/staging.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

let temporary: string;

beforeEach(() => {
  temporary = temporaryDirectory();
});

afterEach(() => {
  removeDirectory(temporary);
});

const versions: TableVersions = { files: 1, chunks: 2, references: 3 };

function schemas(vectorStorage: "float16" | "float32" = "float16") {
  return {
    fileSchema: LanceStore.fileArrowSchema(),
    chunkSchema: LanceStore.chunkArrowSchema(4, vectorStorage),
    referenceSchema: LanceStore.referenceArrowSchema(),
  };
}

function job(root: string, projectId: string, jobId: string): StagingJob {
  return new StagingJob(root, projectId, { ...schemas(), jobId });
}

function schemaShape(schema: Schema): unknown[] {
  return schema.fields.map((field) => ({
    name: field.name,
    nullable: field.nullable,
    type: field.type.toString(),
  }));
}

function file(fileId = "file-a"): StoredFile {
  return {
    file_id: fileId,
    project_id: "project",
    path: `${fileId}.py`,
    language: "python",
    size: 10,
    mtime_ns: 1n,
    content_hash: `hash-${fileId}`,
    has_errors: false,
    error: null,
    indexed_at: 1,
  };
}

function chunk(fileId = "file-a", chunkId = "chunk-a"): StoredChunk {
  return {
    chunk_id: chunkId,
    file_id: fileId,
    path: `${fileId}.py`,
    language: "python",
    kind: "function",
    symbol: "answer",
    qualified_symbol: "answer",
    parent_symbol: null,
    start_byte: 0,
    end_byte: 10,
    start_line: 1,
    end_line: 1,
    content: "def answer",
    identifier_terms: "answer",
    content_hash: `hash-${fileId}`,
    part_index: 0,
    vector: [0, 0, 0, 1],
  };
}

function reference(fileId = "file-a", referenceId = "reference-a"): ReferenceRecord {
  return {
    reference_id: referenceId,
    record_kind: "coverage",
    file_id: fileId,
    project_id: "project",
    path: `${fileId}.py`,
    language: "python",
    kind: null,
    source_qualified_symbol: null,
    written_name: null,
    target_name: null,
    module_path: null,
    imported_name: null,
    alias: null,
    receiver_text: null,
    start_byte: null,
    end_byte: null,
    start_line: null,
    end_line: null,
    shape_json: null,
    content_hash: `hash-${fileId}`,
    schema_version: 4,
  };
}

class RecoveryStore implements StagingStore {
  restored: Array<{ projectId: string; versions: TableVersions; restoreReferences: boolean }> = [];
  states: Array<{ projectId: string; state: string }> = [];
  removed = new Set<string>();
  fail = false;

  async tableVersions(): Promise<TableVersions> {
    return versions;
  }

  async replaceFilesFromArrow(): Promise<void> {}

  async restoreVersions(
    projectId: string,
    target: TableVersions,
    options: { restoreReferences?: boolean } = {},
  ): Promise<boolean> {
    if (this.fail) throw new Error("restore failed");
    this.restored.push({
      projectId,
      versions: target,
      restoreReferences: options.restoreReferences ?? true,
    });
    return !this.removed.has(projectId);
  }

  async markProjectState(projectId: string, state: string): Promise<boolean> {
    this.states.push({ projectId, state });
    return true;
  }
}

describe("Arrow IPC staging", () => {
  test("stages durable Arrow payloads and batches complete files, including zero-row replacements", async () => {
    const staged = job(path.join(temporary, "staging"), "project", "job");
    await staged.begin();
    await staged.stageFile(file());
    await staged.stageChunks([chunk("file-a", "a-1"), chunk("file-a", "a-2")]);
    await staged.stageChunks([chunk("file-b", "b-1")]);
    await staged.stageReferences([reference("file-a")]);
    staged.markReplaced("file-a");
    staged.markReplaced("file-b");
    staged.markReplaced("file-empty");
    staged.markReferencesReplaced("file-a");
    await staged.beginCommit(versions);

    const filesPayload = tableFromIPC(await fs.readFile(path.join(staged.directory, FILES_NAME)));
    const chunksPayload = tableFromIPC(await fs.readFile(path.join(staged.directory, CHUNKS_NAME)));
    expect(filesPayload.numRows).toBe(1);
    expect(schemaShape(filesPayload.schema)).toEqual(schemaShape(LanceStore.fileArrowSchema()));
    expect(schemaShape(chunksPayload.schema)).toEqual(schemaShape(LanceStore.chunkArrowSchema(4)));
    expect(chunksPayload.getChild("vector")?.type.children[0]?.type.precision).toBe(
      new Float16().precision,
    );
    expect(await staged.filesTable()).toEqual([file()]);
    expect(await staged.iterChunkBatches({ maxFiles: 1 })).toEqual([
      { fileIds: ["file-a"], rows: [chunk("file-a", "a-1"), chunk("file-a", "a-2")] },
      { fileIds: ["file-b"], rows: [chunk("file-b", "b-1")] },
      { fileIds: ["file-empty"], rows: [] },
    ]);
    expect(await staged.iterReferenceBatches()).toEqual([
      { fileIds: ["file-a"], rows: [reference("file-a")] },
    ]);
    await expect(hasPendingRecovery(path.join(temporary, "staging"), "project")).resolves.toBe(
      true,
    );
    await staged.complete();
    await expect(fs.stat(staged.directory)).rejects.toMatchObject({ code: "ENOENT" });
  });

  test("writes correct schemas even when every staged payload has zero rows", async () => {
    const staged = job(path.join(temporary, "staging"), "project", "empty");
    await staged.begin();
    await staged.beginCommit(versions);

    for (const [name, expected] of [
      [FILES_NAME, LanceStore.fileArrowSchema()],
      [CHUNKS_NAME, LanceStore.chunkArrowSchema(4)],
      ["references.arrow", LanceStore.referenceArrowSchema()],
    ] as const) {
      const payload = tableFromIPC(await fs.readFile(path.join(staged.directory, name)));
      expect(payload.numRows).toBe(0);
      expect(schemaShape(payload.schema)).toEqual(schemaShape(expected));
    }
  });

  test("rejects non-contiguous file batches and retains a committing journal on discard", async () => {
    const staged = job(path.join(temporary, "staging"), "project", "job");
    await staged.begin();
    await staged.stageChunks([chunk("file-a")]);
    await staged.stageChunks([chunk("file-b")]);
    await expect(staged.stageChunks([chunk("file-a", "a-2")])).rejects.toThrow("contiguous");
    await staged.beginCommit(versions);
    await staged.discard();

    expect(
      JSON.parse(await fs.readFile(path.join(staged.directory, JOURNAL_NAME), "utf8")).phase,
    ).toBe(PHASE_COMMITTING);
  });

  test("rejects mixed-file chunk and reference batches", async () => {
    const staged = job(path.join(temporary, "staging"), "project", "job");
    await staged.begin();
    await expect(staged.stageChunks([chunk("file-a"), chunk("file-b")])).rejects.toThrow(
      "one file",
    );
    await expect(
      staged.stageReferences([reference("file-a"), { ...reference("file-b"), reference_id: "rb" }]),
    ).rejects.toThrow("one file");
  });

  test("commit batches respect the file, row, and byte limits", async () => {
    const staged = job(path.join(temporary, "staging"), "project", "job");
    await staged.begin();
    for (let index = 0; index < 5; index += 1) {
      await staged.stageChunks([chunk(`file-${index}`, `chunk-${index}`)]);
      staged.markReplaced(`file-${index}`);
    }
    await staged.beginCommit(versions);
    const byFiles = await staged.iterChunkBatches({ maxFiles: 2 });
    expect(byFiles.map((batch) => batch.fileIds)).toEqual([
      ["file-0", "file-1"],
      ["file-2", "file-3"],
      ["file-4"],
    ]);
  });

  test("a single file may exceed the row and byte bounds", async () => {
    const staged = job(path.join(temporary, "staging"), "project", "job");
    await staged.begin();
    await staged.stageChunks([
      chunk("file-a", "a-1"),
      chunk("file-a", "a-2"),
      chunk("file-a", "a-3"),
    ]);
    staged.markReplaced("file-a");
    await staged.beginCommit(versions);
    const batches = await staged.iterChunkBatches({ maxFiles: 1, maxRows: 1, maxBytes: 1 });
    expect(batches).toHaveLength(1);
    expect(batches[0]?.fileIds).toEqual(["file-a"]);
    expect(batches[0]?.rows).toHaveLength(3);
  });
});

describe("staged commit recovery", () => {
  test("restores committing journals, removes staging leftovers, and leaves removed projects alone", async () => {
    const root = path.join(temporary, "staging");
    const staged = job(root, "project", "commit");
    await staged.begin();
    await staged.beginCommit(versions);
    const staging = job(root, "project", "staging");
    await staging.begin();
    const missing = job(root, "missing", "missing");
    await missing.begin();
    await missing.beginCommit(versions);
    const store = new RecoveryStore();
    store.removed.add("missing");

    expect(await recoverStagedCommits(root, store)).toBe(1);
    expect(store.restored).toEqual([
      { projectId: "missing", versions, restoreReferences: true },
      { projectId: "project", versions, restoreReferences: true },
    ]);
    await expect(fs.stat(staged.directory)).rejects.toMatchObject({ code: "ENOENT" });
    await expect(fs.stat(staging.directory)).rejects.toMatchObject({ code: "ENOENT" });
    await expect(fs.stat(missing.directory)).rejects.toMatchObject({ code: "ENOENT" });
  });

  test("bounds permanent recovery failures and marks the project error", async () => {
    const root = path.join(temporary, "staging");
    const staged = job(root, "project", "commit");
    await staged.begin();
    await staged.beginCommit(versions);
    const store = new RecoveryStore();
    store.fail = true;

    for (let attempt = 1; attempt < MAX_RECOVERY_ATTEMPTS; attempt += 1) {
      expect(await recoverStagedCommits(root, store)).toBe(0);
      const journal = JSON.parse(
        await fs.readFile(path.join(staged.directory, JOURNAL_NAME), "utf8"),
      );
      expect(journal.recovery_attempts).toBe(attempt);
    }
    expect(await recoverStagedCommits(root, store)).toBe(0);
    expect(store.states).toEqual([{ projectId: "project", state: "error" }]);
    await expect(fs.stat(staged.directory)).rejects.toMatchObject({ code: "ENOENT" });
  });

  test("rolls back a legacy two-table journal without restoring references", async () => {
    const root = path.join(temporary, "staging");
    const directory = path.join(root, "project", "legacy");
    await fs.mkdir(directory, { recursive: true });
    await fs.writeFile(
      path.join(directory, JOURNAL_NAME),
      JSON.stringify({
        version: 1,
        phase: PHASE_COMMITTING,
        project_id: "project",
        files_version: 1,
        chunks_version: 2,
      }),
    );
    const store = new RecoveryStore();
    expect(await recoverStagedCommits(root, store)).toBe(1);
    expect(store.restored).toEqual([
      {
        projectId: "project",
        versions: { files: 1, chunks: 2, references: 0 },
        restoreReferences: false,
      },
    ]);
  });

  test("repeated recovery is idempotent", async () => {
    const root = path.join(temporary, "staging");
    const staged = job(root, "project", "commit");
    await staged.begin();
    await staged.beginCommit(versions);
    const store = new RecoveryStore();
    expect(await recoverStagedCommits(root, store)).toBe(1);
    expect(await recoverStagedCommits(root, store)).toBe(0);
    expect(store.restored).toHaveLength(1);
  });
});
