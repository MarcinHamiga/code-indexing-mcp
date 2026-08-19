import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { runtimePathsFromEnvironment } from "../src/application.ts";
import {
  REPEATED_EDITS,
  RETRIEVAL_TOPICS,
  SEARCH_ITERATIONS,
  buildRetrievalCorpus,
  directoryPhysicalBytes,
  durationSummary,
  runIndexBenchmark,
  runPrecisionBenchmark,
  runPrecisionBenchmarkCommand,
  runSearchBenchmark,
  writeBenchmarkCorpus,
} from "../src/benchmark.ts";
import type { Embedder } from "../src/embedding.ts";
import { isCodeIndexingError } from "../src/errors.ts";
import {
  IndexReport,
  MaintenanceReport,
  ProjectInfo,
  ProjectStorageStats,
  SearchHit,
  SearchResponse,
  StorageStatus,
  TableStorageStats,
} from "../src/models.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

let temporary: string;

beforeEach(() => {
  temporary = temporaryDirectory();
});

afterEach(() => {
  removeDirectory(temporary);
});

class BenchmarkApplication {
  readonly root: string;
  readonly durationMs: number;
  readonly forceCalls: boolean[] = [];
  readonly storageCalls: string[] = [];
  readonly maintenanceCalls: Array<[string | null, boolean]> = [];

  constructor(root: string, { durationMs = 100 }: { durationMs?: number } = {}) {
    this.root = root;
    this.durationMs = durationMs;
  }

  async initProject(target: string): Promise<ProjectInfo> {
    expect(target).toBe(this.root);
    return ProjectInfo.parse({ id: "benchmark-project", name: "benchmark", root: target });
  }

  async indexProject(
    project: string,
    { force = false }: { force?: boolean } = {},
  ): Promise<IndexReport> {
    expect(project).toBe("benchmark-project");
    this.forceCalls.push(force);
    return IndexReport.parse({
      project_id: project,
      discovered_files: 4,
      indexed_files: 1,
      parsed_files: 1,
      embedded_chunks: 8,
      duration_ms: this.durationMs,
      embedding_backend: "cpu",
      embedding_batch_size: 8,
      staged_reference_rows: 12,
      reference_extraction_duration_ms: 12,
    });
  }

  async storageStatus(project?: string | null): Promise<StorageStatus> {
    this.storageCalls.push(project ?? "");
    const entry = ProjectStorageStats.parse({
      project: ProjectInfo.parse({
        id: "benchmark-project",
        name: "benchmark",
        root: this.root,
      }),
      snapshot_at: "2026-08-11T00:00:00+00:00",
      tables: [],
      partition_physical_bytes: this.storageCalls.length,
      consistent: true,
    });
    return StorageStatus.parse({
      snapshot_at: "2026-08-11T00:00:00+00:00",
      registry: TableStorageStats.parse({ name: "projects" }),
      projects: [entry],
    });
  }

  async maintainStorage(
    project?: string | null,
    { waitForLock = false }: { waitForLock?: boolean } = {},
  ): Promise<MaintenanceReport> {
    this.maintenanceCalls.push([project ?? null, waitForLock]);
    return MaintenanceReport.parse({
      trigger: "manual",
      dry_run: false,
      retention_hours: 24,
      started_at: "2026-08-11T00:00:00+00:00",
      finished_at: "2026-08-11T00:00:01+00:00",
      duration_ms: 1000,
      registry_status: "ok",
    });
  }
}

describe("index benchmark", () => {
  test("runs the storage-growth scenarios", async () => {
    const root = path.join(temporary, "corpus");
    writeBenchmarkCorpus(root, { files: 8, functionsPerFile: 2 });
    const app = new BenchmarkApplication(root);
    const payload = await runIndexBenchmark(app, root);
    expect(app.forceCalls).toEqual([
      true,
      false,
      false,
      ...Array.from({ length: 100 }, () => false),
      true,
      false,
      false,
    ]);
    expect(Object.keys(payload.scenarios as object)).toEqual([
      "cold_start",
      "no_op",
      "single_file_edit",
      "repeated_edits",
      "forced_reindex",
      "single_file_deletion",
      "many_file_deletions",
      "post_maintenance",
    ]);
    expect(payload.schema_version).toBe(2);
    const baseline = payload.storage_baseline as { partition_physical_bytes: number };
    expect(baseline.partition_physical_bytes).toBe(1);
    const cold = (
      payload.scenarios as Record<string, { storage_after: { partition_physical_bytes: number } }>
    ).cold_start;
    expect(cold?.storage_after.partition_physical_bytes).toBe(
      baseline.partition_physical_bytes + 1,
    );
    expect(
      (payload.scenarios as Record<string, { includes_embedder_warmup?: boolean }>).cold_start
        ?.includes_embedder_warmup,
    ).toBe(true);
    expect(app.maintenanceCalls).toEqual([["benchmark-project", true]]);
    expect(app.storageCalls).toHaveLength(9);
    const edited = fs.readFileSync(path.join(root, "module_0000.py"), "utf8");
    expect(edited).toContain("phase_2_single_edit_marker");
    expect(edited).toContain("repeated_edit_marker_0099");
    expect(fs.existsSync(path.join(root, "module_0001.py"))).toBe(false);
    for (let deletedIndex = 2; deletedIndex < 10; deletedIndex += 1) {
      expect(
        fs.existsSync(path.join(root, `module_${String(deletedIndex).padStart(4, "0")}.py`)),
      ).toBe(false);
    }
  });

  test("derives the numbers it publishes", async () => {
    const root = path.join(temporary, "corpus");
    writeBenchmarkCorpus(root, { files: 8, functionsPerFile: 2 });
    const payload = await runIndexBenchmark(new BenchmarkApplication(root), root);
    for (const name of ["cold_start", "no_op", "single_file_edit", "forced_reindex"]) {
      const scenario = (payload.scenarios as Record<string, Record<string, unknown>>)[name];
      expect(scenario?.reported_duration_ms).toBe(100);
      expect(scenario?.chunks_per_second).toBe(80);
      expect(scenario?.structural_records).toBe(12);
      expect(scenario?.reference_extraction_duration_ms).toBe(12);
      expect(scenario?.wall_ms as number).toBeGreaterThanOrEqual(0);
      expect(scenario?.wall_ms as number).toBeLessThan(100);
    }
  });

  test("throughput is null when the indexer reports no duration", async () => {
    const root = path.join(temporary, "corpus");
    writeBenchmarkCorpus(root, { files: 4, functionsPerFile: 1 });
    const payload = await runIndexBenchmark(
      new BenchmarkApplication(root, { durationMs: 0 }),
      root,
    );
    const scenario = (payload.scenarios as Record<string, Record<string, unknown>>).cold_start;
    expect(scenario?.reported_duration_ms).toBe(0);
    expect(scenario?.chunks_per_second).toBeNull();
  });

  test("repeated edits reports a distribution", async () => {
    const root = path.join(temporary, "corpus");
    writeBenchmarkCorpus(root, { files: 4, functionsPerFile: 1 });
    const payload = await runIndexBenchmark(new BenchmarkApplication(root), root);
    const summary = (
      payload.scenarios as Record<string, { per_edit_ms: Record<string, number>; wall_ms: number }>
    ).repeated_edits;
    expect(summary?.per_edit_ms.count).toBe(REPEATED_EDITS);
    expect(summary?.per_edit_ms.min_ms).toBeLessThanOrEqual(summary?.per_edit_ms.median_ms ?? 0);
    expect((summary?.per_edit_ms.total_ms ?? 0) <= (summary?.wall_ms ?? 0) + 0.1).toBe(true);
  });

  test("duration summary computes order statistics", () => {
    const summary = durationSummary(Array.from({ length: 20 }, (_, index) => index + 1));
    expect(summary).toEqual({
      count: 20,
      total_ms: 210,
      min_ms: 1,
      max_ms: 20,
      median_ms: 10.5,
      p95_ms: 19,
      first_decile_mean_ms: 1.5,
      last_decile_mean_ms: 19.5,
    });
    expect(durationSummary([])).toEqual({ count: 0 });
  });

  test("corpus is deterministic", () => {
    const first = path.join(temporary, "first");
    const second = path.join(temporary, "second");
    expect(writeBenchmarkCorpus(first, { files: 3, functionsPerFile: 4 })).toBe(
      writeBenchmarkCorpus(second, { files: 3, functionsPerFile: 4 }),
    );
    const names = fs.readdirSync(first).sort();
    expect(names.map((name) => fs.readFileSync(path.join(first, name)))).toEqual(
      names.map((name) => fs.readFileSync(path.join(second, name))),
    );
  });
});

class SearchApp {
  readonly initCalls: string[] = [];
  readonly searchCalls: number[] = [];

  async initProject(target: string): Promise<ProjectInfo> {
    this.initCalls.push(target);
    return ProjectInfo.parse({
      id: `id-${path.basename(target)}`,
      name: path.basename(target),
      root: target,
    });
  }

  async indexProject(
    _project: string,
    { force = false }: { force?: boolean } = {},
  ): Promise<IndexReport> {
    expect(force).toBe(true);
    return IndexReport.parse({ project_id: _project, duration_ms: 1 });
  }

  async searchCode(
    query: string,
    { projects, limit = 8 }: { projects: readonly string[]; limit?: number },
  ): Promise<SearchResponse> {
    this.searchCalls.push(projects.length);
    return SearchResponse.parse({
      query,
      hits: Array.from({ length: Math.min(limit, projects.length) }, (_, index) =>
        SearchHit.parse({
          chunk_id: `chunk-${index}`,
          project_id: projects[index % projects.length],
          project_name: projects[index % projects.length],
          path: "mod.py",
          language: "python",
          kind: "function",
          start_line: index,
          end_line: index,
          score: 1 - index * 0.01,
          snippet: "",
        }),
      ),
    });
  }
}

class FlakySearchApp extends SearchApp {
  override async searchCode(
    query: string,
    options: { projects: readonly string[]; limit?: number },
  ): Promise<SearchResponse> {
    const response = await super.searchCode(query, options);
    if (this.searchCalls.length % 2 === 0) {
      return SearchResponse.parse({ query: response.query, hits: [...response.hits].reverse() });
    }
    return response;
  }
}

describe("search benchmark", () => {
  test("measures one, eight, and fifty project scopes", async () => {
    const roots = Array.from({ length: 50 }, (_, index) => path.join(temporary, `p${index}`));
    const app = new SearchApp();
    const payload = await runSearchBenchmark(app, roots);
    expect(payload.schema_version).toBe(1);
    expect(payload.projects).toBe(50);
    expect(Object.keys(payload.scopes as object)).toEqual(["1", "8", "50"]);
    expect(app.searchCalls).toEqual([1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 50, 50, 50, 50, 50]);
    expect(SEARCH_ITERATIONS).toBe(3);
  });

  test("caps scopes to available projects", async () => {
    const app = new SearchApp();
    const payload = await runSearchBenchmark(
      app,
      Array.from({ length: 3 }, (_, index) => path.join(temporary, `p${index}`)),
    );
    expect(Object.keys(payload.scopes as object)).toEqual(["1", "3"]);
  });

  test("reports non-deterministic ranking", async () => {
    const app = new FlakySearchApp();
    const payload = await runSearchBenchmark(
      app,
      [path.join(temporary, "p0"), path.join(temporary, "p1")],
      {
        iterations: 1,
      },
    );
    expect((payload.scopes as Record<string, { deterministic: boolean }>)["2"]?.deterministic).toBe(
      false,
    );
  });
});

class TopicPrecisionEmbedder implements Embedder {
  readonly modelId = "test/precision";
  readonly dimension = 16;

  #vector(topic: number, jitter: number): number[] {
    const vector = Array.from({ length: this.dimension }, () => 0);
    vector[topic] = 1;
    vector[8 + topic] = jitter;
    return vector;
  }

  embedPassages(texts: string[]): number[][] {
    return texts.map((_, index) =>
      this.#vector(index % RETRIEVAL_TOPICS.length, ((index % 9) + 1) * 0.1),
    );
  }

  embedQuery(text: string): number[] {
    const lowered = text.toLowerCase();
    const topic = RETRIEVAL_TOPICS.findIndex((terms) =>
      terms.some((term) => lowered.includes(term)),
    );
    return this.#vector(topic, 0);
  }
}

describe("precision benchmark", () => {
  test("retrieval corpus is deterministic", () => {
    const [firstCorpus, firstQueries] = buildRetrievalCorpus({ passages: 40 });
    const [secondCorpus, secondQueries] = buildRetrievalCorpus({ passages: 40 });
    expect(firstCorpus).toEqual(secondCorpus);
    expect(firstQueries).toEqual(secondQueries);
    expect(firstQueries).toHaveLength(RETRIEVAL_TOPICS.length);
  });

  test("retrieval corpus rejects too few passages", () => {
    expect(() => buildRetrievalCorpus({ passages: RETRIEVAL_TOPICS.length - 1 })).toThrow();
  });

  test("precision command rejects passages below the corpus minimum", async () => {
    try {
      await runPrecisionBenchmarkCommand(runtimePathsFromEnvironment(), {
        passages: RETRIEVAL_TOPICS.length - 1,
        iterations: 1,
        recallFloor: 0.99,
        rankFloor: 0.95,
      });
      throw new Error("expected rejection");
    } catch (error) {
      expect(isCodeIndexingError(error)).toBe(true);
      if (isCodeIndexingError(error)) expect(error.code).toBe("INVALID_CONFIGURATION");
    }
  });

  test("precision experiment contract and self-consistency", async () => {
    const report = await runPrecisionBenchmark(new TopicPrecisionEmbedder(), temporary, {
      passages: 40,
      topK: 5,
      iterations: 2,
    });
    expect(report.schema_version).toBe(1);
    expect(new Set(Object.keys(report.variants as object))).toEqual(
      new Set(["float32_exact", "float32_hnsw_sq8", "float16_exact", "float16_hnsw_sq8"]),
    );
    const baseline = (report.variants as Record<string, Record<string, unknown>>).float32_exact;
    expect(baseline?.error).toBeUndefined();
    expect(baseline?.recall_at_k).toBeCloseTo(1);
    expect(baseline?.rank_correlation).toBeCloseTo(1);
  }, 60_000);

  test("directory physical bytes ignores symlinks", () => {
    const measured = path.join(temporary, "measured");
    fs.mkdirSync(measured);
    fs.writeFileSync(path.join(measured, "real.bin"), Buffer.alloc(100, 120));
    const outside = path.join(temporary, "outside.bin");
    fs.writeFileSync(outside, Buffer.alloc(4096, 121));
    fs.symlinkSync(outside, path.join(measured, "link.bin"));
    expect(directoryPhysicalBytes(measured)).toBe(100);
  });

  test("precision report recomputes float16 recall independently", async () => {
    // The published recall must be the arithmetic it claims: rank the f32
    // reference, rank the f16-halved copy, and intersect the top-5 sets.
    const embedder = new TopicPrecisionEmbedder();
    const report = (await runPrecisionBenchmark(embedder, temporary, {
      passages: 40,
      topK: 5,
      iterations: 1,
    })) as { variants: Record<string, { recall_at_k: number }> };
    const [corpus, queries] = buildRetrievalCorpus({ passages: 40 });
    const passages = (await embedder.embedPassages(corpus.map((item) => item.content))).map(
      (vector) => [...vector],
    );
    const halve = (vector: number[]): number[] => {
      const view = new DataView(new ArrayBuffer(2 * vector.length));
      vector.forEach((value, index) => {
        view.setFloat16(index * 2, value, true);
      });
      return Array.from({ length: vector.length }, (_, index) => view.getFloat16(index * 2, true));
    };
    const normalize = (vector: number[]): number[] => {
      const length = Math.hypot(...vector);
      return vector.map((value) => value / (length === 0 ? 1 : length));
    };
    const halved = passages.map(halve);
    const normalizedPassages = passages.map(normalize);
    const halvedNormalized = halved.map(normalize);
    const top = (row: number[], table: number[][]): Set<string> => {
      const scores = table.map((candidate) =>
        row.reduce((sum, value, i) => sum + value * candidate[i]!, 0),
      );
      const order = scores
        .map((score, index) => [score, index] as const)
        .sort((left, right) => (right[0] - left[0] !== 0 ? right[0] - left[0] : left[1] - right[1]))
        .slice(0, 5);
      return new Set(order.map(([, index]) => corpus[index]?.chunk_id ?? ""));
    };
    let expected = 0;
    for (const query of queries) {
      const vector = await embedder.embedQuery(query.text);
      const reference = top(normalize([...vector]), normalizedPassages);
      const candidate = top(normalize([...vector]), halvedNormalized);
      expected += [...reference].filter((id) => candidate.has(id)).length / 5;
    }
    expected /= queries.length;
    expect(report.variants.float16_exact?.recall_at_k).toBeCloseTo(expected, 5);
  }, 60_000);
});
