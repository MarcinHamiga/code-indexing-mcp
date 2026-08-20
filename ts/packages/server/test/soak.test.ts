/** Unit tests for the Phase 9 soak comparator. */

import { describe, expect, test } from "bun:test";
import { SoakSnapshot, compareSoakSnapshots } from "../src/soak.ts";

function chunk(chunkId: string, symbol: string): Record<string, unknown> {
  return {
    chunk_id: chunkId,
    file_id: "file:0",
    path: "src/example.py",
    language: "python",
    kind: "function",
    symbol,
    qualified_symbol: symbol,
    parent_symbol: null,
    start_byte: 0,
    end_byte: 10,
    start_line: 1,
    end_line: 1,
    content: "def f():\n    pass\n",
    identifier_terms: symbol,
    part_index: 0,
    content_hash: "abc",
  };
}

function snapshot(
  build: "python" | "typescript",
  repositories: {
    name?: string;
    projectId?: string;
    chunks?: Record<string, unknown>[];
    queries?: { query: string; hits: { chunk_id: string; score: number }[] }[];
  }[],
): SoakSnapshot {
  return SoakSnapshot.parse({
    schema_version: 1,
    build,
    revision: "0".repeat(40),
    model_id: "jinaai/jina-embeddings-v2-base-code",
    repositories: repositories.map((repository) => ({
      name: repository.name ?? "repo",
      path: `/repos/${repository.name ?? "repo"}`,
      project_id: repository.projectId ?? "p-1",
      chunk_count: (repository.chunks ?? [chunk("p-1:0", "alpha")]).length,
      chunks: repository.chunks ?? [chunk("p-1:0", "alpha")],
      queries: repository.queries ?? [
        { query: "alpha", hits: [{ chunk_id: "p-1:0", score: 0.9 }] },
      ],
    })),
  });
}

describe("compareSoakSnapshots", () => {
  test("identical snapshots pass every gate", () => {
    const report = compareSoakSnapshots(
      snapshot("python", [{ name: "repo" }]),
      snapshot("typescript", [{ name: "repo" }]),
    );
    expect(report.passed).toBe(true);
    expect(report.gates.map((gate) => gate.name)).toEqual([
      "chunk_rows_identical",
      "recall_at_k",
      "rank_correlation",
    ]);
  });

  test("a field difference fails the chunk gate and is reported", () => {
    const report = compareSoakSnapshots(
      snapshot("python", [{ chunks: [chunk("p-1:0", "alpha")] }]),
      snapshot("typescript", [{ chunks: [chunk("p-1:0", "beta")] }]),
    );
    expect(report.passed).toBe(false);
    const repository = report.repositories[0]?.chunks;
    expect(repository?.field_difference_count).toBe(3);
    expect(repository?.field_difference_examples[0]?.field).toBe("symbol");
    expect(repository?.matched).toBe(0);
  });

  test("rows present on only one side are counted separately", () => {
    const report = compareSoakSnapshots(
      snapshot("python", [{ chunks: [chunk("p-1:0", "a"), chunk("p-1:1", "b")] }]),
      snapshot("typescript", [{ chunks: [chunk("p-1:0", "a"), chunk("p-1:2", "c")] }]),
    );
    const repository = report.repositories[0]?.chunks;
    expect(repository?.matched).toBe(1);
    expect(repository?.only_python_count).toBe(1);
    expect(repository?.only_python_examples).toEqual(["p-1:1"]);
    expect(repository?.only_typescript_count).toBe(1);
    expect(repository?.only_typescript_examples).toEqual(["p-1:2"]);
    expect(report.passed).toBe(false);
  });

  test("mismatched project ids fail the chunk gate", () => {
    const report = compareSoakSnapshots(
      snapshot("python", [{ projectId: "p-1" }]),
      snapshot("typescript", [{ projectId: "p-2" }]),
    );
    expect(report.repositories[0]?.chunks.project_ids_match).toBe(false);
    expect(report.passed).toBe(false);
  });

  test("mismatched model ids add a model gate failure", () => {
    const python = snapshot("python", [{ name: "repo" }]);
    const typescript = SoakSnapshot.parse({
      ...snapshot("typescript", [{ name: "repo" }]),
      model_id: "other/model",
    });
    const report = compareSoakSnapshots(python, typescript);
    expect(report.model_ids_match).toBe(false);
    expect(report.gates.map((gate) => gate.name)).toContain("model_ids_match");
    expect(report.passed).toBe(false);
  });

  test("recall drops when the candidate misses reference hits", () => {
    const queries = [
      {
        query: "alpha",
        hits: [
          { chunk_id: "p-1:0", score: 0.9 },
          { chunk_id: "p-1:1", score: 0.8 },
        ],
      },
    ];
    const report = compareSoakSnapshots(
      snapshot("python", [{ chunks: [chunk("p-1:0", "a"), chunk("p-1:1", "b")], queries }]),
      snapshot("typescript", [
        {
          chunks: [chunk("p-1:0", "a"), chunk("p-1:1", "b")],
          queries: [{ query: "alpha", hits: [{ chunk_id: "p-1:0", score: 0.9 }] }],
        },
      ]),
    );
    expect(report.repositories[0]?.ranking.queries[0]?.recall_at_k).toBe(0.5);
    expect(report.passed).toBe(false);
  });

  test("queries both builds cannot answer are skipped, not failed", () => {
    const empty = [{ query: "nothing", hits: [] as { chunk_id: string; score: number }[] }];
    const report = compareSoakSnapshots(
      snapshot("python", [{ queries: empty }]),
      snapshot("typescript", [{ queries: empty }]),
    );
    const ranking = report.repositories[0]?.ranking;
    expect(ranking?.skipped_queries).toBe(1);
    expect(ranking?.min_recall_at_k).toBe(null);
    // No comparable ranking pair exists, which the rank gate treats as a failure.
    expect(ranking?.rank_correlation).toBe(null);
    expect(report.passed).toBe(false);
  });

  test("hits the reference cannot find count as zero recall", () => {
    const report = compareSoakSnapshots(
      snapshot("python", [{ queries: [{ query: "nothing", hits: [] }] }]),
      snapshot("typescript", [
        { queries: [{ query: "nothing", hits: [{ chunk_id: "p-1:0", score: 0.5 }] }] },
      ]),
    );
    expect(report.repositories[0]?.ranking.queries[0]?.recall_at_k).toBe(0);
  });

  test("reordered rankings lower the rank correlation below the floor", () => {
    const queries = (order: string[]) => [
      {
        query: "alpha",
        hits: order.map((chunkId, index) => ({ chunk_id: chunkId, score: 1 - index * 0.1 })),
      },
    ];
    const report = compareSoakSnapshots(
      snapshot("python", [
        {
          chunks: ["a", "b", "c"].map((symbol, index) => chunk(`p-1:${String(index)}`, symbol)),
          queries: queries(["p-1:0", "p-1:1", "p-1:2"]),
        },
      ]),
      snapshot("typescript", [
        {
          chunks: ["a", "b", "c"].map((symbol, index) => chunk(`p-1:${String(index)}`, symbol)),
          queries: queries(["p-1:1", "p-1:0", "p-1:2"]),
        },
      ]),
    );
    // One adjacent swap: (a, b) discordant, (a, c) and (b, c) concordant.
    expect(report.repositories[0]?.ranking.rank_correlation).toBeCloseTo(1 / 3, 12);
    expect(report.passed).toBe(false);
  });

  test("a repository missing from one snapshot is an input error", () => {
    expect(() =>
      compareSoakSnapshots(
        snapshot("python", [{ name: "a" }, { name: "b" }]),
        snapshot("typescript", [{ name: "a" }]),
      ),
    ).toThrow("repository missing from the TypeScript snapshot: b");
  });

  test("example lists are bounded while counts stay exact", () => {
    const pythonChunks = Array.from({ length: 30 }, (_, index) =>
      chunk(`p-1:${String(index)}`, `symbol_${String(index)}`),
    );
    const report = compareSoakSnapshots(
      snapshot("python", [{ chunks: pythonChunks }]),
      snapshot("typescript", [{ chunks: pythonChunks.slice(0, 1) }]),
      { maxExamples: 5 },
    );
    const repository = report.repositories[0]?.chunks;
    expect(repository?.only_python_count).toBe(29);
    expect(repository?.only_python_examples.length).toBe(5);
  });
});
