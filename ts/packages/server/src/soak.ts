/**
 * Dual-run soak comparison: the Phase 9 cutover gate of the migration plan.
 *
 * §8 of the plan defines the soak as indexing the same real repositories with
 * both builds and then (a) diffing the chunk tables row-by-row and (b)
 * comparing search rankings for a query set. Each build writes a snapshot
 * with the same shape (`scripts/write_soak_snapshot.py` for the Python build,
 * `scripts/write_soak_snapshot.ts` for this one), and this module compares
 * two snapshots and evaluates the cutover gates.
 *
 * The chunk diff is exact: extraction is deterministic and already held to
 * golden fixtures, so any field difference on a real repository is a
 * migration bug by definition. Search rankings are not exact -- the two
 * builds round vectors to float16 independently, and near-tied scores can
 * legitimately reorder -- so rankings are gated with the same metrics the
 * precision benchmark uses, `recall_at_k` and `topKRankCorrelation`, against
 * floors that leave room for tie reordering while still failing systematic
 * divergence.
 */

import { z } from "zod";
import { topKRankCorrelation } from "./acceptance.ts";

/**
 * At the default limit of 8 one swapped near-tie costs 0.125 of recall, so the
 * floor has to sit below that or it fires on float16 rounding rather than
 * divergence. The measured cross-build swap rate on the Phase 9 validation
 * soak was one tail swap in four queries; systematic divergence looks nothing
 * like that, and the rank-correlation gate catches what recall alone could
 * miss.
 */
export const DEFAULT_SOAK_RECALL_FLOOR = 0.85;
export const DEFAULT_SOAK_RANK_FLOOR = 0.9;

export const SoakHit = z.object({
  chunk_id: z.string(),
  score: z.number(),
});
export type SoakHit = z.infer<typeof SoakHit>;

export const SoakQueryResult = z.object({
  query: z.string(),
  hits: z.array(SoakHit),
});
export type SoakQueryResult = z.infer<typeof SoakQueryResult>;

/** One repository's chunk rows (the `list_chunks` columns, vector excluded). */
export const SoakRepositorySnapshot = z.object({
  name: z.string(),
  path: z.string(),
  project_id: z.string(),
  chunk_count: z.int(),
  chunks: z.array(z.record(z.string(), z.unknown())),
  queries: z.array(SoakQueryResult),
});
export type SoakRepositorySnapshot = z.infer<typeof SoakRepositorySnapshot>;

export const SoakSnapshot = z.object({
  schema_version: z.int(),
  build: z.string(),
  revision: z.string().nullable(),
  model_id: z.string(),
  repositories: z.array(SoakRepositorySnapshot),
});
export type SoakSnapshot = z.infer<typeof SoakSnapshot>;

/**
 * The repositories and queries both builds index. `path` entries are resolved
 * against the working directory of whichever writer runs, and the project id
 * is carried between runs by the `.ci-mcp/project.toml` marker the first run
 * writes into the repository -- which is why the writers must not pass
 * `--force-new-id`.
 */
export const SoakManifest = z.object({
  schema_version: z.int(),
  repositories: z.array(
    z.object({
      path: z.string(),
      name: z.string().optional(),
    }),
  ),
  queries: z.array(z.string()),
  limit: z.number().optional(),
});
export type SoakManifest = z.infer<typeof SoakManifest>;

export interface SoakChunkDifference {
  chunk_id: string;
  field: string;
  python: unknown;
  typescript: unknown;
}

export interface SoakRepositoryChunkComparison {
  name: string;
  python_project_id: string;
  typescript_project_id: string;
  project_ids_match: boolean;
  python_chunks: number;
  typescript_chunks: number;
  matched: number;
  only_python_count: number;
  only_typescript_count: number;
  field_difference_count: number;
  only_python_examples: string[];
  only_typescript_examples: string[];
  field_difference_examples: SoakChunkDifference[];
}

export interface SoakQueryRanking {
  query: string;
  reference_hits: number;
  candidate_hits: number;
  shared_hits: number;
  /** `null` when both builds returned nothing -- a query neither can answer. */
  recall_at_k: number | null;
}

export interface SoakRepositoryRanking {
  name: string;
  queries: SoakQueryRanking[];
  skipped_queries: number;
  min_recall_at_k: number | null;
  mean_recall_at_k: number | null;
  /** Kendall tau-b over the queries where both builds returned hits. */
  rank_correlation: number | null;
}

export interface SoakRepositoryComparison {
  name: string;
  chunks: SoakRepositoryChunkComparison;
  ranking: SoakRepositoryRanking;
}

export interface SoakGate {
  name: string;
  passed: boolean;
  detail: string;
}

export interface SoakComparisonReport {
  schema_version: 1;
  python_revision: string | null;
  typescript_revision: string | null;
  model_ids_match: boolean;
  thresholds: { recall_floor: number; rank_floor: number };
  repositories: SoakRepositoryComparison[];
  gates: SoakGate[];
  passed: boolean;
}

export interface SoakCompareOptions {
  recallFloor?: number;
  rankFloor?: number;
  /** Bound on each example list in the report; counts are always exact. */
  maxExamples?: number;
}

function chunkIdOf(row: Record<string, unknown>): string {
  const value = row.chunk_id;
  return typeof value === "string" ? value : "";
}

function compareChunks(
  python: SoakRepositorySnapshot,
  typescript: SoakRepositorySnapshot,
  maxExamples: number,
): SoakRepositoryChunkComparison {
  const pythonRows = new Map(
    python.chunks.map((row): [string, Record<string, unknown>] => [chunkIdOf(row), row]),
  );
  const typescriptRows = new Map(
    typescript.chunks.map((row): [string, Record<string, unknown>] => [chunkIdOf(row), row]),
  );
  const onlyPython: string[] = [];
  const onlyTypescript: string[] = [];
  const fieldDifferences: SoakChunkDifference[] = [];
  let matched = 0;
  for (const [chunkId, pythonRow] of pythonRows) {
    const typescriptRow = typescriptRows.get(chunkId);
    if (typescriptRow === undefined) {
      onlyPython.push(chunkId);
      continue;
    }
    const fields = new Set([...Object.keys(pythonRow), ...Object.keys(typescriptRow)]);
    let identical = true;
    for (const field of fields) {
      const left = pythonRow[field];
      const right = typescriptRow[field];
      if (left !== right) {
        identical = false;
        fieldDifferences.push({
          chunk_id: chunkId,
          field,
          python: left === undefined ? "<absent>" : left,
          typescript: right === undefined ? "<absent>" : right,
        });
      }
    }
    if (identical) matched += 1;
  }
  for (const chunkId of typescriptRows.keys()) {
    if (!pythonRows.has(chunkId)) onlyTypescript.push(chunkId);
  }
  return {
    name: python.name,
    python_project_id: python.project_id,
    typescript_project_id: typescript.project_id,
    project_ids_match: python.project_id === typescript.project_id,
    python_chunks: python.chunks.length,
    typescript_chunks: typescript.chunks.length,
    matched,
    only_python_count: onlyPython.length,
    only_typescript_count: onlyTypescript.length,
    field_difference_count: fieldDifferences.length,
    only_python_examples: onlyPython.slice(0, maxExamples),
    only_typescript_examples: onlyTypescript.slice(0, maxExamples),
    field_difference_examples: fieldDifferences.slice(0, maxExamples),
  };
}

function compareRankings(
  python: SoakRepositorySnapshot,
  typescript: SoakRepositorySnapshot,
): SoakRepositoryRanking {
  const candidateByQuery = new Map(
    typescript.queries.map((entry): [string, SoakQueryResult] => [entry.query, entry]),
  );
  const queries: SoakQueryRanking[] = [];
  const referenceOrders: (readonly string[])[] = [];
  const candidateOrders: (readonly string[])[] = [];
  let skipped = 0;
  for (const reference of python.queries) {
    const candidate = candidateByQuery.get(reference.query);
    if (candidate === undefined) {
      throw new Error(`query missing from the TypeScript snapshot: ${reference.query}`);
    }
    const referenceIds = reference.hits.map((hit) => hit.chunk_id);
    const candidateIds = candidate.hits.map((hit) => hit.chunk_id);
    const shared = referenceIds.filter((id) => candidateIds.includes(id)).length;
    let recall: number | null;
    if (referenceIds.length === 0) {
      // Zero reference hits means the Python build found nothing; a candidate
      // that still returns hits is divergence, not a rounding artifact.
      recall = candidateIds.length === 0 ? null : 0;
      if (candidateIds.length === 0) skipped += 1;
    } else {
      recall = shared / referenceIds.length;
      referenceOrders.push(referenceIds);
      candidateOrders.push(candidateIds);
    }
    queries.push({
      query: reference.query,
      reference_hits: referenceIds.length,
      candidate_hits: candidateIds.length,
      shared_hits: shared,
      recall_at_k: recall,
    });
  }
  const recalls = queries
    .map((entry) => entry.recall_at_k)
    .filter((value): value is number => value !== null);
  return {
    name: python.name,
    queries,
    skipped_queries: skipped,
    min_recall_at_k: recalls.length === 0 ? null : Math.min(...recalls),
    mean_recall_at_k:
      recalls.length === 0
        ? null
        : recalls.reduce((total, value) => total + value, 0) / recalls.length,
    rank_correlation:
      referenceOrders.length === 0 ? null : topKRankCorrelation(referenceOrders, candidateOrders),
  };
}

function chunkGate(repositories: SoakRepositoryComparison[]): SoakGate {
  const failures = repositories.filter(
    (entry) =>
      !entry.chunks.project_ids_match ||
      entry.chunks.only_python_count > 0 ||
      entry.chunks.only_typescript_count > 0 ||
      entry.chunks.field_difference_count > 0,
  );
  if (failures.length === 0) {
    return {
      name: "chunk_rows_identical",
      passed: true,
      detail: "every chunk row matches field for field",
    };
  }
  const details = failures.map((entry) => {
    const chunks = entry.chunks;
    const parts = [`${chunks.python_chunks} python / ${chunks.typescript_chunks} typescript rows`];
    if (!chunks.project_ids_match) {
      parts.push(`project ids ${chunks.python_project_id} vs ${chunks.typescript_project_id}`);
    }
    if (chunks.only_python_count > 0) parts.push(`${chunks.only_python_count} python-only`);
    if (chunks.only_typescript_count > 0) {
      parts.push(`${chunks.only_typescript_count} typescript-only`);
    }
    if (chunks.field_difference_count > 0) {
      parts.push(`${chunks.field_difference_count} field differences`);
    }
    return `${entry.name}: ${parts.join(", ")}`;
  });
  return { name: "chunk_rows_identical", passed: false, detail: details.join("; ") };
}

function recallGate(repositories: SoakRepositoryComparison[], floor: number): SoakGate {
  const failures: string[] = [];
  for (const entry of repositories) {
    for (const query of entry.ranking.queries) {
      if (query.recall_at_k !== null && query.recall_at_k < floor) {
        failures.push(
          `${entry.name} "${query.query}": recall ${query.recall_at_k.toFixed(3)} < ${floor}`,
        );
      }
    }
  }
  return {
    name: "recall_at_k",
    passed: failures.length === 0,
    detail: failures.length === 0 ? "every answerable query meets the floor" : failures.join("; "),
  };
}

function rankGate(repositories: SoakRepositoryComparison[], floor: number): SoakGate {
  const failures: string[] = [];
  for (const entry of repositories) {
    const correlation = entry.ranking.rank_correlation;
    if (correlation === null) {
      failures.push(`${entry.name}: no query where both builds returned hits`);
    } else if (correlation < floor) {
      failures.push(`${entry.name}: rank correlation ${correlation.toFixed(3)} < ${floor}`);
    }
  }
  return {
    name: "rank_correlation",
    passed: failures.length === 0,
    detail:
      failures.length === 0 ? "every ranking correlation meets the floor" : failures.join("; "),
  };
}

export function compareSoakSnapshots(
  python: SoakSnapshot,
  typescript: SoakSnapshot,
  options: SoakCompareOptions = {},
): SoakComparisonReport {
  const recallFloor = options.recallFloor ?? DEFAULT_SOAK_RECALL_FLOOR;
  const rankFloor = options.rankFloor ?? DEFAULT_SOAK_RANK_FLOOR;
  const maxExamples = options.maxExamples ?? 20;
  const pythonRepositories = new Map(
    python.repositories.map((entry): [string, SoakRepositorySnapshot] => [entry.name, entry]),
  );
  const typescriptRepositories = new Map(
    typescript.repositories.map((entry): [string, SoakRepositorySnapshot] => [entry.name, entry]),
  );
  const repositories: SoakRepositoryComparison[] = [];
  for (const [name, entry] of pythonRepositories) {
    const counterpart = typescriptRepositories.get(name);
    if (counterpart === undefined) {
      throw new Error(`repository missing from the TypeScript snapshot: ${name}`);
    }
    repositories.push({
      name,
      chunks: compareChunks(entry, counterpart, maxExamples),
      ranking: compareRankings(entry, counterpart),
    });
  }
  for (const name of typescriptRepositories.keys()) {
    if (!pythonRepositories.has(name)) {
      throw new Error(`repository missing from the Python snapshot: ${name}`);
    }
  }
  const modelIdsMatch = python.model_id === typescript.model_id;
  const gates: SoakGate[] = [
    chunkGate(repositories),
    recallGate(repositories, recallFloor),
    rankGate(repositories, rankFloor),
  ];
  if (!modelIdsMatch) {
    gates.push({
      name: "model_ids_match",
      passed: false,
      detail: `${python.model_id} vs ${typescript.model_id}`,
    });
  }
  return {
    schema_version: 1,
    python_revision: python.revision,
    typescript_revision: typescript.revision,
    model_ids_match: modelIdsMatch,
    thresholds: { recall_floor: recallFloor, rank_floor: rankFloor },
    repositories,
    gates,
    passed: gates.every((gate) => gate.passed),
  };
}
