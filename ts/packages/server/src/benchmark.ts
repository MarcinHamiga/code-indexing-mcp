/** Deterministic end-to-end indexing benchmarks with JSON-ready results. */

import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { connect, Index, MultiMatchQuery, Operator, rerankers, type Table } from "@lancedb/lancedb";
import { Field, FixedSizeList, Float16, Float32, Schema, Utf8 } from "apache-arrow";
import { topKRankCorrelation } from "./acceptance.ts";
import { Application, type RuntimePaths } from "./application.ts";
import type { Embedder } from "./embedding.ts";
import { CodeIndexingError } from "./errors.ts";
import type {
  IndexReport,
  MaintenanceReport,
  ProjectInfo,
  SearchResponse,
  StorageStatus,
} from "./models.ts";
import { resolvePath } from "./paths.ts";
import { indexSettingsFromEnvironment } from "./settings.ts";
import { checkoutHead } from "./update-check.ts";

export const REPEATED_EDITS = 100;
export const SEARCH_SCOPES = [1, 8, 50] as const;
export const SEARCH_ITERATIONS = 3;
export const PRECISION_TOP_K = 8;
export const PRECISION_ITERATIONS = 5;
export const DEFAULT_RECALL_FLOOR = 0.99;
export const DEFAULT_RANK_FLOOR = 0.95;

export const RETRIEVAL_TOPICS: readonly (readonly string[])[] = [
  ["authorize", "permission", "credential", "session"],
  ["invoice", "billing", "tax", "subtotal"],
  ["hybrid", "ranking", "lexical", "semantic"],
  ["partition", "journal", "rollback", "fragment"],
  ["grammar", "syntax", "node", "token"],
  ["socket", "retry", "timeout", "listener"],
  ["mutex", "atomic", "deadlock", "thread"],
  ["metric", "trace", "span", "histogram"],
];

export interface IndexBenchmarkApplication {
  initProject(root: string): Promise<ProjectInfo>;
  indexProject(project: string, options?: { force?: boolean }): Promise<IndexReport>;
  storageStatus(project?: string | null): Promise<StorageStatus>;
  maintainStorage(
    project?: string | null,
    options?: { waitForLock?: boolean },
  ): Promise<MaintenanceReport>;
}

export interface SearchBenchmarkApplication {
  initProject(root: string): Promise<ProjectInfo>;
  indexProject(project: string, options?: { force?: boolean }): Promise<IndexReport>;
  searchCode(
    query: string,
    options: { projects: readonly string[]; limit?: number },
  ): Promise<SearchResponse>;
}

export function writeBenchmarkCorpus(
  root: string,
  { files = 128, functionsPerFile = 2 }: { files?: number; functionsPerFile?: number } = {},
): number {
  if (files < 1 || functionsPerFile < 1) {
    throw new Error("benchmark corpus dimensions must be positive");
  }
  fs.mkdirSync(root, { recursive: false });
  let total = 0;
  for (let fileIndex = 0; fileIndex < files; fileIndex += 1) {
    let source = "";
    for (let functionIndex = 0; functionIndex < functionsPerFile; functionIndex += 1) {
      source +=
        `def function_${String(fileIndex).padStart(4, "0")}_${String(functionIndex).padStart(4, "0")}(value: int) -> int:\n` +
        `    return value + ${fileIndex + functionIndex}\n\n`;
    }
    const encoded = Buffer.from(source);
    fs.writeFileSync(path.join(root, `module_${String(fileIndex).padStart(4, "0")}.py`), encoded);
    total += encoded.length;
  }
  return total;
}

export interface RetrievalPassage {
  readonly chunk_id: string;
  readonly content: string;
  readonly identifier_terms: string;
}

export interface RetrievalQuery {
  readonly text: string;
  readonly relevant: readonly string[];
}

export function buildRetrievalCorpus({
  passages = 240,
}: {
  passages?: number;
} = {}): [RetrievalPassage[], RetrievalQuery[]] {
  if (passages < RETRIEVAL_TOPICS.length) {
    throw new Error(`the retrieval corpus needs at least ${RETRIEVAL_TOPICS.length} passages`);
  }
  const corpus: RetrievalPassage[] = [];
  const relevantByTopic: string[][] = RETRIEVAL_TOPICS.map(() => []);
  for (let index = 0; index < passages; index += 1) {
    const topic = index % RETRIEVAL_TOPICS.length;
    const terms = RETRIEVAL_TOPICS[topic] as readonly string[];
    const within = Math.floor(index / RETRIEVAL_TOPICS.length);
    const name = `${terms[0]}_${String(index).padStart(4, "0")}`;
    const chunkId = `precision-${String(index).padStart(6, "0")}`;
    corpus.push({
      chunk_id: chunkId,
      content:
        `def ${name}(request, context):\n` +
        `    # ${terms[1]} policy for ${terms[2]} handling\n` +
        `    validate_${terms[3]}(request, context)\n` +
        `    return audit_${terms[2]}(context)\n`,
      identifier_terms: `${name} ${terms.join(" ")} ${String(within).padStart(4, "0")}`,
    });
    relevantByTopic[topic]?.push(chunkId);
  }
  const queries = RETRIEVAL_TOPICS.map((terms, topic) => ({
    text: `where is ${terms[0]} ${terms[3]} validated`,
    relevant: relevantByTopic[topic] ?? [],
  }));
  return [corpus, queries];
}

async function storageSnapshot(
  app: IndexBenchmarkApplication,
  projectId: string,
): Promise<Record<string, unknown>> {
  const status = await app.storageStatus(projectId);
  const entry = status.projects.find((stats) => stats.project.id === projectId);
  return entry === undefined ? {} : { ...entry };
}

async function measure(
  action: () => Promise<IndexReport>,
  snapshotAfter?: () => Promise<Record<string, unknown>>,
): Promise<Record<string, unknown>> {
  const started = process.hrtime.bigint();
  const report = await action();
  const wallMs = Number(process.hrtime.bigint() - started) / 1_000_000;
  const reportedMs = report.duration_ms;
  const result: Record<string, unknown> = {
    wall_ms: round(wallMs, 3),
    reported_duration_ms: reportedMs,
    chunks_per_second:
      reportedMs > 0 ? round((report.embedded_chunks * 1000) / reportedMs, 3) : null,
    structural_records: report.staged_reference_rows,
    reference_extraction_duration_ms: report.reference_extraction_duration_ms ?? 0,
    report: { ...report },
  };
  if (snapshotAfter !== undefined) result.storage_after = await snapshotAfter();
  return result;
}

export function durationSummary(samples: readonly number[]): Record<string, unknown> {
  if (samples.length === 0) return { count: 0 };
  const ordered = [...samples].sort((left, right) => left - right);
  const decile = Math.max(1, Math.floor(samples.length / 10));
  const p95Index = Math.min(ordered.length - 1, Math.ceil((95 * ordered.length) / 100) - 1);
  return {
    count: ordered.length,
    total_ms: round(sum(ordered), 3),
    min_ms: round(ordered[0] ?? 0, 3),
    median_ms: round(median(ordered), 3),
    p95_ms: round(ordered[p95Index] ?? 0, 3),
    max_ms: round(ordered[ordered.length - 1] ?? 0, 3),
    first_decile_mean_ms: round(mean(samples.slice(0, decile)), 3),
    last_decile_mean_ms: round(mean(samples.slice(-decile)), 3),
  };
}

export async function runIndexBenchmark(
  app: IndexBenchmarkApplication,
  root: string,
): Promise<Record<string, unknown>> {
  const project = await app.initProject(root);
  const scenarios: Record<string, Record<string, unknown>> = {};
  const snapshot = () => storageSnapshot(app, project.id);
  const storageBaseline = await snapshot();

  scenarios.cold_start = await measure(
    () => app.indexProject(project.id, { force: true }),
    snapshot,
  );
  scenarios.cold_start.includes_embedder_warmup = true;
  scenarios.no_op = await measure(() => app.indexProject(project.id, { force: false }), snapshot);

  const edited = path.join(root, "module_0000.py");
  fs.appendFileSync(
    edited,
    "\ndef phase_2_single_edit_marker(value: int) -> int:\n    return value + 1\n",
  );
  scenarios.single_file_edit = await measure(
    () => app.indexProject(project.id, { force: false }),
    snapshot,
  );

  const repeatedStarted = process.hrtime.bigint();
  const perEditMs: number[] = [];
  for (let editIndex = 0; editIndex < REPEATED_EDITS; editIndex += 1) {
    fs.appendFileSync(
      edited,
      `\ndef repeated_edit_marker_${String(editIndex).padStart(4, "0")}(value: int) -> int:\n` +
        `    return value + ${editIndex}\n`,
    );
    const editStarted = process.hrtime.bigint();
    await app.indexProject(project.id, { force: false });
    perEditMs.push(Number(process.hrtime.bigint() - editStarted) / 1_000_000);
  }
  scenarios.repeated_edits = {
    wall_ms: round(Number(process.hrtime.bigint() - repeatedStarted) / 1_000_000, 3),
    edits: REPEATED_EDITS,
    per_edit_ms: durationSummary(perEditMs),
    storage_after: await snapshot(),
  };

  scenarios.forced_reindex = await measure(
    () => app.indexProject(project.id, { force: true }),
    snapshot,
  );

  const removedSingle = unlinkIfPresent(path.join(root, "module_0001.py"));
  scenarios.single_file_deletion = await measure(
    () => app.indexProject(project.id, { force: false }),
    snapshot,
  );
  scenarios.single_file_deletion.removed_files = removedSingle;

  let removedGroup = 0;
  for (let deletedIndex = 2; deletedIndex < 10; deletedIndex += 1) {
    removedGroup += unlinkIfPresent(
      path.join(root, `module_${String(deletedIndex).padStart(4, "0")}.py`),
    );
  }
  scenarios.many_file_deletions = await measure(
    () => app.indexProject(project.id, { force: false }),
    snapshot,
  );
  scenarios.many_file_deletions.removed_files = removedGroup;

  const maintenanceStarted = process.hrtime.bigint();
  const maintenance = await app.maintainStorage(project.id, { waitForLock: true });
  scenarios.post_maintenance = {
    wall_ms: round(Number(process.hrtime.bigint() - maintenanceStarted) / 1_000_000, 3),
    report: { ...maintenance },
    storage_after: await snapshot(),
  };
  return {
    schema_version: 2,
    storage_baseline: storageBaseline,
    scenarios,
  };
}

function unlinkIfPresent(target: string): number {
  try {
    fs.unlinkSync(target);
    return 1;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return 0;
    throw error;
  }
}

export async function runSearchBenchmark(
  app: SearchBenchmarkApplication,
  roots: readonly string[],
  { iterations = SEARCH_ITERATIONS, query = "function returns value" } = {},
): Promise<Record<string, unknown>> {
  if (roots.length < 1) throw new Error("the search benchmark needs at least one project");
  if (iterations < 1) throw new Error("the search benchmark needs at least one iteration");
  const projectIds: string[] = [];
  for (const root of roots) {
    writeBenchmarkCorpus(root, { files: 2, functionsPerFile: 2 });
    const project = await app.initProject(root);
    await app.indexProject(project.id, { force: true });
    projectIds.push(project.id);
  }
  const scopes = [
    ...new Set([...SEARCH_SCOPES.filter((scope) => scope <= projectIds.length), projectIds.length]),
  ].sort((left, right) => left - right);
  const scenarios: Record<string, Record<string, unknown>> = {};
  for (const scope of scopes) {
    const selected = projectIds.slice(0, scope);
    const samples: number[] = [];
    for (let index = 0; index < iterations; index += 1) {
      const started = process.hrtime.bigint();
      await app.searchCode(query, { projects: selected, limit: 8 });
      samples.push(Number(process.hrtime.bigint() - started) / 1_000_000);
    }
    const first = await app.searchCode(query, { projects: selected, limit: 8 });
    const second = await app.searchCode(query, { projects: selected, limit: 8 });
    scenarios[String(scope)] = {
      projects: selected.length,
      latency_ms: durationSummary(samples),
      deterministic:
        first.hits.map((hit) => hit.chunk_id).join("\0") ===
        second.hits.map((hit) => hit.chunk_id).join("\0"),
      top_hits: first.hits.map((hit) => ({
        project_id: hit.project_id,
        path: hit.path,
        start_line: hit.start_line,
      })),
    };
  }
  return { schema_version: 1, projects: projectIds.length, query, scopes: scenarios };
}

export function directoryPhysicalBytes(directory: string): number {
  let total = 0;
  const walk = (current: string): void => {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const candidate = path.join(current, entry.name);
      if (entry.isSymbolicLink()) continue;
      if (entry.isDirectory()) {
        walk(candidate);
        continue;
      }
      if (entry.isFile()) total += fs.statSync(candidate).size;
    }
  };
  walk(directory);
  return total;
}

function precisionSchema(dimension: number, storage: "float32" | "float16"): Schema {
  return new Schema([
    new Field("chunk_id", new Utf8()),
    new Field("content", new Utf8()),
    new Field("identifier_terms", new Utf8()),
    new Field(
      "vector",
      new FixedSizeList(
        dimension,
        new Field("item", storage === "float32" ? new Float32() : new Float16()),
      ),
    ),
  ]);
}

function toFloat16(vector: readonly number[]): number[] {
  const buffer = new ArrayBuffer(vector.length * 2);
  const view = new DataView(buffer);
  for (let index = 0; index < vector.length; index += 1) {
    view.setFloat16(index * 2, vector[index] ?? 0, true);
  }
  const out: number[] = [];
  for (let index = 0; index < vector.length; index += 1) out.push(view.getFloat16(index * 2, true));
  return out;
}

async function runPrecisionVariant(
  directory: string,
  {
    corpus,
    queryTexts,
    passageVectors,
    queryVectors,
    storage,
    exact,
    topK,
    iterations,
    groundTruth,
  }: {
    corpus: readonly RetrievalPassage[];
    queryTexts: readonly string[];
    passageVectors: readonly (readonly number[])[];
    queryVectors: readonly (readonly number[])[];
    storage: "float32" | "float16";
    exact: boolean;
    topK: number;
    iterations: number;
    groundTruth: readonly (readonly string[])[];
  },
): Promise<Record<string, unknown>> {
  const dimension = passageVectors[0]?.length ?? 0;
  const rows = corpus.map((passage, index) => {
    const vector = passageVectors[index] ?? [];
    return {
      chunk_id: passage.chunk_id,
      content: passage.content,
      identifier_terms: passage.identifier_terms,
      vector: storage === "float32" ? [...vector] : toFloat16(vector),
    };
  });
  const started = process.hrtime.bigint();
  const database = await connect(directory);
  const table = await database.createTable("chunks", rows, {
    schema: precisionSchema(dimension, storage),
  });
  const tableBuildMs = Number(process.hrtime.bigint() - started) / 1_000_000;

  const indexStarted = process.hrtime.bigint();
  for (const column of ["content", "identifier_terms"]) {
    await table.createIndex(column, {
      config: Index.fts({ lowercase: true, stem: false, removeStopWords: false }),
      replace: false,
    });
  }
  if (!exact) {
    await table.createIndex("vector", {
      config: Index.hnswSq({ distanceType: "cosine" }),
      replace: false,
    });
  }
  const indexBuildMs = Number(process.hrtime.bigint() - indexStarted) / 1_000_000;

  const resultOrders: string[][] = [];
  for (const queryVector of queryVectors) {
    const search = table
      .query()
      .nearestTo([...queryVector])
      .select(["chunk_id"])
      .limit(topK);
    if (exact) search.bypassVectorIndex();
    resultOrders.push((await search.toArray()).map((row) => String(row.chunk_id)));
  }
  const recallAtK =
    groundTruth.reduce((total, reference, index) => {
      const candidate = new Set(resultOrders[index] ?? []);
      return total + reference.filter((id) => candidate.has(id)).length;
    }, 0) /
    (groundTruth.length * topK);
  const rankCorrelation = topKRankCorrelation(groundTruth, resultOrders);

  const samples: number[] = [];
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    for (const [offset, text] of queryTexts.entries()) {
      const queryVector = queryVectors[offset] ?? [];
      const hybridStarted = process.hrtime.bigint();
      const hybrid = table
        .query()
        .nearestToText(
          new MultiMatchQuery(text, ["content", "identifier_terms"], { operator: Operator.Or }),
        )
        .nearestTo([...queryVector])
        .limit(topK)
        .select(["chunk_id"]);
      if (exact) hybrid.bypassVectorIndex();
      const reranker = await rerankers.RRFReranker.create();
      await hybrid.rerank(reranker).toArray();
      samples.push(Number(process.hrtime.bigint() - hybridStarted) / 1_000_000);
    }
  }

  const physicalBytes = directoryPhysicalBytes(directory);
  await (table as Table).optimize();
  const postOptimizeBytes = directoryPhysicalBytes(directory);
  database.close();
  return {
    storage,
    index: exact ? "exact" : "hnsw_sq8",
    table_build_ms: round(tableBuildMs, 3),
    index_build_ms: round(indexBuildMs, 3),
    recall_at_k: round(recallAtK, 6),
    rank_correlation: round(rankCorrelation, 6),
    hybrid_latency_ms: durationSummary(samples),
    physical_bytes: physicalBytes,
    post_optimize_bytes: postOptimizeBytes,
  };
}

export async function runPrecisionBenchmark(
  embedder: Embedder,
  workspace: string,
  {
    passages = 240,
    topK = PRECISION_TOP_K,
    iterations = PRECISION_ITERATIONS,
    recallFloor = DEFAULT_RECALL_FLOOR,
    rankFloor = DEFAULT_RANK_FLOOR,
  } = {},
): Promise<Record<string, unknown>> {
  const [corpus, queries] = buildRetrievalCorpus({ passages });
  const passageVectors = await Promise.resolve(
    embedder.embedPassages(corpus.map((passage) => passage.content)),
  );
  const queryVectors = await Promise.all(queries.map((query) => embedder.embedQuery(query.text)));
  const normalizedPassages = normalizeRows(passageVectors);
  const normalizedQueries = normalizeRows(queryVectors);
  const groundTruth = normalizedQueries.map((queryRow) => {
    const scores = normalizedPassages.map((passage) => dot(queryRow, passage));
    return argsortDescending(scores)
      .slice(0, topK)
      .map((index) => corpus[index]?.chunk_id ?? "");
  });
  const digest = createHash("sha256");
  for (const passage of corpus) {
    digest.update(passage.content);
    digest.update(passage.identifier_terms);
  }
  for (const query of queries) digest.update(query.text);
  const queryTexts = queries.map((query) => query.text);
  const variants: Record<string, Record<string, unknown>> = {};
  for (const storage of ["float32", "float16"] as const) {
    for (const exact of [true, false]) {
      const key = `${storage}_${exact ? "exact" : "hnsw_sq8"}`;
      try {
        variants[key] = await runPrecisionVariant(path.join(workspace, key), {
          corpus,
          queryTexts,
          passageVectors,
          queryVectors,
          storage,
          exact,
          topK,
          iterations,
          groundTruth,
        });
      } catch (error) {
        variants[key] = {
          error: `${error instanceof Error ? error.name : "Error"}: ${error instanceof Error ? error.message : String(error)}`,
        };
      }
    }
  }
  const gates = Object.fromEntries(
    Object.entries(variants).map(([name, result]) => [
      name,
      {
        recall_ok: typeof result.recall_at_k === "number" && result.recall_at_k >= recallFloor,
        rank_ok:
          typeof result.rank_correlation === "number" && result.rank_correlation >= rankFloor,
      },
    ]),
  );
  return {
    schema_version: 1,
    corpus: {
      passages: corpus.length,
      queries: queries.length,
      topics: RETRIEVAL_TOPICS.length,
      digest: digest.digest("hex"),
    },
    top_k: topK,
    iterations,
    lancedb_version: lancedbVersion(),
    thresholds: { recall_at_k: recallFloor, rank_correlation: rankFloor },
    baseline_self_recall: variants.float32_exact?.recall_at_k,
    variants,
    gates,
  };
}

function lancedbVersion(): string {
  try {
    const required = createRequire(import.meta.url)("@lancedb/lancedb/package.json") as {
      version?: string;
    };
    return required.version ?? "";
  } catch {
    return "";
  }
}

function revision(): string | null {
  return checkoutHead(path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../.."));
}

async function runInWorkspace(
  paths: RuntimePaths,
  workspace: string,
  {
    files,
    functionsPerFile,
    batchSize,
  }: { files: number; functionsPerFile: number; batchSize: number },
): Promise<Record<string, unknown>> {
  const root = path.join(workspace, "corpus");
  const sourceBytes = writeBenchmarkCorpus(root, { files, functionsPerFile });
  const settings = {
    ...indexSettingsFromEnvironment(),
    embeddingBatchSize: batchSize,
    indexExecution: "in-process" as const,
    brokerMode: "off" as const,
  };
  const app = new Application(
    { data: path.join(workspace, "data"), cache: paths.cache },
    {
      cwd: root,
      settings,
    },
  );
  const result = await runIndexBenchmark(app, root);
  result.model_id = app.embedder.modelId;
  result.embedding_backend = app.effectiveBackendSelection.descriptor.accelerator;
  result.embedding_batch_size = batchSize;
  result.corpus = { files, functions_per_file: functionsPerFile, source_bytes: sourceBytes };
  result.revision = revision();
  return result;
}

export async function runIndexBenchmarkCommand(
  paths: RuntimePaths,
  {
    files,
    functionsPerFile,
    batchSize,
    workDir,
  }: { files: number; functionsPerFile: number; batchSize: number; workDir?: string | null },
): Promise<Record<string, unknown>> {
  if (files < 1 || functionsPerFile < 1) {
    throw new CodeIndexingError(
      "INVALID_CONFIGURATION",
      "Benchmark corpus dimensions must be positive",
    );
  }
  if (!(batchSize >= 1 && batchSize <= 256)) {
    throw new CodeIndexingError(
      "INVALID_CONFIGURATION",
      "Benchmark batch size must be from 1 to 256",
    );
  }
  if (workDir !== undefined && workDir !== null) {
    const workspace = resolvePath(workDir);
    fs.mkdirSync(workspace, { recursive: true });
    if (
      fs.existsSync(path.join(workspace, "corpus")) ||
      fs.existsSync(path.join(workspace, "data"))
    ) {
      throw new CodeIndexingError(
        "INVALID_CONFIGURATION",
        `Benchmark work directory is not fresh: ${workspace}`,
      );
    }
    return runInWorkspace(paths, workspace, { files, functionsPerFile, batchSize });
  }
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "code-indexing-mcp-index-benchmark-"));
  try {
    return await runInWorkspace(paths, temporary, { files, functionsPerFile, batchSize });
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
}

export async function runSearchBenchmarkCommand(
  paths: RuntimePaths,
  {
    projects,
    iterations,
    workDir,
  }: { projects: number; iterations: number; workDir?: string | null },
): Promise<Record<string, unknown>> {
  if (!(projects >= 1 && projects <= 200)) {
    throw new CodeIndexingError(
      "INVALID_CONFIGURATION",
      "Benchmark project count must be from 1 to 200",
    );
  }
  if (!(iterations >= 1 && iterations <= 20)) {
    throw new CodeIndexingError(
      "INVALID_CONFIGURATION",
      "Benchmark iterations must be from 1 to 20",
    );
  }
  const run = async (workspace: string): Promise<Record<string, unknown>> => {
    const settings = {
      ...indexSettingsFromEnvironment(),
      indexExecution: "in-process" as const,
      brokerMode: "off" as const,
    };
    const app = new Application(
      { data: path.join(workspace, "data"), cache: paths.cache },
      {
        cwd: workspace,
        settings,
      },
    );
    const roots = Array.from({ length: projects }, (_, index) =>
      path.join(workspace, `project_${String(index).padStart(3, "0")}`),
    );
    const result = await runSearchBenchmark(app, roots, { iterations });
    result.model_id = app.embedder.modelId;
    result.embedding_backend = app.effectiveBackendSelection.descriptor.accelerator;
    result.revision = revision();
    return result;
  };
  if (workDir !== undefined && workDir !== null) {
    const workspace = resolvePath(workDir);
    fs.mkdirSync(workspace, { recursive: true });
    if (
      ["corpus", "data", "project_000"].some((name) => fs.existsSync(path.join(workspace, name)))
    ) {
      throw new CodeIndexingError(
        "INVALID_CONFIGURATION",
        `Benchmark work directory is not fresh: ${workspace}`,
      );
    }
    return run(workspace);
  }
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "code-indexing-mcp-search-benchmark-"));
  try {
    return await run(temporary);
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
}

export async function runPrecisionBenchmarkCommand(
  paths: RuntimePaths,
  {
    passages,
    iterations,
    recallFloor,
    rankFloor,
    workDir,
  }: {
    passages: number;
    iterations: number;
    recallFloor: number;
    rankFloor: number;
    workDir?: string | null;
  },
): Promise<Record<string, unknown>> {
  if (!(passages >= RETRIEVAL_TOPICS.length && passages <= 100_000)) {
    throw new CodeIndexingError(
      "INVALID_CONFIGURATION",
      `Benchmark passage count must be from ${RETRIEVAL_TOPICS.length} to 100000`,
    );
  }
  if (!(iterations >= 1 && iterations <= 20)) {
    throw new CodeIndexingError(
      "INVALID_CONFIGURATION",
      "Benchmark iterations must be from 1 to 20",
    );
  }
  if (!(recallFloor > 0 && recallFloor <= 1) || !(rankFloor > 0 && rankFloor <= 1)) {
    throw new CodeIndexingError(
      "INVALID_CONFIGURATION",
      "Benchmark gate thresholds must be within (0, 1]",
    );
  }
  const run = async (workspace: string): Promise<Record<string, unknown>> => {
    const settings = {
      ...indexSettingsFromEnvironment(),
      indexExecution: "in-process" as const,
      brokerMode: "off" as const,
    };
    const app = new Application(
      { data: path.join(workspace, "data"), cache: paths.cache },
      {
        cwd: workspace,
        settings,
      },
    );
    const result = await runPrecisionBenchmark(app.embedder, path.join(workspace, "precision"), {
      passages,
      iterations,
      recallFloor,
      rankFloor,
    });
    result.model_id = app.embedder.modelId;
    result.embedding_backend = app.effectiveBackendSelection.descriptor.accelerator;
    result.revision = revision();
    return result;
  };
  if (workDir !== undefined && workDir !== null) {
    const workspace = resolvePath(workDir);
    fs.mkdirSync(workspace, { recursive: true });
    if (["precision", "data"].some((name) => fs.existsSync(path.join(workspace, name)))) {
      throw new CodeIndexingError(
        "INVALID_CONFIGURATION",
        `Benchmark work directory is not fresh: ${workspace}`,
      );
    }
    return run(workspace);
  }
  const temporary = fs.mkdtempSync(
    path.join(os.tmpdir(), "code-indexing-mcp-precision-benchmark-"),
  );
  try {
    return await run(temporary);
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function mean(values: readonly number[]): number {
  return values.length === 0 ? 0 : sum(values) / values.length;
}

function median(ordered: readonly number[]): number {
  if (ordered.length === 0) return 0;
  const middle = Math.floor(ordered.length / 2);
  if (ordered.length % 2 === 1) return ordered[middle] ?? 0;
  return ((ordered[middle - 1] ?? 0) + (ordered[middle] ?? 0)) / 2;
}

function normalizeRows(matrix: readonly (readonly number[])[]): number[][] {
  return matrix.map((row) => {
    const norm = Math.sqrt(row.reduce((total, value) => total + value * value, 0));
    return row.map((value) => value / (norm || 1));
  });
}

function dot(left: readonly number[], right: readonly number[]): number {
  return left.reduce((total, value, index) => total + value * (right[index] ?? 0), 0);
}

function argsortDescending(scores: readonly number[]): number[] {
  return scores
    .map((score, index) => [score, index] as const)
    .sort((left, right) => (right[0] === left[0] ? left[1] - right[1] : right[0] - left[0]))
    .map(([, index]) => index);
}
