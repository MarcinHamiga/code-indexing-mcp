/**
 * S1 -- LanceDB Node parity.
 *
 * The blocking spike. Open an index written by the Python build; create tables
 * from Arrow schemas; build FTS + BTree indexes with the config the write path
 * needs; run a hybrid query plus a pushdown predicate. A failure here blocks
 * the migration until the JS SDK gains the API (or we contribute it upstream).
 *
 * The schemas and index configuration below are transcribed from
 * `storage.py` -- `_chunk_schema`, `ensure_indexes`, and `_hybrid_search_rows`
 * -- rather than simplified, because "parity" here means the exact
 * configuration the write path uses, not a representative one.
 *
 * The Python half of the fixture comes from `scripts/write_python_index.py`;
 * point this spike at its output with S1_PYTHON_INDEX.
 */

import { existsSync } from "node:fs";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Field, FixedSizeList, Float16, Int32, Int64, Schema, Utf8 } from "apache-arrow";
import * as lancedb from "@lancedb/lancedb";
import { Index, MultiMatchQuery, Operator, rerankers } from "@lancedb/lancedb";
import { Spike } from "./harness.ts";

const VECTOR_DIMENSION = 768;

const spike = new Spike("s1", "LanceDB Node parity");
spike.header();

const scratch = mkdtempSync(join(tmpdir(), "ci-mcp-s1-"));

/** `storage.py::_chunk_schema` with the float16 vector storage default. */
function chunkSchema(): Schema {
  return new Schema([
    new Field("chunk_id", new Utf8(), false),
    new Field("file_id", new Utf8(), false),
    new Field("path", new Utf8(), false),
    new Field("language", new Utf8(), false),
    new Field("kind", new Utf8(), false),
    new Field("symbol", new Utf8(), false),
    new Field("qualified_symbol", new Utf8(), false),
    new Field("parent_symbol", new Utf8(), false),
    new Field("start_byte", new Int64(), false),
    new Field("end_byte", new Int64(), false),
    new Field("start_line", new Int32(), false),
    new Field("end_line", new Int32(), false),
    new Field("content", new Utf8(), false),
    new Field("identifier_terms", new Utf8(), false),
    new Field("content_hash", new Utf8(), false),
    new Field("part_index", new Int32(), false),
    new Field(
      "vector",
      new FixedSizeList(VECTOR_DIMENSION, new Field("item", new Float16(), true)),
      false,
    ),
  ]);
}

function unitVector(seed: number): number[] {
  // Deterministic and normalized, so cosine distances are meaningful without
  // dragging a real embedder into a storage spike.
  const raw = Array.from({ length: VECTOR_DIMENSION }, (_, index) =>
    Math.sin((index + 1) * (seed + 1) * 0.017),
  );
  const norm = Math.sqrt(raw.reduce((total, value) => total + value * value, 0));
  return raw.map((value) => value / norm);
}

const CORPUS = [
  {
    chunk_id: "spike:0",
    path: "src/embedding/tokenizer.ts",
    symbol: "loadTokenizer",
    content:
      "export function loadTokenizer(directory: string) { return Tokenizer.fromFile(directory) }",
    identifier_terms: "load tokenizer loadTokenizer embedding",
  },
  {
    chunk_id: "spike:1",
    path: "src/storage/partition.ts",
    symbol: "openPartition",
    content:
      "export function openPartition(root: string) { return connect(join(root, 'projects')) }",
    identifier_terms: "open partition openPartition storage",
  },
  {
    chunk_id: "spike:2",
    path: "test/tokenizer.test.ts",
    symbol: "tokenizerRoundtrip",
    content: "test('tokenizer roundtrip', () => { expect(loadTokenizer(path)).toBeDefined() })",
    identifier_terms: "test tokenizer roundtrip tokenizerRoundtrip",
  },
];

function chunkRows(count: number): Record<string, unknown>[] {
  return Array.from({ length: count }, (_, index) => {
    const source = CORPUS[index % CORPUS.length];
    if (source === undefined) throw new Error("empty corpus");
    const suffix = index < CORPUS.length ? "" : `:${index}`;
    return {
      chunk_id: `${source.chunk_id}${suffix}`,
      file_id: `file:${index % CORPUS.length}`,
      path: source.path,
      language: "typescript",
      kind: "function",
      symbol: source.symbol,
      qualified_symbol: source.symbol,
      parent_symbol: "",
      start_byte: 0,
      end_byte: source.content.length,
      start_line: 1,
      end_line: 2,
      content: source.content,
      identifier_terms: source.identifier_terms,
      content_hash: `hash${index}`,
      part_index: 0,
      vector: unitVector(index),
    };
  });
}

let table: lancedb.Table | undefined;

await spike.check("create a table from the chunks Arrow schema", async () => {
  const connection = await lancedb.connect(join(scratch, "partition"));
  // createEmptyTable + add is the shape the write path needs: the schema is
  // authoritative (float16 vectors, non-nullable columns), not inferred from
  // the first batch of rows.
  table = await connection.createEmptyTable("chunks", chunkSchema());
  await table.add(chunkRows(CORPUS.length));
  const rows = await table.countRows();
  const stored = await table.schema();
  const vector = stored.fields.find((field) => field.name === "vector");
  const type = vector?.type.toString() ?? "missing";
  if (!type.toLowerCase().includes("float16")) {
    throw new Error(`vector column came back as ${type}, expected a float16 fixed-size list`);
  }
  return `${rows} rows; vector column is ${type}`;
});

await spike.check("build FTS indexes with the write path's configuration", async () => {
  if (table === undefined) throw new Error("table was not created");
  for (const column of ["content", "identifier_terms"]) {
    // storage.py: FTS(lower_case=True, stem=False, remove_stop_words=False).
    await table.createIndex(column, {
      config: Index.fts({
        lowercase: true,
        stem: false,
        removeStopWords: false,
      }),
      replace: false,
    });
  }
  const built = await table.listIndices();
  const names = built.map((index) => `${index.columns.join("+")}:${index.indexType}`);
  const covered = new Set(built.flatMap((index) => index.columns));
  if (!covered.has("content") || !covered.has("identifier_terms")) {
    throw new Error(`FTS indexes missing, saw ${JSON.stringify(names)}`);
  }
  return `built ${JSON.stringify(names)}`;
});

await spike.check("build BTree indexes on the scalar filter columns", async () => {
  if (table === undefined) throw new Error("table was not created");
  for (const column of ["file_id", "language", "path", "symbol"]) {
    await table.createIndex(column, {
      config: Index.btree(),
      replace: false,
    });
  }
  const built = await table.listIndices();
  const btrees = built.filter((index) => index.indexType.toLowerCase().includes("btree"));
  if (btrees.length < 4) {
    throw new Error(`expected 4 BTree indexes, built ${btrees.length}`);
  }
  return `built ${btrees.length} BTree indexes on ${btrees.flatMap((i) => i.columns).join(", ")}`;
});

await spike.check("run a hybrid query with the write path's multi-column FTS", async () => {
  if (table === undefined) throw new Error("table was not created");
  const reranker = await rerankers.RRFReranker.create();
  const results = await table
    .query()
    // Order is load-bearing and differs from Python's: `nearestToText` is
    // declared on `Query` and `nearestTo` returns a `VectorQuery`, so the text
    // leg must be attached first. Reversing them is a TypeError at runtime,
    // not a type error, because `VectorQuery` simply has no such method.
    .nearestToText(
      // storage.py spans both single-column FTS indexes with a MultiMatchQuery;
      // a plain string would silently search only one of them.
      new MultiMatchQuery("tokenizer", ["content", "identifier_terms"], {
        operator: Operator.Or,
      }),
    )
    .nearestTo(unitVector(0))
    .rerank(reranker)
    .select([
      "chunk_id",
      "path",
      "language",
      "kind",
      "symbol",
      "qualified_symbol",
      "parent_symbol",
      "start_line",
      "end_line",
      "content",
    ])
    .limit(5)
    .toArray();
  if (results.length === 0) throw new Error("hybrid query returned nothing");
  const ids = results.map((row) => String(row.chunk_id));
  return `RRF-reranked ${results.length} rows: ${ids.join(", ")}`;
});

await spike.check("push a path glob predicate into the scan", async () => {
  if (table === undefined) throw new Error("table was not created");
  // path_filter.py emits exactly this shape for the glob "src/**/*.ts".
  const predicate = "(regexp_like(path, '^src/(?:[^/]+/)*[^/]*\\.ts$'))";
  const results = await table
    .query()
    .nearestTo(unitVector(0))
    .where(predicate)
    .select(["chunk_id", "path"])
    .limit(10)
    .toArray();
  const paths = results.map((row) => String(row.path));
  if (paths.length === 0) throw new Error("pushdown predicate matched nothing");
  const leaked = paths.filter((path) => !path.startsWith("src/"));
  if (leaked.length > 0) throw new Error(`predicate leaked ${JSON.stringify(leaked)}`);
  return `matched ${paths.length} rows, all under src/: ${[...new Set(paths)].join(", ")}`;
});

await spike.check("build an HnswSq cosine vector index", async () => {
  if (table === undefined) throw new Error("table was not created");
  // storage.py only builds this past 20k rows; the spike asks the cheaper
  // question of whether the JS SDK can express the same configuration at all.
  await table.add(chunkRows(512).slice(CORPUS.length));
  await table.createIndex("vector", {
    config: Index.hnswSq({ distanceType: "cosine" }),
    replace: false,
  });
  const built = await table.listIndices();
  const vectorIndex = built.find((index) => index.columns.includes("vector"));
  if (vectorIndex === undefined) throw new Error("no index on the vector column");
  return `${vectorIndex.indexType} over ${await table.countRows()} rows`;
});

await spike.check("open an index written by the Python build", async () => {
  const source = process.env.S1_PYTHON_INDEX;
  if (source === undefined || !existsSync(source)) {
    return {
      skip:
        "set S1_PYTHON_INDEX to the output of scripts/write_python_index.py " +
        "(run it with the project's own Python environment)",
    };
  }

  const registry = await lancedb.connect(join(source, "registry"));
  const projects = await registry.openTable("projects");
  const registered = await projects.query().limit(10).toArray();
  const first = registered[0];
  if (first === undefined) throw new Error("the Python registry table is empty");

  const partition = await lancedb.connect(join(source, "projects", String(first.id)));
  const pythonChunks = await partition.openTable("chunks");
  const rows = await pythonChunks.countRows();
  const stored = await pythonChunks.schema();
  const vector = stored.fields.find((field) => field.name === "vector");

  // The Python build's own FTS index, queried through the JS SDK: this is the
  // half of D1 that decides whether migrated installs keep their indexes.
  const hits = await pythonChunks
    .query()
    .fullTextSearch(
      new MultiMatchQuery("tokenizer", ["content", "identifier_terms"], {
        operator: Operator.Or,
      }),
    )
    .select(["chunk_id", "path"])
    .limit(5)
    .toArray();
  if (hits.length === 0) {
    throw new Error("Python-built FTS index returned no hits through the JS SDK");
  }

  const indices = await pythonChunks.listIndices();
  return (
    `read ${rows} Python-written rows (vector ${vector?.type.toString() ?? "?"}), ` +
    `${indices.length} Python-built indexes visible, ` +
    `FTS returned ${hits.map((row) => String(row.chunk_id)).join(", ")}`
  );
});

rmSync(scratch, { recursive: true, force: true });
spike.finish();
