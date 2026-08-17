/**
 * S0 -- Bun native-module matrix.
 *
 * Load `@lancedb/lancedb`, `tree-sitter` plus one grammar, `onnxruntime-node`,
 * and `@parcel/watcher`, and exercise one real call through each: open a
 * table, parse a file, run an inference, watch a directory. These are all
 * N-API addons and Bun's N-API layer is the newest part of the stack, so any
 * gap found here picks that module's fallback (WASM tree-sitter, `fs.watch`)
 * or, worst case, runs that one process under Node.
 *
 * Runs under both runtimes on purpose -- `bun run src/s0-native-modules.ts`
 * and `node src/s0-native-modules.ts` produce two reports, and the Node one is
 * the control that tells a Bun N-API gap apart from a broken addon.
 */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { Spike } from "./harness.ts";
import { identityModel } from "./onnx-model.ts";

const spike = new Spike("s0", "Bun native-module matrix");
spike.header();

const scratch = mkdtempSync(join(tmpdir(), "ci-mcp-s0-"));

await spike.check("@lancedb/lancedb loads and opens a table", async () => {
  const lancedb = await import("@lancedb/lancedb");
  const connection = await lancedb.connect(join(scratch, "lance"));
  const table = await connection.createTable("probe", [
    { id: "a", vector: [0.1, 0.2, 0.3, 0.4] },
    { id: "b", vector: [0.5, 0.6, 0.7, 0.8] },
  ]);
  const rows = await table.countRows();
  const names = await connection.tableNames();
  if (rows !== 2) throw new Error(`expected 2 rows, read ${rows}`);
  return `created and reopened a table, ${rows} rows, tables=${JSON.stringify(names)}`;
});

await spike.check("tree-sitter parses and queries with a native grammar", async () => {
  const { default: Parser } = await import("tree-sitter");
  const { default: Python } = await import("tree-sitter-python");
  const parser = new Parser();
  parser.setLanguage(Python as never);
  const tree = parser.parse("def greet(name):\n    return f'hi {name}'\n");
  const root = tree.rootNode;
  if (root.hasError) throw new Error("the fixture parsed with errors");

  // A query is the operation the extractor actually performs, and it is a
  // separate native entry point from parsing.
  const query = new (
    Parser as unknown as {
      Query: new (lang: unknown, source: string) => never;
    }
  ).Query(Python, "(function_definition name: (identifier) @name)") as unknown as {
    captures: (node: unknown) => Array<{ name: string; node: { text: string } }>;
  };
  const captures = query.captures(root);
  const named = captures.map((capture) => capture.node.text);
  if (!named.includes("greet")) throw new Error(`query captured ${JSON.stringify(named)}`);
  return `parsed ${root.namedChildCount} top-level nodes and captured ${JSON.stringify(named)}`;
});

await spike.check("onnxruntime-node runs a real inference", async () => {
  const ort = await import("onnxruntime-node");
  const session = await ort.InferenceSession.create(identityModel([1, 4]));
  const input = new ort.Tensor("float32", Float32Array.from([1, 2, 3, 4]), [1, 4]);
  const output = await session.run({ x: input });
  const y = output.y;
  if (y === undefined)
    throw new Error(`missing output, got ${JSON.stringify(Object.keys(output))}`);
  const values = Array.from(y.data as Float32Array);
  if (values.join(",") !== "1,2,3,4") throw new Error(`identity returned ${values.join(",")}`);
  const providers = ort.env.versions;
  return `inference returned [${values.join(", ")}]; ort versions ${JSON.stringify(providers)}`;
});

await spike.check("@parcel/watcher delivers a filesystem event", async () => {
  const watcher = await import("@parcel/watcher");
  const watched = join(scratch, "watched");
  const { mkdirSync } = await import("node:fs");
  mkdirSync(watched, { recursive: true });

  let subscription: { unsubscribe: () => Promise<void> } | undefined;
  const seen = new Promise<string>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("no event for touched.py within 5s")), 5_000);
    void watcher
      .subscribe(watched, (error, events) => {
        if (error) {
          clearTimeout(timer);
          reject(error);
          return;
        }
        // Match the target file specifically: the directory itself can produce
        // its own event, and accepting that would let the check pass without
        // the watcher having seen the write at all.
        const match = events.find((event) => event.path.endsWith("touched.py"));
        if (match !== undefined) {
          clearTimeout(timer);
          resolve(`${match.type} ${basename(match.path)}`);
        }
      })
      .then((handle) => {
        // Write only once the subscription is live, otherwise the event races
        // the watch and the spike reports a false negative.
        subscription = handle;
        writeFileSync(join(watched, "touched.py"), "x = 1\n");
        return handle;
      })
      .catch(reject);
  });

  try {
    return `observed "${await seen}"`;
  } finally {
    await subscription?.unsubscribe();
  }
});

rmSync(scratch, { recursive: true, force: true });
spike.finish();
