/**
 * Run the Phase 0 spikes and summarize their verdicts.
 *
 * Each spike is a standalone program that can be run on its own -- that is the
 * point of them -- so this runner spawns them as children rather than
 * importing them, which also keeps one spike's native addons out of the next
 * spike's process.
 *
 * Usage:
 *   bun run src/run.ts              # every spike
 *   bun run src/run.ts s0 s2        # a subset
 */

import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

const SPIKES: Record<string, { file: string; title: string }> = {
  s0: { file: "s0-native-modules.ts", title: "Bun native-module matrix" },
  s1: { file: "s1-lancedb-parity.ts", title: "LanceDB Node parity" },
  s2: { file: "s2-grammar-coverage.ts", title: "Grammar coverage" },
  s3: { file: "s3-embedding-parity.ts", title: "Embedding parity" },
  s4: { file: "s4-memory-ceiling.ts", title: "Memory ceiling" },
  s5: { file: "s5-watcher-scanner.ts", title: "Watcher and scanner semantics" },
};

const requested = process.argv.slice(2).filter((argument) => !argument.startsWith("-"));
const selected = requested.length > 0 ? requested : Object.keys(SPIKES);

const verdicts: Array<{ id: string; ok: boolean }> = [];

for (const id of selected) {
  const spike = SPIKES[id];
  if (spike === undefined) {
    process.stderr.write(`unknown spike "${id}"; known: ${Object.keys(SPIKES).join(", ")}\n`);
    process.exit(2);
  }
  const result = spawnSync(process.execPath, [join(HERE, spike.file)], {
    stdio: "inherit",
  });
  verdicts.push({ id, ok: result.status === 0 });
}

process.stdout.write("\n=== Phase 0 spike summary ===\n");
for (const { id, ok } of verdicts) {
  process.stdout.write(`  ${ok ? "PASS" : "FAIL"}  ${id}  ${SPIKES[id]?.title ?? ""}\n`);
}

const failed = verdicts.filter((verdict) => !verdict.ok);
process.stdout.write(`${verdicts.length - failed.length}/${verdicts.length} spikes passed\n`);
process.exit(failed.length > 0 ? 1 : 0);
