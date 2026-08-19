import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Application } from "../src/application.ts";
import { indexSettingsFromEnvironment } from "../src/settings.ts";

const MIB = 1024 ** 2;
const SCANNER_MAX_FILE_BYTES = 1_048_576;
const OVERSHOOT_ALLOWANCE_BYTES = 256 * MIB;

function writeCorpus(root: string): number {
  const source = path.join(root, "src", "benchmark");
  fs.mkdirSync(source, { recursive: true });
  fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'benchmark'\n");
  for (let index = 0; index < 8; index += 1) {
    fs.writeFileSync(
      path.join(source, `module_${index.toString().padStart(2, "0")}.py`),
      `def operation_${index}(value: int) -> int:\n    return value * ${index + 1}\n`,
    );
  }
  const prefix = 'PAYLOAD = "';
  const suffix = '"\n';
  const target = SCANNER_MAX_FILE_BYTES - 4096;
  fs.writeFileSync(
    path.join(source, "near_cap.py"),
    prefix + "x".repeat(target - prefix.length - suffix.length) + suffix,
  );
  fs.writeFileSync(
    path.join(source, "blank_run.py"),
    `"""Blank lines around an oversized token."""\n${"\n".repeat(400)}VALUE = "${"y".repeat(12_288)}"\n`,
  );
  return fs
    .readdirSync(source)
    .reduce((total, name) => total + fs.statSync(path.join(source, name)).size, 0);
}

const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "ci-mcp-ts-memory-"));
const root = path.join(workspace, "corpus");
const cache = process.env.CODE_INDEXING_MEMORY_CACHE ?? path.join(workspace, "cache");
const data = path.join(workspace, "data");
const corpusBytes = writeCorpus(root);
const settings = indexSettingsFromEnvironment({
  ...process.env,
  CODE_INDEXING_BROKER: "off",
  CODE_INDEXING_EMBED_ACCELERATOR: "cpu",
  CODE_INDEXING_EMBED_BATCH_SIZE: "1",
  CODE_INDEXING_EMBED_MAX_TOKENS: process.env.CODE_INDEXING_EMBED_MAX_TOKENS ?? "256",
  CODE_INDEXING_INDEX_EXECUTION: "worker",
});
const application = new Application({ data, cache }, { cwd: root, settings });

try {
  const project = await application.initProject(root);
  const report = await application.indexProject(project.id, { force: true });
  const budget = report.memory_budget_bytes;
  const peak = report.peak_memory_bytes;
  const failures: string[] = [];
  if (!report.worker_used) failures.push("the real embedding worker was not used");
  if (report.errors.length > 0) failures.push(`${report.errors.length} files failed indexing`);
  if (budget === null || budget === undefined)
    failures.push("the worker reported no memory budget");
  if (peak === null || peak === undefined) failures.push("the worker reported no peak RSS");
  if (budget !== null && budget !== undefined && peak !== null && peak !== undefined) {
    const allowed = budget + OVERSHOOT_ALLOWANCE_BYTES;
    if (peak > allowed) {
      failures.push(
        `peak RSS ${Math.ceil(peak / MIB)} MiB exceeded ${Math.ceil(allowed / MIB)} MiB`,
      );
    }
  }
  const result = {
    corpus_bytes: corpusBytes,
    embedding_max_tokens: settings.embeddingMaxTokens,
    overshoot_allowance_bytes: OVERSHOOT_ALLOWANCE_BYTES,
    report,
    passed: failures.length === 0,
    failures,
  };
  const serialized = `${JSON.stringify(result, null, 2)}\n`;
  const output = process.env.CODE_INDEXING_MEMORY_REPORT;
  if (output !== undefined) {
    fs.mkdirSync(path.dirname(output), { recursive: true });
    fs.writeFileSync(output, serialized);
  }
  process.stdout.write(serialized);
  if (failures.length > 0) process.exitCode = 1;
} finally {
  await application.store.close();
  fs.rmSync(workspace, { recursive: true, force: true });
}
