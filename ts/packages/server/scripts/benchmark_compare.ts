/** Run both builds' benchmarks and compare them: the Phase 9 §11 CI gate.
 *
 * Drives the Python CLI through `uv run` and the TypeScript CLI through Bun,
 * parses the JSON each prints, and applies the migration plan's performance
 * bounds: within 15% on index time, no regression (plus a noise allowance) on
 * search latency. Both sides share the ambient `CODE_INDEXING_CACHE_DIR`, so
 * one model download serves the whole comparison.
 *
 * Usage, from the repository root:
 *
 *     bun ts/packages/server/scripts/benchmark_compare.ts [--output report.json]
 *
 * Work directories live under a temporary root removed afterwards; pass
 * --work-root to keep them for inspection.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import {
  DEFAULT_INDEX_TOLERANCE,
  DEFAULT_SEARCH_TOLERANCE,
  buildBenchmarkComparisonReport,
} from "../src/benchmark-compare.ts";
import { dumpJson } from "../src/jsonable.ts";

interface Arguments {
  output: string | undefined;
  workRoot: string | undefined;
  files: number;
  functionsPerFile: number;
  batchSize: number;
  projects: number;
  iterations: number;
  indexTolerance: number;
  searchTolerance: number;
  skip: Set<string>;
}

function parseArguments(argv: string[]): Arguments {
  const values = new Map<string, string>();
  const skip = new Set<string>();
  for (let index = 0; index < argv.length; index++) {
    const flag = argv[index];
    if (flag === undefined || !flag.startsWith("--")) {
      throw new Error(`unexpected argument: ${flag ?? "<none>"}`);
    }
    if (flag === "--skip") {
      const value = argv[index + 1];
      if (value !== "index" && value !== "search") {
        throw new Error("--skip must be index or search");
      }
      skip.add(value);
      index += 1;
      continue;
    }
    const value = argv[index + 1];
    if (value === undefined) {
      throw new Error(`--${flag.slice(2)} needs a value`);
    }
    values.set(flag.slice(2), value);
    index += 1;
  }
  const number = (name: string, fallback: number): number => {
    const raw = values.get(name);
    const parsed = raw === undefined ? fallback : Number(raw);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      throw new Error(`--${name} must be a positive number`);
    }
    return parsed;
  };
  return {
    output: values.get("output"),
    workRoot: values.get("work-root"),
    files: number("files", 128),
    functionsPerFile: number("functions-per-file", 2),
    batchSize: number("batch-size", 8),
    projects: number("projects", 50),
    iterations: number("iterations", 3),
    indexTolerance: number("index-tolerance", DEFAULT_INDEX_TOLERANCE),
    searchTolerance: number("search-tolerance", DEFAULT_SEARCH_TOLERANCE),
    skip,
  };
}

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");
const cliPath = path.join(repositoryRoot, "ts", "packages", "server", "src", "cli.ts");

function run(label: string, command: string, arguments_: string[]): Record<string, unknown> {
  process.stderr.write(`${label}: ${command} ${arguments_.join(" ")}\n`);
  const result = spawnSync(command, arguments_, {
    cwd: repositoryRoot,
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  if (result.error !== undefined || result.status !== 0) {
    process.stderr.write(result.stderr ?? "");
    throw new Error(`${label} failed: ${result.error?.message ?? `exit ${String(result.status)}`}`);
  }
  try {
    return JSON.parse(result.stdout ?? "") as Record<string, unknown>;
  } catch (error) {
    throw new Error(`${label} printed no parsable benchmark JSON: ${String(error)}`);
  }
}

const arguments_ = parseArguments(process.argv.slice(2));
const ownWorkRoot = arguments_.workRoot === undefined;
const workRoot =
  arguments_.workRoot ?? fs.mkdtempSync(path.join(os.tmpdir(), "ci-mcp-benchmark-compare-"));

try {
  const pythonIndex = arguments_.skip.has("index")
    ? undefined
    : run("python index benchmark", "uv", [
        "run",
        "code-indexing-mcp",
        "benchmark",
        "index",
        "--files",
        String(arguments_.files),
        "--functions-per-file",
        String(arguments_.functionsPerFile),
        "--batch-size",
        String(arguments_.batchSize),
        "--work-dir",
        path.join(workRoot, "python-index"),
      ]);
  const typescriptIndex = arguments_.skip.has("index")
    ? undefined
    : run("typescript index benchmark", process.execPath, [
        cliPath,
        "benchmark",
        "index",
        "--files",
        String(arguments_.files),
        "--functions-per-file",
        String(arguments_.functionsPerFile),
        "--batch-size",
        String(arguments_.batchSize),
        "--work-dir",
        path.join(workRoot, "typescript-index"),
      ]);
  const pythonSearch = arguments_.skip.has("search")
    ? undefined
    : run("python search benchmark", "uv", [
        "run",
        "code-indexing-mcp",
        "benchmark",
        "search",
        "--projects",
        String(arguments_.projects),
        "--iterations",
        String(arguments_.iterations),
        "--work-dir",
        path.join(workRoot, "python-search"),
      ]);
  const typescriptSearch = arguments_.skip.has("search")
    ? undefined
    : run("typescript search benchmark", process.execPath, [
        cliPath,
        "benchmark",
        "search",
        "--projects",
        String(arguments_.projects),
        "--iterations",
        String(arguments_.iterations),
        "--work-dir",
        path.join(workRoot, "typescript-search"),
      ]);
  const report = buildBenchmarkComparisonReport({
    pythonIndex,
    typescriptIndex,
    pythonSearch,
    typescriptSearch,
    indexTolerance: arguments_.indexTolerance,
    searchTolerance: arguments_.searchTolerance,
  });
  const serialized = `${dumpJson(report, { indent: 2 })}\n`;
  if (arguments_.output !== undefined) {
    fs.mkdirSync(path.dirname(path.resolve(arguments_.output)), { recursive: true });
    fs.writeFileSync(arguments_.output, serialized, "utf8");
  }
  process.stdout.write(serialized);
  if (report.index !== null) {
    process.stderr.write(
      `${report.index.passed ? "PASS" : "FAIL"} index within +${report.targets.index_within}\n`,
    );
  }
  if (report.search !== null) {
    process.stderr.write(
      `${report.search.passed ? "PASS" : "FAIL"} search within +${report.targets.search_within}\n`,
    );
  }
  if (!report.passed) process.exitCode = 1;
} finally {
  if (ownWorkRoot) fs.rmSync(workRoot, { recursive: true, force: true });
}
