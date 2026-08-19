#!/usr/bin/env bun

/**
 * The installer-facing probe executable.
 *
 * The Python build ships the same contract as `python -m
 * code_indexing_mcp.accelerator_probe`: one JSON object on stdout, the success
 * shape carrying the record fields the installer copies into `accelerator.json`
 * and the failure shape carrying a single `error` string, with a non-zero exit
 * status. The installer parses the last JSON line and quotes `error` back to
 * the operator, so nothing else may escape as a traceback.
 *
 * Unlike the Python build the probe runs in the same runtime as the server:
 * there is no second prepared environment to enter (see the migration plan
 * §5.3), so the interpreter recorded is this process's own Bun binary.
 */

import path from "node:path";
import { probeAccelerator, type AcceleratorProbeReport } from "./accelerator-probe.ts";
import { parseAccelerator } from "./backends.ts";
import { DEFAULT_DIMENSION, DEFAULT_MODEL } from "./embedding.ts";

interface ProbeOptions {
  readonly accelerator: string;
  readonly model: string;
  readonly dimension: number;
  readonly offline: boolean;
}

/** Write sink the run reports through; `process.stdout`/`process.stderr` in production. */
export interface JsonOutput {
  write(chunk: string): void;
}

/** The stdout stream carrying the JSON protocol and the stderr for diagnosis. */
export interface ProbeCliOutput {
  readonly stdout: JsonOutput;
  readonly stderr: JsonOutput;
}

const USAGE =
  "usage: code-indexing-mcp-accelerator-probe " +
  "--accelerator <accelerator> [--model <model>] [--dimension <dimension>] [--offline]";

export async function runProbeCli(
  argv: readonly string[],
  output: ProbeCliOutput = { stdout: process.stdout, stderr: process.stderr },
): Promise<number> {
  const options = parseArguments(argv);
  if (typeof options === "string") {
    output.stderr.write(`${USAGE}\n`);
    output.stderr.write(`code-indexing-mcp-accelerator-probe: error: ${options}\n`);
    return 2;
  }
  try {
    // Imported here so a broken accelerator runtime is reported as a probe
    // failure with its own message rather than as an import error before the
    // argument handling that decides how to report it.
    const { runtimePathsFromEnvironment } = await import("./application.ts");
    const report = await probeAccelerator(parseAccelerator(options.accelerator), {
      cacheDirectory: path.join(runtimePathsFromEnvironment().cache, "models"),
      offline: options.offline,
      modelId: options.model,
      dimension: options.dimension,
    });
    output.stdout.write(`${JSON.stringify(installerContract(report))}\n`);
    return 0;
  } catch (error) {
    const rendered =
      error instanceof Error ? `${error.constructor.name}: ${error.message}` : String(error);
    output.stdout.write(`${JSON.stringify({ ok: false, error: rendered })}\n`);
    return 1;
  }
}

/** The success payload in the field names the installer copies into the record. */
export function installerContract(report: AcceleratorProbeReport): Record<string, unknown> {
  return {
    ok: true,
    accelerator: report.accelerator,
    interpreter: report.interpreter,
    providers: [...report.providers],
    resolved_providers: [...report.resolvedProviders],
    runtime_version: report.runtimeVersion,
    python_version: report.pythonVersion,
    platform: report.platform,
    device: report.device,
    dimension: report.dimension,
    model_id: report.modelId,
    detail: report.detail,
  };
}

function parseArguments(argv: readonly string[]): ProbeOptions | string {
  let accelerator: string | undefined;
  let model = DEFAULT_MODEL;
  let dimension = DEFAULT_DIMENSION;
  let offline = false;
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === undefined) break;
    switch (argument) {
      case "--accelerator": {
        const value = argv[index + 1];
        if (value === undefined) return "--accelerator requires a value";
        accelerator = value;
        index += 1;
        break;
      }
      case "--model": {
        const value = argv[index + 1];
        if (value === undefined) return "--model requires a value";
        model = value;
        index += 1;
        break;
      }
      case "--dimension": {
        const value = argv[index + 1];
        if (value === undefined) return "--dimension requires a value";
        dimension = Number(value);
        if (!Number.isInteger(dimension) || dimension <= 0) {
          return `${JSON.stringify(value)} is not a valid dimension`;
        }
        index += 1;
        break;
      }
      case "--offline":
        offline = true;
        break;
      default:
        if (argument.startsWith("--")) return `unknown option: ${argument}`;
        return `unexpected argument: ${argument}`;
    }
  }
  if (accelerator === undefined) return "the following arguments are required: --accelerator";
  return { accelerator, model, dimension, offline };
}

if (import.meta.main) {
  process.exitCode = await runProbeCli(process.argv.slice(2));
}
