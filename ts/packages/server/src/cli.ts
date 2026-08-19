#!/usr/bin/env bun

/** Command-line interface for Code Indexing MCP administration and stdio serving. */

import path from "node:path";
import { fileURLToPath } from "node:url";
import { Command } from "commander";
import { Application, type RuntimePaths, runtimePathsFromEnvironment } from "./application.ts";
import {
  runIndexBenchmarkCommand,
  runPrecisionBenchmarkCommand,
  runSearchBenchmarkCommand,
} from "./benchmark.ts";
import {
  BrokerApplication,
  DaemonServer,
  daemonStatus,
  ensureDaemon,
  requireDaemonSupport,
} from "./daemon.ts";
import { CodeIndexingError, isCodeIndexingError } from "./errors.ts";
import { dumpJson } from "./jsonable.ts";
import { describeProgress, type IndexProgress } from "./models.ts";
import { createServer } from "./server.ts";
import { indexSettingsFromEnvironment } from "./settings.ts";
import { checkoutHead, isDisabled, notice, startBackgroundRefresh } from "./update-check.ts";

export const VERSION = "0.0.0";
const NOTIFY_COMMANDS = new Set(["init", "index", "status", "projects", "model", "storage"]);

export class ProgressPrinter {
  readonly stream: NodeJS.WriteStream;
  readonly interactive: boolean;
  #width = 0;
  #loggedAt: number | null = null;
  #loggedPhase: string | null = null;
  static readonly LOG_INTERVAL_SECONDS = 5;

  constructor(stream: NodeJS.WriteStream) {
    this.stream = stream;
    this.interactive = Boolean(stream.isTTY);
  }

  call = (progress: IndexProgress): void => {
    const line = describeProgress(progress);
    if (this.interactive) {
      this.stream.write(`\r${line.padEnd(this.#width)}`);
      this.#width = line.length;
      return;
    }
    const now = performance.now() / 1000;
    if (
      progress.phase === this.#loggedPhase &&
      this.#loggedAt !== null &&
      now - this.#loggedAt < ProgressPrinter.LOG_INTERVAL_SECONDS
    ) {
      return;
    }
    this.#loggedAt = now;
    this.#loggedPhase = progress.phase;
    this.stream.write(`${line}\n`);
  };

  clear(): void {
    if (this.interactive && this.#width > 0) {
      this.stream.write(`\r${" ".repeat(this.#width)}\r`);
      this.#width = 0;
    }
  }
}

function updateNotice(cacheDirectory: string): string | null {
  // The notice honours the disable switch even when a cache lingers.
  if (isDisabled()) return null;
  return notice(cacheDirectory);
}

function printJson(value: unknown): void {
  process.stdout.write(`${dumpJson(value, { indent: 2 })}\n`);
}

function installerStub(command: string): never {
  throw new CodeIndexingError(
    "UNSUPPORTED_RUNTIME",
    `${command} is implemented in Phase 8 of the TypeScript port`,
  );
}

/**
 * Test seam mirroring the Python suite's `monkeypatch.setattr(cli, "Application", ...)`:
 * the real constructor builds the ONNX embedder; tests substitute a tiny one.
 */
export const applicationFactory = {
  current(paths: RuntimePaths, options: { cwd?: string } = {}): Application {
    return new Application(paths, options);
  },
};

/** Test seam mirroring `monkeypatch.setattr(cli, "create_server", ...)`. */
export const serverFactory = {
  current(app: Application | BrokerApplication): { run(): Promise<void> } {
    return createServer(app);
  },
};

/** Test seam mirroring the benchmark-command monkeypatches. */
export const benchmarkCommands = {
  index: runIndexBenchmarkCommand,
  search: runSearchBenchmarkCommand,
  precision: runPrecisionBenchmarkCommand,
};

export async function main(argv: string[] = process.argv.slice(2)): Promise<number> {
  const program = new Command();
  program.name("code-indexing-mcp").description("Local MCP code indexer").enablePositionalOptions();
  program.option("--version", "show the version and exit");

  program
    .command("serve")
    .description("Run the stdio MCP server")
    .option("--direct", "Bypass the per-user daemon")
    .action(async (options: { direct?: boolean }) => {
      await runServe(options.direct === true);
    });
  program
    .command("init")
    .description("Initialize a local project marker")
    .argument("[path]")
    .option("--name <name>")
    .option("--force-new-id", undefined, false)
    .option("--allow-overlap", undefined, false);
  program
    .command("index")
    .description("Incrementally index a project")
    .argument("[project]")
    .option("--force");
  program.command("status").description("Show project index status").argument("[project]");
  program
    .command("history")
    .description("Show a project's durable indexing history, newest first")
    .argument("[project]")
    .option("--limit <n>", undefined, "20")
    .option("--cursor <cursor>");
  program
    .command("scan")
    .description("Dry-run scan inspection: what an index run would find, without writing")
    .argument("[project]")
    .option("--outcome <outcome>")
    .option("--reason <reason>")
    .option("--limit <n>", undefined, "50")
    .option("--cursor <cursor>");

  const storage = program
    .command("storage")
    .description("Inspect index storage statistics and maintenance");
  storage.command("status").argument("[project]");
  storage.command("vacuum").argument("[project]").option("--execute");

  const projects = program.command("projects").description("Manage registered projects");
  projects.command("list");
  projects.command("remove").argument("<project>");

  const model = program.command("model").description("Manage the local embedding model");
  model.command("pull");
  model.command("status");

  const benchmark = program.command("benchmark").description("Run reproducible local benchmarks");
  benchmark
    .command("index")
    .option("--files <n>", undefined, "128")
    .option("--functions-per-file <n>", undefined, "2")
    .option("--batch-size <n>", undefined, "8")
    .option("--work-dir <dir>");
  benchmark
    .command("search")
    .option("--projects <n>", undefined, "50")
    .option("--iterations <n>", undefined, "3")
    .option("--work-dir <dir>");
  benchmark
    .command("precision")
    .option("--passages <n>", undefined, "240")
    .option("--iterations <n>", undefined, "5")
    .option("--recall-floor <n>", undefined, "0.99")
    .option("--rank-floor <n>", undefined, "0.95")
    .option("--work-dir <dir>");

  const daemon = program.command("daemon").description("Manage the shared indexing daemon");
  daemon.command("run");
  daemon.command("status");
  daemon.command("stop");
  daemon.command("restart");

  program.command("configure").allowUnknownOption();
  program.command("update").allowUnknownOption();
  program.command("uninstall").allowUnknownOption();

  if (argv.includes("--version") && (argv[0] === "--version" || argv.length === 1)) {
    const head = checkoutHead(
      path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../.."),
    );
    process.stdout.write(
      `code-indexing-mcp ${VERSION} (${head === null ? "unknown" : head.slice(0, 7)})\n`,
    );
    return 0;
  }

  program.exitOverride();
  program.configureOutput({
    writeErr: (string) => {
      process.stderr.write(string);
    },
  });
  let parsed: Command;
  try {
    parsed = await program.parseAsync(["node", "code-indexing-mcp", ...argv], { from: "node" });
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "commander.helpDisplayed") {
      return 0;
    }
    // `serve` runs inside the action, so its configuration errors surface here.
    if (isCodeIndexingError(error)) {
      process.stderr.write(`${error}\n`);
      return 2;
    }
    throw error;
  }
  const command = parsed.args[0] ?? parsed.commands.find((item) => item.name() === argv[0])?.name();
  const leaf = leafCommand(parsed);
  const name = leaf?.name() ?? command;
  const paths = runtimePathsFromEnvironment();
  const refresh =
    name !== undefined && NOTIFY_COMMANDS.has(topLevel(argv))
      ? startBackgroundRefresh(paths.cache)
      : null;
  try {
    if (argv[0] === "daemon") {
      return await runDaemon(argv[1] ?? "", paths);
    }
    if (argv[0] === "benchmark") {
      return await runBenchmark(argv[1] ?? "", argv, paths);
    }
    if (argv[0] === "configure" || argv[0] === "update" || argv[0] === "uninstall") {
      installerStub(argv[0]);
    }
    if (argv[0] === "serve") return 0;
    const app = applicationFactory.current(paths, { cwd: process.cwd() });
    let result: unknown;
    if (argv[0] === "init") {
      const options = namedOptions(argv);
      const nameOption = stringOption(options.name);
      result = await app.initProject(argv[1], {
        ...(nameOption === undefined ? {} : { name: nameOption }),
        forceNewId: options.forceNewId === true,
        allowOverlap: options.allowOverlap === true,
      });
    } else if (argv[0] === "index") {
      const printer = new ProgressPrinter(process.stderr);
      try {
        result = await app.indexProject(argv[1], {
          force: argv.includes("--force"),
          onProgress: printer.call,
        });
      } finally {
        printer.clear();
      }
    } else if (argv[0] === "status") {
      result = await app.projectStatus(argv[1]);
    } else if (argv[0] === "history") {
      const options = namedOptions(argv);
      const cursor = stringOption(options.cursor);
      result = await app.indexHistory(argv[1], {
        ...(cursor === undefined ? {} : { cursor }),
        limit: options.limit === undefined ? 20 : Number(options.limit),
      });
    } else if (argv[0] === "scan") {
      const options = namedOptions(argv);
      const outcome = stringOption(options.outcome);
      const reason = stringOption(options.reason);
      const cursor = stringOption(options.cursor);
      result = await app.inspectScan(argv[1], {
        ...(outcome === undefined ? {} : { outcome }),
        ...(reason === undefined ? {} : { reason }),
        ...(cursor === undefined ? {} : { cursor }),
        limit: options.limit === undefined ? 50 : Number(options.limit),
      });
    } else if (argv[0] === "storage" && argv[1] === "status") {
      result = await app.storageStatus(argv[2]);
    } else if (argv[0] === "storage" && argv[1] === "vacuum") {
      result = await app.maintainStorage(argv[2], {
        dryRun: !argv.includes("--execute"),
        waitForLock: true,
      });
    } else if (argv[0] === "projects" && argv[1] === "list") {
      result = await app.listProjects();
    } else if (argv[0] === "projects" && argv[1] === "remove") {
      result = await app.removeProject(String(argv[2]));
    } else if (argv[0] === "model" && argv[1] === "pull") {
      await app.prepareModel();
      result = { model: app.embedder.modelId, prepared: true };
    } else if (argv[0] === "model" && argv[1] === "status") {
      result = await app.modelStatus();
    } else {
      throw new Error(`unreachable command: ${argv.join(" ")}`);
    }
    printJson(result);
    if (refresh !== null) await Promise.race([refresh, sleep(1000)]);
    const shown = updateNotice(paths.cache);
    if (shown !== null) process.stderr.write(`${shown}\n`);
    return 0;
  } catch (error) {
    if (isCodeIndexingError(error)) {
      process.stderr.write(`${error}\n`);
      return 2;
    }
    throw error;
  }
}

async function runServe(direct: boolean): Promise<void> {
  const paths = runtimePathsFromEnvironment();
  const settings = indexSettingsFromEnvironment();
  let useDaemon = !direct && settings.brokerMode !== "off";
  if (useDaemon) {
    try {
      requireDaemonSupport();
    } catch (error) {
      if (settings.brokerMode === "on") throw error;
      console.warn(
        "Unix domain sockets are unavailable on this platform; serving directly instead of via the shared daemon",
      );
      useDaemon = false;
    }
  }
  const app = useDaemon
    ? await ensureDaemon(paths)
    : applicationFactory.current(paths, { cwd: process.cwd() });
  startBackgroundRefresh(paths.cache);
  const shown = updateNotice(paths.cache);
  if (shown !== null) console.info(shown);
  await serverFactory.current(app).run();
}

async function runDaemon(subcommand: string, paths: RuntimePaths): Promise<number> {
  if (subcommand === "run") {
    startBackgroundRefresh(paths.cache);
    const shown = updateNotice(paths.cache);
    if (shown !== null) console.info(shown);
    await new DaemonServer(paths).serve();
    return 0;
  }
  if (subcommand === "status") {
    printJson(await daemonStatus(paths));
    return 0;
  }
  if (subcommand === "stop") {
    const status = await daemonStatus(paths);
    if (status.running) await new BrokerApplication(paths).stop();
    printJson({ stopped: Boolean(status.running) });
    return 0;
  }
  if (subcommand === "restart") {
    if ((await daemonStatus(paths)).running) {
      await new BrokerApplication(paths).stop();
      for (let index = 0; index < 100; index += 1) {
        if (!(await daemonStatus(paths)).running) break;
        await sleep(50);
      }
    }
    const broker = await ensureDaemon(paths);
    printJson({ restarted: true, ...(await broker.ping()) });
    return 0;
  }
  throw new Error(`unknown daemon command: ${subcommand}`);
}

async function runBenchmark(
  subcommand: string,
  argv: string[],
  paths: RuntimePaths,
): Promise<number> {
  const options = namedOptions(argv);
  const workDir = stringOption(options.workDir);
  if (subcommand === "index") {
    printJson(
      await benchmarkCommands.index(paths, {
        files: Number(options.files ?? 128),
        functionsPerFile: Number(options.functionsPerFile ?? 2),
        batchSize: Number(options.batchSize ?? 8),
        ...(workDir === undefined ? {} : { workDir }),
      }),
    );
    return 0;
  }
  if (subcommand === "search") {
    printJson(
      await benchmarkCommands.search(paths, {
        projects: Number(options.projects ?? 50),
        iterations: Number(options.iterations ?? 3),
        ...(workDir === undefined ? {} : { workDir }),
      }),
    );
    return 0;
  }
  if (subcommand === "precision") {
    printJson(
      await benchmarkCommands.precision(paths, {
        passages: Number(options.passages ?? 240),
        iterations: Number(options.iterations ?? 5),
        recallFloor: Number(options.recallFloor ?? 0.99),
        rankFloor: Number(options.rankFloor ?? 0.95),
        ...(workDir === undefined ? {} : { workDir }),
      }),
    );
    return 0;
  }
  throw new Error(`unknown benchmark command: ${subcommand}`);
}

function stringOption(value: string | boolean | undefined): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function namedOptions(argv: string[]): Record<string, string | boolean> {
  const options: Record<string, string | boolean> = {};
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    if (item === undefined || !item.startsWith("--")) continue;
    const key = item.slice(2).replace(/-([a-z])/g, (_, letter: string) => letter.toUpperCase());
    const next = argv[index + 1];
    if (next === undefined || next.startsWith("--")) {
      options[key] = true;
    } else {
      options[key] = next;
      index += 1;
    }
  }
  return options;
}

function leafCommand(program: Command): Command | null {
  const commands = program.commands;
  return commands[commands.length - 1] ?? null;
}

function topLevel(argv: string[]): string {
  return argv.find((item) => !item.startsWith("-")) ?? "";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

if (import.meta.main) {
  void main().then(
    (code) => {
      process.exitCode = code;
    },
    (error: unknown) => {
      if (isCodeIndexingError(error)) {
        process.stderr.write(`${error}\n`);
        process.exitCode = 2;
        return;
      }
      console.error(error);
      process.exitCode = 1;
    },
  );
}
