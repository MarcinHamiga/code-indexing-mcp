import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { Command } from "commander";
import { ACCELERATOR_CHOICES, writeServerLauncher } from "./accelerator.ts";
import { asBool } from "./settings-spec.ts";
import { InstallerError } from "./config-files.ts";
import { defaultInstallDirectory } from "./orchestrator.ts";
import { expandUser, resolveExisting } from "./config-files.ts";

export const DEFAULT_REPOSITORY_URL = "https://github.com/MarcinHamiga/code-indexing-mcp.git";

function runCommand(
  arguments_: string[],
  options: { cwd?: string; environment?: NodeJS.ProcessEnv } = {},
): { stdout: string; stderr: string } {
  const result = spawnSync(arguments_[0] ?? "", arguments_.slice(1), {
    cwd: options.cwd,
    encoding: "utf8",
    env:
      options.environment === undefined ? process.env : { ...process.env, ...options.environment },
  });
  if (result.error !== undefined) {
    throw new InstallerError(`Required command was not found: ${arguments_[0]}`);
  }
  if (result.status !== 0) {
    const detail = (result.stderr || result.stdout || "").trim();
    const command = arguments_.join(" ");
    throw new InstallerError(
      detail === "" ? `Command failed: ${command}` : `Command failed: ${command}\n${detail}`,
    );
  }
  return { stdout: result.stdout, stderr: result.stderr };
}

export function canonicalRepositoryUrl(url: string): string {
  let value = url.trim().replace(/\/+$/, "");
  if (value.endsWith(".git")) value = value.slice(0, -4);
  if (value.startsWith("git@github.com:")) {
    return `github.com/${value.slice("git@github.com:".length).toLowerCase()}`;
  }
  for (const prefix of ["https://github.com/", "http://github.com/", "ssh://git@github.com/"]) {
    if (value.startsWith(prefix)) return `github.com/${value.slice(prefix.length).toLowerCase()}`;
  }
  if (!value.includes("://")) return path.resolve(expandUser(value));
  return value;
}

export function cloneOrUpdateRepository(repositoryUrl: string, installDirectory: string): string {
  const git = runWhich("git");
  if (git === null) throw new InstallerError("Git is required but was not found in PATH");
  const target = resolveExisting(installDirectory);
  if (!fs.existsSync(target)) {
    fs.mkdirSync(path.dirname(target), { recursive: true, mode: 0o700 });
    runCommand([git, "clone", "--", repositoryUrl, target]);
    return "installed";
  }
  if (!fs.existsSync(path.join(target, ".git"))) {
    throw new InstallerError(`Install target exists but is not a Git repository: ${target}`);
  }
  const origin = runCommand([git, "remote", "get-url", "origin"], { cwd: target }).stdout.trim();
  if (canonicalRepositoryUrl(origin) !== canonicalRepositoryUrl(repositoryUrl)) {
    throw new InstallerError(
      `Existing checkout origin does not match the requested repository: ${origin} != ${repositoryUrl}`,
    );
  }
  const status = runCommand([git, "status", "--porcelain"], { cwd: target }).stdout;
  if (status.trim() !== "") {
    throw new InstallerError(
      `Existing checkout has uncommitted changes; update it manually: ${target}`,
    );
  }
  runCommand([git, "pull", "--ff-only"], { cwd: target });
  return "updated";
}

function runWhich(name: string): string | null {
  const result = spawnSync(process.platform === "win32" ? "where" : "which", [name], {
    encoding: "utf8",
  });
  if (result.status !== 0) return null;
  return result.stdout.trim().split(/\r?\n/)[0] ?? null;
}

export function syncEnvironment(installDirectory: string): string {
  const bun = process.execPath;
  const workspace = path.join(installDirectory, "ts");
  if (!fs.existsSync(path.join(workspace, "package.json"))) {
    throw new InstallerError(`No TypeScript workspace at ${workspace}`);
  }
  runCommand([bun, "install", "--frozen-lockfile"], { cwd: workspace });
  const command = writeServerLauncher(installDirectory);
  if (!fs.existsSync(command)) {
    throw new InstallerError(`bun install completed but the MCP executable is missing: ${command}`);
  }
  return command;
}

export function tuiAvailable(): boolean {
  const term = process.env.TERM ?? "";
  return Boolean(process.stdin.isTTY && process.stdout.isTTY && term !== "" && term !== "dumb");
}

function delegate(installDirectory: string, tail: string[]): number {
  const bun = process.execPath;
  const cli = path.join(installDirectory, "ts", "packages", "installer", "src", "cli.ts");
  const result = spawnSync(bun, [cli, ...tail], { cwd: installDirectory, stdio: "inherit" });
  if (result.error !== undefined) {
    process.stderr.write(`Error: could not launch the installer module: ${result.error.message}\n`);
    return 1;
  }
  return result.status ?? 1;
}

export function main(argv: string[] = process.argv.slice(2)): number {
  const parser = new Command();
  parser
    .name("install")
    .option("--install-dir <path>", "checkout location", defaultInstallDirectory())
    .option(
      "--repo-url <url>",
      "Git repository to clone or update",
      process.env.CODE_INDEXING_MCP_REPO_URL ?? DEFAULT_REPOSITORY_URL,
    )
    .addOption(
      parser
        .createOption("--accelerator <name>", "which accelerator to prepare")
        .choices([...ACCELERATOR_CHOICES]),
    )
    .option("--harnesses <selection>")
    .option(
      "--set <NAME=VALUE>",
      "set a managed setting",
      (value, previous: string[]) => [...previous, value],
      [],
    )
    .option(
      "--unset <NAME>",
      "remove a managed setting",
      (value, previous: string[]) => [...previous, value],
      [],
    )
    .option("--bin-dir <path>", undefined, process.env.CODE_INDEXING_MCP_BIN_DIR)
    .option("--no-launcher")
    .option("--no-modify-path")
    .option("--tui")
    .option("--no-tui")
    .option("--no-prompt")
    .option("--offline", undefined, asBool(process.env.CODE_INDEXING_OFFLINE ?? ""))
    .exitOverride();
  parser.parse(["install", ...argv], { from: "user" });
  const args = parser.opts<{
    installDir: string;
    repoUrl: string;
    accelerator?: string;
    harnesses?: string;
    set: string[];
    unset: string[];
    binDir?: string;
    launcher?: boolean;
    modifyPath?: boolean;
    tui?: boolean;
    noTui?: boolean;
    prompt?: boolean;
    offline?: boolean;
  }>();
  const installDirectory = resolveExisting(args.installDir);
  try {
    const action = cloneOrUpdateRepository(args.repoUrl, installDirectory);
    process.stdout.write(
      `${action[0]?.toUpperCase() ?? ""}${action.slice(1)} repository: ${installDirectory}\n`,
    );
    if (action === "updated")
      process.stdout.write("Next time you can run: code-indexing-mcp update\n");
    const command = syncEnvironment(installDirectory);
    process.stdout.write(`Prepared MCP executable: ${command}\n`);
  } catch (error) {
    process.stderr.write(`Error: ${error instanceof Error ? error.message : String(error)}\n`);
    return 1;
  }

  const tail = [
    "--install-dir",
    installDirectory,
    "--accelerator",
    args.accelerator ?? process.env.CODE_INDEXING_MCP_ACCELERATOR ?? "auto",
  ];
  if (args.harnesses !== undefined) tail.push("--harnesses", args.harnesses);
  for (const pair of args.set) tail.push("--set", pair);
  for (const name of args.unset) tail.push("--unset", name);
  if (args.binDir !== undefined && args.binDir !== "") tail.push("--bin-dir", args.binDir);
  if (args.launcher === false) tail.push("--no-launcher");
  if (args.modifyPath === false) tail.push("--no-modify-path");
  if (args.offline === true) tail.push("--offline");
  if (args.prompt === false) tail.push("--no-prompt");
  const scripted = Boolean(
    args.harnesses !== undefined ||
      args.set.length > 0 ||
      args.unset.length > 0 ||
      args.prompt === false,
  );
  const useTui = args.tui === true || (args.noTui !== true && !scripted && tuiAvailable());
  if (useTui) tail.push("--tui");
  const returncode = delegate(installDirectory, tail);
  if (useTui && returncode !== 0 && returncode !== 1 && returncode !== 130) {
    process.stderr.write(
      "The interactive installer failed; re-run with --no-tui for the plain interface.\n",
    );
  }
  return returncode;
}

if (import.meta.main) {
  process.exit(main());
}
