import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { parseArgs } from "node:util";

export const DEFAULT_REPOSITORY_URL = "https://github.com/MarcinHamiga/code-indexing-mcp.git";
const ACCELERATOR_CHOICES = ["auto", "cpu", "cuda", "mlx", "webgpu", "migraphx", "coreml"];

class InstallerError extends Error {}

function expandUser(value: string): string {
  if (value === "~") return os.homedir();
  if (value.startsWith("~/") || value.startsWith("~\\")) {
    return path.join(os.homedir(), value.slice(2));
  }
  if (value.startsWith("~"))
    throw new InstallerError(`Cannot expand another user's home: ${value}`);
  return value;
}

function resolveExisting(value: string): string {
  return path.resolve(expandUser(value));
}

function defaultInstallDirectory(): string {
  const configured = process.env.CODE_INDEXING_MCP_INSTALL_DIR;
  if (configured !== undefined && configured !== "") return resolveExisting(configured);
  return path.join(os.homedir(), ".local", "share", "code-indexing-mcp");
}

function asBool(raw: string): boolean {
  return new Set(["1", "true", "yes", "on"]).has(raw.trim().toLowerCase());
}

type ParsedValues = ReturnType<typeof parseArgs>["values"];

function stringOption(values: ParsedValues, name: string): string | undefined {
  const value = values[name];
  return typeof value === "string" ? value : undefined;
}

function stringOptions(values: ParsedValues, name: string): string[] {
  const value = values[name];
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function boolOption(values: ParsedValues, name: string): boolean {
  return values[name] === true;
}

function serverExecutable(
  installDirectory: string,
  platformName: string = process.platform,
): string {
  return path.join(
    installDirectory,
    "bin",
    platformName.startsWith("win") ? "code-indexing-mcp.cmd" : "code-indexing-mcp",
  );
}

function writeServerLauncher(
  installDirectory: string,
  platformName: string = process.platform,
): string {
  const executable = serverExecutable(installDirectory, platformName);
  const entry = path.join(installDirectory, "ts", "packages", "server", "src", "cli.ts");
  const bun = process.execPath;
  fs.mkdirSync(path.dirname(executable), { recursive: true, mode: 0o700 });
  if (platformName.startsWith("win")) {
    fs.writeFileSync(executable, `@echo off\r\n"${bun}" "${entry}" %*\r\n`);
  } else {
    fs.writeFileSync(executable, `#!/bin/sh\nexec "${bun}" "${entry}" "$@"\n`, {
      mode: 0o755,
    });
    fs.chmodSync(executable, 0o755);
  }
  return executable;
}

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
  let values: ReturnType<typeof parseArgs>["values"];
  try {
    ({ values } = parseArgs({
      args: argv,
      strict: true,
      allowPositionals: false,
      options: {
        "install-dir": { type: "string" },
        "repo-url": { type: "string" },
        accelerator: { type: "string" },
        harnesses: { type: "string" },
        set: { type: "string", multiple: true },
        unset: { type: "string", multiple: true },
        "bin-dir": { type: "string" },
        "no-launcher": { type: "boolean" },
        "no-modify-path": { type: "boolean" },
        tui: { type: "boolean" },
        "no-tui": { type: "boolean" },
        "no-prompt": { type: "boolean" },
        offline: { type: "boolean" },
        help: { type: "boolean", short: "h" },
      },
    }));
  } catch (error) {
    process.stderr.write(`Error: ${error instanceof Error ? error.message : String(error)}\n`);
    return 1;
  }
  if (boolOption(values, "help")) {
    process.stdout.write(
      "Usage: install [--install-dir PATH] [--repo-url URL] [--accelerator NAME] " +
        "[--harnesses LIST] [--set NAME=VALUE] [--unset NAME] [--no-tui]\n",
    );
    return 0;
  }
  const accelerator =
    stringOption(values, "accelerator") ?? process.env.CODE_INDEXING_MCP_ACCELERATOR ?? "auto";
  if (!ACCELERATOR_CHOICES.includes(accelerator)) {
    process.stderr.write(
      `Error: --accelerator must be one of: ${ACCELERATOR_CHOICES.join(", ")}\n`,
    );
    return 1;
  }
  const installDirectory = resolveExisting(
    stringOption(values, "install-dir") ?? defaultInstallDirectory(),
  );
  const repositoryUrl =
    stringOption(values, "repo-url") ??
    process.env.CODE_INDEXING_MCP_REPO_URL ??
    DEFAULT_REPOSITORY_URL;
  try {
    const action = cloneOrUpdateRepository(repositoryUrl, installDirectory);
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

  const tail = ["--install-dir", installDirectory, "--accelerator", accelerator];
  const harnesses = stringOption(values, "harnesses");
  if (harnesses !== undefined) tail.push("--harnesses", harnesses);
  const settings = stringOptions(values, "set");
  const unsets = stringOptions(values, "unset");
  for (const pair of settings) tail.push("--set", pair);
  for (const name of unsets) tail.push("--unset", name);
  const binDirectory = stringOption(values, "bin-dir") ?? process.env.CODE_INDEXING_MCP_BIN_DIR;
  if (binDirectory !== undefined && binDirectory !== "") tail.push("--bin-dir", binDirectory);
  if (boolOption(values, "no-launcher")) tail.push("--no-launcher");
  if (boolOption(values, "no-modify-path")) tail.push("--no-modify-path");
  if (boolOption(values, "offline") || asBool(process.env.CODE_INDEXING_OFFLINE ?? "")) {
    tail.push("--offline");
  }
  if (boolOption(values, "no-prompt")) tail.push("--no-prompt");
  const scripted = Boolean(
    harnesses !== undefined ||
      settings.length > 0 ||
      unsets.length > 0 ||
      boolOption(values, "no-prompt"),
  );
  const useTui =
    boolOption(values, "tui") || (!boolOption(values, "no-tui") && !scripted && tuiAvailable());
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
