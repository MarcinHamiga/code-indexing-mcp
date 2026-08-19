import { afterEach, describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { serverExecutable } from "../src/accelerator.ts";
import { updateMain } from "../src/update.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

interface Installation {
  readonly checkout: string;
  readonly publisher: string;
  readonly remote: string;
}

let workspace = "";
let previousRepositoryUrl: string | undefined;

afterEach(() => {
  if (previousRepositoryUrl === undefined) delete process.env.CODE_INDEXING_MCP_REPO_URL;
  else process.env.CODE_INDEXING_MCP_REPO_URL = previousRepositoryUrl;
  previousRepositoryUrl = undefined;
  if (workspace !== "") removeDirectory(workspace);
  workspace = "";
});

function git(directory: string, ...args: string[]): string {
  const result = spawnSync("git", args, { cwd: directory, encoding: "utf8" });
  if (result.status !== 0) throw new Error(result.stderr || result.stdout);
  return result.stdout.trim();
}

function install(): Installation {
  workspace = temporaryDirectory();
  const remote = path.join(workspace, "remote.git");
  const publisher = path.join(workspace, "publisher");
  const checkout = path.join(workspace, "checkout");
  fs.mkdirSync(publisher);
  git(workspace, "init", "--bare", "-q", remote);
  git(publisher, "init", "-q", "-b", "main");
  git(publisher, "config", "user.name", "Updater Tests");
  git(publisher, "config", "user.email", "updater@example.test");
  fs.mkdirSync(path.join(publisher, "ts"));
  fs.writeFileSync(path.join(publisher, "ts", "package.json"), '{"private":true}\n');
  fs.writeFileSync(path.join(publisher, "version.txt"), "one\n");
  git(publisher, "add", ".");
  git(publisher, "commit", "-qm", "initial");
  git(publisher, "remote", "add", "origin", remote);
  git(publisher, "push", "-qu", "origin", "main");
  git(remote, "symbolic-ref", "HEAD", "refs/heads/main");
  git(workspace, "clone", "-q", remote, checkout);
  const executable = serverExecutable(checkout);
  fs.mkdirSync(path.dirname(executable), { recursive: true });
  fs.writeFileSync(executable, "launcher");
  previousRepositoryUrl = process.env.CODE_INDEXING_MCP_REPO_URL;
  process.env.CODE_INDEXING_MCP_REPO_URL = remote;
  return { checkout, publisher, remote };
}

function publish(installation: Installation): void {
  fs.writeFileSync(path.join(installation.publisher, "version.txt"), "two\n");
  git(installation.publisher, "add", "version.txt");
  git(installation.publisher, "commit", "-qm", "update");
  git(installation.publisher, "push", "-q", "origin", "main");
}

function runCommand(
  argv: readonly string[],
  options: { cwd?: string } = {},
): { stdout: string; status: number } {
  if (argv[0] === process.execPath) return { stdout: "", status: 0 };
  const result = spawnSync(argv[0] ?? "", argv.slice(1), {
    cwd: options.cwd,
    encoding: "utf8",
  });
  if (result.status !== 0) throw new Error(result.stderr || result.stdout);
  return { stdout: result.stdout, status: result.status ?? 0 };
}

describe("self update", () => {
  test("refuses a directory without a prepared installation", async () => {
    workspace = temporaryDirectory();
    const stderr = captureStderr();
    expect(await updateMain({ installDir: workspace })).toBe(1);
    expect(stderr()).toContain("no installation found");
  });

  test("refuses a dirty checkout before syncing or handing off", async () => {
    const installation = install();
    fs.writeFileSync(path.join(installation.checkout, "version.txt"), "edited\n");
    const calls: string[][] = [];
    const stderr = captureStderr();
    expect(
      await updateMain({
        installDir: installation.checkout,
        runCommand: (argv, options) => {
          if (argv[0] === process.execPath) calls.push([...argv]);
          return runCommand(argv, options);
        },
        spawn: (argv) => {
          calls.push([...argv]);
          return 0;
        },
      }),
    ).toBe(1);
    expect(stderr()).toContain("uncommitted changes");
    expect(stderr()).toContain("version.txt");
    expect(calls).toEqual([]);
  });

  test("fast-forwards, syncs, and hands off to the new command entry", async () => {
    const installation = install();
    publish(installation);
    const calls: string[][] = [];
    expect(
      await updateMain({
        installDir: installation.checkout,
        skipAccelerator: true,
        runCommand: (argv, options) => {
          if (argv[0] === process.execPath) calls.push([...argv]);
          return runCommand(argv, options);
        },
        spawn: (argv) => {
          calls.push([...argv]);
          return 7;
        },
      }),
    ).toBe(7);
    expect(fs.readFileSync(path.join(installation.checkout, "version.txt"), "utf8")).toBe("two\n");
    expect(calls[0]).toEqual([process.execPath, "install", "--frozen-lockfile"]);
    expect(
      calls[1]?.some((item) =>
        item.endsWith(path.join("packages", "installer", "src", "command.ts")),
      ),
    ).toBe(true);
    expect(calls[1]).toContain("--skip-accelerator");
  });
});

function captureStderr(): () => string {
  let output = "";
  const original = process.stderr.write.bind(process.stderr);
  process.stderr.write = ((chunk: string | Uint8Array) => {
    output += String(chunk);
    return true;
  }) as typeof process.stderr.write;
  return () => {
    process.stderr.write = original;
    return output;
  };
}
