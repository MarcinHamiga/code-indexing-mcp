import { afterEach, describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { canonicalRepositoryUrl, cloneOrUpdateRepository, main } from "../src/bootstrap.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

let workspace = "";

afterEach(() => {
  if (workspace !== "") removeDirectory(workspace);
  workspace = "";
});

function git(cwd: string, ...arguments_: string[]): void {
  const result = spawnSync("git", arguments_, { cwd, encoding: "utf8" });
  if (result.status !== 0) throw new Error(result.stderr || result.stdout);
}

function createRemote(root: string): [string, string] {
  const publisher = path.join(root, "publisher");
  const remote = path.join(root, "remote.git");
  fs.mkdirSync(publisher, { recursive: true });
  git(publisher, "init", "-b", "main");
  git(publisher, "config", "user.email", "dev@example.com");
  git(publisher, "config", "user.name", "dev");
  fs.writeFileSync(path.join(publisher, "version.txt"), "one\n");
  git(publisher, "add", "version.txt");
  git(publisher, "commit", "-m", "one");
  spawnSync("git", ["clone", "--bare", publisher, remote], { encoding: "utf8" });
  git(publisher, "remote", "add", "origin", remote);
  return [remote, publisher];
}

describe("bootstrap", () => {
  test("canonicalises github urls", () => {
    expect(canonicalRepositoryUrl("https://github.com/MarcinHamiga/code-indexing-mcp.git")).toBe(
      "github.com/marcinhamiga/code-indexing-mcp",
    );
    expect(canonicalRepositoryUrl("git@github.com:MarcinHamiga/code-indexing-mcp.git")).toBe(
      "github.com/marcinhamiga/code-indexing-mcp",
    );
  });

  test("clones then fast-forwards", () => {
    workspace = temporaryDirectory();
    const [remote, publisher] = createRemote(workspace);
    const checkout = path.join(workspace, "installed", "code-indexing-mcp");
    expect(cloneOrUpdateRepository(remote, checkout)).toBe("installed");
    expect(fs.readFileSync(path.join(checkout, "version.txt"), "utf8")).toBe("one\n");
    fs.writeFileSync(path.join(publisher, "version.txt"), "two\n");
    git(publisher, "add", "version.txt");
    git(publisher, "commit", "-m", "update");
    git(publisher, "push", "-u", "origin", "main");
    expect(cloneOrUpdateRepository(remote, checkout)).toBe("updated");
    expect(fs.readFileSync(path.join(checkout, "version.txt"), "utf8")).toBe("two\n");
  });

  test("the downloaded bootstrap runs without sibling modules or dependencies", () => {
    workspace = temporaryDirectory();
    const standalone = path.join(workspace, "install.ts");
    fs.copyFileSync(new URL("../src/bootstrap.ts", import.meta.url), standalone);
    const result = spawnSync(process.execPath, [standalone, "--help"], {
      cwd: workspace,
      encoding: "utf8",
    });
    expect(result.status).toBe(0);
    expect(result.stdout).toContain("Usage: install");
    expect(result.stderr).toBe("");
  });

  test("help does not clone or require an existing installation", () => {
    const stdout = captureStdout();
    expect(main(["--help"])).toBe(0);
    expect(stdout()).toContain("Usage: install");
  });

  test("accepts the root bootstrap's TypeScript runtime selector", () => {
    const stdout = captureStdout();
    expect(main(["--runtime", "ts", "--help"])).toBe(0);
    expect(stdout()).toContain("Usage: install");
  });

  test("rejects another runtime selector", () => {
    expect(main(["--runtime", "python"])).toBe(1);
  });
});

function captureStdout(): () => string {
  let output = "";
  const original = process.stdout.write.bind(process.stdout);
  process.stdout.write = ((chunk: string | Uint8Array) => {
    output += String(chunk);
    return true;
  }) as typeof process.stdout.write;
  return () => {
    process.stdout.write = original;
    return output;
  };
}
