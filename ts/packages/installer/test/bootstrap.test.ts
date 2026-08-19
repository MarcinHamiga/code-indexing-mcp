import { afterEach, describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { canonicalRepositoryUrl, cloneOrUpdateRepository } from "../src/bootstrap.ts";
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
});
