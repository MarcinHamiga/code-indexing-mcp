/** The pathlib semantics the port depends on. */

import { afterEach, beforeEach, expect, test } from "bun:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { expandUser, isRelativeTo, pathParts, resolvePath, sameFile } from "../src/paths.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

let root: string;

beforeEach(() => {
  root = temporaryDirectory();
});

afterEach(() => {
  removeDirectory(root);
});

test("a bare tilde expands and everything else is left alone", () => {
  expect(expandUser("~")).toBe(os.homedir());
  expect(expandUser(path.join("~", "projects"))).toBe(path.join(os.homedir(), "projects"));
  expect(expandUser("~someone/projects")).toBe("~someone/projects");
  expect(expandUser("/absolute/~/x")).toBe("/absolute/~/x");
});

test("resolving succeeds on a path that does not exist yet", () => {
  // A project marker is resolved before the directory holding it is created.
  const missing = path.join(root, "not", "here", "yet");

  expect(resolvePath(missing)).toBe(missing);
});

/**
 * Link *link* to the directory *target*, or report that this machine will not.
 *
 * Windows needs either a privilege or a junction to link a directory, so the
 * type is chosen per platform and a refusal skips rather than fails: symlink
 * handling is a real property to assert where it can be, and nothing at all
 * where the filesystem does not offer it.
 */
function linkDirectory(target: string, link: string): boolean {
  try {
    fs.symlinkSync(target, link, process.platform === "win32" ? "junction" : "dir");
    return true;
  } catch {
    return false;
  }
}

test("resolving follows symlinks in the part that does exist", () => {
  const real = path.join(root, "real");
  const link = path.join(root, "link");
  fs.mkdirSync(real);
  if (!linkDirectory(real, link)) return;

  expect(resolvePath(link)).toBe(real);
  expect(resolvePath(path.join(link, "child.txt"))).toBe(path.join(real, "child.txt"));
});

test("resolving makes a relative path absolute and normalizes it", () => {
  expect(path.isAbsolute(resolvePath("."))).toBe(true);
  expect(resolvePath(path.join(root, "a", "..", "b"))).toBe(path.join(root, "b"));
});

test("path parts count the root as one component", () => {
  const parts = pathParts(path.join(root, "a", "b"));

  expect(parts.length).toBe(pathParts(root).length + 2);
  expect(parts.at(-1)).toBe("b");
  expect(path.join(...parts)).toBe(path.join(root, "a", "b"));
});

test("repeated separators do not invent components", () => {
  expect(pathParts(`${root}//a//b`)).toEqual(pathParts(path.join(root, "a", "b")));
});

test("two spellings of one directory are the same file", () => {
  const repo = path.join(root, "repo");
  fs.mkdirSync(repo);

  expect(sameFile(repo, path.join(root, "repo", "..", "repo"))).toBe(true);
  expect(sameFile(repo, root)).toBe(false);
});

test("a symlink and its target are the same file", () => {
  const real = path.join(root, "real");
  const link = path.join(root, "link");
  fs.mkdirSync(real);
  if (!linkDirectory(real, link)) return;

  expect(sameFile(link, real)).toBe(true);
});

test("two missing paths fall back to comparing their resolved spellings", () => {
  const missing = path.join(root, "missing");

  expect(sameFile(missing, missing)).toBe(true);
  expect(sameFile(missing, path.join(root, "other"))).toBe(false);
});

test("is_relative_to is a pure component check", () => {
  expect(isRelativeTo("/a/b/c", "/a/b")).toBe(true);
  expect(isRelativeTo("/a/b", "/a/b")).toBe(true);
  expect(isRelativeTo("/a/b", "/a/b/c")).toBe(false);
  // A shared string prefix is not containment.
  expect(isRelativeTo("/a/bc", "/a/b")).toBe(false);
});
