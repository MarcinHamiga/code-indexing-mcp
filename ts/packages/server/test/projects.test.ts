/** Project marker creation, upgrade, and resolution. */

import { afterEach, beforeEach, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { stringify as stringifyToml } from "smol-toml";
import { type ErrorCode, isCodeIndexingError } from "../src/errors.ts";
import {
  DEFAULT_INCLUDES,
  LEGACY_DEFAULT_INCLUDES_V1,
  LEGACY_DEFAULT_INCLUDES_V2,
  LEGACY_DEFAULT_INCLUDES_V3,
} from "../src/models.ts";
import { resolvePath } from "../src/paths.ts";
import {
  findProjectRoot,
  initializeProject,
  ProjectResolver,
  projectRootIdentity,
  readProjectMarker,
  rootedUnder,
  sameProjectRoot,
} from "../src/projects.ts";
import { caseInsensitiveAlias, removeDirectory, temporaryDirectory } from "./helpers.ts";

let root: string;

beforeEach(() => {
  root = temporaryDirectory();
});

afterEach(() => {
  removeDirectory(root);
});

function directory(...parts: string[]): string {
  const created = path.join(root, ...parts);
  fs.mkdirSync(created, { recursive: true });
  return created;
}

/** Assert the call is refused with the given code. */
function expectError(call: () => unknown, code: ErrorCode): void {
  let caught: unknown;
  try {
    call();
  } catch (error) {
    caught = error;
  }
  if (!isCodeIndexingError(caught)) {
    throw new Error(`expected a CodeIndexingError, got ${String(caught)}`);
  }
  expect(caught.code).toBe(code);
}

test("initializing a project creates a local marker", () => {
  const demo = directory("demo");

  const project = initializeProject(demo);

  expect(fs.existsSync(path.join(demo, ".ci-mcp", "project.toml"))).toBe(true);
  expect(fs.readFileSync(path.join(demo, ".ci-mcp", ".gitignore"), "utf8")).toBe("*\n");
  expect(project.root).toBe(resolvePath(demo));
  expect(project.name).toBe("demo");
  expect(readProjectMarker(demo)).toEqual(project);
});

test("a legacy marker directory remains readable", () => {
  const demo = directory("legacy");
  const project = initializeProject(demo);
  fs.renameSync(path.join(demo, ".ci-mcp"), path.join(demo, ".code-indexing-mcp"));

  expect(findProjectRoot(path.join(demo, "src"))).toBe(resolvePath(demo));
  expect(readProjectMarker(demo)).toEqual(project);
});

test("initializing is idempotent unless forced", () => {
  const demo = directory("demo");

  const first = initializeProject(demo);
  const second = initializeProject(demo);
  const replacement = initializeProject(demo, { forceNewId: true });

  expect(second.id).toBe(first.id);
  expect(replacement.id).not.toBe(first.id);
});

test("an explicit name overrides the directory name", () => {
  expect(initializeProject(directory("demo"), { name: "Something Else" }).name).toBe(
    "Something Else",
  );
});

test("initializing a directory that does not exist is refused", () => {
  expectError(() => initializeProject(path.join(root, "missing")), "PROJECT_NOT_FOUND");
});

test.each([
  ["v1", LEGACY_DEFAULT_INCLUDES_V1],
  ["v2", LEGACY_DEFAULT_INCLUDES_V2],
  ["v3", LEGACY_DEFAULT_INCLUDES_V3],
] as const)(
  "a %s default marker gains the current languages without rewriting the file",
  (_label, legacyIncludes) => {
    const demo = directory("demo", ".code-indexing-mcp");
    const marker = path.join(demo, "project.toml");
    const contents = stringifyToml({
      version: 1,
      id: "00000000-0000-0000-0000-000000000001",
      name: "demo",
      scan: { include: [...legacyIncludes], exclude: [], max_file_bytes: 1_048_576 },
    });
    fs.writeFileSync(marker, contents, "utf8");

    const project = readProjectMarker(path.join(root, "demo"));

    expect(project.scan.include).toEqual([...DEFAULT_INCLUDES]);
    expect(fs.readFileSync(marker, "utf8")).toBe(contents);
  },
);

test("custom marker includes are preserved", () => {
  const demo = directory("demo", ".code-indexing-mcp");
  fs.writeFileSync(
    path.join(demo, "project.toml"),
    stringifyToml({
      version: 1,
      id: "00000000-0000-0000-0000-000000000001",
      name: "demo",
      scan: { include: ["src/**/*.py"], exclude: [], max_file_bytes: 1_048_576 },
    }),
    "utf8",
  );

  expect(readProjectMarker(path.join(root, "demo")).scan.include).toEqual(["src/**/*.py"]);
});

test.each([
  ["no marker at all", null],
  ["a marker that is not TOML", "id = "],
  ["an unsupported version", stringifyToml({ version: 2, id: "x", name: "demo" })],
  ["an id that is not a uuid", stringifyToml({ version: 1, id: "not-a-uuid", name: "demo" })],
  [
    "a scan section of the wrong shape",
    stringifyToml({
      version: 1,
      id: "00000000-0000-0000-0000-000000000001",
      name: "d",
      scan: { max_file_bytes: 0 },
    }),
  ],
])("an unusable marker reads as a missing project: %s", (_label, contents) => {
  const demo = directory("demo");
  if (contents !== null) {
    fs.mkdirSync(path.join(demo, ".ci-mcp"));
    fs.writeFileSync(path.join(demo, ".ci-mcp", "project.toml"), contents, "utf8");
  }

  expectError(() => readProjectMarker(demo), "PROJECT_NOT_FOUND");
});

test("the marker written by this build is the one it reads back", () => {
  const demo = directory("demo");
  const project = initializeProject(demo);

  // Round-tripping through the file is what a second process does, so the
  // defaults must survive serialization rather than only existing in memory.
  const reread = readProjectMarker(demo);
  expect(reread.scan.max_file_bytes).toBe(1_048_576);
  expect(reread.scan.exclude).toEqual([]);
  expect(reread.version).toBe(1);
  expect(reread.id).toBe(project.id);
});

test("finding the root walks up from a nested path", () => {
  const repo = directory("repo");
  initializeProject(repo);
  const nested = directory("repo", "src", "pkg");

  expect(findProjectRoot(nested)).toBe(resolvePath(repo));
  expect(findProjectRoot(root)).toBeNull();
});

test("finding the root accepts a file and searches from its directory", () => {
  const repo = directory("repo");
  initializeProject(repo);
  const file = path.join(repo, "main.py");
  fs.writeFileSync(file, "");

  expect(findProjectRoot(file)).toBe(resolvePath(repo));
});

test("the resolver prefers an explicit project", () => {
  const oneRoot = directory("one");
  const twoRoot = directory("two");
  const one = initializeProject(oneRoot);
  const two = initializeProject(twoRoot);
  const resolver = new ProjectResolver([one, two]);

  expect(resolver.resolve({ explicit: two.id, roots: [oneRoot], cwd: oneRoot })).toEqual(two);
  expect(resolver.resolve({ explicit: twoRoot, roots: [oneRoot], cwd: oneRoot })).toEqual(two);
  expect(resolver.resolve({ explicit: two.name, roots: [oneRoot] })).toEqual(two);
});

test("the resolver uses a single marked root, then the working directory", () => {
  const repo = directory("repo");
  const nested = directory("repo", "src", "pkg");
  const project = initializeProject(repo);
  const resolver = new ProjectResolver([project]);

  expect(resolver.resolve({ roots: [repo], cwd: root })).toEqual(project);
  expect(resolver.resolve({ roots: [], cwd: nested })).toEqual(project);
});

test("the resolver rejects ambiguous roots", () => {
  const roots = [directory("one"), directory("two")];
  const projects = roots.map((each) => initializeProject(each));
  const resolver = new ProjectResolver(projects);

  expectError(() => resolver.resolve({ roots, cwd: root }), "AMBIGUOUS_PROJECT");
});

test("the resolver rejects an ambiguous explicit name", () => {
  const one = initializeProject(directory("one"), { name: "shared" });
  const two = initializeProject(directory("two"), { name: "shared" });

  expectError(
    () => new ProjectResolver([one, two]).resolve({ explicit: "shared" }),
    "AMBIGUOUS_PROJECT",
  );
});

test("the resolver reports when nothing at all was detected", () => {
  expectError(
    () => new ProjectResolver([]).resolve({ roots: [root], cwd: root }),
    "PROJECT_NOT_FOUND",
  );
  expectError(() => new ProjectResolver([]).resolve({ explicit: "nope" }), "PROJECT_NOT_FOUND");
});

test("project root identity uses filesystem identity, not the spelling", () => {
  const repo = directory("repo");

  const identity = projectRootIdentity(repo);

  expect(identity).toStartWith("inode:");
  expect(projectRootIdentity(path.join(repo, "..", "repo"))).toBe(identity);
  expect(sameProjectRoot(repo, path.join(repo, "..", "repo"))).toBe(true);
});

test("project root identity falls back to the path when the directory is gone", () => {
  const missing = path.join(root, "missing");

  expect(projectRootIdentity(missing)).toBe(`path:${resolvePath(missing)}`);
});

test("rooted_under reports strict nesting only", () => {
  const parent = directory("repo");
  const child = directory("repo", "src");
  const sibling = directory("other");

  expect(rootedUnder(parent, child)).toBe(true);
  expect(rootedUnder(child, parent)).toBe(false);
  expect(rootedUnder(parent, parent)).toBe(false);
  expect(rootedUnder(parent, sibling)).toBe(false);
  // Containment verifies the boundary directory, not the child itself: a missing
  // path whose boundary is a different directory stays unknown.
  expect(rootedUnder(parent, path.join(sibling, "missing"))).toBe(false);
  expect(rootedUnder(parent, path.join(parent, "missing", "deep"))).toBe(true);
});

test("the resolver reuses a registered project for a case-insensitive alias", () => {
  const repo = directory("repo");
  const project = initializeProject(repo);
  const alias = caseInsensitiveAlias(repo);
  if (alias === null) return; // case-sensitive filesystem; nothing to exhibit

  const resolved = new ProjectResolver([project]).resolve({ explicit: alias });

  expect(resolved).toEqual(project);
  expect(resolved.root).toBe(resolvePath(repo));
});

test("rooted_under accepts a case-insensitive parent spelling", () => {
  const parent = directory("repo");
  const child = directory("repo", "src");
  const alias = caseInsensitiveAlias(parent);
  if (alias === null) return;

  expect(rootedUnder(resolvePath(alias), child)).toBe(true);
});
