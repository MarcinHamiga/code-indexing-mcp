/**
 * Scanner behaviour, ported from `tests/test_scanner.py`.
 *
 * Two properties this suite defends are easy to lose in a port and expensive to
 * lose in production: the *set* of files an index sees (Git's rules, nested
 * `.gitignore` stacks, hard exclusions, opaque submodules) and the *syscall
 * budget* it spends getting there -- a repository dominated by unsupported
 * files must not pay a stat per file. Both are asserted directly.
 *
 * The gitignore semantics themselves are held to a generated fixture in
 * `ignore.test.ts` rather than re-derived here.
 */

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { DEFAULT_INCLUDES, LanguageName, ScanConfig } from "../src/models.ts";
import { initializeProject } from "../src/projects.ts";
import {
  GitEnumerationError,
  HARD_EXCLUDED_DIRECTORIES,
  LANGUAGES,
  SourceScanner,
  compileSpec,
  iterWalkBatches,
  languageForExtension,
} from "../src/scanner.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

let temporary: string;

beforeEach(() => {
  temporary = temporaryDirectory();
});

afterEach(() => {
  removeDirectory(temporary);
});

function git(cwd: string, ...argv: string[]): void {
  execFileSync("git", argv, { cwd, stdio: "ignore" });
}

function write(file: string, contents: string): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, contents);
}

function paths(items: Array<{ path: string }>): string[] {
  return items.map((item) => item.path);
}

describe("enumeration", () => {
  test("honours languages, gitignore, and hard exclusions", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const project = initializeProject(root);
    write(path.join(root, "main.py"), "print('ok')\n");
    write(path.join(root, "component.tsx"), "export const App = () => <div />;\n");
    write(path.join(root, "notes.md"), "not source\n");
    write(path.join(root, "ignored.py"), "ignored = True\n");
    write(path.join(root, ".gitignore"), "ignored.py\n");
    write(path.join(root, "node_modules", "vendor.js"), "export default 1\n");

    const result = await new SourceScanner().scan(project);

    expect(result.files.map((item) => [item.path, item.language])).toEqual([
      ["component.tsx", "tsx"],
      ["main.py", "python"],
    ]);
    const reasons = new Set(result.skipped.map((item) => item.reason));
    expect(reasons.has("unsupported")).toBe(true);
    expect(reasons.has("ignored")).toBe(true);
  });

  test("honours git info/exclude", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    git(temporary, "init", "-q", root);
    const project = initializeProject(root);
    write(path.join(root, "main.py"), "value = 1\n");
    write(path.join(root, "local_only.py"), "local = True\n");
    write(path.join(root, ".git", "info", "exclude"), "local_only.py\n");

    const result = await new SourceScanner().scan(project);

    expect(paths(result.files)).toEqual(["main.py"]);
    // Git's own enumeration applies info/exclude before the scanner sees the
    // file, so an excluded file is absent from both files and skipped.
    expect(paths(result.skipped)).not.toContain("local_only.py");
  });

  test("a git repository mixes tracked and untracked files", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    git(temporary, "init", "-q", root);
    const project = initializeProject(root);
    write(path.join(root, "tracked.py"), "tracked = True\n");
    write(path.join(root, "untracked.ts"), "export const x = 1\n");
    write(path.join(root, "notes.md"), "not source\n");
    git(root, "add", "tracked.py");

    const result = await new SourceScanner().scan(project);

    expect(paths(result.files)).toEqual(["tracked.py", "untracked.ts"]);
    expect(
      result.skipped.some((item) => item.path === "notes.md" && item.reason === "unsupported"),
    ).toBe(true);
  });

  test("a tracked-but-ignored file stays eligible", async () => {
    // Git's own rule is that the index wins: a file that is ignored but was
    // force-added stays tracked, so Git enumerates it and the scanner indexes it.
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    git(temporary, "init", "-q", root);
    const project = initializeProject(root);
    write(path.join(root, ".gitignore"), "ignored.py\n");
    write(path.join(root, "ignored.py"), "value = 1\n");
    write(path.join(root, "main.py"), "value = 2\n");
    git(root, "add", "main.py");
    git(root, "add", "-f", "ignored.py");

    const result = await new SourceScanner().scan(project);

    expect(paths(result.files)).toEqual(["ignored.py", "main.py"]);
  });

  test("submodules and nested repositories are opaque", async () => {
    // Submodules (gitlinks) and nested repositories are single non-file entries
    // in Git's enumeration, so their contents are not indexed from the parent.
    const outer = path.join(temporary, "outer");
    fs.mkdirSync(outer);
    git(temporary, "init", "-q", outer);
    const sub = path.join(temporary, "sub");
    fs.mkdirSync(sub);
    git(temporary, "init", "-q", sub);
    write(path.join(sub, "sub.py"), "def sub_symbol():\n    return 1\n");
    git(sub, "add", "sub.py");
    git(sub, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "sub");
    git(outer, "-c", "protocol.file.allow=always", "submodule", "add", "-q", sub, "vendor/sub");
    const nested = path.join(outer, "nested");
    fs.mkdirSync(nested);
    git(temporary, "init", "-q", nested);
    write(path.join(nested, "nested.py"), "def nested_symbol():\n    return 2\n");
    write(path.join(outer, "main.py"), "def main_symbol():\n    return 3\n");
    const project = initializeProject(outer);

    const result = await new SourceScanner().scan(project);

    expect(paths(result.files)).toEqual(["main.py"]);
  });

  test("a git worktree is scanned as a normal checkout", async () => {
    const mainRoot = path.join(temporary, "repo");
    fs.mkdirSync(mainRoot);
    git(temporary, "init", "-q", mainRoot);
    write(path.join(mainRoot, "main.py"), "value = 1\n");
    git(mainRoot, "-c", "user.email=t@t", "-c", "user.name=t", "add", "main.py");
    git(mainRoot, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init");
    const worktree = path.join(temporary, "wt");
    git(mainRoot, "worktree", "add", "-q", worktree);
    const project = initializeProject(worktree);

    const result = await new SourceScanner().scan(project);

    expect(paths(result.files)).toEqual(["main.py"]);
  });

  test("git enumeration order is deterministic", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    git(temporary, "init", "-q", root);
    const project = initializeProject(root);
    for (const name of ["c.py", "a.py", "b.py", "a-.py", "a/file.py"]) {
      write(path.join(root, name), "value = 1\n");
    }

    const first = await new SourceScanner().scan(project);
    const second = await new SourceScanner().scan(project);

    expect(paths(first.files)).toEqual(["a/file.py", "a-.py", "a.py", "b.py", "c.py"]);
    expect(paths(first.files)).toEqual(paths(second.files));
  });

  test("only files whose suffix is supported are statted", async () => {
    // A repository dominated by unsupported files must not pay a stat per file
    // (the 100,000-file gate, at small scale): only pre-filtered candidates reach
    // the filesystem. Git mode never even passes unsupported paths to Git.
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    git(temporary, "init", "-q", root);
    const project = initializeProject(root);
    for (let index = 0; index < 100; index += 1) {
      write(path.join(root, `doc${String(index).padStart(3, "0")}.md`), "not source\n");
    }
    for (let index = 0; index < 5; index += 1) {
      write(path.join(root, `module${index}.py`), "value = 1\n");
    }
    const statted: string[] = [];
    const original = fs.promises.stat.bind(fs.promises);
    const spy = spyOn(fs.promises, "stat").mockImplementation(((...args: unknown[]) => {
      statted.push(String(args[0]));
      return original(...(args as Parameters<typeof fs.promises.stat>));
    }) as never);

    let result: Awaited<ReturnType<SourceScanner["scan"]>>;
    try {
      result = await new SourceScanner().scan(project);
    } finally {
      spy.mockRestore();
    }

    expect(result.files).toHaveLength(5);
    expect(statted.filter((item) => item.endsWith(".md"))).toEqual([]);
  });
});

describe("the streaming walk", () => {
  test("streams per directory in deterministic order", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(path.join(root, "a"), { recursive: true });
    fs.mkdirSync(path.join(root, "b"));
    const project = initializeProject(root);
    write(path.join(root, "a", "z.py"), "value = 1\n");
    write(path.join(root, "b", "a.py"), "value = 2\n");

    const result = await new SourceScanner().scan(project);

    expect(paths(result.files)).toEqual(["a/z.py", "b/a.py"]);
  });

  test("walk batches stay bounded", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const project = initializeProject(root);
    for (let index = 0; index < 10; index += 1) {
      write(path.join(root, `file${String(index).padStart(2, "0")}.py`), "value = 1\n");
    }
    const includeSpec = compileSpec(project.scan.include);

    const sizes: number[] = [];
    for await (const batch of iterWalkBatches(root, includeSpec, 4)) {
      if (Array.isArray(batch)) sizes.push(batch.length);
    }

    expect(sizes).toEqual([4, 4, 2]);
  });

  test("nested repositories are opaque to the walk", async () => {
    // A non-Git walk must not descend into a directory carrying a `.git` entry:
    // a nested repository or submodule is opaque, matching what `git ls-files`
    // reports on the git path.
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const nested = path.join(root, "nested");
    fs.mkdirSync(nested);
    write(path.join(nested, ".git"), "gitdir: ../.git/modules/nested\n");
    write(path.join(nested, "nested.py"), "def nested_symbol():\n    return 2\n");
    write(path.join(root, "main.py"), "def main_symbol():\n    return 3\n");
    const project = initializeProject(root);

    const result = await new SourceScanner().scan(project);

    expect(paths(result.files)).toEqual(["main.py"]);
  });

  test("the walk fallback keeps tracked-but-ignored files eligible", async () => {
    // The walk fallback inside a worktree consults the index (no `--no-index`),
    // so a force-added file that is both tracked and ignored stays eligible
    // exactly as on the git path: the index wins.
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    git(temporary, "init", "-q", root);
    const project = initializeProject(root);
    write(path.join(root, ".gitignore"), "ignored.py\n");
    write(path.join(root, "ignored.py"), "value = 1\n");
    write(path.join(root, "main.py"), "value = 2\n");
    git(root, "add", "main.py");
    git(root, "add", "-f", "ignored.py");
    const scanner = new SourceScanner();
    scanner.iterGitBatches = failingEnumeration;

    const result = await scanner.scan(project);

    expect(paths(result.files)).toEqual(["ignored.py", "main.py"]);
  });

  test("a failed git enumeration falls back to the streaming walk", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    git(temporary, "init", "-q", root);
    const project = initializeProject(root);
    write(path.join(root, "main.py"), "value = 1\n");
    const scanner = new SourceScanner();
    scanner.iterGitBatches = failingEnumeration;

    const result = await scanner.scan(project);

    expect(paths(result.files)).toEqual(["main.py"]);
  });

  test.skipIf(process.platform === "win32")(
    "a failed git process exposes no partial batches",
    async () => {
      const root = path.join(temporary, "repo");
      fs.mkdirSync(root);
      const stub = path.join(temporary, "bin");
      fs.mkdirSync(stub);
      fs.writeFileSync(
        path.join(stub, "git"),
        "#!/bin/sh\ni=0\nwhile [ $i -lt 300 ]; do printf 'file%03d.py\\0' $i; i=$((i + 1)); done\nexit 1\n",
        { mode: 0o755 },
      );
      const originalPath = process.env.PATH;
      process.env.PATH = stub;
      const batches: string[][] = [];

      try {
        await expect(async () => {
          for await (const batch of new SourceScanner().iterGitBatches(root)) batches.push(batch);
        }).toThrow();
      } finally {
        process.env.PATH = originalPath;
      }

      expect(batches).toEqual([]);
    },
  );

  test("the walk fallback passes only supported files to check-ignore", async () => {
    // The rare walk-inside-a-worktree fallback must not hand unsupported files
    // to `git check-ignore`: the pre-filter happens before the batch.
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    git(temporary, "init", "-q", root);
    const project = initializeProject(root);
    for (let index = 0; index < 20; index += 1) {
      write(path.join(root, `doc${String(index).padStart(3, "0")}.md`), "not source\n");
    }
    for (let index = 0; index < 5; index += 1) {
      write(path.join(root, `module${index}.py`), "value = 1\n");
    }
    const scanner = new SourceScanner();
    const batches: string[][] = [];
    scanner.gitIgnoredPaths = async (_root, candidates) => {
      batches.push([...candidates]);
      return new Set();
    };
    scanner.inGitWorktree = async () => true;
    scanner.iterGitBatches = failingEnumeration;

    const result = await scanner.scan(project);

    expect(result.files).toHaveLength(5);
    expect(batches.length).toBeGreaterThan(0);
    expect(batches.flat().filter((item) => !item.endsWith(".py"))).toEqual([]);
  });
});

describe("classification", () => {
  test("the default includes and the extension map describe the same languages", () => {
    // The two lists are edited separately, and either one alone is useless. An
    // extension the scanner can classify but no default pattern matches is never
    // offered a file; a default pattern with no extension entry matches files
    // the scanner then rejects as unsupported.
    expect(DEFAULT_INCLUDES.map((pattern) => pattern.replace(/^\*\*\/\*/, "")).sort()).toEqual(
      Object.keys(LANGUAGES).sort(),
    );
  });

  test("the newer language extensions have stable names", () => {
    const expected: Record<string, string> = {
      ".go": "go",
      ".tf": "terraform",
      ".tfvars": "terraform",
      ".rs": "rust",
      ".c": "c",
      ".h": "c",
      ".cc": "cpp",
      ".cpp": "cpp",
      ".cxx": "cpp",
      ".hh": "cpp",
      ".hpp": "cpp",
      ".hxx": "cpp",
      ".lua": "lua",
    };

    for (const [extension, language] of Object.entries(expected)) {
      expect(LANGUAGES[extension]).toBe(language);
    }
  });

  test("include, exclude, and gitignore matching stay case-sensitive", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const project = initializeProject(root);
    write(path.join(root, "lower.py"), "value = 1\n");
    write(path.join(root, "MAIN.PY"), "value = 2\n");
    write(path.join(root, "Mixed.py"), "value = 3\n");
    write(path.join(root, ".gitignore"), "mixed.py\n");

    const result = await new SourceScanner().scan(project);

    expect(paths(result.files)).toEqual(["Mixed.py", "lower.py"]);
    expect(result.skipped.find((item) => item.path === "MAIN.PY")?.reason).toBe("unsupported");
  });

  test("every default language is discovered", async () => {
    // Driven off `LANGUAGES` rather than a hand-written list so a newly mapped
    // extension cannot quietly go undiscovered: the file is written from the
    // map, so it is missing from the result until the default patterns cover it.
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const project = initializeProject(root);
    for (const extension of Object.keys(LANGUAGES)) {
      write(path.join(root, `sample${extension}`), "sample\n");
    }

    const result = await new SourceScanner().scan(project);

    expect(result.files.map((item) => [item.path, item.language])).toEqual(
      Object.entries(LANGUAGES)
        .filter(([extension]) => languageForExtension(extension) !== undefined)
        .map(([extension, language]) => [`sample${extension}`, language])
        .sort((left, right) => ((left[0] as string) < (right[0] as string) ? -1 : 1)),
    );
  });

  test("malformed UTF-8 invalidates the whole gitignore file", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const project = initializeProject(root);
    write(path.join(root, "main.py"), "value = 1\n");
    fs.writeFileSync(
      path.join(root, ".gitignore"),
      Buffer.from([0xff, 0x0a, ...Buffer.from("main.py\n")]),
    );

    expect(paths((await new SourceScanner().scan(project)).files)).toEqual(["main.py"]);
  });

  test("gitignore parsing recognizes Python's non-LF line boundaries", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const project = initializeProject(root);
    write(path.join(root, "drop.py"), "value = 1\n");
    write(path.join(root, "also_drop.py"), "value = 2\n");
    write(path.join(root, ".gitignore"), "drop.py\ralso_drop.py");

    expect(paths((await new SourceScanner().scan(project)).files)).toEqual([]);
  });

  test("the godot cache directory is excluded without excluding the project file", async () => {
    // `.godot` names both an indexed extension and Godot's asset cache. A Godot
    // project that has been opened in the editor carries a `.godot` directory
    // holding a generated copy of every imported asset. Nothing there is source,
    // and the project's own `project.godot` has to survive the exclusion.
    const root = path.join(temporary, "game");
    fs.mkdirSync(root);
    const project = initializeProject(root);
    write(path.join(root, "project.godot"), "config_version=5\n");
    write(path.join(root, "level.tscn"), '[node name="Player" type="Node2D"]\n');
    write(path.join(root, ".godot", "imported", "level.tscn"), '[node name="Generated"]\n');

    const result = await new SourceScanner().scan(project);

    expect(paths(result.files)).toEqual(["level.tscn", "project.godot"]);
  });

  test("nested gitignore and config excludes both apply", async () => {
    const root = path.join(temporary, "repo");
    const pkg = path.join(root, "package");
    fs.mkdirSync(pkg, { recursive: true });
    const base = initializeProject(root);
    write(path.join(pkg, ".gitignore"), "generated.py\n");
    write(path.join(pkg, "generated.py"), "generated = True\n");
    write(path.join(pkg, "keep.py"), "keep = True\n");
    const project = { ...base, scan: { ...base.scan, exclude: ["package/keep.py"] } };

    const result = await new SourceScanner().scan(project);

    expect(result.files).toEqual([]);
  });

  test("a nested gitignore can re-include a file", async () => {
    const root = path.join(temporary, "repo");
    const pkg = path.join(root, "package");
    fs.mkdirSync(pkg, { recursive: true });
    const project = initializeProject(root);
    write(path.join(root, ".gitignore"), "package/*.py\n");
    write(path.join(pkg, ".gitignore"), "!keep.py\n");
    write(path.join(pkg, "keep.py"), "keep = True\n");
    write(path.join(pkg, "drop.py"), "drop = True\n");

    const result = await new SourceScanner().scan(project);

    expect(paths(result.files)).toEqual(["package/keep.py"]);
  });

  test("oversized and symlinked files are rejected without reading", async () => {
    // Size and symlink checks are stat-only; content checks belong to the
    // indexer. The scanner used to read every changed file to test for NUL bytes
    // and UTF-8 validity, then discard the bytes, so the indexer read it again.
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const base = initializeProject(root);
    const project = { ...base, scan: { ...base.scan, max_file_bytes: 8 } };
    write(path.join(root, "large.py"), "0123456789");
    fs.writeFileSync(path.join(root, "binary.py"), Buffer.from([0x61, 0x00, 0x62]));
    const target = path.join(temporary, "target.py");
    write(target, "x = 1\n");
    fs.symlinkSync(target, path.join(root, "link.py"));

    const result = await new SourceScanner().scan(project);

    const reasons = new Set(result.skipped.map((item) => item.reason));
    expect(reasons.has("oversized")).toBe(true);
    expect(reasons.has("symlink")).toBe(true);
    expect(reasons.has("binary")).toBe(false);
    expect(reasons.has("encoding")).toBe(false);
    // binary.py is 3 bytes, so it passes the stat-only scan.
    expect(paths(result.files)).toEqual(["binary.py"]);
  });

  test.skipIf(process.platform === "win32")(
    "a symlink to a directory is pruned, not recorded as a skipped file",
    async () => {
      // `os.walk` classifies by the *resolved* type, so such an entry reaches
      // Python as a pruned directory and never as a recorded skip. Node's
      // dirents classify by the link itself, so without the resolved-type check
      // a link named like a source file would produce a spurious skip row that
      // `inspect_scan` would show and the Python build never would.
      const root = path.join(temporary, "repo");
      fs.mkdirSync(root);
      const project = initializeProject(root);
      write(path.join(root, "main.py"), "value = 1\n");
      const target = path.join(temporary, "elsewhere");
      fs.mkdirSync(target);
      write(path.join(target, "inner.py"), "value = 2\n");
      fs.symlinkSync(target, path.join(root, "linked.py"));

      const result = await new SourceScanner().scan(project);

      expect(paths(result.files)).toEqual(["main.py"]);
      expect(result.skipped.map((item) => item.path)).not.toContain("linked.py");
    },
  );

  test.skipIf(process.platform === "win32")(
    "a broken symlink with a source suffix is still recorded as a symlink",
    async () => {
      const root = path.join(temporary, "repo");
      fs.mkdirSync(root);
      const project = initializeProject(root);
      fs.symlinkSync(path.join(temporary, "does-not-exist.py"), path.join(root, "dangling.py"));

      const result = await new SourceScanner().scan(project);

      expect(result.skipped).toEqual([{ path: "dangling.py", reason: "symlink", detail: null }]);
    },
  );

  test("scan does not read file contents", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    write(path.join(root, "ok.py"), "def ok():\n    return 1\n");
    const project = initializeProject(root);
    const spy = spyOn(fs.promises, "readFile").mockImplementation((async (file: unknown) => {
      throw new Error(`scan must not read ${String(file)}`);
    }) as never);

    try {
      expect((await new SourceScanner().scan(project)).files).toHaveLength(1);
    } finally {
      spy.mockRestore();
    }
  });

  test("iterScan reads one file's source at a time", async () => {
    // Streaming means laziness: a file's bytes are read only as it is yielded.
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "a.py"), "a = 1\n");
    fs.writeFileSync(path.join(root, "b.py"), "b = 2\n");
    const project = initializeProject(root);
    const reads: string[] = [];
    const original = fs.promises.readFile.bind(fs.promises);
    const spy = spyOn(fs.promises, "readFile").mockImplementation(((...args: unknown[]) => {
      reads.push(String(args[0]));
      return original(...(args as Parameters<typeof fs.promises.readFile>));
    }) as never);

    try {
      const stream = new SourceScanner().iterScan(project);
      const first = (await stream.next()).value as { content: Uint8Array };
      expect(reads).toHaveLength(1);
      const second = (await stream.next()).value as { content: Uint8Array };
      expect(reads).toHaveLength(2);

      // The streaming path hands the source to the caller instead of re-reading.
      expect(new TextDecoder().decode(first.content)).toBe("a = 1\n");
      expect(new TextDecoder().decode(second.content)).toBe("b = 2\n");
      await stream.return(undefined);
    } finally {
      spy.mockRestore();
    }
  });

  test("hard-excluded directories are never walked", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    const project = initializeProject(root);
    write(path.join(root, "main.py"), "print('ok')\n");
    write(path.join(root, "node_modules", "vendor.js"), "export default 1\n");
    write(path.join(root, ".git", "hook.py"), "hook = True\n");
    for (const marker of [".code-indexing-mcp", ".ci-mcp"]) {
      write(path.join(root, marker, "private.py"), "private = True\n");
    }
    const excluded = [...HARD_EXCLUDED_DIRECTORIES];
    const statted: string[] = [];
    const original = fs.promises.stat.bind(fs.promises);
    const spy = spyOn(fs.promises, "stat").mockImplementation(((...args: unknown[]) => {
      const file = String(args[0]);
      // The walk stats `<dir>/.git` to detect a nested repository, which is a
      // per-directory probe rather than a per-file cost.
      if (!file.endsWith(`${path.sep}.git`)) statted.push(file);
      return original(...(args as Parameters<typeof fs.promises.stat>));
    }) as never);

    let result: Awaited<ReturnType<SourceScanner["scan"]>>;
    try {
      result = await new SourceScanner().scan(project);
    } finally {
      spy.mockRestore();
    }

    expect(
      statted.filter((file) => file.split(path.sep).some((part) => excluded.includes(part))),
    ).toEqual([]);
    expect(paths(result.files)).toEqual(["main.py"]);
  });
});

describe("hasSupportedSource", () => {
  test("respects ignore and hard-exclusion rules", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    write(path.join(root, ".gitignore"), "ignored.py\n");
    write(path.join(root, "ignored.py"), "value = 1\n");
    write(path.join(root, "node_modules", "vendor.js"), "export default 1\n");
    const config = ScanConfig.parse({});

    expect(await new SourceScanner().hasSupportedSource(root, config)).toBe(false);

    write(path.join(root, "main.ts"), "export const answer = 42\n");

    expect(await new SourceScanner().hasSupportedSource(root, config)).toBe(true);
  });

  test("applies nested gitignore rules", async () => {
    const root = path.join(temporary, "repo");
    const pkg = path.join(root, "package");
    fs.mkdirSync(pkg, { recursive: true });
    write(path.join(root, ".gitignore"), "package/*.py\n");
    write(path.join(pkg, ".gitignore"), "!keep.py\n");
    const keep = path.join(pkg, "keep.py");
    write(keep, "keep = True\n");
    write(path.join(pkg, "drop.py"), "drop = True\n");
    const scanner = new SourceScanner();
    const config = ScanConfig.parse({});

    expect(await scanner.hasSupportedSource(root, config)).toBe(true);

    fs.unlinkSync(keep);

    expect(await scanner.hasSupportedSource(root, config)).toBe(false);
  });

  test("batches its git-ignore queries", async () => {
    const root = path.join(temporary, "repo");
    fs.mkdirSync(root);
    for (let index = 0; index < 300; index += 1) {
      write(path.join(root, `ignored_${String(index).padStart(3, "0")}.py`), "ignored = True\n");
    }
    const scanner = new SourceScanner();
    const batches: string[][] = [];
    scanner.gitIgnoredPaths = async (base, candidates) => {
      batches.push([...candidates]);
      return new Set(candidates.map((item) => path.relative(base, item)));
    };

    expect(await scanner.hasSupportedSource(root, ScanConfig.parse({}))).toBe(false);
    expect(batches).toHaveLength(2);
    expect(Math.max(...batches.map((batch) => batch.length))).toBe(256);
  });

  test.skipIf(process.platform === "win32")(
    "a hung git check-ignore times out and falls back to the local rules",
    async () => {
      const root = path.join(temporary, "repo");
      fs.mkdirSync(root);
      const file = path.join(root, "main.py");
      write(file, "value = 1\n");
      // A `git` that never exits is exactly the hang this deadline guards: the
      // scanner must report "nothing ignored" rather than wedging the scan. The
      // deadline is shortened so the suite does not wait ten seconds for a path
      // that behaves identically at either value.
      const stub = path.join(temporary, "bin");
      fs.mkdirSync(stub);
      fs.writeFileSync(path.join(stub, "git"), "#!/bin/sh\nsleep 30\n", { mode: 0o755 });
      const originalPath = process.env.PATH;
      process.env.PATH = stub;

      try {
        const scanner = new SourceScanner({ checkIgnoreTimeoutMs: 100 });
        const started = Date.now();

        const ignored = await scanner.gitIgnoredPaths(root, [file]);

        expect(ignored.size).toBe(0);
        expect(Date.now() - started).toBeLessThan(5_000);
      } finally {
        process.env.PATH = originalPath;
      }
    },
  );

  test.skipIf(process.platform === "win32")(
    "a git that cannot run leaves the tree eligible rather than empty",
    async () => {
      const root = path.join(temporary, "repo");
      fs.mkdirSync(root);
      write(path.join(root, "main.py"), "value = 1\n");
      const stub = path.join(temporary, "nogit");
      fs.mkdirSync(stub);
      fs.writeFileSync(path.join(stub, "git"), "#!/bin/sh\nexit 128\n", { mode: 0o755 });
      const originalPath = process.env.PATH;
      process.env.PATH = stub;

      try {
        const project = initializeProject(root);

        // `rev-parse` fails, so this is not a worktree; the walk covers it and
        // `check-ignore` is never consulted.
        expect(paths((await new SourceScanner().scan(project)).files)).toEqual(["main.py"]);
      } finally {
        process.env.PATH = originalPath;
      }
    },
  );
});

test("the LanguageName enum matches the scanner's language set", () => {
  expect([...(LanguageName.options as readonly string[])].sort()).toEqual(
    [...new Set(Object.values(LANGUAGES))].sort(),
  );
});

/**
 * A `git ls-files` enumeration that fails before yielding anything.
 *
 * It must stay a generator: the scanner's fallback is reached only when the
 * failure surfaces during iteration, which is where the real one fails too.
 */
// biome-ignore lint/correctness/useYield: yielding nothing before throwing is the point.
async function* failingEnumeration(): AsyncGenerator<string[]> {
  throw new GitEnumerationError("simulated git failure");
}
