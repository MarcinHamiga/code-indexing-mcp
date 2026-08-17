/**
 * Filesystem scanning with local ignore rules and safety constraints.
 *
 * Two enumeration strategies share one classification pipeline:
 *
 * - Inside a Git worktree the scanner asks Git for the truth up front --
 *   `git ls-files --cached --others --exclude-standard` returns every tracked
 *   file plus every untracked non-ignored file in one bounded stream. Ignore
 *   rules (`.gitignore`, `info/exclude`, global excludes) are applied by Git
 *   itself, so the scanner never re-runs `check-ignore` on those candidates and
 *   never stats a file whose suffix or include pattern already excludes it.
 *   Tracked-but-ignored files stay eligible because Git's own rule is that the
 *   index wins. Submodules and nested repositories appear as single non-file
 *   entries and are not descended into.
 * - Outside Git, the walk streams per-directory, sorted, with nested
 *   `.gitignore` files loaded incrementally and candidate batches bounded in
 *   memory. `git check-ignore` runs only on a rare fallback path (a worktree
 *   whose `ls-files` enumeration failed) and only over pre-filtered batches.
 *
 * ## Why this module is async where Python's is not
 *
 * `scanner.py` is synchronous and the MCP server offloads it to a thread. Here
 * the same work is an `AsyncGenerator`: a stdio MCP server and a JSON-RPC
 * daemon share one event loop with the scan, and a synchronous walk of a large
 * repository would stall every in-flight request until it finished. It also
 * makes the `git ls-files` deadline enforceable without the reader thread and
 * staging queue `_iter_git_batches` needs -- streaming stdout with an abort
 * timer is the same guarantee in a tenth of the code. The yielded items,
 * batching, and ordering are unchanged, so callers see the identical sequence.
 */

import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import ignore, { type Ignore } from "ignore";
import type { ProjectInfo, ScanConfig, ScanResult, ScannedFile, SkippedFile } from "./models.ts";
import { unavailableLanguages } from "./grammars.ts";
import { resolvePath } from "./paths.ts";

export const LANGUAGES: Readonly<Record<string, string>> = {
  ".py": "python",
  ".pyi": "python",
  ".java": "java",
  ".js": "javascript",
  ".jsx": "javascript",
  ".mjs": "javascript",
  ".cjs": "javascript",
  ".ts": "typescript",
  ".mts": "typescript",
  ".cts": "typescript",
  ".tsx": "tsx",
  ".cs": "csharp",
  ".csx": "csharp",
  ".gd": "gdscript",
  ".gdshader": "gdshader",
  ".gdshaderinc": "gdshader",
  ".tres": "godot_resource",
  ".tscn": "godot_resource",
  ".godot": "godot_resource",
  ".sql": "sql",
  ".yaml": "yaml",
  ".yml": "yaml",
  ".json": "json",
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

export const HARD_EXCLUDED_DIRECTORIES: ReadonlySet<string> = new Set([
  ".git",
  ".ci-mcp",
  ".code-indexing-mcp",
  // `.godot` is both an extension this scanner indexes and the name of Godot's
  // own cache directory, which holds a generated copy of every imported asset.
  // Excluding the directory does not exclude a `project.godot` file: only
  // directory names are matched here.
  ".godot",
  ".venv",
  "venv",
  "node_modules",
  "dist",
  "build",
  ".next",
  ".pytest_cache",
  ".mypy_cache",
  ".ruff_cache",
  "__pycache__",
  "coverage",
  "htmlcov",
]);

export const GIT_IGNORE_DISCOVERY_BATCH_SIZE = 256;
export const GIT_CHECK_IGNORE_TIMEOUT_MS = 10_000;
export const GIT_LS_FILES_TIMEOUT_MS = 10_000;

/** Git enumeration failed or timed out; the walk path takes over. */
export class GitEnumerationError extends Error {
  override readonly name = "GitEnumerationError";
}

/**
 * The language for a file extension, or undefined when this build cannot parse it.
 *
 * A grammar missing on a platform (today: GDShader on Windows, §5.5) is a
 * *supported state*, not an error: the extension reads as unsupported, so a
 * Godot repository indexes its scripts and scenes and quietly omits its
 * shaders rather than failing the run.
 */
export function languageForExtension(extension: string): string | undefined {
  const language = LANGUAGES[extension.toLowerCase()];
  if (language === undefined) return undefined;
  return unavailableLanguages().includes(language) ? undefined : language;
}

/** One walk candidate: an absolute path plus the ignore stack in force there. */
type Candidate = readonly [absolute: string, ignoreSpecs: readonly IgnoreSpec[]];
/** A `.gitignore` file's rules, and the repository-relative directory they apply from. */
interface IgnoreSpec {
  readonly base: string;
  readonly spec: Ignore;
}

export interface ScannerOptions {
  /** Deadline for `git check-ignore`, past which its answer is treated as absent. */
  checkIgnoreTimeoutMs?: number;
  /** Deadline for `git ls-files`, past which the streaming walk takes over. */
  lsFilesTimeoutMs?: number;
}

export class SourceScanner {
  readonly #checkIgnoreTimeoutMs: number;
  readonly #lsFilesTimeoutMs: number;

  constructor(options: ScannerOptions = {}) {
    this.#checkIgnoreTimeoutMs = options.checkIgnoreTimeoutMs ?? GIT_CHECK_IGNORE_TIMEOUT_MS;
    this.#lsFilesTimeoutMs = options.lsFilesTimeoutMs ?? GIT_LS_FILES_TIMEOUT_MS;
  }

  /** Whether *root* contains an eligible source file, without reading one. */
  async hasSupportedSource(root: string, scan: ScanConfig): Promise<boolean> {
    const resolved = resolvePath(root);
    const configExcludes = compileSpec(scan.exclude);
    const includeSpec = compileSpec(scan.include);
    let eligible: string[] = [];

    for await (const visit of walkDirectories(resolved, { opaqueRepositories: false })) {
      for (const name of visit.files) {
        const absolute = path.join(visit.directory, name);
        const relative = relativePosix(resolved, absolute);
        const { language } = await classify(relative, absolute, {
          includeSpec,
          configExcludes,
          ignoreSpecs: visit.ignoreSpecs,
        });
        if (language === null) continue;
        try {
          const stat = await fs.promises.stat(absolute);
          if (stat.size > scan.max_file_bytes) continue;
        } catch {
          continue;
        }
        eligible.push(absolute);
        if (eligible.length >= GIT_IGNORE_DISCOVERY_BATCH_SIZE) {
          const batch = eligible;
          eligible = [];
          const ignored = await this.gitIgnoredPaths(resolved, batch);
          if (batch.some((item) => !ignored.has(relativePosix(resolved, item)))) return true;
        }
      }
    }
    const ignored = await this.gitIgnoredPaths(resolved, eligible);
    return eligible.some((item) => !ignored.has(relativePosix(resolved, item)));
  }

  /** Collect stat-only scan results without retaining or reading source bytes. */
  async scan(
    project: ProjectInfo,
    knownFiles?: ReadonlyMap<string, { size: number; mtime_ns: bigint }>,
  ): Promise<ScanResult> {
    const files: ScannedFile[] = [];
    const skipped: SkippedFile[] = [];
    for await (const item of this.iterScan(project, knownFiles, { readContents: false })) {
      if ("language" in item) files.push(item);
      else skipped.push(item);
    }
    return { files, skipped };
  }

  /**
   * Yield scan results one file at a time.
   *
   * When `readContents` is true, changed files carry their raw source bytes so
   * the indexer never reads a file twice. Binary and encoding validation
   * belongs to the indexer, where those bytes are already consumed. The bytes
   * die with the yielded item, so at most one file's source is live at any
   * moment.
   *
   * Git worktrees are enumerated with `git ls-files`, which applies every
   * ignore rule in one pass; everything else streams the walk with incremental
   * nested ignore rules. Either way only files whose suffix and include pattern
   * already admit them are ever statted or passed to Git, and candidate batches
   * stay bounded in memory.
   */
  async *iterScan(
    project: ProjectInfo,
    knownFiles?: ReadonlyMap<string, { size: number; mtime_ns: bigint }>,
    options: { readContents?: boolean } = {},
  ): AsyncGenerator<ScannedFile | SkippedFile> {
    const readContents = options.readContents ?? true;
    const known = knownFiles ?? new Map();
    const root = resolvePath(project.root);
    const configExcludes = compileSpec(project.scan.exclude);
    const includeSpec = compileSpec(project.scan.include);
    const inWorktree = await this.inGitWorktree(root);
    if (inWorktree) {
      try {
        for await (const batch of this.iterGitBatches(root)) {
          for (const item of prefilterGitBatch(batch, root, includeSpec)) {
            if (Array.isArray(item)) {
              yield* scanCandidates(item as readonly Candidate[], {
                root,
                includeSpec,
                configExcludes,
                maxFileBytes: project.scan.max_file_bytes,
                knownFiles: known,
                readContents,
                runCheckIgnore: false,
                scanner: this,
              });
            } else {
              yield item as SkippedFile;
            }
          }
        }
        return;
      } catch (error) {
        // A failed or timed-out git process must not silently produce an empty
        // index: the streaming walk covers the same tree.
        if (!(error instanceof GitEnumerationError)) throw error;
      }
    }
    for await (const walkBatch of iterWalkBatches(root, includeSpec)) {
      if (Array.isArray(walkBatch)) {
        yield* scanCandidates(walkBatch as readonly Candidate[], {
          root,
          includeSpec,
          configExcludes,
          maxFileBytes: project.scan.max_file_bytes,
          knownFiles: known,
          readContents,
          runCheckIgnore: inWorktree,
          scanner: this,
        });
      } else {
        yield walkBatch as SkippedFile;
      }
    }
  }

  /**
   * Whether *root* sits inside a Git worktree.
   *
   * A worktree may carry its repository as a `.git` file or directory, so the
   * authoritative check is Git itself, not the presence of a directory. Outside
   * any repository Git exits 128 and the scanner falls back to the walk.
   */
  async inGitWorktree(root: string): Promise<boolean> {
    const result = await runGit(["-C", root, "rev-parse", "--is-inside-work-tree"], {
      timeoutMs: this.#checkIgnoreTimeoutMs,
    });
    return result !== null && result.code === 0 && result.stdout.toString("utf8").trim() === "true";
  }

  /**
   * Stream `git ls-files` output in bounded, sorted batches.
   *
   * A hung git process cannot wedge a scan: the child is aborted at the
   * deadline and the walk takes over. A non-zero exit (which cannot normally
   * happen for a worktree the probe just accepted) falls back the same way.
   */
  async *iterGitBatches(root: string): AsyncGenerator<string[]> {
    const controller = new AbortController();
    const deadline = setTimeout(() => controller.abort(), this.#lsFilesTimeoutMs);
    const child = spawn(
      "git",
      ["-C", root, "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
      { stdio: ["ignore", "pipe", "ignore"], signal: controller.signal },
    );
    const exited = new Promise<number | null>((resolve, reject) => {
      child.once("close", (code) => resolve(code));
      child.once("error", (error) => reject(new GitEnumerationError(String(error))));
    });
    let batch: string[] = [];
    let pending: Buffer<ArrayBufferLike> = Buffer.alloc(0);
    try {
      for await (const chunk of child.stdout) {
        pending =
          pending.length === 0 ? (chunk as Buffer) : Buffer.concat([pending, chunk as Buffer]);
        let start = 0;
        for (;;) {
          const separator = pending.indexOf(0, start);
          if (separator < 0) break;
          const raw = pending.subarray(start, separator);
          start = separator + 1;
          if (raw.length > 0) batch.push(path.join(root, raw.toString("utf8")));
          if (batch.length >= GIT_IGNORE_DISCOVERY_BATCH_SIZE) {
            batch.sort(comparePaths);
            yield batch;
            batch = [];
          }
        }
        // Compact the consumed prefix once per chunk rather than per path.
        pending = pending.subarray(start);
      }
      const code = await exited;
      if (code !== 0) throw new GitEnumerationError(`git ls-files exited with status ${code}`);
    } catch (error) {
      if (error instanceof GitEnumerationError) throw error;
      throw new GitEnumerationError(String(error));
    } finally {
      clearTimeout(deadline);
      // A consumer that stops early or an exception path must not leave the
      // process running; an already-exited process makes kill a no-op.
      child.kill();
    }
    if (batch.length > 0) {
      batch.sort(comparePaths);
      yield batch;
    }
  }

  /**
   * Paths ignored by Git's complete standard exclude stack.
   *
   * `git check-ignore` applies nested `.gitignore` files, the repository's
   * `info/exclude`, and the user's configured global excludes in one batch. The
   * index is consulted (no `--no-index`), so a tracked-but-ignored file is
   * never reported, matching `git ls-files` semantics where the index wins.
   * Non-Git projects and environments without Git keep using the in-process
   * `.gitignore` fallback loaded by {@link loadIgnoreSpecs}.
   */
  async gitIgnoredPaths(root: string, candidates: readonly string[]): Promise<Set<string>> {
    const relative = candidates.map((item) => relativePosix(root, item));
    if (relative.length === 0) return new Set();
    const payload = Buffer.from(`${relative.join("\0")}\0`, "utf8");
    const result = await runGit(["-C", root, "check-ignore", "--stdin", "-z"], {
      timeoutMs: this.#checkIgnoreTimeoutMs,
      input: payload,
    });
    if (result === null || (result.code !== 0 && result.code !== 1)) return new Set();
    return new Set(
      result.stdout
        .toString("utf8")
        .split("\0")
        .filter((item) => item.length > 0),
    );
  }
}

/**
 * Split one git-enumerated batch into recordable skips and candidates.
 *
 * Git already applied every ignore rule, but the scanner's own safety and
 * include filters still apply before any stat: hard-excluded directories are
 * dropped silently and unsupported suffixes are recorded as skips, so
 * {@link classify} (and its stat calls) never see them.
 */
function* prefilterGitBatch(
  batch: readonly string[],
  root: string,
  includeSpec: Ignore,
): Generator<SkippedFile | readonly Candidate[]> {
  const items: Candidate[] = [];
  for (const absolute of batch) {
    const relative = relativePosix(root, absolute);
    if (inHardExcludedDirectory(relative)) continue;
    if (
      languageForExtension(path.extname(absolute)) === undefined ||
      !matches(includeSpec, relative)
    ) {
      yield { path: relative, reason: "unsupported", detail: null };
      continue;
    }
    items.push([absolute, []]);
  }
  if (items.length > 0) yield items;
}

/**
 * Stream the non-Git tree in deterministic per-directory order.
 *
 * Nested `.gitignore` files are loaded as the walk reaches their directory.
 * Directories that are their own repository (submodules, nested repositories)
 * are opaque, matching what `git ls-files` reports for the git path. Candidates
 * are pre-filtered by suffix and include pattern before any stat or ignore
 * work; batches stay bounded in memory.
 */
export async function* iterWalkBatches(
  root: string,
  includeSpec: Ignore,
  batchSize: number = GIT_IGNORE_DISCOVERY_BATCH_SIZE,
): AsyncGenerator<SkippedFile | readonly Candidate[]> {
  let batch: Candidate[] = [];
  for await (const visit of walkDirectories(root, { opaqueRepositories: true })) {
    for (const name of visit.files) {
      const absolute = path.join(visit.directory, name);
      const relative = relativePosix(root, absolute);
      if (inHardExcludedDirectory(relative)) continue;
      if (
        languageForExtension(path.extname(absolute)) === undefined ||
        !matches(includeSpec, relative)
      ) {
        yield { path: relative, reason: "unsupported", detail: null };
        continue;
      }
      batch.push([absolute, visit.ignoreSpecs]);
      if (batch.length >= batchSize) {
        yield batch;
        batch = [];
      }
    }
  }
  if (batch.length > 0) yield batch;
}

interface DirectoryVisit {
  readonly directory: string;
  readonly files: readonly string[];
  readonly ignoreSpecs: readonly IgnoreSpec[];
}

/**
 * `os.walk` with the scanner's pruning rules, pre-order and sorted.
 *
 * The ignore stack is threaded through the traversal rather than kept in a
 * whole-tree map: an entry is built when a directory is entered and released
 * when its subtree is done, so memory stays proportional to the traversal
 * frontier rather than to the tree.
 */
async function* walkDirectories(
  root: string,
  options: { opaqueRepositories: boolean },
): AsyncGenerator<DirectoryVisit> {
  const stack: Array<{ directory: string; ignoreSpecs: readonly IgnoreSpec[] }> = [
    { directory: root, ignoreSpecs: [] },
  ];
  while (stack.length > 0) {
    const { directory, ignoreSpecs: inherited } = stack.pop() as {
      directory: string;
      ignoreSpecs: readonly IgnoreSpec[];
    };
    let entries: fs.Dirent[];
    try {
      entries = await fs.promises.readdir(directory, { withFileTypes: true });
    } catch {
      continue;
    }
    const files = entries
      .filter((entry) => !entry.isDirectory())
      .map((entry) => entry.name)
      .sort(compareStrings);
    let ignoreSpecs = inherited;
    if (files.includes(".gitignore")) {
      ignoreSpecs = [
        ...inherited,
        ...(await loadIgnoreSpecs(root, [path.join(directory, ".gitignore")])),
      ];
    }
    yield { directory, files, ignoreSpecs };

    const directories: string[] = [];
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      if (HARD_EXCLUDED_DIRECTORIES.has(entry.name)) continue;
      const child = path.join(directory, entry.name);
      if (await isSymbolicLink(child)) continue;
      if (options.opaqueRepositories && (await isRepositoryWorktree(child))) continue;
      directories.push(entry.name);
    }
    directories.sort(compareStrings);
    // Pre-order DFS in sorted order: push reversed so the first name pops first.
    for (let index = directories.length - 1; index >= 0; index -= 1) {
      stack.push({ directory: path.join(directory, directories[index] as string), ignoreSpecs });
    }
  }
}

/**
 * Whether *target* is its own repository checkout.
 *
 * Git treats a directory carrying a `.git` entry (a submodule or a nested
 * repository) as a single opaque non-file, so the walk must not descend into
 * it: the parent repository has no claim on those files.
 */
async function isRepositoryWorktree(target: string): Promise<boolean> {
  try {
    await fs.promises.stat(path.join(target, ".git"));
    return true;
  } catch {
    return false;
  }
}

async function isSymbolicLink(target: string): Promise<boolean> {
  try {
    return (await fs.promises.lstat(target)).isSymbolicLink();
  } catch {
    return false;
  }
}

/**
 * Classify, stat, and optionally read one pre-filtered candidate batch.
 *
 * `runCheckIgnore` is the rare walk fallback inside a worktree whose
 * `ls-files` enumeration failed; regular walks outside Git skip the subprocess
 * entirely because there is nothing for it to apply.
 */
async function* scanCandidates(
  items: readonly Candidate[],
  options: {
    root: string;
    includeSpec: Ignore;
    configExcludes: Ignore;
    maxFileBytes: number;
    knownFiles: ReadonlyMap<string, { size: number; mtime_ns: bigint }>;
    readContents: boolean;
    runCheckIgnore: boolean;
    scanner: SourceScanner;
  },
): AsyncGenerator<ScannedFile | SkippedFile> {
  const { root, includeSpec, configExcludes, maxFileBytes, knownFiles, readContents } = options;
  let ignoredPaths: ReadonlySet<string> = new Set();
  if (options.runCheckIgnore && items.length > 0) {
    ignoredPaths = await options.scanner.gitIgnoredPaths(
      root,
      items.map(([absolute]) => absolute),
    );
  }
  for (const [absolute, ignoreSpecs] of items) {
    const relative = relativePosix(root, absolute);
    const { language, skipReason } = await classify(relative, absolute, {
      includeSpec,
      configExcludes,
      // On the worktree fallback, `git check-ignore` is the authoritative
      // ignore source (it applies `.gitignore`, `info/exclude`, and global
      // excludes) and knows the index: the in-process `.gitignore` specs must
      // not second-guess it, or a tracked-but-ignored file would be dropped as
      // "ignored" instead of staying eligible (the index wins).
      ignoreSpecs: options.runCheckIgnore ? [] : ignoreSpecs,
      standardIgnored: ignoredPaths.has(relative),
    });
    if (language === null) {
      if (skipReason !== null) yield { path: relative, reason: skipReason, detail: null };
      continue;
    }
    let stat: fs.BigIntStats;
    try {
      stat = await fs.promises.stat(absolute, { bigint: true });
    } catch (error) {
      yield { path: relative, reason: "unreadable", detail: String(error) };
      continue;
    }
    const size = Number(stat.size);
    if (size > maxFileBytes) {
      yield { path: relative, reason: "oversized", detail: null };
      continue;
    }
    const previous = knownFiles.get(relative);
    let content: Uint8Array<ArrayBuffer> | null = null;
    if (
      readContents &&
      (previous === undefined || previous.size !== size || previous.mtime_ns !== stat.mtimeNs)
    ) {
      try {
        const raw = await fs.promises.readFile(absolute);
        content = new Uint8Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
      } catch (error) {
        yield { path: relative, reason: "unreadable", detail: String(error) };
        continue;
      }
    }
    yield {
      path: relative,
      absolute_path: absolute,
      language,
      size,
      mtime_ns: stat.mtimeNs,
      content,
    };
  }
}

/**
 * Decide whether a candidate file is eligible for indexing.
 *
 * `language` is set only when the file passes every path-based eligibility
 * check (not in a hard-excluded directory, not a symlink, a regular file, a
 * supported suffix that matches the include spec, and not matched by config
 * excludes or gitignore rules); callers still need to apply their own
 * size/content checks on top. `skipReason` carries the reason string `scan`
 * records as a skipped file (`"symlink"`, `"unsupported"`, or `"ignored"`); it
 * is null for rejections `scan` does not record (hard-excluded directories,
 * non-files, and symlinks whose suffix is not supported to begin with).
 */
async function classify(
  relative: string,
  absolute: string,
  options: {
    includeSpec: Ignore;
    configExcludes: Ignore;
    ignoreSpecs: readonly IgnoreSpec[];
    standardIgnored?: boolean;
  },
): Promise<{ language: string | null; skipReason: string | null }> {
  if (inHardExcludedDirectory(relative)) return { language: null, skipReason: null };
  let stat: fs.Stats;
  try {
    stat = await fs.promises.lstat(absolute);
  } catch {
    return { language: null, skipReason: null };
  }
  if (stat.isSymbolicLink()) {
    if (!(path.extname(absolute).toLowerCase() in LANGUAGES)) {
      return { language: null, skipReason: null };
    }
    // A symlink that resolves to a *directory* is not a skipped file at all:
    // `os.walk` classifies by the resolved type, so such an entry reaches
    // Python as a pruned directory and never as a recorded skip. Node's
    // dirents classify by the link itself, so the resolved type is checked
    // here -- one extra stat, only for a symlink whose name carries a source
    // suffix, which is already the rare case.
    try {
      if ((await fs.promises.stat(absolute)).isDirectory()) {
        return { language: null, skipReason: null };
      }
    } catch {
      // A broken link resolves to nothing; Python records it as a symlink skip
      // because `is_dir()` is false for it too.
    }
    return { language: null, skipReason: "symlink" };
  }
  if (!stat.isFile()) return { language: null, skipReason: null };
  const language = languageForExtension(path.extname(absolute));
  if (language === undefined || !matches(options.includeSpec, relative)) {
    return { language: null, skipReason: "unsupported" };
  }
  if (
    (options.standardIgnored ?? false) ||
    matches(options.configExcludes, relative) ||
    isIgnored(relative, options.ignoreSpecs)
  ) {
    return { language: null, skipReason: "ignored" };
  }
  return { language, skipReason: null };
}

function inHardExcludedDirectory(relative: string): boolean {
  const parts = relative.split("/");
  return parts.slice(0, -1).some((part) => HARD_EXCLUDED_DIRECTORIES.has(part));
}

async function loadIgnoreSpecs(root: string, gitignores: readonly string[]): Promise<IgnoreSpec[]> {
  const specs: IgnoreSpec[] = [];
  for (const file of gitignores) {
    const relative = relativePosix(root, file);
    if (inHardExcludedDirectory(relative)) continue;
    try {
      const text = await fs.promises.readFile(file, "utf8");
      specs.push({ base: posixDirname(relative), spec: compileSpec(text.split(/\r?\n/)) });
    } catch {
      // An unreadable or non-UTF-8 `.gitignore` contributes no rules rather
      // than failing the scan, as `_load_ignore_specs` does.
    }
  }
  return specs;
}

function isIgnored(relative: string, specs: readonly IgnoreSpec[]): boolean {
  let ignored = false;
  for (const { base, spec } of specs) {
    let candidate: string;
    if (base === "") candidate = relative;
    else if (relative.startsWith(`${base}/`)) candidate = relative.slice(base.length + 1);
    else continue;
    const verdict = spec.test(candidate);
    // pathspec reports the *last* matching pattern's polarity and `None` when
    // nothing matched; a rule that matched nothing must leave an outer
    // directory's verdict standing rather than resetting it to "not ignored".
    if (verdict.ignored) ignored = true;
    else if (verdict.unignored) ignored = false;
  }
  return ignored;
}

/** Gitignore-syntax pattern matching, the `pathspec.GitIgnoreSpec` of this port. */
export function compileSpec(lines: readonly string[]): Ignore {
  return ignore({ allowRelativePaths: true }).add([...lines]);
}

function matches(spec: Ignore, relative: string): boolean {
  return relative.length > 0 && spec.ignores(relative);
}

function relativePosix(root: string, absolute: string): string {
  return path.relative(root, absolute).split(path.sep).join("/");
}

function posixDirname(relative: string): string {
  const separator = relative.lastIndexOf("/");
  return separator < 0 ? "" : relative.slice(0, separator);
}

/** Python's `PurePath.__lt__`: a plain comparison of the whole path string. */
function comparePaths(left: string, right: string): number {
  return compareStrings(left, right);
}

function compareStrings(left: string, right: string): number {
  if (left === right) return 0;
  return left < right ? -1 : 1;
}

/**
 * Run git, returning null when it cannot be started or does not finish in time.
 *
 * Both are "Git has nothing to say" for this module: the caller falls back to
 * the in-process ignore rules or to the streaming walk rather than reporting an
 * empty tree.
 */
async function runGit(
  argv: readonly string[],
  options: { timeoutMs: number; input?: Buffer },
): Promise<{ code: number; stdout: Buffer } | null> {
  return new Promise((resolve) => {
    let child: ReturnType<typeof spawn>;
    try {
      child = spawn("git", [...argv], {
        stdio: [options.input === undefined ? "ignore" : "pipe", "pipe", "ignore"],
      });
    } catch {
      resolve(null);
      return;
    }
    const chunks: Buffer[] = [];
    let settled = false;
    const finish = (value: { code: number; stdout: Buffer } | null): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(value);
    };
    const timer = setTimeout(() => {
      child.kill();
      finish(null);
    }, options.timeoutMs);
    child.stdout?.on("data", (chunk: Buffer) => chunks.push(chunk));
    child.on("error", () => finish(null));
    child.on("close", (code) => finish({ code: code ?? -1, stdout: Buffer.concat(chunks) }));
    if (options.input !== undefined && child.stdin !== null) {
      child.stdin.on("error", () => {});
      child.stdin.end(options.input);
    }
  });
}
