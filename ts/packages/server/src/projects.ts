/** Project marker creation and resolution. */

import { randomUUID } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { parse as parseToml, stringify as stringifyToml } from "smol-toml";
import { CodeIndexingError } from "./errors.ts";
import {
  DEFAULT_INCLUDES,
  LEGACY_DEFAULT_INCLUDES_V1,
  LEGACY_DEFAULT_INCLUDES_V2,
  LEGACY_DEFAULT_INCLUDES_V3,
  ProjectInfo,
  ScanConfig,
} from "./models.ts";
import {
  expandUser,
  fileIdentity,
  resolvePath,
  rootedUnder,
  sameFile as sameProjectRoot,
} from "./paths.ts";

// Defined in paths.ts, where the pathlib semantics they depend on live, and
// exported from here because this is the module that gives them their meaning.
export { fileIdentity as projectRootIdentity, rootedUnder, sameProjectRoot };

export const MARKER_DIRECTORY = ".ci-mcp";
export const LEGACY_MARKER_DIRECTORY = ".code-indexing-mcp";
export const MARKER_FILE = "project.toml";

export function markerPath(root: string): string {
  return path.join(root, MARKER_DIRECTORY, MARKER_FILE);
}

export function legacyMarkerPath(root: string): string {
  return path.join(root, LEGACY_MARKER_DIRECTORY, MARKER_FILE);
}

function isFile(value: string): boolean {
  try {
    return fs.statSync(value).isFile();
  } catch {
    return false;
  }
}

function isDirectory(value: string): boolean {
  try {
    return fs.statSync(value).isDirectory();
  } catch {
    return false;
  }
}

export function existingMarkerPath(root: string): string | null {
  const current = markerPath(root);
  if (isFile(current)) return current;
  const legacy = legacyMarkerPath(root);
  if (isFile(legacy)) return legacy;
  return null;
}

export function initializeProject(
  root: string,
  { name, forceNewId = false }: { name?: string | undefined; forceNewId?: boolean } = {},
): ProjectInfo {
  const resolved = resolvePath(root);
  if (!isDirectory(resolved)) {
    throw new CodeIndexingError(
      "PROJECT_NOT_FOUND",
      `Project directory does not exist: ${resolved}`,
    );
  }
  if (existingMarkerPath(resolved) !== null && !forceNewId) {
    return readProjectMarker(resolved);
  }

  const target = markerPath(resolved);
  const project = ProjectInfo.parse({
    id: randomUUID(),
    name: name ?? path.basename(resolved),
    root: resolved,
  });
  const directory = path.dirname(target);
  fs.mkdirSync(directory, { recursive: true, mode: 0o700 });
  fs.writeFileSync(path.join(directory, ".gitignore"), "*\n", "utf8");
  fs.writeFileSync(
    target,
    stringifyToml({
      version: project.version,
      id: project.id,
      name: project.name,
      scan: project.scan,
    }),
    "utf8",
  );
  return project;
}

export function readProjectMarker(root: string): ProjectInfo {
  const resolved = resolvePath(root);
  const target = existingMarkerPath(resolved) ?? markerPath(resolved);
  try {
    const raw = parseToml(fs.readFileSync(target, "utf8"));
    if (raw.version !== 1) throw new Error("unsupported marker version");
    if (raw.id === undefined) throw new Error("marker has no id");
    const id = String(raw.id);
    if (!isUuid(id)) throw new Error("marker id is not a uuid");
    if (raw.name === undefined) throw new Error("marker has no name");
    let scan = ScanConfig.parse(raw.scan ?? {});
    // A marker still carrying an older default include list is upgraded to the
    // current one, so a project written before a language was supported picks it
    // up. An include list the user has edited is left exactly as written -- and
    // the file itself is never rewritten either way.
    if (isLegacyDefaultIncludes(scan.include)) {
      scan = { ...scan, include: [...DEFAULT_INCLUDES] };
    }
    return ProjectInfo.parse({
      version: raw.version,
      id,
      name: String(raw.name),
      root: resolved,
      scan,
    });
  } catch (cause) {
    throw new CodeIndexingError(
      "PROJECT_NOT_FOUND",
      `Invalid or missing project marker: ${target}`,
      { path: target },
      { cause },
    );
  }
}

function isLegacyDefaultIncludes(include: readonly string[]): boolean {
  return [LEGACY_DEFAULT_INCLUDES_V1, LEGACY_DEFAULT_INCLUDES_V2, LEGACY_DEFAULT_INCLUDES_V3].some(
    (legacy) =>
      legacy.length === include.length &&
      legacy.every((pattern, index) => include[index] === pattern),
  );
}

/**
 * The spellings Python's `UUID()` accepts, which is what wrote these markers.
 *
 * Wider than the canonical form on purpose: a marker hand-edited to
 * `{...}` or a bare 32-hex-digit id validated under the Python build and must
 * keep validating, or the project would read as uninitialized after an upgrade.
 */
function isUuid(value: string): boolean {
  const stripped = value
    .trim()
    .replace(/^urn:uuid:/i, "")
    .replace(/^\{(.*)\}$/, "$1")
    .replaceAll("-", "");
  return /^[0-9a-f]{32}$/i.test(stripped);
}

export function findProjectRoot(start: string): string | null {
  let current = resolvePath(start);
  if (isFile(current)) current = path.dirname(current);
  for (;;) {
    if (existingMarkerPath(current) !== null) return current;
    const parent = path.dirname(current);
    if (parent === current) return null;
    current = parent;
  }
}

export class ProjectResolver {
  readonly #projects: readonly ProjectInfo[];

  constructor(projects: Iterable<ProjectInfo>) {
    this.#projects = [...projects];
  }

  resolve({
    explicit,
    roots = [],
    cwd,
  }: {
    explicit?: string | undefined;
    roots?: readonly string[];
    cwd?: string | undefined;
  } = {}): ProjectInfo {
    if (explicit) return this.#resolveExplicit(explicit);

    const marked = this.#markedProjects(roots);
    if (marked.length === 1) return marked[0] as ProjectInfo;
    if (marked.length > 1) {
      throw new CodeIndexingError(
        "AMBIGUOUS_PROJECT",
        "Multiple MCP roots contain initialized projects",
        { projects: marked.map((project) => project.id) },
      );
    }

    if (cwd !== undefined) {
      const root = findProjectRoot(cwd);
      if (root !== null) return this.#byRootOrMarker(root);
    }
    throw new CodeIndexingError(
      "PROJECT_NOT_FOUND",
      "No active CodeIndexing project was detected; pass an explicit project id, name, or " +
        "path, or run init_project for this directory",
      { searched_roots: [...roots] },
    );
  }

  #resolveExplicit(explicit: string): ProjectInfo {
    const direct = this.#projects.filter(
      (project) => project.id === explicit || project.name === explicit,
    );
    if (direct.length === 1) return direct[0] as ProjectInfo;
    if (direct.length > 1) {
      throw new CodeIndexingError("AMBIGUOUS_PROJECT", `Project name is ambiguous: ${explicit}`, {
        projects: direct.map((project) => project.id),
      });
    }
    const candidate = expandUser(explicit);
    if (fs.existsSync(candidate)) {
      const root = findProjectRoot(candidate);
      if (root !== null) return this.#byRootOrMarker(root);
    }
    throw new CodeIndexingError("PROJECT_NOT_FOUND", `Unknown project: ${explicit}`);
  }

  #markedProjects(roots: readonly string[]): ProjectInfo[] {
    const found = new Map<string, ProjectInfo>();
    for (const candidate of roots) {
      const root = findProjectRoot(candidate);
      if (root !== null) {
        const project = this.#byRootOrMarker(root);
        found.set(project.id, project);
      }
    }
    return [...found.values()];
  }

  #byRootOrMarker(root: string): ProjectInfo {
    const resolved = resolvePath(root);
    for (const project of this.#projects) {
      if (sameProjectRoot(project.root, resolved)) return project;
    }
    return readProjectMarker(resolved);
  }
}
