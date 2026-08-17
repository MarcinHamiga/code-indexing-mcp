/**
 * The `pathlib` operations the port leans on, with `pathlib`'s semantics.
 *
 * Node models a path as a string and `pathlib` models it as an object, so most
 * of the translation is unremarkable. Four operations are not, because their
 * behaviour is load-bearing somewhere: `resolvePath` must succeed on a path that
 * does not exist yet, `sameFile` must see through two spellings of one
 * directory, `rootedUnder` must decide containment by the boundary directory
 * rather than by string prefix, and `fileIdentity` must produce a string a
 * *different process* -- possibly still the Python build, mid-migration -- would
 * produce too.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

/**
 * `Path.expanduser`: a leading bare `~` becomes the home directory.
 *
 * `~user` is left untouched. Python expands it only when the platform can look
 * the account up, and Node offers no such lookup; passing it through unchanged
 * is the same answer Python gives when the lookup fails.
 */
export function expandUser(value: string): string {
  if (value === "~") return os.homedir();
  if (value.startsWith("~/") || value.startsWith(`~${path.sep}`)) {
    return path.join(os.homedir(), value.slice(2));
  }
  return value;
}

/**
 * `Path.expanduser().resolve()`: absolute, with symlinks followed.
 *
 * Non-strict, as `Path.resolve()` has been since Python 3.6: the longest
 * existing prefix is resolved and whatever does not exist yet is appended
 * unchanged. Callers depend on this -- a project marker is resolved before the
 * directory holding it is created.
 */
export function resolvePath(value: string): string {
  const absolute = path.resolve(expandUser(value));
  const trailing: string[] = [];
  let head = absolute;
  for (;;) {
    try {
      return path.resolve(fs.realpathSync(head), ...trailing);
    } catch {
      const parent = path.dirname(head);
      if (parent === head) return absolute;
      trailing.unshift(path.basename(head));
      head = parent;
    }
  }
}

/**
 * `Path.parts`: the root, when there is one, followed by each named component.
 *
 * `/a/b` yields `["/", "a", "b"]` and `C:\a\b` yields `["C:\\", "a", "b"]`, so a
 * component count means the same thing on both platforms.
 */
export function pathParts(value: string): string[] {
  const { root } = path.parse(value);
  const named = value
    .slice(root.length)
    .split(/[\\/]+/)
    .filter((part) => part !== "");
  return root === "" ? named : [root, ...named];
}

/** The filesystem's own identity for a path, or null when it cannot be read. */
function statIdentity(value: string): { dev: bigint; ino: bigint } | null {
  try {
    const info = fs.statSync(value, { bigint: true });
    return { dev: info.dev, ino: info.ino };
  } catch {
    return null;
  }
}

/**
 * `Path.samefile`, falling back to resolved equality when either path is gone.
 *
 * The fallback is what makes this usable on a project root that was renamed out
 * from under a registration: the answer is then a string comparison, which is
 * the best available and never worse than raising.
 */
export function sameFile(left: string, right: string): boolean {
  const leftIdentity = statIdentity(expandUser(left));
  const rightIdentity = statIdentity(expandUser(right));
  if (leftIdentity !== null && rightIdentity !== null) {
    return leftIdentity.dev === rightIdentity.dev && leftIdentity.ino === rightIdentity.ino;
  }
  return resolvePath(left) === resolvePath(right);
}

/**
 * Whether *child* names a location strictly inside *parent*.
 *
 * The boundary directory is compared by filesystem identity rather than by
 * string equality, so differently-cased spellings of one directory (common on
 * macOS and Windows) count as containment exactly as `sameFile` counts them as
 * equality. Both paths must already be resolved; a boundary that cannot be
 * stat'ed means no containment, which keeps an unknown answer from reading as a
 * yes.
 */
export function rootedUnder(parent: string, child: string): boolean {
  const parentParts = pathParts(parent);
  const childParts = pathParts(child);
  if (childParts.length <= parentParts.length) return false;
  const boundary = path.join(...childParts.slice(0, parentParts.length));
  const boundaryIdentity = statIdentity(boundary);
  const parentIdentity = statIdentity(parent);
  if (boundaryIdentity === null || parentIdentity === null) return false;
  return boundaryIdentity.dev === parentIdentity.dev && boundaryIdentity.ino === parentIdentity.ino;
}

/**
 * A cross-process identity for an existing directory.
 *
 * Rendered as `inode:<dev>:<ino>` from the same `stat` fields Python reads, so a
 * TypeScript daemon and a Python CLI agree about which directory they are
 * talking about while both are installed. `dev` and `ino` are read as bigints
 * because a Windows file index does not fit in a double, and a rounded identity
 * would silently collide.
 */
export function fileIdentity(value: string): string {
  const resolved = resolvePath(value);
  const identity = statIdentity(resolved);
  if (identity === null) return `path:${resolved}`;
  return `inode:${identity.dev}:${identity.ino}`;
}

/** `Path.is_relative_to`: pure component-prefix containment, no filesystem. */
export function isRelativeTo(value: string, other: string): boolean {
  const relative = path.relative(other, value);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}
