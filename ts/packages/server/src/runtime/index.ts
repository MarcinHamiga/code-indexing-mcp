/**
 * The one directory allowed to import Bun-only APIs.
 *
 * Everything else in this package imports `node:`-namespaced APIs and Web
 * standards only, so that any single process can be run under Node if a native
 * addon hits a Bun N-API gap (migration plan S3, and the first row of the risk
 * register). Biome enforces the boundary: `bun:*` specifiers are a lint error
 * outside `src/runtime/`.
 *
 * Adapters land here as the phases that need them arrive -- `bun:sqlite`
 * behind the history store's interface in Phase 3, `Bun.spawn` behind the
 * worker launcher's in Phase 4.
 */

export type RuntimeName = "bun" | "node";

/**
 * Which runtime is executing this process.
 *
 * Read through the globalThis bag rather than the `Bun` global directly so the
 * check itself does not become a Bun-only reference under Node's type
 * stripping.
 */
export function runtimeName(): RuntimeName {
  const candidate = (globalThis as { Bun?: { version?: string } }).Bun;
  return typeof candidate?.version === "string" ? "bun" : "node";
}

/** The running runtime's own version string, for diagnostics and probe records. */
export function runtimeVersion(): string {
  const candidate = (globalThis as { Bun?: { version?: string } }).Bun;
  return candidate?.version ?? process.versions.node;
}
