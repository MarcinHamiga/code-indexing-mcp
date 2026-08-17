/**
 * Tree-sitter grammar sourcing -- the npm resolution of `extractor.py::_languages`.
 *
 * Two things differ from the Python side, both settled by spike S2:
 *
 *  - **Sourcing.** The Python build takes all three Godot grammars from
 *    `tree-sitter-language-pack` because PyPI publishes none of them
 *    separately. On npm, `gdscript` and `godot_resource` have dedicated
 *    packages and only `gdshader` needs a pack, so the pack dependency is
 *    narrower here and every other grammar's version moves on its own.
 *  - **Loading.** Several packages' entrypoints cannot run under Bun -- the
 *    scoped ones build their addon filename from the unscoped grammar name
 *    while shipping a scope-mangled file, and the ESM-flavored ones reach the
 *    addon with `await import`, which Bun refuses for Node-API modules. Both
 *    are upstream packaging bugs rather than Bun N-API gaps: the addons load
 *    and run once reached correctly, which is what {@link loadPrebuild} does.
 *
 * One capability is deliberately absent: `gdshader` on Windows (§5.5 of the
 * migration plan). The language pack cannot load its `win32-x64` binding, and
 * it is the only npm source for that grammar. {@link grammarFor} reports the
 * language as unavailable there rather than throwing, and the scanner drops
 * `.gdshader` files on that platform the way it drops an unsupported
 * extension -- failing an index over a Godot repository would be far worse
 * than omitting its shaders.
 */

import { readdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import type Parser from "tree-sitter";

const require = createRequire(import.meta.url);

/** Opaque to us: whatever the addon hands back for `Parser.setLanguage`. */
export type Grammar = Parser.Language;

interface GrammarSource {
  readonly specifier: string;
  /** `tree-sitter-typescript` publishes two grammars from one package. */
  readonly member?: string;
  /** Take this grammar from the language pack, for want of a package of its own. */
  readonly packLanguage?: string;
}

const GRAMMARS: Record<string, GrammarSource> = {
  python: { specifier: "tree-sitter-python" },
  java: { specifier: "tree-sitter-java" },
  javascript: { specifier: "tree-sitter-javascript" },
  typescript: { specifier: "tree-sitter-typescript", member: "typescript" },
  tsx: { specifier: "tree-sitter-typescript", member: "tsx" },
  csharp: { specifier: "tree-sitter-c-sharp" },
  sql: { specifier: "@derekstride/tree-sitter-sql" },
  go: { specifier: "tree-sitter-go" },
  terraform: { specifier: "@tree-sitter-grammars/tree-sitter-hcl" },
  rust: { specifier: "tree-sitter-rust" },
  c: { specifier: "tree-sitter-c" },
  cpp: { specifier: "tree-sitter-cpp" },
  lua: { specifier: "@tree-sitter-grammars/tree-sitter-lua" },
  gdscript: { specifier: "tree-sitter-gdscript" },
  gdshader: {
    specifier: "@kreuzberg/tree-sitter-language-pack",
    packLanguage: "gdshader",
  },
  godot_resource: { specifier: "tree-sitter-godot-resource" },
  yaml: { specifier: "@tree-sitter-grammars/tree-sitter-yaml" },
  json: { specifier: "tree-sitter-json" },
};

/**
 * Languages with no grammar on a platform.
 *
 * A declared gap is a *supported state*, not an error: `grammarFor` returns
 * undefined and the scanner treats the extension as unsupported. Nothing here
 * is a permanent exemption -- {@link unavailableLanguages} is derived from the
 * same table, so release notes and the scanner cannot drift apart.
 */
const KNOWN_GAPS: ReadonlyArray<{ language: string; platform: NodeJS.Platform }> = [
  // "LoadLibrary failed: The specified module could not be found" -- a missing
  // transitive DLL in the pack's win32-x64 binding. Accepted 2026-08-17 rather
  // than shipping a VC++ redistributable, running a second WASM parser stack
  // for one language, or vendoring the grammar's C sources.
  { language: "gdshader", platform: "win32" },
];

/** Languages that cannot be parsed on this platform, in stable order. */
export function unavailableLanguages(platform: NodeJS.Platform = process.platform): string[] {
  return KNOWN_GAPS.filter((gap) => gap.platform === platform)
    .map((gap) => gap.language)
    .sort();
}

const cache = new Map<string, Grammar>();

/**
 * The grammar for one language name, or undefined where the platform has none.
 *
 * Grammars are compiled once per process and cached: the daemon serves each
 * client on its own thread over one extractor, and loading an addon per file
 * would dwarf the parse itself.
 */
export function grammarFor(language: string): Grammar | undefined {
  const cached = cache.get(language);
  if (cached !== undefined) return cached;
  if (unavailableLanguages().includes(language)) return undefined;
  const source = GRAMMARS[language];
  if (source === undefined) return undefined;
  const grammar = resolveGrammar(source);
  cache.set(language, grammar);
  return grammar;
}

/** Every language this build can parse here, in stable order. */
export function supportedLanguages(): string[] {
  const unavailable = new Set(unavailableLanguages());
  return Object.keys(GRAMMARS)
    .filter((language) => !unavailable.has(language))
    .sort();
}

function resolveGrammar(source: GrammarSource): Grammar {
  if (source.packLanguage !== undefined) return loadFromPack(source.packLanguage);
  try {
    const module = require(source.specifier) as Record<string, unknown> & {
      default?: Record<string, unknown>;
    };
    const root = (module.default ?? module) as Record<string, unknown>;
    const grammar = source.member === undefined ? root : root[source.member];
    if (grammar === undefined) throw new Error(`no export "${source.member}"`);
    return grammar as Grammar;
  } catch (entrypointError) {
    if (source.member !== undefined) {
      // A multi-grammar package's members exist only on its entrypoint, so
      // there is no raw addon to fall back to.
      throw entrypointError;
    }
    return loadPrebuild(source.specifier);
  }
}

/** The directory holding a package's own files, found without importing it. */
function packageRoot(specifier: string): string {
  return dirname(require.resolve(`${specifier}/package.json`));
}

/**
 * `require` the platform's addon directly, whatever it is named.
 *
 * Searches the same two places `node-gyp-build` does, in the same order: the
 * shipped prebuild for this platform, then the output of a local source build.
 * The second matters for grammars that publish no prebuilds at all
 * (`@derekstride/tree-sitter-sql` compiles in its install script), where the
 * artifact is named after the gyp target rather than the package.
 */
function loadPrebuild(specifier: string): Grammar {
  const root = packageRoot(specifier);
  const candidates = [
    join(root, "prebuilds", `${process.platform}-${process.arch}`),
    join(root, "build", "Release"),
  ];
  for (const directory of candidates) {
    let entries: string[];
    try {
      entries = readdirSync(directory);
    } catch {
      continue;
    }
    const file = entries.find((entry) => entry.endsWith(".node"));
    if (file !== undefined) return require(join(directory, file)) as Grammar;
  }
  throw new Error(`${specifier} has no addon for ${process.platform}-${process.arch}`);
}

/**
 * The pack fetches parsers from a GitHub release on first use, and that
 * endpoint has transient outages -- CI once saw one drop every connection for
 * tens of seconds -- so the backoff window (1+2+4+8+16s) is sized to ride out
 * such a window rather than a single failed request. Mirrors
 * `_PACK_DOWNLOAD_ATTEMPTS`/`_PACK_DOWNLOAD_BACKOFF_SECONDS`.
 */
export const PACK_DOWNLOAD_ATTEMPTS = 6;
const PACK_DOWNLOAD_BACKOFF_MS = 1_000;

/**
 * Run `load`, retrying a failure with exponential backoff.
 *
 * Exported because the retry is the interesting part and the thing that can
 * regress: a suite that had to stand up a fake release endpoint to exercise it
 * would test the fake instead.
 */
export function withDownloadRetry<T>(
  load: () => T,
  options: { attempts?: number; backoffMs?: number } = {},
): T {
  const attempts = options.attempts ?? PACK_DOWNLOAD_ATTEMPTS;
  const backoffMs = options.backoffMs ?? PACK_DOWNLOAD_BACKOFF_MS;
  for (let attempt = 0; ; attempt += 1) {
    try {
      return load();
    } catch (error) {
      if (attempt >= attempts - 1) throw error;
      if (backoffMs > 0) sleepSync(backoffMs * 2 ** attempt);
    }
  }
}

/**
 * Resolve a grammar the pack downloads on first use, retrying transiently.
 *
 * Downloaded parsers are cached on disk, exactly like the PyPI pack's Godot
 * download -- which is why CI warms the grammar cache before running offline.
 */
function loadFromPack(language: string): Grammar {
  const pack = require("@kreuzberg/tree-sitter-language-pack") as {
    manifestLanguages: () => string[];
    downloadedLanguages: () => string[];
    download: (names: string[]) => number;
    getLanguage: (name: string) => unknown;
  };
  if (!pack.manifestLanguages().includes(language)) {
    throw new Error(`the language pack does not publish "${language}"`);
  }
  return withDownloadRetry(() => {
    if (!pack.downloadedLanguages().includes(language)) pack.download([language]);
    return pack.getLanguage(language) as Grammar;
  });
}

/**
 * Block the current thread.
 *
 * Extraction is synchronous all the way up (as it is in Python), and this is
 * the one place inside it that has to wait on a network round trip. `Atomics.wait`
 * is the only portable synchronous sleep on either runtime.
 */
function sleepSync(milliseconds: number): void {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, milliseconds);
}
