/**
 * Loading tree-sitter grammars under Bun.
 *
 * Most grammar packages resolve their native addon through `node-gyp-build`,
 * but every recent one carries a hand-written fast path guarded on
 * `process.versions.bun`, and two things go wrong with it:
 *
 *  - The scoped packages (`@tree-sitter-grammars/*`) build the filename from
 *    the *unscoped* grammar name, while the file they ship is scope-mangled
 *    (`@tree-sitter-grammars+tree-sitter-yaml.node`). The require fails on a
 *    path that never existed.
 *  - The ESM-flavored packages (`tree-sitter-c-sharp`) reach the addon with
 *    `await import(...)`, and Bun rejects importing a Node-API module outright:
 *    "use require() or process.dlopen instead of import".
 *
 * Both are upstream package bugs rather than Bun N-API gaps -- the addons
 * themselves load and run fine under Bun once reached correctly, which is what
 * `resolveGrammar`'s fallback does: locate the package root, then require
 * whatever `.node` file actually sits in the platform's prebuild directory.
 *
 * Phase 2 inherits this module. The alternative -- pinning to Node for
 * extraction, or waiting on upstream fixes for four packages -- costs far more
 * than the twenty lines below.
 */

import { readdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);

export interface GrammarResolution {
  readonly language: unknown;
  /** Which path produced the grammar, for the spike's results table. */
  readonly via: "entrypoint" | "prebuild-fallback" | "language-pack";
  readonly detail: string;
}

/**
 * Resolve a grammar from `@kreuzberg/tree-sitter-language-pack`.
 *
 * The npm analogue of the PyPI `tree-sitter-language-pack` the Python build
 * already leans on, and used here for the same reason and under the same rule:
 * only for grammars that have no dedicated package of their own. Everything
 * else keeps its own package so its version moves independently, which is the
 * rationale `pyproject.toml` records for the PyPI side.
 *
 * Grammars are fetched on first use and cached, exactly like the PyPI pack's
 * Godot download — so CI must warm the cache before running offline, the way
 * the Python workflow already does.
 */
async function loadFromPack(language: string): Promise<unknown> {
  const pack = require("@kreuzberg/tree-sitter-language-pack") as {
    manifestLanguages: () => string[];
    downloadedLanguages: () => string[];
    download: (names: string[]) => number;
    getLanguage: (name: string) => unknown;
  };
  if (!pack.manifestLanguages().includes(language)) {
    throw new Error(`the language pack does not publish "${language}"`);
  }
  if (!pack.downloadedLanguages().includes(language)) {
    pack.download([language]);
  }
  return pack.getLanguage(language);
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
 * (`@derekstride/tree-sitter-sql` compiles in its `install` script), where the
 * artifact is named after the gyp target rather than the package.
 */
function loadPrebuild(specifier: string): { language: unknown; file: string } {
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
    if (file !== undefined) {
      return { language: require(join(directory, file)) as unknown, file };
    }
  }
  throw new Error(`${specifier} has no addon for ${process.platform}-${process.arch}`);
}

/**
 * Resolve a grammar, preferring the package's own entrypoint so that any
 * metadata it attaches (node type info, bundled queries) is present, and
 * falling back to the raw addon when that entrypoint cannot run here.
 *
 * `packLanguage` names a grammar to take from the language pack instead of
 * from a package of its own.
 */
export async function resolveGrammar(
  specifier: string,
  member?: string,
  packLanguage?: string,
): Promise<GrammarResolution> {
  if (packLanguage !== undefined) {
    return {
      language: await loadFromPack(packLanguage),
      via: "language-pack",
      detail: `@kreuzberg/tree-sitter-language-pack: ${packLanguage}`,
    };
  }
  try {
    const module = (await import(specifier)) as Record<string, unknown> & {
      default?: Record<string, unknown>;
    };
    const root = (module.default ?? module) as Record<string, unknown>;
    const language = member === undefined ? root : root[member];
    if (language === undefined) throw new Error(`no export "${member}"`);
    return { language, via: "entrypoint", detail: "package entrypoint" };
  } catch (entrypointError) {
    if (member !== undefined) {
      // A multi-grammar package's members only exist on its entrypoint, so
      // there is nothing to fall back to.
      throw entrypointError;
    }
    const { language, file } = loadPrebuild(specifier);
    return {
      language,
      via: "prebuild-fallback",
      detail: `entrypoint failed, loaded ${file} directly`,
    };
  }
}
