/**
 * S2 -- Grammar coverage.
 *
 * Load all the languages' grammars under Node bindings, compile the committed
 * `.scm` queries against them, and run them over the extractor fixture corpus.
 *
 * Scope note: the plan's S2 also asks to "diff chunk output against the Python
 * snapshot". That diff cannot be produced in Phase 0 -- `extractor_snapshot.json`
 * records the output of the chunking algorithm, not of the queries, and that
 * algorithm is 1,583 lines of Phase 2 work. What this spike answers is the
 * question Phase 0 actually needs answered: does every grammar exist as an npm
 * package, load under Bun, and accept the query pack we already ship without
 * edits? Capture counts are recorded per language so Phase 2 has a baseline to
 * diff against when the chunker lands.
 *
 * The interesting part is the grammar *sourcing*. The Python build takes three
 * Godot grammars from `tree-sitter-language-pack`; on npm two of them have
 * dedicated packages and only `gdshader` needs a pack, which is why the pack
 * dependency is narrower here than on the Python side.
 */

import { readFileSync } from "node:fs";
import { readdirSync } from "node:fs";
import { extname, join } from "node:path";
import { type GrammarResolution, resolveGrammar } from "./grammar-loader.ts";
import { Spike, describe, repoRoot } from "./harness.ts";

const ROOT = repoRoot();
const QUERY_DIR = join(ROOT, "src", "code_indexing_mcp", "queries");
const REFERENCE_QUERY_DIR = join(ROOT, "src", "code_indexing_mcp", "reference_queries");
const CORPUS_DIR = join(ROOT, "tests", "fixtures", "extractor_corpus");

/**
 * `extractor.py::_languages`, resolved to npm packages.
 *
 * `specifier` is the npm module and `member` the export to read the grammar
 * from -- `tree-sitter-typescript` publishes two grammars from one package,
 * exactly as the PyPI distribution does.
 */
interface GrammarSource {
  readonly specifier: string;
  readonly member?: string;
  /**
   * Take this grammar from `@kreuzberg/tree-sitter-language-pack` rather than
   * from a package of its own. Used only where no dedicated package exists,
   * mirroring why the Python build reaches for the PyPI pack.
   */
  readonly packLanguage?: string;
  /** Recorded so the results table can show the PyPI/npm version drift. */
  readonly pythonPackage: string;
}

const GRAMMARS: Record<string, GrammarSource> = {
  python: {
    specifier: "tree-sitter-python",
    pythonPackage: "tree-sitter-python",
  },
  java: { specifier: "tree-sitter-java", pythonPackage: "tree-sitter-java" },
  javascript: {
    specifier: "tree-sitter-javascript",
    pythonPackage: "tree-sitter-javascript",
  },
  typescript: {
    specifier: "tree-sitter-typescript",
    member: "typescript",
    pythonPackage: "tree-sitter-typescript",
  },
  tsx: {
    specifier: "tree-sitter-typescript",
    member: "tsx",
    pythonPackage: "tree-sitter-typescript",
  },
  csharp: {
    specifier: "tree-sitter-c-sharp",
    pythonPackage: "tree-sitter-c-sharp",
  },
  sql: {
    specifier: "@derekstride/tree-sitter-sql",
    pythonPackage: "tree-sitter-sql",
  },
  go: { specifier: "tree-sitter-go", pythonPackage: "tree-sitter-go" },
  terraform: {
    specifier: "@tree-sitter-grammars/tree-sitter-hcl",
    pythonPackage: "tree-sitter-hcl",
  },
  rust: { specifier: "tree-sitter-rust", pythonPackage: "tree-sitter-rust" },
  c: { specifier: "tree-sitter-c", pythonPackage: "tree-sitter-c" },
  cpp: { specifier: "tree-sitter-cpp", pythonPackage: "tree-sitter-cpp" },
  lua: {
    specifier: "@tree-sitter-grammars/tree-sitter-lua",
    pythonPackage: "tree-sitter-lua",
  },
  gdscript: {
    specifier: "tree-sitter-gdscript",
    pythonPackage: "tree-sitter-language-pack",
  },
  // The one language with no dedicated npm package, so it comes from the pack
  // -- the same source, and the same reason, as the Python build.
  gdshader: {
    specifier: "@kreuzberg/tree-sitter-language-pack",
    packLanguage: "gdshader",
    pythonPackage: "tree-sitter-language-pack",
  },
  godot_resource: {
    specifier: "tree-sitter-godot-resource",
    pythonPackage: "tree-sitter-language-pack",
  },
  yaml: {
    specifier: "@tree-sitter-grammars/tree-sitter-yaml",
    pythonPackage: "tree-sitter-yaml",
  },
  json: { specifier: "tree-sitter-json", pythonPackage: "tree-sitter-json" },
};

/** `scanner.py::LANGUAGES`, restricted to the extensions the corpus uses. */
const EXTENSIONS: Record<string, string> = {
  ".py": "python",
  ".java": "java",
  ".js": "javascript",
  ".ts": "typescript",
  ".tsx": "tsx",
  ".cs": "csharp",
  ".gd": "gdscript",
  ".gdshader": "gdshader",
  ".tscn": "godot_resource",
  ".sql": "sql",
  ".yaml": "yaml",
  ".json": "json",
  ".go": "go",
  ".tf": "terraform",
  ".rs": "rust",
  ".c": "c",
  ".cpp": "cpp",
  ".lua": "lua",
};

/**
 * Grammars known to be unavailable on a platform, reported as skips.
 *
 * A declared gap keeps the spike's verdict honest -- it reports SKIP rather
 * than PASS, so the run is visibly incomplete -- without leaving CI red on
 * something no commit here can fix. Removing an entry is how the gap gets
 * closed; nothing else in this file needs to change.
 */
const KNOWN_GAPS: Array<{ language: string; platform: NodeJS.Platform; why: string }> = [
  {
    language: "gdshader",
    platform: "win32",
    why:
      "@kreuzberg/tree-sitter-language-pack cannot load its win32-x64 binding " +
      '("LoadLibrary failed: The specified module could not be found", which is ' +
      "a missing transitive DLL rather than a missing addon). This is a " +
      "regression against the Python build, whose PyPI language pack ships " +
      "working Windows wheels, and must be resolved before Phase 2 ships",
  },
];

function knownGap(language: string): string | undefined {
  return KNOWN_GAPS.find((gap) => gap.language === language && gap.platform === process.platform)
    ?.why;
}

const spike = new Spike("s2", "Grammar coverage");
spike.header();

const { default: Parser } = await import("tree-sitter");
// node-tree-sitter hangs Query off the Parser export rather than shipping a
// named one, and its types do not describe it.
const Query = (Parser as unknown as { Query: QueryConstructor }).Query;

type QueryConstructor = new (
  language: unknown,
  source: string,
) => { captures: (node: unknown) => Array<{ name: string }> };

const loaded = new Map<string, GrammarResolution>();

async function loadGrammar(name: string): Promise<GrammarResolution> {
  const cached = loaded.get(name);
  if (cached !== undefined) return cached;

  const source = GRAMMARS[name];
  if (source === undefined) throw new Error(`no npm source recorded for ${name}`);

  const resolved = await resolveGrammar(source.specifier, source.member, source.packLanguage);
  loaded.set(name, resolved);
  return resolved;
}

const corpusFiles = readdirSync(CORPUS_DIR);

function fixtureFor(languageName: string): string | undefined {
  return corpusFiles.find((file) => EXTENSIONS[extname(file)] === languageName);
}

// One check per language, so a single missing grammar reports as one failure
// against a full table rather than aborting the run.
for (const name of Object.keys(GRAMMARS)) {
  await spike.check(`${name}: grammar loads, query compiles, corpus parses`, async () => {
    const gap = knownGap(name);
    if (gap !== undefined) {
      // Confirm the gap is still real before skipping, so an entry that has
      // quietly been fixed upstream shows up as a skip that should be deleted
      // rather than silently masking a working grammar.
      try {
        await loadGrammar(name);
      } catch {
        return { skip: `known gap on ${process.platform}: ${gap}` };
      }
      throw new Error(
        `the recorded ${process.platform} gap for ${name} no longer reproduces — remove it from KNOWN_GAPS`,
      );
    }
    const grammar = await loadGrammar(name);

    const querySource = readFileSync(join(QUERY_DIR, `${name}.scm`), "utf8");
    let query: { captures: (node: unknown) => Array<{ name: string }> };
    try {
      query = new Query(grammar.language, querySource);
    } catch (error) {
      throw new Error(`queries/${name}.scm did not compile: ${describe(error)}`);
    }

    const fixture = fixtureFor(name);
    if (fixture === undefined) {
      return `loaded via ${grammar.via}, query compiled; no corpus fixture for this language`;
    }

    const parser = new Parser();
    parser.setLanguage(grammar.language as never);
    const text = readFileSync(join(CORPUS_DIR, fixture), "utf8");
    const tree = parser.parse(text);
    const captures = query.captures(tree.rootNode);
    const kinds = [...new Set(captures.map((capture) => capture.name))].sort();

    if (captures.length === 0) {
      throw new Error(`query captured nothing in ${fixture}`);
    }
    return (
      `${grammar.detail}; ${fixture} -> ${captures.length} captures ` +
      `over ${kinds.length} capture names (${kinds.slice(0, 4).join(", ")}${kinds.length > 4 ? ", ..." : ""})`
    );
  });
}

// The reference queries drive `reference_service.py` and are a separate query
// pack compiled against the same grammars.
const referenceQueries = readdirSync(REFERENCE_QUERY_DIR)
  .filter((file) => file.endsWith(".scm"))
  .map((file) => file.slice(0, -".scm".length))
  .sort();

for (const name of referenceQueries) {
  await spike.check(`${name}: reference query compiles`, async () => {
    const grammar = await loadGrammar(name);
    const source = readFileSync(join(REFERENCE_QUERY_DIR, `${name}.scm`), "utf8");
    const query = new Query(grammar.language, source);
    const fixture = fixtureFor(name);
    if (fixture === undefined) return "compiled; no corpus fixture for this language";
    const parser = new Parser();
    parser.setLanguage(grammar.language as never);
    const tree = parser.parse(readFileSync(join(CORPUS_DIR, fixture), "utf8"));
    const captures = query.captures(tree.rootNode);
    return `compiled; ${fixture} -> ${captures.length} reference captures`;
  });
}

spike.finish();
