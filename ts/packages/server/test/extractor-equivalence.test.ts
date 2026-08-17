/**
 * Output-equivalence gate for the ported extractor.
 *
 * `tests/fixtures/extractor_snapshot.json` records what the shipping Python
 * build emits for every fixture in the extractor corpus -- every chunk's kind,
 * symbol, byte range, line range, part index, and content digests, plus every
 * structural reference and declaration shape. The Python suite gates its own
 * refactors on that file; this suite is held to the *same* file, so a
 * divergence between the two builds fails here instead of surfacing later as a
 * search result the Python build would not have returned.
 *
 * That makes it the migration plan's §8 "golden fixtures" idea at its
 * strongest: the fixture is not a re-derived oracle, it is the shipping
 * build's own output, and the fingerprint covers exactly the fields chunk
 * identity is digested from. A chunk id that shifts silently invalidates every
 * stored chunk and breaks incremental indexing, which is a bug no type checker
 * can catch.
 *
 * The snapshot is regenerated only from Python
 * (`python -m tests.test_extractor_equivalence`), never from this side.
 */

import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { TreeSitterExtractor } from "../src/extractor.ts";
import type { ExtractionResult } from "../src/models.ts";
import { LANGUAGES, languageForExtension } from "../src/scanner.ts";
import { repositoryRoot } from "./helpers.ts";

const CORPUS_DIRECTORY = path.join(repositoryRoot(), "tests", "fixtures", "extractor_corpus");
const SNAPSHOT_PATH = path.join(repositoryRoot(), "tests", "fixtures", "extractor_snapshot.json");

interface Fingerprint {
  has_errors: boolean;
  chunks: unknown[][];
  references: unknown[][];
  declarations: unknown[][];
}

const snapshot: Record<string, Fingerprint> = JSON.parse(fs.readFileSync(SNAPSHOT_PATH, "utf8"));

function digest(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex").slice(0, 16);
}

/** Everything about a chunk that a consumer or a chunk id depends on. */
function chunkFingerprint(result: ExtractionResult): unknown[][] {
  return result.chunks.map((chunk) => [
    chunk.kind,
    chunk.symbol,
    chunk.qualified_symbol,
    chunk.parent_symbol,
    chunk.start_byte,
    chunk.end_byte,
    chunk.start_line,
    chunk.end_line,
    chunk.part_index,
    digest(chunk.content),
    digest(chunk.embedding_text),
    digest(chunk.search_text),
    digest(chunk.embedding_prefix),
    digest(chunk.search_suffix),
  ]);
}

/** Everything a resolver or a rename depends on for one structural reference. */
function referenceFingerprint(result: ExtractionResult): unknown[][] {
  return result.references.map((reference) => [
    reference.kind,
    reference.target_name,
    reference.written_name,
    reference.module_path,
    reference.imported_name,
    reference.alias,
    reference.start_byte,
    reference.end_byte,
    reference.source_qualified_symbol,
    reference.call_shape === null
      ? null
      : [
          reference.call_shape.positional_count,
          reference.call_shape.keywords,
          reference.call_shape.has_positional_spread,
          reference.call_shape.has_keyword_spread,
          reference.call_shape.type_argument_count,
          reference.call_shape.constructor,
        ],
  ]);
}

/** Everything a rename/refactor depends on for one declaration shape. */
function declarationFingerprint(result: ExtractionResult): unknown[][] {
  return result.declarations.map((declaration) => [
    declaration.qualified_symbol,
    declaration.kind,
    declaration.parameters.map((parameter) => [
      parameter.name,
      parameter.kind,
      parameter.required,
      parameter.position,
    ]),
  ]);
}

const extractor = new TreeSitterExtractor();

function fingerprintFor(name: string): Fingerprint {
  const file = path.join(CORPUS_DIRECTORY, name);
  const language = LANGUAGES[path.extname(name).toLowerCase()] as string;
  const result = extractor.extract(
    name,
    language,
    new Uint8Array(fs.readFileSync(file).buffer.slice(0)),
  );
  return {
    has_errors: result.has_errors,
    chunks: chunkFingerprint(result),
    references: referenceFingerprint(result),
    declarations: declarationFingerprint(result),
  };
}

/** Corpus files this platform has no grammar for, so their absence is expected. */
function unsupportedOnThisPlatform(name: string): boolean {
  return languageForExtension(path.extname(name)) === undefined;
}

describe("parity with the Python extractor snapshot", () => {
  test("the corpus covers every language the scanner maps", () => {
    const languages = new Set(
      fs
        .readdirSync(CORPUS_DIRECTORY)
        .map((name) => LANGUAGES[path.extname(name).toLowerCase()] as string),
    );

    expect([...languages].sort()).toEqual([...new Set(Object.values(LANGUAGES))].sort());
  });

  test("the snapshot names exactly the corpus files", () => {
    expect(Object.keys(snapshot).sort()).toEqual(fs.readdirSync(CORPUS_DIRECTORY).sort());
  });

  const names = Object.keys(snapshot).sort();
  for (const name of names) {
    test(`${name} extracts exactly what Python extracts`, () => {
      if (unsupportedOnThisPlatform(name)) {
        // §5.5: GDShader has no grammar on Windows and the scanner skips those
        // files there, so there is no output to compare. Every other platform
        // still holds the file to the snapshot.
        expect(languageForExtension(path.extname(name))).toBeUndefined();
        return;
      }
      const actual = fingerprintFor(name);
      const expected = snapshot[name] as Fingerprint;

      expect(actual.has_errors).toBe(expected.has_errors);
      expect(actual.chunks).toEqual(expected.chunks);
      expect(actual.references).toEqual(expected.references);
      expect(actual.declarations).toEqual(expected.declarations);
    });
  }
});

describe("scaling", () => {
  function generatedSource(definitions: number): Uint8Array {
    const body = Array.from(
      { length: definitions },
      (_, index) => `def f${index}(a, b):\n    return a + b + ${index}\n`,
    ).join("\n");
    return new TextEncoder().encode(body);
  }

  test.each([500, 2000])(
    "%d definitions stay within a linear time budget",
    (definitions: number) => {
      // Deliberately loose bounds -- roughly 40x the measured cost -- so they
      // survive a loaded CI machine while still failing hard if the
      // per-definition rebuilds that made extraction quadratic come back.
      const local = new TreeSitterExtractor();
      const source = generatedSource(definitions);
      local.extract("warm.py", "python", source); // warm the query cache

      const started = performance.now();
      const result = local.extract("generated.py", "python", source);
      const elapsed = (performance.now() - started) / 1000;

      expect(result.chunks).toHaveLength(definitions);
      expect(elapsed).toBeLessThan(definitions / 1000);
    },
  );

  test("a definition-dense file at the scan ceiling is not quadratic", () => {
    const local = new TreeSitterExtractor();
    const source = generatedSource(16_384);
    expect(source.length).toBeLessThan(1_048_576);

    const started = performance.now();
    const result = local.extract("huge.py", "python", source);
    const elapsed = (performance.now() - started) / 1000;

    expect(result.chunks).toHaveLength(16_384);
    expect(elapsed).toBeLessThan(5.0);
  });
});
