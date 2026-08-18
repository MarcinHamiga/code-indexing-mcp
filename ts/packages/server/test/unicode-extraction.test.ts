/**
 * Non-ASCII extraction, held to the Python build's own offsets.
 *
 * This exists because of the single most dangerous difference in the port:
 * tree-sitter's Node binding reports **UTF-16 code-unit indices** where the
 * Python binding reports **UTF-8 byte offsets**. Those offsets are digested
 * into chunk ids, stored on every chunk and reference row, and handed to
 * callers as edit spans -- so an unconverted index corrupts data silently, and
 * only for files containing a non-ASCII character.
 *
 * The shared extractor corpus is entirely ASCII, where the two coordinate
 * systems coincide, so `extractor-equivalence.test.ts` cannot catch this class
 * of bug at all. `scripts/write_unicode_extraction_parity.py` records what the
 * shipping Python build extracts from sources covering all three regimes that
 * differ -- two-byte, three-byte, and four-byte characters -- plus a BOM and an
 * oversized line built from multi-byte characters, which is where Python's
 * code-point string slicing parts company with JavaScript's UTF-16 slicing.
 *
 * Regenerate the fixture from Python whenever either extractor changes; never
 * edit it to make this pass.
 */

import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import fs from "node:fs";
import { TreeSitterExtractor } from "../src/extractor.ts";

interface Case {
  name: string;
  language: string;
  source: string;
  has_errors: boolean;
  chunks: unknown[][];
  references: unknown[][];
  declarations: unknown[][];
}

const fixture: { cases: Case[] } = JSON.parse(
  fs.readFileSync(new URL("./fixtures/unicode-extraction.json", import.meta.url), "utf8"),
);

function digest(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex").slice(0, 16);
}

function pythonRstrip(value: string): string {
  let end = value.length;
  while (end > 0) {
    const code = value.charCodeAt(end - 1);
    const whitespace =
      code === 0x20 ||
      (code >= 0x09 && code <= 0x0d) ||
      (code >= 0x1c && code <= 0x1f) ||
      code === 0x85 ||
      code === 0xa0 ||
      code === 0x1680 ||
      (code >= 0x2000 && code <= 0x200a) ||
      code === 0x2028 ||
      code === 0x2029 ||
      code === 0x202f ||
      code === 0x205f ||
      code === 0x3000;
    if (!whitespace) break;
    end -= 1;
  }
  return value.slice(0, end);
}

const extractor = new TreeSitterExtractor();

describe("parity with Python on non-ASCII sources", () => {
  test.each(fixture.cases.map((entry) => [entry.name, entry] as const))(
    "%s extracts the same byte offsets Python does",
    (_name, entry) => {
      const result = extractor.extract(
        entry.name,
        entry.language,
        new TextEncoder().encode(entry.source),
      );

      expect(result.has_errors).toBe(entry.has_errors);
      // Typed as `unknown[][]` so the comparison is against the fixture's own
      // shape rather than against a narrowed inference of the actual value.
      const chunks: unknown[][] = result.chunks.map((chunk) => [
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
      ]);
      const references: unknown[][] = result.references.map((reference) => [
        reference.kind,
        reference.target_name,
        reference.written_name,
        reference.start_byte,
        reference.end_byte,
        reference.start_line,
        reference.end_line,
        reference.source_qualified_symbol,
      ]);
      const declarations: unknown[][] = result.declarations.map((declaration) => [
        declaration.qualified_symbol,
        declaration.kind,
        declaration.start_byte,
        declaration.end_byte,
        declaration.start_line,
        declaration.end_line,
        declaration.parameters.map((parameter) => [
          parameter.name,
          parameter.kind,
          parameter.required,
        ]),
      ]);

      expect(chunks).toEqual(entry.chunks);
      expect(references).toEqual(entry.references);
      expect(declarations).toEqual(entry.declarations);
    },
  );

  test.each(fixture.cases.map((entry) => [entry.name, entry] as const))(
    "%s reports offsets that slice the right bytes back out",
    (_name, entry) => {
      // The strongest statement available about an offset: a chunk's stored
      // range, applied to the file's bytes, must yield the chunk's own content.
      // A uniformly-shifted offset would satisfy an equality against the fixture
      // only if the fixture were wrong too; this cannot be satisfied at all
      // unless the offsets really do address those bytes.
      const bytes = new TextEncoder().encode(entry.source.replace(/^﻿/, ""));
      const result = extractor.extract(
        entry.name,
        entry.language,
        new TextEncoder().encode(entry.source),
      );

      for (const chunk of result.chunks) {
        const sliced = new TextDecoder("utf-8", { ignoreBOM: true }).decode(
          bytes.subarray(chunk.start_byte, chunk.end_byte),
        );
        expect(pythonRstrip(sliced)).toBe(chunk.content);
      }
      for (const reference of result.references) {
        const sliced = new TextDecoder("utf-8", { ignoreBOM: true }).decode(
          bytes.subarray(reference.start_byte, reference.end_byte),
        );
        // A reference's range covers the occurrence, which for an import or an
        // aliased form is wider than the name -- so containment, not equality.
        expect(sliced).toContain(
          reference.written_name.slice(reference.written_name.lastIndexOf(".") + 1),
        );
      }
    },
  );

  test("the fixture actually contains non-ASCII sources", () => {
    // A fixture that lost its non-ASCII content would turn this whole file
    // green while testing nothing -- the exact failure mode it exists to
    // prevent.
    for (const entry of fixture.cases) {
      expect(new TextEncoder().encode(entry.source).length).toBeGreaterThan(entry.source.length);
    }
  });
});
