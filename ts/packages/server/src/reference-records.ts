/**
 * Building one file's structural generation from extractor output.
 *
 * Pure record assembly, kept beside the row schema rather than inside the
 * indexer: this is the exact shape {@link ReferenceService} reads, and defining
 * the two together is what makes the reader's assumptions checkable. Phase 5's
 * indexer calls this at commit time; the resolver suite calls it to build a
 * store from real extraction, so the resolver is tested against the rows the
 * pipeline actually produces rather than against hand-written ones.
 *
 * Ported from `indexing.py::_reference_rows`.
 */

import { createHash } from "node:crypto";
import type { ExtractedDeclarationShape, ExtractedReference, StoredFile } from "./models.ts";
import { REFERENCE_SCHEMA_VERSION, type ReferenceRecord } from "./reference-store.ts";

/** The content and identity digest used throughout the index. */
export function digest(value: string | Uint8Array): string {
  return createHash("sha256")
    .update(typeof value === "string" ? Buffer.from(value, "utf8") : value)
    .digest("hex");
}

/** A file's stable identity within a project. */
export function fileId(projectId: string, filePath: string): string {
  return digest(`${projectId}\0${filePath}`);
}

/**
 * Build one deterministic structural generation for a file.
 *
 * A coverage record is deliberately emitted even when the parser found no
 * occurrences. It is the durable proof that the current structural schema
 * parsed this exact content, rather than merely an empty query result from an
 * unindexed legacy project.
 */
export function referenceRows(
  projectId: string,
  file: StoredFile,
  references: readonly ExtractedReference[],
  declarations: readonly ExtractedDeclarationShape[],
): ReferenceRecord[] {
  // `kind` belongs in the digest because one byte range legitimately carries
  // two references: a superclass is both `inheritance` and a `read`, and a
  // decorator call is both `decorator` and `call`. Omitting it gave those rows
  // one id, and a merge rejects two source rows matching a single target --
  // which permanently broke every later incremental index of the project.
  const identity = (
    recordKind: string,
    kind: string | null,
    startByte: number | null,
    endByte: number | null,
  ): string =>
    digest(
      [
        file.file_id,
        recordKind,
        kind ?? "",
        String(startByte ?? -1),
        String(endByte ?? -1),
        String(REFERENCE_SCHEMA_VERSION),
      ].join("\0"),
    );

  const base = {
    file_id: file.file_id,
    project_id: projectId,
    path: file.path,
    language: file.language,
    content_hash: file.content_hash,
    schema_version: REFERENCE_SCHEMA_VERSION,
  };

  const rows: ReferenceRecord[] = references.map((reference) => ({
    ...base,
    reference_id: identity("reference", reference.kind, reference.start_byte, reference.end_byte),
    record_kind: "reference",
    kind: reference.kind,
    source_qualified_symbol: reference.source_qualified_symbol,
    written_name: reference.written_name,
    target_name: reference.target_name,
    module_path: reference.module_path,
    imported_name: reference.imported_name,
    alias: reference.alias,
    receiver_text: reference.receiver_text,
    start_byte: reference.start_byte,
    end_byte: reference.end_byte,
    start_line: reference.start_line,
    end_line: reference.end_line,
    shape_json: reference.call_shape === null ? null : JSON.stringify(reference.call_shape),
  }));

  for (const declaration of declarations) {
    rows.push({
      ...base,
      reference_id: identity(
        "declaration",
        declaration.kind,
        declaration.start_byte,
        declaration.end_byte,
      ),
      record_kind: "declaration",
      kind: declaration.kind,
      source_qualified_symbol: declaration.qualified_symbol,
      written_name: declaration.symbol,
      target_name: declaration.symbol,
      module_path: null,
      imported_name: null,
      alias: null,
      receiver_text: null,
      start_byte: declaration.start_byte,
      end_byte: declaration.end_byte,
      start_line: declaration.start_line,
      end_line: declaration.end_line,
      // Sorted keys and no whitespace, so a shape compares byte-for-byte across
      // runs the way the Python writer's `json.dumps(..., sort_keys=True)` does.
      shape_json: JSON.stringify(
        declaration.parameters.map((parameter) =>
          Object.fromEntries(
            Object.entries(parameter).sort(([left], [right]) => (left < right ? -1 : 1)),
          ),
        ),
      ),
    });
  }

  rows.push({
    ...base,
    reference_id: identity("coverage", null, null, null),
    record_kind: "coverage",
    kind: null,
    source_qualified_symbol: null,
    written_name: null,
    target_name: null,
    module_path: null,
    imported_name: null,
    alias: null,
    receiver_text: null,
    start_byte: null,
    end_byte: null,
    start_line: null,
    end_line: null,
    shape_json: null,
  });
  return rows;
}
