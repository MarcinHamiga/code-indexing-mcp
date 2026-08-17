/** The domain models, which are also the wire contract. */

import { describe, expect, test } from "bun:test";
import { z } from "zod";
import {
  CodeChunk,
  DEFAULT_INCLUDES,
  DeclarationSelector,
  IndexReport,
  isBackfillComplete,
  LEGACY_DEFAULT_INCLUDES_V1,
  LEGACY_DEFAULT_INCLUDES_V2,
  LEGACY_DEFAULT_INCLUDES_V3,
  ProjectInfo,
  RefactorAnalysis,
  refactorFindings,
  RefactorOperation,
  ReferenceBackfillReport,
  RunAudit,
  ScanConfig,
  ScannedFile,
  StoredChunk,
} from "../src/models.ts";

test("field names stay snake_case, because they are the wire format", () => {
  const project = ProjectInfo.parse({ id: "i", name: "n", root: "/tmp/x" });

  expect(Object.keys(project).sort()).toEqual(["id", "name", "root", "scan", "version"]);
  expect(Object.keys(project.scan).sort()).toEqual(["exclude", "include", "max_file_bytes"]);
});

test("defaults fill in the way pydantic's default factories did", () => {
  const scan = ScanConfig.parse({});

  expect(scan.include).toEqual([...DEFAULT_INCLUDES]);
  expect(scan.exclude).toEqual([]);
  expect(scan.max_file_bytes).toBe(1_048_576);
});

test("a default list is not shared between two parsed models", () => {
  // `default_factory=list` existed so two projects could not alias one array.
  const first = ScanConfig.parse({});
  const second = ScanConfig.parse({});

  expect(first.include).not.toBe(second.include);
});

test("the legacy include lists are prefixes of the current one", () => {
  // The marker upgrade in projects.ts recognises a default list by equality, so
  // a legacy list that stopped being a prefix would mean a language silently
  // stopped being picked up on upgrade.
  for (const legacy of [
    LEGACY_DEFAULT_INCLUDES_V1,
    LEGACY_DEFAULT_INCLUDES_V2,
    LEGACY_DEFAULT_INCLUDES_V3,
  ]) {
    expect(DEFAULT_INCLUDES.slice(0, legacy.length)).toEqual([...legacy]);
  }
  expect(DEFAULT_INCLUDES.length).toBeGreaterThan(LEGACY_DEFAULT_INCLUDES_V3.length);
});

test("a max_file_bytes of zero is refused", () => {
  expect(ScanConfig.safeParse({ max_file_bytes: 0 }).success).toBe(false);
});

describe("nanosecond mtimes", () => {
  test("survive as bigints, which a number could not", () => {
    // Around 1.7e18: two hundred times past Number.MAX_SAFE_INTEGER, so a
    // `number` would round the low digits away and make every file in a
    // migrated index look modified exactly once.
    const exact = 1_755_400_000_123_456_789n;
    const file = ScannedFile.parse({
      path: "a.py",
      absolute_path: "/repo/a.py",
      language: "python",
      size: 10,
      mtime_ns: exact,
    });

    expect(file.mtime_ns).toBe(exact);
    expect(BigInt(Number(exact))).not.toBe(exact);
  });

  test("a streaming scan may attach bytes and a collected one may not", () => {
    const base = {
      path: "a.py",
      absolute_path: "/repo/a.py",
      language: "python",
      size: 10,
      mtime_ns: 1n,
    };

    expect(ScannedFile.parse(base).content).toBeNull();
    expect(ScannedFile.parse({ ...base, content: new Uint8Array([1, 2]) }).content).toEqual(
      new Uint8Array([1, 2]),
    );
  });
});

describe("the declaration selector", () => {
  test("accepts a chunk id alone", () => {
    expect(DeclarationSelector.safeParse({ chunk_id: "c" }).success).toBe(true);
  });

  test("accepts a full location", () => {
    expect(
      DeclarationSelector.safeParse({ project: "p", path: "a.py", qualified_symbol: "A.b" })
        .success,
    ).toBe(true);
  });

  test("refuses a chunk id combined with any part of a location", () => {
    const result = DeclarationSelector.safeParse({ chunk_id: "c", path: "a.py" });

    expect(result.success).toBe(false);
    expect(result.error?.issues[0]?.message).toContain("cannot be combined");
  });

  test("refuses a partial location", () => {
    const result = DeclarationSelector.safeParse({ project: "p", path: "a.py" });

    expect(result.success).toBe(false);
    expect(result.error?.issues[0]?.message).toContain("Provide exactly");
  });

  test("refuses an empty selector", () => {
    expect(DeclarationSelector.safeParse({}).success).toBe(false);
  });
});

describe("the refactor operation union", () => {
  test("routes on the discriminator", () => {
    expect(RefactorOperation.parse({ kind: "rename", new_name: "x" })).toEqual({
      kind: "rename",
      new_name: "x",
    });
    expect(RefactorOperation.parse({ kind: "signature_change", parameters: [] }).kind).toBe(
      "signature_change",
    );
  });

  test("requires the discriminator to be present, as the tagged union always did", () => {
    expect(RefactorOperation.safeParse({ new_name: "x" }).success).toBe(false);
  });

  test("findings keep the caller-facing priority order", () => {
    const hit = (reference_id: string) => ({
      reference_id,
      project_id: "p",
      path: "a.py",
      language: "python",
      kind: "call" as const,
      start_line: 1,
      end_line: 1,
      start_byte: 0,
      end_byte: 1,
      resolution: "exact" as const,
      reason_code: "r",
      explanation: "e",
    });
    const analysis = RefactorAnalysis.parse({
      selected: {
        project_id: "p",
        file_id: "f",
        path: "a.py",
        language: "python",
        symbol: "s",
        qualified_symbol: "s",
        kind: "function",
        start_line: 1,
        end_line: 2,
      },
      operation: { kind: "rename", new_name: "t" },
      review: [hit("review")],
      must_change: [hit("must")],
      evidence: [hit("evidence")],
      likely_change: [hit("likely")],
    });

    expect(refactorFindings(analysis).map((finding) => finding.reference_id)).toEqual([
      "must",
      "likely",
      "review",
      "evidence",
    ]);
    // A finding is a reference hit plus the edit span, so the inherited fields
    // are all present.
    expect(analysis.must_change[0]?.edit_required).toBe(false);
    expect(analysis.must_change[0]?.edit_start_byte).toBeNull();
    expect(analysis.counts.must_change).toBe(0);
    expect(analysis.completeness.state).toBe("complete");
  });
});

test("a backfill report is complete only when nothing was left behind", () => {
  const report = ReferenceBackfillReport.parse({ project_id: "p" });

  expect(isBackfillComplete(report)).toBe(true);
  expect(isBackfillComplete({ ...report, stale_paths: ["a.py"] })).toBe(false);
  expect(isBackfillComplete({ ...report, incomplete_paths: ["a.py"] })).toBe(false);
});

test("the chunk returned to a caller carries no vector and no derived text", () => {
  const fields = Object.keys(
    CodeChunk.parse({
      chunk_id: "c",
      file_id: "f",
      project_id: "p",
      path: "a.py",
      language: "python",
      kind: "function",
      start_byte: 0,
      end_byte: 1,
      start_line: 1,
      end_line: 1,
      content: "x",
      content_hash: "h",
    }),
  );

  // Inheriting the storage row once shipped a 768-float vector and the code
  // three times over to MCP clients. Adding a storage column must not silently
  // widen this payload.
  expect(fields).not.toContain("vector");
  expect(fields).not.toContain("identifier_terms");
  expect(Object.keys(StoredChunk.shape)).toContain("vector");
});

test("a report validates from nothing but a project id", () => {
  // Every optional field exists so older clients keep validating newer reports
  // and newer clients keep validating older stored ones.
  const report = IndexReport.parse({ project_id: "p" });

  expect(report.trigger).toBe("manual");
  expect(report.embedding_backend).toBe("cpu");
  expect(report.scan_ms).toBeNull();
  expect(report.skip_reasons).toEqual({});
});

test("an audit record stamps the owning process", () => {
  const audit = RunAudit.parse({ run_id: "r", project_id: "p", trigger: "startup" });

  // Startup uses this to tell a crashed run from one another live process is
  // still executing.
  expect(audit.pid).toBe(process.pid);
  expect(audit.state).toBe("running");
});

test("a path field emits a plain string schema", () => {
  // Pydantic marked `pathlib.Path` with a non-standard `"format": "path"` that
  // made strict MCP clients warn, which is why `_PathAsPlainString` existed.
  // A TypeScript path is a string, so the emitted schema is right by default --
  // asserted here so a later change back to a branded type cannot reintroduce
  // the warning unnoticed.
  const schema = z.toJSONSchema(ProjectInfo) as unknown as {
    properties: { root: { type: string; format?: string } };
  };

  expect(schema.properties.root).toEqual({ type: "string" });
});
