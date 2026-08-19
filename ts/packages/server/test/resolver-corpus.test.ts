/**
 * Resolver corpus: small multi-file repos with per-defect expected outcomes.
 *
 * Ported from `tests/test_resolver_corpus.py`, and reading the *same* fixture
 * directories under `tests/fixtures/resolver_corpus/` -- a corpus that existed
 * twice would be a corpus that could disagree with itself. Each case is named
 * for the defect it pins (E1-E9, R1-R4 from the reference-index hardening plan);
 * a case that stops passing means the port lost a fix the Python build has.
 *
 * Hard gate, never relaxed: zero false positives in the `exact` resolution
 * category. A same-named symbol reachable only through an unrelated re-export
 * chain must never bind `exact`.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { DeclarationSelector, type ParameterShape, type RefactorOperation } from "../src/models.ts";
import { ReferenceService } from "../src/reference-service.ts";
import type { ReferenceRecord } from "../src/reference-store.ts";
import { removeDirectory, repositoryRoot, temporaryDirectory } from "./helpers.ts";
import { type InMemoryReferenceStore, indexedStore } from "./reference-fixtures.ts";

const CORPUS_ROOT = path.join(repositoryRoot(), "tests", "fixtures", "resolver_corpus");

let temporary: string;

beforeEach(() => {
  temporary = temporaryDirectory();
});

afterEach(() => {
  removeDirectory(temporary);
});

/** Read every file under a corpus case directory into a path -> source map. */
function loadCase(language: string, name: string): Record<string, string> {
  const directory = path.join(CORPUS_ROOT, language, name);
  const files: Record<string, string> = {};
  const walk = (current: string): void => {
    for (const entry of fs
      .readdirSync(current, { withFileTypes: true })
      .sort((a, b) => (a.name < b.name ? -1 : 1))) {
      const absolute = path.join(current, entry.name);
      if (entry.isDirectory()) walk(absolute);
      else {
        files[path.relative(directory, absolute).split(path.sep).join("/")] = fs.readFileSync(
          absolute,
          "utf8",
        );
      }
    }
  };
  walk(directory);
  if (Object.keys(files).length === 0) {
    throw new Error(`resolver corpus case ${language}/${name} has no fixture files`);
  }
  return files;
}

function indexedCase(
  language: string,
  name: string,
): { service: ReferenceService; store: InMemoryReferenceStore; projectId: string } {
  const store = indexedStore(path.join(temporary, "repo"), loadCase(language, name));
  return { service: new ReferenceService(store), store, projectId: store.project.id };
}

function select(projectId: string, file: string, symbol: string): DeclarationSelector {
  return DeclarationSelector.parse({ project: projectId, path: file, qualified_symbol: symbol });
}

function rename(newName: string): RefactorOperation {
  return { kind: "rename", new_name: newName };
}

function referenceRowsFor(store: InMemoryReferenceStore, file: string): ReferenceRecord[] {
  return store.records.filter((row) => row.path === file && row.record_kind === "reference");
}

describe("E1: TS/TSX class heritage captures the identifier, not the raw clause", () => {
  test("renaming a TS base class surfaces `extends Base`", async () => {
    // Verbatim backlog repro: applying only the reported edits (declaration +
    // import) used to leave `extends Base` dangling in child.ts.
    const { service, projectId } = indexedCase("typescript", "e1_inheritance_base_foundation");

    const analysis = await service.analyzeRefactor(
      select(projectId, "base.ts", "Base"),
      rename("Foundation"),
    );

    const hit = [...analysis.must_change, ...analysis.likely_change].find(
      (item) => item.path === "child.ts" && item.kind === "inheritance",
    );
    expect(["exact", "likely"]).toContain(hit?.resolution ?? "");
  });
});

describe("E2: TS generic/union inner names are their own references", () => {
  test("a generic argument is a separate type_use", async () => {
    const { service, projectId } = indexedCase("typescript", "e2_generic_type");

    const response = await service.findReferences(select(projectId, "box.ts", "Item"));

    expect(response.hits.find((hit) => hit.kind === "type_use")?.path).toBe("main.ts");
  });
});

describe("E3: barrel re-exports keep their module path", () => {
  test("a bare `export *` emits a barrel export row", async () => {
    const { store } = indexedCase("typescript", "e3_export_star");

    const row = referenceRowsFor(store, "index.ts")[0];

    expect(row?.kind).toBe("export");
    expect(row?.module_path).toBe("./lib");
  });

  test("`export * as ns` keeps its module path", async () => {
    const { store } = indexedCase("typescript", "e3_export_star");

    const row = referenceRowsFor(store, "index_namespace.ts")[0];

    expect(row?.kind).toBe("export");
    expect(row?.alias).toBe("ns");
    expect(row?.module_path).toBe("./lib");
  });
});

describe("E4: unusual call forms are still calls", () => {
  test("a python generator as the sole argument is a call", async () => {
    const { service, projectId } = indexedCase("python", "e4_generator_argument");

    const response = await service.findReferences(select(projectId, "lib.py", "summarize"));

    const call = response.hits.find((hit) => hit.kind === "call");
    expect(call?.path).toBe("main.py");
    expect(call?.resolution).toBe("exact");
  });

  test("a JS tagged template is a call", async () => {
    const { service, projectId } = indexedCase("javascript", "e4_tagged_template");

    const response = await service.findReferences(select(projectId, "lib.js", "tag"));

    const call = response.hits.find((hit) => hit.kind === "call");
    expect(call?.path).toBe("main.js");
    expect(call?.resolution).toBe("exact");
  });

  test("a JS `new` without parentheses is a call", async () => {
    const { service, projectId } = indexedCase("javascript", "e4_new_expression");

    const response = await service.findReferences(select(projectId, "lib.js", "Widget"));

    const call = response.hits.find((hit) => hit.kind === "call");
    expect(call?.path).toBe("main.js");
    expect(call?.resolution).toBe("exact");
  });
});

describe("E5: member writes and reads are recorded", () => {
  test("a python member write and read carry their own lines", async () => {
    const { store } = indexedCase("python", "e5_member_write");

    const rows = referenceRowsFor(store, "main.py");
    const write = rows.find(
      (row) => row.kind === "write" && (row.target_name ?? "").split(".").pop() === "TIMEOUT",
    );
    const read = rows.find(
      (row) => row.kind === "read" && (row.target_name ?? "").split(".").pop() === "TIMEOUT",
    );

    expect(write?.start_line).toBe(3);
    expect(read?.start_line).toBe(4);
  });

  test("a JS shorthand property is a read", async () => {
    const { service, projectId } = indexedCase("javascript", "e5_shorthand");

    const response = await service.findReferences(select(projectId, "widget.js", "onSave"));

    expect(
      response.hits.find((hit) => hit.kind === "read" && hit.path === "main.js")?.start_line,
    ).toBe(3);
  });
});

describe("E6: JS/TS decorators produce references", () => {
  test("a TS decorator is a reference", async () => {
    const { service, projectId } = indexedCase("typescript", "e6_decorator");

    const response = await service.findReferences(select(projectId, "lib.ts", "Sealed"));

    expect(response.hits.find((hit) => hit.kind === "decorator")?.path).toBe("main.ts");
  });
});

describe("E7: destructured parameters keep a usable shape", () => {
  test("a TSX destructured parameter name is not raw pattern text", async () => {
    const { store } = indexedCase("tsx", "e7_destructured_parameter");

    const declaration = store.records.find(
      (row) => row.record_kind === "declaration" && row.target_name === "Widget",
    );
    const parameters = JSON.parse(declaration?.shape_json as string) as ParameterShape[];

    expect(parameters).toHaveLength(1);
    expect(parameters[0]?.name).not.toContain("{");
    expect(parameters[0]?.name).not.toContain("}");
  });
});

describe("E9: bare and dynamic module edges stay visible", () => {
  test("a side-effect import and a require keep their module paths", async () => {
    const { store } = indexedCase("javascript", "e9_module_edges");

    const rows = referenceRowsFor(store, "main.js");
    const bareImport = rows.find(
      (row) => row.kind === "import" && row.module_path === "./polyfill",
    );
    const requireCall = rows.find((row) => row.kind === "call" && row.target_name === "require");

    expect(bareImport?.imported_name).toBeNull();
    expect(requireCall?.module_path).toBe("./lib");
  });
});

describe("R1: override analysis", () => {
  test("a python override is a likely change", async () => {
    const { service, projectId } = indexedCase("python", "r1_override");

    const analysis = await service.analyzeRefactor(
      select(projectId, "base.py", "Base.handle"),
      rename("process"),
    );

    expect(
      analysis.likely_change.find(
        (item) => item.path === "child.py" && item.reason_code === "override_of_renamed_method",
      )?.resolution,
    ).toBe("likely");
  });
});

describe("R2: re-export chains", () => {
  test("a re-export chain resolves exactly", async () => {
    const { service, projectId } = indexedCase("python", "r2_reexport_chain");

    const response = await service.findReferences(select(projectId, "pkg/impl.py", "b"));

    const call = response.hits.find((hit) => hit.path === "main.py" && hit.kind === "call");
    expect(call?.resolution).toBe("exact");
    expect(call?.reason_code).toBe("reexport_chain");
  });

  test("HARD GATE: an unrelated re-export chain never binds exact", async () => {
    // Zero false positives in `exact`: a same-named symbol reachable only
    // through a *different* re-export chain must never bind.
    const { service, projectId } = indexedCase("python", "r2_reexport_chain_false_positive");

    const response = await service.findReferences(select(projectId, "pkg_a/impl.py", "shared"));

    const calls = response.hits.filter((hit) => hit.path === "main.py" && hit.kind === "call");
    expect(calls.length).toBeGreaterThan(0);
    expect(calls.every((call) => call.resolution !== "exact")).toBe(true);
  });
});

describe("R3: a declaration and its export are one edit", () => {
  test.each([
    ["r3_duplicate_declaration_export", "answer.ts", "answer", "result"],
    ["r3_duplicate_declaration_export_const", "answer.ts", "answer", "result"],
    ["r3_duplicate_declaration_export_default_class", "foo.ts", "Foo", "Foundation"],
  ])("%s counts one required edit", async (name, file, symbol, newName) => {
    const { service, projectId } = indexedCase("typescript", name);

    const analysis = await service.analyzeRefactor(
      select(projectId, file, symbol),
      rename(newName),
    );

    expect(analysis.counts.must_change).toBe(1);
  });
});

describe("R4: page-independent completeness and counts", () => {
  test("the last page's completeness accounts for earlier pages", async () => {
    const { service, projectId } = indexedCase("python", "r4_pagination_review");
    const selector = select(projectId, "mod.py", "send");
    const operation: RefactorOperation = {
      kind: "signature_change",
      parameters: [
        { name: "message", kind: "positional", required: true, position: 0, destructured: false },
      ],
    };

    const first = await service.analyzeRefactor(selector, operation, { limit: 2 });
    expect(first.cursor).not.toBeNull();
    expect(first.review.some((item) => item.reason_code === "spread_uncertainty")).toBe(true);

    const second = await service.analyzeRefactor(selector, operation, {
      limit: 2,
      cursor: first.cursor,
    });
    expect(second.cursor).toBeNull();
    expect(second.completeness.state).not.toBe("complete");
  });
});
