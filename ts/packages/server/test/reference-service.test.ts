/**
 * Reference resolution, ported from `tests/test_references.py`.
 *
 * The hard gate, which is never relaxed: an `exact` resolution must never be
 * wrong. Several cases below are negative controls that exist only to prove a
 * decoy stays out of `exact` -- a resolver that graded everything `likely`
 * would pass the positive tests and fail these.
 *
 * The store is in-memory (see `reference-fixtures.ts`) because LanceDB lands in
 * Phase 3 and the indexer in Phase 5; the extraction and row assembly are the
 * real ones. Two Python tests are deliberately absent until then and are named
 * in the Phase 2 notes: the one that renames a `.lance` directory on disk to
 * simulate a never-built table, and the one that heals a stale file by
 * re-running the indexer.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { isCodeIndexingError } from "../src/errors.ts";
import { DeclarationSelector } from "../src/models.ts";
import { ReferenceService } from "../src/reference-service.ts";
import { REFERENCE_SCHEMA_VERSION } from "../src/reference-store.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";
import { type InMemoryReferenceStore, indexedStore } from "./reference-fixtures.ts";

let temporary: string;

beforeEach(() => {
  temporary = temporaryDirectory();
});

afterEach(() => {
  removeDirectory(temporary);
});

function indexed(files: Readonly<Record<string, string>>): {
  service: ReferenceService;
  store: InMemoryReferenceStore;
  projectId: string;
} {
  const store = indexedStore(path.join(temporary, "repo"), files);
  return { service: new ReferenceService(store), store, projectId: store.project.id };
}

function select(projectId: string, file: string, symbol: string): DeclarationSelector {
  return DeclarationSelector.parse({
    project: projectId,
    path: file,
    qualified_symbol: symbol,
  });
}

async function errorCode(action: () => Promise<unknown>): Promise<string> {
  try {
    await action();
  } catch (error) {
    if (isCodeIndexingError(error)) return error.code;
    throw error;
  }
  throw new Error("expected a CodeIndexingError");
}

describe("import resolution", () => {
  test("a direct python import alias resolves exactly", async () => {
    const { service, projectId } = indexed({
      "lib.py": "def answer():\n    return 42\n",
      "main.py": "from lib import answer as local\n\ndef caller():\n    return local()\n",
    });

    const response = await service.findReferences(select(projectId, "lib.py", "answer"));

    const call = response.hits.find((hit) => hit.kind === "call");
    expect(call?.resolution).toBe("exact");
    expect(call?.reason_code).toBe("direct_import_alias");
    expect(call?.snippet).toBe("local");
  });

  test("an absolute import within a package resolves exactly", async () => {
    // A same-package absolute import written inside `mypkg/main.py` must anchor
    // at the project root, not at `mypkg/` itself -- otherwise the generated
    // candidate is `mypkg/mypkg/lib.py`, which never matches (finding 2).
    const { service, projectId } = indexed({
      "mypkg/__init__.py": "",
      "mypkg/lib.py": "def answer():\n    return 42\n",
      "mypkg/main.py": "from mypkg.lib import answer\n\ndef caller():\n    return answer()\n",
    });

    const response = await service.findReferences(select(projectId, "mypkg/lib.py", "answer"));

    const call = response.hits.find((hit) => hit.kind === "call");
    expect(call?.resolution).toBe("exact");
    expect(call?.reason_code).toBe("direct_import_alias");
  });

  test("an absolute import under a src layout resolves exactly", async () => {
    // Same as above, but the package sits under a `src/` sub-root: `src/mypkg/
    // main.py` importing `mypkg.lib` absolutely must anchor at `src/`, found by
    // walking the unbroken `__init__.py` chain up to (but not past) `src`.
    const { service, projectId } = indexed({
      "src/mypkg/__init__.py": "",
      "src/mypkg/lib.py": "def answer():\n    return 42\n",
      "src/mypkg/main.py": "from mypkg.lib import answer\n\ndef caller():\n    return answer()\n",
    });

    const response = await service.findReferences(select(projectId, "src/mypkg/lib.py", "answer"));

    const call = response.hits.find((hit) => hit.kind === "call");
    expect(call?.resolution).toBe("exact");
    expect(call?.reason_code).toBe("direct_import_alias");
  });

  test("a src-layout absolute import does not bind the wrong directory", async () => {
    // Negative control for the case above: a same-named module at the (wrong)
    // project root must never be treated as *exact* just because some candidate
    // set happened to include it. The resolver classifies per selected
    // declaration, so asked about this decoy the same call textually matches on
    // name and is reported conservatively -- never upgraded to `exact`.
    const { service, projectId } = indexed({
      "src/mypkg/__init__.py": "",
      "src/mypkg/lib.py": "def answer():\n    return 42\n",
      "src/mypkg/main.py": "from mypkg.lib import answer\n\ndef caller():\n    return answer()\n",
      "mypkg/lib.py": "def answer():\n    return 0\n",
    });

    const response = await service.findReferences(select(projectId, "mypkg/lib.py", "answer"));

    expect(response.hits.length).toBeGreaterThan(0);
    expect(response.hits.every((hit) => hit.resolution !== "exact")).toBe(true);
  });

  test.each([
    ["lib.py", "def answer():\n    return 42\n", "main.py", "import lib as ns\n\nns.answer()\n"],
    ["lib.py", "def answer():\n    return 42\n", "main.py", "import lib\n\nlib.answer()\n"],
    [
      "lib.js",
      "export function answer() { return 42; }\n",
      "main.js",
      "import * as ns from './lib';\nns.answer();\n",
    ],
    [
      "lib.ts",
      "export function answer(): number { return 42; }\n",
      "main.ts",
      "import * as ns from './lib';\nns.answer();\n",
    ],
    [
      "lib.tsx",
      "export function answer(): number { return 42; }\n",
      "main.tsx",
      "import * as ns from './lib';\nns.answer();\n",
    ],
  ])(
    "namespace member imports resolve exactly (%s)",
    async (targetPath, targetSource, sourcePath, source) => {
      const { service, projectId } = indexed({ [targetPath]: targetSource, [sourcePath]: source });

      const response = await service.findReferences(select(projectId, targetPath, "answer"));

      const call = response.hits.find((hit) => hit.kind === "call");
      expect(call?.resolution).toBe("exact");
      expect(call?.reason_code).toBe("known_namespace_member");
      expect(response.hits.some((hit) => hit.kind === "import")).toBe(false);
      expect(response.limitations.some((item) => item.code === "wildcard_import")).toBe(false);
    },
  );

  test("a direct import named like its module is not a namespace", async () => {
    const { service, projectId } = indexed({
      "lib.py": "def answer():\n    return 42\n",
      "main.py": "from lib import lib as ns\n\nns.answer()\n",
    });

    const response = await service.findReferences(select(projectId, "lib.py", "answer"));

    const call = response.hits.find((hit) => hit.kind === "call");
    expect(call?.resolution).toBe("likely");
    expect(call?.reason_code).toBe("unknown_receiver");
  });

  test("a python wildcard import remains unresolved", async () => {
    const { service, projectId } = indexed({
      "lib.py": "def answer():\n    return 42\n",
      "main.py": "from lib import *\n\nanswer()\n",
    });

    const response = await service.findReferences(select(projectId, "lib.py", "answer"));

    const call = response.hits.find((hit) => hit.kind === "call");
    expect(call?.resolution).toBe("unresolved");
    expect(call?.reason_code).toBe("wildcard_import");
  });

  test("a renaming two-hop barrel alias is followed", async () => {
    // A reference row's own `target_name` can be an arbitrary local alias.
    // `pkg/__init__.py` re-exports `answer` under a different local name, and
    // `importer.py` imports that renamed binding under yet another alias -- so
    // the call site's own `target_name` is `"x2"`, a spelling no single
    // predicate could predict from `"answer"` (E3 finding 1). This is the case
    // backing the decision to leave the reference-side scan unfiltered while
    // narrowing declarations.
    const { service, projectId } = indexed({
      "lib.py": "def answer():\n    return 42\n",
      "pkg/__init__.py": "from lib import answer as ans_alias\n",
      "importer.py": "from pkg import ans_alias as x2\n\ndef use():\n    return x2()\n",
    });

    const response = await service.findReferences(select(projectId, "lib.py", "answer"));

    const calls = response.hits.filter((hit) => hit.kind === "call" && hit.path === "importer.py");
    expect(calls).toHaveLength(1);
    expect(calls[0]?.written_name).toBe("x2");
    expect(calls[0]?.resolution).toBe("exact");
    expect(calls[0]?.reason_code).toBe("reexport_chain");
  });

  test("coverage rows keep absolute imports from binding a sibling decoy", async () => {
    // Narrowing the *declaration* fetch (S4/E3) must not narrow `knownPaths`.
    // `mypkg/__init__.py` is empty -- zero declarations, zero references -- so
    // the only way its path reaches `knownPaths` is through its coverage row,
    // which is what lets the package root be recognized. Without it the decoy
    // `mypkg/utils.py` would be indistinguishable from the real `utils.py`.
    const { service, projectId } = indexed({
      "mypkg/__init__.py": "",
      "mypkg/utils.py": "def f():\n    return 'decoy'\n",
      "mypkg/other.py": "from utils import f\n\ndef use():\n    return f()\n",
      "utils.py": "def f():\n    return 'real'\n",
    });

    const real = await service.findReferences(select(projectId, "utils.py", "f"));
    const calls = real.hits.filter((hit) => hit.kind === "call");
    expect(calls).toHaveLength(1);
    expect(calls[0]?.path).toBe("mypkg/other.py");
    expect(calls[0]?.resolution).toBe("exact");

    // The bare call `f()` is still a name-only *candidate* against the decoy
    // (any same-named declaration is), but it must not be graded `exact`.
    const decoy = await service.findReferences(select(projectId, "mypkg/utils.py", "f"));
    const decoyHit = decoy.hits.find((hit) => hit.path === "mypkg/other.py");
    expect(decoyHit?.resolution).not.toBe("exact");
  });
});

describe("bare names and receivers", () => {
  test("one project-wide candidate is a name-only match", async () => {
    // Exercises the project-wide ambiguity fallback (E2), which looks candidates
    // up through a `target_name` query rather than scanning every declaration
    // per reference row -- it must still find exactly the declarations that
    // share the bare name, nothing more or less.
    const { service, projectId } = indexed({
      "lib.py": "def answer():\n    return 42\n",
      "main.py": "def caller():\n    return answer()\n",
    });

    const response = await service.findReferences(select(projectId, "lib.py", "answer"));

    const call = response.hits.find((hit) => hit.kind === "call");
    expect(call?.resolution).toBe("likely");
    expect(call?.reason_code).toBe("name_only_candidate");
  });

  test("two project-wide candidates are ambiguous", async () => {
    const { service, projectId } = indexed({
      "lib.py": "def answer():\n    return 42\n",
      "other.py": "def answer():\n    return 0\n",
      "main.py": "def caller():\n    return answer()\n",
    });

    const response = await service.findReferences(select(projectId, "lib.py", "answer"));

    const call = response.hits.find((hit) => hit.kind === "call");
    expect(call?.resolution).toBe("unresolved");
    expect(call?.reason_code).toBe("ambiguous_symbol");
  });

  test("an unknown member receiver is never exact", async () => {
    const { service, projectId } = indexed({
      "lib.py": "def answer():\n    return 42\n",
      "main.py": "def caller(thing):\n    return thing.answer()\n",
    });

    const response = await service.findReferences(select(projectId, "lib.py", "answer"));

    const call = response.hits.find((hit) => hit.kind === "call");
    expect(call?.resolution).toBe("likely");
    expect(call?.reason_code).toBe("unknown_receiver");
    expect(response.limitations.some((item) => item.code === "unknown_receiver")).toBe(true);
  });

  test("a same-file shadowed call does not bind the selected declaration", async () => {
    const { service, projectId } = indexed({
      "lib.py":
        "def target():\n" +
        "    return 1\n\n" +
        "def direct():\n" +
        "    return target()\n\n" +
        "def outer():\n" +
        "    def target():\n" +
        "        return 2\n" +
        "    return target()\n",
    });

    const response = await service.findReferences(select(projectId, "lib.py", "target"));

    expect(
      response.hits
        .filter((hit) => hit.kind === "call")
        .map((hit) => [hit.start_line, hit.resolution]),
    ).toEqual([[5, "exact"]]);
  });

  test("a class body does not shadow a module-level function", async () => {
    // Python and JS/TS both resolve a bare `helper()` inside `Gate.run` to the
    // module-level `helper`. Treating the class body as an enclosing scope made
    // the resolver discard that call site, so a rename reported no callers.
    const { service, projectId } = indexed({
      "app.py":
        "def helper():\n" +
        "    return 1\n" +
        "\n" +
        "class Gate:\n" +
        "    def helper(self):\n" +
        "        return 2\n" +
        "\n" +
        "    def run(self):\n" +
        "        return helper()\n",
    });

    const response = await service.findReferences(select(projectId, "app.py", "helper"));

    expect(
      response.hits
        .filter((hit) => hit.kind === "call")
        .map((hit) => [hit.start_line, hit.resolution]),
    ).toEqual([[9, "exact"]]);
  });

  test("a method still shadows a reference made through its own receiver", async () => {
    const { service, projectId } = indexed({
      "app.py":
        "def helper():\n" +
        "    return 1\n" +
        "\n" +
        "class Gate:\n" +
        "    def helper(self):\n" +
        "        return 2\n" +
        "\n" +
        "    def run(self):\n" +
        "        return self.helper()\n",
    });

    const response = await service.findReferences(select(projectId, "app.py", "helper"));

    // `self.helper()` names the method, so the module function must not claim it
    // as an exact use.
    expect(
      response.hits.filter((hit) => hit.kind === "call").every((hit) => hit.resolution !== "exact"),
    ).toBe(true);
  });
});

describe("coverage and staleness", () => {
  test("files without reference extraction are reported as a coverage gap", async () => {
    const { service, projectId } = indexed({
      "lib.py": "def answer():\n    return 42\n",
      "svc.go": "package main\n\nfunc Run() int {\n\treturn 1\n}\n",
    });

    const response = await service.findReferences(select(projectId, "lib.py", "answer"));

    const limitation = response.limitations.find((item) => item.code === "unsupported_language");
    expect(limitation?.explanation).toContain("svc.go");
    expect(limitation?.explanation).toContain("go");
  });

  test("references from a file changed since extraction are suppressed as stale", async () => {
    const { service, projectId } = indexed({
      "lib.py": "def answer():\n    return 42\n",
      "main.py": "from lib import answer\n\ndef caller():\n    return answer()\n",
    });
    // The file changes on disk without a reindex: its stored rows now describe
    // bytes that no longer exist, so their offsets must not be served against
    // the new content.
    fs.writeFileSync(
      path.join(temporary, "repo", "main.py"),
      "from lib import answer\n\n\ndef caller():\n    return answer()\n",
    );

    const response = await service.findReferences(select(projectId, "lib.py", "answer"));

    expect(response.hits).toEqual([]);
    const limitation = response.limitations.find((item) => item.code === "stale_file");
    expect(limitation?.explanation).toContain("main.py");
  });

  test("a missing reference table is reported distinctly", async () => {
    // A legacy or never-built reference index must not read as "no references"
    // (S5): both collapse to an empty row set, so the distinction has to be
    // asked for directly.
    const { service, store, projectId } = indexed({ "lib.py": "def answer():\n    return 42\n" });
    const selector = select(projectId, "lib.py", "answer");
    expect((await service.findReferences(selector)).hits).toEqual([]);

    store.referenceTableExists = false;

    expect(await errorCode(() => service.findReferences(selector))).toBe(
      "REFERENCE_INDEX_UNAVAILABLE",
    );
  });

  test("stale schema-version rows are not served", async () => {
    // The version bump was meant to discard the previous generation's id
    // scheme; a row that survives a partial reindex under the old version
    // carries stale offsets and a colliding id shape, so it must be excluded on
    // the read path (finding 9).
    const { service, store, projectId } = indexed({
      "lib.py": "def answer():\n    return 42\n",
      "main.py": "from lib import answer\n\ndef caller():\n    return answer()\n",
    });
    const selector = select(projectId, "lib.py", "answer");
    const before = (await service.findReferences(selector)).hits.filter(
      (hit) => hit.kind === "call",
    );
    expect(before).toHaveLength(1);

    const current = store.records.find(
      (row) => row.path === "main.py" && row.kind === "call",
    ) as (typeof store.records)[number];
    store.records.push({
      ...current,
      reference_id: "stale-v3-call",
      schema_version: REFERENCE_SCHEMA_VERSION - 1,
    });

    const after = (await service.findReferences(selector)).hits.filter(
      (hit) => hit.kind === "call",
    );
    expect(after).toHaveLength(1);
    expect(after.map((hit) => hit.reference_id)).toEqual(before.map((hit) => hit.reference_id));
  });
});

describe("pushdowns", () => {
  test("the declaration fetch narrows to files holding a reference", async () => {
    // `unrelated.py` holds only its own declaration -- nothing in lexical or
    // class-scope resolution ever looks it up, so it must never be fetched.
    // Passing "every known file" here would be a redundant round trip back to
    // data already in hand (E3 finding 2); this proves the set is a proper
    // subset.
    const { service, store, projectId } = indexed({
      "lib.py": "def answer():\n    return 42\n",
      "main.py": "from lib import answer\n\ndef caller():\n    return answer()\n",
      "unrelated.py": "def other():\n    pass\n",
    });
    const declarationCalls: Array<Set<string>> = [];
    const targetCalls: Array<[string, string | undefined]> = [];
    const recordKindsSeen: Array<readonly string[] | undefined> = [];
    const realDeclarations = store.declarationsForFiles.bind(store);
    const realTargets = store.targetNameCandidates.bind(store);
    const realRecords = store.listReferenceRecords.bind(store);
    store.declarationsForFiles = async (project, fileIds, options) => {
      declarationCalls.push(new Set(fileIds));
      return realDeclarations(project, fileIds, options);
    };
    store.targetNameCandidates = async (project, targetName, options) => {
      targetCalls.push([targetName, options.recordKind]);
      return realTargets(project, targetName, options);
    };
    store.listReferenceRecords = async (project, options) => {
      recordKindsSeen.push(options.recordKinds);
      return realRecords(project, options);
    };

    const response = await service.findReferences(select(projectId, "lib.py", "answer"));

    // The pushdowns were actually used, not just left available.
    expect(declarationCalls).toHaveLength(1);
    expect(targetCalls).toEqual([["answer", "declaration"]]);
    expect(recordKindsSeen).toEqual([["reference", "coverage"]]);

    // And the narrowing is real: the fetched file set excludes the file with no
    // candidate reference, while still covering the one that matters.
    const pathsById = new Map(store.records.map((row) => [row.file_id, row.path]));
    const fetched = new Set(
      [...(declarationCalls[0] as Set<string>)].map((id) => pathsById.get(id)),
    );
    expect(fetched.has("unrelated.py")).toBe(false);
    expect(fetched.has("main.py")).toBe(true);

    const calls = response.hits.filter((hit) => hit.kind === "call");
    expect(calls).toHaveLength(1);
    expect(calls[0]?.path).toBe("main.py");
    expect(calls[0]?.resolution).toBe("exact");
  });
});

describe("cursors", () => {
  test("a cursor is filter-bound and reads its original snapshot", async () => {
    const { service, store, projectId } = indexed({
      "lib.py": "def answer():\n    return 42\n",
      "main.py": "from lib import answer\n\ndef a(): return answer()\ndef b(): return answer()\n",
    });
    const selector = select(projectId, "lib.py", "answer");

    const first = await service.findReferences(selector, { limit: 1 });
    expect(first.cursor).not.toBeNull();
    const second = await service.findReferences(selector, { limit: 1, cursor: first.cursor });
    expect(second.hits.length).toBeGreaterThan(0);

    // A cursor/filter mismatch is a structured, machine-readable error rather
    // than a bare throw that would reach the client as a raw message (T2).
    expect(
      await errorCode(() =>
        service.findReferences(selector, { kinds: new Set(["call"]), cursor: first.cursor }),
      ),
    ).toBe("INVALID_CURSOR");
    expect(
      await errorCode(() => service.findReferences(selector, { limit: 2, cursor: first.cursor })),
    ).toBe("INVALID_CURSOR");
    expect(
      await errorCode(() =>
        service.findReferences(selector, { limit: 1, cursor: "not-a-real-cursor" }),
      ),
    ).toBe("INVALID_CURSOR");

    // The snapshot the cursor pins keeps answering after the table moves on.
    const pinned = await service.findReferences(selector, { limit: 1, cursor: first.cursor });
    expect(pinned.hits.length).toBeGreaterThan(0);
    expect(pinned.snapshot_version).toBe(first.snapshot_version);

    // ...and once that version is genuinely gone, the cursor reports as stale
    // rather than silently answering from a different generation.
    store.version += 1;
    expect(
      await errorCode(() => service.findReferences(selector, { limit: 1, cursor: first.cursor })),
    ).toBe("STALE_CURSOR");
  });

  test("an unrelated store failure is not mislabeled as a stale cursor", async () => {
    const { service, store, projectId } = indexed({ "lib.py": "def answer():\n    return 42\n" });
    const failure = new Error("storage unavailable");
    store.listReferenceRecords = async () => {
      throw failure;
    };

    try {
      await service.findReferences(select(projectId, "lib.py", "answer"));
      throw new Error("expected the store failure");
    } catch (error) {
      expect(error).toBe(failure);
    }
  });

  test("a well-formed cursor with foreign keys is rejected as invalid", async () => {
    // The defect this closes: a missing key used to surface as a bare property
    // error, leaking an internal message straight to the client.
    const { service, projectId } = indexed({ "lib.py": "def answer():\n    return 42\n" });
    const foreign = Buffer.from(JSON.stringify({ version: 1, offset: 0 }), "utf8").toString(
      "base64url",
    );

    expect(
      await errorCode(() =>
        service.findReferences(select(projectId, "lib.py", "answer"), { cursor: foreign }),
      ),
    ).toBe("INVALID_CURSOR");
  });

  test("a limit outside the allowed range is rejected", async () => {
    const { service, projectId } = indexed({ "lib.py": "def answer():\n    return 42\n" });
    const selector = select(projectId, "lib.py", "answer");

    expect(await errorCode(() => service.findReferences(selector, { limit: 0 }))).toBe(
      "INVALID_FILTER",
    );
    expect(await errorCode(() => service.findReferences(selector, { limit: 501 }))).toBe(
      "INVALID_FILTER",
    );
  });
});

describe("selection", () => {
  test.each([
    [{}],
    [{ chunk_id: "chunk", project: "project", path: "x.py", qualified_symbol: "x" }],
    [{ project: "project", path: "x.py" }],
  ])("the selector requires one complete mode (%p)", (payload) => {
    expect(() => DeclarationSelector.parse(payload)).toThrow();
  });

  test("a symbol this project lacks is a distinct answer from having no references", async () => {
    const { service, projectId } = indexed({
      "lib.py": "def answer():\n    return 42\n",
      "other.py": "class Holder:\n    def answer(self):\n        return 1\n",
    });

    try {
      await service.findReferences(select(projectId, "lib.py", "missing"));
      throw new Error("expected a CodeIndexingError");
    } catch (error) {
      if (!isCodeIndexingError(error)) throw error;
      expect(error.code).toBe("AMBIGUOUS_SYMBOL");
      expect(error.message).toContain("No declaration missing");
    }
  });

  test("declaration shapes use Python's persisted JSON encoding", () => {
    const { store } = indexed({ "lib.py": "def naïve(café):\n    return café\n" });
    const declaration = store.records.find(
      (row) => row.record_kind === "declaration" && row.source_qualified_symbol === "naïve",
    );

    expect(declaration?.shape_json).toBe(
      '[{"destructured":false,"kind":"positional","name":"caf\\u00e9","position":0,"required":true}]',
    );
  });

  test("a non-structural language is refused with its supported set", async () => {
    const { service, projectId } = indexed({
      "svc.go": "package main\n\nfunc Run() int {\n\treturn 1\n}\n",
    });

    try {
      await service.findReferences(select(projectId, "svc.go", "Run"));
      throw new Error("expected a CodeIndexingError");
    } catch (error) {
      if (!isCodeIndexingError(error)) throw error;
      expect(error.code).toBe("UNSUPPORTED_LANGUAGE");
      expect(error.message).toContain("javascript, python, tsx, typescript");
    }
  });

  test("a chunk id selects the declaration it names", async () => {
    const { service, store, projectId } = indexed({
      "lib.py": "def answer():\n    return 42\n",
      "main.py": "from lib import answer\n\ndef caller():\n    return answer()\n",
    });
    const declaration = store.chunks.find(
      (chunk) => chunk.path === "lib.py" && chunk.qualified_symbol === "answer",
    );

    const response = await service.findReferences(
      DeclarationSelector.parse({ chunk_id: declaration?.chunk_id }),
    );

    expect(response.selected.qualified_symbol).toBe("answer");
    expect(response.selected.project_id).toBe(projectId);
    expect(response.hits.some((hit) => hit.kind === "call")).toBe(true);
  });

  test("a chunk id that names no declaration is refused", async () => {
    const { service } = indexed({ "lib.py": "def answer():\n    return 42\n" });

    expect(
      await errorCode(() =>
        service.findReferences(DeclarationSelector.parse({ chunk_id: "nope:123" })),
      ),
    ).toBe("AMBIGUOUS_SYMBOL");
  });
});
