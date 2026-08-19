/**
 * Refactor analysis, ported from `tests/test_refactors.py`.
 *
 * Two properties carry the weight here. An `edit_start_byte`/`edit_end_byte`
 * pair is an instruction to splice bytes into a file, so several tests apply
 * the edit and assert on the resulting text rather than on the offsets alone --
 * an off-by-one that a numeric assertion would tolerate corrupts source. And
 * `counts`/`completeness` are computed from the full, unsliced result set, so
 * a mid-stream page reports the same totals as the last one; the cursor, not
 * completeness, is what signals that more pages remain.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { isCodeIndexingError } from "../src/errors.ts";
import {
  DeclarationSelector,
  type ParameterShape,
  type RefactorOperation,
  refactorFindings,
} from "../src/models.ts";
import { ReferenceService } from "../src/reference-service.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";
import { type InMemoryReferenceStore, indexedStore } from "./reference-fixtures.ts";

let temporary: string;

beforeEach(() => {
  temporary = temporaryDirectory();
});

afterEach(() => {
  removeDirectory(temporary);
});

function indexed(
  files: Readonly<Record<string, string>>,
  directory = "repo",
): { service: ReferenceService; store: InMemoryReferenceStore; projectId: string; root: string } {
  const root = path.join(temporary, directory);
  const store = indexedStore(root, files, { projectId: `project-${directory}` });
  return { service: new ReferenceService(store), store, projectId: store.project.id, root };
}

function select(projectId: string, file: string, symbol: string): DeclarationSelector {
  return DeclarationSelector.parse({ project: projectId, path: file, qualified_symbol: symbol });
}

function rename(newName: string): RefactorOperation {
  return { kind: "rename", new_name: newName };
}

function signature(parameters: ParameterShape[]): RefactorOperation {
  return { kind: "signature_change", parameters };
}

function parameter(
  name: string,
  kind: ParameterShape["kind"],
  required: boolean,
  position: number,
): ParameterShape {
  return { name, kind, required, position, destructured: false };
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

function bytesOf(root: string, file: string): Buffer {
  return fs.readFileSync(path.join(root, file));
}

describe("rename", () => {
  test("the imported name is marked but an alias call is not", async () => {
    const { service, projectId, root } = indexed({
      "auth.py": "def authorize(user):\n    return user\n",
      "consumer.py":
        "from auth import authorize as check\n\ndef run(user):\n    return check(user)\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "auth.py", "authorize"),
      rename("validate"),
    );

    expect(analysis.must_change.map((item) => `${item.path}:${item.kind}`).sort()).toEqual([
      "auth.py:write",
      "consumer.py:import",
    ]);
    expect(analysis.counts.must_change).toBe(2);
    expect(analysis.counts.evidence).toBe(1);
    const aliasCall = refactorFindings(analysis).find((item) => item.kind === "call");
    expect(aliasCall?.resolution).toBe("exact");
    expect(aliasCall?.edit_required).toBe(false);

    // The import's own range covers "authorize as check"; only the imported name
    // may be rewritten or the alias is destroyed along with it.
    const imported = analysis.must_change.find((item) => item.kind === "import");
    const source = bytesOf(root, "consumer.py");
    expect(
      source
        .subarray(imported?.edit_start_byte as number, imported?.edit_end_byte as number)
        .toString(),
    ).toBe("authorize");
  });

  test("exact qualified member calls are marked for edit", async () => {
    const { service, projectId } = indexed({
      "auth.py": "class Gate:\n    def authorize(self):\n        return self.authorize()\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "auth.py", "Gate.authorize"),
      rename("validate"),
    );

    const call = analysis.must_change.find((item) => item.kind === "call");
    expect(call?.written_name).toBe("self.authorize");
    expect(call?.edit_required).toBe(true);
  });

  test("identifier reads are marked for edit", async () => {
    const { service, projectId } = indexed({
      "lib.py": "def answer():\n    return 42\n\ncallback = answer\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "lib.py", "answer"),
      rename("result"),
    );

    expect(analysis.must_change.find((item) => item.kind === "read")?.written_name).toBe("answer");
  });

  test("a qualified call edits only the member name", async () => {
    // The reference range is wider than the identifier to rewrite:
    // `auth.authorize(u)` spans `auth.authorize`, and replacing that whole
    // range with the new name drops the module qualifier and breaks the call.
    const { service, projectId, root } = indexed({
      "auth.py": "def authorize(user):\n    return user\n",
      "caller.py": "import auth\n\ndef run(user):\n    return auth.authorize(user)\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "auth.py", "authorize"),
      rename("permit"),
    );

    const call = analysis.must_change.find((item) => item.path === "caller.py");
    const source = bytesOf(root, "caller.py");
    expect(source.subarray(call?.start_byte, call?.end_byte).toString()).toBe("auth.authorize");
    expect(
      source.subarray(call?.edit_start_byte as number, call?.edit_end_byte as number).toString(),
    ).toBe("authorize");
    const edited = Buffer.concat([
      source.subarray(0, call?.edit_start_byte as number),
      Buffer.from("permit"),
      source.subarray(call?.edit_end_byte as number),
    ]);
    expect(edited.toString()).toContain("return auth.permit(user)");
  });

  test("the declaration finding points at its own name", async () => {
    const { service, projectId, root } = indexed({
      "auth.py": "def authorize(user):\n    return user\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "auth.py", "authorize"),
      rename("permit"),
    );

    const declaration = analysis.must_change.find((item) => item.reason_code === "declaration");
    expect(
      bytesOf(root, "auth.py")
        .subarray(declaration?.edit_start_byte as number, declaration?.edit_end_byte as number)
        .toString(),
    ).toBe("authorize");
  });

  test("required rename edits are deduplicated by span", async () => {
    const { service, projectId } = indexed({
      "models.py": "class Model:\n    pass\n\nclass FrozenModel(Model):\n    pass\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "models.py", "Model"),
      rename("Entity"),
    );

    const spans = analysis.must_change
      .filter((item) => item.edit_start_byte !== null && item.edit_end_byte !== null)
      .map((item) => `${item.path}:${item.edit_start_byte}:${item.edit_end_byte}`);
    expect(spans.length).toBe(new Set(spans).size);
    expect(analysis.counts.must_change).toBe(2);
  });

  test.each(["$answer", "class"])("python rename rejects %s", async (newName) => {
    const { service, projectId } = indexed({ "lib.py": "def answer():\n    return 42\n" });

    expect(
      await errorCode(() =>
        service.analyzeRefactor(select(projectId, "lib.py", "answer"), rename(newName)),
      ),
    ).toBe("INVALID_REFACTOR");
  });

  test("rename validation uses the selected language", async () => {
    const python = indexed({ "lib.py": "def answer():\n    return 42\n" }, "python");
    const javascript = indexed({ "lib.js": "function answer() { return 42; }\n" }, "javascript");

    const pythonAnalysis = await python.service.analyzeRefactor(
      select(python.projectId, "lib.py", "answer"),
      rename("_answer"),
    );
    const javascriptAnalysis = await javascript.service.analyzeRefactor(
      select(javascript.projectId, "lib.js", "answer"),
      rename("$answer"),
    );

    expect((pythonAnalysis.operation as { new_name: string }).new_name).toBe("_answer");
    expect((javascriptAnalysis.operation as { new_name: string }).new_name).toBe("$answer");
  });

  test("edit spans from a stale file are suppressed", async () => {
    const { service, projectId } = indexed({
      "auth.py": "def authorize(user):\n    return user\n",
      "consumer.py": "from auth import authorize\n\ndef run(user):\n    return authorize(user)\n",
    });
    // consumer.py changed on disk without a reindex: its stored offsets describe
    // bytes that no longer exist, so no edit may be derived from them -- the
    // wrong-edit hazard the serve-time hash gate exists for.
    fs.writeFileSync(
      path.join(temporary, "repo", "consumer.py"),
      "from auth import authorize\n\n\ndef run(user):\n    return authorize(user)\n",
    );

    const analysis = await service.analyzeRefactor(
      select(projectId, "auth.py", "authorize"),
      rename("validate"),
    );

    expect(analysis.must_change.every((item) => item.path !== "consumer.py")).toBe(true);
    expect(
      analysis.limitations.some(
        (item) => item.code === "stale_file" && item.explanation.includes("consumer.py"),
      ),
    ).toBe(true);
    expect(analysis.completeness.state).toBe("incomplete");
  });

  test("a stale selected declaration has no edit span", async () => {
    const { service, projectId, root } = indexed({
      "auth.py": "def authorize(user):\n    return user\n",
    });
    fs.writeFileSync(
      path.join(root, "auth.py"),
      "# authorize moved\ndef authorize(user):\n    return user\n",
    );

    const analysis = await service.analyzeRefactor(
      select(projectId, "auth.py", "authorize"),
      rename("validate"),
    );
    const declaration = analysis.must_change.find((item) => item.reason_code === "declaration");

    expect([declaration?.edit_start_byte, declaration?.edit_end_byte]).toEqual([null, null]);
    expect(analysis.limitations.some((item) => item.code === "stale_file")).toBe(true);
    expect(analysis.completeness.state).toBe("incomplete");
  });
});

describe("signature change", () => {
  test("a renamed keyword marks the exact call with a stable reason", async () => {
    const { service, projectId } = indexed({
      "mail.py": "def send(message):\n    return message\n",
      "consumer.py": "from mail import send\n\nsend(message='hi')\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "mail.py", "send"),
      signature([parameter("body", "positional", true, 0)]),
    );

    expect(analysis.must_change.find((item) => item.kind === "call")?.reason_code).toBe(
      "invalid_keyword",
    );
  });

  test("spread calls are reviewed, not silently ignored", async () => {
    const { service, projectId } = indexed({
      "mail.py": "def send(message):\n    return message\n",
      "consumer.py": "from mail import send\n\nargs = ('hi',)\nsend(*args)\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "mail.py", "send"),
      signature([
        parameter("message", "positional", true, 0),
        parameter("timeout", "positional", true, 1),
      ]),
    );

    expect(analysis.review.find((item) => item.kind === "call")?.reason_code).toBe(
      "spread_uncertainty",
    );
  });

  test("a keyword satisfies a required positional parameter", async () => {
    const { service, projectId } = indexed({
      "mail.py": "def send(message):\n    return message\n",
      "consumer.py": "from mail import send\n\nsend(message='hi')\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "mail.py", "send"),
      signature([parameter("message", "positional", true, 0)]),
    );

    expect(analysis.evidence.find((item) => item.kind === "call")?.reason_code).toBe(
      "direct_import_alias",
    );
  });

  test("a bound receiver does not consume a call argument", async () => {
    const { service, projectId } = indexed({
      "mail.py": "class Mailer:\n    def send(self, message):\n        return self.send(message)\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "mail.py", "Mailer.send"),
      signature([
        parameter("self", "positional", true, 0),
        parameter("message", "positional", true, 1),
      ]),
    );

    expect(analysis.evidence.find((item) => item.kind === "call")?.reason_code).toBe(
      "known_owner_member",
    );
  });

  test("a keyword variadic parameter accepts otherwise unknown keywords", async () => {
    const { service, projectId } = indexed({
      "mail.py": "def send(**kwargs):\n    return kwargs\n\nsend(extra=1)\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "mail.py", "send"),
      signature([parameter("kwargs", "keyword_variadic", true, 0)]),
    );

    expect(analysis.must_change.find((item) => item.kind === "call")).toBeUndefined();
    expect(analysis.evidence.find((item) => item.kind === "call")).toBeDefined();
  });

  test.each([
    [
      "abstract class Base {\n  abstract run(a: number, b: number): void;\n  invoke(): void { this.run(1, 2); }\n}\n",
      "Base.run",
    ],
    [
      "class Service {\n  run = (a: number, b: number): number => a + b;\n  invoke(): number { return this.run(1, 2); }\n}\n",
      "Service.run",
    ],
  ])("typescript callable class members keep parameters (%#)", async (source, qualifiedSymbol) => {
    const { service, projectId } = indexed({ "service.ts": source });

    const analysis = await service.analyzeRefactor(
      select(projectId, "service.ts", qualifiedSymbol),
      signature([parameter("b", "positional", true, 0), parameter("a", "positional", true, 1)]),
    );

    expect(analysis.must_change.find((item) => item.kind === "call")?.reason_code).toBe(
      "positional_order_change",
    );
  });

  test("the old shape is fetched once, not once per call site", async () => {
    const { service, store, projectId } = indexed({
      "auth.py": "def authorize(user, level):\n    return user\n",
      "main.py":
        "from auth import authorize\n\n\n" +
        "def run():\n    return authorize(1, 2)\n\n\n" +
        "def run_again():\n    return authorize(3, 4)\n",
    });
    const calls: string[] = [];
    const real = store.declarationShapes.bind(store);
    store.declarationShapes = async (project, qualifiedSymbol, options) => {
      calls.push(qualifiedSymbol);
      return real(project, qualifiedSymbol, options);
    };

    const analysis = await service.analyzeRefactor(
      select(projectId, "auth.py", "authorize"),
      signature([
        parameter("user", "positional", true, 0),
        parameter("level", "positional", true, 1),
        parameter("extra", "positional", true, 2),
      ]),
    );

    // One fetch total -- two call sites are classified, so a per-hit rescan
    // would show up as more than one call here.
    expect(calls).toEqual(["authorize"]);
    expect(analysis.must_change).toHaveLength(2);
  });
});

describe("overrides", () => {
  test("renaming a base method surfaces the subclass override", async () => {
    const { service, projectId } = indexed({
      "base.py": "class Base:\n    def handle(self):\n        return 1\n",
      "child.py":
        "from base import Base\n\n\nclass Child(Base):\n    def handle(self):\n        return 2\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "base.py", "Base.handle"),
      rename("process"),
    );

    const override = analysis.likely_change.find(
      (item) => item.path === "child.py" && item.reason_code === "override_of_renamed_method",
    );
    expect(override?.resolution).toBe("likely");
    expect(
      analysis.must_change.some(
        (item) => item.path === "child.py" && item.reason_code === "override_of_renamed_method",
      ),
    ).toBe(false);
  });

  test("a base class imported through a barrel surfaces the override", async () => {
    const { service, projectId } = indexed({
      "pkg/impl.py": "class Base:\n    def handle(self):\n        return 1\n",
      "pkg/__init__.py": "from .impl import Base\n",
      "child.py":
        "from pkg import Base\n\n\nclass Child(Base):\n    def handle(self):\n        return 2\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "pkg/impl.py", "Base.handle"),
      rename("process"),
    );

    expect(
      analysis.likely_change.find(
        (item) => item.path === "child.py" && item.reason_code === "override_of_renamed_method",
      )?.resolution,
    ).toBe("likely");
  });

  test("an aliased base class still surfaces the override", async () => {
    // The inheritance check must consult the same alias-to-imported-name
    // mapping the direct-import path uses, not just the base class's real name,
    // or the override is silently dropped and completeness lies about having
    // fully accounted for the rename (finding 5).
    const { service, projectId } = indexed({
      "base.py": "class Base:\n    def handle(self):\n        return 1\n",
      "child.py":
        "from base import Base as B\n\n\nclass C(B):\n    def handle(self):\n        return 2\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "base.py", "Base.handle"),
      rename("process"),
    );

    const override = analysis.likely_change.find(
      (item) => item.path === "child.py" && item.reason_code === "override_of_renamed_method",
    );
    expect(override?.resolution).toBe("likely");
    expect(analysis.completeness.state).toBe("complete_with_dynamic_limitations");
  });

  test("an unrelated aliased import is not treated as the base class", async () => {
    const { service, projectId } = indexed({
      "base.py":
        "class Base:\n    def handle(self):\n        return 1\n\n" +
        "class Other:\n    def handle(self):\n        return 2\n",
      "child.py":
        "from base import Other as B\n\nclass Child(B):\n    def handle(self):\n        return 3\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "base.py", "Base.handle"),
      rename("process"),
    );

    expect(
      analysis.likely_change.some((item) => item.reason_code === "override_of_renamed_method"),
    ).toBe(false);
  });

  test("same-file namespace heritage is not bound to a local namesake", async () => {
    const { service, projectId } = indexed({
      "base.py":
        "import other\n\n" +
        "class Base:\n    def handle(self):\n        return 1\n\n" +
        "class Child(other.Base):\n    def handle(self):\n        return 2\n",
      "other.py": "class Base:\n    def handle(self):\n        return 3\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "base.py", "Base.handle"),
      rename("process"),
    );

    expect(
      analysis.likely_change.some((item) => item.reason_code === "override_of_renamed_method"),
    ).toBe(false);
  });

  test("a namespace-imported base class surfaces the override", async () => {
    const { service, projectId } = indexed({
      "base.py": "class Base:\n    def handle(self):\n        return 1\n",
      "child.py":
        "import base\n\nclass Child(base.Base):\n    def handle(self):\n        return 2\n",
    });

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

  test("a transitive subclass override is surfaced", async () => {
    const { service, projectId } = indexed({
      "base.py": "class Base:\n    def handle(self):\n        return 1\n",
      "mid.py":
        "from base import Base\n\n\nclass Mid(Base):\n    def handle(self):\n        return 2\n",
      "leaf.py":
        "from mid import Mid\n\n\nclass Leaf(Mid):\n    def handle(self):\n        return 3\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "base.py", "Base.handle"),
      rename("process"),
    );

    expect(
      analysis.likely_change
        .filter((item) => item.reason_code === "override_of_renamed_method")
        .map((item) => item.path)
        .sort(),
    ).toEqual(["leaf.py", "mid.py"]);
  });

  test("javascript overrides are surfaced", async () => {
    const { service, projectId } = indexed({
      "base.js": "export class Base {\n  handle() {\n    return 1;\n  }\n}\n",
      "child.js":
        "import { Base } from './base';\n\nexport class Child extends Base {\n  handle() {\n    return 2;\n  }\n}\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "base.js", "Base.handle"),
      rename("process"),
    );

    expect(
      analysis.likely_change.find(
        (item) => item.path === "child.js" && item.reason_code === "override_of_renamed_method",
      )?.resolution,
    ).toBe("likely");
  });

  test("typescript namespace heritage surfaces the override", async () => {
    const { service, projectId } = indexed({
      "base.ts": "export class Base { handle(): void {} }\n",
      "child.ts":
        "import * as ns from './base';\nexport class Child extends ns.Base { handle(): void {} }\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "base.ts", "Base.handle"),
      rename("process"),
    );

    expect(
      analysis.likely_change.find(
        (item) => item.path === "child.ts" && item.reason_code === "override_of_renamed_method",
      )?.resolution,
    ).toBe("likely");
  });

  test("declarations are fetched by exact qualified symbol, never project-wide", async () => {
    // The declaration lookup, the transitive-override walk, and the old-shape
    // comparison all used to scan the record set for declaration rows -- which
    // no longer carries them at all, so this also proves the rename path does
    // not silently fall back to an empty declaration set (S4/E3).
    const { service, store, projectId } = indexed({
      "base.py": "class Base:\n    def handle(self):\n        return 1\n",
      "mid.py":
        "from base import Base\n\n\nclass Mid(Base):\n    def handle(self):\n        return 2\n",
    });
    const calls: string[] = [];
    const real = store.declarationShapes.bind(store);
    store.declarationShapes = async (project, qualifiedSymbol, options) => {
      calls.push(qualifiedSymbol);
      return real(project, qualifiedSymbol, options);
    };

    const analysis = await service.analyzeRefactor(
      select(projectId, "base.py", "Base.handle"),
      rename("process"),
    );

    // Exactly the qualified symbols this rename actually needed -- the renamed
    // declaration, the owner class for the override walk, and the one override
    // found while walking it.
    expect([...new Set(calls)].sort()).toEqual(["Base", "Base.handle", "Mid.handle"]);
    expect(analysis.must_change.find((item) => item.reason_code === "declaration")?.path).toBe(
      "base.py",
    );
    expect(
      analysis.likely_change.find((item) => item.reason_code === "override_of_renamed_method")
        ?.path,
    ).toBe("mid.py");
  });
});

describe("pagination, counts and completeness", () => {
  const callers = Array.from(
    { length: 501 },
    (_, index) => `def caller_${index}():\n    return answer()\n\n`,
  ).join("");

  test("pagination is independent of completeness and counts", async () => {
    // A mid-stream page is not a coverage gap: `cursor` alone signals more pages
    // remain, while completeness and counts come from the full, unsliced result
    // set and so are identical on every page (R4).
    const { service, projectId } = indexed({
      "lib.py": `def answer():\n    return 42\n\n${callers}`,
    });
    const selector = select(projectId, "lib.py", "answer");

    const first = await service.analyzeRefactor(selector, rename("result"));
    expect(first.cursor).not.toBeNull();
    expect(first.completeness.state).toBe("complete");
    expect(first.counts.must_change).toBe(502);

    const second = await service.analyzeRefactor(selector, rename("result"), {
      cursor: first.cursor,
    });
    expect(second.cursor).toBeNull();
    expect(second.completeness.state).toBe("complete");
    expect(second.counts.must_change).toBe(502);
  });

  test("the cursor is bound to the operation and the page limit", async () => {
    // Neither dimension used to be bound into the cursor payload, so page 2
    // could accept a different `new_name` (or apply a rename's edits under a
    // signature-change operation) or a different page size than page 1 used.
    const { service, projectId } = indexed({
      "lib.py": `def answer():\n    return 42\n\n${callers}`,
    });
    const selector = select(projectId, "lib.py", "answer");

    const first = await service.analyzeRefactor(selector, rename("result"));
    expect(first.cursor).not.toBeNull();

    expect(
      await errorCode(() =>
        service.analyzeRefactor(selector, rename("different"), { cursor: first.cursor }),
      ),
    ).toBe("INVALID_CURSOR");
    expect(
      await errorCode(() =>
        service.analyzeRefactor(selector, rename("result"), { cursor: first.cursor, limit: 10 }),
      ),
    ).toBe("INVALID_CURSOR");

    // The identical operation and limit are accepted, unaffected by binding.
    const second = await service.analyzeRefactor(selector, rename("result"), {
      cursor: first.cursor,
    });
    expect(second.cursor).toBeNull();

    const calls = [first, second]
      .flatMap((analysis) => analysis.must_change)
      .filter((item) => item.kind === "call");
    expect(calls).toHaveLength(501);
  });

  test("a Unicode rename uses Python's operation digest", async () => {
    const { service, projectId } = indexed({
      "lib.py": `def answer():\n    return 42\n\n${callers}`,
    });

    const first = await service.analyzeRefactor(
      select(projectId, "lib.py", "answer"),
      rename("café"),
    );
    const payload = JSON.parse(
      Buffer.from(first.cursor as string, "base64url").toString("utf8"),
    ) as { operation_digest: string };

    expect(payload.operation_digest).toBe("3c7376a9bf2c93e7");
  });

  test("an unanalyzable language makes the analysis incomplete", async () => {
    const { service, projectId } = indexed({
      "auth.py": "def authorize(user):\n    return user\n",
      "client.go": "package main\n\nfunc Run() int {\n\treturn 1\n}\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "auth.py", "authorize"),
      rename("permit"),
    );

    expect(analysis.completeness.state).toBe("incomplete");
    expect(analysis.limitations.some((item) => item.code === "unsupported_language")).toBe(true);
  });

  test("a declaration without reference extraction is refused", async () => {
    // Answering at all would mean reporting "rename one line" for a Go function
    // whose callers this index never looked at.
    const { service, projectId } = indexed({
      "svc.go": "package main\n\nfunc Authorize(u string) string {\n\treturn u\n}\n",
      "use.go": 'package main\n\nfunc Run() string {\n\treturn Authorize("a")\n}\n',
    });

    expect(
      await errorCode(() =>
        service.analyzeRefactor(select(projectId, "svc.go", "Authorize"), rename("Permit")),
      ),
    ).toBe("UNSUPPORTED_LANGUAGE");
  });

  test("an unproven call keeps the analysis out of the complete state", async () => {
    const { service, projectId } = indexed({
      "lib.py": "def answer():\n    return 42\n",
      "main.py": "def caller(thing):\n    return thing.answer()\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "lib.py", "answer"),
      rename("result"),
    );

    expect(analysis.counts.likely_change).toBeGreaterThanOrEqual(1);
    expect(analysis.completeness.state).toBe("complete_with_dynamic_limitations");
  });

  test("an ambiguous selector is refused", async () => {
    const { service, projectId } = indexed({ "lib.py": "def answer():\n    return 1\n" });

    expect(
      await errorCode(() =>
        service.analyzeRefactor(select(projectId, "lib.py", "missing"), rename("result")),
      ),
    ).toBe("AMBIGUOUS_SYMBOL");
  });
});

describe("work is not repeated", () => {
  test("the reference table is fetched only once", async () => {
    // analyzeRefactor must reuse findReferences' fetch, not re-scan (S4).
    const { service, store, projectId } = indexed({
      "auth.py": "def authorize(user):\n    return user\n",
    });
    const versions: Array<number | undefined> = [];
    const real = store.listReferenceRecords.bind(store);
    store.listReferenceRecords = async (project, options) => {
      versions.push(options.version);
      return real(project, options);
    };

    await service.analyzeRefactor(select(projectId, "auth.py", "authorize"), rename("permit"));

    expect(versions).toHaveLength(1);
  });

  test("the full hit list is classified only once", async () => {
    // The full-table fetch was already de-duplicated; the remaining, more
    // expensive duplication was the classification pass itself -- which walks
    // every reference row and reads every referenced file -- running once inside
    // findReferences and again inside analyzeRefactor to get an unpaginated
    // list (E1). The pass issues exactly one narrowed declaration fetch, so
    // counting those counts passes.
    const { service, store, projectId } = indexed({
      "auth.py": "def authorize(user):\n    return user\n",
      "main.py":
        "from auth import authorize\n\n\n" +
        "def run():\n    return authorize(1)\n\n\n" +
        "def run_again():\n    return authorize(2)\n",
    });
    let passes = 0;
    const real = store.declarationsForFiles.bind(store);
    store.declarationsForFiles = async (project, fileIds, options) => {
      passes += 1;
      return real(project, fileIds, options);
    };

    const analysis = await service.analyzeRefactor(
      select(projectId, "auth.py", "authorize"),
      rename("permit"),
    );

    expect(passes).toBe(1);
    // The single pass must still be a correct, consistent result.
    expect([...new Set(analysis.must_change.map((item) => item.path))].sort()).toEqual([
      "auth.py",
      "main.py",
    ]);
    expect(analysis.completeness.state).toBe("complete");
  });
});

describe("typescript scopes carry no blanket limitation", () => {
  test("a plain function rename reaches the complete state", async () => {
    const { service, projectId } = indexed({
      "lib.ts": "export function answer(): number { return 42; }\n",
      "main.ts": "import { answer } from './lib';\nanswer();\n",
    });

    const response = await service.findReferences(select(projectId, "lib.ts", "answer"));
    const analysis = await service.analyzeRefactor(
      select(projectId, "lib.ts", "answer"),
      rename("result"),
    );

    expect(response.limitations.some((item) => item.code === "extraction_gaps")).toBe(false);
    expect(analysis.limitations.some((item) => item.code === "extraction_gaps")).toBe(false);
    expect(analysis.completeness.state).toBe("complete");
  });

  test("a class rename finds the heritage reference", async () => {
    // E1 is fixed: class-heritage extraction surfaces `extends Base`, so
    // renaming a base class is no longer a known-wrong-answer case.
    const { service, projectId } = indexed({
      "base.ts": "export class Base {\n  run(): number { return 1; }\n}\n",
      "child.ts": "import { Base } from './base';\n\nexport class Child extends Base {}\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "base.ts", "Base"),
      rename("Foundation"),
    );

    const inheritance = [...analysis.must_change, ...analysis.likely_change].find(
      (item) => item.path === "child.ts" && item.kind === "inheritance",
    );
    expect(["exact", "likely"]).toContain(inheritance?.resolution ?? "");
    expect(analysis.limitations.some((item) => item.code === "extraction_gaps")).toBe(false);
    expect(analysis.completeness.state).not.toBe("incomplete");
  });

  test("a JSX component use resolves exactly", async () => {
    // E14: a `<Widget />` use is its own `type_use` reference row, so it
    // resolves exactly and a TSX scope needs no narrow limitation naming a gap.
    const { service, projectId } = indexed({
      "widget.tsx": "export function Widget(): JSX.Element {\n  return <div />;\n}\n",
      "main.tsx":
        "import { Widget } from './widget';\nexport function App(): JSX.Element {\n  return <Widget />;\n}\n",
    });

    const response = await service.findReferences(select(projectId, "widget.tsx", "Widget"));

    expect(response.limitations.some((item) => item.code === "extraction_gaps")).toBe(false);
    expect(
      response.hits.find((item) => item.path === "main.tsx" && item.kind === "type_use")
        ?.resolution,
    ).toBe("exact");
  });

  test("a python-only scope is unaffected", async () => {
    const { service, projectId } = indexed({
      "lib.py": "class Base:\n    def run(self):\n        return 1\n",
    });

    const analysis = await service.analyzeRefactor(
      select(projectId, "lib.py", "Base"),
      rename("Foundation"),
    );

    expect(analysis.limitations.some((item) => item.code === "extraction_gaps")).toBe(false);
    expect(analysis.completeness.state).toBe("complete");
  });
});
