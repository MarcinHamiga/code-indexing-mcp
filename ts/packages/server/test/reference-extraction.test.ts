/**
 * Structural reference extraction, ported from `tests/test_reference_extraction.py`.
 *
 * These are the E1-E14 hardening cases: every one is a *silent* defect -- a
 * dropped, mis-shaped, or fabricated reference row changes nothing about the
 * chunks, so nothing fails and a rename simply misses a call site. The
 * committed snapshot covers the corpus; this covers the rules.
 *
 * The fixtures are all ASCII, so a character index into the source is also its
 * byte offset, which is what the offset assertions rely on.
 */

import { describe, expect, test } from "bun:test";
import { TreeSitterExtractor } from "../src/extractor.ts";
import type { ExtractedReference, ExtractionResult } from "../src/models.ts";

const encoder = new TextEncoder();

function extract(text: string, language = "python"): ExtractionResult {
  return new TreeSitterExtractor().extract(`sample.${language}`, language, encoder.encode(text));
}

function references(text: string, language = "python"): ExtractedReference[] {
  return extract(text, language).references;
}

function byKindAndName(items: ExtractedReference[]): Map<string, ExtractedReference> {
  return new Map(items.map((item) => [`${item.kind}:${item.written_name}`, item]));
}

function required(map: Map<string, ExtractedReference>, key: string): ExtractedReference {
  const value = map.get(key);
  if (value === undefined) throw new Error(`no reference ${key} in ${[...map.keys()].join(", ")}`);
  return value;
}

function lastIndexOf(text: string, needle: string): number {
  return text.lastIndexOf(needle);
}

describe("identifier reads", () => {
  test.each([
    ["python", "def answer():\n    return 42\n\ncallback = answer\n"],
    ["javascript", "function answer() { return 42; }\nconst callback = answer;\n"],
    ["typescript", "function answer(): number { return 42; }\nconst callback = answer;\n"],
    ["tsx", "function answer(): number { return 42; }\nconst callback = answer;\n"],
  ])("%s records identifier value reads", (language, text) => {
    const reads = references(text, language).filter((item) => item.kind === "read");

    expect(reads.map((item) => item.written_name)).toEqual(["answer"]);
    const start = lastIndexOf(text, "answer");
    expect([reads[0]?.start_byte, reads[0]?.end_byte]).toEqual([start, start + "answer".length]);
  });
});

describe("python structural references", () => {
  test("structural references carry exact ranges", () => {
    const text =
      "from pkg import Widget as LocalWidget\n" +
      "import tools as util\n\n" +
      "@trace(enabled=True)\n" +
      "class Child(Base, protocol.Marker):\n" +
      "    value: LocalWidget\n\n" +
      "    def run(self, first, /, second: LocalWidget, *items, option: int, **kwargs)" +
      " -> LocalWidget:\n" +
      "        return util.make(first, *items, option=option, **kwargs)\n";

    const result = extract(text, "python");
    const refs = byKindAndName(result.references);

    const imported = required(refs, "import:LocalWidget");
    expect([imported.module_path, imported.imported_name, imported.alias]).toEqual([
      "pkg",
      "Widget",
      "LocalWidget",
    ]);
    const importStart = text.indexOf("Widget as LocalWidget");
    expect([
      imported.start_byte,
      imported.end_byte,
      imported.start_line,
      imported.end_line,
    ]).toEqual([importStart, importStart + "Widget as LocalWidget".length, 1, 1]);
    expect([
      required(refs, "import:util").module_path,
      required(refs, "import:util").imported_name,
    ]).toEqual(["tools", null]);
    expect(required(refs, "decorator:trace").source_qualified_symbol).toBe("Child");
    expect(
      [
        required(refs, "inheritance:Base").written_name,
        required(refs, "inheritance:protocol.Marker").written_name,
      ].sort(),
    ).toEqual(["Base", "protocol.Marker"]);
    // The receiver-bearing type_use lands on one of the two enclosing scopes
    // depending on where the annotation sits.
    expect(["Child", "Child.run"]).toContain(
      required(refs, "type_use:LocalWidget").source_qualified_symbol ?? "",
    );

    const call = required(refs, "call:util.make");
    expect(call.source_qualified_symbol).toBe("Child.run");
    expect(call.call_shape?.positional_count).toBe(1);
    expect(call.call_shape?.keywords).toEqual(["option"]);
    expect(call.call_shape?.has_positional_spread).toBe(true);
    expect(call.call_shape?.has_keyword_spread).toBe(true);
    const callStart = text.indexOf("util.make");
    expect([call.start_byte, call.end_byte, call.start_line, call.end_line]).toEqual([
      callStart,
      callStart + "util.make".length,
      9,
      9,
    ]);

    const declaration = result.declarations.find((item) => item.qualified_symbol === "Child.run");
    expect(
      declaration?.parameters.map((item) => [item.name, item.kind, item.required, item.position]),
    ).toEqual([
      ["self", "positional_only", true, 0],
      ["first", "positional_only", true, 1],
      ["second", "positional", true, 2],
      ["items", "variadic", true, 3],
      ["option", "keyword_only", true, 4],
      ["kwargs", "keyword_variadic", true, 5],
    ]);
  });

  test("parameter modes, import targets and a direct call", () => {
    const text =
      "from pkg.auth import enforce as check\n\n" +
      "def run(self, user, *, strict=True):\n" +
      "    return check(user, strict=strict)\n";
    const result = extract(text, "python");
    const refs = byKindAndName(result.references);

    const imported = required(refs, "import:check");
    expect([
      imported.target_name,
      imported.imported_name,
      imported.alias,
      imported.module_path,
    ]).toEqual(["enforce", "enforce", "check", "pkg.auth"]);
    const call = required(refs, "call:check");
    expect(call.target_name).toBe("check");
    expect(call.source_qualified_symbol).toBe("run");
    expect([call.call_shape?.positional_count, call.call_shape?.keywords]).toEqual([1, ["strict"]]);
    const callStart = text.indexOf("check(user");
    expect([call.start_byte, call.end_byte, call.start_line, call.end_line]).toEqual([
      callStart,
      callStart + "check".length,
      4,
      4,
    ]);
    expect(
      result.declarations
        .find((item) => item.qualified_symbol === "run")
        ?.parameters.map((item) => [item.name, item.kind, item.required]),
    ).toEqual([
      ["self", "positional", true],
      ["user", "positional", true],
      ["strict", "keyword_only", false],
    ]);
  });

  test("relative and wildcard imports", () => {
    const imports = new Map(
      references("from . import x\nfrom ..pkg import y\nfrom pkg import *\n")
        .filter((item) => item.kind === "import")
        .map((item) => [item.written_name, item]),
    );

    expect([
      imports.get("x")?.target_name,
      imports.get("x")?.module_path,
      imports.get("x")?.alias,
    ]).toEqual(["x", ".", null]);
    expect([
      imports.get("y")?.target_name,
      imports.get("y")?.module_path,
      imports.get("y")?.alias,
    ]).toEqual(["y", "..pkg", null]);
    expect([
      imports.get("*")?.target_name,
      imports.get("*")?.module_path,
      imports.get("*")?.alias,
    ]).toEqual(["*", "pkg", null]);
  });

  test("aliased relative and wildcard imports", () => {
    const imports = new Map(
      references("from .pkg import x as y\nfrom ..pkg import a as b\nfrom ...pkg import *\n")
        .filter((item) => item.kind === "import")
        .map((item) => [item.written_name, item]),
    );

    expect([
      imports.get("y")?.target_name,
      imports.get("y")?.module_path,
      imports.get("y")?.alias,
    ]).toEqual(["x", ".pkg", "y"]);
    expect([
      imports.get("b")?.target_name,
      imports.get("b")?.module_path,
      imports.get("b")?.alias,
    ]).toEqual(["a", "..pkg", "b"]);
    expect([
      imports.get("*")?.target_name,
      imports.get("*")?.module_path,
      imports.get("*")?.alias,
    ]).toEqual(["*", "...pkg", null]);
  });

  test("member access read and write carry the receiver", () => {
    // E5: attribute assignment/read are no longer swallowed by the `left` exclusion.
    const items = references("config.TIMEOUT = 10\nprint(config.TIMEOUT)\n", "python");

    const write = items.find((item) => item.kind === "write");
    const read = items.find(
      (item) => item.kind === "read" && item.written_name === "config.TIMEOUT",
    );
    expect([write?.target_name, write?.written_name, write?.receiver_text]).toEqual([
      "config.TIMEOUT",
      "config.TIMEOUT",
      "config",
    ]);
    expect([read?.target_name, read?.written_name, read?.receiver_text]).toEqual([
      "config.TIMEOUT",
      "config.TIMEOUT",
      "config",
    ]);
    // The bare receiver identifier still surfaces as its own `read` on the read line.
    expect(items.some((item) => item.kind === "read" && item.written_name === "config")).toBe(true);
  });

  test("a member call does not duplicate as a member access", () => {
    const text = "widget.render()\n";
    const start = text.indexOf("widget.render");
    const matching = references(text, "python").filter(
      (item) => item.start_byte === start && item.end_byte === start + "widget.render".length,
    );

    expect(matching.map((item) => item.kind)).toEqual(["call"]);
  });

  test("a lambda default parameter value is a read", () => {
    // Mirrors the JS/TS case below: the outer `lambda`-level exclusion used to
    // blanket-exclude its whole `parameters` field, undoing the correct decision
    // the parameter-defaults walk already made for the default value.
    const text = "LIMIT = 5\nf = lambda a=LIMIT: a\n";
    const reads = references(text, "python").filter((item) => item.kind === "read");
    const names = reads.map((item) => item.written_name);

    // `LIMIT` (the default value) and `a` (the body's use of the parameter) are
    // both genuine reads; the parameter *binding* itself is not.
    expect(names.filter((name) => name === "LIMIT")).toHaveLength(1);
    expect(names.filter((name) => name === "a")).toHaveLength(1);
    const limit = reads.find((item) => item.written_name === "LIMIT");
    const start = text.indexOf("LIMIT", text.indexOf("lambda"));
    expect([limit?.start_byte, limit?.end_byte]).toEqual([start, start + "LIMIT".length]);
  });

  test("an import list ignores a comment between names", () => {
    // A comment among parenthesized `from`-import names is a named "extra" node
    // too -- it used to fall through the `aliased_import` check and produce a
    // bogus `import` reference for the comment text itself.
    const imports = references("from pkg import (\n    a,\n    # comment\n    b,\n)\n").filter(
      (item) => item.kind === "import",
    );

    expect(imports.map((item) => item.written_name)).toEqual(["a", "b"]);
  });

  test("class heritage ignores a comment between base classes", () => {
    const inheritance = references(
      "class Child(\n    Base,\n    # comment\n    Other,\n):\n    pass\n",
      "python",
    ).filter((item) => item.kind === "inheritance");

    expect(inheritance.map((item) => item.written_name).sort()).toEqual(["Base", "Other"]);
  });
});

describe("javascript, typescript and tsx structural references", () => {
  test("structural syntax across the family", () => {
    const javascript =
      "import Default, { named as local } from 'pkg';\n" +
      "import * as ns from 'space';\n" +
      "export { local as exposed } from 'pkg';\n" +
      "class Child extends Base {\n" +
      "  method(first, ...rest) { this.run(first); ns.make(...rest, {ok: true}); }\n" +
      "}\n";
    const js = byKindAndName(references(javascript, "javascript"));
    expect(required(js, "import:Default").imported_name).toBe("default");
    expect(required(js, "import:local").imported_name).toBe("named");
    expect(required(js, "import:ns").imported_name).toBe("*");
    expect(required(js, "export:exposed").module_path).toBe("pkg");
    expect(required(js, "inheritance:Base").source_qualified_symbol).toBe("Child");
    expect(required(js, "call:this.run").source_qualified_symbol).toBe("Child.method");
    expect(required(js, "call:ns.make").call_shape?.has_positional_spread).toBe(true);

    const typescript =
      "interface Contract<T> extends Base<T> { value: T }\n" +
      "type Alias = Contract<string>;\n" +
      "function make<T>(value: T = undefined as T, ...rest: T[]): Contract<T> " +
      "{ return build<T>(value, ...rest); }\n";
    const tsResult = extract(typescript, "typescript");
    const ts = byKindAndName(tsResult.references);
    expect(required(ts, "inheritance:Base<T>").source_qualified_symbol).toBe("Contract");
    // `type Alias = Contract<string>;` -- the generic head descends to its own
    // type_use row instead of the whole expression verbatim (E2). `string` is a
    // predefined_type, not a type_identifier, so it stays out of scope here.
    expect(
      tsResult.references.some(
        (item) =>
          item.kind === "type_use" &&
          item.written_name === "Contract" &&
          item.source_qualified_symbol === "Alias",
      ),
    ).toBe(true);
    expect(required(ts, "call:build").call_shape?.type_argument_count).toBe(1);
    expect(
      tsResult.declarations
        .find((item) => item.qualified_symbol === "make")
        ?.parameters.map((item) => [item.name, item.kind, item.required]),
    ).toEqual([
      ["value", "positional", false],
      ["rest", "variadic", false],
    ]);

    const tsxResult = extract(
      "import View from './view';\n" +
        "type Props = { item: Item };\n" +
        "export function Screen({ item }: Props) { return <View item={item} />; }\n",
      "tsx",
    );
    expect(
      tsxResult.references.some((item) => item.kind === "import" && item.written_name === "View"),
    ).toBe(true);
    expect(
      tsxResult.references.some(
        (item) => item.kind === "type_use" && item.written_name === "Props",
      ),
    ).toBe(true);
    expect(tsxResult.declarations.some((item) => item.qualified_symbol === "Screen")).toBe(true);
  });

  test.each(["javascript", "typescript", "tsx"])(
    "%s: qualified class heritage is an inheritance reference",
    (language) => {
      const items = references("class Child extends ns.Base {}\n", language);
      const inheritance = items.filter((item) => item.kind === "inheritance");

      expect(inheritance.map((item) => [item.written_name, item.source_qualified_symbol])).toEqual([
        ["ns.Base", "Child"],
      ]);
      expect(items.some((item) => item.kind === "read" && item.written_name === "ns.Base")).toBe(
        false,
      );
    },
  );

  test.each(["typescript", "tsx"])(
    "%s: callable class members include parameter shapes",
    (language) => {
      const result = extract(
        "abstract class Base {\n" +
          "  abstract run(a: number, b: number): void;\n" +
          "  callback = (first: number, second: number): number => first + second;\n" +
          "}\n",
        language,
      );
      const declarations = new Map(
        result.declarations.map((item) => [item.qualified_symbol, item]),
      );

      expect(declarations.get("Base.run")?.parameters.map((item) => item.name)).toEqual(["a", "b"]);
      expect(declarations.get("Base.callback")?.parameters.map((item) => item.name)).toEqual([
        "first",
        "second",
      ]);
    },
  );

  test.each([
    [
      "javascript",
      "const run = (first = 1, ...rest) => rest;\nconst outer = function (value = 1, ...more) { return more; };\n",
    ],
    [
      "typescript",
      "const run = (first: number = 1, ...rest: number[]) => rest;\nconst outer = function (value: number = 1, ...more: number[]) { return more; };\n",
    ],
    [
      "tsx",
      "const run = (first: number = 1, ...rest: number[]) => <>{rest}</>;\nconst outer = function (value: number = 1, ...more: number[]) { return <>{more}</>; };\n",
    ],
  ])("%s: variable-assigned callables keep default and rest parameters", (language, text) => {
    const declarations = new Map(
      extract(text, language).declarations.map((item) => [item.qualified_symbol, item]),
    );

    for (const [qualified, first, second] of [
      ["run", "first", "rest"],
      ["outer", "value", "more"],
    ] as const) {
      const declaration = declarations.get(qualified);
      expect(declaration?.symbol).toBe(qualified);
      expect(declaration?.parameters.map((item) => [item.name, item.kind, item.required])).toEqual([
        [first, "positional", false],
        [second, "variadic", false],
      ]);
    }
  });

  test.each(["javascript", "typescript", "tsx"])(
    "%s: local, default and declaration exports",
    (language) => {
      const text =
        language === "tsx"
          ? "export { foo };\nexport default foo;\nexport function Screen() { return <div />; }\n"
          : "export { foo };\nexport default foo;\nexport function Screen() {}\n";
      const exports = new Map(
        references(text, language)
          .filter((item) => item.kind === "export")
          .map((item) => [item.written_name, item]),
      );

      expect([...exports.keys()].sort()).toEqual(["Screen", "default", "foo"]);
      expect([exports.get("foo")?.target_name, exports.get("foo")?.module_path]).toEqual([
        "foo",
        null,
      ]);
      expect([exports.get("default")?.target_name, exports.get("default")?.module_path]).toEqual([
        "foo",
        null,
      ]);
      expect([exports.get("Screen")?.target_name, exports.get("Screen")?.module_path]).toEqual([
        "Screen",
        null,
      ]);
    },
  );

  test.each(["typescript", "tsx"])("%s: optional parameters are not required", (language) => {
    const declaration = extract(
      "function f(value?: string) { return value; }\n",
      language,
    ).declarations.find((item) => item.qualified_symbol === "f");

    expect(declaration?.parameters.map((item) => [item.name, item.kind, item.required])).toEqual([
      ["value", "positional", false],
    ]);
  });

  test.each(["javascript", "typescript", "tsx"])(
    "%s: lexical and named default exports",
    (language) => {
      const declared = language === "javascript" ? "export let beta;" : "export let beta: number;";
      const body = language === "tsx" ? "{ return <div />; }" : "{}";
      const items = references(
        `export const alpha = 1, gamma = 2;\n${declared}\nexport default function named() ${body}\n`,
        language,
      );
      const exports = items.filter((item) => item.kind === "export");
      const byName = new Map(exports.map((item) => [item.written_name, item]));

      expect(exports).toHaveLength(4);
      expect(
        Object.fromEntries(
          ["alpha", "gamma", "beta", "default"].map((name) => [
            name,
            byName.get(name)?.target_name,
          ]),
        ),
      ).toEqual({ alpha: "alpha", gamma: "gamma", beta: "beta", default: "named" });
    },
  );

  test.each(["javascript", "typescript", "tsx"])(
    "%s: exports name binding identifiers and commented defaults",
    (language) => {
      const body = language === "tsx" ? "{ return <div />; }" : "{}";
      const items = references(
        "export var legacy = 3;\n" +
          "export const { first, renamed: local, nested: [second = 2, ...rest] } = obj;\n" +
          `export /* comment */ default function Commented() ${body}\n`,
        language,
      );
      const exports = items.filter((item) => item.kind === "export");
      const byName = new Map(exports.map((item) => [item.written_name, item]));

      expect(exports).toHaveLength(6);
      expect(
        Object.fromEntries(
          ["legacy", "first", "local", "second", "rest"].map((name) => [
            name,
            byName.get(name)?.target_name,
          ]),
        ),
      ).toEqual({
        legacy: "legacy",
        first: "first",
        local: "local",
        second: "second",
        rest: "rest",
      });
      expect([byName.get("default")?.target_name, byName.get("default")?.written_name]).toEqual([
        "Commented",
        "default",
      ]);
    },
  );

  test.each(["javascript", "typescript", "tsx"])(
    "%s: export-star and namespace export carry a module path",
    (language) => {
      // E3: barrel re-exports emit an `export` row instead of nothing (or a
      // bogus `read`).
      const items = references("export * from './x';\nexport * as ns from './x';\n", language);
      const exports = items.filter((item) => item.kind === "export");

      expect(exports).toHaveLength(2);
      const [bare, namespaced] = exports as [ExtractedReference, ExtractedReference];
      expect([bare.target_name, bare.written_name, bare.module_path, bare.alias]).toEqual([
        "*",
        "*",
        "./x",
        null,
      ]);
      expect([
        namespaced.target_name,
        namespaced.written_name,
        namespaced.module_path,
        namespaced.alias,
      ]).toEqual(["*", "ns", "./x", "ns"]);
      // The namespace alias must not also surface as a bare `read`.
      expect(items.some((item) => item.kind === "read" && item.written_name === "ns")).toBe(false);
    },
  );

  test.each(["javascript", "typescript", "tsx"])("%s: module edges stay visible", (language) => {
    // E9: side-effect imports and require()/dynamic import() keep their module path.
    const items = references(
      "import './polyfill';\nconst lazy = require('./lazy');\nconst dynamic = import('./dynamic');\n",
      language,
    );

    const bareImport = items.find((item) => item.kind === "import");
    expect([bareImport?.module_path, bareImport?.imported_name]).toEqual(["./polyfill", null]);

    const calls = new Map(
      items.filter((item) => item.kind === "call").map((item) => [item.target_name, item]),
    );
    expect(calls.get("require")?.module_path).toBe("./lazy");
    expect(calls.get("import")?.module_path).toBe("./dynamic");
  });

  test.each(["javascript", "typescript", "tsx"])(
    "%s: member access read and write carry the receiver",
    (language) => {
      // E5: member-expression assignment targets and plain reads are recorded.
      const items = references("target.TIMEOUT = 5;\ntarget.TIMEOUT;\n", language);

      const write = items.find((item) => item.kind === "write");
      const read = items.find(
        (item) => item.kind === "read" && item.written_name === "target.TIMEOUT",
      );
      expect([write?.target_name, write?.written_name, write?.receiver_text]).toEqual([
        "target.TIMEOUT",
        "target.TIMEOUT",
        "target",
      ]);
      expect([read?.target_name, read?.written_name, read?.receiver_text]).toEqual([
        "target.TIMEOUT",
        "target.TIMEOUT",
        "target",
      ]);
    },
  );

  test.each(["javascript", "typescript", "tsx"])(
    "%s: decorators produce decorator references",
    (language) => {
      // E6: `@Name`, `@ns.Name`, and `@Factory()` all yield a `decorator` row;
      // the factory call keeps its own additional `call` row.
      const text =
        "@sealed\n" +
        "class Plain {}\n\n" +
        "@ns.sealed\n" +
        "class Namespaced {}\n\n" +
        "@factory()\n" +
        "class Factored {\n" +
        "  @readonly\n" +
        "  handle() {}\n" +
        "}\n";
      const items = references(text, language);
      const bySpan = new Map(
        items.filter((item) => item.kind === "decorator").map((item) => [item.start_byte, item]),
      );

      const plain = bySpan.get(text.indexOf("sealed"));
      expect([plain?.target_name, plain?.written_name, plain?.source_qualified_symbol]).toEqual([
        "sealed",
        "sealed",
        "Plain",
      ]);
      const namespaced = bySpan.get(text.indexOf("ns.sealed"));
      expect([namespaced?.target_name, namespaced?.source_qualified_symbol]).toEqual([
        "ns.sealed",
        "Namespaced",
      ]);
      const factory = bySpan.get(text.indexOf("factory()"));
      expect([factory?.target_name, factory?.source_qualified_symbol]).toEqual([
        "factory",
        "Factored",
      ]);
      // The factory call keeps its own `call` row in addition to the decorator row.
      expect(items.some((item) => item.kind === "call" && item.written_name === "factory")).toBe(
        true,
      );
      expect(bySpan.get(text.indexOf("readonly"))?.source_qualified_symbol).toBe("Factored.handle");

      // No duplicate `read`/member-access row shares a decorator's span.
      const spans = new Set(
        items
          .filter((item) => item.kind === "decorator")
          .map((item) => `${item.start_byte}:${item.end_byte}`),
      );
      for (const item of items) {
        if (item.kind === "read" || item.kind === "write") {
          expect(spans.has(`${item.start_byte}:${item.end_byte}`)).toBe(false);
        }
      }
    },
  );

  test.each(["javascript", "typescript", "tsx"])(
    "%s: a destructured parameter is one marked positional slot",
    (language) => {
      // Expanding to N flat params would corrupt positional matching for every
      // caller (E7), so the extractor marks the slot and synthesizes a name.
      const text =
        language === "javascript"
          ? "function describe({ title, subtitle, footnote }) { return title; }\n"
          : "function describe({ title, subtitle, footnote }: " +
            "{ title: string; subtitle: string; footnote: string }) { return title; }\n";
      const declaration = extract(text, language).declarations.find(
        (item) => item.qualified_symbol === "describe",
      );

      expect(declaration?.parameters).toHaveLength(1);
      const parameter = declaration?.parameters[0];
      expect(parameter?.kind).toBe("positional");
      expect(parameter?.position).toBe(0);
      expect(parameter?.destructured).toBe(true);
      expect(parameter?.name).not.toContain("{");
      expect(parameter?.name).not.toContain("}");
    },
  );

  test.each(["typescript", "tsx"])(
    "%s: a callback-typed parameter is required, not defaulted",
    (language) => {
      // E8: a `=>` inside a parameter's callback type must not misfire a
      // text-based default heuristic and mark the parameter optional.
      const declaration = extract(
        "function bind(handler: (event: Event) => void, retries: number) { return retries; }\n",
        language,
      ).declarations.find((item) => item.qualified_symbol === "bind");

      expect(declaration?.parameters.map((item) => [item.name, item.kind, item.required])).toEqual([
        ["handler", "positional", true],
        ["retries", "positional", true],
      ]);
    },
  );

  test.each([
    ["javascript", "const LIMIT = 5;\nfunction f(a = LIMIT) {}\n"],
    ["typescript", "const LIMIT = 5;\nfunction f(a = LIMIT) {}\n"],
    ["javascript", "const LIMIT = 5;\nconst f = (a = LIMIT) => {};\n"],
  ])("%s: a default parameter value is a read", (language, text) => {
    // A plain `assignment_pattern` default (`a = LIMIT`) exposes its value under
    // the `right` field, not `value` -- only TS's typed parameter wrapper uses
    // `value`. Missing `right` silently dropped every identifier read inside an
    // untyped JS/TS default parameter (finding 6).
    const items = references(text, language);
    const reads = items.filter((item) => item.kind === "read");

    expect(reads.map((item) => item.written_name)).toEqual(["LIMIT"]);
    const start = lastIndexOf(text, "LIMIT");
    expect([reads[0]?.start_byte, reads[0]?.end_byte]).toEqual([start, start + "LIMIT".length]);
    // The parameter's own name must stay excluded -- it is a binding, not a read.
    expect(items.some((item) => item.written_name === "a")).toBe(false);
  });
});

describe("comments are never mistaken for content", () => {
  test.each([
    ["python", "def g(a, b):\n    pass\n\ng(1,  # note\n  2)\n"],
    ["javascript", "function g(a, b) {}\ng(1, /* c */ 2);\n"],
    ["typescript", "function g(a: number, b: number) {}\ng(1, /* c */ 2);\n"],
  ])("%s: a comment is not a positional argument", (language, text) => {
    const call = references(text, language).find((item) => item.kind === "call");

    expect(call?.call_shape?.positional_count).toBe(2);
    expect(call?.call_shape?.keywords).toEqual([]);
  });

  test("a comment between keyword arguments stays uncounted", () => {
    const call = references("def g(a=None, b=None):\n    pass\n\ng(a=1,  # note\n  b=2)\n").find(
      (item) => item.kind === "call",
    );

    expect(call?.call_shape?.positional_count).toBe(0);
    expect(call?.call_shape?.keywords).toEqual(["a", "b"]);
  });

  test("a module path survives a leading comment in the call arguments", () => {
    const call = references("require(/* c */ './mod');\n", "javascript").find(
      (item) => item.kind === "call",
    );

    expect(call?.module_path).toBe("./mod");
  });

  test("a namespace import alias survives a leading comment", () => {
    const imported = references("import * as /* c */ ns from 'mod';\n", "javascript").find(
      (item) => item.kind === "import",
    );

    expect([imported?.written_name, imported?.alias]).toEqual(["ns", "ns"]);
  });

  test("a namespace export alias survives a leading comment", () => {
    const exported = references("export * as /* c */ ns from './x';\n", "javascript").find(
      (item) => item.kind === "export",
    );

    expect(exported?.written_name).toBe("ns");
  });

  test("a decorator target survives a leading comment", () => {
    const decorator = references(
      "class A {\n  @/* c */ dec\n  method() {}\n}\n",
      "javascript",
    ).find((item) => item.kind === "decorator");

    expect(decorator?.written_name).toBe("dec");
  });

  test("a default import is not fabricated from a comment", () => {
    const imports = references("import Default, /* c */ { a } from 'mod';\n", "javascript").filter(
      (item) => item.kind === "import",
    );

    expect(imports.map((item) => item.written_name).sort()).toEqual(["Default", "a"]);
  });

  test("an extends_type_clause ignores a comment", () => {
    const inheritance = references("interface I extends /* c */ Base {}\n", "typescript").filter(
      (item) => item.kind === "inheritance",
    );

    expect(inheritance.map((item) => item.written_name)).toEqual(["Base"]);
  });

  test("a type annotation survives a leading comment", () => {
    const typeUses = references("function f(x: /* c */ Widget) {}\n", "typescript").filter(
      (item) => item.kind === "type_use",
    );

    expect(typeUses.map((item) => item.written_name)).toEqual(["Widget"]);
  });

  test("a comment among type arguments does not inflate the count", () => {
    const call = references(
      "function make<T>(value: T): Contract<T> { return build</* c */ T>(value); }\n",
      "typescript",
    ).find((item) => item.kind === "call" && item.target_name === "build");

    expect(call?.call_shape?.type_argument_count).toBe(1);
  });

  test("a rest parameter name survives a leading comment", () => {
    const declaration = extract("function f(.../* c */ rest) {}\n", "javascript").declarations.find(
      (item) => item.qualified_symbol === "f",
    );

    expect(declaration?.parameters.map((item) => [item.name, item.kind])).toEqual([
      ["rest", "variadic"],
    ]);
  });

  test("a destructured rest binding survives a leading comment", () => {
    const exported = references("export const [.../* c */ rest] = arr;\n", "javascript").find(
      (item) => item.kind === "export",
    );

    expect(exported?.written_name).toBe("rest");
  });
});
