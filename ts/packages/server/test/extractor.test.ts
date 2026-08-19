/**
 * Behavioural extractor tests, ported from `tests/test_extractor.py`.
 *
 * The committed snapshot in `extractor-equivalence.test.ts` proves the port
 * reproduces Python's output on the corpus; these prove *why* each output is
 * what it is, so a future change that happens to keep the corpus stable while
 * breaking a rule still fails.
 */

import { describe, expect, spyOn, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import Parser from "tree-sitter";
import { normalizeIdentifier, QUERY_DIRECTORIES, TreeSitterExtractor } from "../src/extractor.ts";
import {
  PACK_DOWNLOAD_ATTEMPTS,
  supportedLanguages,
  unavailableLanguages,
  withDownloadRetry,
} from "../src/grammars.ts";
import { CHUNK_KINDS } from "../src/models.ts";
import { LANGUAGES } from "../src/scanner.ts";
import { LineIndex } from "../src/source-text.ts";

const encoder = new TextEncoder();
function source(text: string): Uint8Array {
  return encoder.encode(text);
}

function symbolsOf(result: { chunks: Array<{ kind: string; qualified_symbol: string | null }> }) {
  return new Set(result.chunks.map((chunk) => `${chunk.kind}\u0000${chunk.qualified_symbol}`));
}

function expectSymbols(
  result: { chunks: Array<{ kind: string; qualified_symbol: string | null }> },
  expected: Array<[string, string]>,
): void {
  const symbols = symbolsOf(result);
  const missing = expected.filter(([kind, name]) => !symbols.has(`${kind}\u0000${name}`));
  expect(missing).toEqual([]);
}

describe("symbol extraction", () => {
  test("python symbols carry qualified methods and module code", () => {
    const result = new TreeSitterExtractor().extract(
      "pkg/greet.py",
      "python",
      source(
        'import os\nVALUE = 1\n\n@registered\nclass Greeter:\n    """Greets people."""\n\n    def hello(self, name: str) -> str:\n        return f"Hello {name}"\n\ndef standalone() -> None:\n    pass\n',
      ),
    );

    expectSymbols(result, [
      ["class", "Greeter"],
      ["method", "Greeter.hello"],
      ["function", "standalone"],
    ]);
    const classChunk = result.chunks.find((chunk) => chunk.kind === "class");
    expect(classChunk?.content.startsWith("@registered\nclass Greeter:")).toBe(true);
    expect(classChunk?.content).not.toContain("def hello");
    expect(
      result.chunks.some((chunk) => chunk.kind === "module" && chunk.content.includes("import os")),
    ).toBe(true);
  });

  test.each([
    [
      "javascript",
      "web/app.js",
      "const fetchUser = async (id) => id;\nclass Api { load() { return 1; } }\n",
      [
        ["function", "fetchUser"],
        ["class", "Api"],
        ["method", "Api.load"],
      ] as Array<[string, string]>,
    ],
    [
      "typescript",
      "web/types.ts",
      "interface User { name: string }\ntype UserId = string;\nenum Role { Admin }\n",
      [
        ["interface", "User"],
        ["type", "UserId"],
        ["enum", "Role"],
      ] as Array<[string, string]>,
    ],
    [
      "tsx",
      "web/app.tsx",
      "export const App = () => <main>Hello</main>;\n",
      [["function", "App"]] as Array<[string, string]>,
    ],
  ])("%s symbols", (language, file, text, expected) => {
    expectSymbols(new TreeSitterExtractor().extract(file, language, source(text)), expected);
  });

  test("TS abstract class members are qualified and the class is selectable", () => {
    // E10: abstract classes, abstract method signatures, and class-field arrows
    // used to produce no declaration at all, so members inside an abstract class
    // lost their `Worker.` qualification and the class itself was unselectable.
    const result = new TreeSitterExtractor().extract(
      "worker.ts",
      "typescript",
      source(
        "abstract class Worker {\n  abstract run(): number;\n\n  handle = (a: number, b: number): number => a + b;\n}\n",
      ),
    );

    expectSymbols(result, [
      ["class", "Worker"],
      ["method", "Worker.run"],
      ["method", "Worker.handle"],
    ]);
    const bodyRead = result.references.find(
      (reference) => reference.kind === "read" && reference.target_name === "a",
    );
    expect(bodyRead?.source_qualified_symbol).toBe("Worker.handle");
  });

  test.each([
    [
      "go",
      "pkg/user.go",
      "package user\n\ntype User struct {\n    Name string\n}\n\nfunc (u User) Greeting() string { return u.Name }\nfunc NewUser() *User { return &User{} }\nconst Version = 1\n",
      [
        ["class", "User"],
        ["method", "Greeting"],
        ["function", "NewUser"],
        ["constant", "Version"],
      ] as Array<[string, string]>,
    ],
    [
      "terraform",
      "infra/main.tf",
      'variable "region" {\n  default = "eu"\n}\n\nresource "aws_instance" "web" {\n  ami = "ami-123"\n}\n',
      [
        ["object", "region"],
        ["object", "aws_instance"],
      ] as Array<[string, string]>,
    ],
    [
      "rust",
      "src/lib.rs",
      "struct User {\n    name: String,\n}\n\nenum State { Ready, Done }\nfn build_user() -> User { User { name: String::new() } }\nconst VERSION: u32 = 1;\n",
      [
        ["struct", "User"],
        ["enum", "State"],
        ["function", "build_user"],
        ["constant", "VERSION"],
      ] as Array<[string, string]>,
    ],
    [
      "c",
      "src/math.c",
      "#define LIMIT 10\ntypedef struct User { int id; } User;\nint add(int left, int right) { return left + right; }\n",
      [
        ["constant", "LIMIT"],
        ["struct", "User"],
        ["function", "add"],
      ] as Array<[string, string]>,
    ],
    [
      "cpp",
      "src/user.cpp",
      "class User {\npublic:\n    void greet() {}\n};\n\nint build_user() { return 1; }\n",
      [
        ["class", "User"],
        ["method", "User.greet"],
        ["function", "build_user"],
      ] as Array<[string, string]>,
    ],
    [
      "lua",
      "lua/user.lua",
      'local function greet(name)\n  return "hello " .. name\nend\n',
      [["function", "greet"]] as Array<[string, string]>,
    ],
  ])("%s symbols", (language, file, text, expected) => {
    const result = new TreeSitterExtractor().extract(file, language, source(text));

    expect(result.has_errors).toBe(false);
    expectSymbols(result, expected);
  });

  test("java symbols carry precise kinds and nested qualification", () => {
    const result = new TreeSitterExtractor().extract(
      "src/demo/Types.java",
      "java",
      source(`package demo;

@interface Flag {
    String value();
}

interface Service {
    void run();
}

enum State {
    ON;

    void reset() {}
}

record User(String name) {
    User {}

    String value() {
        return name;
    }
}

class Outer {
    Outer() {}

    class Inner {
        void work() {}
    }
}
`),
    );

    expectSymbols(result, [
      ["annotation", "Flag"],
      ["method", "Flag.value"],
      ["interface", "Service"],
      ["method", "Service.run"],
      ["enum", "State"],
      ["method", "State.reset"],
      ["record", "User"],
      ["constructor", "User.User"],
      ["method", "User.value"],
      ["class", "Outer"],
      ["constructor", "Outer.Outer"],
      ["class", "Outer.Inner"],
      ["method", "Outer.Inner.work"],
    ]);
    const record = result.chunks.find((chunk) => chunk.kind === "record");
    expect(record?.content.startsWith("record User(String name)")).toBe(true);
    expect(record?.content).not.toContain("String value()");
  });

  test("csharp symbols carry precise kinds and nested qualification", () => {
    const result = new TreeSitterExtractor().extract(
      "src/Catalog.cs",
      "csharp",
      source(`namespace Demo;

public delegate int Transform(int value);

public enum State { On, Off }

public interface IService
{
    void Run();
    string Name { get; }
}

public record User(string Name);

public struct Point
{
    public Point(int x) => X = x;
    public int X { get; }
}

public class Outer
{
    public Outer() { }

    public void Run()
    {
        void Local() { }
        Local();
    }

    public class Inner
    {
        public void Work() { }
    }
}
`),
    );

    expectSymbols(result, [
      ["type", "Transform"],
      ["enum", "State"],
      ["constant", "State.On"],
      ["interface", "IService"],
      ["method", "IService.Run"],
      ["property", "IService.Name"],
      ["record", "User"],
      ["struct", "Point"],
      ["constructor", "Point.Point"],
      ["property", "Point.X"],
      ["class", "Outer"],
      ["constructor", "Outer.Outer"],
      ["method", "Outer.Run"],
      ["function", "Outer.Run.Local"],
      ["class", "Outer.Inner"],
      ["method", "Outer.Inner.Work"],
    ]);
  });

  test("sql data definitions", () => {
    const result = new TreeSitterExtractor().extract(
      "db/schema.sql",
      "sql",
      source(`CREATE TABLE public.users (
  id BIGSERIAL PRIMARY KEY,
  email TEXT NOT NULL
);

CREATE INDEX idx_users_email ON users (email);

CREATE VIEW active_users AS SELECT id FROM users;

CREATE FUNCTION normalize_email(address TEXT) RETURNS TEXT AS $$
  SELECT lower(address);
$$ LANGUAGE SQL;

CREATE TRIGGER users_audit AFTER INSERT ON users
  FOR EACH ROW EXECUTE FUNCTION audit_users();

CREATE TYPE mood AS ENUM ('sad', 'ok');
`),
    );

    expectSymbols(result, [
      ["table", "users"],
      ["index", "idx_users_email"],
      ["view", "active_users"],
      ["function", "normalize_email"],
      // A trigger names its table and its handler as well as itself; the first
      // object reference is the one that names the trigger.
      ["trigger", "users_audit"],
      ["type", "mood"],
    ]);
  });

  test("gdscript signals and inner classes", () => {
    const result = new TreeSitterExtractor().extract(
      "game/player.gd",
      "gdscript",
      source(
        "class_name Player\nextends CharacterBody2D\n\nsignal health_changed(amount: int)\n\nenum State { IDLE, RUN }\n\nconst MAX_SPEED := 300.0\n\nclass Inventory:\n\tfunc add(item) -> void:\n\t\tpass\n\nfunc _ready() -> void:\n\tpass\n",
      ),
    );

    expectSymbols(result, [
      ["class", "Player"],
      ["signal", "health_changed"],
      ["enum", "State"],
      ["constant", "MAX_SPEED"],
      ["class", "Inventory"],
      ["method", "Inventory.add"],
      ["function", "_ready"],
    ]);
  });

  test.each([
    ["yaml", "compose.yaml", 'version: "3.9"\nservices:\n  web:\n    ports:\n      - "80:80"\n'],
    ["json", "package.json", '{"version": "3.9", "services": {"web": {"ports": ["80:80"]}}}\n'],
  ])("%s indexes only collection-valued keys", (language, file, text) => {
    // Capturing scalar leaves too would turn a large config file into thousands
    // of one-line chunks; the scalars still reach the index as part of the
    // enclosing key's chunk or as module text.
    const result = new TreeSitterExtractor().extract(file, language, source(text));

    expectSymbols(result, [
      ["object", "services"],
      ["object", "services.web"],
      ["array", "services.web.ports"],
    ]);
    expect(result.chunks.some((chunk) => chunk.qualified_symbol === "version")).toBe(false);
  });

  // §5.5: GDShader has no grammar on Windows, and the scanner treats the
  // extension as unsupported there. Checked against the declared gap rather
  // than by attempting a load, so the decision -- not a network round trip --
  // is what decides whether this runs.
  test.skipIf(unavailableLanguages().includes("gdshader"))(
    "gdshader uniforms, functions, structs and constants",
    () => {
      const result = new TreeSitterExtractor().extract(
        "shaders/water.gdshader",
        "gdshader",
        source(
          "shader_type spatial;\n\nuniform vec4 albedo : source_color = vec4(1.0);\nuniform sampler2D noise_tex;\n\nconst float PI2 = 6.28318;\n\nvarying vec3 world_pos;\n\nstruct Ray {\n\tvec3 origin;\n};\n\nfloat wave(float x) {\n\treturn sin(x);\n}\n\nvoid fragment() {\n\tALBEDO = albedo.rgb;\n}\n",
        ),
      );

      expectSymbols(result, [
        // A uniform is the shader's exposed parameter, so it indexes as a property.
        ["property", "albedo"],
        ["property", "noise_tex"],
        ["constant", "PI2"],
        ["struct", "Ray"],
        ["function", "wave"],
        ["function", "fragment"],
      ]);
      // A type hint holds an identifier of its own; it must not become the name.
      expect(result.chunks.some((chunk) => chunk.qualified_symbol === "source_color")).toBe(false);
    },
  );

  test("godot scene nodes and resource ids arrive without quotes", () => {
    const result = new TreeSitterExtractor().extract(
      "scenes/level.tscn",
      "godot_resource",
      source(`[gd_scene load_steps=3 format=3 uid="uid://scene1"]

[ext_resource type="Script" path="res://player.gd" id="1_script"]

[sub_resource type="RectangleShape2D" id="Rect_1"]
size = Vector2(32, 64)

[node name="Player" type="CharacterBody2D"]
script = ExtResource("1_script")

[connection signal="died" from="Player" to="." method="_on_died"]
`),
    );

    expectSymbols(result, [
      ["object", "1_script"],
      ["object", "Rect_1"],
      ["object", "Player"],
    ]);
    // The grammar has no node for the inside of a quoted string, so the name
    // capture arrives with its quotes attached.
    expect(result.chunks.some((chunk) => (chunk.qualified_symbol ?? "").startsWith('"'))).toBe(
      false,
    );
    // A section is named by `name` or by `id` depending on its heading; a type
    // or a connection endpoint is neither.
    const stray = new Set(["CharacterBody2D", "Script", "died", "Player.gd"]);
    expect(
      result.chunks.some(
        (chunk) => chunk.qualified_symbol !== null && stray.has(chunk.qualified_symbol),
      ),
    ).toBe(false);
  });

  test("a section carrying both name and id is named once, by its heading", () => {
    // Matches arrive in source order, not pattern order. Two unanchored patterns
    // would both match such a section and which one won would depend on
    // attribute order in the file, so each pattern is anchored to its heading.
    const result = new TreeSitterExtractor().extract(
      "odd.tscn",
      "godot_resource",
      source('[node name="Player" id="N_1"]\nscript = 1\n'),
    );

    expect(
      result.chunks
        .filter((chunk) => chunk.qualified_symbol !== null)
        .map((chunk) => [chunk.kind, chunk.qualified_symbol]),
    ).toEqual([["object", "Player"]]);
  });

  test("query predicates are applied when matching", () => {
    // The Godot resource query relies on `#eq?`/`#any-of?` filtering matches,
    // and nothing else in this codebase depends on it. Were an upgrade to stop
    // applying them, every attribute in a scene file would become a symbol
    // rather than just its name, so fail here rather than flooding the index.
    const result = new TreeSitterExtractor().extract(
      "scene.tscn",
      "godot_resource",
      source('[node name="Player" type="CharacterBody2D" parent="."]\nscript = 1\n'),
    );

    const names = new Set(
      result.chunks.map((chunk) => chunk.qualified_symbol).filter((name) => name !== null),
    );
    expect([...names]).toEqual(["Player"]);
  });

  test("a quoted name capture is indexed without its quotes", () => {
    const result = new TreeSitterExtractor().extract(
      "config.yaml",
      "yaml",
      source('"quoted key":\n  nested: 1\nplain:\n  nested: 2\n'),
    );

    const symbols = new Set(
      result.chunks.map((chunk) => chunk.qualified_symbol).filter((name) => name !== null),
    );
    expect([...symbols].sort()).toEqual(["plain", "quoted key"]);
  });
});

describe("chunk splitting", () => {
  test("an oversized function splits into bounded parts", () => {
    const body = Array.from({ length: 20 }, (_, index) => `    value_${index} = ${index}`).join(
      "\n",
    );
    const result = new TreeSitterExtractor({
      maxChars: 120,
      maxLines: 6,
      overlapLines: 1,
    }).extract("large.py", "python", source(`def large():\n${body}\n`));

    const parts = result.chunks.filter((chunk) => chunk.symbol === "large");
    expect(parts.length).toBeGreaterThan(1);
    expect(parts.every((chunk) => chunk.kind === "function_part")).toBe(true);
    expect(parts.map((chunk) => chunk.part_index)).toEqual(parts.map((_, index) => index));
    expect(parts.every((chunk) => chunk.start_line <= chunk.end_line)).toBe(true);
  });

  test("a single oversized line splits into bounded chunks", () => {
    const result = new TreeSitterExtractor({ maxChars: 1024 }).extract(
      "payload.py",
      "python",
      source(`payload = '${"x".repeat(10_000)}'\n`),
    );

    expect(result.chunks.length).toBeGreaterThan(1);
    expect(result.chunks.every((chunk) => chunk.content.length <= 1024)).toBe(true);
  });

  test("one oversized line does not split its neighbours per line", () => {
    const body = Array.from({ length: 400 }, (_, index) => `    value_${index} = ${index}`).join(
      "\n",
    );
    const extractor = new TreeSitterExtractor();

    const baseline = extractor.extract("ordinary.py", "python", source(`def big():\n${body}\n`));
    const mixed = extractor.extract(
      "mixed.py",
      "python",
      source(`def big():\n${body}\n    blob = '${"x".repeat(5_000)}'\n`),
    );

    // The long line only adds its own fragments; it must not force every
    // surrounding line onto a chunk of its own.
    expect(mixed.chunks.length).toBeLessThan(baseline.chunks.length + 10);
    expect(mixed.chunks.every((chunk) => chunk.content.length <= extractor.maxChars)).toBe(true);
  });

  test("blank runs around an oversized line produce no empty chunks", () => {
    const lines: string[] = [];
    for (let index = 0; index < 50; index += 1) {
      lines.push(`    value_${index} = ${index}`, "");
    }
    lines.push(`    blob = '${"x".repeat(5_000)}'`);
    const result = new TreeSitterExtractor().extract(
      "spaced.py",
      "python",
      source(`def spaced():\n${lines.join("\n")}\n`),
    );

    expect(result.chunks.length).toBeGreaterThan(0);
    expect(result.chunks.every((chunk) => chunk.content.trim().length > 0)).toBe(true);
  });

  test("oversized line fragments carry contiguous byte ranges", () => {
    const prefix = "value = 1\n";
    const text = `${prefix}blob = '${"x".repeat(5_000)}'\n`;
    const result = new TreeSitterExtractor({ maxChars: 1024 }).extract(
      "offsets.py",
      "python",
      source(text),
    );
    const fragments = result.chunks.filter((chunk) => chunk.start_byte >= prefix.length);

    expect(fragments.length).toBeGreaterThan(1);
    for (let index = 1; index < fragments.length; index += 1) {
      expect((fragments[index - 1] as { end_byte: number }).end_byte).toBe(
        (fragments[index] as { start_byte: number }).start_byte,
      );
    }
    expect((fragments[fragments.length - 1] as { end_byte: number }).end_byte).toBeLessThanOrEqual(
      text.length,
    );
  });
});

describe("qualification and errors", () => {
  test("python syntax errors are reported but valid symbols survive", () => {
    const result = new TreeSitterExtractor().extract(
      "broken.py",
      "python",
      source("def valid():\n    return 1\n\ndef broken(:\n"),
    );

    expect(result.has_errors).toBe(true);
    expect(result.chunks.some((chunk) => chunk.symbol === "valid")).toBe(true);
  });

  test("java syntax errors are reported but valid symbols survive", () => {
    const result = new TreeSitterExtractor().extract(
      "broken.java",
      "java",
      source("class Valid {}\n\nclass Broken { void run( { }\n"),
    );

    expect(result.has_errors).toBe(true);
    expect(result.chunks.some((chunk) => chunk.symbol === "Valid")).toBe(true);
  });

  test("a java declaration-only file creates no module chunk", () => {
    const result = new TreeSitterExtractor().extract(
      "OnlyType.java",
      "java",
      source("class OnlyType { void run() {} }"),
    );

    expect(result.chunks.some((chunk) => chunk.kind === "module")).toBe(false);
  });

  test("identifier normalization splits code and path tokens", () => {
    expect(normalizeIdentifier("HTTPServer_v2/path-name.ts")).toBe("http server v2 path name ts");
  });

  test("java local classes are qualified through the enclosing method", () => {
    const result = new TreeSitterExtractor().extract(
      "A.java",
      "java",
      source(`class A {
    void m() {
        class Local {
            Local() {}

            void run() {}
        }
    }

    void n() {
        class Local {
            void run() {}
        }
    }
}
`),
    );

    expectSymbols(result, [
      ["class", "A.m.Local"],
      ["constructor", "A.m.Local.Local"],
      ["method", "A.m.Local.run"],
      ["class", "A.n.Local"],
      ["method", "A.n.Local.run"],
    ]);
  });

  test("java enum constant bodies qualify their methods", () => {
    const result = new TreeSitterExtractor().extract(
      "E.java",
      "java",
      source("enum E {\n    A(1) {\n        void go() {}\n    },\n    B;\n\n    void go() {}\n}\n"),
    );

    expectSymbols(result, [
      ["constant", "E.A"],
      ["method", "E.A.go"],
      ["method", "E.go"],
    ]);
    expect(result.chunks.some((chunk) => chunk.symbol === "B")).toBe(false);
  });

  test("container chunks stop before nested type declarations", () => {
    const result = new TreeSitterExtractor().extract(
      "Outer.java",
      "java",
      source("class Outer {\n    class Inner {\n        void work() {}\n    }\n}\n"),
    );

    expect(result.chunks.find((chunk) => chunk.qualified_symbol === "Outer")?.content).toBe(
      "class Outer {",
    );
    expect(result.chunks.find((chunk) => chunk.qualified_symbol === "Outer.Inner")?.content).toBe(
      "class Inner {",
    );
  });

  test("python closures are qualified through the enclosing callable", () => {
    const result = new TreeSitterExtractor().extract(
      "mod.py",
      "python",
      source(
        "def outer():\n    def inner():\n        pass\n\n\nclass A:\n    def m(self):\n        def helper():\n            pass\n",
      ),
    );

    expectSymbols(result, [
      ["function", "outer.inner"],
      ["function", "A.m.helper"],
    ]);
  });

  test.each([
    ["python", "mod.py", "def outer():\n    def inner():\n        pass\n"],
    ["javascript", "app.js", "function outer() { function inner() {} }\n"],
    ["java", "E.java", "enum E {\n    A { void go() {} }\n    void go() {}\n}\n"],
    [
      "java",
      "A.java",
      "class A {\n    void m() { class Local {} }\n    void n() { class Local {} }\n}\n",
    ],
  ])("qualified symbols are unique within %s %s", (language, file, text) => {
    const result = new TreeSitterExtractor().extract(file, language, source(text));
    const keys = result.chunks
      .filter((chunk) => chunk.symbol !== null && !chunk.kind.endsWith("_part"))
      .map((chunk) => `${chunk.kind}\u0000${chunk.qualified_symbol}`);

    expect(keys.length).toBe(new Set(keys).size);
  });

  test("java exotic declarations extract surrounding symbols", () => {
    const result = new TreeSitterExtractor().extract(
      "Shapes.java",
      "java",
      source(`sealed interface Shape permits Circle, Square {
}

final class Circle implements Shape {
    static final double PI = 3.14;

    static {
        int ignored = 1;
    }

    {
        int alsoIgnored = 2;
    }

    <T> T pick(T value) {
        return value;
    }

    String describe() {
        String text = """
                multi line
                """;
        java.util.function.Supplier<String> supplier = () -> text;
        return supplier.get();
    }
}

final class Square implements Shape {
}

@Deprecated
class Old {
}
`),
    );

    expect(result.has_errors).toBe(false);
    expectSymbols(result, [
      ["interface", "Shape"],
      ["class", "Circle"],
      ["method", "Circle.pick"],
      ["method", "Circle.describe"],
      ["class", "Square"],
      ["class", "Old"],
    ]);
    expect(
      result.chunks
        .find((chunk) => chunk.qualified_symbol === "Old")
        ?.content.startsWith("@Deprecated"),
    ).toBe(true);
  });
});

describe("packaging invariants", () => {
  test("every scanned language has a grammar and a query", () => {
    // The extension map, the grammar table, and the packaged .scm files are
    // three separate places; adding a language to only one of them is the easy
    // mistake, and it fails per file at index time rather than here.
    const scanned = new Set(Object.values(LANGUAGES));
    // §5.5: one language is deliberately unavailable on Windows.
    const available = new Set(supportedLanguages());

    expect([...available].sort()).toEqual(
      [...scanned].filter((name) => available.has(name)).sort(),
    );
    for (const language of [...scanned].sort()) {
      expect(fs.existsSync(path.join(QUERY_DIRECTORIES.chunks, `${language}.scm`))).toBe(true);
    }
  });

  test("the packaged queries are the ones the Python build ships", () => {
    // During the dual-maintenance window both trees carry the same query pack.
    // A copy that can drift is a copy that will, so byte-identity is asserted
    // rather than assumed; once the Python tree retires, this simply stops
    // having a source to compare against.
    const pythonSource = path.join(
      path.dirname(QUERY_DIRECTORIES.chunks),
      "..",
      "..",
      "..",
      "..",
      "src",
      "code_indexing_mcp",
    );
    const packs = [
      [QUERY_DIRECTORIES.chunks, path.join(pythonSource, "queries")],
      [QUERY_DIRECTORIES.references, path.join(pythonSource, "reference_queries")],
    ] as const;
    if (!fs.existsSync(pythonSource)) return;
    for (const [packaged, python] of packs) {
      for (const name of fs.readdirSync(packaged)) {
        expect(fs.readFileSync(path.join(packaged, name), "utf8")).toBe(
          fs.readFileSync(path.join(python, name), "utf8"),
        );
      }
    }
  });

  test("ChunkKind covers every kind the queries capture", () => {
    const declared = new Set<string>(CHUNK_KINDS);
    const captured = new Set<string>();
    for (const language of new Set(Object.values(LANGUAGES))) {
      const text = fs.readFileSync(path.join(QUERY_DIRECTORIES.chunks, `${language}.scm`), "utf8");
      for (const line of text.split("\n")) {
        if (!line.includes("@definition.")) continue;
        const after = line.split("@definition.")[1] as string;
        captured.add((after.split(/\s/)[0] as string).replace(/\)+$/, ""));
      }
    }

    expect([...captured].filter((kind) => !declared.has(kind))).toEqual([]);
    expect(
      [...captured].filter((kind) => kind !== "module" && !declared.has(`${kind}_part`)),
    ).toEqual([]);
  });

  test("a compiled query is built once per language", () => {
    // The .scm files are package data and grammars are cached in the module.
    // Re-reading and recompiling per file cost 44% of extraction time over 35
    // files, so the cache is load-bearing rather than tidy.
    const reads: string[] = [];
    const originalRead = fs.readFileSync.bind(fs);
    const counting = spyOn(fs, "readFileSync").mockImplementation(((
      ...args: Parameters<typeof fs.readFileSync>
    ) => {
      const file = String(args[0]);
      if (file.endsWith(".scm")) reads.push(file);
      return originalRead(...args);
    }) as never);
    try {
      const extractor = new TreeSitterExtractor();
      for (let index = 0; index < 5; index += 1) {
        extractor.extract("a.py", "python", source("def one():\n    return 1\n"));
      }
      extractor.extract("b.ts", "typescript", source("export const x = 1;\n"));
    } finally {
      counting.mockRestore();
    }

    // Two languages across two query packs (chunks and references, since both
    // are structural languages) -- four files, each read exactly once however
    // many files are extracted.
    expect(reads).toHaveLength(4);
    expect(new Set(reads).size).toBe(4);
  });

  test("Parser.Query is the constructor the extractor compiles through", () => {
    // Guards the one runtime assumption the port makes about node-tree-sitter's
    // shape: the query class hangs off the Parser export.
    expect(typeof Parser.Query).toBe("function");
  });
});

describe("line index", () => {
  test("matches a naive newline count at every offset", () => {
    const bytes = encoder.encode("alpha\nbeta\n\ngamma\r\ndelta");
    const index = new LineIndex(bytes);

    for (let offset = 0; offset <= bytes.length; offset += 1) {
      let newlines = 0;
      for (let position = 0; position < offset; position += 1) {
        if (bytes[position] === 0x0a) newlines += 1;
      }
      expect(index.lineAt(offset)).toBe(newlines + 1);
    }
  });

  test("handles empty and newline-only sources", () => {
    expect(new LineIndex(encoder.encode("")).lineAt(0)).toBe(1);
    expect(new LineIndex(encoder.encode("\n\n\n")).lineAt(3)).toBe(4);
  });
});

describe("language-pack downloads", () => {
  test("a transient download failure is retried", () => {
    // A single 503 from the pack's release download must not fail the run.
    let attempts = 0;
    const value = withDownloadRetry(
      () => {
        attempts += 1;
        if (attempts < PACK_DOWNLOAD_ATTEMPTS) throw new Error("transient outage");
        return "grammar";
      },
      { backoffMs: 0 },
    );

    expect(value).toBe("grammar");
    expect(attempts).toBe(PACK_DOWNLOAD_ATTEMPTS);
  });

  test("a download that keeps failing gives up", () => {
    let attempts = 0;

    expect(() =>
      withDownloadRetry(
        () => {
          attempts += 1;
          throw new Error("down");
        },
        { backoffMs: 0 },
      ),
    ).toThrow("down");
    expect(attempts).toBe(PACK_DOWNLOAD_ATTEMPTS);
  });
});
