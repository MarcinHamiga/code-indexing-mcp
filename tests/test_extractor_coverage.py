"""EXT-2: every meaningful source line must land in at least one chunk."""

from __future__ import annotations

from pathlib import Path

import pytest
from tree_sitter import Node, Parser

from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.models import ExtractedChunk
from code_indexing_mcp.scanner import LANGUAGES

CORPUS = Path(__file__).parent / "fixtures" / "extractor_corpus"
# Lines made only of closing punctuation carry nothing to search for, so a
# container's trailing `}` is allowed to stay unindexed.
_PUNCTUATION_ONLY = set("}])>;, \t\r")


def uncovered_lines(source: bytes, chunks: list[ExtractedChunk]) -> list[str]:
    """Return non-blank, non-punctuation-only source lines no chunk covers."""
    spans = [(chunk.start_byte, chunk.end_byte) for chunk in chunks]
    missing: list[str] = []
    offset = 0
    for raw in source.split(b"\n"):
        text = raw.decode("utf-8", errors="replace")
        stripped = text.strip()
        if stripped and not set(stripped) <= _PUNCTUATION_ONLY:
            position = offset + len(raw) - len(raw.lstrip())
            if not any(start <= position < end for start, end in spans):
                missing.append(stripped)
        offset += len(raw) + 1
    return missing


def _extract(language: str, name: str, source: str) -> tuple[bytes, list[ExtractedChunk]]:
    data = source.encode()
    return data, TreeSitterExtractor().extract(Path(name), language, data).chunks


TYPESCRIPT = """\
@Controller('users')
export class UsersController {
  @Get(':id')
  find(id: string) {
    return id;
  }

  @Get(':id/profile')
  @UseGuards(AuthGuard)
  profile(id: string) {
    return id;
  }

  private readonly retryLimit = 3;

  @Post('bulk-import')
  bulk() {
    return 1;
  }
}
"""

JAVA = """\
public class Service {
    /** First. */
    public void a() {
    }

    private static final int MAX_RETRIES = 3;

    /** Second doc. */
    public void b() {
    }
}
"""

PYTHON = """\
class Client:
    RETRY_LIMIT = 5

    @property
    def first(self):
        return 1

    ENDPOINT = "http://x"

    def second(self):
        return 2

    TRAILING = 9
"""

YAML = """\
services:
  web:
    image: nginx
    ports:
      - "80:80"
    restart: always
"""

JSON = """\
{
  "scripts": {
    "build": "tsc"
  },
  "deploy": "make deploy"
}
"""

PRISMA = """\
model User {
  /// the id
  id Int @id
  /// the name
  name String
  /// the email
  email String
}
"""

VUE = """\
<template>
  <div>hi</div>
  <style>
    .a { color: red; }
  </style>
  <script>
    export default { name: 'x' }
  </script>
</template>
"""


@pytest.mark.parametrize(
    ("language", "name", "source"),
    [
        ("typescript", "users.ts", TYPESCRIPT),
        ("java", "Service.java", JAVA),
        ("python", "client.py", PYTHON),
        ("yaml", "compose.yaml", YAML),
        ("json", "package.json", JSON),
        ("prisma", "schema.prisma", PRISMA),
        ("vue", "App.vue", VUE),
    ],
)
def test_container_chunks_cover_text_after_nested_definitions(
    language: str, name: str, source: str
) -> None:
    data, chunks = _extract(language, name, source)

    assert uncovered_lines(data, chunks) == []


def test_python_class_chunk_does_not_end_with_the_next_methods_decorator() -> None:
    _, chunks = _extract("python", "client.py", PYTHON)

    head = next(chunk for chunk in chunks if chunk.kind == "class")

    method = next(chunk for chunk in chunks if chunk.qualified_symbol == "Client.first")
    assert not head.content.rstrip().endswith("@property")
    assert head.end_byte <= method.start_byte
    assert (head.start_line, head.end_line) == (1, 2)
    assert method.content.startswith("@property")


def test_gap_chunks_carry_the_container_symbol() -> None:
    _, chunks = _extract("python", "client.py", PYTHON)

    gaps = [c for c in chunks if c.kind == "class_part" and "ENDPOINT" in c.content]

    assert [(c.symbol, c.qualified_symbol) for c in gaps] == [("Client", "Client")]


def test_chunk_ids_stay_unique_for_gap_chunks() -> None:
    _, chunks = _extract("python", "client.py", PYTHON)

    keys = [(c.kind, c.qualified_symbol, c.start_byte, c.end_byte, c.part_index) for c in chunks]

    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("path", sorted(CORPUS.iterdir()), ids=lambda p: p.name)
def test_every_fixture_line_is_covered(path: Path) -> None:
    data = path.read_bytes()
    chunks = TreeSitterExtractor().extract(Path(path.name), LANGUAGES[path.suffix.lower()], data)

    assert uncovered_lines(data, chunks.chunks) == []


def test_split_parts_count_lines_by_newline_only() -> None:
    """EXT-9: a form feed or lone CR is not a line break for line numbers."""
    body = "".join(f"x{i} = {i}\n" for i in range(450))
    source = "# page\x0cbreak\n" + body
    extractor = TreeSitterExtractor(max_lines=200, overlap_lines=20)

    result = extractor.extract(Path("a.py"), "python", source.encode())

    raw = source.encode()
    for chunk in result.chunks:
        expected_line = raw.count(b"\n", 0, chunk.start_byte) + 1
        assert chunk.start_line == expected_line
        assert chunk.end_line == chunk.start_line + chunk.content.count("\n")


def test_unicode_whitespace_gaps_emit_no_empty_module_chunk() -> None:
    """EXT-11: a gap of only NBSP / U+3000 must not become an empty chunk."""
    source = "def g(): pass\n\N{NO-BREAK SPACE}\n\N{IDEOGRAPHIC SPACE}\ndef f(): pass\n"

    _, chunks = _extract("python", "a.py", source)

    assert all(chunk.content.strip() for chunk in chunks)
    assert [chunk.kind for chunk in chunks] == ["function", "function"]


@pytest.mark.parametrize(
    "member",
    ['pame => "outer";', 'public pame => "outer";', 'public string => "o";'],
)
def test_missing_name_nodes_yield_no_empty_symbol_or_reference(member: str) -> None:
    """EXT-11: mid-edit C# can leave a zero-width MISSING name node."""
    source = f"class Outer {{\n  {member}\n}}\n".encode()

    result = TreeSitterExtractor().extract(Path("a.cs"), "csharp", source)

    assert all(chunk.symbol for chunk in result.chunks if chunk.kind != "module")
    assert all(not (chunk.qualified_symbol or "").endswith(".") for chunk in result.chunks)
    assert all(ref.end_byte > ref.start_byte and ref.target_name for ref in result.references)


def test_svelte_5_syntax_does_not_flag_parse_errors() -> None:
    """EXT-10: the packaged grammar predates `{@render}` / `{@attach}`."""
    source = b"<script>\n  let { children } = $props();\n</script>\n{@render children?.()}\n"

    result = TreeSitterExtractor().extract(Path("a.svelte"), "svelte", source)

    assert result.has_errors is False
    assert any(chunk.symbol == "script" for chunk in result.chunks)


def test_deeply_nested_type_expressions_do_not_overflow_the_stack() -> None:
    """EXT-8: deep type nesting used to raise RecursionError and fail the whole file."""
    extractor = TreeSitterExtractor()
    members = " | ".join(f"'k{i}'" for i in range(1100))
    ts = f"function icon(name: {members}) {{}}\n"
    assert extractor.extract(Path("a.ts"), "typescript", ts.encode()).chunks

    # Rust and SQL are exercised at the descent helpers: whole-file extraction of
    # inputs this deep is dominated by unrelated per-reference ancestor climbs.
    rust = ("fn f(x: " + "Vec<" * 900 + "Foo" + ">" * 900 + ") {}\n").encode()
    tree = Parser(extractor._languages["rust"]).parse(rust)
    parameter = _first_node(tree.root_node, "type_identifier", b"Vec")
    parent = parameter.parent
    assert parent is not None
    names = TreeSitterExtractor._rust_descend_type_names(parent)
    assert [name.text for name in names][-1] == b"Foo"

    sql = ("SELECT COALESCE(" + " || ".join(f"c{i}" for i in range(1100)) + ") FROM t;\n").encode()
    tree = Parser(extractor._languages["sql"]).parse(sql)
    invocation = _first_node_of_type(tree.root_node, "invocation")
    argument = invocation.child_by_field_name("parameter")
    assert argument is not None

    # Must complete without RecursionError; the `field` operands are owned by
    # another handler, so no rows are expected.
    extractor._sql_emit_column_uses(argument, sql, lambda *args, **kwargs: None)


def _first_node(root: Node, node_type: str, text: bytes | None = None) -> Node:
    stack = [root]
    while stack:
        current = stack.pop()
        if current.type == node_type and (text is None or current.text == text):
            return current
        stack.extend(reversed(current.children))
    raise AssertionError(node_type)


def _first_node_of_type(root: Node, node_type: str) -> Node:
    return _first_node(root, node_type)
