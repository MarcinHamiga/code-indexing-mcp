"""Tree-sitter based symbol and module chunk extraction."""

from __future__ import annotations

import re
import threading
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Final

import tree_sitter_c_sharp
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_json
import tree_sitter_python
import tree_sitter_sql
import tree_sitter_typescript
import tree_sitter_yaml
from tree_sitter import Language, Node, Parser, Query, QueryCursor
from tree_sitter_language_pack import get_language

from .models import ExtractedChunk, ExtractionResult

_CAMEL_BOUNDARY_1: Final = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_BOUNDARY_2: Final = re.compile(r"([A-Z]+)([A-Z][a-z])")
_NON_WORD: Final = re.compile(r"[^A-Za-z0-9]+")
_CONTAINER_KINDS: Final = frozenset(
    {
        "annotation",
        "array",
        "class",
        "constant",
        "enum",
        "interface",
        "object",
        "record",
        "struct",
    }
)
_CALLABLE_KINDS: Final = frozenset({"constructor", "function", "method"})
_QUOTE_CHARACTERS: Final = ("'", '"')


def _capture_name(source: bytes, node: Node) -> str:
    """Return the symbol text of a ``@name`` capture, without surrounding quotes.

    Most grammars name a definition with an identifier token and this is a plain
    decode. A few have no node for the inside of a quoted name -- Godot's
    resource format hands back `"Player"` including the quotes, and a quoted YAML
    key does the same -- which would otherwise index the quotes as part of the
    symbol. Only a matched leading/trailing pair is stripped, so an identifier
    that merely contains a quote is left alone.
    """
    name = source[node.start_byte : node.end_byte].decode("utf-8")
    if len(name) >= 2 and name[0] == name[-1] and name[0] in _QUOTE_CHARACTERS:
        return name[1:-1]
    return name


def normalize_identifier(value: str) -> str:
    value = _CAMEL_BOUNDARY_2.sub(r"\1 \2", value)
    value = _CAMEL_BOUNDARY_1.sub(r"\1 \2", value)
    return " ".join(_NON_WORD.sub(" ", value).lower().split())


def _languages() -> dict[str, Language]:
    return {
        "python": Language(tree_sitter_python.language()),
        "java": Language(tree_sitter_java.language()),
        "javascript": Language(tree_sitter_javascript.language()),
        "typescript": Language(tree_sitter_typescript.language_typescript()),
        "tsx": Language(tree_sitter_typescript.language_tsx()),
        "csharp": Language(tree_sitter_c_sharp.language()),
        "sql": Language(tree_sitter_sql.language()),
        # No standalone GDScript grammar is published to PyPI; the language pack
        # is the only packaged source. It already returns a Language, not a
        # PyCapsule, so it is not wrapped like the others. The two sibling Godot
        # formats come from the same pack for the same reason.
        "gdscript": get_language("gdscript"),
        "gdshader": get_language("gdshader"),
        "godot_resource": get_language("godot_resource"),
        "yaml": Language(tree_sitter_yaml.language()),
        "json": Language(tree_sitter_json.language()),
    }


@dataclass(frozen=True)
class _Definition:
    node: Node
    kind: str
    name: str


@dataclass(frozen=True)
class _DefinitionIndex:
    """Per-file lookups the definition walk needs, built once instead of per definition.

    ``_has_definition_ancestor``, ``_symbol_context``, and ``_content_range`` each
    rebuilt a whole-file dict or set on every call, making extraction quadratic in
    definition count: a 699 KB generated file with 16,384 definitions spent 31 s
    here against 8 ms of parsing.
    """

    definitions: list[_Definition]
    by_node_id: dict[int, _Definition]
    starts: list[int]

    @classmethod
    def build(cls, definitions: list[_Definition]) -> _DefinitionIndex:
        return cls(
            definitions=definitions,
            by_node_id={definition.node.id: definition for definition in definitions},
            # _definitions returns rows sorted by (start_byte, -end_byte), so this
            # is ascending and safe to bisect.
            starts=[definition.node.start_byte for definition in definitions],
        )


class _LineIndex:
    """Byte offsets of every newline in one file, for O(log n) line lookups.

    ``source[:start].count(b"\\n")`` is O(file size) per chunk, so computing a start
    line per chunk was O(chunks x file size). ``bytes.find`` scans at C speed, so
    building this costs one pass and one append per line.
    """

    __slots__ = ("_newlines",)

    def __init__(self, source: bytes) -> None:
        newlines: list[int] = []
        position = source.find(b"\n")
        while position != -1:
            newlines.append(position)
            position = source.find(b"\n", position + 1)
        self._newlines = newlines

    def line_at(self, byte_offset: int) -> int:
        """Return the 1-based line number containing *byte_offset*."""
        return bisect_left(self._newlines, byte_offset) + 1


class TreeSitterExtractor:
    def __init__(
        self,
        *,
        max_chars: int = 4_096,
        max_lines: int = 200,
        overlap_lines: int = 20,
    ) -> None:
        self.max_chars = max_chars
        self.max_lines = max_lines
        self.overlap_lines = min(overlap_lines, max(0, max_lines - 1))
        self._languages = _languages()
        self._queries: dict[str, Query] = {}
        # Indexer holds one extractor and the daemon serves each client on its own
        # thread, so the lazy compile must not build two queries concurrently. Same
        # double-checked shape as FastEmbedder's model load.
        self._queries_lock = threading.Lock()

    def _query(self, language_name: str) -> Query:
        """Return the compiled query for *language_name*, compiling once per process.

        The .scm files are package data and never change at runtime, but the previous
        code re-read and recompiled one per extracted file, which measured at 44% of
        extraction time across a 35-file pass.
        """
        cached = self._queries.get(language_name)
        if cached is not None:
            return cached
        with self._queries_lock:
            cached = self._queries.get(language_name)
            if cached is not None:
                return cached
            text = files("incode_mcp.queries").joinpath(f"{language_name}.scm").read_text()
            compiled = Query(self._languages[language_name], text)
            self._queries[language_name] = compiled
            return compiled

    def extract(self, path: Path, language: str, source: bytes) -> ExtractionResult:
        language_impl = self._languages[language]
        normalized_source = source.decode("utf-8-sig").encode("utf-8")
        tree = Parser(language_impl).parse(normalized_source)
        definitions = self._definitions(language, tree.root_node, normalized_source)
        index = _DefinitionIndex.build(definitions)
        line_index = _LineIndex(normalized_source)
        chunks: list[ExtractedChunk] = []
        covered: list[tuple[int, int]] = []

        for definition in definitions:
            outer = self._outer_node(definition.node)
            if not self._has_definition_ancestor(definition.node, index):
                covered.append((outer.start_byte, outer.end_byte))
            kind, parent, qualified = self._symbol_context(definition, index)
            start, end = self._content_range(outer, definition.node, kind, index)
            chunks.extend(
                self._chunks_for_range(
                    path=path,
                    language=language,
                    kind=kind,
                    symbol=definition.name,
                    qualified=qualified,
                    parent=parent,
                    source=normalized_source,
                    start=start,
                    end=end,
                    line_index=line_index,
                )
            )

        chunks.extend(self._module_chunks(path, language, normalized_source, covered, line_index))
        chunks.sort(key=lambda chunk: (chunk.start_byte, chunk.end_byte, chunk.kind))
        return ExtractionResult(chunks=chunks, has_errors=tree.root_node.has_error)

    def _definitions(self, language_name: str, root: Node, source: bytes) -> list[_Definition]:
        matches = QueryCursor(self._query(language_name)).matches(root)
        found: dict[tuple[int, int, str], _Definition] = {}
        for _, captures in matches:
            name_nodes = captures.get("name", [])
            if not name_nodes:
                continue
            name = _capture_name(source, name_nodes[0])
            for capture, nodes in captures.items():
                if not capture.startswith("definition."):
                    continue
                kind = capture.removeprefix("definition.")
                node = nodes[0]
                found[(node.start_byte, node.end_byte, kind)] = _Definition(node, kind, name)
        return sorted(found.values(), key=lambda item: (item.node.start_byte, -item.node.end_byte))

    @staticmethod
    def _outer_node(node: Node) -> Node:
        outer = node
        if node.parent is not None and node.parent.type == "decorated_definition":
            outer = node.parent
        if outer.parent is not None and outer.parent.type in {
            "export_statement",
            "lexical_declaration",
        }:
            outer = outer.parent
        if outer.parent is not None and outer.parent.type == "export_statement":
            outer = outer.parent
        return outer

    @staticmethod
    def _has_definition_ancestor(node: Node, index: _DefinitionIndex) -> bool:
        parent = node.parent
        while parent is not None:
            if parent.id in index.by_node_id:
                return True
            parent = parent.parent
        return False

    @staticmethod
    def _symbol_context(
        definition: _Definition, index: _DefinitionIndex
    ) -> tuple[str, str | None, str]:
        chain: list[_Definition] = []
        parent = definition.node.parent
        while parent is not None:
            candidate = index.by_node_id.get(parent.id)
            if candidate is not None:
                chain.append(candidate)
            parent = parent.parent
        chain.reverse()
        if any(item.kind in _CALLABLE_KINDS for item in chain):
            scope = chain
        else:
            scope = [item for item in chain if item.kind in _CONTAINER_KINDS]
        parent_name = ".".join(item.name for item in scope) or None
        qualified = f"{parent_name}.{definition.name}" if parent_name else definition.name
        kind = definition.kind
        if scope and scope[-1].kind in _CONTAINER_KINDS and kind == "function":
            kind = "method"
        return kind, parent_name, qualified

    @staticmethod
    def _content_range(
        outer: Node, node: Node, kind: str, index: _DefinitionIndex
    ) -> tuple[int, int]:
        if kind not in _CONTAINER_KINDS:
            return outer.start_byte, outer.end_byte
        # The old code scanned every definition to take the minimum qualifying start.
        # Definitions are start-ascending, so the first qualifying one after
        # outer.start_byte *is* that minimum. A definition starting at or after
        # outer.end_byte cannot end inside outer, which bounds the scan.
        position = bisect_right(index.starts, outer.start_byte)
        while position < len(index.definitions):
            candidate = index.definitions[position].node
            if candidate.start_byte >= outer.end_byte:
                break
            if candidate.id != node.id and candidate.end_byte <= outer.end_byte:
                return outer.start_byte, candidate.start_byte
            position += 1
        return outer.start_byte, outer.end_byte

    def _chunks_for_range(
        self,
        *,
        path: Path,
        language: str,
        kind: str,
        symbol: str | None,
        qualified: str | None,
        parent: str | None,
        source: bytes,
        start: int,
        end: int,
        line_index: _LineIndex,
    ) -> list[ExtractedChunk]:
        content = source[start:end].decode("utf-8").rstrip()
        start_line = line_index.line_at(start)
        if len(content) <= self.max_chars and content.count("\n") + 1 <= self.max_lines:
            return [
                self._make_chunk(
                    path,
                    language,
                    kind,
                    symbol,
                    qualified,
                    parent,
                    start,
                    end,
                    start_line,
                    content,
                    0,
                )
            ]

        lines = content.splitlines(keepends=True)
        # Cumulative UTF-8 byte offset of each line, so chunk offsets stay linear
        # to compute instead of re-encoding every preceding line per part.
        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line.encode("utf-8")))
        part_kind = f"{kind}_part" if kind != "module" else "module"
        chunks: list[ExtractedChunk] = []
        cursor = 0
        part = 0
        while cursor < len(lines):
            part_lines: list[str] = []
            char_count = 0
            end_cursor = cursor
            while end_cursor < len(lines) and len(part_lines) < self.max_lines:
                line = lines[end_cursor]
                if char_count + len(line) > self.max_chars:
                    # With no lines accumulated this means the line is oversized
                    # on its own, and the fragment path below handles it.
                    break
                part_lines.append(line)
                char_count += len(line)
                end_cursor += 1
            if not part_lines:
                # This single line is wider than max_chars on its own. Split just
                # that line into bounded fragments; the surrounding lines keep
                # using the ordinary line windows below.
                for offset, fragment in self._line_fragments(lines[cursor]):
                    fragment_content = fragment.rstrip()
                    if not fragment_content:
                        continue
                    byte_start = start + line_offsets[cursor] + offset
                    chunks.append(
                        self._make_chunk(
                            path,
                            language,
                            part_kind,
                            symbol,
                            qualified,
                            parent,
                            byte_start,
                            byte_start + len(fragment.encode("utf-8")),
                            start_line + cursor,
                            fragment_content,
                            part,
                        )
                    )
                    part += 1
                cursor += 1
                continue
            part_content = "".join(part_lines).rstrip()
            if part_content:
                byte_start = start + line_offsets[cursor]
                byte_end = byte_start + len("".join(part_lines).encode("utf-8"))
                chunks.append(
                    self._make_chunk(
                        path,
                        language,
                        part_kind,
                        symbol,
                        qualified,
                        parent,
                        byte_start,
                        byte_end,
                        start_line + cursor,
                        part_content,
                        part,
                    )
                )
                part += 1
            if end_cursor >= len(lines):
                break
            if len(lines[end_cursor]) > self.max_chars:
                # The window stopped because the next line is oversized, not
                # because it filled up. Overlapping back into it would emit a
                # near-duplicate window per line until the cursor crawls there.
                cursor = end_cursor
            else:
                cursor = max(cursor + 1, end_cursor - self.overlap_lines)
        return chunks

    def _line_fragments(self, line: str) -> Iterator[tuple[int, str]]:
        """Yield (byte offset within *line*, fragment) for an oversized line."""
        cursor = 0
        byte_offset = 0
        while cursor < len(line):
            fragment = line[cursor : cursor + self.max_chars]
            yield byte_offset, fragment
            cursor += len(fragment)
            byte_offset += len(fragment.encode("utf-8"))

    def _module_chunks(
        self,
        path: Path,
        language: str,
        source: bytes,
        covered: list[tuple[int, int]],
        line_index: _LineIndex,
    ) -> list[ExtractedChunk]:
        chunks: list[ExtractedChunk] = []
        cursor = 0
        for start, end in sorted(covered):
            if cursor < start and source[cursor:start].strip():
                chunks.extend(
                    self._chunks_for_range(
                        path=path,
                        language=language,
                        kind="module",
                        symbol=None,
                        qualified=None,
                        parent=None,
                        source=source,
                        start=cursor,
                        end=start,
                        line_index=line_index,
                    )
                )
            cursor = max(cursor, end)
        if cursor < len(source) and source[cursor:].strip():
            chunks.extend(
                self._chunks_for_range(
                    path=path,
                    language=language,
                    kind="module",
                    symbol=None,
                    qualified=None,
                    parent=None,
                    source=source,
                    start=cursor,
                    end=len(source),
                    line_index=line_index,
                )
            )
        return chunks

    @staticmethod
    def _make_chunk(
        path: Path,
        language: str,
        kind: str,
        symbol: str | None,
        qualified: str | None,
        parent: str | None,
        start_byte: int,
        end_byte: int,
        start_line: int,
        content: str,
        part_index: int,
    ) -> ExtractedChunk:
        context = [f"language: {language}", f"path: {path.as_posix()}", f"kind: {kind}"]
        if qualified:
            context.append(f"symbol: {qualified}")
        prefix = "\n".join(context)
        embedding_text = f"{prefix}\n{content}"
        normalized = normalize_identifier(
            " ".join(filter(None, [path.as_posix(), qualified or "", symbol or ""]))
        )
        return ExtractedChunk(
            kind=kind,
            symbol=symbol,
            qualified_symbol=qualified,
            parent_symbol=parent,
            start_byte=start_byte,
            end_byte=end_byte,
            start_line=start_line,
            end_line=start_line + content.count("\n"),
            content=content,
            embedding_text=embedding_text,
            search_text=f"{embedding_text}\n{normalized}",
            part_index=part_index,
            embedding_prefix=prefix,
            search_suffix=normalized,
        )
