"""Tree-sitter based symbol and module chunk extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Final

import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser, Query, QueryCursor

from .models import ExtractedChunk, ExtractionResult

_CAMEL_BOUNDARY_1: Final = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_BOUNDARY_2: Final = re.compile(r"([A-Z]+)([A-Z][a-z])")
_NON_WORD: Final = re.compile(r"[^A-Za-z0-9]+")
_CONTAINER_KINDS: Final = frozenset({"annotation", "class", "enum", "interface", "record"})
_CALLABLE_KINDS: Final = frozenset({"constructor", "function", "method"})


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
    }


@dataclass(frozen=True)
class _Definition:
    node: Node
    kind: str
    name: str


class TreeSitterExtractor:
    def __init__(
        self,
        *,
        max_chars: int = 12_000,
        max_lines: int = 200,
        overlap_lines: int = 20,
    ) -> None:
        self.max_chars = max_chars
        self.max_lines = max_lines
        self.overlap_lines = min(overlap_lines, max(0, max_lines - 1))
        self._languages = _languages()

    def extract(self, path: Path, language: str, source: bytes) -> ExtractionResult:
        language_impl = self._languages[language]
        normalized_source = source.decode("utf-8-sig").encode("utf-8")
        tree = Parser(language_impl).parse(normalized_source)
        definitions = self._definitions(language, language_impl, tree.root_node, normalized_source)
        chunks: list[ExtractedChunk] = []
        covered: list[tuple[int, int]] = []

        for definition in definitions:
            outer = self._outer_node(definition.node)
            if not self._has_definition_ancestor(definition.node, definitions):
                covered.append((outer.start_byte, outer.end_byte))
            kind, parent, qualified = self._symbol_context(definition, definitions)
            start, end = self._content_range(outer, kind, definitions)
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
                )
            )

        chunks.extend(self._module_chunks(path, language, normalized_source, covered))
        chunks.sort(key=lambda chunk: (chunk.start_byte, chunk.end_byte, chunk.kind))
        return ExtractionResult(chunks=chunks, has_errors=tree.root_node.has_error)

    @staticmethod
    def _definitions(
        language_name: str, language: Language, root: Node, source: bytes
    ) -> list[_Definition]:
        query_text = files("incode_mcp.queries").joinpath(f"{language_name}.scm").read_text()
        matches = QueryCursor(Query(language, query_text)).matches(root)
        found: dict[tuple[int, int, str], _Definition] = {}
        for _, captures in matches:
            name_nodes = captures.get("name", [])
            if not name_nodes:
                continue
            name_node = name_nodes[0]
            name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")
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
    def _has_definition_ancestor(node: Node, definitions: list[_Definition]) -> bool:
        definition_nodes = [definition.node for definition in definitions]
        parent = node.parent
        while parent is not None:
            if any(parent == definition_node for definition_node in definition_nodes):
                return True
            parent = parent.parent
        return False

    @staticmethod
    def _symbol_context(
        definition: _Definition, definitions: list[_Definition]
    ) -> tuple[str, str | None, str]:
        containers: list[_Definition] = []
        parent = definition.node.parent
        while parent is not None:
            for candidate in definitions:
                if candidate.kind in _CONTAINER_KINDS and candidate.node == parent:
                    containers.append(candidate)
                    break
            parent = parent.parent
        containers.reverse()
        parent_name = ".".join(item.name for item in containers) or None
        qualified = f"{parent_name}.{definition.name}" if parent_name else definition.name
        kind = definition.kind
        if parent_name and kind == "function":
            kind = "method"
        return kind, parent_name, qualified

    @staticmethod
    def _content_range(outer: Node, kind: str, definitions: list[_Definition]) -> tuple[int, int]:
        if kind not in _CONTAINER_KINDS:
            return outer.start_byte, outer.end_byte
        nested_starts = [
            item.node.start_byte
            for item in definitions
            if item.node.start_byte > outer.start_byte
            and item.node.end_byte <= outer.end_byte
            and item.kind in _CALLABLE_KINDS
        ]
        end = min(nested_starts) if nested_starts else outer.end_byte
        return outer.start_byte, end

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
    ) -> list[ExtractedChunk]:
        content = source[start:end].decode("utf-8").rstrip()
        start_line = source[:start].count(b"\n") + 1
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
        chunks: list[ExtractedChunk] = []
        cursor = 0
        part = 0
        while cursor < len(lines):
            part_lines: list[str] = []
            char_count = 0
            end_cursor = cursor
            while end_cursor < len(lines) and len(part_lines) < self.max_lines:
                line = lines[end_cursor]
                if part_lines and char_count + len(line) > self.max_chars:
                    break
                part_lines.append(line)
                char_count += len(line)
                end_cursor += 1
            if not part_lines:
                part_lines = [lines[cursor][: self.max_chars]]
                end_cursor = cursor + 1
            part_content = "".join(part_lines).rstrip()
            byte_start = start + len("".join(lines[:cursor]).encode("utf-8"))
            byte_end = byte_start + len("".join(part_lines).encode("utf-8"))
            part_kind = f"{kind}_part" if kind != "module" else "module"
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
            if end_cursor >= len(lines):
                break
            cursor = max(cursor + 1, end_cursor - self.overlap_lines)
            part += 1
        return chunks

    def _module_chunks(
        self, path: Path, language: str, source: bytes, covered: list[tuple[int, int]]
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
        embedding_text = "\n".join([*context, content])
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
        )
