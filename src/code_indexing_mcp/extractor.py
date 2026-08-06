"""Tree-sitter based symbol and module chunk extraction."""

from __future__ import annotations

import re
import threading
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterator
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
from tree_sitter import Query as StructuralQuery
from tree_sitter_language_pack import get_language

from .models import (
    CallShape,
    ExtractedChunk,
    ExtractedDeclarationShape,
    ExtractedReference,
    ExtractionResult,
    ParameterKind,
    ParameterShape,
    ReferenceKind,
)

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
_STRUCTURAL_LANGUAGES: Final = frozenset({"python", "javascript", "typescript", "tsx"})
_ReferenceAdder = Callable[..., None]


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
        self._structural_queries: dict[str, Query] = {}
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
            text = files("code_indexing_mcp.queries").joinpath(f"{language_name}.scm").read_text()
            compiled = Query(self._languages[language_name], text)
            self._queries[language_name] = compiled
            return compiled

    def _structural_query(self, language_name: str) -> Query:
        """Return the cached structural query for one supported source grammar."""
        cached = self._structural_queries.get(language_name)
        if cached is not None:
            return cached
        with self._queries_lock:
            cached = self._structural_queries.get(language_name)
            if cached is not None:
                return cached
            text = (
                files("code_indexing_mcp.reference_queries")
                .joinpath(f"{language_name}.scm")
                .read_text()
            )
            compiled = StructuralQuery(self._languages[language_name], text)
            self._structural_queries[language_name] = compiled
            return compiled

    def extract(self, path: Path, language: str, source: bytes) -> ExtractionResult:
        language_impl = self._languages[language]
        normalized_source = source.decode("utf-8-sig").encode("utf-8")
        tree = Parser(language_impl).parse(normalized_source)
        definitions = self._definitions(language, tree.root_node, normalized_source)
        index = _DefinitionIndex.build(definitions)
        line_index = _LineIndex(normalized_source)
        references: list[ExtractedReference] = []
        declarations: list[ExtractedDeclarationShape] = []
        if language in _STRUCTURAL_LANGUAGES:
            references, declarations = self._structural_records(
                language, tree.root_node, normalized_source, index, line_index
            )
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
        return ExtractionResult(
            chunks=chunks,
            references=references,
            declarations=declarations,
            has_errors=tree.root_node.has_error,
        )

    def _structural_records(
        self,
        language: str,
        root: Node,
        source: bytes,
        index: _DefinitionIndex,
        line_index: _LineIndex,
    ) -> tuple[list[ExtractedReference], list[ExtractedDeclarationShape]]:
        """Extract syntax facts using the already parsed tree and definition index."""
        matches = QueryCursor(self._structural_query(language)).matches(root)
        parameter_nodes: dict[int, Node] = {}
        for _, captures in matches:
            for parameter_node in captures.get("declaration.parameters", []):
                owner = parameter_node.parent
                while owner is not None and owner.id not in index.by_node_id:
                    owner = owner.parent
                if owner is not None:
                    parameter_nodes[owner.id] = parameter_node
        declarations = self._declaration_shapes(language, index, line_index, parameter_nodes)
        declaration_by_node = {
            definition.node.id: declaration
            for definition, declaration in zip(index.definitions, declarations, strict=True)
        }
        references: list[ExtractedReference] = []
        seen: set[tuple[str, int, int]] = set()

        def add(
            kind: ReferenceKind,
            node: Node,
            *,
            written_name: str | None = None,
            target_name: str,
            module_path: str | None = None,
            imported_name: str | None = None,
            alias: str | None = None,
            receiver_text: str | None = None,
            call_shape: CallShape | None = None,
        ) -> None:
            key = (kind, node.start_byte, node.end_byte)
            if key in seen:
                return
            seen.add(key)
            name = written_name or _capture_name(source, node)
            references.append(
                ExtractedReference(
                    kind=kind,
                    written_name=name,
                    target_name=target_name,
                    source_qualified_symbol=self._enclosing_symbol(node, declaration_by_node),
                    module_path=module_path,
                    imported_name=imported_name,
                    alias=alias,
                    receiver_text=receiver_text,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    start_line=line_index.line_at(node.start_byte),
                    end_line=line_index.line_at(max(node.start_byte, node.end_byte - 1)),
                    call_shape=call_shape,
                )
            )

        for _, captures in matches:
            for capture, nodes in captures.items():
                if not capture.startswith("reference."):
                    continue
                for node in nodes:
                    if capture == "reference.identifier":
                        self._identifier_record(language, node, source, add)
                    elif language == "python":
                        self._python_records(node, source, add)
                    else:
                        self._javascript_records(node, source, add)
        references.sort(key=lambda item: (item.start_byte, item.end_byte, item.kind))
        return references, declarations

    @staticmethod
    def _identifier_record(
        language: str, node: Node, source: bytes, add_reference: _ReferenceAdder
    ) -> None:
        """Record identifier values while excluding bindings and richer structural uses."""

        def contains(outer: Node | None) -> bool:
            return bool(
                outer is not None
                and outer.start_byte <= node.start_byte
                and node.end_byte <= outer.end_byte
            )

        current = node
        while (parent := current.parent) is not None:
            if parent.type in {
                "import_statement",
                "import_from_statement",
                "export_clause",
                "decorator",
                "type",
                "type_annotation",
                "generic_type",
                "class_heritage",
                "extends_type_clause",
            }:
                return
            if parent.type in {"parameters", "formal_parameters", "lambda_parameters"}:
                parameter = node
                while parameter.parent is not None and parameter.parent != parent:
                    parameter = parameter.parent
                    if contains(parameter.child_by_field_name("value")):
                        break
                else:
                    return
            excluded_fields: tuple[str, ...] = ()
            if parent.type in {
                "function_definition",
                "function_expression",
                "generator_function_declaration",
                "generator_function",
                "class_definition",
                "function_declaration",
                "class_declaration",
                "method_definition",
                "variable_declarator",
            }:
                excluded_fields = ("name",)
            elif parent.type in {
                "assignment",
                "assignment_expression",
                "augmented_assignment",
                "named_expression",
                "for_statement",
                "for_in_clause",
            }:
                excluded_fields = ("left", "name")
            elif parent.type in {"arrow_function", "lambda"}:
                excluded_fields = ("parameter", "parameters")
            elif parent.type in {"attribute", "member_expression"}:
                excluded_fields = ("attribute", "property")
            elif parent.type in {"call", "call_expression", "new_expression"}:
                excluded_fields = ("function", "constructor")
            elif parent.type == "keyword_argument":
                excluded_fields = ("name",)
            elif parent.type in {"as_pattern", "catch_clause"}:
                excluded_fields = ("alias", "parameter")
            elif parent.type == "export_statement":
                excluded_fields = ("value",)
            elif language != "python" and parent.type in {"pair", "pair_pattern"}:
                excluded_fields = ("key",)
            if any(contains(parent.child_by_field_name(field)) for field in excluded_fields):
                return
            current = parent

        name = _capture_name(source, node)
        add_reference("read", node, target_name=name, written_name=name)

    @staticmethod
    def _enclosing_symbol(
        node: Node, declarations: dict[int, ExtractedDeclarationShape]
    ) -> str | None:
        current: Node | None = node
        while current is not None:
            declaration = declarations.get(current.id)
            if declaration is not None:
                return declaration.qualified_symbol
            if current.type == "decorated_definition":
                child = current.child_by_field_name("definition")
                if child is not None and (declaration := declarations.get(child.id)) is not None:
                    return declaration.qualified_symbol
            current = current.parent
        return None

    def _declaration_shapes(
        self,
        language: str,
        index: _DefinitionIndex,
        line_index: _LineIndex,
        parameter_nodes: dict[int, Node],
    ) -> list[ExtractedDeclarationShape]:
        rows: list[ExtractedDeclarationShape] = []
        for definition in index.definitions:
            kind, _, qualified = self._symbol_context(definition, index)
            rows.append(
                ExtractedDeclarationShape(
                    symbol=definition.name,
                    qualified_symbol=qualified,
                    kind=kind,
                    start_byte=definition.node.start_byte,
                    end_byte=definition.node.end_byte,
                    start_line=line_index.line_at(definition.node.start_byte),
                    end_line=line_index.line_at(
                        max(definition.node.start_byte, definition.node.end_byte - 1)
                    ),
                    parameters=self._parameter_shapes(
                        language, parameter_nodes.get(definition.node.id)
                    ),
                )
            )
        return rows

    @staticmethod
    def _parameter_shapes(language: str, parameters: Node | None) -> list[ParameterShape]:
        if parameters is None:
            return []
        rows: list[ParameterShape] = []
        positional_only = False
        keyword_only = False
        for child in parameters.named_children:
            if child.type == "positional_separator":
                rows = [
                    row.model_copy(update={"kind": "positional_only"})
                    if row.kind == "positional"
                    else row
                    for row in rows
                ]
                positional_only = False
                continue
            if child.type == "keyword_separator":
                keyword_only = True
                continue
            name_node = child if child.type in {"identifier", "property_identifier"} else None
            name_node = (
                name_node
                or child.child_by_field_name("name")
                or child.child_by_field_name("pattern")
            )
            if name_node is None:
                name_node = child.named_child(0)
            if name_node is None:
                continue
            name = (name_node.text or b"").decode("utf-8")
            child_text = (child.text or b"").decode("utf-8")
            kind: ParameterKind
            if (
                child.type in {"list_splat_pattern", "rest_pattern"}
                or name_node.type == "rest_pattern"
            ):
                kind = "variadic"
                name = name.removeprefix("*").removeprefix("...")
            elif child.type == "dictionary_splat_pattern":
                kind = "keyword_variadic"
                name = name.removeprefix("**")
            elif positional_only:
                kind = "positional_only"
            elif keyword_only:
                kind = "keyword_only"
            else:
                kind = "positional"
            default = child.child_by_field_name("value") is not None or "=" in child_text
            required = not default and child.type != "optional_parameter"
            if language != "python" and kind == "variadic":
                required = False
            rows.append(ParameterShape(name=name, kind=kind, required=required, position=len(rows)))
            if language == "python" and child.type == "list_splat_pattern":
                keyword_only = True
        return rows

    @staticmethod
    def _call_shape(node: Node) -> CallShape:
        arguments = node.child_by_field_name("arguments")
        positional_count = 0
        keywords: list[str] = []
        positional_spread = False
        keyword_spread = False
        if arguments is not None:
            for argument in arguments.named_children:
                if argument.type in {"list_splat", "spread_element"}:
                    positional_spread = True
                elif argument.type == "dictionary_splat":
                    keyword_spread = True
                elif argument.type == "keyword_argument":
                    name = argument.child_by_field_name("name")
                    if name is not None:
                        keywords.append((name.text or b"").decode("utf-8"))
                else:
                    positional_count += 1
        type_arguments = node.child_by_field_name("type_arguments")
        return CallShape(
            positional_count=positional_count,
            keywords=keywords,
            has_positional_spread=positional_spread,
            has_keyword_spread=keyword_spread,
            type_argument_count=(
                len(type_arguments.named_children) if type_arguments is not None else None
            ),
            constructor=node.type == "new_expression",
        )

    @staticmethod
    def _binding_identifiers(node: Node) -> Iterator[Node]:
        """Yield local binding identifiers without mistaking object property keys for bindings."""
        if node.type in {"identifier", "shorthand_property_identifier_pattern"}:
            yield node
        elif node.type == "pair_pattern":
            value = node.child_by_field_name("value")
            if value is not None:
                yield from TreeSitterExtractor._binding_identifiers(value)
        elif node.type == "assignment_pattern":
            left = node.child_by_field_name("left")
            if left is not None:
                yield from TreeSitterExtractor._binding_identifiers(left)
        elif node.type == "rest_pattern":
            child = node.named_child(0)
            if child is not None:
                yield from TreeSitterExtractor._binding_identifiers(child)
        elif node.type in {"array_pattern", "object_pattern"}:
            for child in node.named_children:
                yield from TreeSitterExtractor._binding_identifiers(child)

    def _python_records(self, node: Node, source: bytes, add_reference: _ReferenceAdder) -> None:
        if node.type == "import_from_statement":
            module = node.child_by_field_name("module_name")
            module_path = _capture_name(source, module) if module is not None else None
            for child in node.named_children:
                if child == module:
                    continue
                imported = (
                    child.child_by_field_name("name") if child.type == "aliased_import" else child
                )
                alias_node = (
                    child.child_by_field_name("alias") if child.type == "aliased_import" else None
                )
                imported_name = (
                    _capture_name(source, imported)
                    if imported is not None
                    else _capture_name(source, child)
                )
                alias = _capture_name(source, alias_node) if alias_node is not None else None
                add_reference(
                    "import",
                    child,
                    target_name=imported_name,
                    written_name=alias or imported_name,
                    module_path=module_path,
                    imported_name=imported_name,
                    alias=alias,
                )
        elif node.type == "import_statement":
            for child in node.named_children:
                imported = (
                    child.child_by_field_name("name") if child.type == "aliased_import" else child
                )
                alias_node = (
                    child.child_by_field_name("alias") if child.type == "aliased_import" else None
                )
                imported_name = (
                    _capture_name(source, imported)
                    if imported is not None
                    else _capture_name(source, child)
                )
                alias = _capture_name(source, alias_node) if alias_node is not None else None
                add_reference(
                    "import",
                    child,
                    target_name=imported_name,
                    written_name=alias or imported_name,
                    module_path=imported_name,
                    imported_name=None,
                    alias=alias,
                )
        elif node.type == "decorator":
            target = node.child_by_field_name("function") or node.named_child(0)
            if target is not None and target.type == "call":
                target = target.child_by_field_name("function")
            if target is not None:
                add_reference(
                    "decorator",
                    target,
                    target_name=_capture_name(source, target),
                    written_name=_capture_name(source, target),
                )
        elif node.type == "class_definition":
            superclasses = node.child_by_field_name("superclasses")
            if superclasses is not None:
                for item in superclasses.named_children:
                    add_reference(
                        "inheritance",
                        item,
                        target_name=_capture_name(source, item),
                        written_name=_capture_name(source, item),
                    )
        elif node.type == "call":
            function = node.child_by_field_name("function")
            if function is not None:
                receiver = function.child_by_field_name("object")
                add_reference(
                    "call",
                    function,
                    target_name=_capture_name(source, function),
                    written_name=_capture_name(source, function),
                    receiver_text=_capture_name(source, receiver) if receiver is not None else None,
                    call_shape=self._call_shape(node),
                )
        elif node.type == "type":
            target = node.named_child(0)
            if target is not None:
                add_reference(
                    "type_use",
                    target,
                    target_name=_capture_name(source, target),
                    written_name=_capture_name(source, target),
                )

    def _javascript_records(
        self, node: Node, source: bytes, add_reference: _ReferenceAdder
    ) -> None:
        if node.type == "import_statement":
            source_node = node.child_by_field_name("source")
            module_path = (
                _capture_name(source, source_node).strip("'\"") if source_node is not None else None
            )
            clause = next(
                (child for child in node.named_children if child.type == "import_clause"), None
            )
            if clause is not None:
                for item in clause.named_children:
                    if item.type == "named_imports":
                        for specifier in item.named_children:
                            name = specifier.child_by_field_name("name")
                            alias = specifier.child_by_field_name("alias")
                            imported = (
                                _capture_name(source, name)
                                if name is not None
                                else _capture_name(source, specifier)
                            )
                            alias_text = _capture_name(source, alias) if alias is not None else None
                            add_reference(
                                "import",
                                specifier,
                                target_name=imported,
                                written_name=alias_text or imported,
                                module_path=module_path,
                                imported_name=imported,
                                alias=alias_text,
                            )
                    elif item.type == "namespace_import":
                        alias = item.named_child(0)
                        if alias is not None:
                            add_reference(
                                "import",
                                alias,
                                target_name="*",
                                written_name=_capture_name(source, alias),
                                module_path=module_path,
                                imported_name="*",
                                alias=_capture_name(source, alias),
                            )
                    else:
                        add_reference(
                            "import",
                            item,
                            target_name="default",
                            written_name=_capture_name(source, item),
                            module_path=module_path,
                            imported_name="default",
                        )
        elif node.type == "export_statement":
            source_node = node.child_by_field_name("source")
            module_path = (
                _capture_name(source, source_node).strip("'\"") if source_node is not None else None
            )
            for clause in node.named_children:
                if clause.type == "export_clause":
                    for specifier in clause.named_children:
                        name = specifier.child_by_field_name("name")
                        alias = specifier.child_by_field_name("alias")
                        exported = _capture_name(source, alias or name or specifier)
                        add_reference(
                            "export",
                            specifier,
                            target_name=_capture_name(source, name)
                            if name is not None
                            else exported,
                            written_name=exported,
                            module_path=module_path,
                            imported_name=_capture_name(source, name)
                            if name is not None
                            else exported,
                            alias=_capture_name(source, alias) if alias is not None else None,
                        )
            declaration = node.child_by_field_name("declaration")
            if declaration is not None:
                if declaration.type in {"lexical_declaration", "variable_declaration"}:
                    for declarator in declaration.named_children:
                        if declarator.type != "variable_declarator":
                            continue
                        name = declarator.child_by_field_name("name")
                        if name is None:
                            continue
                        for binding in self._binding_identifiers(name):
                            exported = _capture_name(source, binding)
                            add_reference(
                                "export",
                                binding,
                                target_name=exported,
                                written_name=exported,
                            )
                else:
                    name = declaration.child_by_field_name("name")
                    if name is not None:
                        exported = _capture_name(source, name)
                        is_default = any(child.type == "default" for child in node.children)
                        add_reference(
                            "export",
                            name,
                            target_name=exported,
                            written_name="default" if is_default else exported,
                        )
            value = node.child_by_field_name("value")
            if value is not None:
                add_reference(
                    "export",
                    value,
                    target_name=_capture_name(source, value),
                    written_name="default",
                )
        elif node.type in {"class_heritage", "extends_type_clause"}:
            for item in node.named_children:
                add_reference(
                    "inheritance",
                    item,
                    target_name=_capture_name(source, item),
                    written_name=_capture_name(source, item),
                )
        elif node.type in {"call_expression", "new_expression"}:
            function = node.child_by_field_name("function") or node.child_by_field_name(
                "constructor"
            )
            if function is not None:
                receiver = function.child_by_field_name("object")
                add_reference(
                    "call",
                    function,
                    target_name=_capture_name(source, function),
                    written_name=_capture_name(source, function),
                    receiver_text=_capture_name(source, receiver) if receiver is not None else None,
                    call_shape=self._call_shape(node),
                )
        elif node.type == "generic_type":
            add_reference(
                "type_use",
                node,
                target_name=_capture_name(source, node),
                written_name=_capture_name(source, node),
            )
        elif node.type == "type_annotation":
            target = node.named_child(0)
            if target is not None and target.type != "generic_type":
                add_reference(
                    "type_use",
                    target,
                    target_name=_capture_name(source, target),
                    written_name=_capture_name(source, target),
                )

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
