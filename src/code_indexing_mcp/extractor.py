"""Tree-sitter based symbol and module chunk extraction."""

from __future__ import annotations

import re
import threading
import time
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Final

import tree_sitter_c
import tree_sitter_c_sharp
import tree_sitter_cpp
import tree_sitter_go
import tree_sitter_hcl
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_json
import tree_sitter_lua
import tree_sitter_python
import tree_sitter_rust
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
STRUCTURAL_LANGUAGES: Final = frozenset({"python", "javascript", "typescript", "tsx"})
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


def _first_named_child(node: Node) -> Node | None:
    """Return *node*'s first named child that is not "extra" trivia (a comment).

    ``node.named_child(0)`` picks whatever sits first in source order, so a
    comment placed before the real content (`require(/* c */ './mod')`,
    `import * as /* c */ ns from 'mod'`) is silently mistaken for it instead
    of being skipped, the same way a comment was mistaken for a positional
    call argument (finding 7).
    """
    for child in node.named_children:
        if not child.is_extra:
            return child
    return None


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
        "go": Language(tree_sitter_go.language()),
        "terraform": Language(tree_sitter_hcl.language()),
        "rust": Language(tree_sitter_rust.language()),
        "c": Language(tree_sitter_c.language()),
        "cpp": Language(tree_sitter_cpp.language()),
        "lua": Language(tree_sitter_lua.language()),
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
        reference_extraction_ns = 0
        if language in STRUCTURAL_LANGUAGES:
            started = time.monotonic_ns()
            references, declarations = self._structural_records(
                language, tree.root_node, normalized_source, index, line_index
            )
            reference_extraction_ns = time.monotonic_ns() - started
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
            reference_extraction_ns=reference_extraction_ns,
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
                "namespace_export",
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
                    # TS's `required_parameter`/`optional_parameter` wrapper
                    # exposes a default under a `value` field; bare JS/TS
                    # `assignment_pattern` (untyped `a = LIMIT`) exposes it
                    # under `right` instead (mirrors the E8 note on
                    # `_parameter_shapes` below). Checking only `value` drops
                    # every identifier read inside a plain JS default.
                    if contains(parameter.child_by_field_name("value")) or contains(
                        parameter.child_by_field_name("right")
                    ):
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
                # TS declaration names surface as type_identifier now that the
                # identifier fallback covers it too (Task 2.2); their own name is
                # a binding, not a reference.
                "interface_declaration",
                "type_alias_declaration",
                "type_parameter",
                # JSX element names get their own `type_use`
                # component-reference row (E14); everything else inside
                # the element (attribute values, children) stays a plain
                # identifier reference.
                "jsx_opening_element",
                "jsx_self_closing_element",
                "jsx_closing_element",
            }:
                excluded_fields = ("name",)
            elif parent.type in {
                "assignment",
                "assignment_expression",
                "augmented_assignment",
                "named_expression",
                "for_statement",
                "for_in_clause",
                # JS `for (const item of items)` -- the loop binding is not a
                # reference to an existing `item` (E11).
                "for_in_statement",
            }:
                excluded_fields = ("left", "name")
            elif parent.type in {"arrow_function", "lambda"}:
                # `parameter` covers a parenless single-identifier arrow param
                # (`x => x`), which has no `formal_parameters` wrapper of its
                # own to be caught by the block above. `parameters` is
                # deliberately NOT excluded here: it names the
                # `formal_parameters`/`lambda_parameters` node wrapping every
                # other case, which the parameter-defaults block above has
                # already walked and decided correctly (including whether an
                # identifier inside a default value is a real read); blanket-
                # excluding the whole field here would undo that decision for
                # every arrow-function/lambda default (`(a = LIMIT) => a`,
                # `lambda a=LIMIT: a`).
                excluded_fields = ("parameter",)
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
            if current.type == "decorator":
                # JS/TS: a method/field decorator is a preceding sibling of the
                # `method_definition`/`public_field_definition` it decorates, not
                # its parent (unlike Python's `decorated_definition` wrapper, and
                # unlike a TS/JS *class* decorator, which the grammar does nest
                # inside `class_declaration`) -- attribute it to that sibling.
                sibling = current.next_sibling
                if (
                    sibling is not None
                    and (declaration := declarations.get(sibling.id)) is not None
                ):
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
            name_node = (
                child
                if child.type
                in {"identifier", "property_identifier", "object_pattern", "array_pattern"}
                else None
            )
            name_node = (
                name_node
                or child.child_by_field_name("name")
                or child.child_by_field_name("pattern")
            )
            if name_node is None:
                # e.g. a bare `rest_pattern` (`...rest`) -- its identifier is
                # a plain child, not a named field. A leading comment
                # (`.../* c */ rest`) must not be mistaken for it (same class
                # as finding 7/8).
                name_node = _first_named_child(child)
            if name_node is None:
                continue
            # A destructured slot (`{ a, b }` / `[a, b]`) collapses to ONE
            # positional parameter marked `destructured`, never N flat ones --
            # expanding it would corrupt positional matching for every caller
            # (E7). It can appear bare -- JS, or TS without a wrapper, where
            # `child` itself IS the pattern -- or as the `pattern`/`name`
            # field of a `required_parameter`/`optional_parameter` wrapper
            # (TS) -- `name_node` above already resolves to it in both cases.
            destructured = name_node.type in {"object_pattern", "array_pattern"}
            if destructured:
                binding_names = [
                    (binding.text or b"").decode("utf-8")
                    for binding in TreeSitterExtractor._binding_identifiers(name_node)
                ]
                name = ",".join(binding_names) if binding_names else "destructured"
            else:
                name = (name_node.text or b"").decode("utf-8")
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
            # A default value is authoritative via node structure, never text
            # matching (E8): TS's `required_parameter`/`optional_parameter`
            # wrapper exposes it as a `value` field; bare JS/TS
            # `assignment_pattern` (untyped `a = 1`) exposes it as `right`.
            # A callback type's `=>` in the parameter's raw text is not a
            # default and must never be mistaken for one.
            default = (
                child.child_by_field_name("value") is not None
                or child.child_by_field_name("right") is not None
            )
            required = not default and child.type != "optional_parameter"
            if language != "python" and kind == "variadic":
                required = False
            rows.append(
                ParameterShape(
                    name=name,
                    kind=kind,
                    required=required,
                    position=len(rows),
                    destructured=destructured,
                )
            )
            if language == "python" and child.type == "list_splat_pattern":
                keyword_only = True
        return rows

    # Node types that hold a genuine positional/keyword argument list. Anything
    # else reachable through the `arguments` field (a tagged template's
    # `template_string`, a `new` with no parens at all) is not a positional arg
    # list and must not have its children miscounted as one (E4).
    _ARGUMENT_LIST_TYPES: Final = frozenset({"arguments", "argument_list"})

    @staticmethod
    def _call_shape(node: Node) -> CallShape:
        arguments = node.child_by_field_name("arguments")
        positional_count = 0
        keywords: list[str] = []
        positional_spread = False
        keyword_spread = False
        if arguments is not None and arguments.type in TreeSitterExtractor._ARGUMENT_LIST_TYPES:
            for argument in arguments.named_children:
                if argument.is_extra:
                    # A comment is a named "extra" node inside the argument
                    # list, not an argument -- `g(1,  # note\n  2)` must count
                    # 2 positional args, not 3 (E4-adjacent).
                    continue
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
        elif arguments is not None and arguments.type == "generator_expression":
            # Python `summarize(x for x in items)` -- a single argument whose
            # contents are not a positional list. Model it as one positional
            # with spread-like uncertainty so signature analysis routes to
            # `review` instead of fabricating a match (E4).
            positional_count = 1
            positional_spread = True
        # else: e.g. a tagged template's `template_string` -- no positional
        # args at all; its `string_fragment`/`template_substitution` children
        # are not call arguments and must not be counted (E4).
        type_arguments = node.child_by_field_name("type_arguments")
        return CallShape(
            positional_count=positional_count,
            keywords=keywords,
            has_positional_spread=positional_spread,
            has_keyword_spread=keyword_spread,
            type_argument_count=(
                sum(1 for argument in type_arguments.named_children if not argument.is_extra)
                if type_arguments is not None
                else None
            ),
            constructor=node.type == "new_expression",
        )

    @staticmethod
    def _string_literal_argument(node: Node, source: bytes) -> str | None:
        """Return the quote-stripped text of a call's sole string-literal argument.

        Used for `require('./mod')` and dynamic `import('./mod')` (E9): both
        keep an ordinary `call` row for signature purposes, but gain a
        `module_path` so the module edge stays visible to the resolver.
        """
        arguments = node.child_by_field_name("arguments")
        if arguments is None:
            return None
        # A leading comment (`require(/* c */ './mod')`) is a named "extra"
        # node in source order before the real argument; `named_child(0)`
        # would grab it instead and silently drop the module edge.
        first = _first_named_child(arguments)
        if first is None or first.type != "string":
            return None
        return _capture_name(source, first).strip("'\"")

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
            # A comment right after `...` (`[.../* c */ rest]`) must not be
            # mistaken for the bound identifier -- same class of bug as
            # finding 7/8.
            child = _first_named_child(node)
            if child is not None:
                yield from TreeSitterExtractor._binding_identifiers(child)
        elif node.type in {"array_pattern", "object_pattern"}:
            for child in node.named_children:
                yield from TreeSitterExtractor._binding_identifiers(child)

    _TYPE_WRAPPER_TYPES: Final = frozenset(
        {"union_type", "intersection_type", "array_type", "type_arguments"}
    )

    @staticmethod
    def _descend_type_names(node: Node | None) -> list[Node]:
        """Descend a TS wrapper node to its identifying name leaf(ves).

        Unwraps `generic_type` (the head name plus one entry per type argument),
        `union_type`, `intersection_type`, `array_type`, `function_type` (its
        `return_type` only -- the parameter list is a binding context, not a type
        reference), and `type_arguments`, stopping at `type_identifier`/`identifier`/
        `predefined_type` (`number`, `string`, `void`, ...) leaves. Qualified
        names such as `ns.Base` stay intact as one resolvable leaf. Anything
        else (for example an object type literal) yields nothing.
        """
        if node is None:
            return []
        if node.type in {
            "type_identifier",
            "identifier",
            "predefined_type",
            "member_expression",
            "nested_type_identifier",
        }:
            return [node]
        if node.type == "generic_type":
            names = TreeSitterExtractor._descend_type_names(node.child_by_field_name("name"))
            names.extend(
                TreeSitterExtractor._descend_type_names(node.child_by_field_name("type_arguments"))
            )
            return names
        if node.type == "function_type":
            return TreeSitterExtractor._descend_type_names(node.child_by_field_name("return_type"))
        if node.type in TreeSitterExtractor._TYPE_WRAPPER_TYPES:
            names = []
            for child in node.named_children:
                names.extend(TreeSitterExtractor._descend_type_names(child))
            return names
        return []

    @staticmethod
    def _emit_type_use_names(
        node: Node | None, source: bytes, add_reference: _ReferenceAdder
    ) -> None:
        """Emit a `type_use` row per identifying name reachable by descending `node`."""
        for name in TreeSitterExtractor._descend_type_names(node):
            add_reference(
                "type_use",
                name,
                target_name=_capture_name(source, name),
                written_name=_capture_name(source, name),
            )

    @staticmethod
    def _emit_heritage_name(
        node: Node | None, source: bytes, add_reference: _ReferenceAdder
    ) -> None:
        """Emit the head name of a heritage clause as `inheritance`, extras as `type_use`.

        `extends Base<T>` yields an `inheritance` row for `Base` and a `type_use`
        row for `T`; `extends Base` (no type arguments) yields just the former.
        """
        names = TreeSitterExtractor._descend_type_names(node)
        if not names:
            return
        head, *rest = names
        add_reference(
            "inheritance",
            head,
            target_name=_capture_name(source, head),
            written_name=_capture_name(source, head),
        )
        for extra in rest:
            add_reference(
                "type_use",
                extra,
                target_name=_capture_name(source, extra),
                written_name=_capture_name(source, extra),
            )

    @staticmethod
    def _is_assignment_target(node: Node) -> bool:
        """True if `node` is the LHS of a plain or augmented assignment."""
        parent = node.parent
        if parent is None:
            return False
        if parent.type in {
            "assignment",
            "augmented_assignment",
            "assignment_expression",
            "augmented_assignment_expression",
        }:
            return parent.child_by_field_name("left") == node
        return False

    @staticmethod
    def _emit_member_access(node: Node, source: bytes, add_reference: _ReferenceAdder) -> None:
        """Emit `read`/`write` for a member-access node that is not itself a call.

        Handles Python `attribute` and JS/TS `member_expression` (E5). Three
        cases already own this exact span with a different `kind` and must
        stay singly represented, not duplicated as a `read`/`write` too:
        a call's `function`/`constructor` (its own `call` row), a Python
        decorator's target (its own `decorator` row), and a class's
        superclass entry (its own `inheritance` row).
        """
        parent = node.parent
        if parent is not None:
            ancestor: Node | None = parent
            while ancestor is not None:
                if ancestor.type in {"class_heritage", "extends_type_clause"}:
                    return
                ancestor = ancestor.parent
            if (
                parent.type in {"call", "call_expression"}
                and parent.child_by_field_name("function") == node
            ):
                return
            if (
                parent.type == "new_expression"
                and parent.child_by_field_name("constructor") == node
            ):
                return
            if parent.type == "decorator":
                return
            if (
                parent.type == "argument_list"
                and parent.parent is not None
                and parent.parent.type == "class_definition"
                and parent.parent.child_by_field_name("superclasses") == parent
            ):
                return
        property_field = node.child_by_field_name("attribute") or node.child_by_field_name(
            "property"
        )
        if property_field is None:
            return
        object_field = node.child_by_field_name("object")
        text = _capture_name(source, node)
        kind: ReferenceKind = "write" if TreeSitterExtractor._is_assignment_target(node) else "read"
        add_reference(
            kind,
            node,
            target_name=text,
            written_name=text,
            receiver_text=_capture_name(source, object_field) if object_field is not None else None,
        )

    def _python_records(self, node: Node, source: bytes, add_reference: _ReferenceAdder) -> None:
        if node.type == "import_from_statement":
            module = node.child_by_field_name("module_name")
            module_path = _capture_name(source, module) if module is not None else None
            for child in node.named_children:
                if child == module or child.is_extra:
                    # A comment among the imported names (`from pkg import (a,
                    # # note\n b)`) is a named "extra" node too -- without this
                    # it would fall through as a bogus import of itself.
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
                if child.is_extra:
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
                    if item.is_extra:
                        # A comment among the base classes (`class Child(Base,
                        # # note\n Other):`) is a named "extra" node too, not
                        # a base class.
                        continue
                    if item.type == "keyword_argument":
                        # e.g. `metaclass=Meta` -- the value already surfaces as a
                        # `read` via the plain identifier fallback; the clause
                        # itself (`metaclass=Meta`) is not a base class (E12).
                        continue
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
        elif node.type == "attribute":
            self._emit_member_access(node, source, add_reference)
        elif node.type in {"assignment", "augmented_assignment"}:
            # `__all__ = [...]`/`__all__ += [...]` (E13) -- the query already
            # restricts the match to a literal left-hand `__all__` via #eq?,
            # so no name check is needed here. Each string entry becomes an
            # `export` row naming the symbol it re-publishes; a rename of
            # that symbol must also touch its `__all__` entry.
            right = node.child_by_field_name("right")
            if right is not None and right.type in {"list", "tuple"}:
                for entry in right.named_children:
                    if entry.type != "string":
                        continue
                    exported = _capture_name(source, entry).strip("'\"")
                    add_reference(
                        "export",
                        entry,
                        target_name=exported,
                        written_name=exported,
                        imported_name=exported,
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
                    if item.is_extra:
                        # A comment between clause items (`import Default,
                        # /* c */ { a } from 'mod'`) is a named "extra" node
                        # too -- the `else` branch below would otherwise
                        # mistake it for a bare default-import identifier.
                        continue
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
                        alias = _first_named_child(item)
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
            else:
                # `import './polyfill'` -- no import_clause at all: a
                # side-effect import that still opens a module edge (E9).
                add_reference(
                    "import",
                    node,
                    target_name=module_path or "",
                    written_name=module_path or "",
                    module_path=module_path,
                    imported_name=None,
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
                elif clause.type == "namespace_export":
                    # `export * as ns from './x'` -- the namespace alias lives
                    # under the wrapper node, not as a direct export_statement
                    # child (E3).
                    alias_node = _first_named_child(clause)
                    alias_text = (
                        _capture_name(source, alias_node) if alias_node is not None else None
                    )
                    add_reference(
                        "export",
                        clause,
                        target_name="*",
                        written_name=alias_text or "*",
                        module_path=module_path,
                        imported_name="*",
                        alias=alias_text,
                    )
            if module_path is not None and any(child.type == "*" for child in node.children):
                # `export * from './x'` -- bare barrel re-export: no clause at
                # all, just a literal `*` token directly under export_statement
                # (E3). `export * as ns ...` is handled above via
                # `namespace_export`, whose own `*` is nested one level deeper
                # so it never reaches this branch.
                star = next(child for child in node.children if child.type == "*")
                add_reference(
                    "export",
                    star,
                    target_name="*",
                    written_name="*",
                    module_path=module_path,
                    imported_name="*",
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
        elif node.type == "decorator":
            # `@Name`, `@ns.Name`, `@Factory()` -- mirrors the Python decorator
            # handler (E6). The factory-call form keeps its own `call` row from
            # the `call_expression` branch; this row is additional.
            target = _first_named_child(node)
            if target is not None and target.type == "call_expression":
                target = target.child_by_field_name("function")
            if target is not None:
                add_reference(
                    "decorator",
                    target,
                    target_name=_capture_name(source, target),
                    written_name=_capture_name(source, target),
                )
        elif node.type == "class_heritage":
            for clause in node.named_children:
                if clause.type == "extends_clause":
                    # TS: `extends Base<T>` -- the identifier is under a `value`
                    # field, not the clause itself (E1).
                    self._emit_heritage_name(
                        clause.child_by_field_name("value"), source, add_reference
                    )
                elif clause.type == "implements_clause":
                    # TS: `implements Foo, Bar<T>` -- each interface name is a
                    # direct named child of the clause (E1).
                    for interface in clause.named_children:
                        self._emit_heritage_name(interface, source, add_reference)
                else:
                    # JS: the grammar puts the identifier directly under
                    # class_heritage (no extends_clause wrapper); already worked.
                    self._emit_heritage_name(clause, source, add_reference)
        elif node.type == "extends_type_clause":
            # TS interface heritage: named children are already type_identifier
            # (or generic_type, handled by the E2 generic_type branch below) --
            # left untouched, see hardening plan Task 2.1.
            for item in node.named_children:
                if item.is_extra:
                    # A comment (`extends /* c */ Base`) is a named "extra"
                    # node too, not an interface name.
                    continue
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
                is_module_call = function.type == "import" or (
                    function.type == "identifier" and _capture_name(source, function) == "require"
                )
                module_path = (
                    self._string_literal_argument(node, source) if is_module_call else None
                )
                add_reference(
                    "call",
                    function,
                    target_name=_capture_name(source, function),
                    written_name=_capture_name(source, function),
                    receiver_text=_capture_name(source, receiver) if receiver is not None else None,
                    call_shape=self._call_shape(node),
                    module_path=module_path,
                )
        elif node.type == "generic_type":
            # `Box<Item>` -- one type_use for `Box`, one per type argument (E2).
            self._emit_type_use_names(node, source, add_reference)
        elif node.type == "type_annotation":
            # `: A | B`, `: C & D`, `: Widget[]`, `: (e: Event) => Widget` -- unwrap
            # to the inner type names instead of capturing the whole expression
            # verbatim (E2). A nested generic_type is also matched by its own
            # top-level pattern above; `add_reference` dedupes the identical span.
            # A leading comment (`: /* c */ Widget`) is a named "extra" node
            # too and must not be mistaken for the annotated type -- that
            # silently dropped the type_use row entirely.
            target = _first_named_child(node)
            self._emit_type_use_names(target, source, add_reference)
        elif node.type == "member_expression":
            self._emit_member_access(node, source, add_reference)
        elif node.type in {
            "jsx_opening_element",
            "jsx_self_closing_element",
            "jsx_closing_element",
        }:
            # `<Widget />`, `<Widget>...</Widget>` -- a component-reference
            # row per element name, opening/self-closing and closing alike,
            # so a rename finds every JSX use (E14, TSX only). Lower-case
            # names (`<div>`) are intrinsic HTML tags, not project symbols,
            # but resolving that distinction is the resolver's job, not the
            # extractor's -- an unmatched `type_use` is simply never a hit.
            target = node.child_by_field_name("name")
            if target is not None:
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
