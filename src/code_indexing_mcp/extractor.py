"""Tree-sitter based symbol and module chunk extraction."""

from __future__ import annotations

import re
import threading
import time
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterator, Sequence
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
from tree_sitter_language_pack import DownloadError, get_language

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
STRUCTURAL_LANGUAGES: Final = frozenset({"python", "javascript", "typescript", "tsx", "go", "rust"})
# Per-language handler for the non-`reference.identifier` captures of that
# language's structural query. Both handlers share one signature
# `(node, source, add_reference)` and re-dispatch on `node.type`, so a new
# structured language slots in as one map entry beside its `.scm` file.
_STRUCTURAL_RECORD_HANDLERS: Final[dict[str, str]] = {
    "python": "_python_records",
    "javascript": "_javascript_records",
    "typescript": "_javascript_records",
    "tsx": "_javascript_records",
    "go": "_go_records",
    "rust": "_rust_records",
}
_PACK_DOWNLOAD_ATTEMPTS: Final = 6
_PACK_DOWNLOAD_BACKOFF_SECONDS: Final = 1.0
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


def _pack_language(name: str) -> Language:
    """Resolve a language the pack downloads on first use, retrying transiently.

    The pack fetches its manifest and parsers from a GitHub release the first
    time a language is resolved, then caches the result on disk. That endpoint
    has transient outages (503s, dropped connections) that can outlast a
    couple of quick retries -- CI observed one dropping every connection for
    tens of seconds -- so the backoff window (1+2+4+8+16s) is sized to ride
    out such a window rather than just a single failed request.
    """

    for attempt in range(_PACK_DOWNLOAD_ATTEMPTS):
        try:
            return get_language(name)
        except DownloadError:
            if attempt == _PACK_DOWNLOAD_ATTEMPTS - 1:
                raise
            time.sleep(_PACK_DOWNLOAD_BACKOFF_SECONDS * (2**attempt))
    raise AssertionError("unreachable")


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
        "gdscript": _pack_language("gdscript"),
        "gdshader": _pack_language("gdshader"),
        "godot_resource": _pack_language("godot_resource"),
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
        # A definition can own several captured parameter lists -- Go's
        # `method_declaration` captures its receiver list AND its parameter
        # list. Slot order follows source order (receiver first), which is
        # exactly slot 0 = Python's `self` convention.
        parameter_nodes: dict[int, list[Node]] = {}
        for _, captures in matches:
            for parameter_node in captures.get("declaration.parameters", []):
                owner = parameter_node.parent
                while owner is not None and owner.id not in index.by_node_id:
                    owner = owner.parent
                if owner is not None:
                    bucket = parameter_nodes.setdefault(owner.id, [])
                    if all(existing != parameter_node for existing in bucket):
                        bucket.append(parameter_node)
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

        handler = _STRUCTURAL_RECORD_HANDLERS[language]
        method = getattr(self, handler)
        for _, captures in matches:
            for capture, nodes in captures.items():
                if not capture.startswith("reference."):
                    continue
                for node in nodes:
                    if capture == "reference.identifier":
                        self._identifier_record(language, node, source, add)
                    else:
                        # Reference-query captures are decorative: every language
                        # handler re-dispatches on `node.type`, so adding a
                        # language is one `.scm` file plus one map entry.
                        method(node, source, add)
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
                # Rust `use` trees bind spellings; the import rows own them.
                "use_declaration",
                "use_as_clause",
                "scoped_use_list",
                "use_list",
                "use_wildcard",
                # `&self`/`&mut self` receivers are not reads of a `self`
                # symbol (method-body `self` reads ride field expressions).
                "self_parameter",
            }:
                return
            if parent.type in {
                "parameters",
                "formal_parameters",
                "lambda_parameters",
                "parameter_list",
                "closure_parameters",
            }:
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
                # Go declaration-name fields surface as identifier or
                # field_identifier and are bindings, not references. The
                # receiver/parameter names are already cut by the
                # parameter_list walk above; these cut the named owners.
                "method_declaration",
                "type_spec",
                "type_alias",
                "field_declaration",
                "method_elem",
                # A var/const spec's annotated type is emitted as `type_use` by
                # the handler; the spec's own name is a binding.
                "var_spec",
                "const_spec",
                # Rust named owners are bindings; their type fields are owned
                # by the handler (`return_type` covers the bare `-> Widget`
                # form, `trait` the implemented trait of an `impl`).
                "function_item",
                "function_signature_item",
                "struct_item",
                "enum_item",
                "trait_item",
                "mod_item",
                "const_item",
                "static_item",
                "enum_variant",
                "impl_item",
                "struct_expression",
            }:
                # `result` covers a function/method's bare result type
                # (`func load() Store`): the handler owns its `type_use` row,
                # so a parallel plain read would only duplicate it with the
                # wrong kind. Parameters are cut by the parameter_list walk
                # above; parenthesized results by it too.
                excluded_fields = ("name", "type", "result", "return_type", "trait")
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
                # Go write targets and new bindings: `x = v` writes, `x := v`
                # declares, `range`'s left side declares. The `write` row for
                # an LHS member access comes from the selector handler; a bare
                # LHS identifier is a pure binding here.
                "assignment_statement",
                "short_var_declaration",
                "range_clause",
                # Rust `count += 1` mutates the existing binding.
                "compound_assignment_expr",
            }:
                excluded_fields = ("left", "name")
            elif parent.type in {"let_declaration", "for_expression"}:
                # Rust `let x = ...` / `for x in ...`: the pattern names a new
                # binding; the optional `: Type` annotation is the handler's
                # `type_use` and the value stays a plain read.
                excluded_fields = ("pattern",)
            elif parent.type == "scoped_identifier":
                # Path segments are namespace spellings; the final `name` may
                # be a real value read (`State::Ready`) and stays eligible.
                # Inside a `use` tree the climb above still cuts the leaf.
                excluded_fields = ("path",)
            elif parent.type == "field_initializer":
                # `Widget { label: text }` -- the field key is a binding, not
                # a read of a `label` symbol (a shorthand `Widget { label }`
                # keeps its read: the identifier is the initializer there).
                excluded_fields = ("field",)
            elif parent.type in {"inc_statement", "dec_statement"}:
                # Go `x++`: the bare operand is mutated in place, and the
                # selector handler owns the write row for `p.x++` -- the same
                # single-representation rule as the assignment LHS above (the
                # statement has no named field to exclude, hence the direct
                # return).
                return
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
            elif language != "python" and parent.type in {
                "pair",
                "pair_pattern",
                # Go composite-literal keys (`Widget{Name: "x"}`) are field
                # bindings, not reads.
                "keyed_element",
            }:
                excluded_fields = ("key",)
            elif language != "python" and parent.type in {
                # Go/Rust type wrappers: every identifier directly inside one
                # is a type position, already emitted as `type_use` by the
                # handler -- a parallel plain read would only duplicate it.
                # (An array length expression nested deeper is untouched by
                # this direct-parent rule.)
                "pointer_type",
                "slice_type",
                "array_type",
                "map_type",
                "channel_type",
                "function_type",
                "parenthesized_type",
                "qualified_type",
                "reference_type",
                "tuple_type",
                "scoped_type_identifier",
                "dynamic_type",
                "type_arguments",
                "ordered_field_declaration_list",
            }:
                return
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
        parameter_nodes: dict[int, list[Node]],
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
    def _parameter_shapes(
        language: str, parameters: Node | None | Sequence[Node | None]
    ) -> list[ParameterShape]:
        """Shapes for one definition's captured parameter list(s).

        Most languages capture a single `parameters` node and this is the
        flat loop it always was. Go's methods capture two lists in source
        order (receiver first, then parameters); processing them
        sequentially puts the receiver at slot 0, mirroring Python's `self`.
        """
        if parameters is None:
            return []
        nodes = list(parameters) if not isinstance(parameters, Node) else [parameters]
        rows: list[ParameterShape] = []
        for node in nodes:
            rows.extend(TreeSitterExtractor._one_parameter_list(language, node))
        # Slot positions are semantic (signature-compat analysis compares them
        # caller-to-callee), so they are renumbered over the MERGED lists --
        # Go's receiver+parameters pair would otherwise both start at 0.
        return [row.model_copy(update={"position": index}) for index, row in enumerate(rows)]

    @staticmethod
    def _one_parameter_list(language: str, parameters: Node | None) -> list[ParameterShape]:
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
            elif child.type == "variadic_parameter_declaration":
                # Go's `opts ...string` -- a genuine variadic slot.
                kind = "variadic"
                name = name.removeprefix("*")
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
            # Rust `count += 1` -- the field operand is the write target.
            "compound_assignment_expr",
        }:
            return parent.child_by_field_name("left") == node
        # Go wraps each assignment side in an `expression_list` (`s.next =
        # nil`). The list itself is never a symbol; peek one level out without
        # touching the Python/JS shapes, whose LHS identifiers are direct
        # children of their assignment nodes. `short_var_declaration` counts
        # too: `a, p.z := 1, 2` re-assigns the existing `p.z`, so its selector
        # operand is a write, not a read.
        if parent.type == "expression_list":
            grandparent = parent.parent
            if grandparent is not None and grandparent.type in {
                "assignment_statement",
                "augmented_assignment",
                "short_var_declaration",
            }:
                return grandparent.child_by_field_name("left") == parent
        # Go's `p.x++` / `p.x--`: the statement wraps its single operand
        # without a named field, and the operand is exactly the write target.
        return parent.type in {"inc_statement", "dec_statement"}

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

    # Go's predeclared identifiers name no project declaration; a type_use for
    # them is an unmatched row forever, so the descent stops at them instead.
    _GO_PREDECLARED_TYPES: Final = frozenset(
        {
            "bool",
            "byte",
            "complex64",
            "complex128",
            "error",
            "float32",
            "float64",
            "int",
            "int8",
            "int16",
            "int32",
            "int64",
            "rune",
            "string",
            "uint",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "uintptr",
            "any",
        }
    )

    @staticmethod
    def _go_descend_type_names(node: Node | None) -> list[Node]:
        """Descend a Go type expression to its naming `type_identifier` leaves.

        Unwraps pointer/slice/array/map/channel/function wrappers and
        `qualified_type` down to leaf names (`pkg.Widget` contributes only its
        final identifier -- the package qualifier is a namespace spelling, not
        a symbol). Predeclared types (`string`, `error`, ...) yield nothing:
        they can never be project declarations. Anything else yields nothing.
        """
        if node is None:
            return []
        if node.type == "type_identifier":
            return [node]
        if node.type == "qualified_type":
            # tree-sitter-go names the two sides `package` and `name` -- there
            # is no `type` field, so descending it silently dropped every
            # `pkg.Type` annotation. The naming leaf is the `name` field.
            return TreeSitterExtractor._go_descend_type_names(node.child_by_field_name("name"))
        if node.type in {
            "pointer_type",
            "slice_type",
            "array_type",
            "map_type",
            "channel_type",
            "parenthesized_type",
        }:
            names: list[Node] = []
            for child in node.named_children:
                names.extend(TreeSitterExtractor._go_descend_type_names(child))
            return names
        if node.type == "function_type":
            # Parameter/result lists are binding contexts of the function
            # type itself; its result type still names something.
            return TreeSitterExtractor._go_descend_type_names(node.child_by_field_name("result"))
        return []

    @staticmethod
    def _go_emit_type_uses(
        node: Node | None, source: bytes, add_reference: _ReferenceAdder
    ) -> None:
        """Emit one `type_use` per Go type-name leaf reached from *node*."""
        for name_node in TreeSitterExtractor._go_descend_type_names(node):
            written = _capture_name(source, name_node)
            if written in TreeSitterExtractor._GO_PREDECLARED_TYPES:
                continue
            add_reference("type_use", name_node, target_name=written, written_name=written)

    def _go_records(self, node: Node, source: bytes, add_reference: _ReferenceAdder) -> None:
        if node.type == "import_spec":
            path_node = node.child_by_field_name("path")
            module_path = (
                _capture_name(source, path_node).strip("'\"") if path_node is not None else None
            )
            alias_node = node.child_by_field_name("name")
            imported_name = (module_path or "").rsplit("/", 1)[-1]
            if alias_node is not None and alias_node.type == "dot":
                # Dot-imports bind every exported name without any local
                # spelling -- wildcard semantics.
                add_reference(
                    "import",
                    node,
                    target_name="*",
                    written_name="*",
                    module_path=module_path,
                    imported_name="*",
                    alias=None,
                )
                return
            alias = _capture_name(source, alias_node) if alias_node is not None else None
            # A Go import binds a whole package (a namespace), never a single
            # symbol, so `imported_name` stays None -- mirroring Python's plain
            # `import x.y`; member access goes through selector receivers.
            add_reference(
                "import",
                node,
                target_name=alias or imported_name,
                written_name=alias or imported_name,
                module_path=module_path,
                imported_name=None,
                alias=alias,
            )
        elif node.type == "call_expression":
            function = node.child_by_field_name("function")
            if function is None:
                return
            receiver = (
                function.child_by_field_name("operand")
                if function.type == "selector_expression"
                else None
            )
            add_reference(
                "call",
                function,
                target_name=_capture_name(source, function),
                written_name=_capture_name(source, function),
                receiver_text=(_capture_name(source, receiver) if receiver is not None else None),
                call_shape=self._call_shape(node),
            )
        elif node.type == "selector_expression":
            parent = node.parent
            if (
                parent is not None
                and parent.type == "call_expression"
                and parent.child_by_field_name("function") == node
            ):
                # The call row above already owns this span.
                return
            field = node.child_by_field_name("field")
            if field is None:
                return
            text = _capture_name(source, node)
            kind: ReferenceKind = (
                "write" if TreeSitterExtractor._is_assignment_target(node) else "read"
            )
            add_reference(kind, node, target_name=text, written_name=text)
        elif node.type == "type_spec":
            declared = node.child_by_field_name("type")
            if declared is None:
                return
            if declared.type == "struct_type":
                fields = declared.child_by_field_name("body")
                if fields is not None:
                    self._go_fields(fields, source, add_reference)
            elif declared.type == "interface_type":
                self._go_interface(declared, source, add_reference)
            else:
                self._go_emit_type_uses(declared, source, add_reference)
        elif node.type == "type_alias":
            self._go_emit_type_uses(node.child_by_field_name("type"), source, add_reference)
        elif node.type == "field_declaration":
            # Reached for anonymous-struct fields inside var specs (nested
            # struct_type fields), and harmlessly alongside the type_spec
            # walk for ordinary struct members -- add_reference dedupes.
            self._go_field_declaration(node, source, add_reference)
        elif node.type in {
            "var_spec",
            "parameter_declaration",
            "variadic_parameter_declaration",
            "composite_literal",
        }:
            self._go_emit_type_uses(node.child_by_field_name("type"), source, add_reference)
        elif node.type == "qualified_type":
            self._go_emit_type_uses(node, source, add_reference)
        elif node.type in {"function_declaration", "method_declaration"}:
            self._go_exports(node, source, add_reference)
            # A bare result type (`func load() Store`) hangs directly off the
            # declaration, so nothing else captures it; the identifier
            # fallback now cuts it, and this owns its `type_use` row. Pointer
            # results reach the descent the same way, and parenthesized
            # result lists are covered by the `parameter_declaration`
            # captures inside them.
            self._go_emit_type_uses(node.child_by_field_name("result"), source, add_reference)
        elif node.type in {"var_declaration", "const_declaration", "type_declaration"}:
            self._go_exports(node, source, add_reference)

    def _go_fields(self, fields: Node, source: bytes, add_reference: _ReferenceAdder) -> None:
        for field in fields.named_children:
            if field.is_extra:
                continue
            self._go_field_declaration(field, source, add_reference)

    def _go_field_declaration(
        self, node: Node, source: bytes, add_reference: _ReferenceAdder
    ) -> None:
        named = node.child_by_field_name("name")
        declared = node.child_by_field_name("type")
        if named is None and declared is not None:
            # Embedded field (`Reader` inside a struct): promoted methods are
            # inherited, so this is an inheritance edge, not a field typing.
            self._go_embedded(declared, source, add_reference)
            return
        self._go_emit_type_uses(declared, source, add_reference)

    @staticmethod
    def _go_embedded(declared: Node, source: bytes, add_reference: _ReferenceAdder) -> None:
        head = TreeSitterExtractor._go_head_identifier(declared)
        if head is None:
            return
        written = _capture_name(source, head)
        if written in TreeSitterExtractor._GO_PREDECLARED_TYPES:
            return
        add_reference("inheritance", head, target_name=written, written_name=written)

    @staticmethod
    def _go_head_identifier(node: Node | None) -> Node | None:
        """The first naming leaf of an embedded-type expression."""
        current: Node | None = node
        while current is not None:
            if current.type == "type_identifier":
                return current
            if current.type == "qualified_type":
                # The type name sits in the `name` field; `type` does not exist
                # on `qualified_type` in this grammar (see the descent above).
                current = current.child_by_field_name("name")
                continue
            named = current.named_children
            if not named:
                return None
            candidate = _first_named_child(current)
            if candidate is None or candidate is current:
                return None
            current = candidate
        return None

    @staticmethod
    def _go_emit_listed_type_uses(
        node: Node | None, source: bytes, add_reference: _ReferenceAdder
    ) -> None:
        """Type uses for a Go parameter/result list -- or a single bare type.

        `method_elem`'s `parameters`/`result` fields usually hold a
        `parameter_list` wrapping `parameter_declaration` nodes, which the
        leaf-level descent cannot enter (it only unwraps type expressions);
        unwrap the declarations' `type` fields here. A bare unparenthesized
        result (`Read() Error`) is a plain type node and descends directly.
        """
        if node is None:
            return
        if node.type == "parameter_list":
            for child in node.named_children:
                if child.is_extra:
                    continue
                TreeSitterExtractor._go_emit_type_uses(
                    child.child_by_field_name("type"), source, add_reference
                )
            return
        TreeSitterExtractor._go_emit_type_uses(node, source, add_reference)

    def _go_interface(self, interface: Node, source: bytes, add_reference: _ReferenceAdder) -> None:
        for element in interface.named_children:
            if element.is_extra:
                continue
            if element.type == "method_elem":
                # Method names are declaration-site spellings, not uses; only
                # the parameter/result types are real references.
                self._go_emit_listed_type_uses(
                    element.child_by_field_name("parameters"), source, add_reference
                )
                self._go_emit_listed_type_uses(
                    element.child_by_field_name("result"), source, add_reference
                )
            else:
                # Embedded interface (`io.Reader`): an inheritance edge.
                self._go_embedded(element, source, add_reference)

    def _go_exports(self, node: Node, source: bytes, add_reference: _ReferenceAdder) -> None:
        """Export rows for top-level declarations with capitalized names.

        Go exports by capitalization alone; lowercase declarations are
        package-private and must gain no export row. The `source_file`
        anchoring in go.scm keeps this to top-level declarations only --
        nested function literals are not matched here at all.
        """
        specs: list[tuple[Node | None, Node | None]] = []
        if node.type in {"method_declaration", "function_declaration"}:
            specs.append((node.child_by_field_name("name"), None))
        elif node.type == "type_declaration":
            for spec in node.named_children:
                if spec.type in {"type_spec", "type_alias"} and not spec.is_extra:
                    specs.append((spec.child_by_field_name("name"), spec))
        elif node.type in {"var_declaration", "const_declaration"}:
            for spec in node.named_children:
                if spec.is_extra:
                    continue
                if spec.type.endswith("_spec_list"):
                    # Grouped declarations (`var ( A = 1 )`): the grammar wraps
                    # `var` specs in a `var_spec_list`, while const/type groups
                    # keep their specs as direct named children. Unwrap so the
                    # grouped form exports exactly like the flat one.
                    for inner in spec.named_children:
                        if not inner.is_extra:
                            specs.append((inner.child_by_field_name("name"), inner))
                    continue
                specs.append((spec.child_by_field_name("name"), spec))
        for name_node, _spec in specs:
            if name_node is None:
                continue
            exported = _capture_name(source, name_node)
            if not exported[:1].isupper():
                continue
            add_reference("export", name_node, target_name=exported, written_name=exported)

    def _rust_records(self, node: Node, source: bytes, add_reference: _ReferenceAdder) -> None:
        if node.type == "use_declaration":
            self._rust_use(node, source, add_reference)
            self._rust_exports(node, source, add_reference)
            return
        if node.type == "call_expression":
            self._rust_call(node, source, add_reference)
            return
        if node.type == "field_expression":
            parent = node.parent
            if (
                parent is not None
                and parent.type == "call_expression"
                and parent.child_by_field_name("function") == node
            ):
                # The call row above already owns this span.
                return
            text = _capture_name(source, node)
            kind: ReferenceKind = (
                "write" if TreeSitterExtractor._is_assignment_target(node) else "read"
            )
            add_reference(kind, node, target_name=text, written_name=text)
            return
        if node.type == "impl_item":
            # `impl Draw for Widget`: the trait is an inheritance edge, the
            # self type a plain type use. The methods' `Type.method`
            # qualification happens in `_symbol_context`.
            self._rust_emit_type_uses(
                node.child_by_field_name("trait"), source, add_reference, kind="inheritance"
            )
            self._rust_emit_type_uses(node.child_by_field_name("type"), source, add_reference)
            return
        if node.type == "struct_expression":
            self._rust_emit_type_uses(node.child_by_field_name("name"), source, add_reference)
            return
        if node.type in {"function_item", "function_signature_item"}:
            self._rust_emit_type_uses(
                node.child_by_field_name("return_type"), source, add_reference
            )
            parameters = node.child_by_field_name("parameters")
            if parameters is not None:
                for child in parameters.named_children:
                    if child.type != "self_parameter" and not child.is_extra:
                        self._rust_emit_type_uses(
                            child.child_by_field_name("type"), source, add_reference
                        )
            self._rust_exports(node, source, add_reference)
            return
        if node.type in {"let_declaration", "field_declaration", "parameter"}:
            self._rust_emit_type_uses(node.child_by_field_name("type"), source, add_reference)
            return
        if node.type in {"const_item", "static_item"}:
            self._rust_emit_type_uses(node.child_by_field_name("type"), source, add_reference)
            self._rust_exports(node, source, add_reference)
            return
        if node.type == "enum_variant":
            self._rust_emit_type_uses(node.child_by_field_name("body"), source, add_reference)
            return
        self._rust_exports(node, source, add_reference)

    def _rust_use(self, node: Node, source: bytes, add_reference: _ReferenceAdder) -> None:
        argument = node.child_by_field_name("argument")
        if argument is None:
            return
        for anchor, module_path, imported_name, alias in self._rust_use_bindings(
            argument, source, []
        ):
            spelling = alias or imported_name
            # A `use` binds a whole path to one local spelling; the module
            # path minus its final segment is what module resolution anchors.
            add_reference(
                "import",
                anchor,
                target_name=spelling,
                written_name=spelling,
                module_path=module_path,
                imported_name=imported_name,
                alias=alias,
            )

    def _rust_use_bindings(
        self, node: Node, source: bytes, prefix: list[str]
    ) -> list[tuple[Node, str | None, str, str | None]]:
        """Flatten one `use` argument into `(anchor, module_path, name, alias)` rows.

        Nested groups expand one row per leaf (`use a::{b::C, d}`), `as`
        renames carry their alias, and globs carry `imported_name="*"` so the
        wildcard gate holds them unresolved.
        """
        bindings: list[tuple[Node, str | None, str, str | None]] = []
        if node.type == "scoped_identifier":
            segments = prefix + self._rust_path_segments(node, source)
            if len(segments) == 1:
                bindings.append((node, None, segments[0], None))
            elif segments:
                bindings.append((node, "::".join(segments[:-1]), segments[-1], None))
        elif node.type == "identifier":
            name = _capture_name(source, node)
            if prefix:
                bindings.append((node, "::".join(prefix), name, None))
            else:
                bindings.append((node, None, name, None))
        elif node.type == "use_as_clause":
            path = node.child_by_field_name("path")
            alias_node = node.child_by_field_name("alias")
            segments = prefix + self._rust_path_segments(path, source)
            if segments:
                module_path = "::".join(segments[:-1]) or None
                alias = _capture_name(source, alias_node) if alias_node is not None else None
                bindings.append((node, module_path, segments[-1], alias))
        elif node.type == "use_wildcard":
            inner = next((child for child in node.named_children if child.type != "*"), None)
            segments = prefix + self._rust_path_segments(inner, source)
            if segments:
                bindings.append((node, "::".join(segments), "*", None))
        elif node.type == "scoped_use_list":
            path = node.child_by_field_name("path")
            use_list = node.child_by_field_name("list")
            for binding in self._rust_use_bindings_list(
                use_list, source, prefix + self._rust_path_segments(path, source)
            ):
                bindings.append(binding)
        elif node.type == "use_list":
            for binding in self._rust_use_bindings_list(node, source, prefix):
                bindings.append(binding)
        return bindings

    def _rust_use_bindings_list(
        self, node: Node | None, source: bytes, prefix: list[str]
    ) -> list[tuple[Node, str | None, str, str | None]]:
        bindings: list[tuple[Node, str | None, str, str | None]] = []
        if node is None:
            return bindings
        for child in node.named_children:
            if child.is_extra:
                continue
            bindings.extend(self._rust_use_bindings(child, source, prefix))
        return bindings

    def _rust_call(self, node: Node, source: bytes, add_reference: _ReferenceAdder) -> None:
        function = node.child_by_field_name("function")
        if function is None:
            return
        if function.type == "identifier":
            name = _capture_name(source, function)
            add_reference(
                "call",
                function,
                target_name=name,
                written_name=name,
                call_shape=self._call_shape(node),
            )
        elif function.type == "scoped_identifier":
            segments = self._rust_path_segments(function, source)
            if not segments:
                return
            # Rows join path segments with dots (Python/Go convention) so the
            # written-name tail math and rename trimming keep one shape; the
            # receiver keeps its `::` spelling and resolves only through an
            # explicit import, which is exactly the honest bar here.
            add_reference(
                "call",
                function,
                target_name=".".join(segments),
                written_name=".".join(segments),
                receiver_text="::".join(segments[:-1]) or None,
                call_shape=self._call_shape(node),
            )
        elif function.type == "field_expression":
            receiver = function.child_by_field_name("value")
            add_reference(
                "call",
                function,
                target_name=_capture_name(source, function),
                written_name=_capture_name(source, function),
                receiver_text=(_capture_name(source, receiver) if receiver is not None else None),
                call_shape=self._call_shape(node),
            )

    def _rust_exports(self, node: Node, source: bytes, add_reference: _ReferenceAdder) -> None:
        """Export rows for `pub` items; `pub use` rows carry the module path.

        The rust.scm export captures are `source_file`-anchored, so nested
        impl methods and module-private items never reach this. `pub use`
        exports mirror their import rows (module path, imported name, alias)
        so the re-export chain walker can hop through them unchanged.
        """
        if not any(child.type == "visibility_modifier" for child in node.children):
            return
        if node.type == "use_declaration":
            argument = node.child_by_field_name("argument")
            if argument is None:
                return
            for anchor, module_path, imported_name, alias in self._rust_use_bindings(
                argument, source, []
            ):
                spelling = alias or imported_name
                add_reference(
                    "export",
                    anchor,
                    target_name=spelling,
                    written_name=spelling,
                    module_path=module_path,
                    imported_name=imported_name,
                    alias=alias,
                )
            return
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        exported = _capture_name(source, name_node)
        add_reference("export", name_node, target_name=exported, written_name=exported)

    def _rust_emit_type_uses(
        self,
        node: Node | None,
        source: bytes,
        add_reference: _ReferenceAdder,
        kind: ReferenceKind = "type_use",
    ) -> None:
        for leaf in TreeSitterExtractor._rust_descend_type_names(node):
            name = _capture_name(source, leaf)
            add_reference(kind, leaf, target_name=name, written_name=name)

    @staticmethod
    def _rust_path_segments(node: Node | None, source: bytes) -> list[str]:
        """Flatten a `::` path into its spelling segments.

        `crate`/`super`/`self` are keyword tokens, not identifiers, and a
        turbofish segment (`Vec::<u8>::new`) hides the head type behind a
        `generic_type` -- its type arguments are type parameters, not path
        segments.
        """
        if node is None:
            return []
        if node.type in {"identifier", "type_identifier"}:
            return [_capture_name(source, node)]
        if node.type in {"crate", "super", "self"}:
            return [node.type]
        if node.type == "scoped_identifier":
            segments = TreeSitterExtractor._rust_path_segments(
                node.child_by_field_name("path"), source
            )
            name = node.child_by_field_name("name")
            if name is not None:
                segments.append(_capture_name(source, name))
            return segments
        if node.type == "generic_type":
            return TreeSitterExtractor._rust_path_segments(node.child_by_field_name("type"), source)
        return []

    @staticmethod
    def _rust_descend_type_names(node: Node | None) -> list[Node]:
        """Descend a Rust type expression to its naming `type_identifier` leaves.

        Unwraps reference/pointer/slice/array/tuple/parenthesized wrappers,
        `generic_type` (head plus one entry per type argument), `dyn` traits,
        and `scoped_type_identifier` down to leaf names -- a qualified leaf
        contributes only its final identifier, the path being a namespace
        spelling. Primitive types yield nothing: they can never be project
        declarations.
        """
        if node is None:
            return []
        if node.type == "type_identifier":
            return [node]
        if node.type == "scoped_type_identifier":
            return TreeSitterExtractor._rust_descend_type_names(node.child_by_field_name("name"))
        if node.type in {
            "reference_type",
            "pointer_type",
            "slice_type",
            "array_type",
            "tuple_type",
            "parenthesized_type",
            "generic_type",
            "dynamic_type",
            "impl_trait_type",
            "type_arguments",
            "ordered_field_declaration_list",
        }:
            names: list[Node] = []
            for child in node.named_children:
                names.extend(TreeSitterExtractor._rust_descend_type_names(child))
            return names
        return []

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
            elif parent.type == "impl_item":
                # Rust methods live in impl blocks, which are naming scopes but
                # deliberately not chunked definitions (an `impl_item` capture
                # would duplicate the self type's `struct` chunk). Synthesizing
                # a container named by the self type qualifies `fn draw` inside
                # `impl Runner for Widget` as `Widget.draw`, so `self.draw()`
                # resolves exactly through `_same_owner` like Python/TS.
                name = TreeSitterExtractor._impl_self_type_name(parent)
                if name is not None:
                    chain.append(_Definition(parent, "struct", name))
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
    def _impl_self_type_name(impl: Node) -> str | None:
        type_node = impl.child_by_field_name("type")
        if type_node is None:
            return None
        if type_node.type == "type_identifier":
            return (type_node.text or b"").decode("utf-8", errors="replace")
        # A generic self type (`impl<T> Pool<T>`) names its head type.
        current: Node | None = type_node
        while current is not None:
            if current.type == "type_identifier":
                return (current.text or b"").decode("utf-8", errors="replace")
            current = current.named_children[0] if current.named_children else None
        return None

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
