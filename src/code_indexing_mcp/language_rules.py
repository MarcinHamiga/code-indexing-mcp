"""Per-language rules table for structural extraction and reference resolution.

Consolidates language-specific node-type sets, reserved words, identifier validation,
and import candidate resolution rules so adding a new structural language requires
only adding a table row and its handlers.
"""

from __future__ import annotations

import keyword as _keyword_module
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Final, NamedTuple


class _ModuleIndex(NamedTuple):
    """Path arithmetic over one query's snapshot, precomputed instead of rescanned.

    The lookups the language resolvers need repeatedly:
    `directories_by_suffix` maps an import path's segment tuple to every known
    directory whose trailing segments equal it (module-prefix agnostic -- the
    shape Go package imports resolve against without reading go.mod),
    `files_by_directory` lists the files indexed directly inside each such
    directory (Go packages span files, so one directory is many candidates),
    and `receiver_names` maps `(file_id, enclosing qualified symbol)` to its
    first parameter name for method declarations only (Go methods put their
    receiver there; a plain function's first parameter is not a receiver),
    which powers the Go receiver-name rule. `rust_crate_roots` maps every
    Rust directory to the nearest ancestor containing a `lib.rs`/`main.rs`,
    which anchors `crate::` paths without reading Cargo.toml. Python and
    JavaScript resolution ignore the index entirely; only callers whose
    language has a directory-suffix or receiver rule consult it.
    """

    directories_by_suffix: dict[tuple[str, ...], tuple[str, ...]]
    files_by_directory: dict[str, tuple[str, ...]]
    java_files_by_directory: dict[str, tuple[str, ...]]
    namespace_by_path: dict[str, str]
    files_by_namespace: dict[str, tuple[str, ...]]
    names_by_namespace: dict[str, frozenset[str]]
    receiver_names: dict[tuple[str, str], str]
    rust_crate_roots: dict[str, str]


# Reserved words per language family.
_ECMASCRIPT_RESERVED_WORDS: Final = frozenset(
    {
        "await",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "continue",
        "debugger",
        "default",
        "delete",
        "do",
        "else",
        "enum",
        "export",
        "extends",
        "false",
        "finally",
        "for",
        "function",
        "if",
        "implements",
        "import",
        "in",
        "instanceof",
        "interface",
        "let",
        "new",
        "null",
        "package",
        "private",
        "protected",
        "public",
        "return",
        "static",
        "super",
        "switch",
        "this",
        "throw",
        "true",
        "try",
        "typeof",
        "var",
        "void",
        "while",
        "with",
        "yield",
    }
)

_GO_RESERVED_WORDS: Final = frozenset(
    {
        "break",
        "case",
        "chan",
        "const",
        "continue",
        "default",
        "defer",
        "else",
        "fallthrough",
        "for",
        "func",
        "go",
        "goto",
        "if",
        "import",
        "interface",
        "map",
        "package",
        "range",
        "return",
        "select",
        "struct",
        "switch",
        "type",
        "var",
    }
)

_RUST_RESERVED_WORDS: Final = frozenset(
    {
        "Self",
        "abstract",
        "as",
        "async",
        "await",
        "become",
        "box",
        "break",
        "const",
        "continue",
        "crate",
        "do",
        "dyn",
        "else",
        "enum",
        "extern",
        "false",
        "final",
        "fn",
        "for",
        "gen",
        "if",
        "impl",
        "in",
        "let",
        "loop",
        "macro",
        "match",
        "mod",
        "move",
        "mut",
        "override",
        "priv",
        "pub",
        "ref",
        "return",
        "self",
        "static",
        "struct",
        "super",
        "trait",
        "true",
        "try",
        "type",
        "typeof",
        "union",
        "unsafe",
        "unsized",
        "use",
        "virtual",
        "where",
        "while",
        "yield",
    }
)

_JAVA_RESERVED_WORDS: Final = frozenset(
    {
        "_",
        "abstract",
        "assert",
        "boolean",
        "break",
        "byte",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extends",
        "false",
        "final",
        "finally",
        "float",
        "for",
        "goto",
        "if",
        "implements",
        "import",
        "instanceof",
        "int",
        "interface",
        "long",
        "native",
        "new",
        "null",
        "package",
        "private",
        "protected",
        "public",
        "return",
        "short",
        "static",
        "strictfp",
        "super",
        "switch",
        "synchronized",
        "this",
        "throw",
        "throws",
        "transient",
        "true",
        "try",
        "void",
        "volatile",
        "while",
    }
)

_CSHARP_RESERVED_WORDS: Final = frozenset(
    {
        "abstract",
        "as",
        "base",
        "bool",
        "break",
        "byte",
        "case",
        "catch",
        "char",
        "checked",
        "class",
        "const",
        "continue",
        "decimal",
        "default",
        "delegate",
        "do",
        "double",
        "else",
        "enum",
        "event",
        "explicit",
        "extern",
        "false",
        "finally",
        "fixed",
        "float",
        "for",
        "foreach",
        "goto",
        "if",
        "implicit",
        "in",
        "int",
        "interface",
        "internal",
        "is",
        "lock",
        "long",
        "namespace",
        "new",
        "null",
        "object",
        "operator",
        "out",
        "override",
        "params",
        "private",
        "protected",
        "public",
        "readonly",
        "ref",
        "return",
        "sbyte",
        "sealed",
        "short",
        "sizeof",
        "stackalloc",
        "static",
        "string",
        "struct",
        "switch",
        "this",
        "throw",
        "true",
        "try",
        "typeof",
        "uint",
        "ulong",
        "unchecked",
        "unsafe",
        "ushort",
        "using",
        "virtual",
        "void",
        "volatile",
        "while",
    }
)

_PYTHON_RESERVED_WORDS: Final = frozenset(_keyword_module.kwlist)


def _csharp_identifier_valid(name: str) -> bool:
    body = name[1:] if name.startswith("@") else name
    return body.isidentifier() and (name.startswith("@") or body not in _CSHARP_RESERVED_WORDS)


def _python_package_root(directory: PurePosixPath, known_paths: frozenset[str]) -> PurePosixPath:
    """The directory an absolute import from a file in `directory` resolves against."""
    parts = list(directory.parts)
    if not parts or str(PurePosixPath(*parts, "__init__.py")) not in known_paths:
        return PurePosixPath()
    boundary = len(parts)
    while boundary > 0 and str(PurePosixPath(*parts[:boundary], "__init__.py")) in known_paths:
        boundary -= 1
    return PurePosixPath(*parts[:boundary])


def _go_import_candidates(
    source: PurePosixPath,
    module_path: str,
    known_paths: frozenset[str],
    module_index: _ModuleIndex | None,
) -> set[PurePosixPath]:
    if module_index is None:
        return set()
    segments = tuple(part for part in PurePosixPath(module_path).parts if part != ".")
    directories = module_index.directories_by_suffix.get(segments, ())
    candidates: set[PurePosixPath] = set()
    for directory in directories:
        for file_path in module_index.files_by_directory.get(directory, ()):
            candidates.add(PurePosixPath(file_path))
    return candidates


def _rust_import_candidates(
    source: PurePosixPath,
    module_path: str,
    known_paths: frozenset[str],
    module_index: _ModuleIndex | None,
) -> set[PurePosixPath]:
    if module_index is None:
        return set()
    parts = [part for part in module_path.split("::") if part and part != "."]
    if not parts:
        return set()
    source_directory = str(source.parent)
    crate_root = module_index.rust_crate_roots.get(source_directory)

    def under(base: str, tail: list[str]) -> set[PurePosixPath]:
        if not tail:
            return set()
        module = "/".join(tail)
        return {
            PurePosixPath(f"{base}/{module}.rs"),
            PurePosixPath(f"{base}/{module}/mod.rs"),
        }

    head = parts[0]
    if head == "crate":
        return under(crate_root, parts[1:]) if crate_root else set()
    if head in {"self", "super"}:
        base = PurePosixPath(source_directory)
        rest = list(parts)
        while rest and rest[0] in {"self", "super"}:
            keyword = rest.pop(0)
            if keyword == "super":
                if len(base.parts) <= 1:
                    return set()
                base = base.parent
        return under(str(base), rest)
    directory_candidates = under(source_directory, parts)
    root_candidates = under(crate_root, parts) if crate_root else set()
    if directory_candidates and root_candidates:
        return directory_candidates & root_candidates
    return directory_candidates | root_candidates


def _java_import_candidates(
    source: PurePosixPath,
    module_path: str,
    known_paths: frozenset[str],
    module_index: _ModuleIndex | None,
) -> set[PurePosixPath]:
    if module_index is None:
        return set()
    segments = tuple(part for part in module_path.split(".") if part and part != ".")
    if len(segments) < 2:
        return set()
    package, type_stem = segments[:-1], segments[-1]
    java_candidates: set[PurePosixPath] = set()
    for directory in module_index.directories_by_suffix.get(package, ()):
        candidate = f"{directory}/{type_stem}.java"
        if candidate in known_paths:
            java_candidates.add(PurePosixPath(candidate))
    return java_candidates


def _csharp_import_candidates(
    source: PurePosixPath,
    module_path: str,
    known_paths: frozenset[str],
    module_index: _ModuleIndex | None,
) -> set[PurePosixPath]:
    if module_index is None:
        return set()
    namespace_candidates: set[PurePosixPath] = {
        PurePosixPath(path) for path in module_index.files_by_namespace.get(module_path, ())
    }
    parts = [part for part in module_path.split(".") if part]
    if len(parts) >= 2:
        namespace = ".".join(parts[:-1])
        tail = parts[-1]
        if tail in module_index.names_by_namespace.get(namespace, frozenset()):
            namespace_candidates.update(
                PurePosixPath(path) for path in module_index.files_by_namespace.get(namespace, ())
            )
    return namespace_candidates


def _python_import_candidates(
    source: PurePosixPath,
    module_path: str,
    known_paths: frozenset[str],
    module_index: _ModuleIndex | None,
) -> set[PurePosixPath]:
    dots = len(module_path) - len(module_path.lstrip("."))
    suffix = module_path[dots:]
    stem = PurePosixPath(*suffix.split(".")) if suffix else PurePosixPath()
    if dots == 0:
        base = _python_package_root(source.parent, known_paths)
    else:
        base = source.parent
        for _ in range(dots - 1):
            base = base.parent
    return {base / f"{stem}.py", base / stem / "__init__.py"}


def _ecmascript_import_candidates(
    source: PurePosixPath,
    module_path: str,
    known_paths: frozenset[str],
    module_index: _ModuleIndex | None,
) -> set[PurePosixPath]:
    if not module_path.startswith("."):
        return set()
    parts = list(source.parent.parts)
    for part in PurePosixPath(module_path).parts:
        if part == ".":
            continue
        if part == "..":
            if not parts:
                return set()
            parts.pop()
            continue
        parts.append(part)
    normalized = PurePosixPath(*parts)
    extensions = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts")
    candidates = {PurePosixPath(f"{normalized}{extension}") for extension in extensions}
    candidates.update(normalized / f"index{extension}" for extension in extensions)
    return candidates


def _empty_import_candidates(
    source: PurePosixPath,
    module_path: str,
    known_paths: frozenset[str],
    module_index: _ModuleIndex | None,
) -> set[PurePosixPath]:
    return set()


_ImportCandidatesFn = Callable[
    [PurePosixPath, str, frozenset[str], _ModuleIndex | None], set[PurePosixPath]
]


@dataclass(frozen=True)
class _LanguageRules:
    """Consolidated per-language extraction and resolution rules."""

    import_owner_parents: frozenset[str] = field(default_factory=frozenset)
    method_name_field_excluded: bool = False
    name_and_type_parents: frozenset[str] = field(default_factory=frozenset)
    name_and_field_parents: frozenset[str] = field(default_factory=frozenset)
    name_only_parents: frozenset[str] = field(default_factory=frozenset)
    type_only_parents: frozenset[str] = field(default_factory=frozenset)
    parameters_parents: frozenset[str] = field(default_factory=frozenset)
    left_and_type_parents: frozenset[str] = field(default_factory=frozenset)
    function_and_type_parents: frozenset[str] = field(default_factory=frozenset)
    pair_parents: frozenset[str] = field(default_factory=frozenset)
    handler_owned_type_parents: frozenset[str] = field(default_factory=frozenset)
    keyword_only_marker: str | None = None
    variadic_is_optional: bool = True
    reserved_words: frozenset[str] = field(default_factory=frozenset)
    identifier_valid: Callable[[str], bool] = lambda _: False
    bound_receivers: frozenset[str] = field(default_factory=frozenset)
    import_candidates: _ImportCandidatesFn = _empty_import_candidates


_DEFAULT: Final[_LanguageRules] = _LanguageRules()

_SHARED_PAIR_PARENTS: Final[frozenset[str]] = frozenset({"pair", "pair_pattern", "keyed_element"})

_GO_RUST_HANDLER_OWNED_TYPE_PARENTS: Final[frozenset[str]] = frozenset(
    {
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
    }
)

LANGUAGE_RULES: Final[Mapping[str, _LanguageRules]] = {
    "python": _LanguageRules(
        keyword_only_marker="list_splat_pattern",
        variadic_is_optional=False,
        reserved_words=_PYTHON_RESERVED_WORDS,
        identifier_valid=lambda name: name.isidentifier() and not _keyword_module.iskeyword(name),
        bound_receivers=frozenset({"self", "cls"}),
        import_candidates=_python_import_candidates,
    ),
    "javascript": _LanguageRules(
        pair_parents=_SHARED_PAIR_PARENTS,
        reserved_words=_ECMASCRIPT_RESERVED_WORDS,
        identifier_valid=lambda name: (
            name.replace("$", "_").isidentifier() and (name not in _ECMASCRIPT_RESERVED_WORDS)
        ),
        import_candidates=_ecmascript_import_candidates,
    ),
    "typescript": _LanguageRules(
        pair_parents=_SHARED_PAIR_PARENTS,
        reserved_words=_ECMASCRIPT_RESERVED_WORDS,
        identifier_valid=lambda name: (
            name.replace("$", "_").isidentifier() and (name not in _ECMASCRIPT_RESERVED_WORDS)
        ),
        import_candidates=_ecmascript_import_candidates,
    ),
    "tsx": _LanguageRules(
        pair_parents=_SHARED_PAIR_PARENTS,
        reserved_words=_ECMASCRIPT_RESERVED_WORDS,
        identifier_valid=lambda name: (
            name.replace("$", "_").isidentifier() and (name not in _ECMASCRIPT_RESERVED_WORDS)
        ),
        import_candidates=_ecmascript_import_candidates,
    ),
    "go": _LanguageRules(
        pair_parents=_SHARED_PAIR_PARENTS,
        handler_owned_type_parents=_GO_RUST_HANDLER_OWNED_TYPE_PARENTS,
        reserved_words=_GO_RESERVED_WORDS,
        identifier_valid=lambda name: name.isidentifier() and name not in _GO_RESERVED_WORDS,
        import_candidates=_go_import_candidates,
    ),
    "rust": _LanguageRules(
        pair_parents=_SHARED_PAIR_PARENTS,
        handler_owned_type_parents=_GO_RUST_HANDLER_OWNED_TYPE_PARENTS,
        reserved_words=_RUST_RESERVED_WORDS,
        identifier_valid=lambda name: name.isidentifier() and name not in _RUST_RESERVED_WORDS,
        import_candidates=_rust_import_candidates,
    ),
    "java": _LanguageRules(
        import_owner_parents=frozenset(
            {
                "import_declaration",
                "package_declaration",
                "marker_annotation",
                "annotation",
            }
        ),
        name_and_type_parents=frozenset(
            {
                "record_declaration",
                "enum_declaration",
                "annotation_type_declaration",
                "constructor_declaration",
                "compact_constructor_declaration",
                "annotation_type_element_declaration",
                "enum_constant",
                "local_variable_declaration",
                "constant_declaration",
                "enhanced_for_statement",
            }
        ),
        name_and_field_parents=frozenset(
            {
                "method_invocation",
                "field_access",
                "catch_formal_parameter",
                "instanceof_expression",
            }
        ),
        type_only_parents=frozenset({"object_creation_expression"}),
        parameters_parents=frozenset({"lambda_expression"}),
        pair_parents=_SHARED_PAIR_PARENTS,
        handler_owned_type_parents=frozenset(
            {
                "superclass",
                "type_list",
                "throws",
                "catch_type",
                "type_parameter",
                "type_bound",
                "spread_parameter",
            }
        ),
        reserved_words=_JAVA_RESERVED_WORDS,
        identifier_valid=lambda name: name.isidentifier() and name not in _JAVA_RESERVED_WORDS,
        import_candidates=_java_import_candidates,
    ),
    "csharp": _LanguageRules(
        import_owner_parents=frozenset({"using_directive", "attribute"}),
        method_name_field_excluded=True,
        name_and_type_parents=frozenset(
            {
                "record_declaration",
                "struct_declaration",
                "enum_declaration",
                "constructor_declaration",
                "property_declaration",
                "enum_member_declaration",
                "namespace_declaration",
                "delegate_declaration",
                "variable_declaration",
                "catch_declaration",
            }
        ),
        name_only_parents=frozenset(
            {
                "member_access_expression",
                "declaration_expression",
            }
        ),
        type_only_parents=frozenset({"cast_expression"}),
        left_and_type_parents=frozenset({"foreach_statement"}),
        function_and_type_parents=frozenset(
            {
                "invocation_expression",
                "object_creation_expression",
            }
        ),
        pair_parents=_SHARED_PAIR_PARENTS,
        handler_owned_type_parents=frozenset(
            {
                "base_list",
                "generic_name",
                "qualified_name",
                "type_argument_list",
                "array_type",
                "nullable_type",
                "type_parameter_constraints_clause",
            }
        ),
        reserved_words=_CSHARP_RESERVED_WORDS,
        identifier_valid=_csharp_identifier_valid,
        import_candidates=_csharp_import_candidates,
    ),
}
