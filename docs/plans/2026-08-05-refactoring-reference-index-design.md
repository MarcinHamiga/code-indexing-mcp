# Refactoring Reference Index Design

## Context

Code Indexing MCP currently indexes syntax-aware declaration and module chunks and exposes hybrid
search, declaration lookup, file outlines, and full chunk retrieval. The stored model records a
symbol's qualified name and parent, but it does not record relationships between declarations and
their uses. `find_symbol` therefore answers where a declaration is, not which imports, calls, type
uses, or overrides depend on it.

That boundary limits an agent preparing a rename, signature change, deletion, or module move. The
agent can search for matching text, but it cannot distinguish a direct imported alias from an
unrelated local name, or an exact call from a member access on an unknown receiver.

This design adds a structural reference index. The first release targets Python, JavaScript,
TypeScript, and TSX, and focuses on rename and signature-change analysis. Safe deletion and module
moves reuse the same data in later releases.

## Goals

- Find evidence-backed references to one uniquely selected declaration.
- Classify references as exact, likely, or unresolved instead of hiding ambiguity.
- Produce an edit checklist for renames and signature changes.
- Preserve exact byte and line ranges so an agent can make targeted edits.
- Keep analysis local, offline, incremental, and independent of language toolchains.
- Refresh references transactionally with the chunks for a changed file.
- Expose incomplete coverage and dynamic-language limitations prominently.
- Backfill references for an existing index without recomputing unchanged embeddings.

## Non-goals

- Editing source files or applying a refactor automatically.
- Full compiler or language-server type checking.
- Claiming complete resolution of dynamic dispatch, reflection, computed properties, or runtime
  imports.
- A general-purpose control-flow or runtime call graph.
- Languages other than Python, JavaScript, TypeScript, and TSX in the first release.
- Cross-project reference resolution in the first release. A later module-move phase will add an
  explicit multi-project scope rather than searching unrelated registrations implicitly.

## Considered approaches

### Persistent structural occurrences — selected

Extract reference occurrences with Tree-sitter while a file is already being parsed. Persist the
syntax facts separately from embedding-backed chunks and resolve them against the current symbol
and import catalogues at query time.

This reuses the existing incremental scanner, extractor, staging, and per-project storage model.
It also allows a conservative resolver: syntax that cannot identify one declaration remains visible
as likely or unresolved evidence.

### Compiler and language-server adapters

Pyright, TypeScript Server, and comparable language-specific services can resolve more overloads
and inferred receiver types. They would also introduce separate processes, toolchains, project
configuration behavior, and version compatibility for each language. That conflicts with the
server's current toolchain-free installation and would make accuracy depend on external workspace
state. Compiler adapters may become optional providers later, but they are not the baseline.

### On-demand textual analysis

The server could search an identifier and inspect nearby syntax only when asked. This is small to
implement, but it repeats work, cannot page over a stable result set, and performs poorly for
aliases, re-exports, shadowing, overloads, and common member names. It is useful as an explicit
unresolved-reference fallback, not as the primary index.

## Architecture

Indexing produces two coordinated outputs:

1. Existing semantic code chunks, including declaration metadata and embedding vectors.
2. Structural `ReferenceOccurrence` rows, which never receive embeddings.

The Tree-sitter extractor gains language-specific reference queries and returns occurrences beside
its existing chunks. The indexer stages files, chunks, and occurrences together. A normal changed-
file commit replaces that file's chunk and occurrence rows atomically, so a successful refresh
cannot expose new declarations with old references or the reverse.

A new `ReferenceResolver` service reads the selected declaration, symbol catalogue, import/export
bindings, and candidate occurrences. It resolves what can be proven structurally and assigns a
resolution level and reason to everything else. Resolution happens at query time so changing a
declaration or import does not require rewriting reference rows from otherwise unchanged files.

The service layer exposes reference lookup and refactor analysis to the application. The MCP layer
uses the same lazy freshness check as current project-scoped code queries before invoking either
operation.

```text
source file
    |
    v
Tree-sitter extraction -----> semantic chunks -----> chunks table
    |                              |
    +-----> occurrences ----------+---------------> references table
                                                   |
selected declaration ---> ReferenceResolver <------+-- symbol/import catalogues
                               |
                               +--> find_references
                               +--> analyze_refactor
```

## Stored data

### Reference occurrence

`ReferenceOccurrence` records source syntax without asserting that the target has been resolved.
Its core fields are:

- `reference_id`: content-derived identifier.
- `file_id`, `project_id`, `path`, and `language`.
- `source_chunk_id` and `source_qualified_symbol`, when the occurrence has an enclosing chunk or
  declaration.
- `kind`: `import`, `export`, `call`, `type_use`, `inheritance`, `decorator`, `read`, or `write`.
- Exact start/end bytes and lines.
- `written_name`: the source spelling.
- `target_name`: the normalized name used to locate candidates.
- Optional `module_path`, `imported_name`, `alias`, and `receiver_text` syntax facts.
- Optional call shape.
- The containing file content hash and occurrence schema version.

The row does not persist a final target declaration ID. A target resolved today may become
ambiguous after another file adds an overload or changes an export. Persisting the syntax facts and
resolving against the current catalogue avoids maintaining a second dependency graph whenever the
catalogue changes.

### Call shape

Calls store only facts required for signature analysis:

- Positional argument count.
- Keyword or named-argument names.
- Whether positional or keyword spread is present.
- Type-argument count when syntax exposes it.
- Whether the call is a constructor or ordinary invocation.

Declarations receive a corresponding parameter shape: name, order, positional-only, keyword-only,
required/defaulted, and variadic status. JavaScript and TypeScript default/rest parameters map onto
the same representation. An overload set remains multiple shapes rather than being flattened into
one misleading signature.

### Storage and indexes

Each project receives a references table next to its files and chunks tables. It is indexed for
exact filters on file, normalized target name, module path, kind, and containing symbol. Reference
queries do not use vector search.

Table versions are included in analysis cursors. A cursor therefore continues against the same
snapshot even if a later refresh publishes a new table version. Expired or unavailable snapshots
return a structured stale-cursor error rather than mixing pages from different indexes.

## Language extraction and resolution

### Python

The first resolver handles:

- `import module as alias` and `from module import name as alias`.
- Direct calls through local declarations and imported bindings.
- Qualified calls through imported module aliases.
- Lexical shadowing in modules, classes, functions, and nested functions.
- `self` and `cls` references to methods on the enclosing class.
- Base classes, decorators, and type annotations.
- Positional, keyword, `*args`, and `**kwargs` call shapes.

Computed attribute names, monkey-patching, wildcard imports, reflection, and calls through values of
unknown type remain likely or unresolved with an explicit reason.

### JavaScript, TypeScript, and TSX

The first resolver handles:

- Default, named, aliased, and namespace imports.
- Relative module paths and ordinary index-file resolution.
- Direct exports and common re-export forms.
- Lexical shadowing in module, class, and function scopes.
- Calls through local and imported bindings.
- Member calls where the receiver is a known namespace import or imported binding.
- Classes, inheritance, decorators, type references, and generic argument counts.
- Positional, default, rest, and spread arguments.

Calls on inferred object types, prototype mutation, computed property names, dynamic imports, and
framework-provided dependency injection remain likely or unresolved. The resolver does not imitate
the TypeScript compiler.

## Resolution levels

Every returned use has one of three levels:

- `exact`: the structural scope and import/export chain select the requested declaration uniquely.
- `likely`: the written name and context are compatible, but receiver type or dispatch cannot be
  proven without semantic type information.
- `unresolved`: the syntax may be relevant, but the server cannot select a target or rule it out.

Each result includes a short machine-stable reason code and a human-readable explanation. Examples
include `direct_import_alias`, `lexical_binding`, `known_namespace_member`,
`unknown_receiver_member`, `wildcard_import`, and `dynamic_spread`.

The resolver must prefer a lower confidence classification over a guessed exact match. Exactness is
an assertion agents may act on; likely and unresolved entries are prompts for review.

## MCP tools

### `find_references`

The tool accepts either a declaration `chunk_id` or a declaration selector containing `project`,
`path`, and `qualified_symbol`. A bare symbol name may be supplied for convenience, but analysis
continues only when it resolves to one declaration. Ambiguous selection returns candidates with
paths and ranges.

Optional filters select reference kinds and the minimum resolution level. Results contain:

- `reference_id` and containing `chunk_id` when available.
- Project, path, language, and exact source range.
- Reference kind and compact snippet.
- Resolution level, reason code, and explanation.
- The selected declaration and index snapshot identifier.

The response is paginated. Its cursor binds the declaration, filters, and table snapshot. Full
source context remains available through `get_chunk` rather than being duplicated into every
reference response.

### `analyze_refactor`

The tool accepts the same declaration selector plus a discriminated operation:

- `rename`, with `new_name`.
- `signature_change`, with the proposed structured parameter list.

It returns the selected declaration, a summary of the proposed change, paginated findings, counts,
and a completeness report. Findings are grouped by required action:

- `must_change`: exact references whose syntax is directly affected.
- `likely_change`: probable references that need inspection or an edit.
- `review`: unresolved references, dynamic calls, overloads, and other limitations.

Rename analysis covers declarations, import/export bindings, direct calls, type uses, inheritance,
decorators, and qualified member uses. It does not suggest renaming unrelated local aliases unless
the alias itself is the selected declaration.

Signature analysis compares indexed call shapes with the proposed declaration. It identifies
renamed or removed keyword parameters, new required parameters, reordering that affects positional
calls, changes between positional-only and keyword-only parameters, rest/variadic changes, and
calls obscured by spread arguments. When overload resolution is ambiguous, the call is placed in
`review` rather than matched to an arbitrary overload.

Both tools are read operations from the caller's perspective, but they retain the current MCP
annotation used by project-scoped queries because freshness checks may register or refresh a
project.

## Completeness and errors

The API does not return a synthetic `safe: true` value. It returns a completeness state:

- `complete`: all eligible first-release files were indexed successfully and every candidate was
  resolved or surfaced.
- `complete_with_dynamic_limitations`: index coverage is complete, but language behavior leaves
  likely or unresolved uses.
- `incomplete`: one or more relevant files, imports, snapshots, or language constructs could not be
  analyzed.

A top-level `limitations` collection records partial indexes, file parse failures, unsupported
languages, unresolved imports, wildcard or dynamic imports, dynamic dispatch, spread arguments,
and truncated or expired pagination.

Expected structured errors include:

- Ambiguous or missing declaration, with candidates when available.
- Unsupported declaration language.
- Invalid rename or signature-change specification.
- Reference backfill unavailable or still running.
- Snapshot cursor expired.
- The existing index-busy and project-resolution failures from automatic freshness checks.

Parse or extraction failure for one changed file preserves its previous committed chunks and
references. The project remains partial, and refactor analysis reports that previous data may be
stale rather than silently treating it as current evidence.

## Existing-index backfill

The project metadata gains an occurrence schema version. When an existing project has current
embeddings but no compatible references, it becomes structurally stale while ordinary semantic
search remains usable.

Backfill reads and parses eligible source files, stages occurrence rows, and commits the references
table without embedding unchanged chunks. It also verifies that each file still matches the stored
content hash. A file that changed follows the normal indexing path so its files, chunks, vectors,
and occurrences advance together.

The backfill publishes its table and schema version only after a successful project-level commit.
An interrupted backfill leaves the previous semantic index searchable and causes reference tools
to retry or report that structural indexing is unavailable. Progress notifications distinguish
reference extraction from embedding so clients can explain the wait accurately.

## Testing strategy

### Extraction fixtures

Language-specific fixtures assert exact ranges, kinds, enclosing symbols, import/export facts, and
call shapes. They cover aliases, nested scopes, shadowing, re-exports, decorators, inheritance,
keyword or named arguments, rest parameters, spreads, and syntax-error recovery.

### Resolver corpus

Small manually annotated repositories contain exact, likely, and deliberately ambiguous cases.
The curated corpus requires zero false positives in the `exact` category. New constructs start as
likely or unresolved until fixtures prove that exact resolution is sound.

### Storage and migration

Tests cover changed, renamed, removed, and failed files; transactional replacement and rollback;
parse-only backfill; mismatched content hashes; occurrence schema upgrades; table snapshot cursors;
and recovery after an interrupted commit.

### MCP contracts

Contract tests cover unique and ambiguous selectors, filters, stable pagination, invalid operation
schemas, stale and partial indexes, limitation propagation, and tool annotations. Responses must
stay compact enough that agents can page rather than receiving an unbounded reference dump.

### End-to-end refactors

Fixture repositories define known rename and signature changes with expected classifications and
edit locations. Tests invoke the real MCP service and compare the resulting checklist, including
the limitations that prevent a complete verdict.

### Performance

Benchmarks record occurrence extraction time, reference-row count, disk growth, cold-index time,
and changed-file refresh separately from embedding. The initial acceptance gate allows at most 25%
additional cold-index time on the deterministic benchmark corpus. Incremental work must remain
proportional to changed files, and reference extraction must never invoke the embedding backend.

## Delivery sequence

1. Add occurrence models, Tree-sitter queries, storage, staging, and parse-only backfill.
2. Add Python and JavaScript/TypeScript/TSX symbol/import catalogues and conservative resolution.
3. Expose `find_references` with stable pagination and completeness reporting.
4. Add rename analysis to `analyze_refactor`.
5. Add declaration parameter shapes and signature-change analysis.
6. Run the accuracy corpus and performance gates, then update the bundled impact-analysis and
   feature-development skills to prefer the new tools.

Each step is independently testable. The MCP tools do not ship until extraction, migration, and
resolver behavior are covered end to end.

## Later phases

### Safe deletion

Deletion analysis asks whether exact, likely, or unresolved reverse references remain. It also
checks exports, subclassing, decorators, and framework registration patterns. A symbol is reported
as unreferenced only within an explicit coverage scope; dynamic limitations remain visible.

### Module moves

Move analysis adds module identity and import/export path rewriting. It reports affected relative
imports, public re-exports, package entry points, and explicit cross-project consumers. Multi-
project scope follows the existing deliberate-selection model and never searches all registered
repositories implicitly.

Both phases reuse occurrence extraction, resolution levels, pagination, completeness reporting,
and storage. They add policies and syntax-specific rewrite facts rather than separate graph systems.
