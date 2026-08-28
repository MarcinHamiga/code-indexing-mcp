# Structural References for More Languages Design

## Context

The structural reference index (`2026-08-05-refactoring-reference-index-design.md`) extracts
reference occurrences with Tree-sitter and resolves them conservatively, powering
`find_references` and `analyze_refactor`. Extraction is gated by `STRUCTURAL_LANGUAGES`
(`src/code_indexing_mcp/extractor.py`), a frozenset that today contains only Python,
JavaScript, TypeScript, and TSX. Search, by contrast, covers eighteen languages: Go, Rust,
Java, and C# all have chunking queries under `src/code_indexing_mcp/queries/` and are fully
searchable, but selecting a declaration in any of them returns `UNSUPPORTED_LANGUAGE`, and
their indexed files surface as `unsupported_language` coverage gaps in every reference
result.

These four languages are the right next targets because their module systems are more
deterministic than JavaScript's: imports are explicit, file-to-module mapping is
convention-driven, and aliasing is bounded. A conservative resolver of the kind already
built can classify most references as `exact` rather than `likely`.

## Goals

- Extract and resolve structural references for Go, Rust, Java, and C#.
- Reuse the existing occurrence schema, staging, snapshot pinning, cursor semantics, and
  coverage/completeness reporting unchanged.
- Classify references as `exact`, `likely`, or `unresolved` with the same meaning as the
  Python and JS/TS resolvers.
- Backfill references for existing indexes parse-only, exactly as the current
  `reference-backfill` phase does.
- Ship one language at a time, each independently usable behind the same
  `STRUCTURAL_LANGUAGES` gate.

## Non-goals

- C and C++ in this release: the preprocessor, header ambiguity, and overloading make
  conservative resolution materially weaker; revisit with include-graph heuristics later.
- Lua in this release: dynamic global naming has no syntax facts to be conservative with.
- Type inference. Rust generic/trait inference, C# `var`, Java diamond inference, and Go
  embedded-field promotion keep `likely` classification instead of guesses.
- Cross-crate, cross-assembly, or cross-module resolution beyond what declared imports
  and workspace file layout prove. Reading `Cargo.toml`, `go.mod`, or `.csproj` for
  dependency edges is deferred with the cross-project work.
- Compiler and language-server adapters, for the same toolchain-free reasons recorded in
  the original reference-index design.

## Considered approaches

### Extend the existing extractor and resolver per language — selected

Add `reference_queries/{go,rust,java,csharp}.scm` capture sets, extend
`STRUCTURAL_LANGUAGES`, and teach the resolver's module/import matching
(`_module_candidates`, `_import_targets`, `_module_matches` in `reference_service.py`)
each language's naming rules. The occurrence rows are language-agnostic; no storage or
migration work is needed. Each language's resolver branch is small and testable in
isolation, and coverage gaps disappear mechanically once the language joins the frozenset.

### Language-server adapters

gopls, rust-analyzer, jdtls, and Roslyn resolve references exactly, including inferred
receivers. They also require per-language toolchains, project configuration, spawned
processes, and version coupling — the exact cost the original design rejected. They
remain a possible optional provider tier, not the baseline.

### Name-only occurrence index

Record bare identifier occurrences without import analysis and let the resolver do all
the work textually. Cheaper to extract, but it cannot distinguish an imported alias from
an unrelated local without re-reading files at query time, and it would regress the
paging-over-a-pinned-snapshot contract. Rejected.

## Architecture

### Extraction

Each new `.scm` file captures, in the existing occurrence vocabulary (`import`, `export`,
`call`, `type_use`, `inheritance`, `decorator`, `read`, `write`):

- **Go** — import specs (with named-import aliases), call expressions and selector-call
  receivers, type identifiers in declarations, parameters, results, struct fields, and
  composite literals, method receivers, and embedded interface/struct names.
- **Rust** — `use` declarations including nested groups, `as` renames, and globs; path
  expressions and segments; `impl` blocks (self type and implemented trait); `mod`
  declarations (needed to reconstruct the module tree); type references in signatures,
  `let` bindings, and generics.
- **Java** — package declarations; single-type and on-demand imports; qualified names and
  selector expressions; method invocations; type references in `extends`/`implements`/
  `throws` clauses, annotations, generics, and declarations.
- **C#** — namespace declarations; `using` directives including aliases and `static`;
  member access expressions; type references in base lists, constraints, attributes,
  generics, and declarations.

Export detection stays syntax-level per the existing convention: Go capitalization,
Rust `pub`, Java/C# modifier lists.

### Resolution

The resolver gains one module-system branch per language alongside the existing Python
and JS/TS paths in `_module_candidates` and `_import_targets`:

- **Go**: a package is its directory; an import path maps to a known indexed path.
  There is no re-export aliasing, so unaliased imports resolve `exact` when the path and
  base name are unique. Method calls on struct values keep `likely` when the receiver's
  type is not provable from a local declaration.
- **Rust**: module paths reconstruct from `mod` declarations plus file conventions
  (`foo.rs` / `foo/mod.rs`) within the crate; `use` paths resolve against that tree.
  Trait-method calls and generic receivers stay `likely`; glob imports (`use a::*`)
  resolve like JS `export *` — recorded, but the individual target stays `likely` unless
  unique in scope.
- **Java**: packages map to directory trees; single-type imports resolve `exact` on
  unique base name, on-demand imports (`import a.b.*`) resolve `likely` unless exactly
  one indexed type carries that simple name. Receivers that are plain type names are
  `exact` for static members; other receivers stay `likely`.
- **C#**: namespaces are declared, not path-derived, so namespace-qualified matches use
  the declared hierarchy plus `using` scopes; aliases (`using X = ...`) resolve `exact`;
  `var` receivers and extension methods stay `likely`.

### Coverage and completeness

No new machinery: once a language joins `STRUCTURAL_LANGUAGES`, its files stop appearing
under the `unsupported_language` coverage code, and `_coverage_limitations` reports only
genuine `parse_error` and `stale_file` gaps for it. `find_references` and
`analyze_refactor` tool descriptions update their supported-language list from the
frozenset so they never drift.

### Backfill

The existing parse-only `reference-backfill` path applies unchanged: an index that
already embeds Go/Rust/Java/C# chunks gets occurrences without re-embedding, on the first
reference query that touches the project.

## Testing strategy

- **Extraction fixtures**: representative corpus files per language
  (`tests/fixtures/extractor_corpus/sample.{go,rs,java,cs}`) with committed fingerprints,
  covering imports (plain, aliased, grouped, globbed, on-demand), calls, type uses,
  inheritance, and writes.
- **Resolver corpus**: aliasing, shadowing, re-export-equivalent patterns, on-demand
  imports, and receiver ambiguity, asserting `exact`/`likely`/`unresolved` per case and
  the reported limitations.
- **Completeness**: a mixed-language project asserting that coverage codes flip for the
  newly supported languages only.
- **Backfill**: an embedded-without-occurrences index gains occurrences parse-only, with
  `reference_extraction_duration_ms` reported and no embedding pass.
- **MCP contracts**: `UNSUPPORTED_LANGUAGE` disappears for the new selectors; tool
  descriptions list the new languages.

## Delivery sequence

1. **Go**: queries, extraction fixtures, resolver branch, corpus tests.
2. **Rust**: same, including `mod`-tree reconstruction.
3. **Java**: same, including on-demand import classification.
4. **C#**: same, including namespace and `using`-alias resolution.
5. Documentation and tool-description sweep driven off `STRUCTURAL_LANGUAGES`.

Each step lands behind the frozenset and is independently shippable; a step that proves
weaker than expected (Rust trait inference, say) degrades to `likely` classifications
rather than blocking the sequence.

## Later phases

- C and C++ with include-graph heuristics and a macro-aware occurrence kind.
- Lua global-name heuristics with explicit `unresolved` bias.
- Cross-crate and cross-assembly edges once cross-project tracing
  (`2026-08-27-feature-backlog.md`) defines the symbol-catalogue sharing model.
