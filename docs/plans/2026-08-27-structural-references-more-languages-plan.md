# Structural References for More Languages — Implementation Plan

**Goal:** Extend structural reference extraction and resolution to Go, Rust, Java, and C#,
one language at a time behind `STRUCTURAL_LANGUAGES`, so `find_references` and
`analyze_refactor` work for them with the same `exact`/`likely`/`unresolved` contract as
Python and JS/TS, and their files stop surfacing as `unsupported_language` coverage gaps.

**Design reference:** `docs/plans/2026-08-27-structural-references-more-languages-design.md`.
All code coordinates below were verified against the current tree on 2026-08-27.

**Baseline:** `uv run ruff format --check . && uv run ruff check . && uv run mypy src &&
uv run pytest -n auto` green before Step 0, and after every step.

## Decisions settled before implementation

- **D1 — method qualification.** Rust methods get owner-qualified names (`Type.method`) by
  synthesizing a container scope in `_symbol_context` (`src/code_indexing_mcp/extractor.py`)
  when a definition's ancestor chain crosses an `impl_item` (scope named by the self type) —
  *not* by adding `impl_item` to `queries/rust.scm`, which would create a duplicate
  `struct`-kind chunk named after the self type and collide with the real `struct_item`.
  Chunk ids for Rust method chunks change; chunk-id churn on reindex is documented behavior.
  Go methods stay flat-qualified; the receiver-parameter-name rule covers the common
  same-type case without chunk changes (see Step 2 for the honest cap).
- **D2 — C# namespace encoding.** Namespaces are recorded as per-top-level-type `export`
  rows carrying `module_path` = the declared namespace (file-scoped and block namespaces
  both). Zero schema change; the resolver builds `namespace_by_path` from these rows.
- **D3 — Java on-demand imports.** `import a.b.*` resolves `exact` iff the package resolves
  to indexed files and exactly one indexed declaration carries that simple name; otherwise
  `likely` with a new `on_demand_import` reason. Per the design.

## Mechanics discovered during research (accounted for below)

- **`REFERENCE_SCHEMA_VERSION` is the backfill trigger.** A coverage row is "current" iff
  its `schema_version` matches (`src/code_indexing_mcp/indexing.py:752-763`;
  `REFERENCE_SCHEMA_VERSION = 4` at `indexing.py:84`). Go/Rust/Java/C# files *already*
  carry version-4 coverage rows today, so adding a language to `STRUCTURAL_LANGUAGES`
  alone would leave every existing index silently "current" with zero occurrences. Each
  language step bumps the version — that is precisely what makes the parse-only
  `reference-backfill` re-extract those files without an embedding pass.
- **Reference-query captures are decorative; handlers re-dispatch on `node.type`.** Each
  language needs both the `.scm` capture sets and a `_go_records`/`_rust_records`/
  `_java_records`/`_csharp_records` handler (or shared handler branches), wired through a
  dispatch map replacing the python/else-if at `extractor.py:375-385`.
- **`_identifier_record`'s binding-exclusion sets are Python/JS/TS node types only**
  (`extractor.py:389-501`). Every language step extends them with its own binding node
  types, or every declaration name becomes a bogus `read`.
- **`_classify`'s existing branches already cover several new-language cases verbatim**:
  `self`/`this` receivers via `_same_owner`, `known_namespace_member` for imported
  type-name receivers, and `imported_name="*"` rides the wildcard branch. The new work is
  module-candidate arithmetic plus the same-package/same-namespace rules below.
- **Go/Java/C# need an explicit same-scope rule.** Without it, intra-package calls (Go's
  default — no imports involved) degrade to `name_only_candidate`. Same-directory
  (Go/Java) and same-declared-namespace (C#) plus name match → `exact`, with zero false
  positives: a same-named symbol in another package is either imported (ruled out by
  import analysis) or shadowing (not provable, stays `likely` elsewhere).
- **Gates that move with each language:** `tests/fixtures/extractor_snapshot.json`
  (fingerprints refs/declarations for `sample.go`/`sample.rs`/`Service.java`/
  `Catalog.cs` — regenerate deliberately via `uv run python -m
  tests.test_extractor_equivalence`, in the same commit as the extractor change, never to
  make a failing test pass); `tests/test_refactors.py:750` (Go refusal test expecting
  `UNSUPPORTED_LANGUAGE`); hardcoded language lists at `src/code_indexing_mcp/server.py:1382`
  and `:1425`, both `src/code_indexing_mcp/skills/*/SKILL.md` files + `tests/test_skills.py`
  assertions, `README.md:277` and `:301`.

## Cross-cutting invariants (apply to every step)

1. Zero false positives in the `exact` category — the resolver-corpus hard gate
   (`tests/test_resolver_corpus.py` module docstring).
2. Snapshot regeneration is always deliberate and lands in the same commit as the
   extractor change that justifies it.
3. Coverage flips only for the newly added language: `unsupported_language` disappears for
   it and is unchanged for every other language (dedicated mixed-language test).
4. Backfill stays parse-only: occurrences appear with `reference_extraction_ns` > 0
   reported and no embedding pass (extend `tests/test_indexing.py:571` per language).
5. After every batch: `uv run ruff format . && uv run ruff check . && uv run mypy src &&
   uv run pytest -n auto` (per AGENTS.md; format before push — CI rejects unformatted
   code at the Format step and nothing else runs until it passes).

---

## Step 0 — Shared groundwork (no behavior change)

1. **Dispatch table** in `_structural_records` (`extractor.py`): replace the
   `python → _python_records / else → _javascript_records` branch with a per-language
   handler map (`reference.identifier` still routes to `_identifier_record` first), so
   each language's handler slots in without touching the loop.
2. **Frozenset-driven tool descriptions** (`server.py:1382`, `:1425`): generate the
   "Python, JavaScript, TypeScript, or TSX"-style phrase from
   `sorted(STRUCTURAL_LANGUAGES)` for both `find_references` and `analyze_refactor`
   descriptions, so they never drift again (design requirement).
3. **Module-index threading**: `_classify`, `_may_refer`, `_import_targets_symbol`,
   `_reexport_targets_symbol`, `_module_candidates`, `_module_matches` in
   `src/code_indexing_mcp/reference_service.py` gain one optional parameter — a per-query
   `module_index` built once in `_find_references_with_records` alongside the existing
   precomputed maps (`_imports_by_file`, `_declarations_by_file_target`, `_known_paths`,
   `_reexport_rows_by_path`). Python/JS paths ignore it; existing tests are the proof of
   unchanged behavior.

**Verify:** full gate green; no snapshot change; existing Python/JS/TS resolver corpus and
reference tests untouched and passing.

## Step 1 — Go (ships `REFERENCE_SCHEMA_VERSION` 4 → 5)

**Extraction — add `src/code_indexing_mcp/reference_queries/go.scm` + `_go_records`:**

- Import specs (plain, aliased, grouped, dot-imports) → `import` rows: `module_path` =
  import path, `imported_name` = last path segment (the conventional package name),
  `alias` = explicit alias if present. Dot-imports get `imported_name="*"` so the
  existing `wildcard_import` branch classifies them `unresolved`-with-reason.
- `call_expression` (plain and selector-call `recv.Method`) → `call` rows with
  `receiver_text` and `call_shape` via existing `_call_shape` (Go's `argument_list` is
  already in `_ARGUMENT_LIST_TYPES`).
- Type identifiers in declarations, parameter/result types, struct fields, composite
  literals, `qualified_type` → `type_use`.
- Method receivers: emit the receiver parameter as slot 0 of `declaration.parameters`
  (mirrors Python `self`); its type name becomes a `type_use`.
- Embedded struct/interface fields → `inheritance`.
- `selector_expression` (non-call) → `_emit_member_access`-style read/write rows with
  `receiver_text` (skip when it is the parent call's function).
- Export rows: top-level declarations with capitalized names.
- Go binding exclusions in `_identifier_record`: `short_var_declaration` left,
  `var_spec`/`const_spec` names, `range_clause` left, `for_statement` init,
  function/type/parameter name fields, composite-literal keys.

**Resolver — Go branches:**

- `_module_candidates`: import path → all known directories whose trailing parts equal
  the path's segments (module-prefix-agnostic suffix match; reading `go.mod` is a
  non-goal). Candidates = every known `.go` file directly in those directories (Go
  packages span files).
- `same_package_symbol`: bare name in the same directory as the selected declaration →
  `exact` (Go's intra-package default; no imports involved).
- Receiver-name rule: `receiver_text` equals the enclosing method declaration's
  receiver-parameter name (resolved via `source_qualified_symbol` against declaration
  shapes) and the selected name is project-wide unique → `exact`; otherwise
  `likely`/`unknown_receiver`. **Honest cap under D1:** with flat method names, `s.Handle()`
  inside a method whose receiver is also named `s` reaches `exact` only when `Handle` is
  the unique declaration of that name project-wide; two same-named methods on different
  types keep it `likely` (receiver-type inference is a non-goal).
- Unaliased imports bind by last segment; unique target directory + name match → `exact`
  (the `direct_import_alias` path already handles this given the rows above).

**Tests:**

- Unit extraction suite in `tests/test_reference_extraction.py`: imports
  (plain/aliased/grouped/dot), calls, embedded fields, writes via `=`/`:=`, receiver
  parameter shapes.
- Enrich `tests/fixtures/extractor_corpus/sample.go`; regenerate the snapshot
  deliberately.
- Resolver corpus cases under `tests/fixtures/resolver_corpus/go/`:
  `package_import_exact`, `same_package_exact`, `aliased_import`,
  `unknown_receiver_likely`, `dot_import_unresolved`, `embedded_interface`.
- Flip `tests/test_refactors.py:750` from refusal to a real Go rename analysis; move the
  `UNSUPPORTED_LANGUAGE` refusal expectation to a C (or Lua) fixture.
- Mixed-language coverage test: `unsupported_language` disappears for `go` only.
- Parse-only backfill test: an embedded-without-occurrences Go file gains occurrences
  after the version bump, with `reference_extraction_ns` reported and no embedding pass.

## Step 2 — Rust (schema 5 → 6)

**Extraction — add `reference_queries/rust.scm` + `_rust_records`:**

- `use_declaration` including nested groups, `as` renames, globs → `import` rows:
  `module_path` = the `::`-joined use path, `imported_name` = final segment (or `*`),
  `alias` from `as`. `pub use` additionally emits an `export` row with `module_path` so
  the existing `_reexport_targets_symbol` chain machinery walks re-exports unchanged.
- `mod_item` declarations feed the module index (row-free for plain `mod x;`).
- `call_expression` (plain identifier or `scoped_identifier`/`scoped_type_identifier`
  like `Vec::push`) → `call`; `field_expression` (`self.x`, `obj.method()`) →
  call/member rows with `receiver_text` (`self` handled by the existing receiver branch).
- `impl_item`: self type and implemented trait → `type_use` / `inheritance` rows.
- Types in signatures, `let` bindings, generics, return types → `type_use` via a Rust
  `_descend_type_names` equivalent (`generic_type`, `reference_type`, `tuple_type`,
  `array_type`, `trait_bounds`, `type_arguments`).
- Export rows: `pub` items.
- Binding exclusions: `let_declaration`/`let_pattern`, `for_expression` left, closure
  parameters, `function_item` name field; `macro_invocation` interiors stay row-free
  (conservative).

**D1 implementation:** synthesize the impl scope in `_symbol_context` — when walking a
Rust definition's ancestor chain, treat an `impl_item` as a container named by its self
type. Methods become `Type.method` in chunks, declaration shapes, and
`source_qualified_symbol` consistently, enabling `_same_owner` `exact` classification
for `self.method()` within the impl.

**Resolver — Rust branches:**

- Crate root = shallowest ancestor of the source file containing `lib.rs` or `main.rs`
  (from `known_paths`); `crate::` anchors there, `self::` at the file's own directory,
  each `super::` pops one directory.
- Module path `a::b` → `a/b.rs` or `a/b/mod.rs` under the anchor.
- Unprefixed first segment (edition ambiguity; no `Cargo.toml` per non-goals): generate
  both crate-root-relative and current-directory-relative candidates; corpus-pinned as
  `likely` when the two interpretations diverge.
- Glob imports (`use a::*`) ride the wildcard branch; `pub use` re-export chains resolve
  through the unchanged re-export walker.

**Tests:** unit suite; enriched `sample.rs` + deliberate snapshot regen; resolver corpus
`rust/{crate_relative_use, self_super_paths, pub_use_reexport_chain, glob_use_likely,
trait_impl_inheritance, self_call_exact}`; coverage flip; parse-only backfill. The
shipped doc records the migration note: existing Rust indexes align fully on the next
full reindex; parse-only backfill refreshes rows in the window.

## Step 3 — Java (schema 6 → 7)

**Extraction — add `reference_queries/java.scm` + `_java_records`:**

- `import_declaration` (single-type) → `import`: `module_path` = FQN, `imported_name` =
  simple name, no alias. `import static` → same, `module_path` = FQN minus member.
  On-demand `import a.b.*` → `imported_name="*"`, `module_path="a.b"`.
- `package_declaration` → row-free (directory layout carries it).
- `method_invocation` (`obj.method()`, `Type.staticMethod()`, bare `method()`) → `call`
  with `receiver_text` (`this` already exact via `_same_owner`; plain type-name
  receivers resolve through `known_namespace_member` when a single-type import binds
  them).
- `object_creation_expression` → `call` (constructor shape).
- Types in `extends`/`implements`/`throws`/generics/params/returns/fields →
  `inheritance`/`type_use` via a Java `_descend_type_names`; annotations → `decorator`;
  `field_access` → member read/write.
- Export rows: `public` top-level types/methods.
- Binding exclusions: local-variable declarators, formal parameter names, enhanced-for
  variable, catch formals, pattern bindings, lambda parameters, declaration name fields.

**Resolver — Java branches:**

- `import a.b.C` → path suffix `a/b/C.java` matched against `known_paths` (pure path
  arithmetic): unique suffix → `exact` via `direct_import_alias`; ambiguous or absent →
  no candidate → `likely`.
- On-demand per D3: package dir resolves to indexed files and exactly one indexed
  declaration carries the simple name → `exact`; else `likely` with the new
  `on_demand_import` reason.
- Same-directory rule → `exact` (Java's same-package default).

**Tests:** unit suite; enriched `Service.java` + snapshot; resolver corpus
`java/{single_type_import_exact, on_demand_unique_exact, on_demand_ambiguous_likely,
static_import, same_package_exact, this_receiver_exact}`; coverage flip; parse-only
backfill.

## Step 4 — C# (schema 7 → 8)

**Extraction — add `reference_queries/csharp.scm` + `_csharp_records`:**

- `using_directive` (namespace, `using static`, alias `using X = A.B.C`) → `import`
  rows: `module_path` = namespace, `alias` = alias name, `imported_name` = `*` for
  plain/`static` directives (on-demand semantics).
- `namespace_declaration` (block and file-scoped) → per-top-level-type `export` rows
  carrying `module_path` = the declared namespace (D2) — the one per-file fact the
  resolver needs.
- `invocation_expression` / `member_access_expression` → `call`/member rows with
  `receiver_text` (`this` exact; `var`-typed and extension receivers stay `likely`
  automatically).
- `object_creation_expression` → `call`; base lists → `inheritance`;
  `type_parameter_constraints_clause`, attributes, generics (`generic_name`),
  declarations → `type_use`/`decorator`.
- Export rows: top-level types (any accessibility — `internal` is project-visible).
- Binding exclusions: `variable_declaration` declarators, `foreach_statement` left,
  `catch_declaration`, lambda parameters, out-argument declarations, declaration name
  fields, `using`-statement declarations.

**Resolver — C# branches (module index):**

- `namespace_by_path` built from the export rows' `module_path`; namespaces never map
  to directories — no path arithmetic.
- `using N` + selected file's namespace equals `N` (or, for aliases, the alias target
  FQN's tail matches) → `exact`; same-namespace → `exact`; `var`/extension receivers
  stay `likely` untouched.

**Tests:** unit suite; enriched `Catalog.cs` + snapshot; resolver corpus
`csharp/{using_namespace_exact, using_alias_exact, file_scoped_namespace,
using_static, var_receiver_likely}`; coverage flip; parse-only backfill.

## Step 5 — Documentation and description sweep

- Verify the frozenset-driven tool descriptions render correctly for all eight
  languages.
- Update `src/code_indexing_mcp/skills/impact-analysis/SKILL.md` and
  `src/code_indexing_mcp/skills/feature-dev/SKILL.md` language lists; pin
  `tests/test_skills.py` assertions to the full set.
- Update `README.md` tool table (`:277`) and prose (`:301`).
- Add the shipped-note doc under `docs/plans/` per repo convention, including the Rust
  reindex note from Step 2 and the Go flat-name `exact` cap from Step 1.

## Sequencing and degradation

Each step lands behind `STRUCTURAL_LANGUAGES` and is independently shippable; the schema
version bump per step is the compatibility event that parse-only backfill keys on. A
step that proves weaker than expected (Rust trait inference, say) degrades to `likely`
classifications rather than blocking the sequence, per the design's delivery order:
Go → Rust → Java → C# → docs sweep.

## Later phases (out of scope here)

C/C++ with include-graph heuristics; Lua global-name heuristics; cross-crate and
cross-assembly edges once cross-project tracing
(`docs/plans/2026-08-27-feature-backlog.md`) defines the symbol-catalogue sharing model.
