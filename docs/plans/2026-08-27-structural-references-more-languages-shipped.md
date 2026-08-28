# Structural References for More Languages: Shipped

## Outcome

Structural reference extraction and resolution now cover all eight plan languages behind
`STRUCTURAL_LANGUAGES`: the original Python, JavaScript, TypeScript, and TSX set plus the four
languages this plan added — Go (`REFERENCE_SCHEMA_VERSION` 5), Rust (6), Java (7), and C# (8).
`find_references` and `analyze_refactor` serve every one of them with the same
`exact`/`likely`/`unresolved` contract, and Go, Rust, Java, and C# files no longer surface as
`unsupported_language` coverage gaps.

Zero false positives in the `exact` category held throughout: the resolver corpus
(`tests/test_resolver_corpus.py`) gained a per-language section whose every `exact` verdict is
proven by an import, namespace, owner, or uniqueness fact — never by name shape alone.

## Delivered

### Shared groundwork (Step 0)

- A per-language handler table in `_structural_records` (`extractor.py`) replaced the
  python/else-if dispatch; each language step became one `.scm` file plus one map entry.
- Tool descriptions render their language phrase from `sorted(STRUCTURAL_LANGUAGES)` in
  `server.py`, so they cannot drift from the supported set.
- The module index (`_ModuleIndex`) threads through `_classify`/`_may_refer`/import matching
  instead of being recomputed per row.

### Go (Step 1)

- `reference_queries/go.scm` + `_go_records`: import specs (plain/aliased/grouped/dot), calls
  with receiver text and shapes, selector read/write rows, embedded fields as inheritance edges,
  type-position `type_use` rows, receiver parameter as slot 0, and capitalized-name export rows.
- Resolver: import-path suffix matching against known directories (module-prefix agnostic, no
  `go.mod`), the intra-package `same_package_symbol` rule, and the receiver-name rule
  (`same_receiver_member`).
- **Shipped cap (honest):** `s.Handle()` inside a method whose receiver is also named `s`
  reaches `exact` only when `Handle` is the project-wide unique declaration of that name. Two
  same-named methods on different types keep `likely/unknown_receiver` — flat Go method names
  cannot prove which receiver type a local holds, and receiver-type inference stays a non-goal.
  Dot imports stay `unresolved/wildcard_import`.

### Rust (Step 2)

- `reference_queries/rust.scm` + `_rust_records`: `use` trees (nested groups, `as` renames,
  globs), `pub use` re-export rows, calls and field expressions with receivers, impl self/trait
  rows, signature type uses, and `pub` export rows.
- D1 implemented: `_symbol_context` synthesizes an impl-block container scope named by the self
  type, so methods qualify as `Type.method` and `self.method()` resolves exactly through
  `_same_owner`.
- Resolver: crate roots from `lib.rs`/`main.rs` markers, `crate::`/`self::`/`super::` anchoring,
  module-path directory arithmetic (`a/b.rs` or `a/b/mod.rs`), edition-ambiguous unprefixed uses
  pinned to `likely` when the two anchorings diverge, glob uses `unresolved/wildcard_import`, and
  re-export chains through `pub use` barrels.
- **Migration note:** existing Rust indexes were written under older schema versions with zero
  Rust occurrences. The version bump makes the parse-only reference backfill re-extract every
  Rust file — occurrences appear with no embedding pass. Existing indexes align fully on the
  next full reindex; parse-only backfill refreshes rows in the window.

### Java (Step 3)

- `reference_queries/java.scm` + `_java_records`: single-type/static/on-demand/static-on-demand
  imports, calls whose span covers the two-sibling `receiver . name` callee, constructor-shaped
  `new` calls, field-access read/write rows, `extends`/`implements` inheritance edges, throws and
  bounds as type uses, annotations as decorators, and `public` top-level-type export rows.
- The Java grammar names few of its children, so descent is positional: wrappers
  (`generic_type`, `scoped_type_identifier`, `array_type`, heritage/throws/catch lists) unwrap by
  named children, and a `method_declaration`'s return type is the named child before its name.
- Resolver per D3: an on-demand import (`import a.b.*`) resolves `exact/on_demand_import` iff the
  package resolves to indexed files (directory-suffix match) and the name is the declaration's
  only indexed spelling project-wide; otherwise `likely` with the same reason, surfaced as a
  limitation only when not exact. Single-type imports bind by pure path arithmetic
  (`a/b/C.java`); the same-directory `same_package_symbol` rule is shared with Go.

### C# (Step 4)

- `reference_queries/csharp.scm` + `_csharp_records`: using directives (plain/`global`/`static`/
  alias), invocations and constructor creations, member-access read/write rows, base-list
  inheritance edges, attributes as decorators, constraint clauses, casts, and top-level-type
  export rows of any accessibility.
- D2 implemented: each top-level type's export row carries `module_path` = the declared
  namespace, covering block and file-scoped namespaces (whose types are *siblings* of the
  `file_scoped_namespace_declaration`, not descendants — the extractor walks preceding siblings).
  Namespaces never map to directories; the resolver builds `namespace_by_path`,
  `files_by_namespace`, and `names_by_namespace` from these rows.
- Resolver: plain/`static` usings bind `exact/on_demand_import` when the selected declaration is
  provably the one declaration in the used namespace (or, for `using static`, a member of the
  named type); more than one proven binding — the case a C# compiler rejects as ambiguous —
  degrades to `likely`. Alias usings (`using W = Acme.Gadget;`) bind through the ordinary
  import machinery by FQN tail. Same-namespace uses resolve `same_package_symbol`, keyed on
  namespace identity rather than directories. `var`-typed and extension receivers stay
  `likely/unknown_receiver` untouched.

### Documentation sweep (Step 5)

- The derived tool-description phrase renders all eight languages.
- `skills/impact-analysis` and `skills/feature-dev` language lists updated;
  `tests/test_skills.py` now pins the full `STRUCTURAL_LANGUAGES` display set so a future
  language step must sweep the skills in the same change.
- README tool table and reference-workflow prose updated, including on-demand imports in the
  conservative-limitations list.

## Compatibility

Each language step bumped `REFERENCE_SCHEMA_VERSION` (4→5→6→7→8); that bump is the compatibility
event parse-only backfill keys on. Files of a newly added language indexed before its step carry
coverage rows with zero occurrences; after the bump, the parse-only backfill re-extracts them —
occurrences appear, `reference_extraction_ns` is real, and no embedding pass runs. Full semantic
alignment still happens on the next full reindex, unchanged from prior behavior. Chunk ids for
Rust method chunks changed in Step 2 (impl-scope qualification, D1); chunk-id churn on reindex is
documented behavior.

## Out of scope (later phases)

C/C++ with include-graph heuristics and Lua global-name heuristics; cross-crate and
cross-assembly edges once cross-project tracing defines the symbol-catalogue sharing model.
