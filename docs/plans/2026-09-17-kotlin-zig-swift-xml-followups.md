# Follow-ups: Kotlin, Zig, Swift, XML support (PR #69)

Base change: `feat/kotlin-zig-swift-xml-support` (commit `6b474ec`) adds
Kotlin/Swift/Zig structural support and XML definitions-only support.
All items below were noted during implementation and verified against
grammar probes. None blocks the PR. Suggested order: 1 → 3 → 2 → 6 → 4 → 5.

## Phase 1 — make cross-file analysis real

### 1. Export rows for Kotlin, Swift, Zig
Without export rows, `dead-code-report` treats these declarations as
"uncaptured members" and cross-file `find_references` loses exact edges.
- Kotlin: top-level declarations are `public` by default — every top-level
  class/fun/val is exported (no modifier check needed, unlike Java).
- Swift: `public`/`open` modifier check on top-level declarations.
- Zig: `pub` keyword check on top-level `Decl`s.
- Shape: `@reference.export` query captures + `_kotlin_exports` /
  `_swift_exports` / `_zig_exports` handler methods, mirroring
  `_java_exports`. Tests in `test_reference_extraction.py`.

### 2. Kotlin import resolution
The Kotlin row uses `_empty_import_candidates`, so `import a.b.C` never
resolves to an indexed file and cross-file references stay unresolved.
- Investigate: reuse the Java directory/package arithmetic
  (`_java_import_candidates`)? First check whether `_ModuleIndex`
  (`java_files_by_directory`) covers `.kt` files — likely Java-only,
  which would need index changes.
- Fallback if too invasive: keep file-local-only resolution and
  document it as an honest limitation (Swift module imports stay
  unresolvable regardless — no package map exists without a build
  system).
- Tests: resolver-corpus fixtures proving exact resolution, or
  `unresolved` limitation rows if documented as such.

### 3. Heritage rows
Kotlin `class A : B()` delegation specifiers and Swift `class A : B`
inheritance clauses currently land as plain `read`s, so
`impact_radius` cannot traverse class hierarchies.
- Query captures for the delegation/inheritance nodes + handler descent
  emitting `inheritance` rows, mirroring the Java `superclass` /
  `super_interfaces` handling.

## Phase 2 — precision upgrades

### 4. `type_use` descent for Kotlin/Swift annotations
Return types and parameter types are plain `read`s today (Zig already
emits `type_use` via `_zig_emit_type_uses`). Precise kinds let renames
mark annotations exactly instead of as generic reads.
- Handler descent over `user_type` leaves in `_kotlin_records` /
  `_swift_records` + reference-query captures.

### 5. Swift `extension` as a naming scope
Members inside `extension Greeter { ... }` index unqualified, because the
`extension` spelling (a `class_declaration` with an `extension` keyword
and a `user_type` child) is not a definition.
- Precedent: `_symbol_context` synthesizes a container for Rust
  `impl_item` from the self type. Do the same for Swift extensions from
  the extended type name.

### 6. Service-level tests
Coverage today is extractor-rows only. Add resolver-corpus fixtures
(`tests/fixtures/resolver_corpus/`) proving `find_references` resolves
Kotlin/Swift/Zig imports end-to-end (exact/likely), plus rename
validation through the new call shapes in `test_refactors.py`.

## Explicitly deferred

- **Kind refinements** (`Swift` struct/enum recorded as `class`, Kotlin
  `enum class` as `class`, Zig `const X = struct {...}` as `constant`):
  tree-sitter queries cannot express "match X without Y", so
  distinguishing spellings would double-emit same-span rows. Needs a
  dedup design first — revisit only on user request.
- Kotlin secondary constructors (no name node to capture), Swift
  tuple-pattern destructuring, Zig `comptime` params / error sets.
- Extended suffixes (`.zon`, `.xsd`/`.xsl`/`.axml`) — declined in the
  base change in favor of the minimal set; revisit on request.
- Zig default values inside `ParamDecl`/`ContainerField` descend into
  `type_use` rows (documented v1 imprecision).
