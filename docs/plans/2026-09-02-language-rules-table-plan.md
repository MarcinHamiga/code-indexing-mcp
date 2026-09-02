# Per-Language Rules Table — Plan

**Goal:** Collect the per-language node-type sets that `extractor.py` and
`reference_service.py` test inline (`language == "java" and parent.type in {...}`,
`if language == "go": ...`) into one private table, so the sixth structural language is
a table row plus its handlers rather than another sweep of `elif` branches. No behaviour
change: every existing language test is the contract.

**Review finding closed:** the deferred "consolidate per-language rules in `extractor.py`
and `reference_service.py` into a `_LanguageRules` table" item. The review suggested
waiting for the next language; this plan does it first, on its own, so the refactor and
the language land as separate diffs.

**Baseline:** `uv run ruff format --check . && uv run ruff check . && uv run mypy src &&
uv run pytest -n auto`. Green before Step 0 and after every step.

## The one principle

A branch that selects **which node types mean what for a language** moves into the
table. A branch that implements **language-specific logic** (Rust crate-root resolution,
C# `@`-prefixed identifiers, Python's `self`/`cls` receiver) stays a function, but is
reached through the table rather than an `if language ==` chain.

## Decisions settled before implementation

- **D1 — Shape.** A frozen dataclass `_LanguageRules` in a new private module
  `language_rules.py`, one instance per language in a `LANGUAGE_RULES: Mapping[str,
  _LanguageRules]`, plus a `_DEFAULT` instance whose sets are empty. The extractor's
  `STRUCTURAL_LANGUAGES` becomes `frozenset(LANGUAGE_RULES)`; `reference_service.py`
  keeps importing it from `extractor.py` so nothing outside these two modules moves.
- **D2 — Extractor fields.** One `frozenset[str]` per distinct inline set, named for
  what the set means, not for where it is checked. From the 2026-09-02 tree:
  - `import_owner_parents` — parents whose identifiers belong to the import/decorator
    rows and are never reads (`extractor.py:462-468` C#, the Java set just above it).
  - `read_stop_parents` — per-language `parent.type` values that end the read
    classification (the `elif language == ...` ladder at `extractor.py:594-720`),
    split into the fields the ladder actually distinguishes: `declaration_parents`,
    `binding_parents`, `type_position_parents`, `iteration_parents`,
    `lambda_parents`, `construction_parents`, `catch_parents`, `cast_parents`.
  - `handler_owned_type_parents` — the Java and C# sets at `extractor.py:755-790`.
  - `keyword_only_marker` — Python's `list_splat_pattern` (`extractor.py:990`), as an
    `str | None`.
  - `method_name_field_excluded` — the C# `method_declaration` special case at
    `extractor.py:497-500`, as a `bool`.
  Read the ladder once end to end before choosing the final field list; the names
  above are the starting point, and a field that would hold one language's one value
  may be better as a `frozenset` that is empty for the others than as a flag.
- **D3 — Reference-service fields.** `reserved_words: frozenset[str]` (the five
  `_*_RESERVED_WORDS` constants at `reference_service.py:2064-2080`),
  `identifier_valid: Callable[[str], bool]` for the two languages whose rule is not
  "isidentifier and not reserved" (ECMAScript `$`, C# `@`), `bound_receivers:
  frozenset[str]` (Python's `{"self", "cls"}`, `reference_service.py:2287`), and
  `import_candidates: Callable[..., set[PurePosixPath]]` pointing at one function per
  language extracted from the `if language == ...` chain at
  `reference_service.py:3415-3530`. The chain's bodies do not change; they become
  `_go_import_candidates`, `_rust_import_candidates`, and so on, with the shared
  variables passed as arguments.
- **D4 — Lookup, not dispatch.** Call sites read `rules = LANGUAGE_RULES.get(language,
  _DEFAULT)` once at the top of the enclosing method and test `parent.type in
  rules.<field>`. No call site keeps a `language ==` comparison except where it names a
  language family that is not structural (`{"javascript", "typescript", "tsx"}` in
  `_validate_rename`, which becomes a row for each of the three with the same
  `identifier_valid`).
- **D5 — Order of evaluation is preserved.** The extractor ladder is an `if/elif`
  chain, so a node type that appears in two languages' sets under different arms must
  keep the earlier arm's outcome. Step 1 lists every node type that appears in more than
  one arm before any set is moved; if the arms disagree, the field split in D2 is
  refined until they do not.
- **D6 — Tests.** Existing language tests are not edited. One new test,
  `tests/test_language_rules.py`, asserts that every structural language has a row,
  that no row's `reserved_words` is empty, and that `STRUCTURAL_LANGUAGES` equals the
  table's keys, so adding a language without a row fails loudly.

## Steps

**Step 0 — Coordinates and audit.** Re-read the ladder at `extractor.py:585-800` and the
chain at `reference_service.py:3405-3530`; list every `language ==` occurrence in both
files (31 and 25 literal hits on 2026-09-02, some in comments) with the node-type set or
logic it selects. Produce the D5 overlap list.

**Step 1 — Table module.** Add `language_rules.py` with the dataclass, the five (plus
three ECMAScript) rows, and `_DEFAULT`. No call site changes. Baseline green.

**Step 2 — Extractor.** Replace the D2 branches with lookups, one field at a time,
running `pytest tests/test_extractor*.py tests/test_reference*.py` between fields and
the full baseline at the end.

**Step 3 — Reference service.** Replace `_validate_rename`'s chain with
`rules.identifier_valid`, the receiver check with `rules.bound_receivers`, and the
import-candidate chain with `rules.import_candidates(...)`. Baseline green.

**Step 4 — Regression guard (D6).** Add `tests/test_language_rules.py`.

**Step 5 — Docs.** Update the "adding a language" section of the contributing notes (or
add one to the README's development section if none exists) to say: one row in
`language_rules.py`, one handler entry in `_STRUCTURAL_RECORD_HANDLERS`, tests.

**Scope note:** this track touches neither `application.py` nor `storage.py`, so it can
proceed on its own branch alongside the profiling and vector-gate work, but it should
start from `main` after PR #51 merges rather than from `review-remediation`.
