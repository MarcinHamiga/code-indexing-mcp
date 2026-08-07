# Reference Index Hardening Backlog

## Context

The structural reference index and refactor analysis shipped in #23. A review on
2026-08-07 found one defect that permanently corrupted a project's index and a
cluster of defects that made the analysis report incomplete results as complete.
Those were fixed as a hotfix in `aa1f602` and `2eea2f8`.

This document records what the review found and the hotfix deliberately did
*not* address, so a proper hardening pass can be planned. It is a backlog, not a
design: each item states the defect, how it fails, and where it lives. It does
not prescribe the fix.

Severity is about consequence to a caller acting on the output, not effort.
A "silent miss" — a real reference the tool does not report while claiming
completeness — is the worst class, because an agent renames and ships a broken
build.

## What the hotfix already covers

Working now, with regression tests:

- Incremental re-indexing no longer collides on `reference_id` and bricks the
  project.
- Class bodies are no longer treated as a method's lexical scope.
- Files in languages with no reference query are reported as
  `unsupported_language` limitations; selecting a declaration in one raises
  `UNSUPPORTED_LANGUAGE`.
- `unknown_receiver` and `likely_change` now affect the completeness state.
- Rename findings carry `edit_start_byte`/`edit_end_byte` covering just the
  identifier, so applying them does not destroy a module qualifier or an alias.
- A single unparseable file degrades the result instead of disabling both tools
  for the whole project forever.
- Signature analysis reads the pinned snapshot once rather than re-querying per
  hit.

## The gap that matters most

**The completeness signal does not cover extraction gaps.** The hotfix makes the
resolver honest about files it *never looked at* — other languages, parse
failures. It cannot detect a reference the extractor failed to produce for a file
it did parse successfully. Every item in "Extraction gaps" below is therefore
still a silent miss reported as `complete`.

Verified after the hotfix:

```
base.ts:  export class Base { run(): number { return 1; } }
child.ts: import { Base } from './base';
          export class Child extends Base { }

analyze_refactor(Base -> Foundation)
  completeness: complete
  limitations:  []
  findings:     base.ts declaration, base.ts export, child.ts import
```

`extends Base` is missing. Applying the three reported edits renames the import
and leaves `extends Base` dangling. This is the exact failure the feature exists
to prevent, and TypeScript is where it concentrates.

**Practical read:** Python renames are in reasonable shape. TypeScript and
JavaScript should be treated as preview until the extraction items below land —
or `analyze_refactor` should report a standing limitation for TS/JS
inheritance and type references until it can see them.

## Extraction gaps

Reference rows that are never produced, so nothing downstream can report them.
All were reproduced against the extractor during the review.

| # | Defect | Where | Severity |
|---|---|---|---|
| E1 | TS/TSX inheritance captures the raw clause text: `class A extends Base implements C` yields `target_name` `"extends Base"` and `"implements C, Other"`. Renaming a TS base class or interface finds zero references. JS is unaffected — its grammar puts the identifier directly under `class_heritage`. | `reference_queries/typescript.scm:9`, `tsx.scm:9`, handler `extractor.py:784-791` | High |
| E2 | TS generic and union types store the whole type expression: `Box<Item>`, `A \| B`. Inner names are `type_identifier`, which the identifier capture does not match, and `generic_type`/`type_annotation` are excluded. Python is unaffected — `(type (_) @name)` recurses. | `extractor.py:806-812`, `typescript.scm:16`, `tsx.scm:16` | High |
| E3 | `export * from './x'` produces no reference at all; `export * as ns from './x'` produces a bogus `read` and discards the module path. Barrel and index files are invisible edges. | `extractor.py:724-783` | High |
| E4 | The mandatory `arguments:` field drops whole call forms: Python `summarize(x for x in items)` (generator sole argument), JS tagged templates `` gql`…` `` and `` styled.div`…` ``, and `new Widget;` without parentheses. No `read` fallback exists because the `function` field is excluded. | `python.scm:44`, `javascript.scm:10,14`, `typescript.scm:11,15`, `tsx.scm:11,15` | High |
| E5 | No `write` kind is ever emitted, and non-call member access is dropped. `config.TIMEOUT` and `config.TIMEOUT = 5` record only the `import`. Renaming a module-level constant reports zero references. JS object shorthand `{ onSave }` is also missed (`shorthand_property_identifier` is not `identifier`). | `extractor.py:405-406`; `models.py:130` declares `write` but nothing produces it | High |
| E6 | JS/TS decorators produce nothing: `@Component`/`@sealed` yield zero references. `@Injectable()` survives only as `kind="call"`, so a `kinds=["decorator"]` filter returns nothing for the whole JS family while working for Python. | no `decorator` pattern in `javascript.scm`/`typescript.scm`/`tsx.scm`; `extractor.py:361-372` | Medium |
| E7 | Destructured JS/TS parameters produce a wrong, lossy shape. JS `function f({ alpha, beta })` yields one parameter named `alpha`; TS yields one named `{ item, onSelect }`. Signature analysis on React-style props compares against a fabricated list. The existing TSX test uses a single-key pattern, which hides it. | `extractor.py:488-497` | Medium |
| E8 | The `"=" in child_text` default-detection heuristic misfires on TS function types: `handler: (e: Event) => void` is read as optional because `=>` contains `=`, so `missing_required_parameter` never fires for a callback parameter. | `extractor.py:516` | Medium |
| E9 | Bare and dynamic module edges dropped: `import './polyfill'` yields nothing; `require('./a')` and `import('./b')` yield a call to `require`/`import` with no `module_path`. The design says dynamic imports should remain visible as likely/unresolved. | `extractor.py:671-723`, `724-783` | Medium |
| E10 | TS `abstract class A {}` produces no declaration at all, and a member inside it is qualified `run` rather than `A.run`. That corrupts `source_qualified_symbol` for every reference inside an abstract class and makes the class unselectable. Class-field arrow methods (`run = (a,b) => a`) likewise produce no declaration. | `queries/typescript.scm` | Medium |
| E11 | JS `for (const item of items)` records the loop *binding* as a `read`, a spurious hit for any rename of an unrelated symbol named `item`. The exclusion list has `for_statement`/`for_in_clause` but not `for_in_statement`. | `extractor.py:398-401` | Low |
| E12 | Python `class A(Base, metaclass=Meta)` emits `target_name` `"metaclass=Meta"`. Noise only — `Meta` is separately recorded as a `read`. | `extractor.py:639-648` | Low |
| E13 | Python has no `export` kind and `__all__` is ignored, so a rename never flags the `__all__` entry that must change with it. | — | Low |

## Resolution gaps

| # | Defect | Where | Severity |
|---|---|---|---|
| R1 | No override analysis. Renaming `Base.handle` never mentions `Child.handle`, which exists only as a declaration row and is never a reference candidate. Reported `complete`. | `reference_service.py` | High |
| R2 | Re-export chains degrade to `likely`/`unresolved` rather than resolving. `_import_targets` requires the import's module path to resolve to the declaration's own file, so `from pkg import b` where `pkg/__init__.py` re-exports it never binds exactly. Conservative and safe, but the *reason* is not distinguishable from an ordinary unproven name. | `reference_service.py:540-551` | Medium |
| R3 | A rename of a TS `export function answer()` yields two `must_change` entries for one identifier: the synthetic declaration finding and the `same_file_symbol` export hit. `counts.must_change` overstates, and an agent applying both edits sequentially rewrites the same bytes twice. Verified still present after the hotfix. | `reference_service.py:172-215` | Medium |
| R4 | The final page of a paged analysis reports `completeness: "complete"` even though earlier pages held most of the findings and the declaration edit appeared only on page 1. A caller reading one page's completeness field is misled. `tests/test_refactors.py` currently encodes this. | `reference_service.py`, completeness block | Low |

## Storage and indexing

| # | Defect | Where | Severity |
|---|---|---|---|
| S1 | **Chunk/reference divergence that hides itself.** When embedding fails, `stage_failure` writes the `files` row with the *new* content hash but keeps the old chunks. Backfill validates against that new hash, so it happily stages references for content the chunk table does not contain, then commits. Coverage hash now equals the files hash, so the divergence becomes undetectable and the next backfill reports nothing to do. `_select` then resolves declarations from stale chunks against a newer reference generation. This is a normal-operation path, not a crash path. | `indexing.py:545-546`, `indexing.py:315` | High |
| S2 | Reference backfill resets project state from `partial` to `ready`. `_commit_staged` unconditionally writes `"partial" if errors else "ready"`, and backfill never has errors, so the first reference query after a failed index silently promotes the project and `project_status` stops showing the failed files. | `indexing.py:359`, `indexing.py:774-778` | Medium |
| S3 | A binary or undecodable file with a source extension makes the project permanently stale: it is dropped from storage but the scanner keeps yielding it, so `current.keys() != existing.keys()` is true forever. The staleness bug predates #23, but `ensure_reference_index` now turns it into a full re-index under the global lock on *every* reference query. | `indexing.py:611-621`, `application.py:665`, `application.py:617-618` | Medium |
| S4 | `list_reference_records` materializes the entire reference table per query — roughly one row per source line (20,820 rows for this repo's `src/`, ~31 MB, 0.11 s). Extrapolated to a 500k-line repo that is ~750 MB resident and ~2.7 s per call. The hotfix removed the per-hit repetition but not the full materialization. Note the pushdown primitives already exist and now have **no production callers**: `declaration_shapes`, `imports_for`, and `target_name_candidates` are exercised only by `tests/test_storage.py`. A filtered resolver would use them. | `storage.py:388-392`, `855-878` | Medium |
| S5 | A missing reference table is indistinguishable from "no references" at the storage layer: `_reference_rows` returns `[]` and `reference_version` returns `0`. Masked today because `Application` always calls `ensure_reference_index` first, but `ReferenceService` has no guard of its own. Related: `storage.py:265` and `360` rely on bare `assert` for the same invariant, which vanishes under `python -O`. | `storage.py:858-860`, `397-402` | Low |
| S6 | Backfill publishes its `committing` progress phase *after* the commit finished, so a watcher never sees it. The `stale_paths` early return also drops `files_backfilled` from the report. | `indexing.py:351-365` | Low |

## Surface, benchmarks, and tests

| # | Defect | Where | Severity |
|---|---|---|---|
| T1 | `reference_extraction_duration_ms` is not reference-extraction duration — it is the whole parse phase, which wraps tree-sitter parsing, chunk extraction, and reference extraction in one timer. A reader comparing before/after would attribute all of it to this feature. `structural_records` is a whole-project total, so all four scenarios report the same number, including `incremental_index`. | `benchmark.py:56`, `indexing.py:641` | Medium |
| T2 | A well-formed but foreign cursor leaks a raw `KeyError` as the whole error message (`Error executing tool find_references: 'version'`). `_decode_cursor` validates only that the payload is a dict. The remaining cursor mismatches also raise bare `ValueError` rather than a structured error. | `reference_service.py:_decode_cursor` and the cursor block | Medium |
| T3 | `tests/test_benchmark.py:51-52` asserts the *degenerate* values of both new metrics (`== 0`), which the fake application inevitably produces. Neither new benchmark path is actually exercised, and renaming `Application.store` would keep the test green while reporting 0 records forever. | `tests/test_benchmark.py` | Low |
| T4 | `test_every_tool_parameter_is_documented_and_bounded` walks only top-level `properties`, so all four `DeclarationSelector` fields ship with no description. An agent is not told that `path` is repo-relative POSIX, that `project` accepts id/name/path, or that `qualified_symbol` is dotted. | `tests/test_server.py:1490-1501` | Low |
| T5 | `tests/test_skills.py` only asserts that the two tool-name strings appear in the markdown. It does not check that the skills convey language coverage, the resolution contract, or completeness handling. | `tests/test_skills.py:51-55` | Low |
| T6 | `indexed-review/SKILL.md:39` still describes `find_symbol` as finding "definitions and call sites". The same stale claim was corrected in `SERVER_INSTRUCTIONS`, `impact-analysis`, and (in the hotfix) `feature-dev`. | `skills/indexed-review/SKILL.md` | Low |
| T7 | README describes `evidence` only in the rename sense ("aliases that identify the target but need no spelling change"), but for `signature_change` the same bucket holds compatible call sites. | `README.md` | Low |

## Suggested sequencing

1. **S1** first. It is the only remaining item that silently corrupts stored
   state, and it hides its own evidence, so every later measurement is suspect
   while it is live.
2. **E1, E2, E3, E5** — the TypeScript/JavaScript silent misses. Together they
   are what stands between this feature and being trustworthy outside Python.
   Until they land, consider emitting a standing limitation for TS/JS so
   `completeness` stops claiming more than the extractor can deliver.
3. **R1** (overrides) and **E4** (dropped call forms) — the remaining
   silent-miss classes in languages that are otherwise well covered.
4. **S4** before anyone points this at a large repository; the query primitives
   already exist and are unused.
5. Everything else as ordinary cleanup.

A corpus test would catch this class of regression far better than more unit
tests: a small annotated fixture repository per language, asserting the exact
expected reference set, with a rule that a construct starts as `likely` until a
fixture proves exact resolution is sound. The original design called for this
(`2026-08-05-refactoring-reference-index-design.md`, "Resolver corpus"); it was
not built, and its absence is why every defect above shipped green.
