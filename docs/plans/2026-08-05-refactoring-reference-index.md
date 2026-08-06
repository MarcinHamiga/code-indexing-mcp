# Refactoring Reference Index Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add conservative, evidence-backed reference lookup plus rename and signature-change analysis for Python, JavaScript, TypeScript, and TSX.

**Architecture:** Extend the existing single Tree-sitter parse to emit structural records beside semantic chunks. Store those records in a third per-project LanceDB table that participates in the existing Arrow staging and crash-recovery transaction, then resolve them conservatively at query time through a new `ReferenceService`. Expose `find_references` and `analyze_refactor` through the application, daemon broker, and MCP server without modifying source files.

**Tech Stack:** Python 3.12, Pydantic 2, Tree-sitter 0.26, PyArrow 23, LanceDB 0.34, FastMCP, pytest, Ruff, mypy.

**Design reference:** `docs/plans/2026-08-05-refactoring-reference-index-design.md`

**Working directory:** `/Users/mhamiga/Projects/mcps/code-indexing-mcp/.worktrees/refactoring-reference-index`

**Baseline:** `uv run pytest -q` passed with 940 tests and 8 expected opt-in skips. The full test environment requires `uv sync --all-groups --extra cpu --extra tui --locked` because `tests/test_installer_tui.py` imports the optional TUI dependency.

## Implementation constraints

- Use `@superpowers:test-driven-development` for every production-code task.
- Never persist a guessed target declaration. Store syntax facts and resolve against the current symbol/import catalogue at query time.
- Prefer `likely` or `unresolved` over a false `exact` result.
- A successful changed-file commit replaces files, chunks, and structural rows transactionally.
- A failed file preserves its previous chunks and structural rows.
- Structural backfill must not call the embedding backend.
- Every successfully parsed supported file gets a coverage row, even if it contains no references. Coverage rows make parse-only backfill detectable without migrating the central project registry.
- MCP tools are read-only from the caller's perspective but use the existing `_READS_AND_REGISTERS` annotation because freshness checks may write.
- Do not implement deletion or module-move analysis in this plan. They are later phases built on the reference foundation.

### Task 1: Structural domain models and extraction

**Files:**
- Modify: `src/code_indexing_mcp/models.py:24-433`
- Modify: `src/code_indexing_mcp/extractor.py:145-215`
- Create: `src/code_indexing_mcp/reference_queries/__init__.py`
- Create: `src/code_indexing_mcp/reference_queries/python.scm`
- Create: `src/code_indexing_mcp/reference_queries/javascript.scm`
- Create: `src/code_indexing_mcp/reference_queries/typescript.scm`
- Create: `src/code_indexing_mcp/reference_queries/tsx.scm`
- Create: `tests/test_reference_extraction.py`

**Step 1: Add failing model and Python extraction tests**

Cover direct and qualified calls, imported aliases, decorators, inheritance, type uses, positional/keyword/spread call shape, declaration parameters, enclosing qualified symbols, and exact byte/line ranges.

```python
def test_python_references_preserve_import_alias_call_shape_and_scope() -> None:
    source = b'''from pkg.auth import enforce as check

class Guard(BaseGuard):
    @registered
    def run(self, user, *, strict=True):
        return check(user, strict=strict)
'''

    result = TreeSitterExtractor().extract(Path("guard.py"), "python", source)

    call = next(item for item in result.references if item.kind == "call")
    assert call.written_name == "check"
    assert call.source_qualified_symbol == "Guard.run"
    assert call.call_shape == CallShape(positional_count=1, keywords=["strict"])
    assert any(item.kind == "import" and item.alias == "check" for item in result.references)
    shape = next(item for item in result.declarations if item.qualified_symbol == "Guard.run")
    assert [(parameter.name, parameter.kind, parameter.required) for parameter in shape.parameters] == [
        ("self", "positional", True),
        ("user", "positional", True),
        ("strict", "keyword_only", False),
    ]
```

Run: `uv run pytest tests/test_reference_extraction.py -q`

Expected: FAIL because `ExtractionResult` has no `references` or `declarations` fields.

**Step 2: Add the immutable structural models**

Add these closed types and models to `models.py`:

```python
ReferenceKind = Literal[
    "import", "export", "call", "type_use", "inheritance", "decorator", "read", "write"
]
ParameterKind = Literal["positional_only", "positional", "keyword_only", "variadic", "keyword_variadic"]
ResolutionLevel = Literal["exact", "likely", "unresolved"]

class CallShape(FrozenModel):
    positional_count: int = 0
    keywords: list[str] = Field(default_factory=list)
    has_positional_spread: bool = False
    has_keyword_spread: bool = False
    type_argument_count: int | None = None
    constructor: bool = False

class ParameterShape(FrozenModel):
    name: str
    kind: ParameterKind
    required: bool
    position: int

class ExtractedReference(FrozenModel):
    kind: ReferenceKind
    written_name: str
    target_name: str
    source_qualified_symbol: str | None = None
    module_path: str | None = None
    imported_name: str | None = None
    alias: str | None = None
    receiver_text: str | None = None
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    call_shape: CallShape | None = None

class ExtractedDeclarationShape(FrozenModel):
    symbol: str
    qualified_symbol: str
    kind: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    parameters: list[ParameterShape] = Field(default_factory=list)

class ExtractionResult(FrozenModel):
    chunks: list[ExtractedChunk]
    references: list[ExtractedReference] = Field(default_factory=list)
    declarations: list[ExtractedDeclarationShape] = Field(default_factory=list)
    has_errors: bool = False
```

**Step 3: Implement one cached structural query per supported language**

Add `_reference_queries` and the same double-checked locking used by `_query`. Compile package data from `code_indexing_mcp.reference_queries`. Use capture families such as `reference.call`, `reference.import`, `reference.type`, `reference.inheritance`, `reference.decorator`, `declaration.parameters`, `name`, `receiver`, `module`, `alias`, and `arguments`.

Add private helpers in `TreeSitterExtractor`:

```python
def _structural_records(
    self,
    language: str,
    root: Node,
    source: bytes,
    definitions: list[_Definition],
    line_index: _LineIndex,
) -> tuple[list[ExtractedReference], list[ExtractedDeclarationShape]]: ...

def _call_shape(self, language: str, arguments: Node, source: bytes) -> CallShape: ...

def _parameter_shapes(self, language: str, parameters: Node, source: bytes) -> list[ParameterShape]: ...
```

Reuse `_DefinitionIndex` to associate each reference with the nearest containing declaration. Do not parse the source a second time.

**Step 4: Add JS/TS/TSX extraction fixtures**

Cover named/default/namespace imports, aliases, re-exports, `this.method()`, namespace member calls, interfaces/types, generics, default/rest parameters, JSX-containing TSX, and spread arguments.

Run: `uv run pytest tests/test_reference_extraction.py tests/test_extractor.py -q`

Expected: PASS.

**Step 5: Run static checks and commit**

Run: `uv run ruff check src/code_indexing_mcp/models.py src/code_indexing_mcp/extractor.py tests/test_reference_extraction.py`

Run: `uv run mypy src/code_indexing_mcp/models.py src/code_indexing_mcp/extractor.py`

Expected: both commands pass.

```bash
git add src/code_indexing_mcp/models.py src/code_indexing_mcp/extractor.py \
  src/code_indexing_mcp/reference_queries tests/test_reference_extraction.py
git commit -m "feat: extract structural code references"
```

### Task 2: Transactional reference storage and crash recovery

**Files:**
- Modify: `src/code_indexing_mcp/staging.py:1-388`
- Modify: `src/code_indexing_mcp/storage.py:1-707`
- Modify: `tests/test_staging.py`
- Modify: `tests/test_storage.py`

**Step 1: Write failing three-table transaction tests**

Add tests proving:

- `TableVersions` includes `references`.
- `StagingJob` writes `references.arrow` without Python object-vector materialization.
- Replacing a file removes its previous reference rows and inserts the new rows.
- Removing a file deletes files, chunks, and references.
- Commit failure restores all three table versions.
- Startup recovery restores all three versions after a simulated mid-commit crash.
- Read-only reference queries do not materialize an unindexed partition.

```python
def test_a_failed_commit_restores_reference_rows_with_files_and_chunks(tmp_path: Path) -> None:
    indexer, store, project = indexed_reference_project(tmp_path)
    before = store.list_reference_records(project.id)
    versions = store.table_versions(project.id)
    # Stage and partially apply a replacement, then raise from the live write.
    ...
    assert store.table_versions(project.id) == versions
    assert store.list_reference_records(project.id) == before
```

Run: `uv run pytest tests/test_staging.py tests/test_storage.py -q`

Expected: FAIL because project partitions contain only files and chunks.

**Step 2: Add Arrow reference rows and schema**

Add `ReferenceRow` to `staging.py`. Use one structural table with a `record_kind` discriminator:

- `reference`: source occurrence and optional JSON call shape.
- `declaration`: declaration identity and JSON parameter shape.
- `coverage`: one record per successfully parsed file and structural schema version.

The Arrow schema must include `reference_id`, `record_kind`, file/project/path/language, reference kind, source symbol, target/import/alias/receiver facts, ranges, `shape_json`, content hash, and `schema_version`.

Add `reference_schema()` and `reference_arrow_schema()` to `LanceStore`.

**Step 3: Extend staging and journal versions**

Add `REFERENCES_NAME`, a third writer, `stage_references`, and `iter_reference_groups`. Track `replace_reference_file_ids` separately so parse-only backfill cannot accidentally delete unchanged chunks.

Extend `TableVersions` and the journal:

```python
@dataclass(frozen=True)
class TableVersions:
    files: int
    chunks: int
    references: int
```

Recovery must restore `files`, `chunks`, and `references` before marking a journal rolled back.

**Step 4: Extend per-project storage**

Add `references` to `_ProjectTables`, `_tables`, and `_existing_tables`. Extend `replace_files_from_arrow` with `reference_groups` and independent reference replacement IDs. Add exact-filter read methods for coverage, declaration shapes, imports, and target-name candidates.

`ensure_indexes` should create B-tree indexes on `file_id`, `record_kind`, `target_name`, `module_path`, and `kind` for the references table while preserving current chunk indexes.

**Step 5: Run tests and commit**

Run: `uv run pytest tests/test_staging.py tests/test_storage.py -q`

Run: `uv run ruff check src/code_indexing_mcp/staging.py src/code_indexing_mcp/storage.py tests/test_staging.py tests/test_storage.py`

Expected: PASS.

```bash
git add src/code_indexing_mcp/staging.py src/code_indexing_mcp/storage.py \
  tests/test_staging.py tests/test_storage.py
git commit -m "feat: store references transactionally"
```

### Task 3: Normal indexing and parse-only structural backfill

**Files:**
- Modify: `src/code_indexing_mcp/models.py`
- Modify: `src/code_indexing_mcp/indexing.py:62-597`
- Modify: `src/code_indexing_mcp/progress.py:32-75`
- Modify: `src/code_indexing_mcp/application.py:566-627`
- Modify: `tests/test_indexing.py`
- Modify: `tests/test_progress.py`
- Modify: `tests/test_application.py`

**Step 1: Write failing normal-index tests**

Prove that:

- A successful file stages references, declaration shapes, and a coverage row.
- A file with no references still gets coverage.
- Changing a file replaces its references atomically.
- Removing a file removes its references.
- An extraction or embedding failure preserves the previous reference generation.

Run: `uv run pytest tests/test_indexing.py -q`

Expected: FAIL because `_PendingFile` carries only chunks.

**Step 2: Stage structural records on the successful-file path**

Extend `_PendingFile` with `references` and `declarations`. Add `_reference_rows(project_id, file, extraction)` that emits deterministic IDs from file ID, record kind, range, and structural schema version. Stage rows and mark the reference file replaced only after embedding/windowing succeeds.

Keep reference staging outside `_embed_candidates`; it must never add embedding work.

**Step 3: Write failing parse-only backfill tests**

Use a counting embedder and an existing semantic index whose references table has no coverage rows.

```python
def test_reference_backfill_parses_unchanged_files_without_embedding(tmp_path: Path) -> None:
    embedder = RecordingEmbedder()
    indexer, store, project = legacy_semantic_index(tmp_path, embedder)
    batches_before = len(embedder.passage_batches)

    report = indexer.backfill_references(project)

    assert report.files_backfilled > 0
    assert len(embedder.passage_batches) == batches_before
    assert store.reference_coverage(project.id).complete
```

Also test a content-hash mismatch, a backfill interruption, a file with no occurrences, and retry after incomplete coverage.

Run: `uv run pytest tests/test_indexing.py -k reference_backfill -q`

Expected: FAIL because `backfill_references` does not exist.

**Step 4: Implement backfill under the existing global lock**

Add `ReferenceBackfillReport` and `ReferenceCoverage` models. `Indexer.backfill_references` should:

1. Compare stored files with coverage rows for the current structural schema version.
2. Stream only missing/outdated files through the scanner.
3. Verify each file's current content hash equals `StoredFile.content_hash`.
4. Parse once and stage only structural rows plus coverage.
5. Commit the references table through the same journal/rollback mechanism.
6. Abort with a structured stale condition if source content changed; the normal freshness path handles that file before retry.

Use progress phase `extracting_references`; update `IndexProgress.describe` and tests.

**Step 5: Wire automatic backfill into application freshness**

Add an internal `ensure_reference_index(project, roots)` call after the normal stale-index wait and before reference tools execute. Do not backfill during ordinary semantic searches.

Run: `uv run pytest tests/test_indexing.py tests/test_progress.py tests/test_application.py -q`

Expected: PASS.

**Step 6: Commit**

```bash
git add src/code_indexing_mcp/models.py src/code_indexing_mcp/indexing.py \
  src/code_indexing_mcp/progress.py src/code_indexing_mcp/application.py \
  tests/test_indexing.py tests/test_progress.py tests/test_application.py
git commit -m "feat: backfill structural references without embeddings"
```

### Task 4: Conservative reference resolution and pagination

**Files:**
- Create: `src/code_indexing_mcp/reference_service.py`
- Modify: `src/code_indexing_mcp/models.py`
- Modify: `src/code_indexing_mcp/storage.py`
- Create: `tests/test_references.py`

**Step 1: Write failing declaration-selection and resolver tests**

Build small indexed repositories covering:

- Python direct imported aliases and module aliases (`from x import f as g`, `import x as m`).
- Python `self`/`cls` methods and same-file calls.
- Python wildcard/dynamic imports and unknown receivers.
- JS/TS named/default/namespace imports, aliases, re-exports, and `this`.
- Lexical shadowing that prevents an exact result.
- Duplicate declaration names requiring a path/qualified selector.
- Result ordering and reason codes.

Exact imported and known-owner cases must resolve `exact`. Dynamic/unknown receiver cases must be `likely` or `unresolved`, never exact.

**Step 2: Add public result models**

Add `DeclarationSelector`, `SelectedDeclaration`, `ReferenceHit`, `ReferenceLimitation`, and `ReferenceResponse`. A selector accepts either `chunk_id` or the tuple `project`/`path`/`qualified_symbol`; validate that callers do not mix modes.

Each hit includes source range, snippet, kind, resolution, stable reason code, explanation, and `edit_required` only when computed by refactor analysis.

**Step 3: Implement module identity and import catalogues**

In `ReferenceService`:

- Map Python paths to module identities, including `__init__.py`.
- Resolve relative Python imports within the selected project.
- Resolve JS/TS relative paths across supported extensions and `index.*` files.
- Build per-file alias/namespace/re-export maps from structural records.
- Select declarations from chunk metadata without materializing vectors.

Do not implement `pyproject` package discovery, TypeScript `paths`, runtime imports, or inferred receiver types in this release; report them as limitations.

**Step 4: Implement conservative classification**

Implement ordered rules for direct import aliases, known namespace members, same-owner `self`/`cls`/`this`, and unique same-file lexical declarations. Lower-confidence rules must never promote an ambiguous candidate to exact.

**Step 5: Add snapshot-bound cursor paging**

Encode an opaque URL-safe cursor containing the selected declaration, filters, references-table version, and offset. On a later page, open a fresh references table handle at the recorded Lance version. If that version is unavailable, raise `STALE_CURSOR`; never mix versions.

Run: `uv run pytest tests/test_references.py tests/test_storage.py -q`

Expected: PASS.

**Step 6: Commit**

```bash
git add src/code_indexing_mcp/reference_service.py src/code_indexing_mcp/models.py \
  src/code_indexing_mcp/storage.py tests/test_references.py tests/test_storage.py
git commit -m "feat: resolve indexed references conservatively"
```

### Task 5: Expose `find_references` through MCP and the daemon

**Files:**
- Modify: `src/code_indexing_mcp/errors.py:7-22`
- Modify: `src/code_indexing_mcp/application.py:639-681`
- Modify: `src/code_indexing_mcp/daemon.py:293-330,496-510`
- Modify: `src/code_indexing_mcp/server.py:472-1066`
- Modify: `tests/test_application.py`
- Modify: `tests/test_daemon.py`
- Modify: `tests/test_server.py`

**Step 1: Write failing application, broker, and MCP tests**

Add `find_references` to the focused tool set and auto-registering annotations. Test chunk-ID and path/qualified selectors, language/kind/resolution filters, bounded `limit`, cursor paging, ambiguity details, automatic structural backfill, and JSON round trips through `BrokerApplication`.

Run: `uv run pytest tests/test_application.py tests/test_daemon.py tests/test_server.py -q`

Expected: FAIL because the new method and tool are absent.

**Step 2: Add structured errors**

Add `AMBIGUOUS_SYMBOL`, `UNSUPPORTED_LANGUAGE`, `REFERENCE_INDEX_UNAVAILABLE`, `STALE_CURSOR`, and `INVALID_REFACTOR` to `ErrorCode`. Include declaration candidates and limitations in `CodeIndexingError` details where applicable.

**Step 3: Wire application and daemon**

Construct `ReferenceService` from the existing store in `Application.__init__`. Add an application method that resolves the project, ensures semantic freshness and structural coverage, then delegates.

Add explicit daemon dispatch and a `BrokerApplication.find_references` method that validates `ReferenceResponse`.

**Step 4: Register the MCP tool**

Add a fully documented, bounded tool with `_READS_AND_REGISTERS`. Update `_TOOL_INSTRUCTIONS` and server instructions to teach clients: `find_symbol` selects declarations; `find_references` retrieves uses.

Run: `uv run pytest tests/test_application.py tests/test_daemon.py tests/test_server.py -q`

Run: `uv run mypy src/code_indexing_mcp/application.py src/code_indexing_mcp/daemon.py src/code_indexing_mcp/server.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/code_indexing_mcp/errors.py src/code_indexing_mcp/application.py \
  src/code_indexing_mcp/daemon.py src/code_indexing_mcp/server.py \
  tests/test_application.py tests/test_daemon.py tests/test_server.py
git commit -m "feat: expose reference lookup through MCP"
```

### Task 6: Rename-impact analysis

**Files:**
- Modify: `src/code_indexing_mcp/models.py`
- Modify: `src/code_indexing_mcp/reference_service.py`
- Modify: `src/code_indexing_mcp/application.py`
- Modify: `src/code_indexing_mcp/daemon.py`
- Modify: `src/code_indexing_mcp/server.py`
- Create: `tests/test_refactors.py`
- Modify: `tests/test_daemon.py`
- Modify: `tests/test_server.py`

**Step 1: Write failing rename tests**

Cover declaration edits, direct calls, import/export specifiers, aliased calls that identify the target but do not need spelling changes, inheritance/type/decorator uses, ambiguous receivers, invalid identifiers, duplicate declarations, pagination, and incomplete coverage.

```python
def test_rename_marks_import_name_but_not_local_alias_call_for_edit(tmp_path: Path) -> None:
    service, selector = indexed_python_rename_case(tmp_path)

    analysis = service.analyze_refactor(selector, RenameOperation(new_name="authorize"))

    assert locations(analysis.must_change) == {("auth.py", 1), ("consumer.py", 1)}
    alias_call = next(item for item in analysis.findings if item.written_name == "check")
    assert alias_call.resolution == "exact"
    assert not alias_call.edit_required
```

Run: `uv run pytest tests/test_refactors.py -k rename -q`

Expected: FAIL because no refactor models or analysis method exist.

**Step 2: Add operation and analysis models**

```python
class RenameOperation(FrozenModel):
    kind: Literal["rename"] = "rename"
    new_name: str

class SignatureChangeOperation(FrozenModel):
    kind: Literal["signature_change"] = "signature_change"
    parameters: list[ParameterShape]

RefactorOperation = Annotated[
    RenameOperation | SignatureChangeOperation,
    Field(discriminator="kind"),
]
```

Add `RefactorFinding`, `CompletenessReport`, and `RefactorAnalysis` with `must_change`, `likely_change`, `review`, counts, cursor, and limitations.

**Step 3: Implement rename policy**

Always include the declaration. Mark exact occurrences `must_change` only when their source spelling is the selected declaration name or import/export target. Exact aliased calls remain evidence but require no edit. Likely occurrences go to `likely_change`; unresolved occurrences and structural coverage limitations go to `review`.

Validate new identifiers per language without trying to format or apply edits.

**Step 4: Wire `analyze_refactor` through application, daemon, and MCP**

Use one discriminated `operation` input. Preserve the same selector and paging contract as `find_references`.

Run: `uv run pytest tests/test_refactors.py tests/test_daemon.py tests/test_server.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/code_indexing_mcp/models.py src/code_indexing_mcp/reference_service.py \
  src/code_indexing_mcp/application.py src/code_indexing_mcp/daemon.py \
  src/code_indexing_mcp/server.py tests/test_refactors.py tests/test_daemon.py \
  tests/test_server.py
git commit -m "feat: analyze rename impact"
```

### Task 7: Signature-change analysis

**Files:**
- Modify: `src/code_indexing_mcp/reference_service.py`
- Modify: `tests/test_refactors.py`
- Modify: `tests/test_reference_extraction.py`

**Step 1: Write failing signature compatibility tests**

Cover:

- Added required parameters.
- Removed or renamed keyword parameters.
- Reordered positional parameters.
- Positional-only and keyword-only transitions.
- Variadic/rest parameter changes.
- Python `*args`/`**kwargs` and JS/TS spread calls.
- Constructor calls.
- Type-argument count changes.
- Multiple declaration shapes/overloads.

```python
def test_added_required_parameter_marks_exact_calls_must_change(tmp_path: Path) -> None:
    service, selector = indexed_signature_case(tmp_path, "def send(message): ...", "send('hi')")
    operation = SignatureChangeOperation(
        parameters=[
            ParameterShape(name="message", kind="positional", required=True, position=0),
            ParameterShape(name="timeout", kind="positional", required=True, position=1),
        ]
    )

    analysis = service.analyze_refactor(selector, operation)

    assert len(analysis.must_change) == 1
    assert analysis.must_change[0].reason_code == "missing_required_parameter"
```

Run: `uv run pytest tests/test_refactors.py -k signature -q`

Expected: FAIL because signature operations are not evaluated.

**Step 2: Implement call-shape comparison**

Compare the stored old declaration shape and proposed parameters against each resolved call. Produce stable reason codes for missing required arguments, invalid keywords, positional-order changes, parameter-mode changes, variadic uncertainty, spread uncertainty, and overload ambiguity.

Only deterministic incompatibilities on exact calls are `must_change`. Calls hidden by spreads or ambiguous overloads are `review`.

**Step 3: Run extraction and refactor suites**

Run: `uv run pytest tests/test_reference_extraction.py tests/test_references.py tests/test_refactors.py -q`

Run: `uv run ruff check src/code_indexing_mcp/reference_service.py tests/test_refactors.py`

Expected: PASS.

**Step 4: Commit**

```bash
git add src/code_indexing_mcp/reference_service.py tests/test_refactors.py \
  tests/test_reference_extraction.py
git commit -m "feat: analyze signature changes"
```

### Task 8: Documentation, bundled workflows, and release verification

**Files:**
- Modify: `README.md:260-435,810-855`
- Modify: `src/code_indexing_mcp/skills/impact-analysis/SKILL.md`
- Modify: `src/code_indexing_mcp/skills/feature-dev/SKILL.md`
- Modify: `tests/test_skills.py`
- Modify: `tests/test_benchmark.py`
- Modify: `src/code_indexing_mcp/benchmark.py`

**Step 1: Add failing documentation/workflow contract tests**

Update skill tests to require the normalized `find_references` and `analyze_refactor` tool names in impact-analysis guidance. Extend the deterministic benchmark report with structural-record counts and reference extraction duration, then assert the benchmark path never performs a second embedding pass for references.

Run: `uv run pytest tests/test_skills.py tests/test_benchmark.py -q`

Expected: FAIL until skills and benchmark reporting are updated.

**Step 2: Update README and bundled skills**

Document:

- Both new tools and their caller-visible write annotations.
- Supported languages and deliberately unsupported dynamic cases.
- Exact/likely/unresolved resolution.
- Completeness and limitations.
- Parse-only backfill behavior.
- Rename/signature workflows and pagination.

Update `impact-analysis` to select a declaration first, then call `find_references`/`analyze_refactor`, and retain semantic search for indirect/dynamic evidence. Update `feature-dev` to use the new tools when a change targets an existing symbol.

**Step 3: Run targeted subsystem suites**

Run: `uv run pytest tests/test_reference_extraction.py tests/test_references.py tests/test_refactors.py tests/test_staging.py tests/test_storage.py tests/test_indexing.py tests/test_application.py tests/test_daemon.py tests/test_server.py tests/test_skills.py tests/test_benchmark.py -q`

Expected: PASS with no failures.

**Step 4: Run the full verification matrix**

Use `@superpowers:verification-before-completion` before making any completion claim.

Run: `uv run pytest -q`

Expected: all ordinary tests pass; only the existing opt-in MLX/accelerator/model/memory cases skip.

Run: `uv run ruff check .`

Run: `uv run ruff format --check .`

Run: `uv run mypy src`

Run: `git diff --check`

Expected: every command exits 0.

**Step 5: Commit documentation and qualification**

```bash
git add README.md src/code_indexing_mcp/skills tests/test_skills.py \
  src/code_indexing_mcp/benchmark.py tests/test_benchmark.py
git commit -m "docs: document refactoring analysis tools"
```

**Step 6: Review the complete branch**

Run: `git status --short`

Run: `git log --oneline --decorate main..HEAD`

Run: `git diff --stat main...HEAD`

Expected: clean worktree; one implementation-plan commit plus focused feature commits; no unrelated files.

Use `@superpowers:requesting-code-review`, address actionable findings, rerun the full verification matrix, then use `@superpowers:finishing-a-development-branch` to offer merge/PR/cleanup choices.
