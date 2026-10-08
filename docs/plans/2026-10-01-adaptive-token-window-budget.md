# Adaptive Token Window Budget Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent otherwise indexable source files from failing when slicing changes the composed input's token count.

**Architecture:** Keep the existing offset-based planner as the fast path. When exact composed-input measurements exceed the limit, reduce the candidate's content budget by the measured overflow and replan until all windows fit. Keep prefix accounting, deterministic offsets, overlap clamping, and the window-count cap.

**Tech Stack:** Python, pytest, tokenizer-agnostic encoding callbacks, optional tokenizers regression test.

### Task 1: Reproduce boundary inflation

**Files:** `tests/test_token_batching.py`, `tests/test_embedding.py`.

1. Add a deterministic encoding whose composed input costs more tokens than the separately measured prefix and content.
2. Add a real WordPiece regression where slicing into a continuation changes one token into two.
3. Assert exact input limits, full character coverage, stable retries, unchanged normal plans, and bounded fan-out.
4. Run the new cases and verify the current planner raises the reported error.

### Task 2: Adapt the content budget

**Files:** `src/code_indexing_mcp/token_batching.py`.

1. Tokenize the candidate content once and retain its original offsets.
2. Plan and measure every complete model input.
3. If any input overflows, subtract the largest overflow from the content budget and repeat.
4. Return only a completely measured, fitting plan. Preserve explicit errors for exhausted budgets and excessive window counts.
5. Run planner and embedding tests, including a callback that rejects oversized model inputs.

### Task 3: Verify

1. Run `uv run ruff format .`.
2. Run `uv run ruff format --check .`, `uv run ruff check .`, and `uv run mypy src`.
3. Run `uv run pytest -n auto`.
4. Review the diff and record results. Preserve the user's existing README changes.

Use `UV_CACHE_DIR=/private/tmp/code-indexing-mcp-uv-cache` for verification in this sandbox, where the default uv cache is not writable.

### Results

- The original implementation failed all four initial regression cases with the reported composed-input overflow.
- Planner and embedding tests: 46 passed, including the real WordPiece regression with and without a prefix.
- Cached Jina code tokenizer stress check: 450 deterministic, 4,096-character identifier, dense literal, and Unicode inputs. The original planner overflowed on 78 inputs; adaptive planning handled all 450. All 1,650 resulting windows covered their source and measured at or below 1,024 tokens.
- Formatting, lint, type checking, and diff whitespace checks passed.
- Full suite: 2,329 passed and 10 skipped in 190.59 seconds. The initial sandbox run blocked Unix socket binding in daemon tests; the completed run used approved OS access. Optional model, memory, and accelerator acceptance gates were skipped by their existing opt-in conditions.
- Read-only review found no actionable issues. Existing successful cache entries remain compatible because fitting plans do not change; previously overflowing candidates never produced cache entries.
- Deployment requires restarting/updating the server, then `index_project(force=true)` for affected projects: unchanged failed-file records can be skipped by normal incremental indexing.

### Follow-up: retrieve original extracted chunks

The user requested that `get_chunk` return the original extracted chunk when a smaller embedding window matches. Preserve window vectors and snippets for ranking, but record the original byte bounds on each window. Reconstruct the full chunk from overlapping stored windows in the same physical partition and content generation, checking coverage and overlap consistency. Do not consult the current checkout or duplicate the original text per vector.

- `models.py`, `staging.py`, `indexing.py`: carry nullable original byte bounds for windowed chunks. Unwindowed chunks retain their current representation.
- `storage.py`: bump the index schema; reconstruct complete payloads from bounded, projected sibling queries; coalesce window metadata in symbol/declaration lookups.
- `search.py`: deduplicate matches from the same source chunk and span complete declarations in outlines.
- `tests/test_window_retrieval.py`: end-to-end adaptive planning, real LanceDB retrieval, Unicode offsets, early/late/vector matches, selectors, snapshot independence, and reindexing. Add coverage for extracted parts, duplicate declarations, corruption, and cache reuse.
- Verify relevant storage/indexing/search/reference suites, formatting, lint, type checking, then the full suite with approved Unix socket access.

### Follow-up results

- Every adaptive window retrieves the complete original extracted chunk from stored UTF-8 slices, even after deleting the checkout file. Reconstruction pins a LanceDB table version and rejects missing coverage or inconsistent overlaps.
- Window identity includes file, extractor kind/name/part, original content bounds, and file content hash. Separate declarations and extractor parts remain separate retrieval units; reindexing invalidates old IDs and cache reuse updates source offsets.
- Search and symbol/declaration lookups collapse sibling windows. Search refills physical candidate pages when needed, including identical source in multiple pinned checkouts, so windows do not exhaust the logical result limit. Outlines preserve their existing policy for repeated names.
- Metadata schema version is now 6. Existing partitions need rebuilding after updating/restarting the server; `index_project(force=true)` rebuilds the required metadata and retries previously failed files.
- Relevant storage, search, indexing, references, changed-symbols, and staging suites initially passed all 352 tests. The full suite passed 2,345 tests with 10 existing environment/opt-in skips in 226.68 seconds. After the final pinned-checkout refill correction, all 51 focused retrieval/search tests passed, including both new multi-checkout regressions.
- Final formatter, lint, type checks (72 source files), and diff whitespace checks passed. Read-only review found no remaining actionable issues. The user's existing README changes remain untouched.
- PR isolation: rebased only this fix onto `origin/main`, excluding the two unrelated commits from PR #72. The isolated branch passed formatting, lint, mypy (71 source files), and the full CI test gate: 2,322 passed, 10 skipped in 147.40 seconds. Existing README contents were restored byte-for-byte and excluded from the commit.
