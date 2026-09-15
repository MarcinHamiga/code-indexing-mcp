# Scoped Discovery Errors Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Explicit MCP queries remain usable when discovery of an unselected client root fails.

**Architecture:** Keep discovery of new client roots, but defer its failures until explicit selectors can be resolved. Only failures belonging to the selected checkouts propagate. Freshness waits use those same checkout roots, including advertised subdirectories of a selected checkout. Implicit requests and initialization retain their ambiguity errors.

**Tech Stack:** Python, FastMCP, asyncio/AnyIO, pytest, Git worktrees, LanceDB.

The user authorized this fix following the PR #68 review. Work in the existing
`fix/worktree-resolution-follow-up` worktree.

## Design

Filtering discovery failures after scope resolution preserves automatic discovery
for explicit ID/name requests from a new worktree. Skipping all discovery for an
explicit request would incorrectly fall back to the canonical checkout. Removing
the registration ambiguity guard would again arbitrarily select a registration.

## Tasks

1. Add integration regressions in `tests/test_server.py` using disposable Git
   worktrees, `TinyEmbedder`, and an in-memory MCP session. Cover explicit project
   and declaration selectors, search helpers, selected ambiguity, and automatic
   discovery of a uniquely compatible worktree.
2. Run the new tests and confirm that explicit queries fail with the reviewed
   `AMBIGUOUS_PROJECT` error before modifying production code.
3. In `src/code_indexing_mcp/server.py`, add a scope-aware discovery helper used
   by project/declaration-scoped tools. Collect root discovery failures, resolve
   explicit scope, and propagate only relevant failures. Preserve the original
   discovery failure for an explicitly selected unmarked path; retain validation
   errors for invalid selectors and filter combinations.
4. Restrict `_prepare_startup_projects` waits to selected checkouts and their
   advertised subdirectories, retaining background indexing behavior.
5. Run the new regressions, affected module tests, formatting, lint, type checking,
   and the full parallel suite. Inspect the final diff.

## Verification

Use the existing worktree virtual environment; set `UV_CACHE_DIR` to a writable
temporary directory when running under the sandbox.

```sh
uv run --no-sync pytest tests/test_server.py -k 'ambiguous_client_root or discovers_matching_worktree' --tb=short
uv run --no-sync pytest tests/test_projects.py tests/test_application.py tests/test_server.py tests/test_storage.py -n auto --tb=short
uv run --no-sync ruff format .
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync mypy src
uv run --no-sync pytest -n auto --tb=short
```

## Index limitations

Navigation uses the refreshed exact-head review archive. Four files have parse
gaps, including `storage.py`; the server and its tests are indexed. Structural
references for nested MCP tools are reported as unresolved, so their call sites
are inspected directly after indexed navigation.

## Implementation notes

- The shared server helper handles single-project, multi-project, all-project,
  and declaration selectors. It still discovers new roots before ID/name binding.
- Chunk selectors use their logical project routing prefix without loading chunk
  content in the canonical checkout. The reference operation validates the chunk
  against its selected checkout, including malformed and unknown chunk IDs.
- The 42 focused regression/control cases pass. They cover both discovery-root
  orders for a noncanonical worktree chunk, explicit and implicit ambiguity,
  unknown selectors, invalid filter combinations, and selected subdirectories.
- Independent review findings about chunk routing and preservation of validation
  errors were reproduced, covered by tests, and corrected. The final review has
  no remaining actionable findings.

## Final verification

- Full parallel suite: **2,183 passed, 10 skipped** in 165.22 seconds, with local
  socket access for daemon integration tests. Skips cover optional accelerator
  and real-model environments, a Linux-only runtime-directory test, and an
  existing fixture without a selectable declaration.
- `ruff format --check .`, `ruff check .`, `mypy src`, and `git diff --check` pass.
