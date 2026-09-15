# Worktree Resolution Follow-up Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Honor explicit checkout paths and preserve registered project identities on ordinary initialization.

**Architecture:** Keep selector classification in ProjectResolver and carry checkout-specific scope through Application. Existing registrations take precedence over worktree sharing; only an unregistered checkout can join a single compatible registration. Legacy migration remains an explicit remove-by-ID followed by initialization workflow.

**Tech Stack:** Python, pytest, Git worktrees, LanceDB, MCP.

Approved in the conversation on 2026-09-15. Worktree: `.worktrees/worktree-resolution-follow-up`; branch: `fix/worktree-resolution-follow-up`; base: `6f5ec68`.

## Tasks

- [x] Establish the affected-module test baseline in the new worktree (303 passed).
- [x] Add failing resolver and application regressions for explicit paths in both directions, multiple selected checkouts, and unchanged ID/name/implicit behavior.
- [x] Update `src/code_indexing_mcp/projects.py` and `Application._scope_checkouts` to preserve explicit checkout scope and deduplicate by ID plus root.
- [x] Add failing application regressions for split registrations in both insertion orders, ambiguous sharing, and explicit legacy migration preserving the survivor's slots and data.
- [x] Update `Application._initialize_registration` and `_shared_registration`: preserve registered markers, never automatically delete a registration, and reject ambiguous candidates before mutation.
- [x] Add MCP regressions and change `create_server.init_project` to obtain client roots without first discovering/re-registering them.
- [x] Update `worktree_warnings`, MCP descriptions, and README with selector precedence and explicit migration guidance.
- [x] Format, run targeted tests, lint, typecheck, and run the full test suite. Review the final diff.

## Test coverage

- `tests/test_projects.py`: explicit main/worktree paths override ambient roots and cwd; ID/name and implicit resolution follow the requesting checkout; path aliases and selector precedence remain supported.
- `tests/test_application.py`: status, indexing, search, and references use the selected checkout/slot; multiple explicit paths and multi-root ID/name searches preserve all selected branches; split registrations retain IDs, slots, and queryable data after reinitializing either checkout in both insertion orders.
- `tests/test_application.py`: legacy migration removes only the explicitly selected duplicate before reinitializing its checkout; the unique remaining compatible registration survives. Ambiguous registration choices fail without mutation.
- `tests/test_server.py`: conflicting explicit paths and roots work through MCP; initialization does not discover and recreate a removed duplicate before joining the survivor.
- Existing `tests/test_storage.py` coverage checks warning behavior and shared-repository storage compatibility.

## Verification

Use the worktree's virtual environment (CPU and TUI extras installed). `UV_CACHE_DIR` may point to a writable temporary cache and `uv run --no-sync` preserves the prepared environment.

```sh
uv run pytest tests/test_projects.py tests/test_application.py tests/test_server.py tests/test_storage.py
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -n auto
```

All mutating reproductions use disposable repositories and isolated index storage. Live project registrations are outside this implementation's scope.

## Findings during implementation

- MCP wrappers resolved the correct checkout but then forwarded only IDs with ambient roots. Preserve resolved roots through search, example search, symbol lookup, file outlines, manual indexing, automatic indexing retries, and freshness checks. Regression coverage exercises manual and lazy modes, both checkout directions, and a second refresh after a commit.
- Reinitializing split worktrees nested below the main root also requires checking for an existing matching ID/root pair before overlap rejection. Both insertion orders now have nested and sibling worktree coverage. A separate regression ensures relocation cannot introduce a new overlap without `allow_overlap`.
- Canonical-root binding must stop before cwd fallback while keeping the registered scan configuration. The existing scan-configuration freshness regression verifies this.
- New MCP routing tests disable automatic maintenance so unrelated startup compaction cannot race the manual-index request; existing maintenance contention tests remain enabled.

## Final results

- Full parallel suite: **2,141 passed, 10 skipped** in 197.52 seconds (`pytest -n auto --tb=short`). Skips cover optional accelerator/real-model environments, a Linux-only runtime check, and an existing fixture without a selectable declaration.
- Formatting: all 147 files pass `ruff format --check .`; `ruff check .` passes.
- Type checking: `mypy src` passes for all 71 source files.
- Independent code review has no remaining findings after adding the relocation regression and narrowing the overlap exemption to an already registered ID/root pair.
- The first sandboxed full-suite run encountered local socket permission errors. The complete rerun with permitted local socket access passed without code changes.
- Implemented on `fix/worktree-resolution-follow-up` in the requested worktree.
