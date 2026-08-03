# Index Freshness Monitoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep eager and lazy indexes fresh after filesystem changes while preserving manual mode.

**Architecture:** Add a shared stat-level freshness check to the application/daemon boundary.
Eager mode consumes debounced filesystem events through a single-flight coordinator queue; lazy
mode invokes the freshness check on project-scoped queries. Git repositories use Git's standard
ignore engine and non-Git projects keep the current pathspec fallback.

**Tech Stack:** Python 3.12, AnyIO/asyncio, FastMCP, watchfiles, Git, pytest.

### Task 1: Git-standard exclusions

**Files:**
- Modify: `src/code_indexing_mcp/scanner.py`
- Test: `tests/test_scanner.py`

1. Add a failing test proving a supported source listed in `.git/info/exclude` is omitted.
2. Run that test and confirm it fails because the file is returned.
3. Add one batched `git check-ignore --stdin -z --no-index` helper with safe fallback.
4. Run scanner tests and confirm they pass.

### Task 2: Freshness and stale status

**Files:**
- Modify: `src/code_indexing_mcp/application.py`
- Modify: `src/code_indexing_mcp/daemon.py`
- Test: `tests/test_application.py`
- Test: `tests/test_daemon.py`

1. Add failing create/modify/delete freshness and stale-status tests.
2. Add a failing broker RPC test for the same check.
3. Run the focused tests and confirm the methods/state are missing.
4. Implement `project_is_stale`, stale status mapping, daemon dispatch, and broker proxy.
5. Run application and daemon tests and confirm they pass.

### Task 3: Lazy query refresh

**Files:**
- Modify: `src/code_indexing_mcp/server.py`
- Test: `tests/test_server.py`

1. Add a failing test that indexes a symbol, edits it, and queries the replacement in lazy mode.
2. Confirm the replacement is absent and the old symbol remains.
3. Make completed coordinator jobs check freshness before reuse.
4. Confirm lazy modification/create/delete tests pass and manual mode remains unchanged.

### Task 4: Eager background monitoring

**Files:**
- Modify: `src/code_indexing_mcp/server.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tests/test_server.py`

1. Add failing eager tests for post-index changes and a second change during an active refresh.
2. Confirm no automatic refresh occurs.
3. Add `watchfiles`, one watcher per eager root, and a bounded dirty queue.
4. Confirm eager tests pass without timing-dependent sleeps beyond bounded condition polling.

### Task 5: Documentation and verification

**Files:**
- Modify: `src/code_indexing_mcp/server.py`
- Modify: `src/code_indexing_mcp/installer/settings_spec.py`
- Modify: `README.md`
- Test: `tests/test_installer_settings_spec.py`

1. Update mode/status descriptions and remove the statement that watching is excluded.
2. Run focused verification:
   `PYTHONPATH=src python -m pytest -q tests/test_scanner.py tests/test_application.py tests/test_server.py tests/test_daemon.py`.
3. Run full pytest, Ruff, mypy, and `git diff --check`.
4. Review the final diff for concurrency, shutdown, RPC, and documentation consistency.
