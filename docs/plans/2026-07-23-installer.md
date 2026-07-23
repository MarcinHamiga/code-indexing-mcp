# Code Indexing MCP Installer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an idempotent cross-platform installer with interactive harness selection and move
new project markers from `.incode` to `.ci-mcp` without breaking legacy marker discovery.

**Architecture:** A standalone `install.py` owns clone/update, environment synchronization,
interactive selection, and targeted user-configuration edits. A small `install.sh` bootstraps
the Python file for POSIX users. Existing application code gains a new marker directory plus a
read-only compatibility path for legacy markers.

**Tech Stack:** Python 3 standard library, POSIX shell, Git, uv, pytest, Ruff, MyPy.

### Task 1: Move project markers to `.ci-mcp`

**Files:**
- Modify: `tests/test_projects.py`
- Modify: `tests/test_application.py`
- Modify: `tests/test_indexing.py`
- Modify: `tests/test_scanner.py`
- Modify: `src/incode_mcp/projects.py`
- Modify: `src/incode_mcp/scanner.py`

**Step 1: Write the failing tests**

Change marker assertions to `.ci-mcp`, add a project-resolution test with a hand-built legacy
`.incode/project.toml`, and assert both marker directories are excluded from scans.

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_projects.py tests/test_application.py tests/test_indexing.py tests/test_scanner.py -v`

Expected: failures show new markers are still written under `.incode` and `.ci-mcp` is scanned.

**Step 3: Write the minimal implementation**

Set `MARKER_DIRECTORY = ".ci-mcp"`, add `LEGACY_MARKER_DIRECTORY = ".incode"`, resolve the new
marker first and the legacy marker second, and hard-exclude both directory names.

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_projects.py tests/test_application.py tests/test_indexing.py tests/test_scanner.py -v`

Expected: all selected tests pass.

### Task 2: Add format-preserving configuration editors

**Files:**
- Create: `install.py`
- Create: `tests/test_installer.py`

**Step 1: Write failing JSONC editor tests**

Load `install.py` as a module and specify a `merge_json_object_entry()` API. Cover an empty file,
an existing target entry, comments, trailing commas, nested objects, and preservation of
unrelated text.

**Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_installer.py -v`

Expected: import or attribute failure because the installer/editor does not exist.

**Step 3: Implement the minimal JSONC editor**

Add trivia skipping, JSON string parsing, nested value scanning, object member discovery,
formatted insertion/replacement, atomic writes, and changed-file backups using only the
standard library.

**Step 4: Run the JSONC tests**

Run: `.venv/bin/pytest tests/test_installer.py -v`

Expected: JSONC tests pass.

**Step 5: Write failing TOML editor tests**

Specify `merge_codex_server()` behavior for creating a config, replacing only the named MCP
table and its subtables, and preserving unrelated settings and comments.

**Step 6: Implement and verify the TOML editor**

Run: `.venv/bin/pytest tests/test_installer.py -v`

Expected: all editor tests pass.

### Task 3: Add harness adapters and selection

**Files:**
- Modify: `install.py`
- Modify: `tests/test_installer.py`

**Step 1: Write failing harness tests**

Cover menu parsing for numbers, slugs, duplicates, and `all`. Assert the combined Codex entry
and each harness's command schema and environment-aware global configuration path.

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_installer.py -v`

Expected: missing selection and harness-configuration APIs.

**Step 3: Implement minimal adapters**

Add the six choices, path resolution, harness-specific entries, per-harness error isolation,
and combined Codex CLI/Desktop labeling.

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_installer.py -v`

Expected: all harness and editor tests pass.

### Task 4: Add clone, update, sync, and command-line workflow

**Files:**
- Modify: `install.py`
- Modify: `tests/test_installer.py`

**Step 1: Write failing repository lifecycle tests**

Use a temporary local bare Git remote to verify first-run cloning, later fast-forward updates,
mismatched-target rejection, and dirty-checkout rejection.

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_installer.py -v`

Expected: repository lifecycle functions are missing.

**Step 3: Implement repository lifecycle and CLI**

Add argument parsing, prerequisite checks, clone/update validation, `uv sync --locked`, virtual
environment executable resolution, interactive/noninteractive selection, summary output, and
exit status handling.

**Step 4: Run tests and smoke-test help**

Run: `.venv/bin/pytest tests/test_installer.py -v`

Run: `.venv/bin/python install.py --help`

Expected: tests pass and help lists install directory, repository URL, and harness options.

### Task 5: Add the POSIX bootstrap and documentation

**Files:**
- Create: `install.sh`
- Modify: `README.md`

**Step 1: Write the bootstrap**

Make `install.sh` use an adjacent `install.py` when present, otherwise download the raw installer
with curl or wget, attach interactive input to `/dev/tty`, and clean its temporary directory.

**Step 2: Validate shell syntax**

Run: `sh -n install.sh`

Expected: exit status 0.

**Step 3: Update user documentation**

Document one-line POSIX installation, Windows download-and-run instructions, update semantics,
the six menu choices, noninteractive selection, install-directory overrides, and `.ci-mcp`
marker behavior.

### Task 6: Full verification

**Files:**
- Review all changed files

**Step 1: Run all tests**

Run: `.venv/bin/pytest`

Expected: all non-model tests pass and only the opt-in model test is skipped.

**Step 2: Run static checks**

Run: `.venv/bin/ruff check .`

Run: `.venv/bin/ruff format --check .`

Run: `.venv/bin/mypy src`

Expected: all commands exit 0.

**Step 3: Run installer checks**

Run: `sh -n install.sh`

Run: `.venv/bin/python install.py --help`

Expected: both commands exit 0.

**Step 4: Review the diff**

Run: `git diff --check`

Run: `git status --short`

Expected: no whitespace errors and only intended files are modified.
