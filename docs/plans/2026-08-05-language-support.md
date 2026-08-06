# Next Languages Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Tree-sitter indexing support for Go, Terraform, Rust, C, C++, and Lua.

**Architecture:** Extend the existing synchronized extension map, default include list,
language-name type, grammar registry, and packaged query set. Use dedicated grammar
packages for every requested language: tree-sitter-go, tree-sitter-hcl,
tree-sitter-rust, tree-sitter-c, tree-sitter-cpp, and tree-sitter-lua.
Preserve the extractor's generic definition traversal and validate each grammar with
representative fixtures and the committed fingerprint snapshot.

**Tech Stack:** Python 3.12, Tree-sitter 0.25, dedicated tree-sitter wheels,
pytest, Ruff, mypy, uv.

### Task 1: Add failing language coverage tests

**Files:**
- Modify: tests/test_scanner.py
- Modify: tests/test_extractor.py
- Modify: tests/test_extractor_equivalence.py

Write tests that require all six new language names, extensions, grammar/query
resources, and representative symbols to be present. Run the focused tests and
confirm they fail because the new languages are not registered.

### Task 2: Register dependencies and language mappings

**Files:**
- Modify: pyproject.toml
- Modify: uv.lock
- Modify: src/code_indexing_mcp/scanner.py
- Modify: src/code_indexing_mcp/models.py
- Modify: src/code_indexing_mcp/extractor.py

Add dedicated package dependencies, extension mappings, default include patterns,
LanguageName literals, imports, and Language(...) registrations. Resolve the
lockfile with uv and run the focused registry tests.

### Task 3: Add Tree-sitter extraction queries

**Files:**
- Create: src/code_indexing_mcp/queries/go.scm
- Create: src/code_indexing_mcp/queries/terraform.scm
- Create: src/code_indexing_mcp/queries/rust.scm
- Create: src/code_indexing_mcp/queries/c.scm
- Create: src/code_indexing_mcp/queries/cpp.scm
- Create: src/code_indexing_mcp/queries/lua.scm

Add minimal grammar-specific captures for types, functions, methods, constants,
macros, fields, and other definitions represented by the grammars. Run extraction
tests and adjust only for actual grammar node names.

### Task 4: Add corpus fixtures and regenerate the snapshot

**Files:**
- Create: tests/fixtures/extractor_corpus/sample.go
- Create: tests/fixtures/extractor_corpus/sample.tf
- Create: tests/fixtures/extractor_corpus/sample.rs
- Create: tests/fixtures/extractor_corpus/sample.c
- Create: tests/fixtures/extractor_corpus/sample.cpp
- Create: tests/fixtures/extractor_corpus/sample.lua
- Modify: tests/fixtures/extractor_snapshot.json

Use small valid examples that exercise the new queries. Regenerate the snapshot
with python3 -m tests.test_extractor_equivalence only after the equivalence test
has demonstrated the expected failure and the focused extraction tests pass.

### Task 5: Verify and review

Run:

    uv run pytest tests/test_scanner.py tests/test_extractor.py tests/test_extractor_equivalence.py
    uv run ruff check src tests
    uv run mypy
    uv run pytest

Review the final diff for synchronized language lists, package lock consistency,
query resource packaging, and unchanged behavior for existing languages.
