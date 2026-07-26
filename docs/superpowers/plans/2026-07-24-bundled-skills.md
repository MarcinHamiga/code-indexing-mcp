# Bundled Skills for code-indexing-mcp — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship 4 agent skills inside the Python package, symlink them into harness skill directories via `install.py`, and add always-on index-first guidance to the server's `instructions` string.

**Architecture:** Skills live as `SKILL.md` folders under `src/incode_mcp/skills/` (same bundling pattern as `queries/*.scm`, rides along with hatchling's `packages = ["src/incode_mcp"]`). The installer symlinks them into each skill-capable harness's user skill directory, pointing into the cloned repo so `git pull` updates refresh them. The FastMCP `instructions` string becomes a module-level constant with index-first usage guidance.

**Tech Stack:** Python 3.12, FastMCP (`mcp>=1.27`), hatchling, pytest, stdlib-only installer.

Spec: `docs/superpowers/specs/2026-07-24-bundled-skills-design.md`

## Global Constraints

- All skills reference tools ONLY with the `mcp__code-indexing-mcp__*` prefix (never `mcp__incode__*`).
- `install.py` stays stdlib-only; no new dependencies anywhere.
- Skill frontmatter keeps the existing shape: `name`, `description`, `type: prompt`, `whenToUse`, `arguments` (single-line values only — tests parse them with regex, no YAML dependency).
- The 4 bundled skills are exactly: `codebase-exploration`, `feature-dev`, `impact-analysis`, `indexed-review`.
- Harness skill-directory map: `claude-code` → `$CLAUDE_CONFIG_DIR/skills` (default `~/.claude/skills`); `codex`, `kimi-code` → `~/.agents/skills`; `opencode` → `$XDG_CONFIG_HOME/opencode/skills` (default `~/.config/opencode/skills`); `claude-desktop`, `kilocode` → unsupported, skip.
- Test commands: `uv run pytest tests/<file> -v`; full suite `uv run pytest`. Lint: `uv run ruff check .`; types: `uv run mypy`.

---

### Task 1: Index-first server instructions

**Files:**
- Modify: `src/incode_mcp/server.py` (module constants near line 19, constructor at line 187-192)
- Test: `tests/test_server.py` (append; reuses the `TinyEmbedder` / `Application` pattern at lines 15-23 and 80-86)

**Interfaces:**
- Produces: `SERVER_INSTRUCTIONS: str` module constant in `incode_mcp.server`; `create_server(app).instructions` returns it (FastMCP exposes `.instructions`, confirmed against installed SDK `mcp/server/fastmcp/server.py:249`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server.py`:

```python
def test_server_instructions_guide_index_first_usage(tmp_path: Path) -> None:
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    server = create_server(app)

    instructions = server.instructions

    assert instructions is not None
    for tool in (
        "search_code",
        "find_symbol",
        "file_outline",
        "get_chunk",
        "project_status",
        "index_project",
    ):
        assert tool in instructions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py::test_server_instructions_guide_index_first_usage -v`
Expected: FAIL — `"search_code" not in "Local Tree-sitter code indexing and hybrid search."`

- [ ] **Step 3: Add the SERVER_INSTRUCTIONS constant and use it**

In `src/incode_mcp/server.py`, add near the top of the module (with the other module-level definitions, before `class AutoIndexingMCP`):

```python
SERVER_INSTRUCTIONS = (
    "Local Tree-sitter code indexing and hybrid search. "
    "When exploring code, prefer these index tools over grep-style file reading: "
    "search_code (semantic natural-language queries), find_symbol (definitions and call "
    "sites), file_outline (file structure before reading), get_chunk (exact code for a "
    "search hit). Check list_projects/project_status for index freshness first and run "
    "index_project if the index is missing or stale."
)
```

Change the `AutoIndexingMCP.__init__` super call (currently line 187-192) from:

```python
        super().__init__(
            "code-indexing-mcp",
            instructions="Local Tree-sitter code indexing and hybrid search.",
            json_response=True,
            lifespan=self._lifespan,
        )
```

to:

```python
        super().__init__(
            "code-indexing-mcp",
            instructions=SERVER_INSTRUCTIONS,
            json_response=True,
            lifespan=self._lifespan,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS (whole file, including the new test)

- [ ] **Step 5: Commit**

```bash
git add src/incode_mcp/server.py tests/test_server.py
git commit -m "feat: guide index-first tool usage via server instructions"
```

---

### Task 2: Bundle the four skills in the package

**Files:**
- Create: `src/incode_mcp/skills/codebase-exploration/SKILL.md`
- Create: `src/incode_mcp/skills/feature-dev/SKILL.md`
- Create: `src/incode_mcp/skills/impact-analysis/SKILL.md`
- Create: `src/incode_mcp/skills/indexed-review/SKILL.md`
- Test: `tests/test_skills.py` (new)

**Interfaces:**
- Consumes: nothing from Task 1 (independent).
- Produces: `src/incode_mcp/skills/<name>/SKILL.md` × 4 — the exact directory Task 3's installer symlinks and Task 4's wheel check inspects.

- [ ] **Step 1: Write the failing test**

Create `tests/test_skills.py`:

```python
"""Validation for the skills bundled under src/incode_mcp/skills/."""

import re
from pathlib import Path

import pytest

SKILLS_DIR = Path(__file__).resolve().parent.parent / "src" / "incode_mcp" / "skills"
EXPECTED_SKILLS = {
    "codebase-exploration",
    "feature-dev",
    "impact-analysis",
    "indexed-review",
}


def _skill_dirs() -> list[Path]:
    return sorted(path for path in SKILLS_DIR.iterdir() if path.is_dir())


def test_all_expected_skills_are_bundled() -> None:
    assert {path.name for path in _skill_dirs()} == EXPECTED_SKILLS


@pytest.mark.parametrize("skill_dir", _skill_dirs() if SKILLS_DIR.is_dir() else [], ids=lambda p: p.name)
def test_skill_has_valid_frontmatter(skill_dir: Path) -> None:
    skill_md = skill_dir / "SKILL.md"
    assert skill_md.is_file(), f"missing {skill_md}"
    text = skill_md.read_text(encoding="utf-8")
    frontmatter = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
    assert frontmatter is not None, f"{skill_md} has no frontmatter block"
    name = re.search(r"^name: (.+)$", frontmatter.group(1), re.MULTILINE)
    description = re.search(r"^description: (.+)$", frontmatter.group(1), re.MULTILINE)
    assert name is not None and name.group(1).strip() == skill_dir.name
    assert description is not None and description.group(1).strip()


@pytest.mark.parametrize("skill_dir", _skill_dirs() if SKILLS_DIR.is_dir() else [], ids=lambda p: p.name)
def test_skill_references_only_code_indexing_mcp_tools(skill_dir: Path) -> None:
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "mcp__incode__" not in text
    assert "mcp__code-indexing-mcp__" in text
```

Note: the parametrization is evaluated at collection time, so before the skills exist these two parametrized tests collect zero cases (pytest reports them as skipped/empty) while `test_all_expected_skills_are_bundled` fails — that is the expected red state.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_skills.py -v`
Expected: FAIL — `test_all_expected_skills_are_bundled` (`SKILLS_DIR` does not exist / empty set)

- [ ] **Step 3: Create `src/incode_mcp/skills/codebase-exploration/SKILL.md`**

````markdown
---
name: codebase-exploration
description: Explore and navigate a codebase by narrowing down with the code-indexing-mcp index tools (search_code, find_symbol, file_outline, get_chunk) before reading any file
type: prompt
whenToUse: When the user asks where something is, how something works, or wants to understand or navigate code, and the code-indexing-mcp tools are available
arguments:
  - question
---

You are exploring a codebase to answer: `$question`. If the question is missing or too vague to search against, ask the user with the AskUserQuestion tool before starting.

The core rule of this skill: **narrow down with the index first; read files last.** Do not fall back to Grep/Glob/Read for code exploration while the code-indexing-mcp tools are available.

## 1. Ensure the index is ready

1. Call `mcp__code-indexing-mcp__project_status` for the target project. If the project is unknown, call `mcp__code-indexing-mcp__list_projects` and, if needed, `mcp__code-indexing-mcp__init_project`.
2. If the index is missing or stale relative to the files in question, call `mcp__code-indexing-mcp__index_project` first. Tell the user if indexing will take a moment.

## 2. Narrow down semantically

- `mcp__code-indexing-mcp__search_code` — issue several targeted natural-language queries derived from the question (the concept, the mechanism, the data it touches). Not one broad query.
- `mcp__code-indexing-mcp__find_symbol` — once you have a concrete function/class/module name, jump straight to its definitions and references instead of more searching.

Keep a shortlist of candidate files and symbols; discard paths the results rule out.

## 3. Outline before reading

- `mcp__code-indexing-mcp__file_outline` — get each candidate file's structure and decide which symbols matter. Never dump a whole file just to find one thing.

## 4. Read only what matters

- `mcp__code-indexing-mcp__get_chunk` — pull the full body of the specific chunks that answer the question.
- Use Read/Grep only for content the index does not cover (non-code files, unindexed paths, exact literal matches semantic search cannot express) — and say so when you do.

## 5. Answer

Answer the question with `path:line` references for every claim, the shortest explanation that is complete, and an explicit note of anything the index could not cover so the user knows what was not verified.
````

- [ ] **Step 4: Create `src/incode_mcp/skills/impact-analysis/SKILL.md`**

````markdown
---
name: impact-analysis
description: Map the blast radius of changing or removing a symbol, module, or API using code-indexing-mcp (find_symbol for definitions and call sites, search_code for indirect usages) before any edit
type: prompt
whenToUse: When the user plans a rename, refactor, signature change, move, or removal and wants to know what it affects, and the code-indexing-mcp tools are available
arguments:
  - target
---

You are analyzing the impact of changing: `$target` (a symbol, module, or API plus the intended change). If the target or the intended change is unclear, ask the user with the AskUserQuestion tool before starting.

The core rule of this skill: **map usages with the index tools, not with grep.**

## 1. Ensure the index is ready

1. Call `mcp__code-indexing-mcp__project_status` for the target project. If the project is unknown, call `mcp__code-indexing-mcp__list_projects` and, if needed, `mcp__code-indexing-mcp__init_project`.
2. If the index is missing or stale relative to the files the change touches, call `mcp__code-indexing-mcp__index_project` first. Tell the user if indexing will take a moment.

## 2. Resolve the target

- `mcp__code-indexing-mcp__find_symbol` with an exact match on the target name to locate every definition. If several unrelated symbols match, ask the user which one is meant — do not guess.
- `mcp__code-indexing-mcp__get_chunk` — read the definition itself so the analysis is grounded in what the symbol actually does and exposes.

## 3. Map direct usages

- `mcp__code-indexing-mcp__find_symbol` — collect all references and call sites of the resolved symbol.
- For each distinct file in the results, `mcp__code-indexing-mcp__file_outline` to place the usage in context, and `mcp__code-indexing-mcp__get_chunk` where the exact call matters (signature changes, argument reordering).

## 4. Hunt indirect usages

Semantic search finds what name-based lookup misses:

- `mcp__code-indexing-mcp__search_code` — targeted queries for re-exports and wrappers around the symbol, duck-typed or dynamic call sites, serialization or config that names it, and tests that exercise it indirectly.
- Grep is allowed only for literal string occurrences of the exact name (reflection, registries, config keys) — say so when you use it.

## 5. Report

Produce a structured impact report:

- **Target and intended change**, with `path:line` of the definition.
- **Must change** — call sites that break, each with `path:line` and why.
- **Should review** — indirect or dynamic usages that may break silently.
- **Tests** — the test files covering the target, found via the index.
- **Change checklist** — the ordered list of edits the change requires, grounded in the findings above.

Report only what you actually inspected through the tools; list blind spots the index could not cover.
````

- [ ] **Step 5: Create `src/incode_mcp/skills/feature-dev/SKILL.md`**

Copy the current `~/.agents/skills/feature-dev/SKILL.md` verbatim — its tool references already use the `mcp__code-indexing-mcp__*` prefix. Full content:

````markdown
---
name: feature-dev
description: Use when the user asks to implement a new feature, enhancement, or behavior change in a codebase and the code-indexing-mcp tools are available
type: prompt
whenToUse: When the user asks for a feature to be designed and implemented, and the code-indexing-mcp tools are available
arguments:
  - request
---

You are running a structured feature-development workflow. Follow the phases in order. Do not skip or merge phases.

The feature request is `$request`. If it is missing or too vague to plan against (no clear goal or acceptance signal), ask the user with the AskUserQuestion tool before starting Phase 1.

## Phase 1 — Gather current state with code-indexing-mcp

The core rule of this phase: **the code-indexing-mcp tools are the primary navigation mechanism. Do not fall back to Grep/Glob/Read for code exploration while they are available.**

1. Call `mcp__code-indexing-mcp__project_status` for the target project. If the project is unknown, call `mcp__code-indexing-mcp__list_projects` and, if needed, `mcp__code-indexing-mcp__init_project`.
2. If the index is missing or stale relative to the files the feature will touch, call `mcp__code-indexing-mcp__index_project` first. Tell the user if indexing will take a moment.
3. Explore the relevant parts of the codebase:
   - `mcp__code-indexing-mcp__search_code` — issue several targeted semantic queries derived from the feature request (the area being changed, the patterns it must follow, the tests that cover it). Not one broad query.
   - `mcp__code-indexing-mcp__file_outline` — get a file's structure before reading it; decide which parts matter instead of dumping whole files.
   - `mcp__code-indexing-mcp__find_symbol` — trace functions/classes the feature will touch to all definitions and call sites, to map impact.
   - `mcp__code-indexing-mcp__get_chunk` — pull the full body of a chunk when you need exact code.
4. Only use Grep when the task genuinely needs a literal match semantic search cannot express (exact string literals, config keys, TODO/FIXME markers) or when the MCP tools are unavailable — and say so when you do.

End the phase with a short **current-state summary** for the user: the modules/files involved, existing patterns and conventions the feature must follow, extension points, and how that area is tested.

## Phase 2 — Prepare the implementation plan

Ground every step in what Phase 1 actually found — real file paths, real symbols, real test conventions. The plan must state:

- **Changes per file** — which files are created or edited, and what changes in each (new symbols, modified functions, wired-in call sites).
- **Design decisions** — any place you deviate from an existing convention, with the reason.
- **Tests** — which test files, which cases, following the project's existing test patterns.
- **Verification** — the exact commands to run (tests, lint, typecheck) and any manual smoke test.

Keep the plan minimal: the smallest change that delivers the request, no speculative extras.

## Phase 3 — Wait for explicit approval

Present the current-state summary and the plan, then STOP and wait for the user's explicit approval (via AskUserQuestion, or ExitPlanMode if in plan mode).

**No file edits, no implementation, no "starting with the obvious part" before approval.** If the user asks for changes, revise the plan and wait again.

## Phase 4 — Implement and verify

Only after approval:

1. Implement the plan as approved, matching the codebase conventions found in Phase 1.
2. Run the verification commands from the plan and look at the results.
3. Report what was changed (`path:line` references) and the verification outcome. If verification fails, fix it before claiming completion — do not declare done with red tests.

## Red flags — STOP and return to the right phase

- Reading files with Read/Grep as the main exploration while the MCP tools are available
- Writing a plan from memory or convention instead of Phase 1 findings
- Editing any file before the user has approved the plan
- Treating silence, "sounds good so far", or a clarifying question as approval
- Claiming completion without running the verification commands
````

- [ ] **Step 6: Create `src/incode_mcp/skills/indexed-review/SKILL.md`**

This is the renamed, tool-prefix-normalized port of `~/.agents/skills/incode-review/SKILL.md` (`name` changed to `indexed-review`, every `mcp__incode__` replaced with `mcp__code-indexing-mcp__`). Full content:

````markdown
---
name: indexed-review
description: Review the codebase from a user-chosen angle (security, logic, UI, UX, performance, architecture, etc.) using code-indexing-mcp semantic code analysis instead of grep-style searching
type: prompt
whenToUse: When the user asks for a code review, audit, or targeted analysis of the codebase and the code-indexing-mcp tools are available
arguments:
  - angle
  - scope
---

You are running a structured, angle-focused code review. Follow this workflow exactly.

## 1. Require a review angle

The review angle is `$angle` (valid examples: security, logic, UI, UX, performance, architecture, API design, error handling, testing).

If no angle was provided (or it is ambiguous), you MUST stop and ask the user with the AskUserQuestion tool before doing any analysis. Offer these options:

- **security** — vulnerabilities, injection, secrets, authn/authz, unsafe input handling
- **logic** — correctness, edge cases, error handling, race conditions, off-by-one
- **UI / UX** — interface structure, consistency, accessibility, user-facing flows
- **performance** — hot paths, allocations, I/O patterns, caching, complexity

Do not guess the angle. A review without a stated angle is not started.

Scope is `$scope` (a directory, module, feature, or symbol set). If scope is empty, ask the user or infer it from their request (e.g. recent changes); state the scope you settled on before reviewing.

## 2. Prepare the index

1. Call `mcp__code-indexing-mcp__project_status` for the target project. If the project is unknown, call `mcp__code-indexing-mcp__list_projects` and, if needed, `mcp__code-indexing-mcp__init_project`.
2. If the index is missing or stale relative to the files under review, call `mcp__code-indexing-mcp__index_project` first. Tell the user if indexing will take a moment.

## 3. Analyze with the index, not grep

This is the core rule of this skill: **use the code-indexing-mcp tools as the primary navigation and analysis mechanism. Do not fall back to Grep/Glob for code exploration.**

- `mcp__code-indexing-mcp__search_code` — semantic natural-language queries to find relevant code for the angle (see playbooks below). Issue several targeted queries, not one broad one.
- `mcp__code-indexing-mcp__file_outline` — get a file's structure before reading it; decide which parts matter instead of dumping whole files.
- `mcp__code-indexing-mcp__find_symbol` — trace a function/class to all its definitions and call sites when evaluating impact or correctness.
- `mcp__code-indexing-mcp__get_chunk` — pull the full body of a chunk returned by search when you need exact code.

Only use Grep when the task genuinely needs a literal match that semantic search cannot express (exact string literals, secret-looking patterns, TODO/FIXME markers) or when the code-indexing-mcp tools are unavailable — and say so when you do.

## 4. Angle playbooks

Tune your `search_code` queries and checklist to the chosen angle:

- **security**: query for "user input validation", "SQL or query construction", "subprocess / shell command execution", "authentication and permission checks", "file path handling", "deserialization", "token or password handling". Check: injection sinks, missing authz, hardcoded secrets, unsafe defaults, data exposure in logs/errors.
- **logic**: query for the core decision-making code in scope, "boundary handling", "retry and timeout logic", "concurrency and locking". Check: edge cases (empty, None, overflow), error paths that swallow exceptions, incorrect conditionals, state mutation bugs.
- **UI / UX**: query for "rendering / component structure", "user-facing text and messages", "input and interaction handling", "loading and error states". Check: consistency, accessibility, feedback on failure, dead ends in flows.
- **performance**: query for "loops over collections", "database or network calls", "caching", "serialization". Check: N+1 patterns, repeated work in hot paths, unbounded growth, blocking I/O on critical paths.
- **architecture**: query for "module boundaries", "dependency direction", "shared state". Check: layering violations, god objects, leaky abstractions, coupling.
- **API design**: query for "public endpoints / exported functions", "request and response schemas". Check: naming consistency, versioning, error contract, backward compatibility.

For any other angle, derive 3–5 semantic queries from the angle's intent and proceed the same way.

## 5. Report

Produce a structured review report:

- **Angle and scope** actually reviewed, and index freshness used.
- **Findings**, ordered by severity (critical / major / minor). Each finding: `path:line` reference, the evidence (short code excerpt via `get_chunk`), why it matters for this angle, and a concrete fix suggestion.
- **What looks good** — brief, only genuine positives.
- **Blind spots** — areas the semantic search could not cover confidently, so the user knows what was not verified.

Keep findings honest: report only what you actually inspected through the tools, never speculate about code you did not read.
````

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_skills.py -v`
Expected: PASS — 1 + 4 + 4 tests

- [ ] **Step 8: Commit**

```bash
git add src/incode_mcp/skills tests/test_skills.py
git commit -m "feat: bundle codebase-exploration, feature-dev, impact-analysis, indexed-review skills"
```

---

### Task 3: Installer symlinks skills into harness skill directories

**Files:**
- Modify: `install.py` (new functions after `configure_selected_harnesses` at line 719-744; wiring in `main()` at line 820-828)
- Test: `tests/test_installer.py` (append; uses the existing `installer` import alias and `tmp_path` style)

**Interfaces:**
- Consumes: `src/incode_mcp/skills/<name>/SKILL.md` folders from Task 2 (the test fixtures mimic that layout).
- Produces:
  - `skill_directory(slug: str, *, home: Path | None = None, environment: Mapping[str, str] | None = None) -> Path | None`
  - `install_skills(slugs: list[str], install_directory: Path, *, home: Path | None = None, environment: Mapping[str, str] | None = None) -> list[tuple[str, str]]` — returns `(slug, human-readable status)` lines; never raises for per-harness problems.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_installer.py`:

```python
def _skills_source(tmp_path: Path, names: tuple[str, ...] = ("alpha", "beta")) -> Path:
    root = tmp_path / "repo" / "src" / "incode_mcp" / "skills"
    for name in names:
        skill = root / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: test\n---\n", encoding="utf-8"
        )
    return tmp_path / "repo"


def test_skill_directories_cover_supported_harnesses(tmp_path: Path) -> None:
    assert (
        installer.skill_directory("claude-code", home=tmp_path, environment={})
        == tmp_path / ".claude" / "skills"
    )
    assert (
        installer.skill_directory("codex", home=tmp_path, environment={})
        == tmp_path / ".agents" / "skills"
    )
    assert (
        installer.skill_directory("kimi-code", home=tmp_path, environment={})
        == tmp_path / ".agents" / "skills"
    )
    assert (
        installer.skill_directory(
            "opencode", home=tmp_path, environment={"XDG_CONFIG_HOME": str(tmp_path / "xdg")}
        )
        == tmp_path / "xdg" / "opencode" / "skills"
    )
    assert installer.skill_directory("claude-desktop", home=tmp_path, environment={}) is None
    assert installer.skill_directory("kilocode", home=tmp_path, environment={}) is None


def test_skill_directory_honors_claude_config_dir(tmp_path: Path) -> None:
    environment = {"CLAUDE_CONFIG_DIR": str(tmp_path / "custom-claude")}
    assert (
        installer.skill_directory("claude-code", home=tmp_path, environment=environment)
        == tmp_path / "custom-claude" / "skills"
    )


def test_install_skills_links_bundled_skills(tmp_path: Path) -> None:
    repo = _skills_source(tmp_path)

    results = installer.install_skills(["claude-code"], repo, home=tmp_path, environment={})

    skills_dir = tmp_path / ".claude" / "skills"
    for name in ("alpha", "beta"):
        link = skills_dir / name
        assert link.is_symlink()
        assert link.resolve() == (repo / "src" / "incode_mcp" / "skills" / name).resolve()
    assert len(results) == 1
    slug, message = results[0]
    assert slug == "claude-code"
    assert "2 linked" in message


def test_install_skills_is_idempotent(tmp_path: Path) -> None:
    repo = _skills_source(tmp_path)
    installer.install_skills(["codex"], repo, home=tmp_path, environment={})

    results = installer.install_skills(["codex"], repo, home=tmp_path, environment={})

    slug, message = results[0]
    assert "0 linked" in message
    assert "already installed" in message


def test_install_skills_backs_up_clashing_directory(tmp_path: Path) -> None:
    repo = _skills_source(tmp_path)
    clash = tmp_path / ".agents" / "skills" / "alpha"
    clash.mkdir(parents=True)
    (clash / "SKILL.md").write_text("old", encoding="utf-8")

    installer.install_skills(["kimi-code"], repo, home=tmp_path, environment={})

    assert (tmp_path / ".agents" / "skills" / "alpha").is_symlink()
    backup = tmp_path / ".agents" / "skills" / "alpha.bak"
    assert backup.is_dir() and not backup.is_symlink()
    assert (backup / "SKILL.md").read_text(encoding="utf-8") == "old"


def test_install_skills_skips_unsupported_harness(tmp_path: Path) -> None:
    repo = _skills_source(tmp_path)

    results = installer.install_skills(["claude-desktop"], repo, home=tmp_path, environment={})

    slug, message = results[0]
    assert slug == "claude-desktop"
    assert "skipped" in message
    assert not (tmp_path / ".claude").exists()


def test_install_skills_reports_missing_source(tmp_path: Path) -> None:
    results = installer.install_skills(
        ["codex"], tmp_path / "empty-repo", home=tmp_path, environment={}
    )

    slug, message = results[0]
    assert "skipped" in message
    assert not (tmp_path / ".agents").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_installer.py -v -k "skill"`
Expected: FAIL — `AttributeError: module 'install' has no attribute 'skill_directory'`

- [ ] **Step 3: Implement `skill_directory`, `_link_skill`, and `install_skills`**

In `install.py`, after `configure_selected_harnesses` (ends line 744), add:

```python
def skill_directory(
    slug: str,
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    """Return the user-level skill directory for a harness, or None if unsupported."""

    home = home or Path.home()
    environment = os.environ if environment is None else environment
    if slug == "claude-code":
        return (
            _configured_directory(environment, "CLAUDE_CONFIG_DIR", home / ".claude") / "skills"
        )
    if slug in {"codex", "kimi-code"}:
        return home / ".agents" / "skills"
    if slug == "opencode":
        xdg_config = _configured_directory(environment, "XDG_CONFIG_HOME", home / ".config")
        return xdg_config / "opencode" / "skills"
    return None


def _link_skill(source: Path, target: Path) -> bool:
    """Symlink one bundled skill folder, backing up any clashing entry.

    Returns True when a new link was created, False when it already existed.
    """

    if target.is_symlink():
        if Path(os.readlink(target)) == source:
            return False
        target.unlink()
    elif target.exists():
        backup = target.with_name(f"{target.name}.bak")
        if backup.is_symlink() or backup.is_file():
            backup.unlink()
        elif backup.exists():
            shutil.rmtree(backup)
        target.rename(backup)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)
    return True


def install_skills(
    slugs: list[str],
    install_directory: Path,
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Symlink bundled skills into each selected harness's skill directory.

    Returns one (slug, status message) pair per harness; per-harness problems
    become "skipped" messages instead of raising.
    """

    skills_source = install_directory / "src" / "incode_mcp" / "skills"
    if not skills_source.is_dir():
        return [(slug, f"skipped: bundled skills not found at {skills_source}") for slug in slugs]
    skills = sorted(
        entry for entry in skills_source.iterdir() if (entry / "SKILL.md").is_file()
    )
    results: list[tuple[str, str]] = []
    for slug in slugs:
        directory = skill_directory(slug, home=home, environment=environment)
        if directory is None:
            results.append((slug, "skipped: harness has no skill-directory support"))
            continue
        try:
            created = [_link_skill(skill, directory / skill.name) for skill in skills]
        except OSError as exc:
            results.append((slug, f"skipped: {exc}"))
            continue
        linked = sum(created)
        results.append(
            (
                slug,
                f"{linked} linked, {len(created) - linked} already installed in {directory}",
            )
        )
    return results
```

- [ ] **Step 4: Wire into `main()`**

In `install.py` `main()`, change the block after `configure_selected_harnesses` (currently lines 820-828) from:

```python
        successes, failures = configure_selected_harnesses(selected, command)
        for slug, path in successes:
            output_fn(f"Configured {_harness_label(slug)}: {path}")
        for slug, message in failures:
            error_fn(f"Failed to configure {_harness_label(slug)}: {message}")
        if failures:
            return 1
        output_fn("Installation complete. Restart configured clients to load the MCP server.")
        return 0
```

to:

```python
        successes, failures = configure_selected_harnesses(selected, command)
        for slug, path in successes:
            output_fn(f"Configured {_harness_label(slug)}: {path}")
        for slug, message in failures:
            error_fn(f"Failed to configure {_harness_label(slug)}: {message}")
        for slug, message in install_skills(selected, install_directory):
            output_fn(f"Skills for {_harness_label(slug)}: {message}")
        if failures:
            return 1
        output_fn("Installation complete. Restart configured clients to load the MCP server.")
        return 0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_installer.py -v`
Expected: PASS (whole file, including the 7 new tests)

Note: `test_main_runs_noninteractive_install_and_reports_harness_failures` and `test_main_prompts_for_harnesses_when_option_is_omitted` monkeypatch `configure_selected_harnesses` but not `install_skills`; the real `install_skills` will run against their fake install dirs, find no `src/incode_mcp/skills`, and emit one "skipped" output line per selected slug. If either test asserts exact output, add `"skipped: bundled skills not found"` lines to its expected output or monkeypatch `installer.install_skills` to `lambda slugs, directory: []` — match the surrounding test style.

- [ ] **Step 6: Commit**

```bash
git add install.py tests/test_installer.py
git commit -m "feat: symlink bundled skills into harness skill directories during install"
```

---

### Task 4: Packaging verification, docs, and full-suite gate

**Files:**
- Modify: `README.md` (installation section — add a short "Bundled skills" note)
- Create: `docs/superpowers/plans/2026-07-24-bundled-skills.md` (copy of this plan, for the repo record)

**Interfaces:**
- Consumes: everything from Tasks 1-3.

- [ ] **Step 1: Verify the skills ship in the built wheel**

Run:

```bash
rm -rf dist && uv build --wheel && unzip -l dist/*.whl | grep skills
```

Expected: the four `incode_mcp/skills/<name>/SKILL.md` entries are listed. If they are missing, add to `pyproject.toml` under `[tool.hatch.build.targets.wheel]`:

```toml
artifacts = ["src/incode_mcp/skills/**"]
```

(or the equivalent `force-include`) and re-run until the files appear. Clean up: `rm -rf dist`.

- [ ] **Step 2: Add the README note**

In `README.md`, in the installation section (after the installer usage), add:

```markdown
### Bundled skills

The installer also symlinks four agent skills into skill-capable harnesses
(Claude Code, Kimi Code, Codex, OpenCode), pointing into the cloned repo so
they update on every re-install: `codebase-exploration` (index-first
navigation), `feature-dev` (index-grounded feature workflow), `indexed-review`
(angle-based code review), and `impact-analysis` (blast-radius mapping before a
change). Harnesses without skill support are skipped.
```

- [ ] **Step 3: Save the plan to the repo**

```bash
mkdir -p docs/superpowers/plans
cp <this plan file> docs/superpowers/plans/2026-07-24-bundled-skills.md
```

- [ ] **Step 4: Run the full verification suite**

Run:

```bash
uv run pytest && uv run ruff check . && uv run mypy
```

Expected: all tests pass, no lint or type errors. (`tests/test_model_integration.py` carries the `model` marker — if it is deselected by default in this environment, that is pre-existing behavior, not a failure.)

- [ ] **Step 5: Commit**

```bash
git add README.md docs/superpowers/plans/2026-07-24-bundled-skills.md pyproject.toml
git commit -m "docs: document bundled skills and record the implementation plan"
```

(Omit `pyproject.toml` from the add if Step 1 needed no change.)

---

## Self-Review Notes

- **Spec coverage:** instructions (Task 1), 4 bundled skills incl. rename + prefix normalization (Task 2), symlink installer with skip/backup/idempotency (Task 3), wheel packaging check + tests + docs (Tasks 2-4). Out-of-scope items (MCP prompts, uninstall, per-project skills) are untouched.
- **Type consistency:** `skill_directory` / `install_skills` signatures match between implementation and all 7 tests; `_skills_source` fixture mirrors the real `src/incode_mcp/skills/<name>/SKILL.md` layout.
- **Known soft spot:** Task 3 Step 5's note about the two existing `main()` tests — the implementer must look at their actual assertions and adapt output expectations as described.
