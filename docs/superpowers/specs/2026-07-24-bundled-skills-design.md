# Bundled Skills for code-indexing-mcp — Design

Date: 2026-07-24
Status: Approved (pending user spec review)

## Goal

Ship a set of agent skills with the MCP server so harnesses pick them up and use
the index tools efficiently — most importantly, an index-first exploration
workflow that narrows down via the index before any grep/read fallback.

## Decisions (from brainstorming)

- **Pickup style:** both — always-on baseline behavior guidance (server
  `instructions`) plus named, invocable/auto-applicable skill files.
- **Delivery:** installer symlinks skill folders into harness skill
  directories. No MCP `prompts`/`resources` for now.
- **Link style:** symlink into the cloned repo at
  `~/.local/share/code-indexing-mcp`, so `install.py` updates / fast-forward
  pulls refresh skills automatically.
- **Skill set (4):** `codebase-exploration` (new), `feature-dev` (ported),
  `indexed-review` (ported, renamed from `code-indexing-mcp-review`), `impact-analysis`
  (new).
- **Tool-name normalization:** all bundled skills reference
  `mcp__code-indexing-mcp__*` (the server name install.py registers). The
  existing loose copies use inconsistent prefixes (`mcp__code-indexing-mcp__*`).

## Components

### 1. Baseline behavior via server instructions

Expand the `instructions=` string in `create_server()`
(`src/code_indexing_mcp/server.py`, currently the one-liner at ~line 189) into a short
guidance block injected into every MCP client on connect:

- Prefer `search_code` / `find_symbol` / `file_outline` / `get_chunk` over
  grep-style file reading when exploring indexed code.
- Check `list_projects` / `project_status` first; `index_project` if missing
  or stale.

Keep it to a few lines — instructions are injected into every context.

### 2. Bundled skills in the Python package

New directory `src/code_indexing_mcp/skills/`, shipped with the package (same
mechanism as `queries/*.scm` — files inside the package directory ride along
with hatchling's `packages = ["src/code_indexing_mcp"]`; verify the built wheel
contains them). One folder per skill, each with a `SKILL.md`:

- `codebase-exploration/SKILL.md` — index-first narrowing:
  `list_projects`/`project_status` → targeted `search_code` queries →
  `file_outline` before reading any file → `get_chunk` for exact code.
  Grep/Glob only as an explicitly-stated fallback (unindexed files, literal
  matches semantic search can't express).
- `feature-dev/SKILL.md` — ported from `~/.agents/skills/feature-dev/`,
  tool names normalized to `mcp__code-indexing-mcp__*`.
- `indexed-review/SKILL.md` — ported from `~/.agents/skills/code-indexing-mcp-review/`,
  renamed (`name: indexed-review`), tool names normalized.
- `impact-analysis/SKILL.md` — new; plays to `find_symbol`'s strength. Before
  a rename/refactor/signature change: map all definitions and call sites via
  `find_symbol`, assess blast radius with `search_code` for indirect usages
  (strings, duck-typed call sites), produce a change checklist.

The package copies are the canonical source of truth going forward; the loose
copies in `~/.agents/skills/` are replaced by the symlinks.

### 3. Installer step: symlink skills into harness skill dirs

`install.py` gains a skill-install step, run after
`configure_selected_harnesses()`:

- For each selected harness that supports skills, symlink each bundled skill
  folder from the cloned repo into the harness's user-level skill directory:
  - Claude Code: `~/.claude/skills/`
  - Kimi Code: `~/.agents/skills/`
  - Codex, Claude Desktop, OpenCode, KiloCode: verify skill support during
    implementation; skip silently (with a note in the summary) if unsupported.
- Existing clashing entry at the target path: if it's already our symlink,
  leave it; otherwise back up to `<name>.bak` (matching installer's existing
  `.bak` convention) and link.
- Report installed/skipped skills in the install summary.

### 4. Uninstall/cleanup consideration

On re-install after a skill is removed from the package, stale symlinks point
into the repo folder and would dangle only if the repo is deleted — accepted
trade-off of symlinking (chosen by user). No uninstall command is added in
this iteration.

## Error handling

- Harness without skill support → skip with note, never fail the install.
- Unwritable skill dir → warn, continue with remaining harnesses.
- Repo missing `src/code_indexing_mcp/skills/` (old checkout) → skip with note.

## Testing

- `tests/test_installer.py`: unit tests for the skill-link step — creates
  symlinks for skill-capable harnesses, skips unsupported ones, backs up
  clashing non-symlink entries, idempotent on re-run.
- New test (e.g. `tests/test_skills.py`): every folder under
  `src/code_indexing_mcp/skills/` contains a `SKILL.md` with valid frontmatter
  (`name`, `description`) and references only `mcp__code-indexing-mcp__*`
  tool prefixes.
- Existing test suite must stay green (`uv run pytest`).

## Out of scope

- MCP `prompts` / `resources` exposure of the skills.
- An uninstall command for skills.
- Per-project (repo-local) skill installation.
