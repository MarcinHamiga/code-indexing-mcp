# Code Indexing MCP Installer Design

## Goals

Provide one installer that can be downloaded directly, can clone a fresh Code Indexing MCP
checkout, can safely fast-forward an existing installer-managed checkout, and can configure the
MCP server for any selected supported harness. The initial harness list is:

1. Codex (CLI + Desktop)
2. Claude Code
3. Kimi Code
4. Claude Desktop
5. OpenCode
6. KiloCode

Codex CLI and Codex Desktop are one choice because both read the same `config.toml`.

All newly initialized source projects use `.ci-mcp/project.toml`. Existing `.incode` markers
remain readable so an upgrade does not make an already indexed checkout disappear. Both marker
directories are always excluded from source scans.

## Installer Architecture

The primary installer is a standalone, standard-library-only `install.py`. This keeps
configuration behavior consistent across macOS, Linux, and Windows without requiring the
project's dependencies before the repository has been downloaded. A small `install.sh` locates
the adjacent Python installer when run from a checkout or downloads it to a temporary directory
when invoked through `curl`.

The default checkout is `~/.local/share/code-indexing-mcp`. The location, repository URL, and
harness selection can be overridden with command-line options. On a first run, the installer
clones the configured repository. On later runs, it verifies the target is the expected Git
checkout, refuses to overwrite local tracked changes, and performs a fast-forward-only pull.
It then runs `uv sync --locked` and configures clients to launch the absolute executable inside
the checkout's virtual environment. Desktop clients therefore do not depend on the user's shell
`PATH`.

The interactive menu accepts comma-separated numbers or harness slugs, plus `all`. Supplying
`--harnesses` skips the prompt for automation. An empty interactive selection installs or
updates the checkout but does not change any client configuration.

## Configuration Updates

The installer changes only the `code-indexing-mcp` entry in each selected client:

- Codex: `~/.codex/config.toml` (or `$CODEX_HOME/config.toml`)
- Claude Code: `~/.claude.json`
- Kimi Code: `$KIMI_CODE_HOME/mcp.json` or `~/.kimi-code/mcp.json`
- Claude Desktop: the platform's standard `claude_desktop_config.json`
- OpenCode: its global `opencode.json`/`opencode.jsonc`
- KiloCode: its global `kilo.json`/`kilo.jsonc`

Codex uses a targeted TOML table replacement. JSON and JSONC clients use a small structural
editor that understands strings, nesting, comments, and trailing commas. This preserves
unrelated keys and comments rather than deserializing and rewriting an entire user-owned file.
Writes are atomic and an existing file is copied to a `.bak` file immediately before a changed
version is installed. Invalid or unsupported configuration is reported without modifying the
file; failures in one selected harness do not prevent other selections from being attempted.

## Error Handling and Verification

Missing Git, Python, or `uv`; a non-repository install target; a mismatched origin; a dirty
checkout; or a non-fast-forward update produces an actionable error and a nonzero exit status.
Claude Desktop reports an unsupported-platform error where no standard local desktop
configuration exists.

Unit tests cover marker creation and legacy resolution, TOML and JSONC preservation, menu
parsing, harness-specific schemas and paths, fresh clone behavior, and update behavior using a
temporary local Git remote. The final verification runs the complete pytest suite, Ruff, MyPy,
shell syntax validation, and an installer help smoke test.
