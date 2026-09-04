# Terminal TUI First Implementation Plan

## Goal

Add a keyboard-first terminal interface for exploring indexed code without typing the
`code-indexing-mcp` command or manually constructing MCP tool calls.

The primary entry point will be:

```console
cidx
```

The existing executable will retain all current behavior and gain an explicit equivalent:

```console
code-indexing-mcp tui
```

## Current State

- Textual is already the supported TUI framework and is installed by the normal bootstrap through
  the `tui` extra (`pyproject.toml:99-105`, `install.py:23`).
- The current CLI exposes administration but not interactive code queries
  (`src/code_indexing_mcp/cli.py:63-262`, `src/code_indexing_mcp/cli.py:476-519`).
- `ApplicationLike` and `BrokerApplication` already expose search, symbols, chunks, outlines,
  references, impact analysis, status, and indexing
  (`src/code_indexing_mcp/application.py:188-355`,
  `src/code_indexing_mcp/daemon.py:834-1048`).
- Textual background-worker and `run_test()` patterns already exist
  (`src/code_indexing_mcp/installer/tui/panels.py:612-625`,
  `tests/test_installer_tui.py:76-87`).
- The installer currently creates and removes only one launcher. Making `cidx` genuinely available
  on `PATH` therefore requires install, repair, update, and uninstall integration
  (`src/code_indexing_mcp/installer/shell_path.py:87-91`,
  `src/code_indexing_mcp/installer/shell_path.py:154-180`,
  `src/code_indexing_mcp/installer/shell_path.py:277-296`).
- Direct application queries do not perform all MCP lazy-freshness coordination. The runtime TUI
  needs an explicit readiness step before querying (`src/code_indexing_mcp/server.py:573-628`,
  `src/code_indexing_mcp/application.py:1367-1400`).

## First Release Scope

The first release will provide:

- `cidx` as the short interactive command.
- `code-indexing-mcp tui` as the long-form equivalent.
- Current-directory discovery and a selector for registered projects.
- Natural-language search and symbol lookup.
- A search-result list with project, path, symbol, kind, score, and line range.
- Full chunk preview for the selected result.
- Outline, references, and impact-radius actions for a selected declaration.
- Project status plus an explicit refresh/index action.
- Lazy automatic indexing when configured, with no automatic indexing in manual mode.
- Visible loading, indexing progress, empty states, and recoverable errors.
- Keyboard-first operation with mouse support.

The first release will not include:

- Project removal.
- Storage mutation.
- Refactor patch generation or application.
- Installer configuration.
- Cross-project merged search.
- Generic forms for every MCP tool.

## User Interface

The initial layout will use a compact header and status strip, query controls, a result list, and a
detail pane:

```text
+ Project / branch / index state --------------------------------------+
| Search: authentication middleware                         [Search v] |
+ Results -----------------------+ Code preview ------------------------+
| 1  auth/service.py:42          | class AuthenticationService:         |
| 2  api/middleware.py:18        |     ...                              |
| 3  tests/test_auth.py:71       |                                      |
+--------------------------------+--------------------------------------+
| Enter open  r references  o outline  i impact  F5 index  q quit     |
+-----------------------------------------------------------------------+
```

The minimum supported terminal size will be 80 columns by 24 rows. The standard layout will show
results and preview side by side where space permits. At the minimum size, both panes must remain
usable without controls painting over the footer.

Initial key bindings:

| Key | Action |
| --- | --- |
| `/` | Focus the query input |
| `Enter` | Submit the query or open the focused result |
| `o` | Show the selected result's file outline |
| `r` | Show references to the selected declaration |
| `i` | Show the selected declaration's impact radius |
| `F5` | Index or refresh the selected project |
| `Escape` | Return from details to the result list |
| `q` | Quit when an input does not own the key |

## Design Decisions

### Use the application boundary, not a child MCP client

The TUI will call `ApplicationLike` rather than starting a stdio MCP client and parsing protocol
responses. `BrokerApplication` already provides the same typed models over the shared daemon, while
`Application` provides the platform fallback. This avoids a child process, MCP root negotiation,
and duplicated serialization inside one local application.

### Prefer the shared daemon

The service factory will use `ensure_daemon()` when broker mode allows it. This reuses the loaded
embedding model and serializes access through the existing local daemon. On platforms without local
socket support, broker `auto` mode will fall back to a direct `Application`; explicit broker `on`
mode will preserve the existing error behavior.

### Keep readiness behavior explicit

The MCP server's coordinator currently performs discovery and stale-index refresh before query tool
calls. A direct call to `Application.search_code()` only guarantees a compatible generation, not a
fresh source scan. The TUI service will therefore perform project discovery and status checks before
querying.

In `lazy` and `eager` modes, a selected project outside `ready` or `partial` will be indexed with the
`lazy-query` trigger before the query proceeds. In `manual` mode, the TUI will show that indexing is
required and wait for the user to press `F5`.

### Keep Textual off non-TUI paths

Both entry points will import the runtime TUI lazily. Starting the MCP server, daemon, benchmarks,
or existing administrative commands must not import Textual or initialize terminal state.

### Manage `cidx` as an owned secondary launcher

Adding only a project script would make `cidx` available inside the prepared virtual environment,
but not through the launcher directory placed on the user's `PATH`. The installer will therefore
manage both the existing `code-indexing-mcp` launcher and the new `cidx` launcher.

The existing ownership rule remains mandatory: an unrelated executable named `cidx` must never be
overwritten or removed. Failure to install the convenience alias will be a visible warning, not a
reason to undo successful MCP client configuration.

## Implementation Tasks

### 1. Add the runtime service boundary

Create `src/code_indexing_mcp/tui/service.py`.

Add a `TuiService` that accepts an `ApplicationLike`, the current directory, roots, and index mode.
It will own:

- Current-directory project discovery.
- Registered-project listing and selected-project resolution.
- Readiness checks before query operations.
- Explicit indexing.
- Semantic search and symbol lookup.
- Chunk and outline loading.
- Reference and impact queries.
- Conversion of a selected `SearchHit` into a `DeclarationSelector`.

Add a factory that follows the existing serve policy:

- Use `ensure_daemon()` when broker mode permits.
- Fall back to `Application` when sockets are unavailable in auto mode.
- Surface the existing error when broker mode is explicitly required.

Use `index_progress()` for polling so the same progress path works with direct and broker
applications. Do not add a callback-only branch for direct applications.

Keep the module independent of Textual so service behavior can be unit-tested with a fake
`ApplicationLike`.

### 2. Build the Textual application

Create `src/code_indexing_mcp/tui/app.py`.

Add `CodeIndexingApp` with:

- A header showing the selected project, checkout selector, and index state.
- A project selector populated from registered projects.
- A query input.
- A semantic-search or symbol-search mode selector.
- A result list based on `SearchHit`.
- A detail pane for chunks, outlines, references, and impact layers.
- A status line for progress, empty states, and errors.
- A footer containing key bindings.

Use `@work(thread=True, exclusive=True)` for discovery, querying, indexing, and detail loading,
following the installer worker pattern. Route UI mutations back through `call_from_thread`.

Use one result rendering path for `search_code` and `find_symbol`, because both return `SearchHit`
records (`src/code_indexing_mcp/models.py:823-852`).

Selecting a result and pressing Enter will call `get_chunk` and show complete indexed source text.
The `o`, `r`, and `i` actions will replace the detail pane rather than adding separate screens in
the first release.

Keep the selected hit stable while detail calls run. Assign each query a monotonically increasing
request id and discard a worker completion when a newer query has started.

Display `CodeIndexingError` details in the status area without terminating the application. An
error should not erase a previously usable result unless the selected project changed.

Put the Textual styling in `src/code_indexing_mcp/tui/app.tcss` so layout behavior can be tested
without embedding a large stylesheet in Python.

### 3. Add lazy entry points

Create `src/code_indexing_mcp/tui/__init__.py`.

Expose a small `main()` that imports Textual and `CodeIndexingApp` only after invocation, constructs
the service from environment settings, runs the application, and returns its exit code.

Edit `src/code_indexing_mcp/cli.py`.

Add a `tui` subcommand and dispatch it through a lazy import. Do not import the runtime TUI on the
`serve`, daemon, benchmark, or ordinary CLI paths.

Edit `pyproject.toml`.

Add the short project script while retaining the existing script unchanged:

```toml
[project.scripts]
code-indexing-mcp = "code_indexing_mcp.cli:main"
cidx = "code_indexing_mcp.tui:main"
```

### 4. Install the `cidx` launcher

Edit `src/code_indexing_mcp/installer/shell_path.py`.

Generalize launcher path, installation, ownership checks, and removal to accept a command name while
retaining `code-indexing-mcp` as the default. Add `cidx` as the managed secondary launcher,
targeting the generated `cidx` console script in the prepared environment.

Preserve the current safety rule: an unrelated existing executable named `cidx` must never be
overwritten or removed.

Edit `src/code_indexing_mcp/installer/orchestrator.py`.

Install both launchers during install, configure, repair, and update flows. Extend `InstallResult`
with a defaulted `tui_launcher` result so existing positional construction remains valid. Report a
secondary-launcher collision as a warning without undoing successful MCP client configuration.

Edit `src/code_indexing_mcp/installer/uninstall.py`.

Remove both owned launchers. Extend `UninstallResult` with a defaulted `tui_launcher_removed` field
and include it in plain output.

Edit `src/code_indexing_mcp/installer/tui/panels.py`.

Change command-access wording from one launcher to both commands, show both installation outcomes,
include `cidx` in completion instructions, and flag a missing `cidx` launcher as repairable.

### 5. Add service tests

Create `tests/test_tui_service.py`.

Use a fake `ApplicationLike`; do not load a real embedding model. Cover:

- Current-directory discovery.
- Registered-project listing.
- Selected-project scoping.
- Ready and partial queries without reindexing.
- Stale and pending lazy indexing.
- Manual-mode refusal to auto-index.
- Symbol match-mode forwarding.
- Chunk loading.
- Outline forwarding.
- Selector construction for references and impact analysis.
- Broker auto fallback.
- Surfaced `CodeIndexingError` details.

### 6. Add Textual interaction tests

Create `tests/test_tui.py`.

Use `CodeIndexingApp.run_test()` with an injected fake service. Cover:

- Initial project selection and status.
- Semantic search submission.
- Symbol search mode.
- Empty results.
- Keyboard result selection.
- Full chunk preview.
- Outline, reference, and impact bindings.
- Explicit `F5` indexing.
- Progress rendering.
- Recoverable errors.
- Stale worker-result suppression.
- Clean quit behavior.
- An 80x24 layout smoke test.

Avoid snapshot-heavy assertions. Assert widget state, visible semantic text, focus changes, and
service calls.

### 7. Extend command and installer regression tests

Edit `tests/test_cli.py`.

Verify that `code-indexing-mcp tui` lazily invokes the runtime entry point and propagates its exit
code without constructing `Application` or starting the MCP server.

Edit `tests/test_installer_shell_path.py`.

Verify POSIX and Windows paths for both launcher names, creation of `cidx`, replacement of an owned
stale alias, refusal to overwrite an unrelated `cidx`, and ownership-safe removal.

Edit `tests/test_installer_orchestrator.py`.

Verify successful creation of both launchers and warning behavior when only `cidx` cannot be
created.

Edit `tests/test_installer_uninstall.py`.

Verify uninstall removes both owned launchers and leaves a foreign `cidx` untouched.

Edit `tests/test_installer_tui.py`.

Verify command-access and completion panels advertise `cidx`, display its installation failure, and
recommend repair when it is missing.

### 8. Document the workflow

Edit `README.md`.

Add a `Terminal UI` section showing `cidx` and `code-indexing-mcp tui`, key bindings, automatic
versus manual indexing behavior, supported first-release actions, and the fact that the TUI does not
modify source files.

Update installation and uninstallation descriptions to state that both `code-indexing-mcp` and
`cidx` launchers are managed.

## Acceptance Criteria

- A normal installation places both `code-indexing-mcp` and `cidx` in the configured launcher
  directory.
- Running `cidx` from inside a qualifying repository discovers or selects it and opens the TUI.
- Running `code-indexing-mcp tui` opens the same application.
- Semantic and symbol searches never block the Textual event loop.
- Selecting a hit displays its complete indexed chunk.
- Outline, references, and impact actions operate on the selected hit.
- Lazy and eager modes refresh a stale selected project before querying.
- Manual mode asks for explicit indexing instead of refreshing automatically.
- Index progress remains visible whether work runs locally or through the daemon.
- Existing foreign executables named `cidx` are not overwritten or removed.
- Existing CLI and MCP behavior remains unchanged.
- Importing or starting the serve path does not import Textual.
- The interface remains usable at 80 columns by 24 rows.

## Verification

Run focused tests first:

```bash
uv run pytest tests/test_tui_service.py tests/test_tui.py tests/test_cli.py \
  tests/test_installer_shell_path.py tests/test_installer_orchestrator.py \
  tests/test_installer_uninstall.py tests/test_installer_tui.py
```

Run the repository gate:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest -n auto
```

Perform a manual smoke test from an indexed repository:

```bash
uv run cidx
uv run code-indexing-mcp tui
```

Confirm semantic search, symbol lookup, chunk preview, outline, references, impact analysis, explicit
indexing, terminal resize, and clean exit.
