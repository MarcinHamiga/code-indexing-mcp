# MCP Tool Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the nine MCP tools self-describing — real descriptions, host-usable annotations,
constrained schemas, and errors that carry their recovery detail — so a model can pick the right
tool and a host can auto-approve the read-only ones.

**Architecture:** Purely adapter-layer. Every change lands in `src/incode_mcp/server.py` plus a
small rendering helper on `IncodeError`. No application, storage, or indexing logic changes. The
existing `Application` / `BrokerApplication` split is untouched, so the CLI and daemon behave
exactly as before.

**Tech Stack:** `mcp` FastMCP (stdio transport), Pydantic v2 `Annotated`/`Field` for schema
constraints, `typing.Literal` for closed vocabularies.

## Global Constraints

See [the plan index](2026-07-27-review-followups-index.md#global-constraints). Additionally, for
this plan:

- **Tool descriptions must not instruct the model how to behave.** Anthropic Directory review
  treats "always call X first", "you must", and product promotion inside a tool description as
  prompt injection. Describe what the tool does, what it returns, and what it does *not* do.
  Cross-references phrased as capability statements ("find_symbol resolves a known declaration
  name") are fine; imperatives are not.
- **Tool names must stay byte-identical.** `init_project`, `index_project`, `project_status`,
  `list_projects`, `remove_project`, `search_code`, `find_symbol`, `file_outline`, `get_chunk` are
  the documented surface in `README.md:100-101`. Renaming any of them breaks existing clients.
- **Read and write tools stay in separate tools.** They already are; do not merge any.

---

## Problem

Verified by dumping `list_tools()` against a live server on `main`:

```
init_project    description: ''   annotations: None   title: None
index_project   description: ''   annotations: None   title: None
...  (all 9 identical)
TOTAL tool-list chars: 13,515  (~3,378 tokens)
```

None of the nine handlers in `server.py:333-441` has a docstring, and FastMCP derives
`Tool.description` from `__doc__`. The server therefore spends ~3.4k tokens of context every turn
on schemas while telling the model nothing about what any tool does. Four consequences:

1. **No descriptions.** `search_code`, `find_symbol`, and `file_outline` are mutually confusable
   and nothing disambiguates them. `index_project` already has the ideal description text sitting
   in a `#` comment at `server.py:353-356`, where it is discarded.
2. **No annotations.** The five read-only tools cannot be auto-approved, and `remove_project` —
   which `shutil.rmtree`s a partition at `storage.py:352-354` — is indistinguishable from a read.
3. **Loose schemas.** `match: str` accepts anything and is validated only at `search.py:81`, after
   project resolution, so a typo surfaces as `PROJECT_NOT_FOUND`. `limit` is an unbounded integer
   silently clamped to 1–50 at `search.py:42` and `search.py:83`. `kinds` is an open `list[str]`.
   No parameter has a description.
4. **Errors drop their details.** `IncodeError.details` carries `waited_seconds`,
   `wait_timeout_seconds`, `registered_root`, `effective_memory_bytes` — none reaches the client:

   ```
   ToolError: Error executing tool get_chunk: PROJECT_NOT_FOUND: Unknown chunk: deadbeef
      details preserved on wire? False
   ```

   `search.py:123` also raises `PROJECT_NOT_FOUND` for an unknown *chunk*, and no message names a
   recovery path.

## File Structure

| File | Responsibility after this plan |
|---|---|
| `src/incode_mcp/errors.py` | Adds `CHUNK_NOT_FOUND` and `IncodeError.for_client()`, which renders code, message, and details into one client-facing string. `__str__` is left alone so `IndexIssue` messages and daemon frames keep their current shape. |
| `src/incode_mcp/server.py` | Gains `_TOOL_INSTRUCTIONS`, the `_with_error_details` decorator, and full `title`/`description`/`annotations` plus `Annotated` parameter schemas on all nine tools. |
| `src/incode_mcp/search.py` | Raises `CHUNK_NOT_FOUND` with a recovery hint. Keeps its defensive `limit` clamp for direct CLI/daemon callers. |
| `src/incode_mcp/application.py` | One message gains a recovery hint. |
| `src/incode_mcp/projects.py` | One message gains a recovery hint. |
| `tests/test_server.py` | Gains a metadata contract test, a schema constraint test, and an error-detail test. |
| `README.md` | Tool list gains one line per tool describing it. |

Tasks 1–3 are ordered so each leaves the suite green. Task 1 is pure addition; Task 2 rewrites the
tool declarations; Task 3 threads error detail through.

---

### Task 1: Error rendering and a correct code for unknown chunks

**Files:**
- Modify: `src/incode_mcp/errors.py:7-32`
- Modify: `src/incode_mcp/search.py:120-124`
- Modify: `src/incode_mcp/application.py:307-309`
- Modify: `src/incode_mcp/projects.py:125`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ErrorCode.CHUNK_NOT_FOUND`, and `IncodeError.for_client() -> str`. Task 3 calls
  `for_client()` from the MCP boundary.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server.py`:

```python
def test_error_renders_code_message_and_details_for_clients() -> None:
    error = IncodeError(
        ErrorCode.INDEX_BUSY,
        "Another indexing job is already active",
        waited_seconds=3.5,
        wait_timeout_seconds=300,
    )

    rendered = error.for_client()

    assert rendered.startswith("INDEX_BUSY: Another indexing job is already active")
    assert "waited_seconds=3.5" in rendered
    assert "wait_timeout_seconds=300" in rendered
    # __str__ stays detail-free: IndexIssue messages and daemon frames embed it.
    assert str(error) == "INDEX_BUSY: Another indexing job is already active"


def test_error_without_details_renders_as_plain_string() -> None:
    error = IncodeError(ErrorCode.CHUNK_NOT_FOUND, "Unknown chunk: abc")

    assert error.for_client() == "CHUNK_NOT_FOUND: Unknown chunk: abc"
```

No new imports are needed: `tests/test_server.py:11` already has
`from incode_mcp.errors import ErrorCode, IncodeError`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_server.py -k for_client -v`

Expected: FAIL — `AttributeError: 'IncodeError' object has no attribute 'for_client'`, and
`AttributeError: CHUNK_NOT_FOUND` on the `ErrorCode` lookup.

- [ ] **Step 3: Write minimal implementation**

In `src/incode_mcp/errors.py`, add the member after `PROJECT_ID_CONFLICT`:

```python
    CHUNK_NOT_FOUND = "CHUNK_NOT_FOUND"
```

and add the method to `IncodeError`:

```python
    def for_client(self) -> str:
        """Render code, message, and details as one line for an MCP tool error.

        ``__str__`` deliberately omits details: it is embedded in ``IndexIssue``
        messages and in daemon frames that already carry ``details`` as a
        separate field, where appending them would duplicate the payload.
        """
        if not self.details:
            return str(self)
        rendered = "; ".join(f"{key}={value}" for key, value in self.details.items())
        return f"{self} [{rendered}]"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_server.py -k "for_client or plain_string" -v`

Expected: 2 passed.

- [ ] **Step 5: Give the three vaguest errors a recovery path**

In `src/incode_mcp/search.py:120-124`, replace `get_chunk`:

```python
    def get_chunk(self, chunk_id: str) -> CodeChunk:
        chunk = self.store.get_chunk(chunk_id)
        if chunk is None:
            raise IncodeError(
                ErrorCode.CHUNK_NOT_FOUND,
                f"Unknown chunk: {chunk_id}; chunk ids come from search_code or find_symbol "
                "results and change when the file is re-indexed",
                chunk_id=chunk_id,
            )
        return CodeChunk.model_validate(chunk.model_dump())
```

In `src/incode_mcp/application.py:307-309`:

```python
        if not project_ids:
            raise IncodeError(
                ErrorCode.PROJECT_NOT_FOUND,
                "No indexed projects are available; init_project registers one and "
                "index_project builds its index",
            )
```

In `src/incode_mcp/projects.py:125`:

```python
        raise IncodeError(
            ErrorCode.PROJECT_NOT_FOUND,
            "No active Incode project was detected; pass an explicit project id, name, or "
            "path, or run init_project for this directory",
            searched_roots=[str(root) for root in roots],
        )
```

`roots` is an `Iterable[Path]` consumed earlier in `resolve` by `_marked_projects`, so materialise
it once at the top of `resolve` before that call:

```python
        roots = list(roots)
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`

Expected: the pre-existing `190 passed, 3 skipped` plus the 2 new tests, with no failures. If a test
asserts `PROJECT_NOT_FOUND` for a missing chunk, update it to `CHUNK_NOT_FOUND` —
`grep -rn "PROJECT_NOT_FOUND\|Unknown chunk" tests` to find them.

- [ ] **Step 7: Lint, type-check, commit**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy src tests
git add src/incode_mcp/errors.py src/incode_mcp/search.py src/incode_mcp/application.py \
        src/incode_mcp/projects.py tests/test_server.py
git commit -m "feat: render error details for clients and add CHUNK_NOT_FOUND"
```

---

### Task 2: Describe, title, and annotate all nine tools

**Files:**
- Modify: `src/incode_mcp/server.py:272-442`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent; Task 1 only makes Task 3 possible).
- Produces: a `list_tools()` result where every tool has a non-empty `description`, a `title`, and
  an `annotations` object. Task 3 wraps the same handlers.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server.py`. That file uses **`pytest-asyncio`**, so async tests carry
`@pytest.mark.asyncio` — there is no `anyio` marker and no `application` fixture. Each test builds
its own `Application` inline with the `TinyEmbedder` double defined at line 14, exactly as
`test_server_registers_the_focused_tool_suite` does at line 80. Add a small local helper next to it
so the three new tests do not repeat that construction:

```python
READ_ONLY_TOOLS = frozenset(
    {"project_status", "list_projects", "search_code", "find_symbol", "file_outline", "get_chunk"}
)
WRITE_TOOLS = frozenset({"init_project", "index_project", "remove_project"})


def _tiny_application(tmp_path: Path) -> Application:
    return Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )


@pytest.mark.asyncio
async def test_every_tool_declares_description_title_and_annotations(tmp_path: Path) -> None:
    tools = await create_server(_tiny_application(tmp_path), auto_index=False).list_tools()

    assert {tool.name for tool in tools} == READ_ONLY_TOOLS | WRITE_TOOLS
    for tool in tools:
        assert tool.description, f"{tool.name} has no description"
        assert len(tool.description) > 60, f"{tool.name} description is a stub"
        assert tool.title, f"{tool.name} has no title"
        assert tool.annotations is not None, f"{tool.name} has no annotations"
        assert tool.annotations.openWorldHint is False, f"{tool.name} is not local-only"


@pytest.mark.asyncio
async def test_read_and_write_tools_are_annotated_distinctly(tmp_path: Path) -> None:
    tools = await create_server(_tiny_application(tmp_path), auto_index=False).list_tools()
    annotations = {tool.name: tool.annotations for tool in tools}

    for name in READ_ONLY_TOOLS:
        assert annotations[name] is not None and annotations[name].readOnlyHint is True, name
        assert annotations[name].destructiveHint is False, name
    for name in WRITE_TOOLS:
        assert annotations[name] is not None and annotations[name].readOnlyHint is False, name
    # remove_project is the only tool that destroys data.
    assert annotations["remove_project"].destructiveHint is True
    assert annotations["index_project"].destructiveHint is False
    assert annotations["init_project"].destructiveHint is False


@pytest.mark.asyncio
async def test_every_tool_parameter_is_documented_and_bounded(tmp_path: Path) -> None:
    tools = {
        tool.name: tool
        for tool in await create_server(_tiny_application(tmp_path), auto_index=False).list_tools()
    }

    for name, tool in tools.items():
        for parameter, spec in tool.inputSchema.get("properties", {}).items():
            assert "description" in spec, f"{name}.{parameter} has no description"

    limit = tools["search_code"].inputSchema["properties"]["limit"]
    assert (limit["minimum"], limit["maximum"]) == (1, 50)
    assert tools["find_symbol"].inputSchema["properties"]["match"]["enum"] == [
        "exact",
        "prefix",
        "contains",
    ]
```

`test_server_registers_the_focused_tool_suite` (line 80) already asserts the nine names and that
`ctx` never leaks into a schema. Leave it as it is — these tests add the metadata layer on top rather
than replacing it.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_server.py -k "declares_description or annotated_distinctly" -v`

Expected: FAIL — `AssertionError: init_project has no description`.

- [ ] **Step 3: Add the server instructions and the annotation constants**

In `src/incode_mcp/server.py`, add the import:

```python
from mcp.types import Tool as MCPTool, ToolAnnotations
```

Add above `class AutoIndexingMCP`:

```python
_TOOL_INSTRUCTIONS = """\
Local Tree-sitter code indexing with hybrid semantic and full-text search over repositories on \
this machine. No code leaves the machine: embeddings are computed locally and stored in a local \
LanceDB index.

search_code answers "where is the code that does X". find_symbol resolves a declaration whose name \
is already known. file_outline lists one file's structure without returning code. Both search \
tools return chunk_id values that get_chunk expands to full text.

Scope defaults to the active MCP root, or the nearest .ci-mcp/project.toml above the working \
directory. Searching every registered project requires all_projects=true, so cross-project results \
are never mixed in by accident.

In the default lazy mode the first code query builds the initial index, which can take minutes on \
a large repository; it reports progress while it runs."""

# openWorldHint is False on every tool: this server touches only the local
# filesystem and a local index, never the network.
_READ_ONLY = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
_WRITES = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
_DESTRUCTIVE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False
)
```

In `AutoIndexingMCP.__init__`, replace the `instructions` argument:

```python
        super().__init__(
            "code-indexing-mcp",
            instructions=_TOOL_INSTRUCTIONS,
            lifespan=self._lifespan,
        )
```

`json_response=True` is dropped: it configures the streamable-HTTP transport and is inert under the
stdio transport this server runs (`run_server` at the bottom of the file).

- [ ] **Step 4: Declare the three write tools**

Replace `init_project`, `index_project`, and `remove_project` in `create_server`:

```python
    @mcp.tool(
        title="Initialize project",
        description=(
            "Register a directory as an indexable project and write its local "
            ".ci-mcp/project.toml marker, which holds a checkout-local id and the scan "
            "configuration. Returns the project id, name, root, and scan settings. Building the "
            "index is a separate operation (index_project). Re-running on an already-initialized "
            "directory returns the existing project unless force_new_id is set."
        ),
        annotations=_WRITES,
    )
    async def init_project(
        ctx: ServerContext,
        path: Annotated[
            str | None,
            Field(
                description=(
                    "Directory to initialize. Defaults to the single MCP root when exactly one "
                    "is offered, otherwise to the server's working directory."
                )
            ),
        ] = None,
        name: Annotated[
            str | None,
            Field(description="Display name for the project. Defaults to the directory name."),
        ] = None,
        force_new_id: Annotated[
            bool,
            Field(
                description=(
                    "Mint a new project id even if a marker already exists, orphaning the "
                    "previous index for this directory."
                )
            ),
        ] = False,
    ) -> ProjectInfo:
        roots = await _startup_roots(ctx, discover=True)
        return await asyncio.to_thread(
            app.init_project,
            path,
            name,
            force_new_id,
            roots=roots,
        )

    @mcp.tool(
        title="Index project",
        description=(
            "Incrementally index a project: scan for supported source files, parse changed files "
            "with Tree-sitter, embed their chunks, and commit them. Files whose size, mtime, and "
            "content hash are unchanged are skipped without being re-read. Returns per-phase "
            "counts and durations plus any per-file errors. Indexes Python, Java, JavaScript, and "
            "TypeScript only, skipping symlinks, binaries, and files over 1 MiB."
        ),
        annotations=_WRITES,
    )
    async def index_project(
        ctx: ServerContext,
        project: Annotated[
            str | None,
            Field(
                description=(
                    "Project id, name, or path. Defaults to the active MCP root or the nearest "
                    ".ci-mcp/project.toml."
                )
            ),
        ] = None,
        force: Annotated[
            bool,
            Field(
                description=(
                    "Re-parse and re-embed every discovered file, ignoring change detection. "
                    "Use after changing the embedding model or chunking settings."
                )
            ),
        ] = False,
    ) -> IndexReport:
        # Only wait for discovery, not the outcome of the automatic index this tool is
        # meant to let callers manually recover from. If startup indexing is still
        # running, app.index_project's 0-timeout file lock raises INDEX_BUSY, which is
        # acceptable, pre-existing behavior.
        roots = await _startup_roots(ctx, discover=True)
        await ctx.report_progress(0, 1, "Indexing project")
        report = await asyncio.to_thread(app.index_project, project, roots=roots, force=force)
        await ctx.report_progress(1, 1, "Index complete")
        return report

    @mcp.tool(
        title="Remove project",
        description=(
            "Permanently delete a project's registration and its entire on-disk index partition. "
            "The .ci-mcp/project.toml marker in the working tree is left in place, so a later "
            "init_project re-registers the same id with an empty index. Irreversible: the only "
            "way back is a full re-index. Returns whether a registration existed."
        ),
        annotations=_DESTRUCTIVE,
    )
    async def remove_project(
        ctx: ServerContext,
        project: Annotated[
            str, Field(description="Project id, name, or path to remove. Required — no default.")
        ],
    ) -> RemovalReport:
        del ctx
        return await asyncio.to_thread(app.remove_project, project)
```

Add to the imports at the top of `server.py`:

```python
from typing import Annotated, Literal

from pydantic import Field
```

- [ ] **Step 5: Declare the six read tools**

Replace `project_status`, `list_projects`, `search_code`, `find_symbol`, `file_outline`, and
`get_chunk`:

```python
    @mcp.tool(
        title="Project status",
        description=(
            "Report one project's index state — pending, indexing, ready, partial, or error — "
            "with its indexed file count and chunk count. Reads the index without modifying it, "
            "and does not check the filesystem for changes; index_project does that."
        ),
        annotations=_READ_ONLY,
    )
    async def project_status(
        ctx: ServerContext,
        project: Annotated[
            str | None,
            Field(
                description=(
                    "Project id, name, or path. Defaults to the active MCP root or the nearest "
                    ".ci-mcp/project.toml."
                )
            ),
        ] = None,
    ) -> ProjectStatus:
        roots = await _startup_roots(ctx, discover=True)
        return await asyncio.to_thread(app.project_status, project, roots=roots)

    @mcp.tool(
        title="List projects",
        description=(
            "List every project registered with this server — id, name, root directory, and scan "
            "configuration — sorted by name. Takes no arguments and returns registrations only, "
            "not index state; project_status reports that."
        ),
        annotations=_READ_ONLY,
    )
    async def list_projects(ctx: ServerContext) -> list[ProjectInfo]:
        del ctx
        return await asyncio.to_thread(app.list_projects)

    @mcp.tool(
        title="Search code",
        description=(
            "Hybrid semantic and keyword search over indexed code chunks. Returns hits ranked by "
            "relevance, each with a code snippet, file path, line range, and a chunk_id that "
            "get_chunk expands to the full text. Searches indexed source only — not commit "
            "history, not comments in unindexed files, and not files excluded by .gitignore or "
            "the 1 MiB size cap. For a declaration whose name is already known, find_symbol is "
            "direct; for one file's structure, file_outline is cheaper."
        ),
        annotations=_READ_ONLY,
    )
    async def search_code(
        ctx: ServerContext,
        query: Annotated[
            str,
            Field(
                description=(
                    "What to look for, as natural language or keywords. Matched against chunk "
                    "text and against normalized identifier names."
                )
            ),
        ],
        projects: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Restrict the search to these project ids, names, or paths. Mutually "
                    "exclusive with all_projects."
                )
            ),
        ] = None,
        all_projects: Annotated[
            bool,
            Field(
                description=(
                    "Search every registered project. Off by default so results from unrelated "
                    "repositories are never mixed in implicitly."
                )
            ),
        ] = False,
        languages: Annotated[
            list[LanguageName] | None,
            Field(description="Restrict to these languages."),
        ] = None,
        paths: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Restrict to paths matching these glob patterns, relative to the project "
                    "root, for example 'src/*' or '**/*.py'. Patterns match from the right, so "
                    "'*.py' matches any Python file at any depth."
                )
            ),
        ] = None,
        kinds: Annotated[
            list[ChunkKind] | None,
            Field(description="Restrict to these chunk kinds."),
        ] = None,
        limit: Annotated[
            int, Field(ge=1, le=50, description="Maximum hits to return. Hard cap of 50.")
        ] = 8,
    ) -> SearchResponse:
        roots = await _startup_roots(ctx, indexes=True)
        project_ids = await asyncio.to_thread(
            app.resolve_search_scope, projects, all_projects, roots
        )
        await _wait_for_startup_projects(ctx, roots, project_ids)
        return await asyncio.to_thread(
            app.search_code,
            query,
            projects=project_ids,
            all_projects=False,
            languages=languages,
            paths=paths,
            kinds=kinds,
            limit=limit,
            roots=roots,
        )

    @mcp.tool(
        title="Find symbol",
        description=(
            "Look up indexed code chunks by symbol name, matching exactly, by prefix, or by "
            "substring. Returns hits ordered by path and line, each with a snippet and a "
            "chunk_id. Matches declaration names only — not call sites, imports, or other "
            "references. For a conceptual query rather than a known name, search_code applies."
        ),
        annotations=_READ_ONLY,
    )
    async def find_symbol(
        ctx: ServerContext,
        name: Annotated[
            str,
            Field(
                description=(
                    "Symbol name to look up. Either the bare name or the dotted qualified name, "
                    "for example 'LanceStore' or 'LanceStore.upsert_project'."
                )
            ),
        ],
        project: Annotated[
            str | None,
            Field(
                description=(
                    "Project id, name, or path. Defaults to the active MCP root or the nearest "
                    ".ci-mcp/project.toml."
                )
            ),
        ] = None,
        match: Annotated[
            Literal["exact", "prefix", "contains"],
            Field(
                description=(
                    "How to compare name against stored symbols. 'exact' requires a full match "
                    "on the bare or qualified name."
                )
            ),
        ] = "exact",
        kinds: Annotated[
            list[ChunkKind] | None,
            Field(description="Restrict to these chunk kinds."),
        ] = None,
        limit: Annotated[
            int, Field(ge=1, le=50, description="Maximum hits to return. Hard cap of 50.")
        ] = 20,
    ) -> SymbolResponse:
        roots = await _startup_roots(ctx, indexes=True)
        resolved = await asyncio.to_thread(app.resolve_project, project, roots)
        await _wait_for_startup_projects(ctx, roots, [resolved.id])
        return await asyncio.to_thread(
            app.find_symbol,
            name,
            resolved.id,
            match=match,
            kinds=kinds,
            limit=limit,
            roots=roots,
        )

    @mcp.tool(
        title="File outline",
        description=(
            "List the symbols declared in one indexed file, in source order, with kind, "
            "qualified name, parent, and line range. Returns structure metadata only, never code "
            "text, so it is the cheap way to understand a file before fetching parts of it. The "
            "file must already be indexed."
        ),
        annotations=_READ_ONLY,
    )
    async def file_outline(
        ctx: ServerContext,
        path: Annotated[
            str,
            Field(
                description=(
                    "File path relative to the project root, using forward slashes, exactly as "
                    "reported in search_code hits."
                )
            ),
        ],
        project: Annotated[
            str | None,
            Field(
                description=(
                    "Project id, name, or path. Defaults to the active MCP root or the nearest "
                    ".ci-mcp/project.toml."
                )
            ),
        ] = None,
    ) -> OutlineResponse:
        roots = await _startup_roots(ctx, indexes=True)
        resolved = await asyncio.to_thread(app.resolve_project, project, roots)
        await _wait_for_startup_projects(ctx, roots, [resolved.id])
        return await asyncio.to_thread(app.file_outline, path, resolved.id, roots=roots)

    @mcp.tool(
        title="Get chunk",
        description=(
            "Fetch one indexed chunk's full stored text by the chunk_id returned from search_code "
            "or find_symbol, with its path, symbol, and line range. Chunk ids are content-derived "
            "and change when the file is re-indexed, so a stale id returns CHUNK_NOT_FOUND rather "
            "than the wrong code."
        ),
        annotations=_READ_ONLY,
    )
    async def get_chunk(
        ctx: ServerContext,
        chunk_id: Annotated[
            str, Field(description="Chunk id from a search_code or find_symbol hit.")
        ],
    ) -> CodeChunk:
        del ctx
        return await asyncio.to_thread(app.get_chunk, chunk_id)
```

- [ ] **Step 6: Define the two closed vocabularies**

The `ChunkKind` and `LanguageName` aliases used above must exist. Add them to
`src/incode_mcp/models.py`, next to the existing scanning constants, so the extractor's vocabulary
has one home:

```python
# The kinds TreeSitterExtractor emits, plus the "_part" variants it produces when a
# definition is split across chunks. Closed so MCP clients get an enum instead of a
# free-text field; extend both halves together when a query file gains a capture.
ChunkKind = Literal[
    "annotation",
    "class",
    "constant",
    "constructor",
    "enum",
    "function",
    "interface",
    "method",
    "module",
    "record",
    "type",
    "annotation_part",
    "class_part",
    "constant_part",
    "constructor_part",
    "enum_part",
    "function_part",
    "interface_part",
    "method_part",
    "record_part",
    "type_part",
]
```

The ten base kinds are exactly the `@definition.*` captures in `src/incode_mcp/queries/*.scm`
(`annotation`, `class`, `constant`, `constructor`, `enum`, `function`, `interface`, `method`,
`record`, `type` — verified with
`grep -oh "@definition\.[a-z_]*" src/incode_mcp/queries/*.scm | sort -u`). `module` is not a capture;
`_module_chunks` synthesises it. The `_part` variants come from `_chunks_for_range`, which appends
`"_part"` to every kind except `module`.

# Mirrors scanner.LANGUAGES values. Kept here rather than imported from scanner so
# models stays free of scanner imports.
LanguageName = Literal["python", "java", "javascript", "typescript", "tsx"]
```

Add `Literal` to that file's `typing` import, then import both into `server.py`'s existing
`from .models import (...)` block.

- [ ] **Step 7: Guard the vocabularies against drift**

Add to `tests/test_extractor.py`:

```python
def test_chunk_kind_literal_covers_every_kind_the_queries_capture() -> None:
    from typing import get_args

    from incode_mcp.models import ChunkKind

    declared = set(get_args(ChunkKind))
    captured = set()
    for language in ("python", "java", "javascript", "typescript", "tsx"):
        text = files("incode_mcp.queries").joinpath(f"{language}.scm").read_text()
        captured |= {
            line.split("@definition.", 1)[1].split()[0].strip(")")
            for line in text.splitlines()
            if "@definition." in line
        }

    missing = captured - declared
    assert not missing, f"ChunkKind is missing extractor kinds: {sorted(missing)}"
    assert {f"{kind}_part" for kind in captured if kind != "module"} <= declared
```

Add `from importlib.resources import files` to that test file if absent.

And to `tests/test_scanner.py`:

```python
def test_language_name_literal_matches_scanner_languages() -> None:
    from typing import get_args

    from incode_mcp.models import LanguageName
    from incode_mcp.scanner import LANGUAGES

    assert set(get_args(LanguageName)) == set(LANGUAGES.values())
```

- [ ] **Step 8: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_server.py tests/test_extractor.py tests/test_scanner.py -q`

Expected: all pass. If `test_chunk_kind_literal_covers_every_kind_the_queries_capture` fails, the
`ChunkKind` list in Step 6 is missing a kind the `.scm` files capture — add it rather than relaxing
the test.

- [ ] **Step 9: Run the full suite**

Run: `.venv/bin/python -m pytest -q`

Expected: no failures, with the 5 tests from this task added. A pre-existing test that calls a tool
with `limit` above 50 or a bogus `match` will now fail with a Pydantic validation error — that is the
intended new behaviour, so update the test to assert the validation error rather than loosening the
schema.

- [ ] **Step 10: Lint, type-check, commit**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy src tests
git add src/incode_mcp/server.py src/incode_mcp/models.py tests/
git commit -m "feat: describe, title, and annotate every MCP tool"
```

---

### Task 3: Surface error details at the MCP boundary

**Files:**
- Modify: `src/incode_mcp/server.py` (add decorator, apply to nine handlers)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `IncodeError.for_client()` from Task 1; the nine decorated handlers from Task 2.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server.py`:

```python
@pytest.mark.asyncio
async def test_tool_error_carries_code_and_details(tmp_path: Path) -> None:
    server = create_server(_tiny_application(tmp_path), auto_index=False)

    with pytest.raises(ToolError) as caught:
        await server.call_tool("get_chunk", {"chunk_id": "missing"})

    message = str(caught.value)
    assert "CHUNK_NOT_FOUND" in message
    assert "chunk_id=missing" in message
    assert "search_code or find_symbol" in message
```

Add `from mcp.server.fastmcp.exceptions import ToolError` to the imports. `_tiny_application` is the
helper added in Task 2 Step 1; if Task 3 is executed before Task 2, add it here instead.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_server.py -k carries_code_and_details -v`

Expected: FAIL — the message contains `CHUNK_NOT_FOUND: Unknown chunk: missing; ...` but not
`chunk_id=missing`, because `details` is dropped.

- [ ] **Step 3: Write the decorator**

Add to `src/incode_mcp/server.py`, above `create_server`:

```python
def _with_error_details[**P, R](
    handler: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """Re-raise IncodeError as a ToolError that keeps its code and details.

    FastMCP stringifies an uncaught exception, and ``IncodeError.__str__`` omits
    ``details`` on purpose, so the machine-readable half of every error — which
    project, how long it waited, which memory ceiling — never reached the client.
    ``functools.wraps`` keeps the signature FastMCP introspects for the schema.
    """

    @wraps(handler)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await handler(*args, **kwargs)
        except IncodeError as exc:
            raise ToolError(exc.for_client()) from exc

    return wrapper
```

Add the imports:

```python
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import partial, wraps

from mcp.server.fastmcp.exceptions import ToolError
```

- [ ] **Step 4: Apply it to all nine handlers**

Insert `@_with_error_details` between each `@mcp.tool(...)` and its `async def`, so the decorator
order is:

```python
    @mcp.tool(
        title="Get chunk",
        description=(...),
        annotations=_READ_ONLY,
    )
    @_with_error_details
    async def get_chunk(
        ctx: ServerContext,
        chunk_id: Annotated[str, Field(description="Chunk id from a search_code or find_symbol hit.")],
    ) -> CodeChunk:
        del ctx
        return await asyncio.to_thread(app.get_chunk, chunk_id)
```

`@mcp.tool()` must stay outermost so it registers the wrapper, and `wraps` is what lets FastMCP
still see the annotated signature through it.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_server.py -k carries_code_and_details -v`

Expected: PASS.

- [ ] **Step 6: Confirm the schemas survived the decorator**

Run:

```bash
.venv/bin/python - <<'PY'
import asyncio, json, tempfile
from pathlib import Path
from incode_mcp.application import Application, RuntimePaths
from incode_mcp.server import create_server

class F:
    model_id = "f"; dimension = 8
    def embed_passages(self, texts): return [[0.0] * 8 for _ in texts]
    def embed_query(self, text): return [0.0] * 8

tmp = Path(tempfile.mkdtemp())
app = Application(RuntimePaths(data=tmp / "d", cache=tmp / "c"), embedder=F(), cwd=tmp)
tools = asyncio.run(create_server(app, auto_index=False).list_tools())
for tool in tools:
    props = tool.inputSchema["properties"]
    undocumented = [name for name, spec in props.items() if "description" not in spec]
    print(f"{tool.name:16} params={len(props):2} undocumented={undocumented} ctx_leaked={'ctx' in props}")
search = next(t for t in tools if t.name == "search_code")
print("limit bounds:", search.inputSchema["properties"]["limit"].get("minimum"),
      search.inputSchema["properties"]["limit"].get("maximum"))
symbol = next(t for t in tools if t.name == "find_symbol")
print("match enum:", symbol.inputSchema["properties"]["match"].get("enum"))
PY
```

Expected: every line shows `undocumented=[] ctx_leaked=False`, `limit bounds: 1 50`, and
`match enum: ['exact', 'prefix', 'contains']`. If any parameter is undocumented the `Annotated`
wrapper on it was missed in Task 2.

- [ ] **Step 7: Run the full suite, lint, type-check**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy src tests
```

Expected: no failures, clean lint and types. `mypy` matters here: the `[**P, R]` generic on
`_with_error_details` is what proves the decorator preserved each handler's signature.

- [ ] **Step 8: Document the surface**

In `README.md`, replace the bare tool list at lines 100-101 with a table:

```markdown
The server exposes nine tools. Read tools are annotated `readOnlyHint` so hosts may auto-approve
them; `remove_project` is annotated `destructiveHint`.

| Tool | Kind | Purpose |
| --- | --- | --- |
| `init_project` | write | Register a directory and write its `.ci-mcp/project.toml` marker. |
| `index_project` | write | Incrementally scan, parse, embed, and commit changed files. |
| `remove_project` | destructive | Delete a registration and its whole index partition. |
| `project_status` | read | Index state plus file and chunk counts. |
| `list_projects` | read | Every registered project, sorted by name. |
| `search_code` | read | Hybrid semantic and keyword search returning ranked snippets. |
| `find_symbol` | read | Exact, prefix, or substring lookup of declaration names. |
| `file_outline` | read | One file's declared symbols, metadata only. |
| `get_chunk` | read | Full stored text for one `chunk_id`. |

`limit` is capped at 50 and `match` accepts only `exact`, `prefix`, or `contains`; both are
enforced by the tool schema, so an out-of-range value is rejected rather than silently clamped.
```

- [ ] **Step 9: Commit**

```bash
git add src/incode_mcp/server.py tests/test_server.py README.md
git commit -m "feat: surface IncodeError codes and details through MCP tool errors"
```

---

## Self-Review

**Spec coverage.** Review items A1 (descriptions, Task 2), A2 (annotations, Task 2), A4 (schemas,
Task 2 Steps 5–7), A5 (error details and hints, Tasks 1 and 3), A6 (`instructions`, Task 2 Step 3),
A7 (`json_response`, Task 2 Step 3) are all covered. A3 (`get_chunk` payload) is deliberately out of
scope — it is [plan 3](2026-07-27-get-chunk-projection.md).

**Type consistency.** `ChunkKind` and `LanguageName` are defined once in Task 2 Step 6 and used in
`search_code` and `find_symbol` in Step 5. `IncodeError.for_client()` is defined in Task 1 Step 3
and consumed in Task 3 Step 3. `_READ_ONLY` / `_WRITES` / `_DESTRUCTIVE` are defined in Task 2
Step 3 and used in Steps 4–5. `_with_error_details` is defined in Task 3 Step 3 and applied in
Step 4.

**Known follow-on risk.** Task 2 tightens `limit` and `match`, so any existing test that passed
`limit=100` or an invalid `match` through a *tool* now raises a validation error. The service-level
clamps at `search.py:42` and `search.py:83` stay in place because the CLI, the daemon, and
`Application` are callable directly and are not schema-validated.
