"""FastMCP stdio adapter."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from functools import partial, wraps
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.session import ServerSession
from mcp.types import Tool as MCPTool
from mcp.types import ToolAnnotations
from pydantic import Field
from watchfiles import awatch

from .application import Application
from .daemon import BrokerApplication
from .errors import CodeIndexingError, ErrorCode
from .models import (
    ChunkKind,
    CodeChunk,
    IndexReport,
    LanguageName,
    OutlineResponse,
    ProjectInfo,
    ProjectStatus,
    RemovalReport,
    SearchResponse,
    SymbolResponse,
)
from .settings import IndexMode, IndexSettings

logger = logging.getLogger(__name__)

SERVER_INSTRUCTIONS = (
    "Local Tree-sitter code indexing and hybrid search. "
    "When exploring code, prefer these index tools over grep-style file reading: "
    "search_code (semantic natural-language queries), find_symbol (definitions and call "
    "sites), file_outline (file structure before reading), get_chunk (exact code for a "
    "search hit). When correlating code across explicitly related services, use list_projects "
    "to discover them and search_across_projects to search the selected repositories together. "
    "Check list_projects/project_status for index freshness first and run index_project if the "
    "index is missing or stale."
)

# Bounds on the retry cadence used while another indexing job holds the global
# lock. Polling stays cancellable between non-blocking attempts, so the first
# delays are short enough to pick up a briefly held lock quickly, and the cap
# keeps a long wait from spinning.
INITIAL_RETRY_DELAY_SECONDS = 0.05
MAXIMUM_RETRY_DELAY_SECONDS = 1.0

# How often a waiting tool call re-reads the indexing progress snapshot. Fast
# enough that a client's progress bar keeps moving, slow enough that polling a
# small file costs nothing next to the indexing it is watching.
PROGRESS_POLL_SECONDS = 0.5

# Filesystem events are collapsed before they reach the indexing coordinator.
# The bounded dirty queue below provides the second layer: one more refresh is
# enough no matter how many saves happen while an index is already running.
WATCH_DEBOUNCE_MILLISECONDS = 250
WATCH_STEP_MILLISECONDS = 50
WATCH_RUST_TIMEOUT_MILLISECONDS = 1_000
WATCH_RETRY_INITIAL_SECONDS = 1.0
WATCH_RETRY_MAXIMUM_SECONDS = 30.0

# Native filesystem notifications can be lost, and Git's global excludes live
# outside the watched root. A slow stat-only reconciliation is the correctness
# backstop; normal edits still take the event-driven path above.
EAGER_RECONCILE_SECONDS = 30.0


@dataclass
class _StartupJob:
    discovered: anyio.Event = field(default_factory=anyio.Event)
    ready: anyio.Event = field(default_factory=anyio.Event)
    project_id: str | None = None
    discovery_error: BaseException | None = None
    indexing_error: BaseException | None = None
    indexes: bool = False

    @property
    def failed(self) -> bool:
        return self.discovery_error is not None or self.indexing_error is not None


class StartupCoordinator:
    def __init__(
        self,
        application: Application | BrokerApplication,
        task_group: anyio.abc.TaskGroup,
        *,
        mode: IndexMode,
        wait_seconds: int = 300,
    ) -> None:
        self.application = application
        self.task_group = task_group
        self.mode = mode
        self.wait_seconds = wait_seconds
        self._jobs: dict[Path, _StartupJob] = {}
        self._monitors: dict[Path, asyncio.Queue[None]] = {}
        self._lock = asyncio.Lock()
        self._limiter = anyio.CapacityLimiter(1)

    async def schedule(self, roots: list[Path], *, indexes: bool) -> None:
        if self.mode is IndexMode.MANUAL:
            return
        async with self._lock:
            for root in roots:
                root = root.resolve()
                existing = self._jobs.get(root)
                if existing is not None:
                    if not existing.ready.is_set():
                        continue
                    if not existing.failed:
                        if not indexes:
                            continue
                        if (
                            existing.indexes
                            and existing.project_id is not None
                            and not await self._is_stale(existing.project_id)
                        ):
                            continue
                job = _StartupJob(indexes=indexes)
                self._jobs[root] = job
                self.task_group.start_soon(self._run, root, job)

    async def _is_stale(self, project_id: str) -> bool:
        return await anyio.to_thread.run_sync(
            partial(self.application.project_is_stale, project_id),
            abandon_on_cancel=False,
        )

    async def wait_for_discovery(self, roots: list[Path]) -> None:
        for job in await self._jobs_for(roots):
            await job.discovered.wait()
            if job.discovery_error is not None:
                raise job.discovery_error

    async def has_pending_indexing(self, roots: list[Path], project_ids: set[str]) -> bool:
        """Return whether a caller would actually block waiting on these roots."""
        return any(
            job.project_id in project_ids and job.indexes and not job.ready.is_set()
            for job in await self._jobs_for(roots)
        )

    async def wait_for_ready(self, roots: list[Path], project_ids: set[str]) -> None:
        for job in await self._jobs_for(roots):
            if job.project_id not in project_ids:
                continue
            await job.ready.wait()
            if job.discovery_error is not None:
                raise job.discovery_error
            if job.indexing_error is not None:
                raise job.indexing_error

    async def _jobs_for(self, roots: list[Path]) -> list[_StartupJob]:
        async with self._lock:
            return [self._jobs[root.resolve()] for root in roots if root.resolve() in self._jobs]

    async def _run(self, root: Path, job: _StartupJob) -> None:
        try:
            project = await anyio.to_thread.run_sync(
                self.application.discover_project,
                root,
                abandon_on_cancel=False,
            )
            job.project_id = project.id if project is not None else None
            job.discovered.set()
            if project is None or not job.indexes:
                logger.info("Skipping automatic indexing for non-project root: %s", root)
                return
            await self._ensure_monitor(root, project.id)
            report = await self._index_when_free(project.id)
            logger.info(
                "Automatic indexing complete for %s: %s files indexed",
                project.root,
                report.indexed_files,
            )
        except Exception as exc:
            if not job.discovered.is_set():
                job.discovery_error = exc
                job.discovered.set()
            else:
                job.indexing_error = exc
            logger.exception("Automatic indexing failed for %s", root)
        finally:
            job.ready.set()

    async def _ensure_monitor(self, root: Path, project_id: str) -> None:
        if self.mode is not IndexMode.EAGER:
            return
        root = root.resolve()
        async with self._lock:
            if root in self._monitors:
                return
            dirty: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
            # The watch backend takes its first filesystem snapshot when its
            # task begins. Seed one freshness pass so an edit made between
            # project discovery and that snapshot cannot be missed.
            dirty.put_nowait(None)
            self._monitors[root] = dirty
            self.task_group.start_soon(self._watch_root, root, dirty)
            self.task_group.start_soon(self._refresh_dirty_root, root, project_id, dirty)

    async def _watch_root(self, root: Path, dirty: asyncio.Queue[None]) -> None:
        retry_seconds = WATCH_RETRY_INITIAL_SECONDS
        while True:
            try:
                async for _changes in awatch(
                    root,
                    debounce=WATCH_DEBOUNCE_MILLISECONDS,
                    step=WATCH_STEP_MILLISECONDS,
                    rust_timeout=WATCH_RUST_TIMEOUT_MILLISECONDS,
                    ignore_permission_denied=True,
                ):
                    retry_seconds = WATCH_RETRY_INITIAL_SECONDS
                    with suppress(asyncio.QueueFull):
                        dirty.put_nowait(None)
                logger.warning("Filesystem monitor stopped for %s; restarting", root)
            except Exception:
                logger.exception("Filesystem monitor failed for %s; restarting", root)
            await anyio.sleep(retry_seconds)
            retry_seconds = min(retry_seconds * 2, WATCH_RETRY_MAXIMUM_SECONDS)

    async def _refresh_dirty_root(
        self, root: Path, project_id: str, dirty: asyncio.Queue[None]
    ) -> None:
        while True:
            with anyio.move_on_after(EAGER_RECONCILE_SECONDS) as reconcile:
                await dirty.get()
            try:
                # The first pass either starts a refresh or waits for one that
                # was already active. Event-driven refreshes make a second pass
                # to close the race where a save landed after that job's scan;
                # the periodic reconciliation needs only its one stat pass.
                passes = 1 if reconcile.cancel_called else 2
                for _ in range(passes):
                    await self.schedule([root], indexes=True)
                    await self.wait_for_ready([root], {project_id})
            except Exception:
                logger.exception("Automatic refresh after a file change failed for %s", root)

    async def _index_when_free(self, project_id: str) -> IndexReport:
        """Index *project_id* once the machine is free, within ``wait_seconds``.

        Two separate things make a job wait: another root queued ahead of it in
        this process, and another process holding the global index lock. Both
        look identical to a first query - someone else is indexing - so one
        deadline spans both rather than restarting at the hand-off.
        """
        started = time.monotonic()
        deadline = started + self.wait_seconds
        await self._acquire_slot(deadline, started=started)
        try:
            return await self._index_with_backoff(project_id, deadline=deadline, started=started)
        finally:
            self._limiter.release()

    async def _acquire_slot(self, deadline: float, *, started: float) -> None:
        """Take this process's single indexing slot, giving up at *deadline*."""
        try:
            self._limiter.acquire_nowait()
            return
        except anyio.WouldBlock:
            pass
        remaining = deadline - time.monotonic()
        if remaining > 0:
            with anyio.move_on_after(remaining):
                await self._limiter.acquire()
                return
        raise self._busy(time.monotonic() - started)

    async def _index_with_backoff(
        self, project_id: str, *, deadline: float, started: float
    ) -> IndexReport:
        """Index *project_id*, waiting out a competing process up to *deadline*.

        The global index lock is taken non-blockingly so this task stays
        cancellable between attempts; a blocking acquire inside ``run_sync``
        could not be abandoned at shutdown. Without a deadline the retry loop
        would make a first query hang for as long as any other job runs.
        """
        delay = INITIAL_RETRY_DELAY_SECONDS
        while True:
            try:
                return await anyio.to_thread.run_sync(
                    partial(self.application.index_project, project_id),
                    abandon_on_cancel=False,
                )
            except CodeIndexingError as exc:
                if exc.code is not ErrorCode.INDEX_BUSY:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._busy(time.monotonic() - started, cause=exc) from exc
                await anyio.sleep(min(delay, remaining))
                delay = min(delay * 2, MAXIMUM_RETRY_DELAY_SECONDS)

    def _busy(self, waited: float, *, cause: CodeIndexingError | None = None) -> CodeIndexingError:
        message = (
            str(cause.args[0]) if cause is not None else "Another indexing job is already active"
        )
        details = dict(cause.details) if cause is not None else {}
        details["waited_seconds"] = round(waited, 3)
        details["wait_timeout_seconds"] = self.wait_seconds
        return CodeIndexingError(
            ErrorCode.INDEX_BUSY,
            f"{message}; gave up after waiting {waited:.1f}s",
            **details,
        )


ServerContext = Context[ServerSession, StartupCoordinator]


async def _roots(ctx: ServerContext) -> list[Path]:
    try:
        result = await ctx.session.list_roots()
    except Exception:
        return []
    roots: list[Path] = []
    for root in result.roots:
        parsed = urlparse(str(root.uri))
        if parsed.scheme == "file":
            roots.append(Path(url2pathname(unquote(parsed.path))).resolve())
    return roots


def _coordinator(ctx: ServerContext) -> StartupCoordinator | None:
    try:
        return ctx.request_context.lifespan_context
    except ValueError:
        return None


async def _startup_roots(
    ctx: ServerContext, *, discover: bool = False, indexes: bool = False
) -> list[Path]:
    roots = await _roots(ctx)
    coordinator = _coordinator(ctx)
    if coordinator is None:
        return roots
    await coordinator.schedule(roots, indexes=indexes)
    if discover or indexes:
        await coordinator.wait_for_discovery(roots)
    return roots


@dataclass
class _ProgressStream:
    """Forwards the indexing snapshot to the client for as long as it runs.

    The run being watched usually belongs to another process (the per-user
    daemon), so the numbers come from the snapshot that process publishes rather
    than from an in-process callback. Reported progress only ever moves forward,
    as the protocol requires, even when one run's snapshot replaces another's.
    """

    ctx: ServerContext
    application: Application | BrokerApplication
    project_ids: list[str]
    message: str
    highest: float = 0.0

    async def run(self) -> None:
        await self.ctx.report_progress(0, None, self.message)
        while True:
            await asyncio.sleep(PROGRESS_POLL_SECONDS)
            snapshot = next(
                (
                    found
                    for found in (
                        self.application.index_progress(project_id)
                        for project_id in self.project_ids
                    )
                    if found is not None
                ),
                None,
            )
            if snapshot is None:
                continue
            self.highest = max(self.highest, float(snapshot.files_seen))
            await self.ctx.report_progress(self.highest, snapshot.files_total, snapshot.describe())

    async def finish(self, message: str) -> None:
        """Close the bar out at 100%, whatever scale it reached."""

        total = max(self.highest, 1.0)
        await self.ctx.report_progress(total, total, message)


@asynccontextmanager
async def _reporting_index_progress(
    ctx: ServerContext,
    application: Application | BrokerApplication,
    project_ids: list[str],
    *,
    message: str,
) -> AsyncIterator[_ProgressStream]:
    """Report indexing progress for the duration of the enclosed work."""

    stream = _ProgressStream(ctx, application, project_ids, message)
    reporter = asyncio.create_task(stream.run())
    try:
        yield stream
    finally:
        reporter.cancel()
        with suppress(asyncio.CancelledError):
            await reporter


async def _wait_for_startup_projects(
    ctx: ServerContext, roots: list[Path], project_ids: list[str]
) -> None:
    coordinator = _coordinator(ctx)
    if coordinator is None:
        return
    if coordinator.mode is IndexMode.MANUAL:
        return
    projects = await asyncio.gather(
        *(
            asyncio.to_thread(coordinator.application.resolve_project, project_id)
            for project_id in project_ids
        )
    )
    # An explicit project or all_projects query can select registrations that
    # are not among the client's advertised roots. Freshen those too; otherwise
    # lazy mode would silently serve an old index for exactly those scopes.
    selected_roots = list(dict.fromkeys([*roots, *(project.root for project in projects)]))
    statuses = await asyncio.gather(
        *(
            asyncio.to_thread(coordinator.application.project_status, project.id)
            for project in projects
        )
    )
    refresh_roots = [
        project.root
        for project, status in zip(projects, statuses, strict=True)
        if status.state not in {"ready", "partial"}
    ]
    await coordinator.schedule(refresh_roots, indexes=True)
    await coordinator.wait_for_discovery(selected_roots)
    wanted = set(project_ids)
    # A lazy query blocks on any refresh its selected scope needs. Report
    # progress so the client can distinguish a slow index from a hung tool call,
    # and so the wait shows how far along it is rather than just that it exists.
    pending = await coordinator.has_pending_indexing(selected_roots, wanted)
    if not pending:
        await coordinator.wait_for_ready(selected_roots, wanted)
        return
    message = (
        "Building the initial index"
        if all(status.file_count == 0 for status in statuses)
        else "Refreshing the stale index"
    )
    logger.info("%s before serving the code query", message)
    async with _reporting_index_progress(
        ctx, coordinator.application, project_ids, message=message
    ) as stream:
        await coordinator.wait_for_ready(selected_roots, wanted)
    await stream.finish("Index ready")


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

For cross-repository debugging, use list_projects to discover related registrations, then prefer \
search_across_projects with at least two explicit project ids, names, or paths. It searches only \
that deliberate scope and globally ranks the combined results.

In the default lazy mode every project-scoped code query checks freshness and refreshes only when \
the source tree has changed. The initial refresh can take minutes on a large repository and \
reports progress while it runs."""

# openWorldHint is False on every tool: this server touches only the local
# filesystem and a local index, never the network.
_READ_ONLY = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
# Tools that answer a read query but route through _startup_roots first. On a
# root the server has not seen, that registers the project — writing a
# .ci-mcp/project.toml marker and a store row, and for the query tools building
# the initial index — so readOnlyHint would be a lie a host could act on.
_READS_AND_REGISTERS = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
_INITIALIZES = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False
)
_WRITES = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
_DESTRUCTIVE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False
)


class AutoIndexingMCP(FastMCP):
    def __init__(
        self,
        application: Application | BrokerApplication,
        *,
        mode: IndexMode,
        wait_seconds: int = 300,
    ) -> None:
        self.application = application
        self.mode = mode
        self.wait_seconds = wait_seconds
        super().__init__(
            "code-indexing-mcp",
            instructions=f"{_TOOL_INSTRUCTIONS}\n\n{SERVER_INSTRUCTIONS}",
            json_response=True,
            lifespan=self._lifespan,
        )

    @asynccontextmanager
    async def _lifespan(self, _: FastMCP) -> AsyncIterator[StartupCoordinator]:
        async with anyio.create_task_group() as task_group:
            try:
                yield StartupCoordinator(
                    self.application,
                    task_group,
                    mode=self.mode,
                    wait_seconds=self.wait_seconds,
                )
            finally:
                # Lock waiters are cancellable between non-blocking attempts. A worker
                # that has acquired the index lock remains owned until its write
                # finishes because run_sync is not abandoned on cancellation.
                task_group.cancel_scope.cancel()

    async def list_tools(self) -> list[MCPTool]:
        tools = await super().list_tools()
        if self.mode is not IndexMode.EAGER:
            return tools
        context = self.get_context()
        coordinator = _coordinator(context)
        if coordinator is not None:
            await coordinator.schedule(await _roots(context), indexes=True)
        return tools


def _with_error_details[**P, R](
    handler: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """Re-raise CodeIndexingError as a ToolError that keeps its code and details.

    FastMCP stringifies an uncaught exception, and ``CodeIndexingError.__str__`` omits
    ``details`` on purpose, so the machine-readable half of every error — which
    project, how long it waited, which memory ceiling — never reached the client.
    ``functools.wraps`` keeps the signature FastMCP introspects for the schema.
    """

    @wraps(handler)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await handler(*args, **kwargs)
        except CodeIndexingError as exc:
            raise ToolError(exc.for_client()) from exc

    return wrapper


def create_server(
    application: Application | BrokerApplication | None = None,
    *,
    auto_index: bool | None = None,
) -> FastMCP:
    app = application or Application.from_environment()
    settings = IndexSettings.from_environment()
    mode = (
        IndexMode.EAGER
        if auto_index
        else IndexMode.MANUAL
        if auto_index is not None
        else settings.mode
    )
    mcp = AutoIndexingMCP(app, mode=mode, wait_seconds=settings.index_wait_seconds)

    async def search_resolved_projects(
        ctx: ServerContext,
        query: str,
        project_ids: list[str],
        roots: list[Path],
        languages: list[LanguageName] | None,
        paths: list[str] | None,
        kinds: list[ChunkKind] | None,
        limit: int,
    ) -> SearchResponse:
        await _wait_for_startup_projects(ctx, roots, project_ids)
        # The service layer takes open list[str]; the closed Literal lists exist to
        # constrain the tool schema, and list invariance blocks passing them through.
        selected_languages: list[str] | None = list(languages) if languages else None
        selected_kinds: list[str] | None = list(kinds) if kinds else None
        return await asyncio.to_thread(
            app.search_code,
            query,
            projects=project_ids,
            all_projects=False,
            languages=selected_languages,
            paths=paths,
            kinds=selected_kinds,
            limit=limit,
            roots=roots,
        )

    @mcp.tool(
        title="Initialize project",
        description=(
            "Register a directory as an indexable project and write its local "
            ".ci-mcp/project.toml marker, which holds a checkout-local id and the scan "
            "configuration. Returns the project id, name, root, and scan settings. Building the "
            "index is a separate operation (index_project). Re-running on an already-initialized "
            "directory returns the existing project unless force_new_id is set."
        ),
        annotations=_INITIALIZES,
    )
    @_with_error_details
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
            "counts and durations plus any per-file errors. Indexes supported source files, "
            "skipping symlinks, binaries, and files over 1 MiB."
        ),
        annotations=_WRITES,
    )
    @_with_error_details
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
        # Resolved up front because progress is published per project id, and the
        # process doing the work may be the daemon rather than this one.
        resolved = await asyncio.to_thread(app.resolve_project, project, roots)
        async with _reporting_index_progress(
            ctx, app, [resolved.id], message=f"Indexing {resolved.name}"
        ) as stream:
            report = await asyncio.to_thread(
                app.index_project, resolved.id, roots=roots, force=force
            )
        await stream.finish(
            f"Indexed {report.indexed_files} files, {report.embedded_chunks} chunks embedded"
        )
        return report

    @mcp.tool(
        title="Project status",
        description=(
            "Report one project's index state — pending, indexing, ready, partial, stale, or "
            "error — with its indexed file count and chunk count. Compares eligible source "
            "metadata with the index but does not rebuild it; index_project does that. A root "
            "that is not registered yet is registered first, which writes its "
            ".ci-mcp/project.toml marker."
        ),
        annotations=_READS_AND_REGISTERS,
    )
    @_with_error_details
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
    @_with_error_details
    async def list_projects(ctx: ServerContext) -> list[ProjectInfo]:
        del ctx
        return await asyncio.to_thread(app.list_projects)

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
    @_with_error_details
    async def remove_project(
        ctx: ServerContext,
        project: Annotated[
            str, Field(description="Project id, name, or path to remove. Required — no default.")
        ],
    ) -> RemovalReport:
        del ctx
        return await asyncio.to_thread(app.remove_project, project)

    @mcp.tool(
        title="Search code",
        description=(
            "Hybrid semantic and keyword search over indexed code chunks. Returns hits ranked by "
            "relevance, each with a code snippet, file path, line range, and a chunk_id that "
            "get_chunk expands to the full text. Searches indexed source only — not commit "
            "history, not comments in unindexed files, and not files excluded by .gitignore or "
            "the 1 MiB size cap. For a declaration whose name is already known, find_symbol is "
            "direct; for one file's structure, file_outline is cheaper. A root that is not "
            "registered yet is registered and indexed before the first query is answered; later "
            "queries refresh selected projects when source metadata has changed."
        ),
        annotations=_READS_AND_REGISTERS,
    )
    @_with_error_details
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
        roots = await _startup_roots(ctx, discover=True)
        project_ids = await asyncio.to_thread(
            app.resolve_search_scope, projects, all_projects, roots
        )
        return await search_resolved_projects(
            ctx,
            query,
            project_ids,
            roots,
            languages,
            paths,
            kinds,
            limit,
        )

    @mcp.tool(
        title="Search across projects",
        description=(
            "Hybrid semantic and keyword search across an explicit set of related projects for "
            "cross-repository debugging. Accepts project ids, unique names, or paths and requires "
            "at least two distinct resolved projects. Returns one globally ranked hit list with "
            "project metadata; use list_projects first to discover the intended repositories."
        ),
        annotations=_READS_AND_REGISTERS,
    )
    @_with_error_details
    async def search_across_projects(
        ctx: ServerContext,
        query: Annotated[
            str,
            Field(
                description=(
                    "What to look for across the selected projects, as natural language or "
                    "keywords. Matched against chunk text and normalized identifier names."
                )
            ),
        ],
        projects: Annotated[
            list[str],
            Field(
                min_length=2,
                description=(
                    "At least two project ids, unique names, or paths to search together. "
                    "Selectors must resolve to at least two distinct projects."
                ),
            ),
        ],
        languages: Annotated[
            list[LanguageName] | None,
            Field(description="Restrict to these languages across the complete selected scope."),
        ] = None,
        paths: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Restrict to paths matching these glob patterns relative to each selected "
                    "project root, for example 'src/*' or '**/*.py'. Patterns match from the "
                    "right, so '*.py' matches any Python file at any depth."
                )
            ),
        ] = None,
        kinds: Annotated[
            list[ChunkKind] | None,
            Field(description="Restrict to these chunk kinds across the selected projects."),
        ] = None,
        limit: Annotated[
            int,
            Field(
                ge=1,
                le=50,
                description="Maximum globally ranked hits to return. Hard cap of 50.",
            ),
        ] = 8,
    ) -> SearchResponse:
        roots = await _startup_roots(ctx, discover=True)
        resolved_ids = await asyncio.to_thread(app.resolve_search_scope, projects, False, roots)
        project_ids = list(dict.fromkeys(resolved_ids))
        if len(project_ids) < 2:
            raise CodeIndexingError(
                ErrorCode.INVALID_FILTER,
                "search_across_projects requires at least two distinct projects",
            )
        return await search_resolved_projects(
            ctx,
            query,
            project_ids,
            roots,
            languages,
            paths,
            kinds,
            limit,
        )

    @mcp.tool(
        title="Find symbol",
        description=(
            "Look up indexed code chunks by symbol name, matching exactly, by prefix, or by "
            "substring. Returns hits ordered by path and line, each with a snippet and a "
            "chunk_id. Matches declaration names only — not call sites, imports, or other "
            "references. For a conceptual query rather than a known name, search_code applies. A "
            "root that is not registered yet is registered and indexed before the first query is "
            "answered; later queries refresh it when source metadata has changed."
        ),
        annotations=_READS_AND_REGISTERS,
    )
    @_with_error_details
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
        roots = await _startup_roots(ctx, discover=True)
        resolved = await asyncio.to_thread(app.resolve_project, project, roots)
        await _wait_for_startup_projects(ctx, roots, [resolved.id])
        selected_kinds: list[str] | None = list(kinds) if kinds else None
        return await asyncio.to_thread(
            app.find_symbol,
            name,
            resolved.id,
            match=match,
            kinds=selected_kinds,
            limit=limit,
            roots=roots,
        )

    @mcp.tool(
        title="File outline",
        description=(
            "List the symbols declared in one indexed file, in source order, with kind, "
            "qualified name, parent, and line range. Returns structure metadata only, never code "
            "text, so it is the cheap way to understand a file before fetching parts of it. The "
            "file must already be indexed; a root that is not registered yet is registered and "
            "indexed first, and a changed index is refreshed before the outline is returned."
        ),
        annotations=_READS_AND_REGISTERS,
    )
    @_with_error_details
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
        roots = await _startup_roots(ctx, discover=True)
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
    @_with_error_details
    async def get_chunk(
        ctx: ServerContext,
        chunk_id: Annotated[
            str, Field(description="Chunk id from a search_code or find_symbol hit.")
        ],
    ) -> CodeChunk:
        del ctx
        return await asyncio.to_thread(app.get_chunk, chunk_id)

    return mcp


def run_server() -> None:
    create_server().run(transport="stdio")
