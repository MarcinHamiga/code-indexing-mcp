"""FastMCP stdio adapter."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from mcp.types import Tool as MCPTool

from .application import Application
from .daemon import BrokerApplication
from .errors import ErrorCode, IncodeError
from .models import (
    CodeChunk,
    IndexReport,
    OutlineResponse,
    ProjectInfo,
    ProjectStatus,
    RemovalReport,
    SearchResponse,
    SymbolResponse,
)
from .settings import IndexMode, IndexSettings

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.application = application
        self.task_group = task_group
        self.mode = mode
        self._jobs: dict[Path, _StartupJob] = {}
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
                    if not existing.failed and (existing.indexes or not indexes):
                        continue
                job = _StartupJob(indexes=indexes)
                self._jobs[root] = job
                self.task_group.start_soon(self._run, root, job)

    async def wait_for_discovery(self, roots: list[Path]) -> None:
        for job in await self._jobs_for(roots):
            await job.discovered.wait()
            if job.discovery_error is not None:
                raise job.discovery_error

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
            async with self._limiter:
                while True:
                    try:
                        report = await anyio.to_thread.run_sync(
                            partial(self.application.index_project, project.id),
                            abandon_on_cancel=False,
                        )
                        break
                    except IncodeError as exc:
                        if exc.code is not ErrorCode.INDEX_BUSY:
                            raise
                        await anyio.sleep(0.05)
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


async def _wait_for_startup_projects(
    ctx: ServerContext, roots: list[Path], project_ids: list[str]
) -> None:
    coordinator = _coordinator(ctx)
    if coordinator is not None:
        await coordinator.wait_for_ready(roots, set(project_ids))


class AutoIndexingMCP(FastMCP):
    def __init__(self, application: Application | BrokerApplication, *, mode: IndexMode) -> None:
        self.application = application
        self.mode = mode
        super().__init__(
            "code-indexing-mcp",
            instructions="Local Tree-sitter code indexing and hybrid search.",
            json_response=True,
            lifespan=self._lifespan,
        )

    @asynccontextmanager
    async def _lifespan(self, _: FastMCP) -> AsyncIterator[StartupCoordinator]:
        async with anyio.create_task_group() as task_group:
            try:
                yield StartupCoordinator(self.application, task_group, mode=self.mode)
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


def create_server(
    application: Application | BrokerApplication | None = None,
    *,
    auto_index: bool | None = None,
) -> FastMCP:
    app = application or Application.from_environment()
    mode = (
        IndexMode.EAGER
        if auto_index
        else IndexMode.MANUAL
        if auto_index is not None
        else IndexSettings.from_environment().mode
    )
    mcp = AutoIndexingMCP(app, mode=mode)

    @mcp.tool()
    async def init_project(
        ctx: ServerContext,
        path: str | None = None,
        name: str | None = None,
        force_new_id: bool = False,
    ) -> ProjectInfo:
        roots = await _startup_roots(ctx, discover=True)
        return await asyncio.to_thread(
            app.init_project,
            path,
            name,
            force_new_id,
            roots=roots,
        )

    @mcp.tool()
    async def index_project(
        ctx: ServerContext, project: str | None = None, force: bool = False
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

    @mcp.tool()
    async def project_status(ctx: ServerContext, project: str | None = None) -> ProjectStatus:
        roots = await _startup_roots(ctx, discover=True)
        return await asyncio.to_thread(app.project_status, project, roots=roots)

    @mcp.tool()
    async def list_projects(ctx: ServerContext) -> list[ProjectInfo]:
        del ctx
        return await asyncio.to_thread(app.list_projects)

    @mcp.tool()
    async def remove_project(ctx: ServerContext, project: str) -> RemovalReport:
        del ctx
        return await asyncio.to_thread(app.remove_project, project)

    @mcp.tool()
    async def search_code(
        ctx: ServerContext,
        query: str,
        projects: list[str] | None = None,
        all_projects: bool = False,
        languages: list[str] | None = None,
        paths: list[str] | None = None,
        kinds: list[str] | None = None,
        limit: int = 8,
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

    @mcp.tool()
    async def find_symbol(
        ctx: ServerContext,
        name: str,
        project: str | None = None,
        match: str = "exact",
        kinds: list[str] | None = None,
        limit: int = 20,
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

    @mcp.tool()
    async def file_outline(
        ctx: ServerContext, path: str, project: str | None = None
    ) -> OutlineResponse:
        roots = await _startup_roots(ctx, indexes=True)
        resolved = await asyncio.to_thread(app.resolve_project, project, roots)
        await _wait_for_startup_projects(ctx, roots, [resolved.id])
        return await asyncio.to_thread(app.file_outline, path, resolved.id, roots=roots)

    @mcp.tool()
    async def get_chunk(ctx: ServerContext, chunk_id: str) -> CodeChunk:
        del ctx
        return await asyncio.to_thread(app.get_chunk, chunk_id)

    return mcp


def run_server() -> None:
    create_server().run(transport="stdio")
