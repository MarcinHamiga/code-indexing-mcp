"""FastMCP stdio adapter."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from .application import Application
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

logger = logging.getLogger(__name__)


@dataclass
class _StartupJob:
    discovered: anyio.Event = field(default_factory=anyio.Event)
    ready: anyio.Event = field(default_factory=anyio.Event)
    error: BaseException | None = None


class StartupCoordinator:
    def __init__(
        self, application: Application, task_group: anyio.abc.TaskGroup, *, enabled: bool
    ) -> None:
        self.application = application
        self.task_group = task_group
        self.enabled = enabled
        self._jobs: dict[Path, _StartupJob] = {}
        self._lock = asyncio.Lock()
        self._limiter = anyio.CapacityLimiter(1)

    async def schedule(self, roots: list[Path]) -> None:
        if not self.enabled:
            return
        async with self._lock:
            for root in roots:
                root = root.resolve()
                if root in self._jobs:
                    continue
                job = _StartupJob()
                self._jobs[root] = job
                self.task_group.start_soon(self._run, root, job)

    async def wait_for_discovery(self, roots: list[Path]) -> None:
        for job in await self._jobs_for(roots):
            await job.discovered.wait()
            self._raise_error(job)

    async def wait_for_ready(self, roots: list[Path]) -> None:
        for job in await self._jobs_for(roots):
            await job.ready.wait()
            self._raise_error(job)

    async def _jobs_for(self, roots: list[Path]) -> list[_StartupJob]:
        async with self._lock:
            return [self._jobs[root.resolve()] for root in roots if root.resolve() in self._jobs]

    async def _run(self, root: Path, job: _StartupJob) -> None:
        try:
            async with self._limiter:
                project = await asyncio.to_thread(self.application.discover_project, root)
                job.discovered.set()
                if project is None:
                    logger.info("Skipping automatic indexing for non-project root: %s", root)
                    return
                report = await asyncio.to_thread(
                    self.application.index_project,
                    project.id,
                    wait_for_lock=True,
                )
                logger.info(
                    "Automatic indexing complete for %s: %s files indexed",
                    project.root,
                    report.indexed_files,
                )
        except Exception as exc:
            job.error = exc
            if not job.discovered.is_set():
                job.discovered.set()
            logger.exception("Automatic indexing failed for %s", root)
        finally:
            job.ready.set()

    @staticmethod
    def _raise_error(job: _StartupJob) -> None:
        if job.error is not None:
            raise job.error


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


def _auto_index_enabled(value: bool | None) -> bool:
    if value is not None:
        return value
    return os.environ.get("INCODE_AUTO_INDEX", "1").lower() not in {"0", "false", "no"}


def _coordinator(ctx: ServerContext) -> StartupCoordinator | None:
    try:
        return ctx.request_context.lifespan_context
    except ValueError:
        return None


async def _startup_roots(ctx: ServerContext, *, wait_for: str | None = None) -> list[Path]:
    roots = await _roots(ctx)
    coordinator = _coordinator(ctx)
    if coordinator is None:
        return roots
    await coordinator.schedule(roots)
    if wait_for == "discovery":
        await coordinator.wait_for_discovery(roots)
    elif wait_for == "ready":
        await coordinator.wait_for_ready(roots)
    return roots


class AutoIndexingMCP(FastMCP):
    def __init__(self, application: Application, *, auto_index: bool) -> None:
        self.application = application
        self.auto_index = auto_index
        super().__init__(
            "code-indexing-mcp",
            instructions="Local Tree-sitter code indexing and hybrid search.",
            json_response=True,
            lifespan=self._lifespan,
        )

    @asynccontextmanager
    async def _lifespan(self, _: FastMCP) -> AsyncIterator[StartupCoordinator]:
        async with anyio.create_task_group() as task_group:
            yield StartupCoordinator(self.application, task_group, enabled=self.auto_index)

    async def list_tools(self) -> Any:
        tools = await super().list_tools()
        if not self.auto_index:
            return tools
        context = self.get_context()
        coordinator = _coordinator(context)
        if coordinator is not None:
            await coordinator.schedule(await _roots(context))
        return tools


def create_server(
    application: Application | None = None, *, auto_index: bool | None = None
) -> FastMCP:
    app = application or Application.from_environment()
    mcp = AutoIndexingMCP(app, auto_index=_auto_index_enabled(auto_index))

    @mcp.tool()
    async def init_project(
        ctx: ServerContext,
        path: str | None = None,
        name: str | None = None,
        force_new_id: bool = False,
    ) -> ProjectInfo:
        roots = await _startup_roots(ctx, wait_for="discovery")
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
        roots = await _startup_roots(ctx, wait_for="ready")
        await ctx.report_progress(0, 1, "Indexing project")
        report = await asyncio.to_thread(app.index_project, project, roots=roots, force=force)
        await ctx.report_progress(1, 1, "Index complete")
        return report

    @mcp.tool()
    async def project_status(ctx: ServerContext, project: str | None = None) -> ProjectStatus:
        roots = await _startup_roots(ctx, wait_for="discovery")
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
        roots = await _startup_roots(ctx, wait_for="ready")
        return await asyncio.to_thread(
            app.search_code,
            query,
            projects=projects,
            all_projects=all_projects,
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
        roots = await _startup_roots(ctx, wait_for="ready")
        return await asyncio.to_thread(
            app.find_symbol,
            name,
            project,
            match=match,
            kinds=kinds,
            limit=limit,
            roots=roots,
        )

    @mcp.tool()
    async def file_outline(
        ctx: ServerContext, path: str, project: str | None = None
    ) -> OutlineResponse:
        roots = await _startup_roots(ctx, wait_for="ready")
        return await asyncio.to_thread(app.file_outline, path, project, roots=roots)

    @mcp.tool()
    async def get_chunk(ctx: ServerContext, chunk_id: str) -> CodeChunk:
        del ctx
        return await asyncio.to_thread(app.get_chunk, chunk_id)

    return mcp


def run_server() -> None:
    create_server().run(transport="stdio")
