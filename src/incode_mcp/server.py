"""FastMCP stdio adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

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

ServerContext = Context[ServerSession, None]


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


def create_server(application: Application | None = None) -> FastMCP:
    app = application or Application.from_environment()
    mcp = FastMCP(
        "incode",
        instructions="Local Tree-sitter code indexing and hybrid search.",
        json_response=True,
    )

    @mcp.tool()
    async def init_project(
        ctx: ServerContext,
        path: str | None = None,
        name: str | None = None,
        force_new_id: bool = False,
    ) -> ProjectInfo:
        del ctx
        return await asyncio.to_thread(app.init_project, path, name, force_new_id)

    @mcp.tool()
    async def index_project(
        ctx: ServerContext, project: str | None = None, force: bool = False
    ) -> IndexReport:
        roots = await _roots(ctx)
        await ctx.report_progress(0, 1, "Indexing project")
        report = await asyncio.to_thread(app.index_project, project, roots=roots, force=force)
        await ctx.report_progress(1, 1, "Index complete")
        return report

    @mcp.tool()
    async def project_status(ctx: ServerContext, project: str | None = None) -> ProjectStatus:
        return await asyncio.to_thread(app.project_status, project, roots=await _roots(ctx))

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
        return await asyncio.to_thread(
            app.search_code,
            query,
            projects=projects,
            all_projects=all_projects,
            languages=languages,
            paths=paths,
            kinds=kinds,
            limit=limit,
            roots=await _roots(ctx),
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
        return await asyncio.to_thread(
            app.find_symbol,
            name,
            project,
            match=match,
            kinds=kinds,
            limit=limit,
            roots=await _roots(ctx),
        )

    @mcp.tool()
    async def file_outline(
        ctx: ServerContext, path: str, project: str | None = None
    ) -> OutlineResponse:
        return await asyncio.to_thread(app.file_outline, path, project, roots=await _roots(ctx))

    @mcp.tool()
    async def get_chunk(ctx: ServerContext, chunk_id: str) -> CodeChunk:
        del ctx
        return await asyncio.to_thread(app.get_chunk, chunk_id)

    return mcp


def run_server() -> None:
    create_server().run(transport="stdio")
