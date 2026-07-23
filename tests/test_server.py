import asyncio
import threading
from pathlib import Path

import pytest
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session

from incode_mcp.application import Application, RuntimePaths
from incode_mcp.server import create_server


class TinyEmbedder:
    model_id = "test/tiny"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, float(len(text))]


class BlockingEmbedder(TinyEmbedder):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.started.set()
        assert self.release.wait(timeout=5)
        return super().embed_passages(texts)


@pytest.mark.asyncio
async def test_server_registers_the_focused_tool_suite(tmp_path: Path) -> None:
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    server = create_server(app)

    tools = await server.list_tools()

    assert {tool.name for tool in tools} == {
        "init_project",
        "index_project",
        "project_status",
        "list_projects",
        "remove_project",
        "search_code",
        "find_symbol",
        "file_outline",
        "get_chunk",
    }
    assert all("ctx" not in tool.inputSchema.get("properties", {}) for tool in tools)


@pytest.mark.asyncio
async def test_server_starts_background_indexing_when_client_lists_tools(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    (root / "main.py").write_text("def answer():\n    return 42\n")
    embedder = BlockingEmbedder()
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=embedder,
        cwd=tmp_path,
    )
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        tools = await client.list_tools()

        assert tools
        assert await asyncio.to_thread(embedder.started.wait, 1)
        assert app.project_status(roots=[root]).state == "indexing"

        embedder.release.set()
        for _ in range(20):
            if app.project_status(roots=[root]).state == "ready":
                break
            await asyncio.sleep(0.05)

    assert app.project_status(roots=[root]).state == "ready"


@pytest.mark.asyncio
async def test_code_query_waits_for_startup_index_to_finish(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package.json").write_text('{"name": "project"}\n')
    (root / "main.js").write_text("export function answer() { return 42; }\n")
    embedder = BlockingEmbedder()
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=embedder,
        cwd=tmp_path,
    )
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.list_tools()
        assert await asyncio.to_thread(embedder.started.wait, 1)

        query = asyncio.create_task(client.call_tool("search_code", {"query": "answer"}))
        await asyncio.sleep(0.05)
        assert not query.done()

        embedder.release.set()
        result = await query

    assert result


@pytest.mark.asyncio
async def test_auto_index_can_be_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    (root / "main.py").write_text("value = 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    monkeypatch.setenv("INCODE_AUTO_INDEX", "0")
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.list_tools()
        await client.call_tool("search_code", {"query": "value"})
        await asyncio.sleep(0.05)

    assert not (root / ".incode").exists()
    assert app.list_projects() == []
