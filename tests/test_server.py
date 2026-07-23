import asyncio
import threading
from pathlib import Path

import pytest
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session

from incode_mcp.application import Application, RuntimePaths
from incode_mcp.errors import ErrorCode, IncodeError
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


class FlakyEmbedder(TinyEmbedder):
    """Fails the first call to embed_passages, then behaves like TinyEmbedder.

    Raises MODEL_UNAVAILABLE specifically because the indexer otherwise treats a
    per-file embedding failure as a recoverable per-file issue (recorded on the
    file, indexing still "completes"); MODEL_UNAVAILABLE is the one error the
    indexer re-raises so it fails the whole indexing job, which is what we need
    to exercise the startup coordinator's job-level failure/retry handling.
    """

    def __init__(self) -> None:
        self.calls = 0

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == 1:
            raise IncodeError(ErrorCode.MODEL_UNAVAILABLE, "embedding backend unavailable")
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
        for _ in range(100):
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


@pytest.mark.asyncio
async def test_failed_startup_index_is_retried_on_next_tool_call(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    (root / "main.py").write_text("def answer():\n    return 42\n")
    embedder = FlakyEmbedder()
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

        first = await client.call_tool("search_code", {"query": "answer"})
        assert first.isError

        second = await client.call_tool("search_code", {"query": "answer"})
        assert not second.isError


@pytest.mark.asyncio
async def test_index_project_tool_recovers_after_startup_failure(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    (root / "main.py").write_text("def answer():\n    return 42\n")
    embedder = FlakyEmbedder()
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
        failed = await client.call_tool("search_code", {"query": "answer"})
        assert failed.isError

    # A brand-new session gets a fresh StartupCoordinator with no memory of the
    # earlier failure, and this client doesn't advertise any roots, so the
    # explicit index_project call below can't race a background retry for the
    # same project - it resolves the project by name and indexes it directly.
    project_name = app.list_projects()[0].name

    async def list_roots_none(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots_none
    ) as client:
        recovered = await client.call_tool("index_project", {"project": project_name})

    assert not recovered.isError


@pytest.mark.asyncio
async def test_discovery_is_not_blocked_by_concurrent_indexing(tmp_path: Path) -> None:
    root_a = tmp_path / "project_a"
    root_a.mkdir()
    (root_a / "pyproject.toml").write_text("[project]\nname = 'project-a'\n")
    (root_a / "main.py").write_text("def a():\n    return 1\n")

    root_b = tmp_path / "project_b"
    root_b.mkdir()
    (root_b / "pyproject.toml").write_text("[project]\nname = 'project-b'\n")
    (root_b / "main.py").write_text("def b():\n    return 2\n")

    embedder = BlockingEmbedder()
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=embedder,
        cwd=tmp_path,
    )
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(
            roots=[types.Root(uri=root_a.as_uri()), types.Root(uri=root_b.as_uri())]
        )

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.list_tools()
        assert await asyncio.to_thread(embedder.started.wait, 1)

        # Discovery for both roots should complete quickly even though one of
        # them is now stuck indexing (blocked on the embedder) - discovery no
        # longer shares the capacity limiter with indexing.
        for _ in range(100):
            if (root_a / ".incode" / "project.toml").exists() and (
                root_b / ".incode" / "project.toml"
            ).exists():
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("expected both roots to be discovered while indexing was still blocked")

        # Exactly one of the two roots won the capacity limiter and is stuck
        # indexing (blocked on the embedder); target the other one so this
        # assertion exercises discovery, not indexing, of that root.
        blocked_root = root_a if app.project_status(roots=[root_a]).state == "indexing" else root_b
        other_root = root_b if blocked_root is root_a else root_a

        result = await asyncio.wait_for(
            client.call_tool("project_status", {"project": other_root.name}), timeout=5
        )

        embedder.release.set()

    assert not result.isError
