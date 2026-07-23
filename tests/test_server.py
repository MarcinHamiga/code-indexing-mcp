import asyncio
import threading
from pathlib import Path

import pytest
from filelock import FileLock
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


class SwitchableBlockingEmbedder(TinyEmbedder):
    def __init__(self) -> None:
        self.block = False
        self.started = threading.Event()
        self.release = threading.Event()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if self.block:
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


class FailingEmbedder(TinyEmbedder):
    def __init__(self) -> None:
        self.calls = 0

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        raise IncodeError(ErrorCode.MODEL_UNAVAILABLE, "embedding backend unavailable")


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
async def test_first_automatic_index_materializes_project_tree_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    server = create_server(app)
    walks = 0
    original_walk = app.indexer.scanner._walk

    def counted_walk(path: Path) -> tuple[list[Path], list[Path]]:
        nonlocal walks
        walks += 1
        return original_walk(path)

    monkeypatch.setattr(app.indexer.scanner, "_walk", counted_walk)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.list_tools()
        result = await client.call_tool("search_code", {"query": "answer"})

    assert not result.isError
    assert walks == 1


@pytest.mark.asyncio
async def test_server_shutdown_waits_for_active_startup_index(tmp_path: Path) -> None:
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
    entered = asyncio.Event()
    leave = asyncio.Event()

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async def open_session() -> None:
        async with create_connected_server_and_client_session(
            server, list_roots_callback=list_roots
        ) as client:
            await client.list_tools()
            assert await asyncio.to_thread(embedder.started.wait, 1)
            entered.set()
            await leave.wait()

    session = asyncio.create_task(open_session())
    await asyncio.wait_for(entered.wait(), timeout=2)
    try:
        leave.set()
        await asyncio.sleep(0.05)
        assert not session.done()
    finally:
        embedder.release.set()
        await asyncio.wait_for(session, timeout=2)

    assert app.project_status(roots=[root]).state == "ready"


@pytest.mark.asyncio
async def test_server_shutdown_cancels_startup_job_waiting_for_index_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    project = app.init_project(root)
    server = create_server(app)
    attempted = threading.Event()
    original_index = app.indexer.index

    def observed_index(*args: object, **kwargs: object) -> object:
        attempted.set()
        return original_index(*args, **kwargs)

    monkeypatch.setattr(app.indexer, "index", observed_index)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async def open_session() -> None:
        async with create_connected_server_and_client_session(
            server, list_roots_callback=list_roots
        ) as client:
            await client.list_tools()
            assert await asyncio.to_thread(attempted.wait, 1)

    lock = FileLock(paths.data / "locks" / f"{project.id}.lock")
    with lock:
        await asyncio.wait_for(open_session(), timeout=2)

    for _ in range(50):
        if app.project_status(project.id).state == "ready":
            break
        await asyncio.sleep(0.01)

    assert app.project_status(project.id).state == "pending"


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
async def test_explicit_code_query_ignores_unrelated_startup_index(tmp_path: Path) -> None:
    ready_root = tmp_path / "ready"
    ready_root.mkdir()
    (ready_root / "main.py").write_text("def answer():\n    return 42\n")

    startup_root = tmp_path / "startup"
    startup_root.mkdir()
    (startup_root / "pyproject.toml").write_text("[project]\nname = 'startup'\n")
    (startup_root / "slow.py").write_text("def slow():\n    return True\n")

    embedder = SwitchableBlockingEmbedder()
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=embedder,
        cwd=tmp_path,
    )
    ready_project = app.init_project(ready_root)
    app.index_project(ready_project.id)
    embedder.block = True
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=startup_root.as_uri())])

    try:
        async with create_connected_server_and_client_session(
            server, list_roots_callback=list_roots
        ) as client:
            await client.list_tools()
            assert await asyncio.to_thread(embedder.started.wait, 1)

            result = await asyncio.wait_for(
                client.call_tool(
                    "search_code",
                    {"query": "answer", "projects": [ready_project.id]},
                ),
                timeout=0.5,
            )

            assert not result.isError
    finally:
        embedder.release.set()


@pytest.mark.asyncio
async def test_explicit_code_query_ignores_unrelated_startup_failure(tmp_path: Path) -> None:
    ready_root = tmp_path / "ready"
    ready_root.mkdir()
    (ready_root / "main.py").write_text("def answer():\n    return 42\n")
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    setup_app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    ready_project = setup_app.init_project(ready_root)
    setup_app.index_project(ready_project.id)

    failing_root = tmp_path / "failing"
    failing_root.mkdir()
    (failing_root / "pyproject.toml").write_text("[project]\nname = 'failing'\n")
    (failing_root / "broken.py").write_text("def broken():\n    return True\n")

    embedder = FailingEmbedder()
    app = Application(paths, embedder=embedder, cwd=tmp_path)
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=failing_root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.list_tools()
        for _ in range(100):
            if embedder.calls == 1:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("expected the unrelated startup index to fail")

        result = await client.call_tool(
            "search_code",
            {"query": "answer", "projects": [ready_project.id]},
        )

    assert not result.isError


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

    assert not (root / ".ci-mcp").exists()
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
            if (root_a / ".ci-mcp" / "project.toml").exists() and (
                root_b / ".ci-mcp" / "project.toml"
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
