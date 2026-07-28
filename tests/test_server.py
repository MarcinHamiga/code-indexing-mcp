import asyncio
import threading
from pathlib import Path

import pytest
from filelock import FileLock
from mcp import types
from mcp.server.fastmcp.exceptions import ToolError
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
async def test_default_server_defers_indexing_until_first_code_query(tmp_path: Path) -> None:
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
        assert not embedder.started.is_set()
        assert not (root / ".ci-mcp").exists()

        query = asyncio.create_task(client.call_tool("search_code", {"query": "answer"}))
        assert await asyncio.to_thread(embedder.started.wait, 5)
        assert not query.done()
        embedder.release.set()
        result = await query

    assert not result.isError
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
    server = create_server(app, auto_index=True)
    entered = asyncio.Event()
    leave = asyncio.Event()

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async def open_session() -> None:
        async with create_connected_server_and_client_session(
            server, list_roots_callback=list_roots
        ) as client:
            await client.list_tools()
            assert await asyncio.to_thread(embedder.started.wait, 5)
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
    server = create_server(app, auto_index=True)
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
            assert await asyncio.to_thread(attempted.wait, 5)

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
    server = create_server(app, auto_index=True)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.list_tools()
        assert await asyncio.to_thread(embedder.started.wait, 5)

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
    server = create_server(app, auto_index=True)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=startup_root.as_uri())])

    try:
        async with create_connected_server_and_client_session(
            server, list_roots_callback=list_roots
        ) as client:
            await client.list_tools()
            assert await asyncio.to_thread(embedder.started.wait, 5)

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
    server = create_server(app, auto_index=True)

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
    server = create_server(app, auto_index=True)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(
            roots=[types.Root(uri=root_a.as_uri()), types.Root(uri=root_b.as_uri())]
        )

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.list_tools()
        assert await asyncio.to_thread(embedder.started.wait, 5)

        # Discovery for both roots should complete quickly even though one of
        # them is now stuck indexing (blocked on the embedder) - discovery no
        # longer shares the capacity limiter with indexing. Wait for the store
        # registration too: the marker file is written before the project is
        # upserted, and project_status below needs the registered project.
        for _ in range(100):
            if (
                (root_a / ".ci-mcp" / "project.toml").exists()
                and (root_b / ".ci-mcp" / "project.toml").exists()
                and len(app.list_projects()) == 2
            ):
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


@pytest.mark.asyncio
async def test_first_query_reports_progress_while_the_initial_index_runs(
    tmp_path: Path,
) -> None:
    """A blocking lazy index must look like work in progress, not a hung tool."""
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
    progress: list[str | None] = []

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async def on_progress(_: float, __: float | None, message: str | None) -> None:
        progress.append(message)

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        query = asyncio.create_task(
            client.call_tool("search_code", {"query": "answer"}, progress_callback=on_progress)
        )
        assert await asyncio.to_thread(embedder.started.wait, 5)
        embedder.release.set()
        result = await query

    assert not result.isError
    assert "Building the initial index" in progress


@pytest.mark.asyncio
async def test_first_query_fails_when_a_competing_index_holds_the_lock_past_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A held global lock must surface INDEX_BUSY at the deadline, not hang."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    (root / "main.py").write_text("def answer():\n    return 42\n")
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    monkeypatch.setenv("INCODE_INDEX_WAIT_SECONDS", "1")
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    lock_directory = paths.data / "locks"
    lock_directory.mkdir(parents=True, exist_ok=True)
    # A separate FileLock instance contends for the same file, which is exactly
    # what a second process (or a second MCP client) does.
    competing = FileLock(lock_directory / "index-global.lock")
    with competing:
        async with create_connected_server_and_client_session(
            server, list_roots_callback=list_roots
        ) as client:
            result = await asyncio.wait_for(
                client.call_tool("search_code", {"query": "answer"}), timeout=20
            )

    assert result.isError
    message = "".join(
        block.text for block in result.content if isinstance(block, types.TextContent)
    )
    assert ErrorCode.INDEX_BUSY.value in message


@pytest.mark.asyncio
async def test_a_second_root_gives_up_instead_of_queueing_behind_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wait budget covers queueing inside this process, not only the file lock.

    One capacity limiter serializes roots in a session, so a second root waits
    even though no other process holds the global lock. To a first query the two
    are indistinguishable, so both count against INCODE_INDEX_WAIT_SECONDS.
    """
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
    monkeypatch.setenv("INCODE_INDEX_WAIT_SECONDS", "0")
    server = create_server(app, auto_index=True)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(
            roots=[types.Root(uri=root_a.as_uri()), types.Root(uri=root_b.as_uri())]
        )

    try:
        async with create_connected_server_and_client_session(
            server, list_roots_callback=list_roots
        ) as client:
            await client.list_tools()
            assert await asyncio.to_thread(embedder.started.wait, 5)

            # Discovery writes the on-disk marker before it registers the
            # project, so a root can resolve to an id the store does not know
            # yet. Wait for both registrations before reading any state.
            for _ in range(100):
                if len(app.list_projects()) == 2:
                    break
                await asyncio.sleep(0.02)
            else:
                pytest.fail("expected both roots to be registered")

            # Exactly one root won the limiter and is stuck on the embedder;
            # target the other, which is the one that had to wait.
            blocked = root_a if app.project_status(roots=[root_a]).state == "indexing" else root_b
            waiting = root_b if blocked is root_a else root_a
            waiting_id = app.project_status(roots=[waiting]).project.id

            result = await asyncio.wait_for(
                client.call_tool("search_code", {"query": "value", "projects": [waiting_id]}),
                timeout=10,
            )
    finally:
        embedder.release.set()

    assert result.isError
    message = "".join(
        block.text for block in result.content if isinstance(block, types.TextContent)
    )
    assert ErrorCode.INDEX_BUSY.value in message


@pytest.mark.asyncio
async def test_first_query_succeeds_when_the_index_lock_frees_before_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    (root / "main.py").write_text("def answer():\n    return 42\n")
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    monkeypatch.setenv("INCODE_INDEX_WAIT_SECONDS", "60")
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    lock_directory = paths.data / "locks"
    lock_directory.mkdir(parents=True, exist_ok=True)
    competing = FileLock(lock_directory / "index-global.lock")

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        competing.acquire()
        try:
            query = asyncio.create_task(client.call_tool("search_code", {"query": "answer"}))
            await asyncio.sleep(0.3)
            assert not query.done()
        finally:
            competing.release()
        result = await asyncio.wait_for(query, timeout=20)

    assert not result.isError
    assert app.project_status(roots=[root]).state == "ready"


@pytest.mark.asyncio
async def test_a_ready_project_does_not_report_indexing_progress(tmp_path: Path) -> None:
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
    progress: list[str | None] = []

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async def on_progress(_: float, __: float | None, message: str | None) -> None:
        progress.append(message)

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.call_tool("search_code", {"query": "answer"})
        progress.clear()
        result = await client.call_tool(
            "search_code", {"query": "answer"}, progress_callback=on_progress
        )

    assert not result.isError
    assert progress == []


def test_server_instructions_guide_index_first_usage(tmp_path: Path) -> None:
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    server = create_server(app)

    instructions = server.instructions

    assert instructions is not None
    for tool in (
        "search_code",
        "find_symbol",
        "file_outline",
        "get_chunk",
        "project_status",
        "index_project",
    ):
        assert tool in instructions


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


READ_ONLY_TOOLS = frozenset({"list_projects", "get_chunk"})
# These answer read queries but first go through _startup_roots, which registers
# an unknown root as a project — writing its marker — before serving the call.
AUTO_REGISTERING_TOOLS = frozenset({"project_status", "search_code", "find_symbol", "file_outline"})
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

    assert {tool.name for tool in tools} == READ_ONLY_TOOLS | AUTO_REGISTERING_TOOLS | WRITE_TOOLS
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
    # Registering a root is a write, so these cannot be auto-approved as reads.
    for name in AUTO_REGISTERING_TOOLS:
        assert annotations[name] is not None and annotations[name].readOnlyHint is False, name
        assert annotations[name].destructiveHint is False, name
        assert annotations[name].idempotentHint is True, name
    for name in WRITE_TOOLS:
        assert annotations[name] is not None and annotations[name].readOnlyHint is False, name
    # force_new_id lets init_project overwrite a marker and orphan the old
    # registration, so the tool as a whole cannot claim additive/idempotent use.
    assert annotations["remove_project"].destructiveHint is True
    assert annotations["remove_project"].idempotentHint is True
    assert annotations["index_project"].destructiveHint is False
    assert annotations["index_project"].idempotentHint is True
    assert annotations["init_project"].destructiveHint is True
    assert annotations["init_project"].idempotentHint is False


@pytest.mark.asyncio
async def test_project_status_registers_an_unmarked_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = _tiny_application(tmp_path)
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        assert not (root / ".ci-mcp").exists()

        result = await client.call_tool("project_status", {})

    assert not result.isError
    # The marker and the registration are both new: project_status wrote to the
    # repository and to the store, which is why it cannot claim readOnlyHint.
    assert (root / ".ci-mcp" / "project.toml").is_file()
    assert [project.root for project in app.list_projects()] == [root.resolve()]


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


@pytest.mark.asyncio
async def test_tool_error_carries_code_and_details(tmp_path: Path) -> None:
    server = create_server(_tiny_application(tmp_path), auto_index=False)

    with pytest.raises(ToolError) as caught:
        await server.call_tool("get_chunk", {"chunk_id": "missing"})

    message = str(caught.value)
    assert "CHUNK_NOT_FOUND" in message
    assert "chunk_id=missing" in message
    assert "search_code or find_symbol" in message
