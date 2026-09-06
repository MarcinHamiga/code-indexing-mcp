"""Host deadlines must not strand indexing or roots-dependent MCP requests."""

import asyncio
from pathlib import Path

import pytest
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session
from test_server import SwitchableBlockingEmbedder, _wait_until

from code_indexing_mcp import server as server_module
from code_indexing_mcp.application import Application, RuntimePaths
from code_indexing_mcp.server import create_server
from code_indexing_mcp.settings import IndexSettings


def _application(tmp_path: Path) -> tuple[Application, SwitchableBlockingEmbedder, str]:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    embedder = SwitchableBlockingEmbedder()
    embedder.block = True
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=embedder,
        cwd=root,
        settings=IndexSettings.from_environment({"CODE_INDEXING_AUTO_MAINTENANCE": "0"}),
    )
    return app, embedder, app.init_project(root).id


def _text(result: types.CallToolResult) -> str:
    return " ".join(block.text for block in result.content if isinstance(block, types.TextContent))


@pytest.mark.asyncio
async def test_unanswered_roots_falls_back_and_later_roots_still_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _, project = _application(tmp_path)
    server = create_server(app, auto_index=False)
    # Patch only the peer response, not the roots resolution under test. The
    # blocked response is cancellation-aware, just like the SDK request wait.
    requested = asyncio.Event()
    cancelled = asyncio.Event()

    async def silent_roots(_session: object) -> types.ListRootsResult:
        requested.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("unreachable")

    async def roots(_: object) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=app.cwd.as_uri())])

    monkeypatch.setattr(server_module, "ROOTS_TIMEOUT_SECONDS", 0.05, raising=False)
    monkeypatch.setattr(server_module.ServerSession, "list_roots", silent_roots)
    async with create_connected_server_and_client_session(
        server, list_roots_callback=roots
    ) as client:
        result = await asyncio.wait_for(client.call_tool("project_status", {"project": project}), 1)
        assert not result.isError
        assert requested.is_set() and cancelled.is_set()
        await client.send_ping()
        monkeypatch.setattr(server_module.ServerSession, "list_roots", roots)
        assert not (await client.call_tool("project_status", {})).isError


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["index_project", "search_code", "find_symbol", "file_outline"])
async def test_slow_index_returns_busy_then_completes_without_restarting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    app, embedder, project = _application(tmp_path)
    server = create_server(app, auto_index=False if tool == "index_project" else None)
    monkeypatch.setattr(server_module, "INDEX_RESPONSE_TIMEOUT_SECONDS", 0.1, raising=False)
    arguments = {
        "index_project": {"project": project},
        "search_code": {"query": "answer", "projects": [project]},
        "find_symbol": {"name": "answer", "project": project},
        "file_outline": {"path": "main.py", "project": project},
    }[tool]
    calls = 0
    original = app.index_project

    def observed(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(app, "index_project", observed)
    async with create_connected_server_and_client_session(server) as client:
        try:
            query = asyncio.create_task(client.call_tool(tool, arguments))
            assert await asyncio.to_thread(embedder.started.wait, 5)
            result = await asyncio.wait_for(query, 1)
            assert result.isError
            assert "INDEX_BUSY" in _text(result)
            assert "project_status" in _text(result)
            assert project in _text(result)
            # A retry while the worker is blocked must not start another build.
            repeated = await asyncio.wait_for(client.call_tool(tool, arguments), 1)
            assert repeated.isError
            assert "INDEX_BUSY" in _text(repeated)
            assert calls == 1
            status = await client.call_tool("project_status", {"project": project})
            assert not status.isError
            await client.send_ping()
        finally:
            embedder.release.set()
        await _wait_until(lambda: app.project_status(project).state == "ready")
        # Give the normal report path enough time even on a busy test runner.
        monkeypatch.setattr(server_module, "INDEX_RESPONSE_TIMEOUT_SECONDS", 5)
        completed = await client.call_tool(tool, arguments)
        assert not completed.isError, _text(completed)
        assert app.project_status(project).chunk_count == 1


@pytest.mark.asyncio
async def test_cancelled_manual_request_keeps_one_owned_index_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, embedder, project = _application(tmp_path)
    server = create_server(app, auto_index=False)
    monkeypatch.setattr(server_module, "INDEX_RESPONSE_TIMEOUT_SECONDS", 0.1, raising=False)
    async with create_connected_server_and_client_session(server) as client:
        try:
            request = asyncio.create_task(client.call_tool("index_project", {"project": project}))
            assert await asyncio.to_thread(embedder.started.wait, 5)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            result = await asyncio.wait_for(
                client.call_tool("index_project", {"project": project}), 1
            )
            assert result.isError and "project_status" in _text(result)
        finally:
            embedder.release.set()
        await _wait_until(lambda: app.project_status(project).state == "ready")


@pytest.mark.asyncio
async def test_forced_request_cannot_join_a_nonforced_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, embedder, project = _application(tmp_path)
    server = create_server(app, auto_index=False)
    monkeypatch.setattr(server_module, "INDEX_RESPONSE_TIMEOUT_SECONDS", 5)
    async with create_connected_server_and_client_session(server) as client:
        try:
            first = asyncio.create_task(client.call_tool("index_project", {"project": project}))
            assert await asyncio.to_thread(embedder.started.wait, 5)
            forced = asyncio.create_task(
                client.call_tool("index_project", {"project": project, "force": True})
            )
            # The conflicting request should reject promptly without waiting
            # for or presenting the non-forced job as a completed forced run.
            result = await asyncio.wait_for(forced, 1)
            assert result.isError and "INDEX_BUSY" in _text(result)
            assert "force" in _text(result)
        finally:
            embedder.release.set()
            await first


@pytest.mark.asyncio
async def test_slow_discovery_has_a_response_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    app, _, project = _application(tmp_path)
    server = create_server(app)
    monkeypatch.setattr(server_module, "INDEX_RESPONSE_TIMEOUT_SECONDS", 0.05)
    release = threading.Event()
    original = app.discover_project

    def discover(root: Path):  # type: ignore[no-untyped-def]
        assert release.wait(5)
        return original(root)

    async def roots(_: object) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=app.cwd.as_uri())])

    monkeypatch.setattr(app, "discover_project", discover)
    async with create_connected_server_and_client_session(
        server, list_roots_callback=roots
    ) as client:
        try:
            result = await asyncio.wait_for(
                client.call_tool("project_status", {"project": project}), 1
            )
            assert result.isError and "INDEX_BUSY" in _text(result)
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
async def test_stdio_peer_that_never_answers_roots_does_not_block_discovery(
    tmp_path: Path, eager: bool
) -> None:
    import json
    import os
    import sys

    env = os.environ | {
        "CODE_INDEXING_DATA_DIR": str(tmp_path / "data"),
        "CODE_INDEXING_CACHE_DIR": str(tmp_path / "cache"),
        "CODE_INDEXING_AUTO_MAINTENANCE": "0",
    }
    # Real newline-delimited MCP traffic; deliberately never answer roots/list.
    script = (
        "from code_indexing_mcp import server as s; "
        "s.ROOTS_TIMEOUT_SECONDS = 0.05; "
        f"s.create_server(auto_index={eager!r}).run()"
    )
    with (tmp_path / "stderr.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=log,
            env=env,
            limit=2**20,
        )
        assert process.stdin is not None and process.stdout is not None
        seen_roots = []

        async def send(message: dict) -> None:
            process.stdin.write((json.dumps({"jsonrpc": "2.0", **message}) + "\n").encode())
            await process.stdin.drain()

        async def request(identifier: str, method: str, params: dict) -> dict:
            await send({"id": identifier, "method": method, "params": params})
            async with asyncio.timeout(10):
                while True:
                    line = await process.stdout.readline()
                    assert line, (tmp_path / "stderr.log").read_text()
                    message = json.loads(line)
                    if message.get("method") == "roots/list":
                        seen_roots.append(message)
                    if message.get("id") == identifier:
                        return message

        try:
            response = await request(
                "init",
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"roots": {"listChanged": True}},
                    "clientInfo": {"name": "silent-roots-smoke", "version": "1"},
                },
            )
            assert "result" in response
            await send({"method": "notifications/initialized"})
            assert len((await request("tools", "tools/list", {}))["result"]["tools"]) == 20
            # Explicitly unknown project should return its real error after
            # roots fallback, not hang behind the peer's absent response.
            status = await request(
                "status",
                "tools/call",
                {
                    "name": "project_status",
                    "arguments": {"project": "missing-project"},
                },
            )
            assert "PROJECT_NOT_FOUND" in json.dumps(status)
            assert seen_roots
            assert "result" in await request("ping", "ping", {})
        finally:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                process.kill()
                await process.wait()


@pytest.mark.asyncio
async def test_background_manual_failure_is_recorded_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from code_indexing_mcp.errors import CodeIndexingError, ErrorCode

    app, embedder, project = _application(tmp_path)
    server = create_server(app, auto_index=False)
    monkeypatch.setattr(server_module, "INDEX_RESPONSE_TIMEOUT_SECONDS", 0.1)
    original = embedder.embed_passages

    def fail_after_release(texts: list[str]) -> list[list[float]]:
        original(texts)
        raise CodeIndexingError(ErrorCode.MODEL_UNAVAILABLE, "test backend failed")

    monkeypatch.setattr(embedder, "embed_passages", fail_after_release)
    async with create_connected_server_and_client_session(server) as client:
        try:
            result = await asyncio.wait_for(
                client.call_tool("index_project", {"project": project}), 2
            )
            assert result.isError and "INDEX_BUSY" in _text(result)
        finally:
            embedder.release.set()
        await _wait_until(lambda: app.project_status(project).state == "error")
        history = await client.call_tool("index_history", {"project": project})
        assert not history.isError and "test backend failed" in _text(history)
        monkeypatch.setattr(embedder, "embed_passages", original)
        monkeypatch.setattr(server_module, "INDEX_RESPONSE_TIMEOUT_SECONDS", 5)
        result = await client.call_tool("index_project", {"project": project})
        assert not result.isError, _text(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_before_response", [False, True])
async def test_daemon_build_survives_stdio_client_exit_and_reconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_before_response: bool
) -> None:
    import json
    import os
    import sys
    import tempfile
    import threading

    from code_indexing_mcp.daemon import BrokerApplication, DaemonServer, daemon_supported

    if not daemon_supported():
        pytest.skip("daemon requires local sockets")
    app, embedder, project = _application(tmp_path)
    env = os.environ | {
        "CODE_INDEXING_DATA_DIR": str(app.paths.data),
        "CODE_INDEXING_CACHE_DIR": str(app.paths.cache),
    }
    script = (
        "from code_indexing_mcp import server as s; "
        "from code_indexing_mcp.daemon import BrokerApplication; "
        "s.INDEX_RESPONSE_TIMEOUT_SECONDS = 0.2; "
        "s.create_server(BrokerApplication.from_environment(), auto_index=False).run()"
    )
    with tempfile.TemporaryDirectory(prefix="cim-") as runtime:
        monkeypatch.setenv("XDG_RUNTIME_DIR", runtime)
        env["XDG_RUNTIME_DIR"] = runtime
        daemon = DaemonServer(app.paths, application=app)
        thread = threading.Thread(target=daemon.serve, daemon=True)
        thread.start()
        assert await asyncio.to_thread(daemon.ready.wait, 5)
        broker = BrokerApplication(app.paths)
        processes = []
        with (tmp_path / "peer-stderr.log").open("w") as log:

            async def open_peer():  # type: ignore[no-untyped-def]
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    script,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=log,
                    env=env,
                    limit=2**20,
                )
                processes.append(process)
                return process

            async def send(process, identifier: str | None, method: str, params: dict):  # type: ignore[no-untyped-def]
                message = {"jsonrpc": "2.0", "method": method, "params": params}
                if identifier is not None:
                    message["id"] = identifier
                process.stdin.write((json.dumps(message) + "\n").encode())
                await process.stdin.drain()

            async def receive(process, identifier: str) -> dict:  # type: ignore[no-untyped-def]
                async with asyncio.timeout(10):
                    while True:
                        line = await process.stdout.readline()
                        assert line, (tmp_path / "peer-stderr.log").read_text()
                        message = json.loads(line)
                        if message.get("id") == identifier:
                            return message

            async def initialize(process):  # type: ignore[no-untyped-def]
                await send(
                    process,
                    "init",
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "reconnect-smoke", "version": "1"},
                    },
                )
                assert "result" in await receive(process, "init")
                await send(process, None, "notifications/initialized", {})

            try:
                first = await open_peer()
                await initialize(first)
                await send(
                    first,
                    "index",
                    "tools/call",
                    {
                        "name": "index_project",
                        "arguments": {"project": project},
                    },
                )
                assert await asyncio.to_thread(embedder.started.wait, 5)
                if cancel_before_response:
                    await send(first, None, "notifications/cancelled", {"requestId": "index"})
                else:
                    response = await receive(first, "index")
                    assert response["result"]["isError"]
                    assert "INDEX_BUSY" in json.dumps(response)
                first.kill()
                await first.wait()

                second = await open_peer()
                await initialize(second)
                await send(
                    second,
                    "status",
                    "tools/call",
                    {
                        "name": "project_status",
                        "arguments": {"project": project},
                    },
                )
                status = await receive(second, "status")
                assert not status["result"].get("isError", False)
                assert status["result"]["structuredContent"]["state"] == "indexing"
                embedder.release.set()
                await _wait_until(lambda: app.project_status(project).state == "ready")
                await send(
                    second,
                    "find",
                    "tools/call",
                    {
                        "name": "find_symbol",
                        "arguments": {"name": "answer", "project": project},
                    },
                )
                result = await receive(second, "find")
                assert not result["result"].get("isError", False)
                assert result["result"]["structuredContent"]["hits"][0]["symbol"] == "answer"
                runs = app.index_history(project).runs
                assert len(runs) == 1 and runs[0].state == "completed"
            finally:
                embedder.release.set()
                for process in processes:
                    if process.returncode is None:
                        process.kill()
                    await process.wait()
                await asyncio.to_thread(broker.stop)
                await asyncio.to_thread(thread.join, 5)
                assert not thread.is_alive()
