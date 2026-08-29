import asyncio
import json
import os
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import run_git
from filelock import FileLock
from mcp import types
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.memory import create_connected_server_and_client_session

from code_indexing_mcp import server as server_module
from code_indexing_mcp.application import Application, RuntimePaths
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.server import create_server
from code_indexing_mcp.settings import IndexSettings


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
            raise CodeIndexingError(ErrorCode.MODEL_UNAVAILABLE, "embedding backend unavailable")
        return super().embed_passages(texts)


class FailingEmbedder(TinyEmbedder):
    def __init__(self) -> None:
        self.calls = 0

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        raise CodeIndexingError(ErrorCode.MODEL_UNAVAILABLE, "embedding backend unavailable")


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 15.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not met before the timeout")
        await asyncio.sleep(0.02)


def _write_with_later_mtime(path: Path, content: str) -> None:
    """Write with a visibly later timestamp for watchfiles' test polling backend."""

    previous_mtime = path.stat().st_mtime_ns
    path.write_text(content)
    os.utime(path, ns=(path.stat().st_atime_ns, previous_mtime + 2_000_000_000))


def _observe_freshness_check(app: Application, monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    checked = threading.Event()
    original = app.project_is_stale

    def observed(project_name: str | None = None, *, roots: list[Path] | None = None) -> bool:
        result = original(project_name, roots=roots)
        checked.set()
        return result

    monkeypatch.setattr(app, "project_is_stale", observed)
    return checked


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
        "index_history",
        "inspect_scan",
        "index_storage_status",
        "index_storage_maintenance",
        "list_projects",
        "remove_project",
        "search_code",
        "search_across_projects",
        "find_symbol",
        "find_references",
        "analyze_refactor",
        "emit_refactor_patch",
        "file_outline",
        "get_chunk",
    }
    assert len(tools) == 17
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
async def test_lazy_query_refreshes_a_modified_source(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    source = root / "main.py"
    source.write_text("def before_change():\n    return 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.call_tool("find_symbol", {"name": "before_change"})
        source.write_text("def after_change():\n    return 2\n")

        result = await client.call_tool("find_symbol", {"name": "after_change"})

    project = app.list_projects()[0]
    assert not result.isError
    assert app.find_symbol("after_change", project.id).hits
    assert not app.find_symbol("before_change", project.id).hits


@pytest.mark.asyncio
async def test_lazy_query_refreshes_created_and_deleted_sources(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    removed = root / "removed.py"
    removed.write_text("def removed_symbol():\n    return True\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.call_tool("find_symbol", {"name": "removed_symbol"})
        removed.unlink()
        (root / "added.py").write_text("def added_symbol():\n    return True\n")

        result = await client.call_tool("find_symbol", {"name": "added_symbol"})

    project = app.list_projects()[0]
    assert not result.isError
    assert app.find_symbol("added_symbol", project.id).hits
    assert not app.find_symbol("removed_symbol", project.id).hits


@pytest.mark.asyncio
async def test_lazy_query_refreshes_an_explicit_project_outside_the_active_roots(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def before_change():\n    return 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    app.index_project(project.id)
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[])

    source.write_text("def after_change():\n    return 2\n")
    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        result = await client.call_tool(
            "find_symbol", {"name": "after_change", "project": project.id}
        )

    assert not result.isError
    assert app.find_symbol("after_change", project.id).hits
    assert not app.find_symbol("before_change", project.id).hits


@pytest.mark.asyncio
async def test_manual_mode_does_not_refresh_a_changed_source(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def before_change():\n    return 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    app.index_project(project.id)
    server = create_server(app, auto_index=False)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    source.write_text("def after_change():\n    return 2\n")
    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        result = await client.call_tool("find_symbol", {"name": "after_change"})

    assert not result.isError
    assert not app.find_symbol("after_change", project.id).hits
    assert app.find_symbol("before_change", project.id).hits


@pytest.mark.asyncio
async def test_lazy_merged_search_refreshes_every_requested_checkout(tmp_path: Path) -> None:
    """A merged multi-checkout search freshness-checks each checkout's slot.

    Both advertised roots carry markers of one shared registration. A change
    in the canonical checkout must still be refreshed before the next merged
    answer, even though the worktree's marker would otherwise win a
    single-checkout binding.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git("init", "-q", "--initial-branch", "main", str(repo))
    (repo / "main.py").write_text("def main_branch():\n    return 1\n")
    run_git("add", "main.py", cwd=repo)
    run_git(
        "-c",
        "user.email=test@example.test",
        "-c",
        "user.name=Tests",
        "commit",
        "-qm",
        "main",
        cwd=repo,
    )
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(repo)
    app.index_project(project.id)

    worktree = tmp_path / "wt"
    run_git("worktree", "add", "-q", "--detach", str(worktree), cwd=repo)
    (worktree / "wt.py").write_text("def worktree_branch():\n    return 2\n")
    run_git("add", "wt.py", cwd=worktree)
    run_git(
        "-c",
        "user.email=test@example.test",
        "-c",
        "user.name=Tests",
        "commit",
        "-qm",
        "worktree commit",
        cwd=worktree,
    )
    assert app.init_project(worktree).id == project.id
    app.index_project(project.id, roots=[worktree])
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(
            roots=[types.Root(uri=repo.as_uri()), types.Root(uri=worktree.as_uri())]
        )

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        first = await client.call_tool("search_code", {"query": "return"})
        assert not first.isError

        # A commit changes the canonical checkout's HEAD, which no clean-answer
        # cache may paper over: the fingerprint includes the HEAD OID.
        (repo / "main.py").write_text("def after_change():\n    return 3\n")
        run_git("add", "main.py", cwd=repo)
        run_git(
            "-c",
            "user.email=test@example.test",
            "-c",
            "user.name=Tests",
            "commit",
            "-qm",
            "change main",
            cwd=repo,
        )
        second = await client.call_tool("search_code", {"query": "after_change"})
        assert not second.isError

    symbols = {hit.symbol for hit in app.search_code("return", roots=[repo, worktree]).hits}
    # The canonical checkout's slot was refreshed before the merged answer.
    assert "after_change" in symbols
    # The untouched worktree slot keeps serving its own checkout, still at the
    # pre-commit HEAD. (main_branch is deduplicated against the refreshed
    # main.py chunk: same project, path, and line range across two slots.)
    assert "worktree_branch" in symbols


@pytest.mark.asyncio
async def test_search_across_projects_returns_filtered_globally_limited_hits(
    tmp_path: Path,
) -> None:
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    roots = [tmp_path / "alpha", tmp_path / "beta"]
    for root in roots:
        (root / "src").mkdir(parents=True)
    (roots[0] / "src" / "feature.py").write_text(
        "def shared_feature_alpha():\n    return 'alpha'\n"
    )
    (roots[1] / "src" / "feature.ts").write_text(
        "export function sharedFeatureBeta() { return 'beta'; }\n"
    )
    alpha = app.init_project(roots[0], "alpha-service")
    beta = app.init_project(roots[1], "beta-service")
    app.index_project(alpha.id)
    app.index_project(beta.id)
    server = create_server(app, auto_index=False)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        result = await client.call_tool(
            "search_across_projects",
            {
                "query": "shared feature",
                "projects": [alpha.id, str(roots[1])],
                "languages": ["python", "typescript"],
                "paths": ["src/*"],
                "kinds": ["function"],
                "limit": 2,
            },
        )
        limited = await client.call_tool(
            "search_across_projects",
            {
                "query": "shared feature",
                "projects": [alpha.name, beta.id],
                "languages": ["python", "typescript"],
                "paths": ["src/*"],
                "kinds": ["function"],
                "limit": 1,
            },
        )
        python_only = await client.call_tool(
            "search_across_projects",
            {
                "query": "shared feature",
                "projects": [alpha.id, beta.name],
                "languages": ["python"],
            },
        )

    assert not result.isError
    assert result.structuredContent is not None
    hits = result.structuredContent["hits"]
    assert len(hits) == 2
    assert {hit["project_id"] for hit in hits} == {alpha.id, beta.id}
    assert {hit["project_name"] for hit in hits} == {alpha.name, beta.name}
    assert all(hit["path"].startswith("src/") and hit["kind"] == "function" for hit in hits)
    assert limited.structuredContent is not None
    assert len(limited.structuredContent["hits"]) == 1
    assert python_only.structuredContent is not None
    assert {hit["project_id"] for hit in python_only.structuredContent["hits"]} == {alpha.id}


@pytest.mark.asyncio
async def test_search_across_projects_rejects_duplicate_project_aliases(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = _tiny_application(tmp_path)
    project = app.init_project(root, "service")
    server = create_server(app, auto_index=False)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        result = await client.call_tool(
            "search_across_projects",
            {"query": "answer", "projects": [project.id, project.name]},
        )

    assert result.isError
    message = "".join(
        block.text for block in result.content if isinstance(block, types.TextContent)
    )
    assert ErrorCode.INVALID_FILTER.value in message


@pytest.mark.asyncio
async def test_search_across_projects_preserves_missing_selector_error(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    app = _tiny_application(tmp_path)
    project = app.init_project(root, "service")
    server = create_server(app, auto_index=False)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        result = await client.call_tool(
            "search_across_projects",
            {"query": "answer", "projects": [project.id, "missing-project"]},
        )

    assert result.isError
    message = "".join(
        block.text for block in result.content if isinstance(block, types.TextContent)
    )
    assert ErrorCode.PROJECT_NOT_FOUND.value in message


@pytest.mark.asyncio
async def test_search_across_projects_preserves_ambiguous_selector_error(tmp_path: Path) -> None:
    app = _tiny_application(tmp_path)
    roots = [tmp_path / name for name in ("one", "two", "three")]
    for root in roots:
        root.mkdir()
    app.init_project(roots[0], "shared-service")
    app.init_project(roots[1], "shared-service")
    unique = app.init_project(roots[2], "unique-service")
    server = create_server(app, auto_index=False)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        result = await client.call_tool(
            "search_across_projects",
            {"query": "answer", "projects": ["shared-service", unique.id]},
        )

    assert result.isError
    message = "".join(
        block.text for block in result.content if isinstance(block, types.TextContent)
    )
    assert ErrorCode.AMBIGUOUS_PROJECT.value in message


@pytest.mark.asyncio
async def test_init_project_rejects_overlap_unless_allow_overlap_is_set(tmp_path: Path) -> None:
    root = tmp_path / "project"
    nested = root / "src"
    nested.mkdir(parents=True)
    app = _tiny_application(tmp_path)
    app.init_project(root)
    server = create_server(app, auto_index=False)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        rejected = await client.call_tool("init_project", {"path": str(nested)})
        allowed = await client.call_tool(
            "init_project", {"path": str(nested), "allow_overlap": True}
        )

    assert rejected.isError
    message = "".join(
        block.text for block in rejected.content if isinstance(block, types.TextContent)
    )
    assert ErrorCode.OVERLAPPING_PROJECT.value in message
    assert not allowed.isError
    assert allowed.structuredContent is not None
    assert allowed.structuredContent["name"] == "src"


@pytest.mark.asyncio
async def test_lazy_search_across_projects_refreshes_projects_outside_active_roots(
    tmp_path: Path,
) -> None:
    app = _tiny_application(tmp_path)
    projects = []
    sources = []
    for name in ("one", "two"):
        root = tmp_path / name
        root.mkdir()
        source = root / "main.py"
        source.write_text(f"def before_change_{name}():\n    return 1\n")
        project = app.init_project(root, f"service-{name}")
        app.index_project(project.id)
        projects.append(project)
        sources.append(source)
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[])

    for name, source in zip(("one", "two"), sources, strict=True):
        source.write_text(f"def after_change_{name}():\n    return 2\n")
    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        result = await client.call_tool(
            "search_across_projects",
            {"query": "after change", "projects": [project.id for project in projects]},
        )

    assert not result.isError
    for name, project in zip(("one", "two"), projects, strict=True):
        assert app.find_symbol(f"after_change_{name}", project.id).hits
        assert not app.find_symbol(f"before_change_{name}", project.id).hits


@pytest.mark.asyncio
async def test_manual_search_across_projects_does_not_refresh_changed_sources(
    tmp_path: Path,
) -> None:
    app = _tiny_application(tmp_path)
    projects = []
    sources = []
    for name in ("one", "two"):
        root = tmp_path / name
        root.mkdir()
        source = root / "main.py"
        source.write_text(f"def before_change_{name}():\n    return 1\n")
        project = app.init_project(root, f"service-{name}")
        app.index_project(project.id)
        projects.append(project)
        sources.append(source)
    server = create_server(app, auto_index=False)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[])

    for name, source in zip(("one", "two"), sources, strict=True):
        source.write_text(f"def after_change_{name}():\n    return 2\n")
    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        result = await client.call_tool(
            "search_across_projects",
            {"query": "after change", "projects": [project.id for project in projects]},
        )

    assert not result.isError
    for name, project in zip(("one", "two"), projects, strict=True):
        assert not app.find_symbol(f"after_change_{name}", project.id).hits
        assert app.find_symbol(f"before_change_{name}", project.id).hits


@pytest.mark.asyncio
async def test_eager_monitor_refreshes_created_and_deleted_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # macOS FSEvents are unavailable in the test sandbox. watchfiles' polling
    # backend exercises the same producer/consumer path deterministically.
    monkeypatch.setenv("WATCHFILES_FORCE_POLLING", "true")
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    removed = root / "removed.py"
    removed.write_text("def removed_symbol():\n    return True\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    seed_refresh_finished = _observe_freshness_check(app, monkeypatch)
    server = create_server(app, auto_index=True)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.list_tools()
        await client.call_tool("find_symbol", {"name": "removed_symbol"})
        assert await asyncio.to_thread(seed_refresh_finished.wait, 5)
        project = app.list_projects()[0]
        removed.unlink()
        (root / "added.py").write_text("def added_symbol():\n    return True\n")

        # Two full incremental runs on a contended CI runner (git probes,
        # staging, Lance commits) can outlast the default 5-second budget;
        # the periodic reconcile backstop is 30 seconds, so anything past
        # this window is a real stranding bug, not runner jitter.
        await _wait_until(
            lambda: (
                bool(app.find_symbol("added_symbol", project.id).hits)
                and not app.find_symbol("removed_symbol", project.id).hits
            ),
            timeout=20,
        )

        assert app.find_symbol("added_symbol", project.id).hits
        assert not app.find_symbol("removed_symbol", project.id).hits


@pytest.mark.asyncio
async def test_eager_monitor_repeats_when_source_changes_during_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WATCHFILES_FORCE_POLLING", "true")
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    source = root / "main.py"
    source.write_text("def initial_symbol():\n    return 0\n")
    embedder = SwitchableBlockingEmbedder()
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=embedder,
        cwd=tmp_path,
    )
    seed_refresh_finished = _observe_freshness_check(app, monkeypatch)
    server = create_server(app, auto_index=True)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.list_tools()
        await client.call_tool("find_symbol", {"name": "initial_symbol"})
        assert await asyncio.to_thread(seed_refresh_finished.wait, 5)
        project = app.list_projects()[0]
        embedder.block = True
        _write_with_later_mtime(source, "def first_change():\n    return 1\n")
        assert app.project_is_stale(project.id)
        assert await asyncio.to_thread(embedder.started.wait, 5)

        _write_with_later_mtime(source, "def final_change():\n    return 2\n")
        embedder.release.set()
        # See the budget note in the created-and-deleted monitor test: the
        # release unwinds one blocked run and then a second full incremental
        # run must reconcile this write, all on a possibly contended runner.
        await _wait_until(
            lambda: bool(app.find_symbol("final_change", project.id).hits), timeout=20
        )

        assert not app.find_symbol("first_change", project.id).hits
        assert not app.find_symbol("initial_symbol", project.id).hits


@pytest.mark.asyncio
async def test_eager_monitor_retries_after_a_refresh_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A refresh iteration consumes its sentinel before running, so a failing
    # one used to strand the dirty root until the periodic reconcile. The
    # monitor must requeue its own retry and reconcile the change promptly.
    monkeypatch.setenv("WATCHFILES_FORCE_POLLING", "true")
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    source = root / "main.py"
    source.write_text("def initial_symbol():\n    return 0\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    server = create_server(app, auto_index=True)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.list_tools()
        await client.call_tool("find_symbol", {"name": "initial_symbol"})
        project = app.list_projects()[0]
        await _wait_until(lambda: bool(app.find_symbol("initial_symbol", project.id).hits))

        armed = threading.Event()
        failures = {"count": 0}
        original_index = app.index_project

        def failing_index(project_id=None, **kwargs):
            if kwargs.get("trigger") == "watcher" and armed.is_set():
                failures["count"] += 1
                armed.clear()
                raise CodeIndexingError(ErrorCode.MODEL_UNAVAILABLE, "injected refresh failure")
            return original_index(project_id, **kwargs)

        monkeypatch.setattr(app, "index_project", failing_index)
        armed.set()
        _write_with_later_mtime(source, "def after_failure():\n    return 1\n")

        # A failed iteration backs off for WATCH_RETRY_INITIAL_SECONDS before
        # requeuing, and the retry runs two full index passes; on a slow
        # Windows runner that exceeds the short default deadline.
        await _wait_until(
            lambda: bool(app.find_symbol("after_failure", project.id).hits), timeout=30
        )

        assert failures["count"] == 1
        assert not app.find_symbol("initial_symbol", project.id).hits


@pytest.mark.asyncio
async def test_eager_reconciliation_detects_git_exclusion_changes_without_an_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    source = root / "local_only.py"
    source.write_text("def local_symbol():\n    return True\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    seed_refresh_finished = _observe_freshness_check(app, monkeypatch)

    async def silent_watch(*args: object, **kwargs: object):
        del args, kwargs
        await asyncio.Event().wait()
        yield set()

    monkeypatch.setattr(server_module, "awatch", silent_watch)
    monkeypatch.setattr(server_module, "EAGER_RECONCILE_SECONDS", 0.05)
    server = create_server(app, auto_index=True)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.list_tools()
        await client.call_tool("find_symbol", {"name": "local_symbol"})
        assert await asyncio.to_thread(seed_refresh_finished.wait, 5)
        project = app.list_projects()[0]

        (root / ".git" / "info" / "exclude").write_text("local_only.py\n")
        await _wait_until(lambda: not app.find_symbol("local_symbol", project.id).hits)


@pytest.mark.asyncio
async def test_eager_monitor_restarts_after_a_watcher_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    source = root / "main.py"
    source.write_text("def before_failure():\n    return 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    seed_refresh_finished = _observe_freshness_check(app, monkeypatch)
    change = asyncio.Event()
    watch_calls = 0

    async def flaky_watch(*args: object, **kwargs: object):
        nonlocal watch_calls
        del args, kwargs
        watch_calls += 1
        if watch_calls == 1:
            raise RuntimeError("simulated watcher failure")
        await change.wait()
        yield {(2, str(source))}
        await asyncio.Event().wait()

    monkeypatch.setattr(server_module, "awatch", flaky_watch)
    monkeypatch.setattr(server_module, "WATCH_RETRY_INITIAL_SECONDS", 0.01)
    monkeypatch.setattr(server_module, "WATCH_RETRY_MAXIMUM_SECONDS", 0.02)
    server = create_server(app, auto_index=True)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.list_tools()
        await client.call_tool("find_symbol", {"name": "before_failure"})
        assert await asyncio.to_thread(seed_refresh_finished.wait, 5)
        await _wait_until(lambda: watch_calls >= 2)
        project = app.list_projects()[0]

        source.write_text("def after_recovery():\n    return 2\n")
        change.set()
        # The event-driven refresh runs two full index passes; a slow Windows
        # runner needs more than the short default deadline for both.
        await _wait_until(
            lambda: bool(app.find_symbol("after_recovery", project.id).hits), timeout=30
        )

        assert not app.find_symbol("before_failure", project.id).hits


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
    scans = 0
    original_scan = app.indexer.scanner.iter_scan

    def counted_scan(*args, **kwargs):
        nonlocal scans
        scans += 1
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(app.indexer.scanner, "iter_scan", counted_scan)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.list_tools()
        result = await client.call_tool("search_code", {"query": "answer"})

    assert not result.isError
    assert scans == 1


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
            await _wait_until(embedder.started.is_set, timeout=30)
            entered.set()
            await leave.wait()

    session = asyncio.create_task(open_session())
    # Generous bounds: the startup index must reach the embedding phase and
    # then complete after release, and a cold CI runner (Windows especially)
    # can take far longer than a locally-observed 2s to get there.
    await asyncio.wait_for(entered.wait(), timeout=45)
    try:
        leave.set()
        await asyncio.sleep(0.05)
        assert not session.done()
    finally:
        embedder.release.set()
        await asyncio.wait_for(session, timeout=30)

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
    # Generous bound: a cold CI runner can take far longer to reach the
    # attempted-index signal than the locally-observed 2s.
    with lock:
        await asyncio.wait_for(open_session(), timeout=30)

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
            await _wait_until(embedder.started.is_set, timeout=30)

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
async def test_startup_maintenance_defers_to_startup_indexing(tmp_path: Path) -> None:
    """Scheduled maintenance never competes with the initial index build.

    Regression test for the Windows 3.13 CI timeout: an eager server previously
    ran its maintenance pass at startup, racing the blocked startup index for
    the writer lock and optimizing tables that a tight-timeout query was about
    to read.
    """
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
    timestamp_path = app.paths.data / "maintenance.json"

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    try:
        async with create_connected_server_and_client_session(
            server, list_roots_callback=list_roots
        ) as client:
            await client.list_tools()
            # Startup indexing loads the model before the test can release the
            # embedder; a hard wait(t) bound makes a loaded shared runner fail
            # on timing rather than behavior, so poll to a generous deadline.
            started_deadline = time.monotonic() + 30
            while time.monotonic() < started_deadline and not embedder.started.is_set():
                await asyncio.sleep(0.05)
            assert embedder.started.is_set(), "startup indexing never began embedding"
            await asyncio.sleep(0.3)
            assert not timestamp_path.exists()
            embedder.release.set()
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not timestamp_path.exists():
                await asyncio.sleep(0.05)
            assert timestamp_path.exists()
    finally:
        embedder.release.set()


@pytest.mark.asyncio
async def test_lazy_server_runs_startup_maintenance_before_root_scheduling(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    app.index_project(project.id)
    server = create_server(app)
    timestamp_path = app.paths.data / "maintenance.json"

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.list_tools()
        await _wait_until(timestamp_path.exists)

    assert timestamp_path.exists()


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
        # The startup job embeds on a background task; a hard 1s window makes a
        # loaded shared runner fail on timing rather than behavior. Poll to a
        # generous deadline, and match >= 1 so a file split into several
        # candidates (embed bisection retries) cannot skip the check.
        await _wait_until(lambda: embedder.calls >= 1, timeout=30)

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
    monkeypatch.setenv("CODE_INDEXING_AUTO_INDEX", "0")
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
    monkeypatch.setenv("CODE_INDEXING_INDEX_WAIT_SECONDS", "1")
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
    are indistinguishable, so both count against CODE_INDEXING_INDEX_WAIT_SECONDS.
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
    monkeypatch.setenv("CODE_INDEXING_INDEX_WAIT_SECONDS", "0")
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
    monkeypatch.setenv("CODE_INDEXING_INDEX_WAIT_SECONDS", "60")
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
        "search_across_projects",
        "find_symbol",
        "find_references",
        "analyze_refactor",
        "file_outline",
        "get_chunk",
        "project_status",
        "index_project",
    ):
        assert tool in instructions


def test_error_renders_code_message_and_details_for_clients() -> None:
    error = CodeIndexingError(
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
    error = CodeIndexingError(ErrorCode.CHUNK_NOT_FOUND, "Unknown chunk: abc")

    assert error.for_client() == "CHUNK_NOT_FOUND: Unknown chunk: abc"


READ_ONLY_TOOLS = frozenset({"list_projects", "get_chunk"})
# These answer read queries but first go through _startup_roots, which registers
# an unknown root as a project — writing its marker — before serving the call.
AUTO_REGISTERING_TOOLS = frozenset(
    {
        "project_status",
        "index_history",
        "inspect_scan",
        "index_storage_status",
        "search_code",
        "search_across_projects",
        "find_symbol",
        "find_references",
        "analyze_refactor",
        "emit_refactor_patch",
        "file_outline",
    }
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

    assert {tool.name for tool in tools} == (
        READ_ONLY_TOOLS | AUTO_REGISTERING_TOOLS | WRITE_TOOLS | {"index_storage_maintenance"}
    )
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
async def test_project_status_deduplicates_case_insensitive_client_roots(
    tmp_path: Path, case_insensitive_path_alias: Callable[[Path], Path]
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    (root / "main.py").write_text("def answer():\n    return 42\n")
    alias = case_insensitive_path_alias(root)
    app = _tiny_application(tmp_path)
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(
            roots=[types.Root(uri=root.as_uri()), types.Root(uri=alias.as_uri())]
        )

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        result = await client.call_tool("project_status", {})

    assert not result.isError
    assert len(app.list_projects()) == 1


@pytest.mark.asyncio
async def test_every_tool_parameter_is_documented_and_bounded(tmp_path: Path) -> None:
    tools = {
        tool.name: tool
        for tool in await create_server(_tiny_application(tmp_path), auto_index=False).list_tools()
    }

    for name, tool in tools.items():
        for parameter, spec in tool.inputSchema.get("properties", {}).items():
            assert "description" in spec, f"{name}.{parameter} has no description"
        # T4: a top-level parameter can point at a nested model (e.g.
        # `selector: DeclarationSelector`) whose own fields ship undocumented
        # in `$defs` even though the outer parameter itself has a description.
        for def_name, def_schema in tool.inputSchema.get("$defs", {}).items():
            for field, field_spec in def_schema.get("properties", {}).items():
                assert "description" in field_spec, (
                    f"{name}.$defs.{def_name}.{field} has no description"
                )

    limit = tools["search_code"].inputSchema["properties"]["limit"]
    assert (limit["minimum"], limit["maximum"]) == (1, 50)
    cross_project_schema = tools["search_across_projects"].inputSchema
    assert set(cross_project_schema["properties"]) == {
        "query",
        "projects",
        "languages",
        "paths",
        "kinds",
        "limit",
    }
    assert set(cross_project_schema["required"]) == {"query", "projects"}
    assert cross_project_schema["properties"]["projects"]["minItems"] == 2
    cross_project_limit = cross_project_schema["properties"]["limit"]
    assert (cross_project_limit["minimum"], cross_project_limit["maximum"]) == (1, 50)
    assert set(tools["search_code"].inputSchema["properties"]) == {
        "query",
        "projects",
        "all_projects",
        "languages",
        "paths",
        "kinds",
        "limit",
    }
    assert tools["find_symbol"].inputSchema["properties"]["match"]["enum"] == [
        "exact",
        "prefix",
        "contains",
    ]
    analyze_refactor_schema = tools["analyze_refactor"].inputSchema
    assert set(analyze_refactor_schema["properties"]) == {
        "selector",
        "operation",
        "limit",
        "cursor",
    }
    analyze_limit = analyze_refactor_schema["properties"]["limit"]
    assert (analyze_limit["minimum"], analyze_limit["maximum"]) == (1, 500)
    emit_patch_schema = tools["emit_refactor_patch"].inputSchema
    assert set(emit_patch_schema["properties"]) == {
        "selector",
        "operation",
        "context_lines",
    }
    context_lines = emit_patch_schema["properties"]["context_lines"]
    assert (context_lines["minimum"], context_lines["maximum"]) == (0, 50)


@pytest.mark.asyncio
async def test_analyze_refactor_description_credits_signature_change_evidence(
    tmp_path: Path,
) -> None:
    """T7: `evidence` is described only in the rename sense ("aliases that

    identify the target but need no spelling change") in both the tool
    description and README.md, but for `signature_change` the same bucket
    holds compatible call sites that need no argument edit either.
    """
    tools = {
        tool.name: tool
        for tool in await create_server(_tiny_application(tmp_path), auto_index=False).list_tools()
    }
    description = tools["analyze_refactor"].description or ""
    assert "call sites" in description.lower()

    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    anchor = readme.index("`evidence` includes")
    evidence_paragraph = readme[anchor : readme.index("\n\n", anchor)]
    assert "call sites" in evidence_paragraph.lower()


@pytest.mark.asyncio
async def test_emit_refactor_patch_description_states_the_emission_contract(
    tmp_path: Path,
) -> None:
    """The patch tool must read as emission-only and rename-only.

    The description is the contract callers act on: it must say the tool
    never edits source files, that a signature change is refused (with the
    stable code), and that a partial patch can never read as a finished
    rename -- the callers who skip `analyze_refactor`'s review step otherwise
    apply unproven edits on the tool's apparent authority.
    """
    tools = {
        tool.name: tool
        for tool in await create_server(_tiny_application(tmp_path), auto_index=False).list_tools()
    }
    description = tools["emit_refactor_patch"].description or ""

    lowered = description.lower()
    assert "never edits source" in lowered
    assert "git apply" in lowered
    assert "unsupported_operation" in lowered
    assert "unapplied" in lowered and "conflicted" in lowered
    assert "completeness" in lowered


@pytest.mark.asyncio
async def test_search_across_projects_schema_rejects_one_project(tmp_path: Path) -> None:
    server = create_server(_tiny_application(tmp_path), auto_index=False)

    with pytest.raises(ToolError, match="at least 2"):
        await server.call_tool(
            "search_across_projects",
            {"query": "answer", "projects": ["only-one"]},
        )


@pytest.mark.asyncio
async def test_tool_error_carries_code_and_details(tmp_path: Path) -> None:
    server = create_server(_tiny_application(tmp_path), auto_index=False)

    with pytest.raises(ToolError) as caught:
        await server.call_tool("get_chunk", {"chunk_id": "missing"})

    message = str(caught.value)
    assert "CHUNK_NOT_FOUND" in message
    assert "chunk_id=missing" in message
    assert "search_code or find_symbol" in message


@pytest.mark.asyncio
async def test_index_project_reports_file_counts_while_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manual index must show what it is doing, not an empty bar until the JSON lands."""

    class SlowEmbedder(TinyEmbedder):
        def embed_passages(self, texts: list[str]) -> list[list[float]]:
            time.sleep(0.05)
            return super().embed_passages(texts)

    monkeypatch.setattr(server_module, "PROGRESS_POLL_SECONDS", 0.02)
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    for number in range(8):
        (root / f"module_{number}.py").write_text(f"def answer_{number}():\n    return {number}\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=SlowEmbedder(),
        cwd=tmp_path,
        # The scheduled pass holds the registry's index-global lock while it
        # optimises tables, and this test measures one manual run's progress
        # bar - it must not race that background pass for the lock on a busy
        # runner (observed as an INDEX_BUSY error on CI).
        settings=IndexSettings.from_environment({"CODE_INDEXING_AUTO_MAINTENANCE": "0"}),
    )
    server = create_server(app)
    reports: list[tuple[float, float | None, str | None]] = []

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        reports.append((progress, total, message))

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        result = await client.call_tool(
            "index_project", {"project": str(root)}, progress_callback=on_progress
        )

    assert not result.isError
    assert reports[0][2] == "Indexing project"
    # Mid-run visibility, not a specific phase: the scan phase can complete
    # between two polls on a fast runner, so "Scanning for changed files" is
    # not guaranteed to be sampled. Any live snapshot carrying candidate or
    # file counts proves the bar showed what the run was doing.
    assert any(
        "files" in (message or "") or "candidates" in (message or "")
        for _, _, message in reports[1:-1]
    ), reports
    assert [value for value, _, _ in reports] == sorted(value for value, _, _ in reports)
    assert "chunks embedded" in (reports[-1][2] or "")


@pytest.mark.asyncio
async def test_index_storage_status_tool_reports_installation_statistics(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = _tiny_application(tmp_path)
    # Indexed before the tool runs: the tool auto-registers a root but does not
    # index it, and a registered-but-unindexed project correctly reports no
    # partition at all -- which would make every statistic below trivially zero.
    project = app.init_project(root)
    app.index_project(project.id)
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        scoped = await client.call_tool("index_storage_status", {"project": str(root)})
        installation = await client.call_tool("index_storage_status", {})

    assert not scoped.isError
    assert not installation.isError

    # Assert against what the tool actually returned. Re-deriving the numbers
    # from a fresh app.storage_status() call would re-test the application
    # layer and leave the tool free to serialize an empty or malformed body.
    for result in (scoped, installation):
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert payload["schema_version"] == 2
        assert payload["registry"]["name"] == "projects"
        assert payload["registry"]["row_count"] == 1
        assert payload["registry"]["logical_bytes"] > 0
        assert payload["physical_bytes_total"] > 0
        assert [entry["project"]["id"] for entry in payload["projects"]] == [project.id]
        entry = payload["projects"][0]
        assert entry["consistent"] is True
        assert entry["partition_open_failed"] is False
        assert entry["partition_physical_bytes"] > 0
        assert {table["name"] for table in entry["tables"]} == {"files", "chunks", "references"}


@pytest.mark.asyncio
async def test_index_storage_maintenance_tool_defaults_to_dry_run(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = _tiny_application(tmp_path)
    project = app.init_project(root)
    app.index_project(project.id)
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        scoped = await client.call_tool("index_storage_maintenance", {"project": str(root)})
        installation = await client.call_tool("index_storage_maintenance", {})

    assert not scoped.isError
    assert not installation.isError
    for result in (scoped, installation):
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert payload["schema_version"] == 2
        assert payload["dry_run"] is True
        assert payload["trigger"] == "manual"
        assert payload["retention_hours"] == 24
        entry = payload["projects"][0]
        assert entry["status"] == "skipped"
        assert entry["after"] is None
        assert entry["before"]["partition_physical_bytes"] > 0


@pytest.mark.asyncio
async def test_index_storage_maintenance_tool_can_execute_cleanup(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = _tiny_application(tmp_path)
    project = app.init_project(root)
    app.index_project(project.id)
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        result = await client.call_tool(
            "index_storage_maintenance", {"project": str(root), "dry_run": False}
        )

    assert not result.isError
    payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
    assert payload["dry_run"] is False
    entry = payload["projects"][0]
    assert entry["status"] == "ok"
    assert entry["after"] is not None
    assert payload["registry_after"] is not None
    # The project remains fully usable after maintenance.
    status = app.project_status(project.id)
    assert status.state == "ready"


@pytest.mark.asyncio
async def test_index_history_tool_reports_paginated_runs(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = _tiny_application(tmp_path)
    project = app.init_project(root)
    app.index_project(project.id)
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        result = await client.call_tool("index_history", {"project": str(root), "limit": 1})

    assert not result.isError
    payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
    assert payload["schema_version"] == 1
    assert payload["project"]["id"] == project.id
    assert len(payload["runs"]) == 1
    assert payload["runs"][0]["state"] == "completed"
    assert payload["runs"][0]["trigger"] == "manual"
    assert payload["runs"][0]["run_id"]
    assert payload["runs"][0]["chunks_embedded"] >= 1


@pytest.mark.asyncio
async def test_project_status_includes_progress_and_last_run(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = _tiny_application(tmp_path)
    project = app.init_project(root)
    app.index_project(project.id)
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        result = await client.call_tool("project_status", {"project": str(root)})

    assert not result.isError
    payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
    assert payload["last_run"]["state"] == "completed"
    assert payload["last_run"]["trigger"] == "manual"
    assert payload["last_run"]["eligible_files"] == 1
    assert payload["progress"] is None


@pytest.mark.asyncio
async def test_inspect_scan_tool_returns_paginated_filtered_results(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    (root / "main.py").write_text("def answer():\n    return 42\n")
    (root / "notes.md").write_text("not source\n")
    app = _tiny_application(tmp_path)
    project = app.init_project(root)
    app.index_project(project.id)
    server = create_server(app)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        first = await client.call_tool("inspect_scan", {"project": str(root), "limit": 1})
        second = await client.call_tool(
            "inspect_scan",
            {"project": str(root), "limit": 1, "cursor": cursor(first)},
        )
        eligible = await client.call_tool(
            "inspect_scan", {"project": str(root), "outcome": "eligible"}
        )

    assert not first.isError
    assert not second.isError
    assert not eligible.isError

    first_payload = json.loads(first.content[0].text)  # type: ignore[union-attr]
    assert first_payload["schema_version"] == 1
    assert first_payload["project"]["id"] == project.id
    assert len(first_payload["items"]) == 1
    assert first_payload["next_cursor"]

    second_payload = json.loads(second.content[0].text)  # type: ignore[union-attr]
    assert len(second_payload["items"]) == 1
    assert second_payload["items"][0]["path"] != first_payload["items"][0]["path"]

    eligible_payload = json.loads(eligible.content[0].text)  # type: ignore[union-attr]
    assert [item["path"] for item in eligible_payload["items"]] == ["main.py"]
    assert eligible_payload["items"][0]["outcome"] == "eligible"
    assert eligible_payload["items"][0]["language"] == "python"
    assert eligible_payload["items"][0]["size"] is not None


def cursor(result) -> str:
    payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
    assert payload["next_cursor"] is not None
    return payload["next_cursor"]


@pytest.mark.asyncio
async def test_eager_watcher_marks_the_root_dirty_without_a_freshness_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real filesystem event marks the root dirty immediately; the
    event-driven refresh then schedules and reconciles without running a
    freshness walk, and the mark is cleared once it succeeds."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'project'\n")
    source = root / "main.py"
    source.write_text("def before_change():\n    return 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    stale_checks = 0
    original_is_stale = app.project_is_stale

    def counted_is_stale(*args, **kwargs):
        nonlocal stale_checks
        stale_checks += 1
        return original_is_stale(*args, **kwargs)

    monkeypatch.setattr(app, "project_is_stale", counted_is_stale)
    seed_refresh_finished = _observe_freshness_check(app, monkeypatch)
    change = asyncio.Event()

    async def one_shot_watch(*args: object, **kwargs: object):
        del args, kwargs
        await change.wait()
        yield {(2, str(source))}
        await asyncio.Event().wait()

    monkeypatch.setattr(server_module, "awatch", one_shot_watch)
    monkeypatch.setattr(server_module, "EAGER_RECONCILE_SECONDS", 3600.0)
    server = create_server(app, auto_index=True)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=root.as_uri())])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        await client.list_tools()
        await client.call_tool("find_symbol", {"name": "before_change"})
        project = app.list_projects()[0]
        # The seeded startup pass may legitimately run one freshness walk;
        # every check after it belongs to the event-driven refresh.
        assert await asyncio.to_thread(seed_refresh_finished.wait, 5)
        baseline = stale_checks

        _write_with_later_mtime(source, "def after_change():\n    return 2\n")
        change.set()
        await _wait_until(lambda: bool(app.find_symbol("after_change", project.id).hits))

    # The watcher-driven refresh needed no freshness walk: the dirty mark
    # carried the change signal for both scheduling and the query readiness
    # path.
    assert stale_checks == baseline
