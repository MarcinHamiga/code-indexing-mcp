"""Unit tests for the TuiService runtime boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from code_indexing_mcp.application import ApplicationLike
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.models import (
    CodeChunk,
    CompletenessReport,
    DeclarationSelector,
    ImpactLayer,
    ImpactRadiusResponse,
    IndexProgress,
    IndexReport,
    OutlineItem,
    OutlineResponse,
    ProjectInfo,
    ProjectStatus,
    ReferenceHit,
    ReferenceResponse,
    SearchHit,
    SearchResponse,
    SelectedDeclaration,
    SymbolResponse,
)
from code_indexing_mcp.settings import IndexMode, IndexSettings
from code_indexing_mcp.tui.service import TuiService, create_tui_service


def _sample_project(
    project_id: str = "proj-1", name: str = "test-repo", root: Path | None = None
) -> ProjectInfo:
    return ProjectInfo(
        id=project_id,
        name=name,
        root=root or Path("/workspace/test-repo"),
    )


def _sample_hit(
    chunk_id: str = "chk-1",
    project_id: str = "proj-1",
    path: str = "src/main.py",
    symbol: str = "main",
    score: float = 0.95,
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        project_id=project_id,
        project_name="test-repo",
        path=path,
        language="python",
        kind="function",
        symbol=symbol,
        qualified_symbol=symbol,
        start_line=10,
        end_line=25,
        score=score,
        snippet="def main(): pass",
    )


class FakeApplication:
    """In-memory stub implementing the ApplicationLike interface for TUI tests."""

    def __init__(self, projects: list[ProjectInfo] | None = None) -> None:
        self.projects = projects or [_sample_project()]
        self.statuses: dict[str, str] = {p.id: "ready" for p in self.projects}
        self.index_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.symbol_calls: list[dict[str, Any]] = []
        self.references_calls: list[dict[str, Any]] = []
        self.impact_calls: list[dict[str, Any]] = []

    def discover_project(self, root: Path) -> ProjectInfo | None:
        for p in self.projects:
            if p.root == root or root.is_relative_to(p.root):
                return p
        return None

    def list_projects(self) -> list[ProjectInfo]:
        return list(self.projects)

    def resolve_project(self, explicit: str | None, roots: list[Path] | None = None) -> ProjectInfo:
        for p in self.projects:
            if explicit in (p.id, p.name, str(p.root)):
                return p
        raise CodeIndexingError(ErrorCode.PROJECT_NOT_FOUND, f"No project matching {explicit}")

    def project_status(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> ProjectStatus:
        target = self.resolve_project(project)
        state = self.statuses.get(target.id, "ready")
        return ProjectStatus(
            project=target,
            state=state,
            file_count=42,
            chunk_count=100,
        )

    def project_is_stale(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> bool:
        target = self.resolve_project(project)
        return self.statuses.get(target.id) == "stale"

    def index_project(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        force: bool = False,
        wait_for_lock: bool = False,
        trigger: str = "manual",
    ) -> IndexReport:
        target = self.resolve_project(project)
        self.index_calls.append({"project_id": target.id, "force": force, "trigger": trigger})
        self.statuses[target.id] = "ready"
        return IndexReport(
            project_id=target.id,
            indexed_files=5,
            embedded_chunks=10,
            chunks_staged=10,
        )

    def index_progress(self, project_id: str) -> IndexProgress | None:
        return IndexProgress(
            project_id=project_id,
            phase="embedding",
            candidates_seen=42,
            eligible_files=42,
            chunks_extracted=100,
            chunks_embedded=50,
        )

    def search_code(
        self,
        query: str,
        *,
        projects: list[str] | None = None,
        all_projects: bool = False,
        languages: list[str] | None = None,
        paths: list[str] | None = None,
        kinds: list[str] | None = None,
        limit: int = 8,
        roots: list[Path] | None = None,
    ) -> SearchResponse:
        self.search_calls.append({"query": query, "projects": projects, "limit": limit})
        return SearchResponse(query=query, hits=[_sample_hit()])

    def find_symbol(
        self,
        name: str,
        project: str | None = None,
        *,
        match: str = "exact",
        kinds: list[str] | None = None,
        limit: int = 20,
        roots: list[Path] | None = None,
    ) -> SymbolResponse:
        self.symbol_calls.append({"name": name, "project": project, "match": match, "limit": limit})
        return SymbolResponse(name=name, hits=[_sample_hit(symbol=name)])

    def get_chunk(self, chunk_id: str) -> CodeChunk:
        return CodeChunk(
            chunk_id=chunk_id,
            file_id="file-1",
            project_id="proj-1",
            path="src/main.py",
            language="python",
            kind="function",
            symbol="main",
            qualified_symbol="main",
            start_byte=0,
            end_byte=100,
            start_line=1,
            end_line=10,
            content="def main():\n    print('hello')",
            content_hash="abc123hash",
        )

    def file_outline(
        self, path: str, project: str | None = None, *, roots: list[Path] | None = None
    ) -> OutlineResponse:
        return OutlineResponse(
            project_id=project or "proj-1",
            path=path,
            items=[
                OutlineItem(
                    kind="function",
                    symbol="main",
                    qualified_symbol="main",
                    start_line=1,
                    end_line=10,
                )
            ],
        )

    def find_references(
        self,
        selector: DeclarationSelector,
        *,
        kinds: set[str] | None = None,
        limit: int = 100,
        cursor: str | None = None,
        roots: list[Path] | None = None,
    ) -> ReferenceResponse:
        self.references_calls.append({"selector": selector, "limit": limit})
        decl = SelectedDeclaration(
            project_id="proj-1",
            file_id="file-1",
            path="src/main.py",
            language="python",
            symbol="main",
            qualified_symbol="main",
            kind="function",
            start_line=1,
            end_line=10,
        )
        return ReferenceResponse(
            selected=decl,
            hits=[
                ReferenceHit(
                    reference_id="ref-1",
                    project_id="proj-1",
                    path="src/runner.py",
                    language="python",
                    start_line=5,
                    end_line=5,
                    start_byte=20,
                    end_byte=24,
                    kind="call",
                    snippet="main()",
                    resolution="exact",
                    reason_code="direct_call",
                    explanation="Direct symbol call",
                )
            ],
        )

    def impact_radius(
        self,
        selector: DeclarationSelector,
        *,
        max_depth: int = 2,
        include_likely: bool = False,
        kinds: set[str] | None = None,
        max_nodes: int = 500,
        limit: int = 100,
        cursor: str | None = None,
        roots: list[Path] | None = None,
    ) -> ImpactRadiusResponse:
        self.impact_calls.append({"selector": selector, "max_depth": max_depth})
        decl = SelectedDeclaration(
            project_id="proj-1",
            file_id="file-1",
            path="src/main.py",
            language="python",
            symbol="main",
            qualified_symbol="main",
            kind="function",
            start_line=1,
            end_line=10,
        )
        return ImpactRadiusResponse(
            selected=decl,
            layers=[ImpactLayer(depth=1, edges=[], review=[])],
            visited=3,
            completeness=CompletenessReport(),
        )


def _service(app: FakeApplication, **kwargs: Any) -> TuiService:
    return TuiService(cast(ApplicationLike, app), **kwargs)


def test_tui_service_discovery_and_selection() -> None:
    repo_root = Path("/workspace/test-repo")
    project = _sample_project(root=repo_root)
    app = FakeApplication(projects=[project])
    service = _service(app, cwd=repo_root / "src")

    discovered = service.discover_current_project()
    assert discovered is not None
    assert discovered.id == project.id
    assert service.selected_project == project

    # Fallback to list when discover returns None
    app.projects = [_sample_project(project_id="other", root=Path("/other/repo"))]
    service2 = _service(app, cwd=Path("/somewhere/else"))
    fallback = service2.discover_current_project()
    assert fallback is not None
    assert fallback.id == "other"

    # Select explicit project
    service2.select_project("other")
    assert service2.selected_project is not None
    assert service2.selected_project.id == "other"


def test_tui_service_ready_and_partial_skip_reindex() -> None:
    app = FakeApplication()
    service = _service(app, index_mode=IndexMode.LAZY)
    service.discover_current_project()

    status = service.ensure_ready()
    assert status.state == "ready"
    assert len(app.index_calls) == 0

    app.statuses["proj-1"] = "partial"
    status_partial = service.ensure_ready()
    assert status_partial.state == "partial"
    assert len(app.index_calls) == 0


def test_tui_service_lazy_mode_triggers_lazy_query_index() -> None:
    app = FakeApplication()
    app.statuses["proj-1"] = "stale"
    service = _service(app, index_mode=IndexMode.LAZY)
    service.discover_current_project()

    resp = service.search_code("authentication")
    assert len(resp.hits) == 1
    assert len(app.index_calls) == 1
    assert app.index_calls[0]["trigger"] == "lazy-query"
    assert app.index_calls[0]["project_id"] == "proj-1"


def test_tui_service_manual_mode_refuses_auto_index() -> None:
    app = FakeApplication()
    app.statuses["proj-1"] = "stale"
    service = _service(app, index_mode=IndexMode.MANUAL)
    service.discover_current_project()

    status = service.ensure_ready()
    assert status.state == "stale"
    assert len(app.index_calls) == 0

    with pytest.raises(CodeIndexingError) as exc_info:
        service.search_code("authentication")
    assert exc_info.value.code == ErrorCode.INDEX_INCOMPATIBLE
    assert len(app.index_calls) == 0

    with pytest.raises(CodeIndexingError) as exc_info2:
        service.find_symbol("main")
    assert exc_info2.value.code == ErrorCode.INDEX_INCOMPATIBLE


def test_tui_service_explicit_index_and_progress() -> None:
    app = FakeApplication()
    service = _service(app)
    service.discover_current_project()

    report = service.index_project(force=True)
    assert report.indexed_files == 5
    assert len(app.index_calls) == 1
    assert app.index_calls[0]["force"] is True
    assert app.index_calls[0]["trigger"] == "manual"

    prog = service.index_progress()
    assert prog is not None
    assert prog.phase == "embedding"
    assert prog.chunks_embedded == 50


def test_tui_service_symbol_match_forwarding() -> None:
    app = FakeApplication()
    service = _service(app)
    service.discover_current_project()

    resp = service.find_symbol("my_func", match="prefix", limit=15)
    assert len(resp.hits) == 1
    assert app.symbol_calls[0]["match"] == "prefix"
    assert app.symbol_calls[0]["limit"] == 15


def test_tui_service_chunk_and_outline() -> None:
    app = FakeApplication()
    service = _service(app)
    service.discover_current_project()

    chunk = service.get_chunk("chk-test")
    assert chunk.symbol == "main"
    assert "def main" in chunk.content

    outline = service.file_outline("src/main.py")
    assert len(outline.items) == 1
    assert outline.items[0].symbol == "main"


def test_tui_service_references_and_impact_selector_construction() -> None:
    app = FakeApplication()
    service = _service(app)
    service.discover_current_project()

    hit = _sample_hit(chunk_id="chunk-xyz")
    refs = service.find_references(hit)
    assert len(refs.hits) == 1
    assert app.references_calls[0]["selector"].chunk_id == "chunk-xyz"

    impact = service.impact_radius(hit, max_depth=3)
    assert len(impact.layers) == 1
    assert app.impact_calls[0]["selector"].chunk_id == "chunk-xyz"
    assert app.impact_calls[0]["max_depth"] == 3


def test_tui_service_factory_broker_auto_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import code_indexing_mcp.daemon as daemon_mod

    def fake_require() -> None:
        raise CodeIndexingError(ErrorCode.DAEMON_UNAVAILABLE, "Sockets unavailable")

    monkeypatch.setattr(daemon_mod, "require_daemon_support", fake_require)

    settings = IndexSettings.from_environment(
        {
            "CODE_INDEXING_BROKER": "auto",
            "CODE_INDEXING_DATA_DIR": str(tmp_path / "data"),
            "CODE_INDEXING_CACHE_DIR": str(tmp_path / "cache"),
        }
    )
    service = create_tui_service(cwd=tmp_path, settings=settings)
    assert service.index_mode == IndexMode.LAZY


def test_tui_service_factory_broker_on_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import code_indexing_mcp.daemon as daemon_mod

    def fake_require() -> None:
        raise CodeIndexingError(ErrorCode.DAEMON_UNAVAILABLE, "Sockets unavailable")

    monkeypatch.setattr(daemon_mod, "require_daemon_support", fake_require)

    settings = IndexSettings.from_environment(
        {
            "CODE_INDEXING_BROKER": "on",
            "CODE_INDEXING_DATA_DIR": str(tmp_path / "data"),
            "CODE_INDEXING_CACHE_DIR": str(tmp_path / "cache"),
        }
    )
    with pytest.raises(CodeIndexingError) as exc_info:
        create_tui_service(cwd=tmp_path, settings=settings)
    assert exc_info.value.code == ErrorCode.DAEMON_UNAVAILABLE


def test_source_preview_reads_bounded_working_tree_context(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("\n".join(f"line {i}" for i in range(1, 501)))
    app = FakeApplication(projects=[_sample_project(root=root)])
    service = TuiService(cast(ApplicationLike, app), cwd=root)
    project = service.discover_current_project()
    preview = service.source_preview("main.py", 250, project=project)
    assert preview.start_line == 240
    assert "line 250" in preview.content
    assert len(preview.content.splitlines()) <= 41
    with pytest.raises(CodeIndexingError):
        service.source_preview("../outside.py", 1, project=project)
    (root / "link.py").symlink_to(tmp_path / "outside.py")
    with pytest.raises(CodeIndexingError):
        service.source_preview("link.py", 1, project=project)
