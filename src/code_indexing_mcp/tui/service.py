"""Runtime service boundary for the terminal TUI.

Decouples Textual and UI state from the underlying ApplicationLike backend,
handling project discovery, readiness checks, search, chunk loading, outline,
references, and impact analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..application import ApplicationLike
from ..errors import CodeIndexingError, ErrorCode
from ..models import (
    CodeChunk,
    DeclarationSelector,
    ImpactRadiusResponse,
    IndexProgress,
    IndexReport,
    OutlineResponse,
    ProjectInfo,
    ProjectStatus,
    ReferenceResponse,
    SearchHit,
    SearchResponse,
    SymbolResponse,
)
from ..settings import IndexMode

if TYPE_CHECKING:
    from ..application import RuntimePaths
    from ..settings import IndexSettings

logger = logging.getLogger(__name__)


class TuiService:
    """Service layer backing the Terminal UI.

    Wraps an ``ApplicationLike`` backend (in-process ``Application`` or
    ``BrokerApplication`` fronting the daemon) and manages the active project
    scope, freshness checks, queries, and navigation.
    """

    def __init__(
        self,
        application: ApplicationLike,
        cwd: Path | None = None,
        roots: list[Path] | None = None,
        index_mode: IndexMode | str = IndexMode.LAZY,
    ) -> None:
        self.application = application
        self.cwd = (cwd or Path.cwd()).resolve()
        self.roots = roots if roots is not None else [self.cwd]
        self.index_mode = IndexMode(index_mode) if isinstance(index_mode, str) else index_mode
        self._selected_project: ProjectInfo | None = None

    @property
    def selected_project(self) -> ProjectInfo | None:
        return self._selected_project

    def discover_current_project(self) -> ProjectInfo | None:
        """Discover the project for the current working directory, or select from registered."""
        discovered = self.application.discover_project(self.cwd)
        if discovered is not None:
            self._selected_project = discovered
            return discovered

        projects = self.list_projects()
        if not projects:
            self._selected_project = None
            return None

        # Prefer registered project matching or containing cwd
        matching = [p for p in projects if self.cwd == p.root or self.cwd.is_relative_to(p.root)]
        if matching:
            matching.sort(key=lambda p: len(p.root.parts), reverse=True)
            self._selected_project = matching[0]
            return matching[0]

        # Fallback to the first registered project
        self._selected_project = projects[0]
        return projects[0]

    def list_projects(self) -> list[ProjectInfo]:
        """List all registered projects."""
        return self.application.list_projects()

    def select_project(self, project: ProjectInfo | str) -> ProjectInfo:
        """Set the active project by ProjectInfo or identifier/name."""
        if isinstance(project, ProjectInfo):
            self._selected_project = project
            return project
        resolved = self.application.resolve_project(project, roots=self.roots)
        self._selected_project = resolved
        return resolved

    def _require_project(self, project: ProjectInfo | str | None = None) -> ProjectInfo:
        if project is None:
            if self._selected_project is None:
                discovered = self.discover_current_project()
                if discovered is None:
                    raise CodeIndexingError(
                        ErrorCode.PROJECT_NOT_FOUND, "No project registered or discovered"
                    )
                return discovered
            return self._selected_project
        if isinstance(project, str):
            return self.application.resolve_project(project, roots=self.roots)
        return project

    def project_status(self, project: ProjectInfo | str | None = None) -> ProjectStatus:
        """Get index status for the selected or given project."""
        target = self._require_project(project)
        return self.application.project_status(target.id, roots=[target.root])

    def ensure_ready(self, project: ProjectInfo | None = None) -> ProjectStatus:
        """Ensure project index is ready before a query.

        In lazy or eager mode, refreshes stale or pending indexes with trigger='lazy-query'.
        In manual mode, does not trigger automatic indexing.
        """
        target = self._require_project(project)
        status = self.project_status(target)
        if status.state in {"ready", "partial"}:
            return status

        if self.index_mode is IndexMode.MANUAL:
            return status

        # Trigger lazy indexing
        self.application.index_project(target.id, roots=[target.root], trigger="lazy-query")
        return self.project_status(target)

    def index_project(
        self, project: ProjectInfo | None = None, *, force: bool = False
    ) -> IndexReport:
        """Explicitly run an index for the project (e.g. F5)."""
        target = self._require_project(project)
        return self.application.index_project(
            target.id, roots=[target.root], force=force, trigger="manual"
        )

    def index_progress(self, project: ProjectInfo | None = None) -> IndexProgress | None:
        """Poll current indexing progress for the project."""
        try:
            target = self._require_project(project)
        except CodeIndexingError:
            return None
        return self.application.index_progress(target.id)

    def search_code(
        self, query: str, *, limit: int = 20, project: ProjectInfo | None = None
    ) -> SearchResponse:
        """Execute semantic search within the active project."""
        target = self._require_project(project)
        status = self.ensure_ready(target)
        if status.state not in {"ready", "partial"} and self.index_mode is IndexMode.MANUAL:
            raise CodeIndexingError(
                ErrorCode.INDEX_INCOMPATIBLE,
                f"Project '{target.name}' is {status.state}; index before searching (press F5)",
            )
        return self.application.search_code(
            query, projects=[target.id], limit=limit, roots=[target.root]
        )

    def find_symbol(
        self,
        name: str,
        *,
        match: str = "exact",
        limit: int = 20,
        project: ProjectInfo | None = None,
    ) -> SymbolResponse:
        """Execute symbol lookup within the active project."""
        target = self._require_project(project)
        status = self.ensure_ready(target)
        if status.state not in {"ready", "partial"} and self.index_mode is IndexMode.MANUAL:
            raise CodeIndexingError(
                ErrorCode.INDEX_INCOMPATIBLE,
                f"Project '{target.name}' is {status.state}; index before searching (press F5)",
            )
        return self.application.find_symbol(
            name, project=target.id, match=match, limit=limit, roots=[target.root]
        )

    def get_chunk(self, chunk_id: str) -> CodeChunk:
        """Retrieve chunk content and metadata by chunk_id."""
        return self.application.get_chunk(chunk_id)

    def file_outline(self, path: str, project: ProjectInfo | None = None) -> OutlineResponse:
        """Retrieve hierarchical symbol outline for a file."""
        target = self._require_project(project)
        return self.application.file_outline(path, project=target.id, roots=[target.root])

    def to_selector(self, hit: SearchHit) -> DeclarationSelector:
        """Construct a DeclarationSelector from a SearchHit."""
        return DeclarationSelector(chunk_id=hit.chunk_id)

    def find_references(
        self,
        hit_or_selector: SearchHit | DeclarationSelector,
        *,
        limit: int = 100,
        project: ProjectInfo | None = None,
    ) -> ReferenceResponse:
        """Find references to a selected hit or selector."""
        selector = (
            self.to_selector(hit_or_selector)
            if isinstance(hit_or_selector, SearchHit)
            else hit_or_selector
        )
        target = self._require_project(project)
        roots = [target.root]
        return self.application.find_references(selector, limit=limit, roots=roots)

    def impact_radius(
        self,
        hit_or_selector: SearchHit | DeclarationSelector,
        *,
        project: ProjectInfo | None = None,
        max_depth: int = 2,
        limit: int = 100,
    ) -> ImpactRadiusResponse:
        """Compute impact radius for a selected hit or selector."""
        selector = (
            self.to_selector(hit_or_selector)
            if isinstance(hit_or_selector, SearchHit)
            else hit_or_selector
        )
        target = self._require_project(project)
        roots = [target.root]
        return self.application.impact_radius(
            selector, max_depth=max_depth, limit=limit, roots=roots
        )


def create_tui_service(
    cwd: Path | None = None,
    roots: list[Path] | None = None,
    settings: IndexSettings | None = None,
    paths: RuntimePaths | None = None,
) -> TuiService:
    """Factory to construct TuiService following the standard daemon policy."""
    from ..application import Application, RuntimePaths
    from ..daemon import ensure_daemon, require_daemon_support
    from ..settings import IndexSettings

    cwd = (cwd or Path.cwd()).resolve()
    settings = settings or IndexSettings.from_environment()
    paths = paths or RuntimePaths.from_environment()

    use_daemon = settings.broker_mode != "off"
    if use_daemon:
        try:
            require_daemon_support()
        except CodeIndexingError:
            if settings.broker_mode == "on":
                raise
            logger.warning(
                "Unix domain sockets are unavailable on this platform; "
                "connecting directly instead of via the shared daemon"
            )
            use_daemon = False

    app: ApplicationLike = (
        ensure_daemon(paths) if use_daemon else Application(paths, cwd=cwd, settings=settings)
    )
    return TuiService(app, cwd=cwd, roots=roots, index_mode=settings.mode)
