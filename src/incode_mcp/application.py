"""Application services shared by MCP and CLI adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_path, user_data_path

from .embedding import Embedder, FastEmbedder
from .errors import ErrorCode, IncodeError
from .extractor import TreeSitterExtractor
from .indexing import Indexer
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
from .projects import ProjectResolver, initialize_project
from .scanner import SourceScanner
from .search import SearchService
from .storage import LanceStore


@dataclass(frozen=True)
class RuntimePaths:
    data: Path
    cache: Path

    @classmethod
    def from_environment(cls) -> RuntimePaths:
        data = Path(os.environ.get("INCODE_DATA_DIR", user_data_path("incode")))
        cache = Path(os.environ.get("INCODE_CACHE_DIR", user_cache_path("incode")))
        return cls(data=data.expanduser().resolve(), cache=cache.expanduser().resolve())


class Application:
    def __init__(
        self,
        paths: RuntimePaths,
        *,
        embedder: Embedder | None = None,
        cwd: Path | None = None,
    ) -> None:
        self.paths = paths
        self.cwd = (cwd or Path.cwd()).resolve()
        if embedder is None:
            offline = os.environ.get("INCODE_OFFLINE", "").lower() in {"1", "true", "yes"}
            embedder = FastEmbedder(paths.cache / "models", offline=offline)
        self.embedder = embedder
        self.store = LanceStore(paths.data / "lancedb", vector_dimension=embedder.dimension)
        self.indexer = Indexer(
            store=self.store,
            scanner=SourceScanner(),
            extractor=TreeSitterExtractor(),
            embedder=embedder,
            lock_directory=paths.data / "locks",
        )
        self.search = SearchService(self.store, embedder)

    @classmethod
    def from_environment(cls, *, cwd: Path | None = None) -> Application:
        return cls(RuntimePaths.from_environment(), cwd=cwd)

    def init_project(
        self,
        path: Path | str | None = None,
        name: str | None = None,
        force_new_id: bool = False,
        *,
        roots: list[Path] | None = None,
    ) -> ProjectInfo:
        if path is None and roots:
            if len(roots) > 1:
                raise IncodeError(
                    ErrorCode.AMBIGUOUS_PROJECT,
                    "Multiple MCP roots are available; provide an explicit path",
                )
            path = roots[0]
        project = initialize_project(
            Path(path) if path is not None else self.cwd,
            name=name,
            force_new_id=force_new_id,
        )
        self.store.upsert_project(project, model_id=self.embedder.model_id)
        return project

    def index_project(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        force: bool = False,
    ) -> IndexReport:
        return self.indexer.index(self._resolve(project, roots), force=force)

    def project_status(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> ProjectStatus:
        resolved = self._resolve(project, roots)
        return ProjectStatus(
            project=resolved,
            state=self.store.project_state(resolved.id),
            file_count=len(self.store.list_files(resolved.id)),
            chunk_count=len(self.store.list_chunks([resolved.id])),
        )

    def list_projects(self) -> list[ProjectInfo]:
        return sorted(self.store.list_projects(), key=lambda project: (project.name, project.id))

    def remove_project(self, project: str) -> RemovalReport:
        resolved = self._resolve(project, [])
        return RemovalReport(
            project_id=resolved.id,
            removed=self.store.remove_project(resolved.id),
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
        project_ids = self._search_scope(projects, all_projects, roots)
        return self.search.search_code(
            query,
            project_ids,
            languages=languages,
            paths=paths,
            kinds=kinds,
            limit=limit,
        )

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
        resolved = self._resolve(project, roots)
        return self.search.find_symbol(name, resolved.id, match=match, kinds=kinds, limit=limit)

    def file_outline(
        self, path: str, project: str | None = None, *, roots: list[Path] | None = None
    ) -> OutlineResponse:
        resolved = self._resolve(project, roots)
        return self.search.file_outline(path, resolved.id)

    def get_chunk(self, chunk_id: str) -> CodeChunk:
        return self.search.get_chunk(chunk_id)

    def prepare_model(self) -> None:
        if not isinstance(self.embedder, FastEmbedder):
            return
        self.embedder.prepare()

    def _resolve(self, explicit: str | None, roots: list[Path] | None) -> ProjectInfo:
        return ProjectResolver(self.store.list_projects()).resolve(
            explicit=explicit,
            roots=roots or [],
            cwd=self.cwd,
        )

    def _search_scope(
        self,
        projects: list[str] | None,
        all_projects: bool,
        roots: list[Path] | None,
    ) -> list[str]:
        if projects and all_projects:
            raise IncodeError(
                ErrorCode.INVALID_FILTER,
                "projects and all_projects cannot be used together",
            )
        if all_projects:
            project_ids = [project.id for project in self.list_projects()]
        elif projects:
            project_ids = [self._resolve(project, roots).id for project in projects]
        else:
            project_ids = [self._resolve(None, roots).id]
        if not project_ids:
            raise IncodeError(ErrorCode.PROJECT_NOT_FOUND, "No indexed projects are available")
        return project_ids
