"""Application services shared by MCP and CLI adapters."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from filelock import FileLock
from platformdirs import user_cache_path, user_data_path

from .embedding import Embedder, FastEmbedder
from .embedding_worker import EmbeddingWorkerSession, WorkerConfig
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
    ScanConfig,
    SearchResponse,
    SymbolResponse,
)
from .projects import ProjectResolver, find_project_root, initialize_project, read_project_marker
from .scanner import SourceScanner
from .search import SearchService
from .settings import IndexSettings
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


PROJECT_SHAPE_MARKERS = {
    ".git",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "tsconfig.json",
    "jsconfig.json",
}


class Application:
    def __init__(
        self,
        paths: RuntimePaths,
        *,
        embedder: Embedder | None = None,
        cwd: Path | None = None,
        settings: IndexSettings | None = None,
    ) -> None:
        self.paths = paths
        self.cwd = (cwd or Path.cwd()).resolve()
        self.settings = settings or IndexSettings.from_environment()
        if embedder is None:
            offline = os.environ.get("INCODE_OFFLINE", "").lower() in {"1", "true", "yes"}
            embedder = FastEmbedder(
                paths.cache / "models",
                offline=offline,
                threads=self.settings.embedding_threads,
                enable_cpu_mem_arena=self.settings.embedding_cpu_arena,
            )
        self.embedder = embedder
        self.store = LanceStore(
            paths.data / "lancedb",
            vector_dimension=embedder.dimension,
            vector_index=self.settings.vector_index,
        )
        passage_session_factory: Callable[[], EmbeddingWorkerSession] | None = None
        if isinstance(embedder, FastEmbedder) and self.settings.index_execution == "worker":
            worker_config = WorkerConfig(
                cache_directory=str(embedder.cache_directory),
                offline=embedder.offline,
                threads=self.settings.embedding_threads,
                enable_cpu_mem_arena=self.settings.embedding_cpu_arena,
                dimension=embedder.dimension,
                model_id=embedder.model_id,
            )
            ceiling_bytes = self.settings.index_memory_bytes

            def new_worker_session() -> EmbeddingWorkerSession:
                return EmbeddingWorkerSession(worker_config, configured_ceiling_bytes=ceiling_bytes)

            passage_session_factory = new_worker_session
        self.indexer = Indexer(
            store=self.store,
            scanner=SourceScanner(),
            extractor=TreeSitterExtractor(),
            embedder=embedder,
            lock_directory=paths.data / "locks",
            batch_size=self.settings.embedding_batch_size,
            passage_session_factory=passage_session_factory,
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
        root = Path(path) if path is not None else self.cwd
        # The daemon serves every client on its own thread, so N clients calling
        # this for one root would otherwise all miss the marker and register N
        # ids for the same project. Marker creation and registration are one
        # critical section, keyed the same way as discovery.
        with self._root_lock(root):
            project = initialize_project(root, name=name, force_new_id=force_new_id)
            self._register_project(project)
        return project

    def discover_project(self, root: Path) -> ProjectInfo | None:
        """Find an initialized project or initialize a qualifying client root."""
        root = root.expanduser().resolve()
        if not root.is_dir():
            return None
        with self._root_lock(root):
            marker_root = find_project_root(root)
            if marker_root is not None:
                project = read_project_marker(marker_root)
            else:
                if not self._is_project_shaped(root):
                    return None
                project = initialize_project(root)
            self._register_project(project)
            return project

    def index_project(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        force: bool = False,
        wait_for_lock: bool = False,
    ) -> IndexReport:
        return self.indexer.index(
            self._resolve(project, roots), force=force, wait_for_lock=wait_for_lock
        )

    def project_status(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> ProjectStatus:
        resolved = self._resolve(project, roots)
        return ProjectStatus(
            project=resolved,
            state=self.store.project_state(resolved.id),
            file_count=len(self.store.list_files(resolved.id)),
            chunk_count=self.store.count_chunks([resolved.id]),
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
        project_ids = self.resolve_search_scope(projects, all_projects, roots)
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
        resolved = self.resolve_project(project, roots)
        return self.search.find_symbol(name, resolved.id, match=match, kinds=kinds, limit=limit)

    def file_outline(
        self, path: str, project: str | None = None, *, roots: list[Path] | None = None
    ) -> OutlineResponse:
        resolved = self.resolve_project(project, roots)
        return self.search.file_outline(path, resolved.id)

    def get_chunk(self, chunk_id: str) -> CodeChunk:
        return self.search.get_chunk(chunk_id)

    def prepare_model(self) -> None:
        if not isinstance(self.embedder, FastEmbedder):
            return
        self.embedder.prepare()

    def _root_lock(self, root: Path) -> FileLock:
        """Return the cross-thread, cross-process lock guarding *root*'s marker."""
        directory = self.paths.data / "locks"
        directory.mkdir(parents=True, exist_ok=True)
        digest = sha256(str(root.expanduser().resolve()).encode()).hexdigest()
        return FileLock(directory / f"discover-{digest}.lock")

    def _register_project(self, project: ProjectInfo) -> None:
        """Persist *project*, upserting as pending if new or revalidating if known.

        A brand-new project starts in the "pending" state. An already-known
        project keeps its current state (e.g. "ready" is not reset back to
        "pending"), but the upsert still runs so LanceStore.upsert_project can
        apply its compatibility checks (incompatible embedding model/schema,
        or a project id already active at another root).
        """
        known = {existing.id for existing in self.store.list_projects()}
        state = self.store.project_state(project.id) if project.id in known else "pending"
        self.store.upsert_project(project, model_id=self.embedder.model_id, state=state)

    def _is_project_shaped(self, root: Path) -> bool:
        return any(
            (root / marker).exists() for marker in PROJECT_SHAPE_MARKERS
        ) and self.indexer.scanner.has_supported_source(root, ScanConfig())

    def resolve_project(self, explicit: str | None, roots: list[Path] | None = None) -> ProjectInfo:
        """Resolve one project using the same rules as project-scoped tools."""
        return self._resolve(explicit, roots)

    def resolve_search_scope(
        self,
        projects: list[str] | None,
        all_projects: bool,
        roots: list[Path] | None = None,
    ) -> list[str]:
        """Resolve the project ids a search will use without executing it."""
        return self._search_scope(projects, all_projects, roots)

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
