"""Application services shared by MCP and CLI adapters."""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol, TypeVar, cast

from filelock import FileLock, Timeout
from platformdirs import user_cache_path, user_data_path

from .accelerator_env import EnvironmentStatus
from .backend_coordinator import BackendCoordinator
from .backends import BackendSelection
from .embedding import Embedder, FastEmbedder, SegmentPlan
from .errors import CodeIndexingError, ErrorCode
from .extractor import TreeSitterExtractor
from .git_state import GitProbeOutcome, GitState, WorktreeStatus, head_snapshot, probe_git_state
from .git_state import slot_id as git_slot_id
from .history import HistoryStore
from .indexing import Indexer
from .maintenance import MaintenanceService
from .models import (
    SCAN_SKIP_REASONS,
    CodeChunk,
    DeclarationSelector,
    HistoryPage,
    ImpactRadiusResponse,
    IndexReport,
    IndexTrigger,
    MaintenanceReport,
    ModelStatus,
    OutlineResponse,
    ProjectInfo,
    ProjectSlot,
    ProjectStatus,
    RefactorAnalysis,
    RefactorOperation,
    RefactorPatch,
    ReferenceBackfillReport,
    ReferenceResponse,
    RemovalReport,
    ScanConfig,
    ScanInspectionItem,
    ScanInspectionPage,
    ScannedFile,
    SearchResponse,
    SkippedFile,
    StorageStatus,
    StoredFile,
    SymbolResponse,
)
from .passage_backend import PassageBackendSession
from .probe_cache import ProbeCache
from .progress import IndexProgress, read_progress
from .projects import (
    ProjectResolver,
    existing_marker_path,
    find_project_root,
    initialize_checkout,
    initialize_project,
    project_root_identity,
    read_project_marker,
    same_project_root,
)
from .reference_service import ReferenceService, validate_patch_request
from .scanner import SourceScanner
from .search import SearchService
from .settings import IndexSettings
from .staging import recover_staged_commits
from .storage import (
    ActiveIndexTarget,
    LanceStore,
    PartitionRef,
    decode_status_paths,
    overlapping_registration,
)
from .token_batching import max_token_product_for

logger = logging.getLogger(__name__)

_Result = TypeVar("_Result")

# Startup recovery needs the global index lock, but that lock is held for the
# whole of an index run. Wait only long enough to lose a race against a commit
# that is about to finish; a run genuinely in flight is left to a later start.
RECOVERY_LOCK_TIMEOUT_SECONDS = 5.0

# Upper bound on how many projects a multi-project scope resolves in parallel.
# Each project probes Git and takes its own per-project file lock, so there is
# no cross-project contention to bound against; this simply caps thread count
# for a scope naming many projects at once.
RESOLVE_TARGET_MAX_WORKERS = 8

# Negative freshness answers are cached briefly so a burst of tool calls in one
# agent interaction does not walk the same clean repository once per call. The
# window is short enough that an external edit surfaces on the next check, and
# watcher events, indexing, and registration invalidate it immediately.
FRESHNESS_CACHE_SECONDS = 5.0

SCAN_INSPECTION_MAX_LIMIT = 200

# Written into `data` and `cache` by `RuntimePaths.ensure_private()`. Its
# presence is what tells the uninstaller "this directory is ours to delete"
# even when neither directory has been populated with anything else yet
# (installer/uninstall.py's `_DATA_MARKERS`).
_PRIVATE_DIRECTORY_SENTINEL = ".code-indexing-mcp"


def _tighten_if_owned(directory: Path) -> None:
    """Chmod *directory* to 0700 when this process owns it and it is looser.

    A pre-existing directory (a user-provided `CODE_INDEXING_DATA_DIR`, or one
    created by an older release before this method existed) keeps whatever
    mode it has today; this only narrows it, and only when doing so is safe.
    Foreign ownership -- a directory on a filesystem without real POSIX
    ownership, or one the current account merely has access to -- is left
    alone rather than refused, so an odd filesystem still works, just without
    the tightening. Never raises: a chmod failure here must not stop the
    server from starting.
    """
    if not hasattr(os, "getuid"):
        return  # Windows: mkdir(mode=) already applied what the platform honours.
    try:
        info = directory.lstat()
        if info.st_uid == os.getuid() and info.st_mode & 0o077:
            os.chmod(directory, 0o700)
    except OSError:
        logger.debug("Could not tighten permissions on %s", directory, exc_info=True)


@dataclass(frozen=True)
class RuntimePaths:
    data: Path
    cache: Path

    @classmethod
    def from_environment(cls) -> RuntimePaths:
        data = Path(os.environ.get("CODE_INDEXING_DATA_DIR", user_data_path("code-indexing-mcp")))
        cache = Path(
            os.environ.get("CODE_INDEXING_CACHE_DIR", user_cache_path("code-indexing-mcp"))
        )
        return cls(data=data.expanduser().resolve(), cache=cache.expanduser().resolve())

    def ensure_private(self) -> None:
        """Create `data` and `cache` as user-private directories.

        Both hold chunk text, embeddings, and (for `data`) the daemon's auth
        token -- an indexed repository can be private even when the account
        running the server is shared, so these must not be left at whatever
        the process umask happens to allow. A directory is created `0700`
        outright; one that already existed looser is tightened in place
        rather than refused, since a first run against an
        already-provisioned `CODE_INDEXING_DATA_DIR` must still work.
        Subdirectories underneath keep their own, default modes: a `0700`
        parent already denies traversal into them from outside this account.
        """
        for directory in (self.data, self.cache):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            _tighten_if_owned(directory)
            sentinel = directory / _PRIVATE_DIRECTORY_SENTINEL
            if not sentinel.exists():
                try:
                    sentinel.write_text("", encoding="utf-8")
                except OSError:
                    logger.debug("Could not write sentinel file in %s", directory, exc_info=True)


PROJECT_SHAPE_MARKERS = {
    ".git",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "tsconfig.json",
    "jsconfig.json",
}


class ApplicationLike(Protocol):
    """D8: the query/project surface `server.py` calls polymorphically on
    whichever backend it was handed -- an in-process ``Application`` or a
    ``BrokerApplication`` fronting the shared daemon.

    Signatures here are ``Application``'s: ``BrokerApplication``'s methods
    either match exactly or forward through ``**params: Any``, which
    structurally accepts any keyword this protocol can throw at it. The two
    exceptions -- ``index_project``'s ``on_progress`` and
    ``maintain_storage``'s ``trigger`` -- are dropped rather than widened
    onto the broker: neither crosses the daemon socket (a callback can't,
    and the daemon's own scheduled-maintenance trigger is not something a
    client sets), so they are not part of what a caller can rely on through
    this shared surface. ``tests/test_daemon.py::test_broker_mirrors_application_surface``
    keeps this list itself honest against both concrete classes.
    """

    def project_is_stale(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> bool: ...

    def discover_project(self, root: Path) -> ProjectInfo | None: ...

    def index_project(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        force: bool = False,
        wait_for_lock: bool = False,
        trigger: IndexTrigger = "manual",
    ) -> IndexReport: ...

    def index_progress(self, project_id: str) -> IndexProgress | None: ...

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
    ) -> SearchResponse: ...

    def init_project(
        self,
        path: Path | str | None = None,
        name: str | None = None,
        force_new_id: bool = False,
        allow_overlap: bool = False,
        *,
        roots: list[Path] | None = None,
    ) -> ProjectInfo: ...

    def resolve_project(
        self, explicit: str | None, roots: list[Path] | None = None
    ) -> ProjectInfo: ...

    def project_status(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> ProjectStatus: ...

    def index_history(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> HistoryPage: ...

    def inspect_scan(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        outcome: str | None = None,
        reason: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> ScanInspectionPage: ...

    def storage_status(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> StorageStatus: ...

    def maintain_storage(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        dry_run: bool = False,
        wait_for_lock: bool = False,
    ) -> MaintenanceReport: ...

    def list_projects(self) -> list[ProjectInfo]: ...

    def remove_project(self, project: str) -> RemovalReport: ...

    def resolve_scope_checkouts(
        self,
        projects: list[str] | None,
        all_projects: bool,
        roots: list[Path] | None = None,
    ) -> list[ProjectInfo]: ...

    def find_symbol(
        self,
        name: str,
        project: str | None = None,
        *,
        match: str = "exact",
        kinds: list[str] | None = None,
        limit: int = 20,
        roots: list[Path] | None = None,
    ) -> SymbolResponse: ...

    def find_references(
        self,
        selector: DeclarationSelector,
        *,
        kinds: set[str] | None = None,
        limit: int = 100,
        cursor: str | None = None,
        roots: list[Path] | None = None,
    ) -> ReferenceResponse: ...

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
    ) -> ImpactRadiusResponse: ...

    def analyze_refactor(
        self,
        selector: DeclarationSelector,
        operation: RefactorOperation,
        *,
        limit: int = 500,
        cursor: str | None = None,
        roots: list[Path] | None = None,
    ) -> RefactorAnalysis: ...

    def emit_refactor_patch(
        self,
        selector: DeclarationSelector,
        operation: RefactorOperation,
        *,
        context_lines: int = 3,
        roots: list[Path] | None = None,
    ) -> RefactorPatch: ...

    def file_outline(
        self, path: str, project: str | None = None, *, roots: list[Path] | None = None
    ) -> OutlineResponse: ...

    def get_chunk(self, chunk_id: str) -> CodeChunk: ...


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
        paths.ensure_private()
        self.cwd = (cwd or Path.cwd()).resolve()
        self.settings = settings or IndexSettings.from_environment()
        if embedder is None:
            embedder = FastEmbedder(
                paths.cache / "models",
                offline=self.settings.offline,
                threads=self.settings.embedding_threads,
                enable_cpu_mem_arena=self.settings.embedding_cpu_arena,
            )
        self.embedder = embedder
        self.store = LanceStore(
            paths.data / "lancedb",
            vector_dimension=embedder.dimension,
            vector_index=self.settings.vector_index,
            vector_storage=self.settings.vector_storage,
            branch_cache_limit=self.settings.branch_cache_limit,
        )
        # Durable audit history for indexing runs. A process that died
        # mid-run left its row in "running"; every new process start is the
        # moment history learns that run never finished.
        self.history = HistoryStore(paths.data / "history")
        self.history.mark_interrupted()
        # Roll back any commit interrupted by a crash before queries are
        # accepted, so a half-written project never becomes searchable. The
        # global index lock keeps recovery from running underneath a commit
        # another process is legitimately writing right now.
        #
        # Waiting that lock out is not an option: it is held for the length of
        # an index run, and every CLI invocation and daemon start builds an
        # Application. If a run is in flight, skip recovery -- that run commits
        # or rolls back on its own, and anything left by an older crash is
        # picked up by the next start that finds the lock free.
        lock_directory = paths.data / "locks"
        lock_directory.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(
                lock_directory / "index-global.lock", timeout=RECOVERY_LOCK_TIMEOUT_SECONDS
            ):
                recover_staged_commits(paths.data / "staging", self.store)
        except Timeout:
            logger.warning(
                "Skipping staged-commit recovery: an index run holds the global lock. "
                "Any commit interrupted earlier is rolled back on a later start."
            )
        # Backend and accelerator selection, with batch-size and crossover
        # calibration, is owned by BackendCoordinator (D1 in the split plan).
        # `self.backends` is the collaborator; the properties and methods
        # below it (`backend_selection`, `effective_backend_selection`,
        # `embedding_batch_size`, `batch_calibration`, `serving_providers`,
        # `accelerator_environment`, `probe_cache`, `model_status`,
        # `crossover_characters`) are thin delegates kept for every public
        # member `server.py`, `cli.py`, `daemon.py`, `benchmark.py`, or the
        # tests reference on `Application` itself.
        self.backends = BackendCoordinator(paths, self.settings, embedder)
        # Negative freshness results, keyed by project id: the monotonic
        # deadline and the scan-config fingerprint the answer was computed for.
        # Read and written from asyncio.to_thread workers and daemon request
        # threads, so every access goes through _freshness_lock.
        self._clean_freshness_until: dict[tuple[str, str], tuple[float, str]] = {}
        self._freshness_lock = threading.Lock()

        passage_session_factory: Callable[[], PassageBackendSession] | None = None
        if isinstance(embedder, FastEmbedder) and self.settings.index_execution == "worker":
            # `segment_plan` is read lazily because `self.indexer` (below)
            # does not exist yet -- see BackendCoordinator._passage_session_factory.
            passage_session_factory = self.backends._passage_session_factory(
                embedder, segment_plan=lambda: self.indexer.segment_plan
            )
        self.indexer = Indexer(
            store=self.store,
            scanner=SourceScanner(),
            extractor=TreeSitterExtractor(),
            embedder=embedder,
            lock_directory=paths.data / "locks",
            batch_size=self.embedding_batch_size,
            segment_plan=SegmentPlan(
                max_tokens=self.settings.embedding_max_tokens,
                overlap_tokens=self.settings.embedding_overlap_tokens,
                max_items=self.embedding_batch_size,
                # The padded matrix a microbatch materializes is charged to the
                # same ceiling as everything else the worker holds, so it is
                # budgeted from that ceiling rather than from the constant it
                # was measured at.
                max_token_product=max_token_product_for(
                    self.settings.index_memory_bytes,
                    max_tokens=self.settings.embedding_max_tokens,
                ),
            ),
            passage_session_factory=passage_session_factory,
            staging_directory=paths.data / "staging",
            progress_directory=paths.data / "progress",
            history=self.history,
        )
        self.search = SearchService(self.store, embedder)
        self.references = ReferenceService(self.store)
        # Storage maintenance is owned by MaintenanceService (D2). It needs
        # target resolution only Application can provide -- which projects are
        # registered, resolving a selector to a ProjectInfo, and resolving the
        # live ActiveIndexTarget(s) behind one -- passed as explicit callables
        # so the dependency is visible rather than handing over `self`.
        self.maintenance = MaintenanceService(
            store=self.store,
            paths=paths,
            settings=self.settings,
            list_projects=self.list_projects,
            resolve_project=self._resolve,
            resolve_active_target=lambda project, lock_held: self._resolve_active_target(
                project, lock_held=lock_held
            ),
            resolve_active_targets=lambda projects, include_status: self._resolve_active_targets(
                projects, include_status=include_status
            ),
            run_repository_stable_query=self._run_repository_stable_query,
        )

    @property
    def backend_selection(self) -> BackendSelection:
        """The backend the current process resolved to from static capability."""
        return self.backends.backend_selection

    @property
    def effective_backend_selection(self) -> BackendSelection:
        """The backend the next run will attempt, after any runtime fallback."""
        return self.backends.effective_backend_selection

    @property
    def embedding_batch_size(self) -> int:
        return self.backends.embedding_batch_size

    @property
    def batch_calibration(self) -> str:
        return self.backends.batch_calibration

    @property
    def serving_providers(self) -> tuple[str, ...]:
        return self.backends.serving_providers

    @property
    def accelerator_environment(self) -> EnvironmentStatus:
        return self.backends.accelerator_environment

    @property
    def probe_cache(self) -> ProbeCache:
        return self.backends.probe_cache

    def crossover_characters(self) -> int | None:
        """Return the run size below which this machine should stay on CPU.

        Delegates to :class:`~.backend_coordinator.BackendCoordinator`; see D1
        in docs/plans/2026-09-02-review-remediation-5-application-split-plan.md.
        """
        return self.backends.crossover_characters()

    def model_status(self) -> ModelStatus:
        """Report the resolved embedding stack without loading or probing it.

        Delegates to :class:`~.backend_coordinator.BackendCoordinator`; see D1
        in docs/plans/2026-09-02-review-remediation-5-application-split-plan.md.
        """
        return self.backends.model_status()

    @classmethod
    def from_environment(cls, *, cwd: Path | None = None) -> Application:
        return cls(RuntimePaths.from_environment(), cwd=cwd)

    def init_project(
        self,
        path: Path | str | None = None,
        name: str | None = None,
        force_new_id: bool = False,
        allow_overlap: bool = False,
        *,
        roots: list[Path] | None = None,
    ) -> ProjectInfo:
        if path is None and roots:
            unique_roots: list[Path] = []
            for root in roots:
                if not any(same_project_root(root, existing) for existing in unique_roots):
                    unique_roots.append(root)
            if len(unique_roots) > 1:
                raise CodeIndexingError(
                    ErrorCode.AMBIGUOUS_PROJECT,
                    "Multiple MCP roots are available; provide an explicit path",
                )
            path = unique_roots[0]
        root = Path(path) if path is not None else self.cwd
        # The daemon serves every client on its own thread, so N clients calling
        # this for one root would otherwise all miss the marker and register N
        # ids for the same project. The root lock keys this the same way as
        # discovery; the registration lock additionally serializes overlapping
        # roots, which the per-root lock cannot: two concurrent init calls for a
        # parent and its nested directory must not both pass the overlap check.
        # The check runs before initialize_project so a rejected registration
        # never writes a marker that discovery would later register anyway.
        with self._root_lock(root), self._registration_lock():
            resolved = root.expanduser().resolve()
            if not allow_overlap and not force_new_id:
                existing = overlapping_registration(self.store.list_projects(), resolved)
                if existing is not None:
                    marker = (
                        read_project_marker(resolved)
                        if existing_marker_path(resolved) is not None
                        else None
                    )
                    # A marker whose id already matches the overlapping project
                    # is a re-initialization of that project, not a new overlap.
                    if marker is None or marker.id != existing.id:
                        raise CodeIndexingError(
                            ErrorCode.OVERLAPPING_PROJECT,
                            f"Project root {resolved} overlaps the registered root "
                            f"{existing.root} of project {existing.id!r}; pass "
                            "allow_overlap=true to register it anyway",
                            existing_project=existing.id,
                            new_project=None if marker is None else marker.id,
                        )
            project = self._initialize_registration(resolved, name=name, force_new_id=force_new_id)
            self._register_project(project)
            self._resolve_active_target(project)
            self.invalidate_freshness(project.id)
        return project

    def _initialize_registration(
        self, root: Path, *, name: str | None, force_new_id: bool
    ) -> ProjectInfo:
        """Create or re-read the marker at *root*, joining shared registrations.

        A checkout of an already-registered repository -- a linked worktree --
        joins that repository's project instead of minting its own id: its
        branches occupy slots inside the existing registration. A leftover
        pre-worktree duplicate registration (its own id on a worktree marker)
        folds into the surviving registration when explicitly initialized.
        ``force_new_id`` deliberately splits away and skips both paths.
        """
        if force_new_id:
            return initialize_project(root, name=name, force_new_id=True)
        marker = read_project_marker(root) if existing_marker_path(root) is not None else None
        shared = self._shared_registration(root)
        if marker is not None:
            if shared is not None and shared.id != marker.id:
                # The marker still names a separate pre-worktree registration;
                # unifying drops that registration together with its slots,
                # which the slot-key upgrade has already invalidated.
                self.store.remove_project(marker.id)
                self.invalidate_freshness(marker.id)
                logger.info(
                    "Unified legacy worktree registration %s into %s (%s)",
                    marker.id,
                    shared.name,
                    shared.id,
                )
                return initialize_checkout(root, shared, name=name)
            return marker
        if shared is not None:
            return initialize_checkout(root, shared, name=name)
        return initialize_project(root, name=name)

    def _shared_registration(self, root: Path) -> ProjectInfo | None:
        """Return the registered project *root*'s checkout already belongs to.

        Sharing requires the same Git repository identity and the same
        project prefix: two toplevel checkouts of one repository are one
        project's checkouts, while a subdirectory registration scopes only
        its own subtree even inside the same repository. Without a usable
        Git identity nothing is ever shared.
        """
        state = probe_git_state(root)
        if state.probe is not GitProbeOutcome.GIT:
            return None
        for project in self.store.list_projects():
            registered = Path(project.root)
            if same_project_root(registered, root):
                continue
            candidate = probe_git_state(registered)
            if (
                candidate.probe is GitProbeOutcome.GIT
                and bool(candidate.repository_identity)
                and candidate.repository_identity == state.repository_identity
                and candidate.project_prefix == state.project_prefix
            ):
                return project
        return None

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
                shared = self._shared_registration(root)
                if shared is not None:
                    project = initialize_checkout(root, shared)
                else:
                    project = initialize_project(root)
            self._register_project(project)
            self._resolve_active_target(project)
            self.invalidate_freshness(project.id)
            return project

    def _resolve_active_target(
        self,
        project: ProjectInfo,
        *,
        include_status: bool = False,
        lock_held: bool = False,
    ) -> ActiveIndexTarget:
        """Resolve one immutable Git and physical-partition operation target."""
        git = probe_git_state(project.root, include_status=include_status)
        lock_directory = self.paths.data / "locks"
        lock_directory.mkdir(parents=True, exist_ok=True)

        def current_target() -> ActiveIndexTarget | None:
            partition = self.store.active_partition_ref(project.id)
            if partition is None or partition.slot_id != git_slot_id(project.id, git):
                return None
            slot = self.store.get_slot(partition.slot_id)
            if slot is None or slot.partition_id != partition.partition_id:
                return None
            self.store.touch_slot(slot.slot_id)
            return ActiveIndexTarget(
                project=project,
                slot=slot,
                partition_id=partition.partition_id,
                activation_epoch=partition.activation_epoch,
                git_state=git,
            )

        def activate() -> ActiveIndexTarget:
            current = current_target()
            if current is not None:
                return current
            try:
                partition = self.store.resolve_partition(project, git)
            except CodeIndexingError as exc:
                if exc.code is not ErrorCode.PROJECT_NOT_FOUND:
                    raise
                self.store.upsert_project(project, model_id=self.embedder.model_id, state="pending")
                partition = self.store.resolve_partition(project, git)
            slot = self.store.get_slot(partition.slot_id)
            if slot is None or slot.partition_id != partition.partition_id:
                raise CodeIndexingError(
                    ErrorCode.PROJECT_NOT_FOUND,
                    f"Project {project.id} has no active index slot",
                )
            return ActiveIndexTarget(
                project=project,
                slot=slot,
                partition_id=partition.partition_id,
                activation_epoch=partition.activation_epoch,
                git_state=git,
            )

        if lock_held:
            return activate()
        current = current_target()
        if current is not None:
            return current
        with FileLock(lock_directory / f"{project.id}.lock"):
            return activate()

    def _resolve_active_targets(
        self, projects: Sequence[ProjectInfo], *, include_status: bool = False
    ) -> dict[str, list[ActiveIndexTarget]]:
        """Resolve per-checkout targets keyed by project id, primary checkout first.

        One shared registration may arrive through several live checkouts
        (its main worktree and linked worktrees); each gets its own target so
        a query can read every requested branch slot together. Sorting keeps
        the logical-project lock order fixed while remaining stable within a
        project, preserving the request's checkout order.

        A scope naming more than one checkout resolves them concurrently:
        each probes Git and takes its own per-project file lock, so there is
        nothing to serialize on, and a multi-project scope otherwise pays the
        full per-project probe cost once per project in sequence.
        ``ThreadPoolExecutor.map`` returns results in submission order, so the
        reassembly below sees exactly the sorted, de-duplicated order the
        sequential loop would have produced.
        """
        ordered: list[ProjectInfo] = []
        seen: set[tuple[str, str]] = set()
        for project in sorted(projects, key=lambda item: item.id):
            key = (project.id, project_root_identity(project.root))
            if key in seen:
                continue
            seen.add(key)
            ordered.append(project)

        def resolve(project: ProjectInfo) -> ActiveIndexTarget:
            return self._resolve_active_target(project, include_status=include_status)

        if len(ordered) > 1:
            with ThreadPoolExecutor(
                max_workers=min(RESOLVE_TARGET_MAX_WORKERS, len(ordered))
            ) as pool:
                resolved = list(pool.map(resolve, ordered))
        else:
            resolved = [resolve(project) for project in ordered]

        grouped: dict[str, list[ActiveIndexTarget]] = {}
        for project, target in zip(ordered, resolved, strict=True):
            grouped.setdefault(project.id, []).append(target)
        return grouped

    @staticmethod
    def _primary_target(
        targets: Mapping[str, Sequence[ActiveIndexTarget]], project_id: str
    ) -> ActiveIndexTarget:
        """Return the primary -- first requested -- checkout's target."""
        checkouts = targets.get(project_id) or ()
        if not checkouts:
            raise CodeIndexingError(
                ErrorCode.PROJECT_NOT_FOUND,
                f"No active index slot was resolved for project {project_id}",
            )
        return checkouts[0]

    @staticmethod
    def _target_changed(target: ActiveIndexTarget) -> bool:
        """Whether the resolved target's repository selector or HEAD moved.

        Reads ``HEAD`` directly (``head_snapshot``) instead of re-running the
        full multi-spawn probe on every checked call; only a read failure it
        cannot make sense of falls back to :func:`probe_git_state`. Selector
        kind, selector value, and head OID are compared exactly as the full
        probe path always has -- see ``head_snapshot`` for why that is
        equivalent to comparing slot identity.
        """
        snapshot = head_snapshot(target.git_state)
        if snapshot is None:
            current = probe_git_state(target.project.root)
            return (
                git_slot_id(target.project.id, current) != target.slot.slot_id
                or current.head_oid != target.git_state.head_oid
            )
        selector_kind, selector_value, head_oid = snapshot
        return (
            selector_kind != target.git_state.selector_kind
            or selector_value != target.git_state.selector_value
            or head_oid != target.git_state.head_oid
        )

    def _run_repository_stable_query(
        self,
        projects: Sequence[ProjectInfo],
        operation: Callable[[Mapping[str, Sequence[ActiveIndexTarget]]], _Result],
        *,
        include_status: bool = False,
    ) -> _Result:
        """Retry one read when any checkout's identity changes during execution."""
        for attempt in range(2):
            targets = self._resolve_active_targets(projects, include_status=include_status)
            error: Exception | None = None
            result: _Result | None = None
            try:
                result = operation(targets)
            except Exception as exc:
                error = exc
            changed = [
                project_id
                for project_id, checkouts in sorted(targets.items())
                if any(self._target_changed(target) for target in checkouts)
            ]
            if not changed:
                if error is not None:
                    raise error
                return cast(_Result, result)
            if attempt == 1:
                raise CodeIndexingError(
                    ErrorCode.REPOSITORY_CHANGED,
                    "Repository selector or HEAD changed repeatedly while serving the request",
                    projects=changed,
                ) from error
        raise AssertionError("repository query retry loop did not return")

    def index_project(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        force: bool = False,
        wait_for_lock: bool = False,
        on_progress: Callable[[IndexProgress], None] | None = None,
        trigger: IndexTrigger = "manual",
    ) -> IndexReport:
        resolved = self._resolve(project, roots)
        target = self._resolve_active_target(resolved)
        try:
            return self.indexer.index(
                resolved,
                target=target,
                force=force,
                wait_for_lock=wait_for_lock,
                on_progress=on_progress,
                trigger=trigger,
            )
        finally:
            # Whether the run changed anything or failed, a cached clean answer
            # must not outlive the state it was computed against.
            self.invalidate_freshness(resolved.id)

    def index_progress(self, project_id: str) -> IndexProgress | None:
        """Return the live progress of whichever process is indexing *project_id*."""

        return read_progress(self.paths.data / "progress", project_id)

    def ensure_reference_index(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        _target: ActiveIndexTarget | None = None,
    ) -> ReferenceBackfillReport:
        """Bring structural rows current without running during semantic searches.

        Reference tools call this boundary before resolution. If a source moved
        between freshness inspection and parse-only backfill, advance its normal
        semantic index first and retry once so files, chunks, and references
        remain one coherent generation.

        Files that still could not be covered are returned rather than raised.
        One unparseable file used to disable both reference tools for the whole
        project on every call, with no way to clear it; the resolver now reports
        those paths as limitations so the rest of the analysis stays usable.
        """

        resolved = self._resolve(project, roots)
        target = _target or self._resolve_active_target(resolved)
        if target.project.id != resolved.id:
            raise ValueError("reference target does not belong to project")
        return self._ensure_reference_target(target)

    def _ensure_reference_target(self, target: ActiveIndexTarget) -> ReferenceBackfillReport:
        resolved = target.project
        if self._project_is_stale(resolved, partition_id=target.partition_id):
            self.indexer.index(
                resolved,
                target=target,
                wait_for_lock=True,
                trigger="lazy-query",
            )
        report = self.indexer.backfill_references(
            resolved,
            target=target,
            wait_for_lock=True,
            trigger="reference-backfill",
        )
        if report.stale_paths:
            self.indexer.index(
                resolved,
                target=target,
                wait_for_lock=True,
                trigger="lazy-query",
            )
            report = self.indexer.backfill_references(
                resolved,
                target=target,
                wait_for_lock=True,
                trigger="reference-backfill",
            )
        self.invalidate_freshness(resolved.id)
        return report

    def project_status(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> ProjectStatus:
        resolved = self._resolve(project, roots)
        return self._run_repository_stable_query(
            [resolved],
            lambda targets: self._project_status_for_target(
                self._primary_target(targets, resolved.id)
            ),
            include_status=True,
        )

    def _slot_is_current(self, target: ActiveIndexTarget) -> bool:
        """Whether a clean slot indexed at the current HEAD needs no source scan.

        The one comparison that makes a branch switch-back free: selector slot,
        indexed HEAD, clean state, scan configuration, model, dimension, and
        schema all matching means the scanner walk itself has nothing to add.
        """
        slot, git = target.slot, target.git_state
        if git.probe is not GitProbeOutcome.GIT or git.worktree is not WorktreeStatus.CLEAN:
            return False
        if slot.indexed_head is None or slot.indexed_head != git.head_oid:
            return False
        if slot.indexed_clean is not True:
            return False
        if slot.scan_config_hash != LanceStore._scan_config_hash(target.project):
            return False
        return (
            self.store.incompatibility_reason(
                target.project.id, self.embedder.model_id, partition_id=target.partition_id
            )
            is None
        )

    def _cached_freshness(self, cache_key: tuple[str, str]) -> tuple[float, str] | None:
        with self._freshness_lock:
            return self._clean_freshness_until.get(cache_key)

    def _set_cached_freshness(self, cache_key: tuple[str, str], fingerprint: str) -> None:
        with self._freshness_lock:
            self._clean_freshness_until[cache_key] = (
                time.monotonic() + FRESHNESS_CACHE_SECONDS,
                fingerprint,
            )

    def _pop_cached_freshness(self, cache_key: tuple[str, str]) -> None:
        with self._freshness_lock:
            self._clean_freshness_until.pop(cache_key, None)

    def _project_status_for_target(self, target: ActiveIndexTarget) -> ProjectStatus:
        resolved = target.project
        partition = target.partition
        slot = target.slot
        files = self.store.list_files(resolved.id, partition_id=target.partition_id)
        state = slot.state
        git = target.git_state
        if state in {"ready", "partial"}:
            fingerprint = (
                f"{partition.slot_id}:{partition.activation_epoch}:{git.head_oid}:"
                f"{resolved.scan.model_dump_json()}"
            )
            # Keyed per slot, not per project: several worktrees share one
            # registration id and must not thrash each other's cached answer.
            cache_key = (resolved.id, partition.slot_id)
            cached = self._cached_freshness(cache_key)
            if cached is not None and cached[1] == fingerprint and cached[0] > time.monotonic():
                # A recent check found this exact slot, activation, HEAD, and
                # scan configuration clean; do not walk the repository again
                # for this call.
                pass
            elif self._slot_is_current(target):
                # A clean slot indexed at exactly this HEAD cannot be stale:
                # the switch-back fast path, with no scanner, parser, or
                # embedder work at all.
                self._set_cached_freshness(cache_key, fingerprint)
            elif (
                git.probe is GitProbeOutcome.GIT
                and slot.indexed_head is not None
                and slot.indexed_head != git.head_oid
            ):
                # The slot was indexed at a different HEAD of the same branch:
                # a commit or reset can hide a same-size, same-mtime content
                # change from any metadata walk, so only an index run that
                # validates the diff can prove the slot current. Lazy and
                # eager modes schedule exactly that run.
                self._pop_cached_freshness(cache_key)
                state = "stale"
            else:
                existing_files = {record.path: record for record in files}
                candidates = self._subset_stale_candidates(slot, git)
                is_stale = (
                    self._paths_are_stale(
                        resolved,
                        existing_files,
                        candidates,
                        partition_id=partition.partition_id,
                    )
                    if candidates is not None
                    else self._project_is_stale(
                        resolved, existing_files, partition_id=partition.partition_id
                    )
                )
                if is_stale:
                    self._pop_cached_freshness(cache_key)
                    state = "stale"
                else:
                    self._set_cached_freshness(cache_key, fingerprint)
        return ProjectStatus(
            project=resolved,
            state=state,
            file_count=len(files),
            chunk_count=self.store.count_chunks(
                [resolved.id], partition_ids={resolved.id: partition.partition_id}
            ),
            progress=self.index_progress(resolved.id),
            last_run=self.history.recent(resolved.id),
            active_slot_id=slot.slot_id,
            git_selector_kind=git.selector_kind.value,
            git_selector_value=git.selector_value,
            git_head=git.head_oid,
            git_probe=git.probe.value,
            git_clean=(
                None
                if git.worktree is WorktreeStatus.UNKNOWN
                else git.worktree is WorktreeStatus.CLEAN
            ),
            branch_build_pending=(
                slot.state != "ready"
                or (
                    git.probe is GitProbeOutcome.GIT
                    and slot.indexed_head is not None
                    and slot.indexed_head != git.head_oid
                )
            ),
            checkout_root=str(resolved.root),
        )

    def invalidate_freshness(self, project_id: str) -> None:
        """Forget a cached clean answer, forcing the next status check to scan.

        Called after anything that changes what the index holds -- registration,
        a completed index or reference backfill, removal -- and by eager-mode
        watchers the moment a file system event lands.
        """
        with self._freshness_lock:
            for key in [k for k in self._clean_freshness_until if k[0] == project_id]:
                self._clean_freshness_until.pop(key, None)

    def index_history(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> HistoryPage:
        """One page of a project's durable indexing history, newest first.

        Deliberately bounded: never more than ``MAX_RUNS_PER_PROJECT`` rows are
        retained, and this reads at most *limit* + 1 of them.
        """

        resolved = self._resolve(project, roots)
        if limit < 1:
            raise CodeIndexingError(ErrorCode.INVALID_FILTER, "history limit must be at least 1")
        try:
            return self.history.list_runs(resolved.id, cursor=cursor, limit=limit, project=resolved)
        except ValueError as exc:
            # Cursors are opaque, user-supplied tokens that legitimately go
            # stale or arrive mangled; that is a structured client error, not
            # a traceback.
            raise CodeIndexingError(ErrorCode.INVALID_CURSOR, "invalid history cursor") from exc

    def storage_status(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> StorageStatus:
        """Read-only storage statistics for one project or the whole installation.

        Delegates to :class:`~.maintenance.MaintenanceService`; see D2 in
        docs/plans/2026-09-02-review-remediation-5-application-split-plan.md.
        """
        return self.maintenance.storage_status(project, roots=roots)

    def maintain_storage(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        dry_run: bool = False,
        wait_for_lock: bool = False,
        trigger: str = "manual",
    ) -> MaintenanceReport:
        """Compact tables and remove verified versions older than the retention window.

        Delegates to :class:`~.maintenance.MaintenanceService`; see D2 in
        docs/plans/2026-09-02-review-remediation-5-application-split-plan.md.
        """
        return self.maintenance.maintain_storage(
            project,
            roots=roots,
            dry_run=dry_run,
            wait_for_lock=wait_for_lock,
            trigger=trigger,
        )

    def maybe_run_maintenance(self) -> MaintenanceReport | None:
        """Run scheduled maintenance when it is due.

        Delegates to :class:`~.maintenance.MaintenanceService`; see D2 in
        docs/plans/2026-09-02-review-remediation-5-application-split-plan.md.
        """
        return self.maintenance.maybe_run_maintenance()

    def project_is_stale(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> bool:
        """Return whether eligible source metadata differs from the live index."""
        resolved = self._resolve(project, roots)
        return self._run_repository_stable_query(
            [resolved],
            lambda targets: self._project_is_stale(
                resolved,
                partition_id=self._primary_target(targets, resolved.id).partition_id,
            ),
        )

    def _project_is_stale(
        self,
        project: ProjectInfo,
        existing: dict[str, StoredFile] | None = None,
        *,
        partition_id: str | None = None,
    ) -> bool:
        if existing is None:
            existing = {
                record.path: record
                for record in self.store.list_files(project.id, partition_id=partition_id)
            }
        current = {
            item.path.as_posix(): item
            for item in self.indexer.scanner.iter_scan(project, existing, read_contents=False)
            if isinstance(item, ScannedFile)
        }
        if current.keys() != existing.keys():
            return True
        return any(
            item.size != existing[path].size
            or item.mtime_ns != existing[path].mtime_ns
            or item.language != existing[path].language
            for path, item in current.items()
        )

    @staticmethod
    def _subset_stale_candidates(slot: ProjectSlot, git: GitState) -> set[str] | None:
        """Return the paths a freshness check needs to stat, or None for a full walk.

        D2 of the query-path-overhead plan: a status fingerprint alone is not
        proof of currency (a file dirty at index time and edited again shares
        the fingerprint of an untouched dirty file), so the rule is: same HEAD
        and same fingerprint needs only today's dirty and untracked paths;
        same HEAD with a different fingerprint also needs the paths that were
        dirty or untracked *when this slot was indexed* -- a file reverted
        back to its indexed content, or one that was freshly cleaned, leaves
        no trace in the current status at all. Anything else (no HEAD match,
        no stored fingerprint, or a path list too large to have been stored)
        cannot be answered from a subset and falls back to the full walk.
        """
        if (
            git.probe is not GitProbeOutcome.GIT
            or slot.indexed_head is None
            or slot.indexed_head != git.head_oid
            or slot.indexed_status_fingerprint is None
            or git.status_fingerprint is None
        ):
            return None
        candidates = set(git.dirty_paths) | set(git.untracked_paths)
        if slot.indexed_status_fingerprint != git.status_fingerprint:
            stored = decode_status_paths(slot.indexed_status_paths)
            if stored is None:
                return None
            candidates |= stored
        return candidates

    def _paths_are_stale(
        self,
        project: ProjectInfo,
        existing: dict[str, StoredFile],
        candidates: Iterable[str],
        *,
        partition_id: str | None = None,
    ) -> bool:
        """Whether any of *candidates* differs from what the index holds.

        Only *candidates* is statted -- never the whole tree -- so this is
        cheap even on a large repository with one dirty file. A candidate
        that disappeared from disk, or that the index has and the current
        scan does not (deleted, or newly ineligible), also counts as stale,
        matching what a full :meth:`_project_is_stale` walk would report for
        the same path.
        """
        relative = {Path(path).as_posix() for path in candidates}
        if not relative:
            return False
        current = {
            item.path.as_posix(): item
            for item in self.indexer.scanner.scan_paths(project, relative, existing)
            if isinstance(item, ScannedFile)
        }
        for path in relative:
            indexed = existing.get(path)
            scanned = current.get(path)
            if (indexed is None) != (scanned is None):
                return True
            if (
                indexed is not None
                and scanned is not None
                and (
                    scanned.size != indexed.size
                    or scanned.mtime_ns != indexed.mtime_ns
                    or scanned.language != indexed.language
                )
            ):
                return True
        return False

    def inspect_scan(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        outcome: str | None = None,
        reason: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> ScanInspectionPage:
        """Dry-run scan inspection: what an index run would find, page by page.

        Read-only and stat-only: no file is read for content, nothing is
        embedded, the index is not mutated, and no manifest is persisted. The
        cursor is an opaque offset into the *matched* stream; a page re-scans
        from the start, which keeps the tool stateless at the cost of walking
        the project again for every page.
        """
        resolved = self._resolve(project, roots)
        if limit < 1 or limit > SCAN_INSPECTION_MAX_LIMIT:
            raise CodeIndexingError(
                ErrorCode.INVALID_FILTER,
                f"scan limit must be between 1 and {SCAN_INSPECTION_MAX_LIMIT}",
            )
        if outcome not in {None, "eligible", "skipped"}:
            raise CodeIndexingError(
                ErrorCode.INVALID_FILTER, "scan outcome must be 'eligible' or 'skipped'"
            )
        if reason is not None and reason not in SCAN_SKIP_REASONS:
            raise CodeIndexingError(ErrorCode.INVALID_FILTER, f"unknown scan skip reason: {reason}")
        if cursor is not None:
            try:
                skip = int(cursor)
            except ValueError:
                raise CodeIndexingError(ErrorCode.INVALID_CURSOR, "invalid scan cursor") from None
            if skip < 0:
                raise CodeIndexingError(ErrorCode.INVALID_CURSOR, "invalid scan cursor")
        else:
            skip = 0
        items: list[ScanInspectionItem] = []
        stream = self.indexer.scanner.iter_scan(resolved, read_contents=False)
        try:
            next_cursor: str | None = None
            matched = 0
            for item in stream:
                if outcome == "eligible" and not isinstance(item, ScannedFile):
                    continue
                if outcome == "skipped" and not isinstance(item, SkippedFile):
                    continue
                if reason is not None and (
                    not isinstance(item, SkippedFile) or item.reason != reason
                ):
                    continue
                matched += 1
                if matched <= skip:
                    continue
                if len(items) >= limit:
                    next_cursor = str(skip + len(items))
                    break
                if isinstance(item, ScannedFile):
                    items.append(
                        ScanInspectionItem(
                            path=item.path,
                            outcome="eligible",
                            language=item.language,
                            size=item.size,
                            mtime_ns=item.mtime_ns,
                        )
                    )
                else:
                    items.append(
                        ScanInspectionItem(
                            path=item.path,
                            outcome="skipped",
                            reason=item.reason,
                            detail=item.detail,
                        )
                    )
            return ScanInspectionPage(project=resolved, items=items, next_cursor=next_cursor)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

    def list_projects(self) -> list[ProjectInfo]:
        return sorted(self.store.list_projects(), key=lambda project: (project.name, project.id))

    def remove_project(self, project: str) -> RemovalReport:
        resolved = self._resolve(project, [])
        self.invalidate_freshness(resolved.id)
        lock_directory = self.paths.data / "locks"
        lock_directory.mkdir(parents=True, exist_ok=True)
        with (
            FileLock(lock_directory / "index-global.lock"),
            FileLock(lock_directory / f"{resolved.id}.lock"),
            FileLock(lock_directory / f"active-{resolved.id}.lock"),
        ):
            removed = self.store.remove_project(resolved.id, locks_held=True)
            if removed:
                shutil.rmtree(self.paths.data / "staging" / resolved.id, ignore_errors=True)
                progress = self.paths.data / "progress" / f"{resolved.id}.json"
                progress.unlink(missing_ok=True)
        return RemovalReport(project_id=resolved.id, removed=removed)

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
        resolved = self._scope_checkouts(projects, all_projects, roots)
        project_ids = list(dict.fromkeys(project.id for project in resolved))

        def search(targets: Mapping[str, Sequence[ActiveIndexTarget]]) -> SearchResponse:
            self._ensure_query_generations(targets)
            return self.search.search_code(
                query,
                project_ids,
                languages=languages,
                paths=paths,
                kinds=kinds,
                limit=limit,
                partitions={
                    project_id: [target.partition for target in targets.get(project_id, ())]
                    for project_id in project_ids
                },
            )

        return self._run_repository_stable_query(
            resolved,
            search,
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
        return self._run_repository_stable_query(
            [resolved],
            lambda targets: self._find_symbol_for_target(
                name,
                self._primary_target(targets, resolved.id),
                match=match,
                kinds=kinds,
                limit=limit,
            ),
        )

    def _find_symbol_for_target(
        self,
        name: str,
        target: ActiveIndexTarget,
        *,
        match: str,
        kinds: list[str] | None,
        limit: int,
    ) -> SymbolResponse:
        self._ensure_query_generations({target.project.id: [target]})
        return self.search.find_symbol(
            name,
            target.project.id,
            match=match,
            kinds=kinds,
            limit=limit,
            partition=target.partition,
        )

    def file_outline(
        self, path: str, project: str | None = None, *, roots: list[Path] | None = None
    ) -> OutlineResponse:
        resolved = self.resolve_project(project, roots)
        return self._run_repository_stable_query(
            [resolved],
            lambda targets: self._file_outline_for_target(
                path, self._primary_target(targets, resolved.id)
            ),
        )

    def _file_outline_for_target(self, path: str, target: ActiveIndexTarget) -> OutlineResponse:
        self._ensure_query_generations({target.project.id: [target]})
        return self.search.file_outline(path, target.project.id, partition=target.partition)

    def get_chunk(self, chunk_id: str) -> CodeChunk:
        project_id = self.store.chunk_project_id(chunk_id)
        if project_id is None:
            return self.search.get_chunk(chunk_id)
        resolved = self._resolve(project_id, None)
        return self._run_repository_stable_query(
            [resolved],
            lambda targets: self._get_chunk_for_target(
                chunk_id, self._primary_target(targets, project_id)
            ),
        )

    def _get_chunk_for_target(self, chunk_id: str, target: ActiveIndexTarget) -> CodeChunk:
        self._ensure_query_generations({target.project.id: [target]})
        return self.search.get_chunk(chunk_id, partition=target.partition)

    def _prepare_reference_query(
        self,
        selector: DeclarationSelector,
        target: ActiveIndexTarget,
    ) -> tuple[DeclarationSelector, ReferenceBackfillReport, PartitionRef]:
        if selector.project is not None:
            selector = selector.model_copy(update={"project": target.project.id})
        else:
            chunk_project_id = self.store.chunk_project_id(selector.chunk_id or "")
            if chunk_project_id is None:
                # Preserve the established CHUNK_NOT_FOUND error contract for
                # malformed and retired chunk ids.
                self.search.get_chunk(selector.chunk_id or "", partition=target.partition)
                raise AssertionError("get_chunk unexpectedly returned without a project id")
            if chunk_project_id != target.project.id:
                raise ValueError("reference target does not own selector chunk")
        self._ensure_query_generations({target.project.id: [target]})
        report = self.ensure_reference_index(target.project.id, _target=target)
        return selector, report, target.partition

    def _ensure_query_generations(self, targets: Mapping[str, Sequence[ActiveIndexTarget]]) -> None:
        """Rebuild incompatible partitions before any query can observe them."""
        for project_id in sorted(targets):
            for target in targets[project_id]:
                if (
                    self.store.incompatibility_reason(
                        project_id,
                        self.embedder.model_id,
                        partition_id=target.partition_id,
                    )
                    is not None
                ):
                    self.indexer.index(
                        target.project,
                        partition=target.partition,
                        wait_for_lock=True,
                        trigger="lazy-query",
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
        resolved = self._resolve_reference_project(selector, roots)
        return self._run_repository_stable_query(
            [resolved],
            lambda targets: self._find_references_for_target(
                selector,
                self._primary_target(targets, resolved.id),
                kinds=kinds,
                limit=limit,
                cursor=cursor,
            ),
        )

    def _find_references_for_target(
        self,
        selector: DeclarationSelector,
        target: ActiveIndexTarget,
        *,
        kinds: set[str] | None,
        limit: int,
        cursor: str | None,
    ) -> ReferenceResponse:
        selector, report, partition = self._prepare_reference_query(selector, target)
        with self.store.partition_access(report.project_id, partition_id=partition.partition_id):
            return self.references.find_references(
                selector,
                kinds=kinds,
                limit=limit,
                cursor=cursor,
                backfill=report,
                partition=partition,
                root=target.project.root,
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
        resolved = self._resolve_reference_project(selector, roots)
        return self._run_repository_stable_query(
            [resolved],
            lambda targets: self._impact_radius_for_target(
                selector,
                self._primary_target(targets, resolved.id),
                max_depth=max_depth,
                include_likely=include_likely,
                kinds=kinds,
                max_nodes=max_nodes,
                limit=limit,
                cursor=cursor,
            ),
        )

    def _impact_radius_for_target(
        self,
        selector: DeclarationSelector,
        target: ActiveIndexTarget,
        *,
        max_depth: int,
        include_likely: bool,
        kinds: set[str] | None,
        max_nodes: int,
        limit: int,
        cursor: str | None,
    ) -> ImpactRadiusResponse:
        selector, report, partition = self._prepare_reference_query(selector, target)
        with self.store.partition_access(report.project_id, partition_id=partition.partition_id):
            return self.references.impact_radius(
                selector,
                max_depth=max_depth,
                include_likely=include_likely,
                kinds=kinds,
                max_nodes=max_nodes,
                limit=limit,
                cursor=cursor,
                backfill=report,
                partition=partition,
                root=target.project.root,
            )

    def analyze_refactor(
        self,
        selector: DeclarationSelector,
        operation: RefactorOperation,
        *,
        limit: int = 500,
        cursor: str | None = None,
        roots: list[Path] | None = None,
    ) -> RefactorAnalysis:
        resolved = self._resolve_reference_project(selector, roots)
        return self._run_repository_stable_query(
            [resolved],
            lambda targets: self._analyze_refactor_for_target(
                selector,
                operation,
                self._primary_target(targets, resolved.id),
                limit=limit,
                cursor=cursor,
            ),
        )

    def _analyze_refactor_for_target(
        self,
        selector: DeclarationSelector,
        operation: RefactorOperation,
        target: ActiveIndexTarget,
        *,
        limit: int,
        cursor: str | None,
    ) -> RefactorAnalysis:
        selector, report, partition = self._prepare_reference_query(selector, target)
        with self.store.partition_access(report.project_id, partition_id=partition.partition_id):
            return self.references.analyze_refactor(
                selector,
                operation,
                limit=limit,
                cursor=cursor,
                backfill=report,
                partition=partition,
                root=target.project.root,
            )

    def emit_refactor_patch(
        self,
        selector: DeclarationSelector,
        operation: RefactorOperation,
        *,
        context_lines: int = 3,
        roots: list[Path] | None = None,
    ) -> RefactorPatch:
        validate_patch_request(operation, context_lines)
        resolved = self._resolve_reference_project(selector, roots)
        return self._run_repository_stable_query(
            [resolved],
            lambda targets: self._emit_refactor_patch_for_target(
                selector,
                operation,
                self._primary_target(targets, resolved.id),
                context_lines=context_lines,
            ),
        )

    def _emit_refactor_patch_for_target(
        self,
        selector: DeclarationSelector,
        operation: RefactorOperation,
        target: ActiveIndexTarget,
        *,
        context_lines: int,
    ) -> RefactorPatch:
        selector, report, partition = self._prepare_reference_query(selector, target)
        with self.store.partition_access(report.project_id, partition_id=partition.partition_id):
            return self.references.emit_refactor_patch(
                selector,
                operation,
                context_lines=context_lines,
                backfill=report,
                partition=partition,
                root=target.project.root,
            )

    def _resolve_reference_project(
        self, selector: DeclarationSelector, roots: list[Path] | None
    ) -> ProjectInfo:
        if selector.project is not None:
            return self._resolve(selector.project, roots)
        project_id = self.store.chunk_project_id(selector.chunk_id or "")
        if project_id is None:
            self.search.get_chunk(selector.chunk_id or "")
            raise AssertionError("get_chunk unexpectedly returned without a project id")
        return self._resolve(project_id, roots)

    def prepare_model(self) -> None:
        if not isinstance(self.embedder, FastEmbedder):
            return
        self.embedder.prepare()

    def _root_lock(self, root: Path) -> FileLock:
        """Return the cross-thread, cross-process lock guarding *root*'s marker."""
        directory = self.paths.data / "locks"
        directory.mkdir(parents=True, exist_ok=True)
        digest = sha256(project_root_identity(root).encode()).hexdigest()
        return FileLock(directory / f"discover-{digest}.lock")

    def _registration_lock(self) -> FileLock:
        """Return the lock serializing overlap checks and registration.

        The root lock only guards one root at a time, but overlap is a
        cross-root property: without this lock, concurrent init calls for a
        parent directory and its nested child would both observe the
        pre-registration state and both register. Always acquired after a root
        lock, never before, so lock ordering stays deadlock-free.
        """
        directory = self.paths.data / "locks"
        directory.mkdir(parents=True, exist_ok=True)
        return FileLock(directory / "registration.lock")

    def _register_project(self, project: ProjectInfo) -> None:
        """Persist *project*, upserting as pending if new or revalidating if known.

        A brand-new project starts in the "pending" state. An already-known
        project keeps its current state (e.g. "ready" is not reset back to
        "pending"), but the upsert still runs so LanceStore.upsert_project can
        apply its compatibility checks. A project whose stored generation was
        written by an incompatible embedding model or schema version is marked
        "rebuild_required" rather than rejected: registration and discovery
        keep working, and the next index run rebuilds the partition.
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
        return list(
            dict.fromkeys(
                project.id for project in self._scope_checkouts(projects, all_projects, roots)
            )
        )

    def resolve_scope_checkouts(
        self,
        projects: list[str] | None,
        all_projects: bool,
        roots: list[Path] | None = None,
    ) -> list[ProjectInfo]:
        """Resolve every checkout behind a search scope, primary checkout first.

        One entry per requested checkout, so freshness can be verified for
        each slot a merged search will actually read instead of only the
        primary checkout's.
        """
        return self._scope_checkouts(projects, all_projects, roots)

    def _scope_checkouts(
        self,
        projects: list[str] | None,
        all_projects: bool,
        roots: list[Path] | None = None,
    ) -> list[ProjectInfo]:
        """Resolve every checkout behind a search scope.

        An explicit selector binds its project to the request's own checkout
        when possible; an unscoped request returns all requested checkouts of
        the single in-scope registration so their branch slots are searched
        together. ``all_projects`` keeps each registration's canonical root.
        """
        resolver = ProjectResolver(self.store.list_projects())
        if projects and all_projects:
            raise CodeIndexingError(
                ErrorCode.INVALID_FILTER,
                "projects and all_projects cannot be used together",
            )
        if all_projects:
            scope = [
                resolver.resolve_scope(explicit=project.id)[0] for project in self.list_projects()
            ]
        elif projects:
            selected: dict[str, ProjectInfo] = {}
            for selector in projects:
                project = resolver.resolve_scope(
                    explicit=selector, roots=roots or [], cwd=self.cwd
                )[0]
                selected.setdefault(project.id, project)
            # Explicit selection answers with each requested checkout of the
            # selected registrations, not just one: when the request's roots
            # carry several markers of a shared registration -- a main
            # checkout and linked worktrees -- every one of those checkouts
            # joins the scope behind its primary.
            scope = []
            seen: set[tuple[str, str]] = set()
            candidates = [*selected.values(), *resolver._marked_checkouts(roots or [])]
            for project in candidates:
                if project.id not in selected:
                    continue
                key = (project.id, project_root_identity(project.root))
                if key in seen:
                    continue
                seen.add(key)
                scope.append(project)
        else:
            scope = resolver.resolve_scope(roots=roots or [], cwd=self.cwd)
        if not scope:
            raise CodeIndexingError(
                ErrorCode.PROJECT_NOT_FOUND,
                "No indexed projects are available; init_project registers one and "
                "index_project builds its index",
            )
        return scope

    def _resolve(self, explicit: str | None, roots: list[Path] | None) -> ProjectInfo:
        return ProjectResolver(self.store.list_projects()).resolve(
            explicit=explicit,
            roots=roots or [],
            cwd=self.cwd,
        )
