"""Application services shared by MCP and CLI adapters."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from filelock import FileLock, Timeout
from platformdirs import user_cache_path, user_data_path

from .accelerator_env import apply_environment, load_environment
from .backends import (
    CPU_BACKEND,
    BackendDescriptor,
    BackendSelection,
    available_execution_providers,
    backend_for,
    describe_environment,
    platform_fingerprint,
    runtime_version,
    select_backend,
)
from .calibration import crossover_characters
from .embedding import Embedder, FastEmbedder, SegmentPlan
from .embedding_worker import EmbeddingWorkerSession, WorkerConfig, default_launcher
from .errors import ErrorCode, IncodeError
from .extractor import TreeSitterExtractor
from .indexing import Indexer
from .models import (
    CodeChunk,
    IndexReport,
    ModelStatus,
    OutlineResponse,
    ProjectInfo,
    ProjectStatus,
    RemovalReport,
    ScanConfig,
    SearchResponse,
    SymbolResponse,
)
from .passage_backend import PassageBackendSession
from .probe_cache import ProbeCache, ProbeKey, ProbeRecord, model_artifact_fingerprint
from .projects import ProjectResolver, find_project_root, initialize_project, read_project_marker
from .scanner import SourceScanner
from .search import SearchService
from .settings import MAX_CROSSOVER_CHARACTERS, IndexSettings
from .staging import recover_staged_commits
from .storage import LanceStore
from .token_batching import max_token_product_for
from .worker_launcher import ExternalInterpreterLauncher, WorkerLauncher

logger = logging.getLogger(__name__)

# Startup recovery needs the global index lock, but that lock is held for the
# whole of an index run. Wait only long enough to lose a race against a commit
# that is about to finish; a run genuinely in flight is left to a later start.
RECOVERY_LOCK_TIMEOUT_SECONDS = 5.0


def _rate(record: ProbeRecord | None) -> float | None:
    """Return a measured rate, or None for one that was never measured.

    A stored zero means the record predates its measurement, and reporting it
    as zero characters per second would describe a backend that never finishes.
    """
    if record is None or record.characters_per_second <= 0:
        return None
    return record.characters_per_second


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
        # Passage embedding is the only role acceleration targets. The query
        # model stays in this process on CPU so a search never waits on a
        # worker spawning or a model loading onto a device.
        #
        # An accelerator usually lives in a second environment the installer
        # prepared, so what this process can execute is not the whole story:
        # the providers that environment reported are candidates too, and a
        # backend chosen from them runs in its interpreter rather than ours.
        self.serving_providers = available_execution_providers()
        self.accelerator_environment = load_environment(paths.data)
        self.backend_selection = self._select_backend()
        self.probe_cache = ProbeCache(paths.cache / "backend-probes.json")
        self._probe_key: ProbeKey | None = None
        # Set when a run actually tried the selected accelerator and it failed.
        # Only successful probes are cached, so without this memo every index
        # run in a long-lived daemon would re-spawn a known-dead backend and
        # reload its model onto the device before giving up again.
        self._runtime_fallback: BackendSelection | None = None
        self.embedding_batch_size = self.settings.embedding_batch_size
        self.batch_calibration = "explicit"
        if self.settings.embedding_batch_auto:
            self.batch_calibration = "default"
            if self.backend_selection.uses_accelerator:
                self._probe_key = self._build_probe_key(embedder)
                cached = self.probe_cache.load(self._probe_key)
                if cached is not None and cached.batch_size > 0:
                    self.embedding_batch_size = cached.batch_size
                    # A size a ceiling overrun forced down is not the size
                    # calibration chose, and a machine pinned low by one bad run
                    # has to be able to say so.
                    self.batch_calibration = (
                        "reduced" if cached.limited_by == "memory" else "measured"
                    )

        passage_session_factory: Callable[[], PassageBackendSession] | None = None
        if isinstance(embedder, FastEmbedder) and self.settings.index_execution == "worker":
            passage_session_factory = self._passage_session_factory(embedder)
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
        )
        self.search = SearchService(self.store, embedder)

    def _select_backend(self) -> BackendSelection:
        """Choose a backend from everything this machine can actually execute."""
        record = self.accelerator_environment.environment
        providers = list(self.serving_providers)
        if record is not None:
            # The prepared environment vouches for the one accelerator it was
            # probed for, not for every provider its runtime happens to ship.
            # Widening any further would offer a backend on the strength of a
            # record that never exercised it -- and would let selection land on
            # an accelerator whose device and driver this record cannot describe.
            prepared = backend_for(record.accelerator)
            if prepared is not None and prepared.provider in record.providers:
                providers.append(prepared.provider)
        selection = select_backend(
            self.settings.embedding_accelerator, available_providers=providers
        )
        if record is not None and selection.uses_accelerator:
            selection = selection.described_as(apply_environment(selection.descriptor, record))
        rejection = self.accelerator_environment.reason
        if rejection is not None and not selection.uses_accelerator:
            # A record that was found and refused explains the CPU outcome far
            # better than "no accelerator is prepared" does.
            selection = selection.diagnosed(rejection)
        return selection

    def _runs_externally(self, descriptor: BackendDescriptor) -> bool:
        """Whether a worker for *descriptor* needs the prepared environment.

        A provider this interpreter already exposes needs no second environment
        -- an explicitly requested Core ML on macOS runs in the serving
        environment's own runtime. Anything offered only by the prepared
        accelerator environment runs in that environment's interpreter.
        """
        record = self.accelerator_environment.environment
        return record is not None and descriptor.provider not in self.serving_providers

    def _accelerator_launcher(self, descriptor: BackendDescriptor) -> WorkerLauncher:
        """Return where a worker for *descriptor* has to be started."""
        record = self.accelerator_environment.environment
        if record is None or not self._runs_externally(descriptor):
            return default_launcher()
        return ExternalInterpreterLauncher(
            record.interpreter,
            environment_name=f"{record.accelerator.value} environment",
        )

    @property
    def effective_backend_selection(self) -> BackendSelection:
        """The backend the next run will attempt, after any runtime fallback.

        ``backend_selection`` records what selection resolved to from static
        capability alone. Once a run has tried it and been degraded, that
        verdict stands for the life of this process.
        """
        return self._runtime_fallback or self.backend_selection

    def _remember_fallback(self, degraded: BackendSelection) -> None:
        logger.warning(
            "Pinning passage embedding to CPU for the rest of this process: %s",
            degraded.fallback_reason,
        )
        self._runtime_fallback = degraded

    def _build_probe_key(self, embedder: Embedder) -> ProbeKey:
        descriptor = self.backend_selection.descriptor
        cache_directory = getattr(embedder, "cache_directory", self.paths.cache / "models")
        return ProbeKey(
            model_id=embedder.model_id,
            model_artifact=model_artifact_fingerprint(Path(cache_directory), embedder.model_id),
            accelerator=descriptor.accelerator.value,
            provider=descriptor.provider,
            # The record's version describes the environment that will run the
            # backend; this process's own runtime is only the fallback answer.
            runtime_version=descriptor.runtime_version or runtime_version(descriptor.runtime),
            platform=platform_fingerprint(),
            device=descriptor.device,
            driver_version=descriptor.driver_version,
        )

    def _cpu_probe_key(self) -> ProbeKey:
        """The key CPU's own calibration is stored under.

        CPU needs no probe to be trusted, but the crossover is a comparison and
        a comparison needs both sides measured -- under a key that moves when
        the model, the platform, or this process's runtime does, for the same
        reasons the accelerator's does.
        """
        cache_directory = getattr(self.embedder, "cache_directory", self.paths.cache / "models")
        return ProbeKey(
            model_id=self.embedder.model_id,
            model_artifact=model_artifact_fingerprint(
                Path(cache_directory), self.embedder.model_id
            ),
            accelerator=CPU_BACKEND.accelerator.value,
            provider=CPU_BACKEND.provider,
            runtime_version=runtime_version(CPU_BACKEND.runtime),
            platform=platform_fingerprint(),
            device=CPU_BACKEND.device,
        )

    def _cpu_max_items(self) -> int:
        """Return the microbatch size measured for CPU, if one was.

        0 means CPU keeps whatever the indexer planned, which is correct both
        when nothing has been measured and when the operator set a size
        explicitly -- an explicit size is a size for the whole installation.
        """
        if not self.settings.embedding_batch_auto:
            return 0
        record = self.probe_cache.load(self._cpu_probe_key())
        return 0 if record is None else record.batch_size

    def _measurements(self) -> tuple[ProbeRecord | None, ProbeRecord | None]:
        """Return what calibration recorded for CPU and for the accelerator."""
        selection = self.effective_backend_selection
        if not selection.uses_accelerator:
            return self.probe_cache.load(self._cpu_probe_key()), None
        key = self._probe_key or self._build_probe_key(self.embedder)
        return self.probe_cache.load(self._cpu_probe_key()), self.probe_cache.load(key)

    def crossover_characters(self) -> int:
        """Return the run size below which this machine should stay on CPU.

        0 means "start the accelerator immediately", which is the answer when
        the operator turned deferral off and also when nothing has been measured
        yet -- the first run on a machine is what does the measuring, and it
        cannot defer on numbers it is in the middle of producing.
        """
        if not self.settings.embedding_crossover_auto:
            return self.settings.embedding_crossover_characters
        cpu, accelerator = self._measurements()
        if cpu is None or accelerator is None:
            return 0
        measured = crossover_characters(
            accelerator_load_ns=accelerator.load_ns,
            cpu_load_ns=cpu.load_ns,
            cpu_characters_per_second=cpu.characters_per_second,
            accelerator_characters_per_second=accelerator.characters_per_second,
        )
        # No crossover means the accelerator never overtakes CPU. Deferring
        # every run is then exactly right, and MAX_CROSSOVER_CHARACTERS is the
        # largest run the configuration admits.
        return MAX_CROSSOVER_CHARACTERS if measured is None else measured

    def _recommended_override(
        self, cpu: ProbeRecord | None, accelerator: ProbeRecord | None
    ) -> str | None:
        """Return the one setting change the measurements argue for, if any."""
        if accelerator is not None and accelerator.limited_by == "memory":
            return (
                "INCODE_EMBED_MEMORY_MB (a batch overran the ceiling and was reduced to "
                f"{accelerator.batch_size})"
            )
        if (
            cpu is not None
            and accelerator is not None
            and cpu.characters_per_second > 0
            and accelerator.characters_per_second > 0
            and accelerator.characters_per_second <= cpu.characters_per_second
        ):
            return (
                "INCODE_EMBED_ACCELERATOR=cpu (the accelerator measured no faster than CPU "
                "on this machine)"
            )
        return None

    def _passage_session_factory(
        self, embedder: FastEmbedder
    ) -> Callable[[], PassageBackendSession]:
        """Build the factory that opens one passage session per index run.

        Both backends are described up front, but neither process is started
        until indexing asks for one, and the accelerator's is only started if
        the selection actually chose it.
        """
        ceiling_bytes = self.settings.index_memory_bytes
        strict = self.settings.embedding_strict
        probe_key = self._probe_key

        def worker_config(providers: tuple[str, ...], accelerator: str) -> WorkerConfig:
            return WorkerConfig(
                cache_directory=str(embedder.cache_directory),
                offline=embedder.offline,
                threads=self.settings.embedding_threads,
                enable_cpu_mem_arena=self.settings.embedding_cpu_arena,
                dimension=embedder.dimension,
                model_id=embedder.model_id,
                providers=providers,
                accelerator=accelerator,
            )

        descriptor = self.backend_selection.descriptor
        accelerator_config = worker_config(descriptor.providers, descriptor.accelerator.value)
        cpu_config = worker_config(CPU_BACKEND.providers, CPU_BACKEND.accelerator.value)
        accelerator_launcher = self._accelerator_launcher(descriptor)

        def session(config: WorkerConfig, launcher: WorkerLauncher) -> EmbeddingWorkerSession:
            return EmbeddingWorkerSession(
                config, configured_ceiling_bytes=ceiling_bytes, launcher=launcher
            )

        def new_passage_session() -> PassageBackendSession:
            return PassageBackendSession(
                # Read per run, not captured: a fallback recorded by an earlier
                # run keeps this one from paying for the same dead backend.
                self.effective_backend_selection,
                accelerator_factory=lambda: session(accelerator_config, accelerator_launcher),
                # The fallback never depends on a prepared environment: it is
                # what a failed accelerator falls back *to*.
                cpu_factory=lambda: session(cpu_config, default_launcher()),
                strict=strict,
                probe_cache=self.probe_cache,
                probe_key=probe_key,
                cpu_probe_key=self._cpu_probe_key(),
                # Only calibration establishes one. A configured default
                # recorded here would make ``model status`` report it as a
                # measurement that never ran.
                calibrated_batch_size=0,
                dimension=embedder.dimension,
                on_degrade=self._remember_fallback,
                # Read per run: the first run on a machine writes the numbers
                # every later run defers on, and a daemon must not have to be
                # restarted to start using them.
                crossover_characters=self.crossover_characters(),
                calibration_plan=(
                    self.indexer.segment_plan if self.settings.embedding_calibrate else None
                ),
                # The plan the indexer builds is packed for the accelerator,
                # because that is whose batch size calibration adopted. A run
                # that defers or degrades to CPU is packed for CPU instead.
                cpu_max_items=self._cpu_max_items(),
            )

        return new_passage_session

    def model_status(self) -> ModelStatus:
        """Report the resolved embedding stack without loading or probing it."""
        selection: BackendSelection = self.effective_backend_selection
        descriptor = describe_environment(selection.descriptor)
        if not selection.uses_accelerator:
            # CPU is the reference backend; it needs no probe to be trusted.
            probe_state = "not-applicable"
        else:
            key = self._probe_key or self._build_probe_key(self.embedder)
            probe_state = self.probe_cache.state(key)
        record = self.accelerator_environment.environment
        external = selection.uses_accelerator and self._runs_externally(descriptor)
        cpu, accelerator = self._measurements()
        # Reported as a number only when there is one. An accelerator that never
        # overtakes CPU has no crossover, and printing the largest admissible run
        # would read as a threshold some run could pass.
        measured_crossover: int | None = None
        if cpu is not None and accelerator is not None:
            measured_crossover = crossover_characters(
                accelerator_load_ns=accelerator.load_ns,
                cpu_load_ns=cpu.load_ns,
                cpu_characters_per_second=cpu.characters_per_second,
                accelerator_characters_per_second=accelerator.characters_per_second,
            )
        return ModelStatus(
            embedding_model=self.embedder.model_id,
            dimension=self.embedder.dimension,
            requested_accelerator=selection.requested.value,
            resolved_accelerator=descriptor.accelerator.value,
            device=descriptor.device,
            execution_provider=descriptor.provider,
            available_providers=list(selection.available_providers),
            stability=descriptor.stability.value,
            precision=descriptor.precision.value,
            runtime_version=descriptor.runtime_version,
            driver_version=descriptor.driver_version,
            # Where passage embedding will run. None means this process's own
            # environment, which is always the answer for CPU.
            accelerator_environment=str(record.interpreter) if external and record else None,
            accelerator_prepared=None if record is None else record.accelerator.value,
            batch_size=self.embedding_batch_size,
            batch_calibration=self.batch_calibration,
            probe_cache_state=probe_state,
            strict=self.settings.embedding_strict,
            fallback_reason=selection.fallback_reason,
            cpu_characters_per_second=_rate(cpu),
            accelerator_characters_per_second=_rate(accelerator),
            accelerator_load_ms=None if accelerator is None else accelerator.load_ns // 1_000_000,
            crossover_characters=measured_crossover,
            recommended_override=self._recommended_override(cpu, accelerator),
        )

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
            raise IncodeError(
                ErrorCode.PROJECT_NOT_FOUND,
                "No indexed projects are available; init_project registers one and "
                "index_project builds its index",
            )
        return project_ids
