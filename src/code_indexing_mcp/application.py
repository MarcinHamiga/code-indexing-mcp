"""Application services shared by MCP and CLI adapters."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
from .calibration import LIMITED_BY_MEMORY, crossover_characters
from .embedding import Embedder, FastEmbedder, SegmentPlan
from .embedding_worker import EmbeddingWorkerSession, WorkerConfig, default_launcher
from .errors import CodeIndexingError, ErrorCode
from .extractor import TreeSitterExtractor
from .history import HistoryStore
from .indexing import Indexer
from .models import (
    SCAN_SKIP_REASONS,
    CodeChunk,
    DeclarationSelector,
    HistoryPage,
    IndexReport,
    IndexTrigger,
    MaintenanceProjectResult,
    MaintenanceReport,
    ModelStatus,
    OutlineResponse,
    ProjectInfo,
    ProjectStatus,
    ProjectStorageStats,
    RefactorAnalysis,
    RefactorOperation,
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
    TableStorageStats,
)
from .passage_backend import PassageBackendSession
from .probe_cache import ProbeCache, ProbeKey, ProbeRecord, model_artifact_fingerprint
from .progress import IndexProgress, read_progress
from .projects import (
    ProjectResolver,
    find_project_root,
    initialize_project,
    project_root_identity,
    read_project_marker,
    same_project_root,
)
from .reference_service import ReferenceService
from .scanner import SourceScanner
from .search import SearchService
from .settings import IndexSettings
from .staging import has_pending_recovery, recover_staged_commits
from .storage import LanceStore, overlap_warnings, overlapping_registration, worktree_warnings
from .token_batching import max_token_product_for
from .worker_launcher import ExternalInterpreterLauncher, WorkerLauncher

logger = logging.getLogger(__name__)

# Startup recovery needs the global index lock, but that lock is held for the
# whole of an index run. Wait only long enough to lose a race against a commit
# that is about to finish; a run genuinely in flight is left to a later start.
RECOVERY_LOCK_TIMEOUT_SECONDS = 5.0

# Automatic maintenance repeats its overdue check at most this often. The check
# itself is gated by the persisted last-successful-maintenance timestamp.
MAINTENANCE_CHECK_INTERVAL = timedelta(hours=24)

# Negative freshness answers are cached briefly so a burst of tool calls in one
# agent interaction does not walk the same clean repository once per call. The
# window is short enough that an external edit surfaces on the next check, and
# watcher events, indexing, and registration invalidate it immediately.
FRESHNESS_CACHE_SECONDS = 5.0

SCAN_INSPECTION_MAX_LIMIT = 200

MAINTENANCE_TIMESTAMP_FILE = "maintenance.json"
MAINTENANCE_LOCK_FILE = "maintenance-schedule.lock"


def _read_maintenance_timestamp(path: Path) -> datetime | None:
    """Return the last successful maintenance time, or None when unreadable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        timestamp = datetime.fromisoformat(payload["last_maintenance_at"])
        return timestamp if timestamp.utcoffset() is not None else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_maintenance_timestamp(path: Path) -> None:
    """Persist the maintenance timestamp atomically so readers never parse a partial file."""
    payload = json.dumps(
        {
            "schema_version": 1,
            "last_maintenance_at": datetime.now(UTC).isoformat(),
        },
        sort_keys=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _estimate_reclaimable(stats: ProjectStorageStats) -> int:
    """Estimate bytes a table could reclaim: physical minus logical, floor zero."""
    return sum(max(0, table.physical_bytes - table.logical_bytes) for table in stats.tables)


def _versions_removed(before: ProjectStorageStats, after: ProjectStorageStats) -> int:
    """Sum retained versions removed across the tables shared by both snapshots."""
    removed = 0
    after_by_name = {table.name: table for table in after.tables}
    for table in before.tables:
        after_table = after_by_name.get(table.name)
        if after_table is not None:
            removed += max(0, table.retained_version_count - after_table.retained_version_count)
    return removed


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
        data = Path(os.environ.get("CODE_INDEXING_DATA_DIR", user_data_path("code-indexing-mcp")))
        cache = Path(
            os.environ.get("CODE_INDEXING_CACHE_DIR", user_cache_path("code-indexing-mcp"))
        )
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
            offline = os.environ.get("CODE_INDEXING_OFFLINE", "").lower() in {"1", "true", "yes"}
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
        # Negative freshness results, keyed by project id: the monotonic
        # deadline and the scan-config fingerprint the answer was computed for.
        self._clean_freshness_until: dict[str, tuple[float, str]] = {}
        self.embedding_batch_size = self.settings.embedding_batch_size
        self.batch_calibration = "explicit"
        if self.settings.embedding_batch_auto:
            self.batch_calibration = "default"
            if self.backend_selection.uses_accelerator:
                self._probe_key = self._build_probe_key(embedder)
                cached = self.probe_cache.load(self._probe_key)
                if cached is not None and cached.batch_size > 0:
                    self.embedding_batch_size = cached.batch_size
                    # A size something forced down is not the size calibration
                    # chose, and a machine pinned low by one bad run has to be
                    # able to say so. Which kind of ceiling stopped it is a
                    # question for the recommendation, not for this label.
                    self.batch_calibration = "reduced" if cached.limited_by else "measured"

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
            progress_directory=paths.data / "progress",
            history=self.history,
        )
        self.search = SearchService(self.store, embedder)
        self.references = ReferenceService(self.store)

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

    def crossover_characters(self) -> int | None:
        """Return the run size below which this machine should stay on CPU.

        0 means "start the accelerator immediately", which is the answer when
        the operator turned deferral off and also when nothing has been measured
        yet -- the first run on a machine is what does the measuring, and it
        cannot defer on numbers it is in the middle of producing.

        ``None`` means the accelerator never overtakes CPU, so no run is large
        enough to be worth starting it for. That is not the same statement as a
        very large threshold, and reporting it as one would name a size some run
        could conceivably pass -- and would collide with an operator who pinned
        that size deliberately.

        Strict mode is a third such answer. It exists for a caller who would
        rather fail than quietly index at CPU speed, and a deferral is quiet
        CPU indexing that no degradation reports -- so under strict mode the
        accelerator that was asked for is the one that runs, whatever the run
        turns out to cost.
        """
        if self.settings.embedding_strict:
            return 0
        if not self.settings.embedding_crossover_auto:
            return self.settings.embedding_crossover_characters
        cpu, accelerator = self._measurements()
        if cpu is None or accelerator is None:
            return 0
        return self._measured_crossover()

    def _measured_crossover(self) -> int | None:
        """Return the crossover the recorded measurements imply, if both exist.

        What the machine measured, with no policy applied. ``model status``
        reports this, so an explicit threshold or strict mode changes which runs
        defer without changing what this machine was found to be.
        """
        cpu, accelerator = self._measurements()
        if cpu is None or accelerator is None:
            return None
        return crossover_characters(
            accelerator_load_ns=accelerator.load_ns,
            cpu_load_ns=cpu.load_ns,
            cpu_characters_per_second=cpu.characters_per_second,
            accelerator_characters_per_second=accelerator.characters_per_second,
        )

    def _recommended_override(
        self, cpu: ProbeRecord | None, accelerator: ProbeRecord | None
    ) -> str | None:
        """Return the one setting change the measurements argue for, if any.

        Only a memory ceiling has a setting behind it. A batch that took the
        worker down with it was reduced just the same, but raising the ceiling
        is not what answers a device that could not make the allocation, so
        that case is reported without advice attached.
        """
        if accelerator is not None and accelerator.limited_by == LIMITED_BY_MEMORY:
            return (
                "CODE_INDEXING_EMBED_MEMORY_MB (a batch overran the ceiling and was reduced to "
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
                "CODE_INDEXING_EMBED_ACCELERATOR=cpu (the accelerator measured no faster than CPU "
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
        # What was measured, not what policy does with it: an explicit setting
        # or strict mode changes which runs defer, and neither changes what this
        # machine turned out to be.
        measured_crossover = self._measured_crossover()
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
        # ids for the same project. Marker creation, overlap rejection, and
        # registration are one critical section, keyed the same way as discovery.
        with self._root_lock(root):
            project = initialize_project(root, name=name, force_new_id=force_new_id)
            if not allow_overlap and not force_new_id:
                existing = overlapping_registration(self.store.list_projects(), project.root)
                if existing is not None and existing.id != project.id:
                    raise CodeIndexingError(
                        ErrorCode.OVERLAPPING_PROJECT,
                        f"Project root {project.root} overlaps the registered root "
                        f"{existing.root} of project {existing.id!r}; pass allow_overlap=true "
                        "to register it anyway",
                        existing_project=existing.id,
                        new_project=project.id,
                    )
            self._register_project(project)
            self.invalidate_freshness(project.id)
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
            self.invalidate_freshness(project.id)
            return project

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
        try:
            return self.indexer.index(
                resolved,
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
        self, project: str | None = None, *, roots: list[Path] | None = None
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
        if self._project_is_stale(resolved):
            self.indexer.index(resolved, wait_for_lock=True, trigger="lazy-query")
        report = self.indexer.backfill_references(
            resolved, wait_for_lock=True, trigger="reference-backfill"
        )
        if report.stale_paths:
            self.indexer.index(resolved, wait_for_lock=True, trigger="lazy-query")
            report = self.indexer.backfill_references(
                resolved, wait_for_lock=True, trigger="reference-backfill"
            )
        self.invalidate_freshness(resolved.id)
        return report

    def project_status(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> ProjectStatus:
        resolved = self._resolve(project, roots)
        files = self.store.list_files(resolved.id)
        state = self.store.project_state(resolved.id)
        if state in {"ready", "partial"}:
            fingerprint = resolved.scan.model_dump_json()
            cached = self._clean_freshness_until.get(resolved.id)
            if cached is not None and cached[1] == fingerprint and cached[0] > time.monotonic():
                # A recent check found this exact scan configuration clean;
                # do not walk the repository again for this call.
                pass
            elif self._project_is_stale(resolved, {record.path: record for record in files}):
                self._clean_freshness_until.pop(resolved.id, None)
                state = "stale"
            else:
                self._clean_freshness_until[resolved.id] = (
                    time.monotonic() + FRESHNESS_CACHE_SECONDS,
                    fingerprint,
                )
        return ProjectStatus(
            project=resolved,
            state=state,
            file_count=len(files),
            chunk_count=self.store.count_chunks([resolved.id]),
            progress=self.index_progress(resolved.id),
            last_run=self.history.recent(resolved.id),
        )

    def invalidate_freshness(self, project_id: str) -> None:
        """Forget a cached clean answer, forcing the next status check to scan.

        Called after anything that changes what the index holds -- registration,
        a completed index or reference backfill, removal -- and by eager-mode
        watchers the moment a file system event lands.
        """
        self._clean_freshness_until.pop(project_id, None)

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

        Never mutates the index: a registered project with no partition reports
        zeroed tables instead of materializing one. Root-overlap and shared-Git
        worktree warnings are advisory and best-effort.
        """
        snapshot_at = datetime.now(UTC).isoformat()
        registry_before = self.store.registry_stats()
        registered = self.list_projects()
        if project is not None:
            resolved = self._resolve(project, roots)
            projects = [self.store.storage_stats_for(resolved)]
        else:
            projects = [
                self.store.storage_stats_for(registered_project)
                for registered_project in registered
            ]
        registry_after = self.store.registry_stats()
        return StorageStatus(
            snapshot_at=snapshot_at,
            registry=registry_after,
            projects=projects,
            physical_bytes_total=registry_after.physical_bytes
            + sum(stats.partition_physical_bytes for stats in projects),
            consistent=registry_before.current_version == registry_after.current_version
            and all(stats.consistent for stats in projects),
            overlap_warnings=overlap_warnings(registered),
            worktree_warnings=worktree_warnings(registered),
        )

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

        Runs under the same global and per-project writer locks indexing uses,
        so a pass never races a commit. Automatic passes attempt the locks
        without waiting and record busy projects as skipped; manual passes wait
        them out. A dry run collects the before statistics, a reclaimable-bytes
        estimate, and no after statistics (``after`` stays null) but mutates
        nothing and takes no locks.

        ``trigger`` labels the pass as ``manual`` or ``scheduled`` for audit
        purposes. The registry is maintained once under the global lock; when an
        automatic pass cannot get that lock, the registry is reported without
        maintenance rather than failing the whole run.
        """
        started = time.monotonic_ns()
        started_at = datetime.now(UTC).isoformat()
        retention = timedelta(hours=self.settings.version_retention_hours)
        registered = self.list_projects()
        if project is not None:
            resolved = self._resolve(project, roots)
            scope: list[ProjectInfo] = [resolved]
        else:
            scope = registered
        lock_directory = self.paths.data / "locks"
        lock_directory.mkdir(parents=True, exist_ok=True)

        def acquire(lock: FileLock) -> bool:
            if wait_for_lock:
                lock.acquire()
                return True
            try:
                lock.acquire(timeout=0)
            except Timeout:
                return False
            return True

        results: list[MaintenanceProjectResult] = []
        for registered_project in scope:
            before: ProjectStorageStats | None = None
            estimate = 0
            if dry_run:
                try:
                    before = self.store.storage_stats_for(registered_project)
                    estimate = _estimate_reclaimable(before)
                    if before.partition_open_failed:
                        results.append(
                            MaintenanceProjectResult(
                                project=registered_project,
                                before=before,
                                status="error",
                                error="Partition exists but its tables could not be opened",
                                reclaimable_bytes_estimate=estimate,
                            )
                        )
                    else:
                        results.append(
                            MaintenanceProjectResult(
                                project=registered_project,
                                before=before,
                                skip_reason="not-indexed" if not before.tables else "dry-run",
                                reclaimable_bytes_estimate=estimate,
                            )
                        )
                except Exception as exc:
                    results.append(
                        MaintenanceProjectResult(
                            project=registered_project,
                            status="error",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                continue

            global_lock = FileLock(lock_directory / "index-global.lock")
            project_lock = FileLock(lock_directory / f"{registered_project.id}.lock")
            global_acquired = False
            project_acquired = False
            try:
                global_acquired = acquire(global_lock)
                if not global_acquired:
                    results.append(
                        MaintenanceProjectResult(
                            project=registered_project,
                            skip_reason="busy",
                        )
                    )
                    continue
                project_acquired = acquire(project_lock)
                if not project_acquired:
                    results.append(
                        MaintenanceProjectResult(
                            project=registered_project,
                            skip_reason="busy",
                        )
                    )
                    continue
                if has_pending_recovery(self.paths.data / "staging", registered_project.id):
                    results.append(
                        MaintenanceProjectResult(
                            project=registered_project,
                            skip_reason="recovery-pending",
                        )
                    )
                    continue
                before = self.store.storage_stats_for(registered_project)
                estimate = _estimate_reclaimable(before)
                if before.partition_open_failed:
                    results.append(
                        MaintenanceProjectResult(
                            project=registered_project,
                            before=before,
                            status="error",
                            error="Partition exists but its tables could not be opened",
                            reclaimable_bytes_estimate=estimate,
                        )
                    )
                    continue
                if not before.tables:
                    results.append(
                        MaintenanceProjectResult(
                            project=registered_project,
                            before=before,
                            skip_reason="not-indexed",
                            reclaimable_bytes_estimate=estimate,
                        )
                    )
                    continue
                self.store.maintain_project(registered_project.id, cleanup_older_than=retention)
                after = self.store.storage_stats_for(registered_project)
                if after.partition_open_failed or not after.tables:
                    raise RuntimeError("Partition became unreadable during maintenance")
                results.append(
                    MaintenanceProjectResult(
                        project=registered_project,
                        before=before,
                        after=after,
                        status="ok",
                        versions_removed=_versions_removed(before, after),
                        bytes_reclaimed=max(
                            0, before.partition_physical_bytes - after.partition_physical_bytes
                        ),
                        reclaimable_bytes_estimate=estimate,
                    )
                )
            except Exception as exc:  # one project must not abort the pass
                results.append(
                    MaintenanceProjectResult(
                        project=registered_project,
                        before=before,
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                        reclaimable_bytes_estimate=estimate,
                    )
                )
            finally:
                if project_acquired:
                    project_lock.release()
                if global_acquired:
                    global_lock.release()

        registry_before: TableStorageStats | None = None
        registry_after: TableStorageStats | None = None
        registry_status = "skipped"
        registry_skip_reason: str | None = "dry-run" if dry_run else None
        registry_error: str | None = None
        registry_versions_removed = 0
        registry_bytes_reclaimed = 0
        if dry_run:
            registry_before = self.store.registry_stats()
        else:
            global_lock = FileLock(lock_directory / "index-global.lock")
            global_acquired = False
            try:
                global_acquired = acquire(global_lock)
                if not global_acquired:
                    registry_skip_reason = "busy"
                else:
                    registry_before = self.store.registry_stats()
                    self.store.maintain_registry(cleanup_older_than=retention)
                    registry_after = self.store.registry_stats()
                    registry_status = "ok"
                    registry_versions_removed = max(
                        0,
                        registry_before.retained_version_count
                        - registry_after.retained_version_count,
                    )
                    registry_bytes_reclaimed = max(
                        0, registry_before.physical_bytes - registry_after.physical_bytes
                    )
            except Exception as exc:
                registry_status = "error"
                registry_error = f"{type(exc).__name__}: {exc}"
            finally:
                if global_acquired:
                    global_lock.release()

        finished_at = datetime.now(UTC).isoformat()
        duration_ms = (time.monotonic_ns() - started) // 1_000_000
        skipped = [
            result.project.id
            for result in results
            if result.status == "skipped" and result.skip_reason != "busy"
        ]
        busy = [
            result.project.id
            for result in results
            if result.status == "skipped" and result.skip_reason == "busy"
        ]
        failed = [result.project.id for result in results if result.status == "error"]
        return MaintenanceReport(
            trigger=trigger,
            dry_run=dry_run,
            retention_hours=self.settings.version_retention_hours,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            projects=results,
            registry_before=registry_before,
            registry_after=registry_after,
            registry_status=registry_status,
            registry_skip_reason=registry_skip_reason,
            registry_error=registry_error,
            registry_versions_removed=registry_versions_removed,
            registry_bytes_reclaimed=registry_bytes_reclaimed,
            versions_removed_total=sum(result.versions_removed for result in results),
            bytes_reclaimed_total=sum(result.bytes_reclaimed for result in results),
            reclaimable_bytes_estimate_total=sum(
                result.reclaimable_bytes_estimate for result in results
            ),
            skipped_projects=skipped,
            busy_projects=busy,
            failed_projects=failed,
        )

    def maybe_run_maintenance(self) -> MaintenanceReport | None:
        """Run scheduled maintenance when it is due, persisting the last-run stamp.

        Gated by ``CODE_INDEXING_AUTO_MAINTENANCE`` and checked at most once per
        24 hours, using the timestamp of the last complete pass. Runs in any
        indexing mode: maintenance never scans source files or creates a new
        logical generation, so lazy, eager, and manual modes are all eligible.
        Busy, damaged, or recovery-dependent projects leave the stamp stale so
        a later startup retries them instead of waiting another 24 hours.
        """
        if not self.settings.auto_maintenance:
            return None
        timestamp_path = self.paths.data / MAINTENANCE_TIMESTAMP_FILE
        lock_directory = self.paths.data / "locks"
        lock_directory.mkdir(parents=True, exist_ok=True)
        schedule_lock = FileLock(lock_directory / MAINTENANCE_LOCK_FILE)
        try:
            schedule_lock.acquire(timeout=0)
        except Timeout:
            return None
        try:
            # Re-check after acquiring the cross-process lock: another startup
            # may have completed the overdue pass while this process waited to run.
            last = _read_maintenance_timestamp(timestamp_path)
            if last is not None and datetime.now(UTC) - last < MAINTENANCE_CHECK_INTERVAL:
                return None
            report = self.maintain_storage(dry_run=False, wait_for_lock=False, trigger="scheduled")
            projects_complete = all(
                result.status == "ok"
                or (result.status == "skipped" and result.skip_reason == "not-indexed")
                for result in report.projects
            )
            if report.registry_status == "ok" and projects_complete:
                _write_maintenance_timestamp(timestamp_path)
            return report
        finally:
            schedule_lock.release()

    def project_is_stale(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> bool:
        """Return whether eligible source metadata differs from the live index."""
        return self._project_is_stale(self._resolve(project, roots))

    def _project_is_stale(
        self, project: ProjectInfo, existing: dict[str, StoredFile] | None = None
    ) -> bool:
        if existing is None:
            existing = {record.path: record for record in self.store.list_files(project.id)}
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
        self._clean_freshness_until.pop(resolved.id, None)
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
        self._ensure_query_generations(project_ids, roots)
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
        self._ensure_query_generations([resolved.id], roots)
        return self.search.find_symbol(name, resolved.id, match=match, kinds=kinds, limit=limit)

    def file_outline(
        self, path: str, project: str | None = None, *, roots: list[Path] | None = None
    ) -> OutlineResponse:
        resolved = self.resolve_project(project, roots)
        self._ensure_query_generations([resolved.id], roots)
        return self.search.file_outline(path, resolved.id)

    def get_chunk(self, chunk_id: str) -> CodeChunk:
        project_id = self.store._chunk_project_id(chunk_id)
        if project_id is not None:
            self._ensure_query_generations([project_id], None)
        return self.search.get_chunk(chunk_id)

    def _prepare_reference_query(
        self,
        selector: DeclarationSelector,
        roots: list[Path] | None,
    ) -> tuple[DeclarationSelector, ReferenceBackfillReport]:
        if selector.project is not None:
            resolved = self._resolve(selector.project, roots)
            selector = selector.model_copy(update={"project": resolved.id})
            project_id = resolved.id
        else:
            chunk_project_id = self.store._chunk_project_id(selector.chunk_id or "")
            if chunk_project_id is None:
                # Preserve the established CHUNK_NOT_FOUND error contract for
                # malformed and retired chunk ids.
                self.search.get_chunk(selector.chunk_id or "")
                raise AssertionError("get_chunk unexpectedly returned without a project id")
            project_id = chunk_project_id
        self._ensure_query_generations([project_id], roots)
        return selector, self.ensure_reference_index(project_id, roots=roots)

    def _ensure_query_generations(self, project_ids: list[str], roots: list[Path] | None) -> None:
        """Rebuild incompatible partitions before any query can observe them."""
        for project_id in project_ids:
            if self.store.incompatibility_reason(project_id, self.embedder.model_id) is not None:
                self.index_project(
                    project_id,
                    roots=roots,
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
        selector, report = self._prepare_reference_query(selector, roots)
        with self.store.partition_access(report.project_id):
            return self.references.find_references(
                selector, kinds=kinds, limit=limit, cursor=cursor, backfill=report
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
        selector, report = self._prepare_reference_query(selector, roots)
        with self.store.partition_access(report.project_id):
            return self.references.analyze_refactor(
                selector, operation, limit=limit, cursor=cursor, backfill=report
            )

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
            raise CodeIndexingError(
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
            raise CodeIndexingError(
                ErrorCode.PROJECT_NOT_FOUND,
                "No indexed projects are available; init_project registers one and "
                "index_project builds its index",
            )
        return project_ids
