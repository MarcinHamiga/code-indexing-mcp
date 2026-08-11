import shutil
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from code_indexing_mcp.accelerator_env import (
    RECORD_FILENAME,
    AcceleratorEnvironment,
    running_python_version,
    write_environment,
)
from code_indexing_mcp.application import Application, RuntimePaths
from code_indexing_mcp.backends import CPU_BACKEND, Accelerator
from code_indexing_mcp.embedding_worker import default_launcher
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.models import (
    DeclarationSelector,
    ProjectInfo,
    ReferenceBackfillReport,
    RenameOperation,
)
from code_indexing_mcp.settings import IndexSettings
from code_indexing_mcp.token_batching import DEFAULT_MAX_TOKEN_PRODUCT, REFERENCE_MEMORY_BYTES
from code_indexing_mcp.worker_launcher import ExternalInterpreterLauncher


class TinyEmbedder:
    model_id = "test/tiny"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, float(len(text))]


class OtherModelTinyEmbedder:
    """Same vector dimension as TinyEmbedder but a different model_id, to
    exercise LanceStore's incompatible-model detection."""

    model_id = "test/other-tiny"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, float(len(text))]


def test_concurrent_init_project_registers_one_project(tmp_path: Path) -> None:
    """The daemon serves each client on its own thread; one root, one project id."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    callers = 8
    barrier = threading.Barrier(callers)
    identifiers: list[str] = []
    lock = threading.Lock()

    def initialize() -> None:
        barrier.wait(timeout=10)
        project = app.init_project(root)
        with lock:
            identifiers.append(project.id)

    threads = [threading.Thread(target=initialize) for _ in range(callers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert len(set(identifiers)) == 1
    assert len(app.list_projects()) == 1


def test_application_orchestrates_default_project_lifecycle(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def locate_feature():\n    return True\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )

    project = app.init_project(root)
    assert app.project_status(roots=[root]).state == "pending"
    report = app.index_project(roots=[root])
    status = app.project_status(roots=[root])

    assert status.state == "ready"
    search = app.search_code("locate feature", roots=[root])
    removal = app.remove_project(project.id)

    assert report.project_id == project.id
    assert status.file_count == 1
    assert status.chunk_count >= 1
    assert search.hits[0].symbol == "locate_feature"
    assert removal.removed is True
    assert app.list_projects() == []
    assert (root / ".ci-mcp" / "project.toml").exists()


def test_application_can_ensure_the_structural_index_without_a_semantic_search(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    report = app.ensure_reference_index(project.id)

    assert report.complete is True
    assert report.files_current == 1


def test_reference_query_reports_an_incomplete_structural_index(tmp_path: Path) -> None:
    """An uncoverable file degrades the answer instead of disabling the tool.

    Refusing outright meant one unparseable file anywhere in a repository made
    both reference tools permanently unusable, because such a file never gains
    coverage and so fails the same way on every later call.
    """

    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    with patch.object(
        app.indexer,
        "backfill_references",
        return_value=ReferenceBackfillReport(project_id=project.id, incomplete_paths=["broken.py"]),
    ):
        response = app.find_references(
            DeclarationSelector(
                project=project.id,
                path="main.py",
                qualified_symbol="answer",
            )
        )
        analysis = app.analyze_refactor(
            DeclarationSelector(
                project=project.id,
                path="main.py",
                qualified_symbol="answer",
            ),
            RenameOperation(new_name="result"),
        )

    limitation = next(item for item in response.limitations if item.code == "parse_error")
    assert "broken.py" in limitation.explanation
    assert analysis.completeness.state == "incomplete"


def test_modified_source_marks_an_index_stale(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("value = 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    assert app.project_is_stale(project.id) is False

    source.write_text("value = 200\n")

    assert app.project_is_stale(project.id) is True
    assert app.project_status(project.id).state == "stale"


def test_created_source_marks_an_index_stale(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("value = 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    (root / "added.py").write_text("added = True\n")

    assert app.project_is_stale(project.id) is True


def test_deleted_source_marks_an_index_stale(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("value = 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    source.unlink()

    assert app.project_is_stale(project.id) is True


def test_a_rejected_file_does_not_make_the_project_permanently_stale(tmp_path: Path) -> None:
    """A NUL-byte file used to vanish from storage while the scanner kept

    yielding it, so `current.keys() != existing.keys()` was true forever and
    every reference query triggered a full re-index under the global lock
    (S3). The file must instead persist as a tombstone row, so freshness
    settles once its size/mtime stop changing.
    """

    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    (root / "garbage.py").write_bytes(b"def broken(\x00):\n    pass\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    assert app.project_is_stale(project.id) is False
    assert app.project_status(project.id).state in {"ready", "partial"}

    with patch.object(app.indexer, "index", wraps=app.indexer.index) as index_spy:
        first = app.ensure_reference_index(project.id)
        second = app.ensure_reference_index(project.id)

    assert index_spy.call_count == 0
    assert first.files_current == second.files_current


def test_init_project_defaults_to_the_single_client_root(tmp_path: Path) -> None:
    root = tmp_path / "client-root"
    root.mkdir()
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )

    project = app.init_project(roots=[root])

    assert project.root == root.resolve()


def test_case_insensitive_root_alias_is_one_registration_and_one_lock(
    tmp_path: Path, case_insensitive_path_alias: Callable[[Path], Path]
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    alias = case_insensitive_path_alias(root)

    from_alias = app.init_project(alias)
    from_duplicate_roots = app.init_project(roots=[root, alias])

    assert from_alias.id == project.id
    assert from_duplicate_roots.id == project.id
    assert app._root_lock(root).lock_file == app._root_lock(alias).lock_file
    assert app.list_projects() == [project]


def test_discover_project_requires_marker_and_supported_source(tmp_path: Path) -> None:
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    source_only = tmp_path / "source-only"
    source_only.mkdir()
    (source_only / "main.py").write_text("value = 1\n")

    assert app.discover_project(source_only) is None
    assert not (source_only / ".ci-mcp").exists()

    (source_only / "pyproject.toml").write_text("[project]\nname = 'source-only'\n")

    project = app.discover_project(source_only)

    assert project is not None
    assert project.root == source_only.resolve()
    assert app.project_status(project.id).state == "pending"
    assert (source_only / ".ci-mcp" / "project.toml").exists()


def test_discover_project_accepts_javascript_manifest_and_existing_marker(tmp_path: Path) -> None:
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    javascript = tmp_path / "javascript"
    javascript.mkdir()
    (javascript / "package.json").write_text('{"name": "javascript"}\n')
    (javascript / "main.ts").write_text("export const value = 1\n")

    project = app.discover_project(javascript)

    assert project is not None
    empty = tmp_path / "empty"
    empty.mkdir()
    existing = app.init_project(empty)

    assert app.discover_project(empty) == existing


def test_application_supports_explicit_cross_project_search(tmp_path: Path) -> None:
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    ids = []
    for name in ("one", "two"):
        root = tmp_path / name
        root.mkdir()
        (root / "main.py").write_text(f"def {name}_feature():\n    return True\n")
        project = app.init_project(root)
        app.index_project(project.id)
        ids.append(project.id)

    selected = app.search_code("feature", projects=ids)
    all_projects = app.search_code("feature", all_projects=True)

    assert {hit.project_id for hit in selected.hits} == set(ids)
    assert {hit.project_id for hit in all_projects.hits} == set(ids)


def test_duplicate_live_project_marker_is_rejected_but_moved_checkout_is_adopted(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    duplicate = tmp_path / "duplicate"
    original.mkdir()
    (original / "main.py").write_text("value = 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(original)
    shutil.copytree(original, duplicate)

    with pytest.raises(CodeIndexingError) as raised:
        app.index_project(str(duplicate))
    assert raised.value.code is ErrorCode.PROJECT_ID_CONFLICT

    shutil.rmtree(original)
    report = app.index_project(str(duplicate))
    assert report.project_id == project.id
    assert app.list_projects()[0].root == duplicate.resolve()


def test_duplicate_legacy_project_marker_is_still_rejected(tmp_path: Path) -> None:
    original = tmp_path / "original"
    duplicate = tmp_path / "duplicate"
    original.mkdir()
    (original / "main.py").write_text("value = 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    app.init_project(original)
    (original / ".ci-mcp").rename(original / ".code-indexing-mcp")
    shutil.copytree(original, duplicate)

    with pytest.raises(CodeIndexingError) as raised:
        app.index_project(str(duplicate))

    assert raised.value.code is ErrorCode.PROJECT_ID_CONFLICT


def test_reregistering_a_known_project_preserves_state_and_still_validates_compatibility(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def locate_feature():\n    return True\n")
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=root)
    project = app.init_project(root)
    app.index_project(project.id)
    assert app.project_status(project.id).state == "ready"

    # Re-initializing (or re-discovering) an already-known, ready project
    # must not reset its state back to "pending".
    app.init_project(root)
    assert app.project_status(project.id).state == "ready"
    app.discover_project(root)
    assert app.project_status(project.id).state == "ready"

    other_app = Application(paths, embedder=OtherModelTinyEmbedder(), cwd=root)
    with pytest.raises(CodeIndexingError) as raised_init:
        other_app.init_project(root)
    assert raised_init.value.code is ErrorCode.INDEX_INCOMPATIBLE

    with pytest.raises(CodeIndexingError) as raised_discover:
        other_app.discover_project(root)
    assert raised_discover.value.code is ErrorCode.INDEX_INCOMPATIBLE


def test_the_application_resolves_a_backend_and_reports_it(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    settings = replace(IndexSettings.from_environment({}), embedding_accelerator=Accelerator.CPU)

    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path, settings=settings)

    status = app.model_status()
    assert app.backend_selection.accelerator is Accelerator.CPU
    assert status.embedding_model == "test/tiny"
    assert status.batch_calibration == "default"
    assert status.probe_cache_state == "not-applicable"


def test_an_explicit_batch_size_is_not_overridden_by_calibration(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    settings = replace(
        IndexSettings.from_environment({}),
        embedding_batch_size=12,
        embedding_batch_auto=False,
    )

    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path, settings=settings)

    assert app.model_status().batch_size == 12
    assert app.model_status().batch_calibration == "explicit"


def test_the_microbatch_token_budget_follows_the_memory_ceiling(tmp_path: Path) -> None:
    """The padded matrix a microbatch materializes is charged to the same
    ceiling the operator configured, so it has to move with it."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    settings = replace(
        IndexSettings.from_environment({}), index_memory_bytes=REFERENCE_MEMORY_BYTES * 2
    )

    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path, settings=settings)

    assert app.indexer.segment_plan.max_token_product == DEFAULT_MAX_TOKEN_PRODUCT * 2


def test_the_query_model_stays_in_process_regardless_of_the_backend(tmp_path: Path) -> None:
    """Acceleration targets passage indexing; a query must never wait on it."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    embedder = TinyEmbedder()

    app = Application(paths, embedder=embedder, cwd=tmp_path)

    assert app.search.embedder is embedder
    assert app.embedder is embedder


def test_a_backend_that_failed_once_is_not_attempted_again(tmp_path: Path) -> None:
    """Only successful probes are cached, so the failure has to be remembered.

    Without this the daemon would spawn a worker, load the model onto a dead
    device, and terminate it again before every single index run.
    """
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    settings = replace(IndexSettings.from_environment({}), embedding_accelerator=Accelerator.CPU)
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path, settings=settings)
    resolved = app.backend_selection
    assert app.effective_backend_selection is resolved

    app._remember_fallback(resolved.fell_back_to(CPU_BACKEND, "the device fell off the bus"))

    assert app.effective_backend_selection is not resolved
    assert app.effective_backend_selection.accelerator is Accelerator.CPU
    # backend_selection still records what capability alone resolved to; only
    # the effective selection carries the verdict a real run reached.
    assert app.backend_selection is resolved


def test_model_status_reports_a_runtime_fallback_rather_than_the_original_choice(
    tmp_path: Path,
) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    app._remember_fallback(
        app.backend_selection.fell_back_to(CPU_BACKEND, "the accelerator died on load")
    )

    status = app.model_status()
    assert status.resolved_accelerator == "cpu"
    assert status.fallback_reason == "the accelerator died on load"


def _prepared_cuda_environment(tmp_path: Path, **overrides: object) -> Path:
    """Write the record an installer leaves behind for a prepared CUDA machine."""
    interpreter = tmp_path / "venv-accel" / "python"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("", encoding="utf-8")
    settings: dict[str, object] = {
        "accelerator": Accelerator.CUDA,
        "interpreter": interpreter,
        "providers": ("CUDAExecutionProvider", "CPUExecutionProvider"),
        "runtime_version": "1.23.2",
        "driver_version": "550.54.14",
        "device": "cuda:0",
        "python_version": running_python_version(),
    }
    settings.update(overrides)
    write_environment(
        tmp_path / "data" / RECORD_FILENAME,
        AcceleratorEnvironment(**settings),  # type: ignore[arg-type]
    )
    return interpreter


def test_auto_selects_a_prepared_accelerator_this_process_cannot_execute_itself(
    tmp_path: Path,
) -> None:
    """The serving environment is CPU-only; the record is what makes CUDA reachable."""
    interpreter = _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")

    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    assert app.backend_selection.accelerator is Accelerator.CUDA
    status = app.model_status()
    assert status.resolved_accelerator == "cuda"
    assert status.accelerator_environment == str(interpreter)
    assert status.accelerator_prepared == "cuda"
    # Diagnostics the probe cache is keyed on come from the environment that was
    # probed, not from this process's own CPU runtime.
    assert status.driver_version == "550.54.14"
    assert status.runtime_version == "1.23.2"
    assert status.device == "cuda:0"


def _measure(app: Application, *, cpu: float, accelerator: float, load_ns: int) -> None:
    """Record the calibration a first run would have left behind."""
    app.probe_cache.store(
        app._build_probe_key(app.embedder),
        batch_size=8,
        dimension=app.embedder.dimension,
        characters_per_second=accelerator,
        load_ns=load_ns,
    )
    app.probe_cache.store(
        app._cpu_probe_key(),
        batch_size=1,
        dimension=app.embedder.dimension,
        characters_per_second=cpu,
        load_ns=0,
    )


def test_the_crossover_is_computed_from_both_measured_backends(tmp_path: Path) -> None:
    _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    _measure(app, cpu=1_000.0, accelerator=2_000.0, load_ns=2_000_000_000)

    status = app.model_status()
    assert app.crossover_characters() == 4_000
    assert status.crossover_characters == 4_000
    assert status.accelerator_characters_per_second == 2_000.0
    assert status.cpu_characters_per_second == 1_000.0
    assert status.accelerator_load_ms == 2_000
    # The measured batch size is adopted by the next process to start, which is
    # where the segment plan for a run is built.
    assert (
        Application(paths, embedder=TinyEmbedder(), cwd=tmp_path).model_status().batch_calibration
        == "measured"
    )


def test_an_unmeasured_accelerator_has_no_crossover_and_starts_at_once(
    tmp_path: Path,
) -> None:
    """Nothing has been measured, so there is no size to defer below and the
    accelerator is used exactly as it was before any of this existed."""
    _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")

    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    assert app.crossover_characters() == 0
    assert app.model_status().crossover_characters is None


def test_an_explicit_crossover_wins_over_the_measured_one(tmp_path: Path) -> None:
    _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    settings = replace(
        IndexSettings.from_environment({}),
        embedding_crossover_characters=99,
        embedding_crossover_auto=False,
    )
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path, settings=settings)

    _measure(app, cpu=1_000.0, accelerator=2_000.0, load_ns=2_000_000_000)

    assert app.crossover_characters() == 99


def test_strict_mode_refuses_to_defer_to_cpu_at_all(tmp_path: Path) -> None:
    """Strict mode is for a caller who would rather fail than index quietly on
    CPU, and a deferral is quiet CPU indexing no degradation reports."""
    _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    settings = replace(IndexSettings.from_environment({}), embedding_strict=True)
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path, settings=settings)

    _measure(app, cpu=1_000.0, accelerator=2_000.0, load_ns=2_000_000_000)

    # The measurement still stands and is still reported; only the deferral it
    # would otherwise drive is refused.
    assert app.model_status().crossover_characters == 4_000
    assert app.crossover_characters() == 0


def test_an_accelerator_that_lost_to_cpu_recommends_the_override(tmp_path: Path) -> None:
    """There is no run size at which it wins, so the useful thing to report is
    not a threshold but that this machine should stop preparing it."""
    _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    _measure(app, cpu=2_000.0, accelerator=1_000.0, load_ns=2_000_000_000)

    status = app.model_status()
    assert status.crossover_characters is None
    assert status.recommended_override is not None
    assert "CODE_INDEXING_EMBED_ACCELERATOR=cpu" in status.recommended_override
    # The same None reaches the session, which is what stops it starting a
    # backend no run is large enough to justify. Reporting the largest
    # admissible run instead would name a threshold and defer against it.
    assert app.crossover_characters() is None


def test_a_batch_size_a_ceiling_overrun_reduced_is_reported_as_reduced(
    tmp_path: Path,
) -> None:
    _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    app.probe_cache.store(
        app._build_probe_key(app.embedder),
        batch_size=1,
        dimension=app.embedder.dimension,
        characters_per_second=500.0,
        load_ns=1,
        limited_by="memory",
    )

    rebuilt = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    status = rebuilt.model_status()
    assert status.batch_size == 1
    assert status.batch_calibration == "reduced"
    assert status.recommended_override is not None
    assert "CODE_INDEXING_EMBED_MEMORY_MB" in status.recommended_override


def test_a_prepared_accelerator_runs_in_its_own_interpreter(tmp_path: Path) -> None:
    interpreter = _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    launcher = app._accelerator_launcher(app.backend_selection.descriptor)

    assert isinstance(launcher, ExternalInterpreterLauncher)
    assert launcher.executable == interpreter
    # The fallback is what a failed accelerator falls back *to*, so it must not
    # depend on the environment that just failed.
    assert not isinstance(default_launcher(), ExternalInterpreterLauncher)


def test_a_backend_this_process_already_offers_needs_no_second_environment(
    tmp_path: Path,
) -> None:
    """An explicit Core ML override runs in the serving environment's own runtime."""
    _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    in_process = replace(app.backend_selection.descriptor, provider=app.serving_providers[0])

    assert not isinstance(app._accelerator_launcher(in_process), ExternalInterpreterLauncher)


def test_a_refused_record_explains_the_cpu_outcome(tmp_path: Path) -> None:
    """ "No accelerator is prepared" is the wrong diagnosis when one nearly was."""
    _prepared_cuda_environment(tmp_path, python_version="3.7")
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")

    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    assert app.backend_selection.accelerator is Accelerator.CPU
    status = app.model_status()
    assert "built for Python 3.7" in (status.fallback_reason or "")
    assert status.accelerator_environment is None
    assert status.accelerator_prepared is None


def test_a_record_offers_only_the_accelerator_it_was_probed_for(tmp_path: Path) -> None:
    """A provider the prepared runtime happens to ship is not evidence of anything.

    A CUDA environment's ONNX Runtime lists more providers than CUDA. Offering
    all of them would select a backend no probe ever exercised there, and then
    describe it with a record that cannot say which device or driver it ran on.
    """
    _prepared_cuda_environment(
        tmp_path, providers=("CUDAExecutionProvider", "MIGraphXExecutionProvider")
    )
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    settings = replace(
        IndexSettings.from_environment({}), embedding_accelerator=Accelerator.MIGRAPHX
    )

    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path, settings=settings)

    assert app.backend_selection.accelerator is Accelerator.CPU
    assert app.backend_selection.honored is False
    assert "not among the execution providers" in (app.model_status().fallback_reason or "")
    # The record's own accelerator is still reachable; only the rest is not.
    assert Application(
        paths, embedder=TinyEmbedder(), cwd=tmp_path
    ).backend_selection.accelerator is (Accelerator.CUDA)


def test_storage_status_reports_registry_project_and_totals(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def locate_feature():\n    return True\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    status = app.storage_status()

    assert status.schema_version == 1
    assert status.consistent is True
    assert status.overlap_warnings == []
    assert status.worktree_warnings == []
    assert status.registry.name == "projects"
    assert status.registry.row_count == 1
    assert status.registry.logical_bytes > 0
    assert len(status.projects) == 1
    project_stats = status.projects[0]
    assert project_stats.project.id == project.id
    assert {table.name for table in project_stats.tables} == {"files", "chunks", "references"}
    assert project_stats.partition_physical_bytes > 0
    assert project_stats.consistent is True
    assert status.physical_bytes_total >= (
        status.registry.physical_bytes + project_stats.partition_physical_bytes
    )

    scoped = app.storage_status(project.id)

    assert [entry.project.id for entry in scoped.projects] == [project.id]


def test_storage_status_reports_registered_root_overlaps(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    first = tmp_path / "first"
    first.mkdir()
    app.init_project(first)
    nested = first / "nested"
    nested.mkdir()
    app.store.upsert_project(
        ProjectInfo(id="nested-id", name="nested", root=nested), model_id="test/model"
    )

    status = app.storage_status()

    assert any("nested" in warning for warning in status.overlap_warnings)
    assert len(status.projects) == 2


def test_storage_status_raises_for_an_unknown_project(tmp_path: Path) -> None:
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )

    with pytest.raises(CodeIndexingError) as raised:
        app.storage_status("no-such-project")

    assert raised.value.code is ErrorCode.PROJECT_NOT_FOUND
