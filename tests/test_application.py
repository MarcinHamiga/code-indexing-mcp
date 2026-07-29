import shutil
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from incode_mcp.accelerator_env import (
    RECORD_FILENAME,
    AcceleratorEnvironment,
    running_python_version,
    write_environment,
)
from incode_mcp.application import Application, RuntimePaths
from incode_mcp.backends import CPU_BACKEND, Accelerator
from incode_mcp.embedding_worker import default_launcher
from incode_mcp.errors import ErrorCode, IncodeError
from incode_mcp.settings import IndexSettings
from incode_mcp.worker_launcher import ExternalInterpreterLauncher


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

    with pytest.raises(IncodeError) as raised:
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
    (original / ".ci-mcp").rename(original / ".incode")
    shutil.copytree(original, duplicate)

    with pytest.raises(IncodeError) as raised:
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
    with pytest.raises(IncodeError) as raised_init:
        other_app.init_project(root)
    assert raised_init.value.code is ErrorCode.INDEX_INCOMPATIBLE

    with pytest.raises(IncodeError) as raised_discover:
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
