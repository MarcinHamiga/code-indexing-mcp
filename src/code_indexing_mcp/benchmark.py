"""Deterministic end-to-end indexing benchmarks with JSON-ready results."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from . import update_check
from .application import Application, RuntimePaths
from .errors import CodeIndexingError, ErrorCode
from .models import IndexReport, ProjectInfo, StorageStatus
from .settings import IndexSettings

# The repeated_edits scenario applies this many consecutive edits to one file,
# indexing after each one, so per-edit write amplification shows up as version
# growth over a meaningful sample.
REPEATED_EDITS = 100


class IndexBenchmarkApplication(Protocol):
    def init_project(self, path: Path) -> ProjectInfo: ...

    def index_project(self, project: str, *, force: bool = False) -> IndexReport: ...

    def storage_status(self, project: str | None = None) -> StorageStatus: ...


def write_benchmark_corpus(root: Path, *, files: int = 128, functions_per_file: int = 2) -> int:
    """Write a fixed Python corpus and return its source byte count."""
    if files < 1 or functions_per_file < 1:
        raise ValueError("benchmark corpus dimensions must be positive")
    root.mkdir(parents=True, exist_ok=False)
    total = 0
    for file_index in range(files):
        source = "".join(
            (
                f"def function_{file_index:04d}_{function_index:04d}(value: int) -> int:\n"
                f"    return value + {file_index + function_index}\n\n"
            )
            for function_index in range(functions_per_file)
        )
        encoded = source.encode()
        (root / f"module_{file_index:04d}.py").write_bytes(encoded)
        total += len(encoded)
    return total


def _storage_snapshot(app: IndexBenchmarkApplication, project_id: str) -> dict[str, Any]:
    """Return *project_id*'s storage statistics as a JSON-ready dict."""
    status = app.storage_status(project_id)
    entry = next((stats for stats in status.projects if stats.project.id == project_id), None)
    return entry.model_dump(mode="json") if entry is not None else {}


def _measure(
    action: Callable[[], IndexReport],
    *,
    storage_after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.monotonic_ns()
    report = action()
    wall_ms = (time.monotonic_ns() - started) / 1_000_000
    measured_ms = float(report.duration_ms) if report.duration_ms > 0 else max(wall_ms, 0.001)
    throughput = report.embedded_chunks * 1_000 / measured_ms
    result: dict[str, Any] = {
        "wall_ms": round(wall_ms, 3),
        "chunks_per_second": round(throughput, 3),
        # This run's own staged structural rows (T1) -- not a whole-project
        # table read, which would report the same total for every scenario
        # regardless of how much work that scenario actually did.
        "structural_records": report.staged_reference_rows,
        # Reference extraction's own timing (T1), separate from
        # `parse_duration_ms`, which also covers parsing and chunking.
        "reference_extraction_duration_ms": report.reference_extraction_duration_ms or 0,
        "report": report.model_dump(mode="json"),
    }
    if storage_after is not None:
        result["storage_after"] = storage_after
    return result


def run_index_benchmark(app: IndexBenchmarkApplication, root: Path) -> dict[str, Any]:
    """Run the storage-growth scenarios against one isolated application.

    Every scenario records the project's storage statistics after it finishes,
    so table-version deltas and physical growth are observable per scenario
    (contract version 2). ``post_maintenance`` currently captures the
    post-deletion snapshot; the maintenance release extends it with real
    cleanup work without changing the JSON contract.
    """
    project = app.init_project(root)
    scenarios: dict[str, dict[str, Any]] = {}

    def snapshot() -> dict[str, Any]:
        return _storage_snapshot(app, project.id)

    storage_baseline = snapshot()

    scenarios["cold_start"] = _measure(
        lambda: app.index_project(project.id, force=True), storage_after=snapshot()
    )
    scenarios["no_op"] = _measure(
        lambda: app.index_project(project.id, force=False), storage_after=snapshot()
    )

    edited = root / "module_0000.py"
    with edited.open("a", encoding="utf-8") as stream:
        stream.write("\ndef phase_2_single_edit_marker(value: int) -> int:\n    return value + 1\n")
    scenarios["single_file_edit"] = _measure(
        lambda: app.index_project(project.id, force=False), storage_after=snapshot()
    )

    repeated_started = time.monotonic_ns()
    for edit_index in range(REPEATED_EDITS):
        with edited.open("a", encoding="utf-8") as stream:
            stream.write(
                f"\ndef repeated_edit_marker_{edit_index:04d}(value: int) -> int:\n"
                f"    return value + {edit_index}\n"
            )
        app.index_project(project.id, force=False)
    scenarios["repeated_edits"] = {
        "wall_ms": round((time.monotonic_ns() - repeated_started) / 1_000_000, 3),
        "edits": REPEATED_EDITS,
        "storage_after": snapshot(),
    }

    scenarios["forced_reindex"] = _measure(
        lambda: app.index_project(project.id, force=True), storage_after=snapshot()
    )

    removed_single = _unlink_if_present(root / "module_0001.py")
    scenarios["single_file_deletion"] = _measure(
        lambda: app.index_project(project.id, force=False), storage_after=snapshot()
    )
    scenarios["single_file_deletion"]["removed_files"] = removed_single

    removed_group = 0
    for deleted_index in range(2, 10):
        removed_group += _unlink_if_present(root / f"module_{deleted_index:04d}.py")
    scenarios["many_file_deletions"] = _measure(
        lambda: app.index_project(project.id, force=False), storage_after=snapshot()
    )
    scenarios["many_file_deletions"]["removed_files"] = removed_group

    maintenance_started = time.monotonic_ns()
    scenarios["post_maintenance"] = {
        "wall_ms": round((time.monotonic_ns() - maintenance_started) / 1_000_000, 3),
        "storage_after": snapshot(),
    }
    return {
        "schema_version": 2,
        "storage_baseline": storage_baseline,
        "scenarios": scenarios,
    }


def _unlink_if_present(path: Path) -> int:
    """Delete *path*, returning 1 when it existed. Bounds a deletion group to the corpus."""
    try:
        path.unlink()
    except FileNotFoundError:
        return 0
    return 1


def _run_in_workspace(
    paths: RuntimePaths,
    workspace: Path,
    *,
    files: int,
    functions_per_file: int,
    batch_size: int,
) -> dict[str, Any]:
    root = workspace / "corpus"
    source_bytes = write_benchmark_corpus(root, files=files, functions_per_file=functions_per_file)
    settings = replace(
        IndexSettings.from_environment(),
        embedding_batch_size=batch_size,
        index_execution="in-process",
        broker_mode="off",
    )
    app = Application(
        RuntimePaths(data=workspace / "data", cache=paths.cache),
        cwd=root,
        settings=settings,
    )
    result = run_index_benchmark(app, root)
    result.update(
        {
            "model_id": app.embedder.model_id,
            "embedding_backend": "cpu",
            "embedding_batch_size": batch_size,
            "corpus": {
                "files": files,
                "functions_per_file": functions_per_file,
                "source_bytes": source_bytes,
            },
            "revision": update_check.checkout_head(Path(__file__).resolve().parents[2]),
        }
    )
    return result


def run_index_benchmark_command(
    paths: RuntimePaths,
    *,
    files: int,
    functions_per_file: int,
    batch_size: int,
    work_dir: Path | None,
) -> dict[str, Any]:
    """Create an isolated workspace and run the CLI benchmark."""
    if files < 1 or functions_per_file < 1:
        raise CodeIndexingError(
            ErrorCode.INVALID_CONFIGURATION,
            "Benchmark corpus dimensions must be positive",
        )
    if not 1 <= batch_size <= 256:
        raise CodeIndexingError(
            ErrorCode.INVALID_CONFIGURATION,
            "Benchmark batch size must be from 1 to 256",
        )
    if work_dir is not None:
        workspace = work_dir.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        if (workspace / "corpus").exists() or (workspace / "data").exists():
            raise CodeIndexingError(
                ErrorCode.INVALID_CONFIGURATION,
                f"Benchmark work directory is not fresh: {workspace}",
            )
        return _run_in_workspace(
            paths,
            workspace,
            files=files,
            functions_per_file=functions_per_file,
            batch_size=batch_size,
        )
    with tempfile.TemporaryDirectory(prefix="code-indexing-mcp-index-benchmark-") as temporary:
        return _run_in_workspace(
            paths,
            Path(temporary),
            files=files,
            functions_per_file=functions_per_file,
            batch_size=batch_size,
        )
