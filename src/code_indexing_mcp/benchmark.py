"""Deterministic end-to-end indexing benchmarks with JSON-ready results."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from .application import Application, RuntimePaths
from .errors import CodeIndexingError, ErrorCode
from .models import IndexReport, ProjectInfo
from .settings import IndexSettings


class IndexBenchmarkApplication(Protocol):
    def init_project(self, path: Path) -> ProjectInfo: ...

    def index_project(self, project: str, *, force: bool = False) -> IndexReport: ...


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


def _measure(
    action: Callable[[], IndexReport], structural_record_count: Callable[[str], int]
) -> dict[str, Any]:
    started = time.monotonic_ns()
    report = action()
    wall_ms = (time.monotonic_ns() - started) / 1_000_000
    measured_ms = float(report.duration_ms) if report.duration_ms > 0 else max(wall_ms, 0.001)
    throughput = report.embedded_chunks * 1_000 / measured_ms
    return {
        "wall_ms": round(wall_ms, 3),
        "chunks_per_second": round(throughput, 3),
        "structural_records": structural_record_count(report.project_id),
        "reference_extraction_duration_ms": report.parse_duration_ms or report.parse_ms or 0,
        "report": report.model_dump(mode="json"),
    }


def _structural_record_count(app: IndexBenchmarkApplication, project_id: str) -> int:
    """Read persisted structural facts; indexing already extracted them in its parse pass."""
    store = getattr(app, "store", None)
    if store is None:
        return 0
    return len(store.list_reference_records(project_id))


def run_index_benchmark(app: IndexBenchmarkApplication, root: Path) -> dict[str, Any]:
    """Run the four Phase 1 scenarios against one isolated application."""
    project = app.init_project(root)
    scenarios: dict[str, dict[str, Any]] = {}

    def records(project_id: str) -> int:
        return _structural_record_count(app, project_id)

    scenarios["cold_start"] = _measure(lambda: app.index_project(project.id, force=True), records)
    scenarios["warm_index"] = _measure(lambda: app.index_project(project.id, force=True), records)

    incremental = root / "module_0000.py"
    with incremental.open("a", encoding="utf-8") as stream:
        stream.write("\ndef phase_1_incremental_marker(value: int) -> int:\n    return value + 1\n")
    scenarios["incremental_index"] = _measure(
        lambda: app.index_project(project.id, force=False), records
    )
    scenarios["forced_reindex"] = _measure(
        lambda: app.index_project(project.id, force=True), records
    )
    return {"schema_version": 1, "scenarios": scenarios}


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
