from pathlib import Path

import benchmark_index_memory

from code_indexing_mcp.benchmark import REPEATED_EDITS, run_index_benchmark, write_benchmark_corpus
from code_indexing_mcp.models import (
    IndexReport,
    ProjectInfo,
    ProjectStorageStats,
    StorageStatus,
    TableStorageStats,
)
from code_indexing_mcp.settings import IndexSettings


class BenchmarkApplication:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.force_calls: list[bool] = []
        self.storage_calls: list[str] = []

    def init_project(self, path: Path) -> ProjectInfo:
        assert path == self.root
        return ProjectInfo(id="benchmark-project", name="benchmark", root=path)

    def index_project(self, project: str, *, force: bool = False) -> IndexReport:
        assert project == "benchmark-project"
        self.force_calls.append(force)
        return IndexReport(
            project_id=project,
            discovered_files=4,
            indexed_files=1,
            parsed_files=1,
            embedded_chunks=8,
            duration_ms=100,
            embedding_backend="cpu",
            embedding_batch_size=8,
            staged_reference_rows=12,
            reference_extraction_duration_ms=12,
        )

    def storage_status(self, project: str | None = None) -> StorageStatus:
        self.storage_calls.append(project or "")
        entry = ProjectStorageStats(
            project=ProjectInfo(id="benchmark-project", name="benchmark", root=self.root),
            snapshot_at="2026-08-11T00:00:00+00:00",
            tables=[],
            # A distinguishable value per call, so a scenario that forgets to
            # snapshot cannot pass for one that did (T3).
            partition_physical_bytes=len(self.storage_calls),
            consistent=True,
        )
        return StorageStatus(
            snapshot_at="2026-08-11T00:00:00+00:00",
            registry=TableStorageStats(name="projects"),
            projects=[entry],
        )


def test_benchmark_runs_the_storage_growth_scenarios(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    write_benchmark_corpus(root, files=8, functions_per_file=2)
    app = BenchmarkApplication(root)

    payload = run_index_benchmark(app, root)

    assert app.force_calls == [True, False, False] + [False] * 100 + [True, False, False]
    assert list(payload["scenarios"]) == [
        "cold_start",
        "no_op",
        "single_file_edit",
        "repeated_edits",
        "forced_reindex",
        "single_file_deletion",
        "many_file_deletions",
        "post_maintenance",
    ]
    assert payload["schema_version"] == 2
    # The baseline is captured before any index work and every scenario records
    # its own post-run storage snapshot, so version deltas and physical growth
    # are computable per scenario from the contract alone.
    assert payload["storage_baseline"]["partition_physical_bytes"] == 1
    # The snapshot must be taken after the scenario's index work: cold_start's
    # storage_after is exactly one collection newer than the pre-index baseline,
    # not equal to it (the baseline and a pre-action snapshot would be the same).
    assert (
        payload["scenarios"]["cold_start"]["storage_after"]["partition_physical_bytes"]
        == payload["storage_baseline"]["partition_physical_bytes"] + 1
    )
    for name in (
        "cold_start",
        "no_op",
        "single_file_edit",
        "repeated_edits",
        "forced_reindex",
        "single_file_deletion",
        "many_file_deletions",
        "post_maintenance",
    ):
        after = payload["scenarios"][name]["storage_after"]
        assert after["project"]["id"] == "benchmark-project"
        assert after["partition_physical_bytes"] > 0
        assert after["consistent"] is True
    assert payload["scenarios"]["repeated_edits"]["edits"] == REPEATED_EDITS
    # Storage is snapshotted once per scenario; the 100 edits index but do not
    # each get their own snapshot, so the counter stays proportional to the
    # scenario count rather than the edit count.
    assert len(app.storage_calls) == 9
    # The corpus mutations the scenarios make are real files on disk: the
    # edit markers land in module_0000.py, and each deletion scenario removes
    # its bounded group.
    edited = (root / "module_0000.py").read_text()
    assert "phase_2_single_edit_marker" in edited
    assert "repeated_edit_marker_0099" in edited
    assert not (root / "module_0001.py").exists()
    for deleted_index in range(2, 10):
        assert not (root / f"module_{deleted_index:04d}.py").exists()


def test_benchmark_corpus_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_bytes = write_benchmark_corpus(first, files=3, functions_per_file=4)
    second_bytes = write_benchmark_corpus(second, files=3, functions_per_file=4)

    assert first_bytes == second_bytes
    assert [path.read_bytes() for path in sorted(first.iterdir())] == [
        path.read_bytes() for path in sorted(second.iterdir())
    ]


def test_the_benchmark_pins_the_memory_ceiling_it_reports(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """--memory-mb must win over whatever the developer's shell exports.

    The child inherits os.environ, and CODE_INDEXING_EMBED_MEMORY_MB outranks the
    legacy name -- so an exported ceiling would silently replace the requested
    one while the results still claimed the value that was asked for.
    """
    monkeypatch.setenv("CODE_INDEXING_EMBED_MEMORY_MB", "9999")
    monkeypatch.setenv("CODE_INDEXING_INDEX_MEMORY_MB", "8888")

    environment = benchmark_index_memory._environment(
        data_directory=Path("/tmp/data"),
        cache_directory=Path("/tmp/cache"),
        batch_size=1,
        memory_mb=2048,
        offline=True,
    )

    assert IndexSettings.from_environment(environment).index_memory_bytes == 2048 * 1024 * 1024


def test_the_benchmark_leaves_no_inherited_ceiling_behind(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Without --memory-mb the run uses the default, not the shell's value."""
    monkeypatch.setenv("CODE_INDEXING_EMBED_MEMORY_MB", "9999")

    environment = benchmark_index_memory._environment(
        data_directory=Path("/tmp/data"),
        cache_directory=Path("/tmp/cache"),
        batch_size=1,
        memory_mb=None,
        offline=True,
    )

    assert "CODE_INDEXING_EMBED_MEMORY_MB" not in environment
    assert "CODE_INDEXING_INDEX_MEMORY_MB" not in environment
