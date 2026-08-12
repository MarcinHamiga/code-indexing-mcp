from pathlib import Path

import benchmark_index_memory

from code_indexing_mcp.benchmark import (
    REPEATED_EDITS,
    _duration_summary,
    run_index_benchmark,
    write_benchmark_corpus,
)
from code_indexing_mcp.models import (
    IndexReport,
    MaintenanceReport,
    ProjectInfo,
    ProjectStorageStats,
    StorageStatus,
    TableStorageStats,
)
from code_indexing_mcp.settings import IndexSettings


class BenchmarkApplication:
    def __init__(self, root: Path, *, duration_ms: int = 100) -> None:
        self.root = root
        self.duration_ms = duration_ms
        self.force_calls: list[bool] = []
        self.storage_calls: list[str] = []
        self.maintenance_calls: list[tuple[str | None, bool]] = []

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
            duration_ms=self.duration_ms,
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

    def maintain_storage(
        self, project: str | None = None, *, wait_for_lock: bool = False
    ) -> MaintenanceReport:
        self.maintenance_calls.append((project, wait_for_lock))
        return MaintenanceReport(
            trigger="manual",
            dry_run=False,
            retention_hours=24,
            started_at="2026-08-11T00:00:00+00:00",
            finished_at="2026-08-11T00:00:01+00:00",
            duration_ms=1_000,
            registry_status="ok",
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
    maintenance = payload["scenarios"]["post_maintenance"]
    assert maintenance["wall_ms"] >= 0
    assert maintenance["report"]["duration_ms"] == 1_000
    assert app.maintenance_calls == [("benchmark-project", True)]
    assert payload["scenarios"]["cold_start"]["includes_embedder_warmup"] is True
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


def test_the_benchmark_derives_the_numbers_it_publishes(tmp_path: Path) -> None:
    """The reported metrics must be the arithmetic they claim, not just present.

    Scenario ordering can be right while every published number is wrong: a
    swapped numerator or a milliseconds-to-seconds slip is invisible unless the
    derived values are pinned against the report they came from.
    """
    root = tmp_path / "corpus"
    write_benchmark_corpus(root, files=8, functions_per_file=2)

    payload = run_index_benchmark(BenchmarkApplication(root), root)

    for name in ("cold_start", "no_op", "single_file_edit", "forced_reindex"):
        scenario = payload["scenarios"][name]
        # 8 chunks over the report's own 100 ms is 80 chunks/second.
        assert scenario["reported_duration_ms"] == 100
        assert scenario["chunks_per_second"] == 80.0
        # Structural rows are this run's staged rows, not a whole-table count.
        assert scenario["structural_records"] == 12
        assert scenario["reference_extraction_duration_ms"] == 12
        # Wall time is measured independently of the report's own duration, so
        # a fake that never sleeps must not inherit the reported 100 ms.
        assert 0 <= scenario["wall_ms"] < 100
        assert scenario["report"]["embedded_chunks"] == 8


def test_throughput_is_null_when_the_indexer_reports_no_duration(tmp_path: Path) -> None:
    """Wall time must not stand in for the indexer's own clock.

    Substituting it would publish one field name computed two different ways,
    so runs would be compared against each other on different measurements.
    """
    root = tmp_path / "corpus"
    write_benchmark_corpus(root, files=4, functions_per_file=1)

    payload = run_index_benchmark(BenchmarkApplication(root, duration_ms=0), root)

    scenario = payload["scenarios"]["cold_start"]
    assert scenario["reported_duration_ms"] == 0
    assert scenario["chunks_per_second"] is None
    assert scenario["wall_ms"] >= 0


def test_repeated_edits_reports_a_distribution_not_only_a_total(tmp_path: Path) -> None:
    """100 edits is a real sample; the total alone cannot show per-edit drift."""
    root = tmp_path / "corpus"
    write_benchmark_corpus(root, files=4, functions_per_file=1)

    payload = run_index_benchmark(BenchmarkApplication(root), root)

    summary = payload["scenarios"]["repeated_edits"]["per_edit_ms"]
    assert summary["count"] == REPEATED_EDITS
    assert summary["min_ms"] <= summary["median_ms"] <= summary["p95_ms"] <= summary["max_ms"]
    # Head and tail means make write amplification visible: a last decile well
    # above the first is growth, which the aggregate total hides entirely.
    assert summary["first_decile_mean_ms"] >= 0
    assert summary["last_decile_mean_ms"] >= 0
    # The summary covers the indexing inside the scenario's own wall time; the
    # tolerance absorbs per-sample rounding, not a real discrepancy.
    assert summary["total_ms"] <= payload["scenarios"]["repeated_edits"]["wall_ms"] + 0.1


def test_duration_summary_computes_order_statistics() -> None:
    """Pin the summary arithmetic directly, free of any timing jitter."""
    summary = _duration_summary([float(value) for value in range(1, 21)])

    assert summary["count"] == 20
    assert summary["total_ms"] == 210.0
    assert summary["min_ms"] == 1.0
    assert summary["max_ms"] == 20.0
    assert summary["median_ms"] == 10.5
    # Nearest-rank p95 of 20 ordered samples is the 19th.
    assert summary["p95_ms"] == 19.0
    assert summary["first_decile_mean_ms"] == 1.5
    assert summary["last_decile_mean_ms"] == 19.5
    assert _duration_summary([]) == {"count": 0}


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
