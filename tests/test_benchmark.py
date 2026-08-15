from pathlib import Path

import benchmark_index_memory
import numpy as np
import pytest

from code_indexing_mcp.application import RuntimePaths
from code_indexing_mcp.benchmark import (
    REPEATED_EDITS,
    RETRIEVAL_TOPICS,
    SEARCH_ITERATIONS,
    _directory_physical_bytes,
    _duration_summary,
    build_retrieval_corpus,
    run_index_benchmark,
    run_precision_benchmark,
    run_precision_benchmark_command,
    run_search_benchmark,
    write_benchmark_corpus,
)
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.models import (
    IndexReport,
    MaintenanceReport,
    ProjectInfo,
    ProjectStorageStats,
    SearchHit,
    SearchResponse,
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


class SearchBenchmarkApplication:
    def __init__(self) -> None:
        self.init_calls: list[Path] = []
        self.search_calls: list[int] = []

    def init_project(self, path: Path) -> ProjectInfo:
        self.init_calls.append(path)
        return ProjectInfo(id=f"id-{path.name}", name=path.name, root=path)

    def index_project(self, project: str, *, force: bool = False) -> IndexReport:
        assert force is True
        return IndexReport(project_id=project, duration_ms=1)

    def search_code(self, query: str, *, projects: list[str], limit: int = 8) -> SearchResponse:
        self.search_calls.append(len(projects))
        return SearchResponse(
            query=query,
            hits=[
                SearchHit(
                    chunk_id=f"chunk-{index}",
                    project_id=projects[index % len(projects)],
                    project_name=projects[index % len(projects)],
                    path="mod.py",
                    language="python",
                    kind="function",
                    start_line=index,
                    end_line=index,
                    score=1.0 - index * 0.01,
                    snippet="",
                )
                for index in range(min(limit, len(projects)))
            ],
        )


class FlakySearchBenchmarkApplication(SearchBenchmarkApplication):
    def search_code(self, query: str, *, projects: list[str], limit: int = 8) -> SearchResponse:
        response = super().search_code(query, projects=projects, limit=limit)
        if len(self.search_calls) % 2 == 0:
            response = SearchResponse(query=response.query, hits=list(reversed(response.hits)))
        return response


def test_search_benchmark_measures_one_eight_and_fifty_project_scopes(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / f"p{index}" for index in range(50)]
    app = SearchBenchmarkApplication()

    payload = run_search_benchmark(app, roots)

    assert payload["schema_version"] == 1
    assert payload["projects"] == 50
    assert list(payload["scopes"]) == ["1", "8", "50"]
    for scope in ("1", "8", "50"):
        scenario = payload["scopes"][scope]
        assert scenario["projects"] == int(scope)
        assert scenario["latency_ms"]["count"] == SEARCH_ITERATIONS
        assert scenario["deterministic"] is True
        assert len(scenario["top_hits"]) == min(int(scope), 8)
    # Every scope times its iterations and then pins ordering twice more.
    assert app.search_calls == [1] * 5 + [8] * 5 + [50] * 5


def test_search_benchmark_caps_scopes_to_available_projects(tmp_path: Path) -> None:
    app = SearchBenchmarkApplication()

    payload = run_search_benchmark(app, [tmp_path / f"p{index}" for index in range(3)])

    assert list(payload["scopes"]) == ["1", "3"]
    assert payload["scopes"]["3"]["projects"] == 3


def test_search_benchmark_reports_non_deterministic_ranking(tmp_path: Path) -> None:
    app = FlakySearchBenchmarkApplication()

    payload = run_search_benchmark(app, [tmp_path / "p0", tmp_path / "p1"], iterations=1)

    # The two-project scope flips its hit order between the pinning runs, so the
    # scenario must report the ranking as non-deterministic rather than silently
    # publishing the last run's order.
    assert payload["scopes"]["2"]["deterministic"] is False


class TopicPrecisionEmbedder:
    """Topic axis plus a deterministic within-topic jitter axis; no model.

    Passages embed positionally exactly as ``build_retrieval_corpus`` assigns
    topics, and queries embed onto their topic's axis, so the float32-exact
    reference ranking is the topic's passages ordered by ascending jitter --
    a ranking whose margins dwarf float16 rounding.
    """

    model_id = "test/precision"
    dimension = 16

    def _vector(self, topic: int, jitter: float) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        vector[topic] = 1.0
        vector[8 + topic] = jitter
        return vector.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [
            self._vector(index % len(RETRIEVAL_TOPICS), ((index % 9) + 1) * 0.1)
            for index in range(len(texts))
        ]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        topic = next(
            index
            for index, terms in enumerate(RETRIEVAL_TOPICS)
            if any(term in lowered for term in terms)
        )
        return self._vector(topic, 0.0)


def test_retrieval_corpus_is_deterministic_with_sound_judgments() -> None:
    first_corpus, first_queries = build_retrieval_corpus(passages=40)
    second_corpus, second_queries = build_retrieval_corpus(passages=40)

    assert first_corpus == second_corpus
    assert first_queries == second_queries
    ids = {passage.chunk_id for passage in first_corpus}
    assert len(ids) == 40
    assert len(first_queries) == len(RETRIEVAL_TOPICS)
    for query in first_queries:
        assert query.text
        assert query.relevant
        assert set(query.relevant) <= ids


def test_retrieval_corpus_rejects_too_few_passages() -> None:
    with pytest.raises(ValueError):
        build_retrieval_corpus(passages=len(RETRIEVAL_TOPICS) - 1)


def test_precision_command_rejects_passages_below_the_corpus_minimum(tmp_path: Path) -> None:
    # The command must reject what build_retrieval_corpus cannot build with
    # the same INVALID_CONFIGURATION error every other bad CLI value raises,
    # not a bare ValueError traceback.
    with pytest.raises(CodeIndexingError) as caught:
        run_precision_benchmark_command(
            paths=RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
            passages=len(RETRIEVAL_TOPICS) - 1,
            iterations=1,
            recall_floor=0.99,
            rank_floor=0.95,
            work_dir=None,
        )

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION
    assert str(len(RETRIEVAL_TOPICS)) in str(caught.value)


def test_precision_experiment_contract_and_self_consistency(tmp_path: Path) -> None:
    report = run_precision_benchmark(
        TopicPrecisionEmbedder(),
        tmp_path,
        passages=40,
        top_k=5,
        iterations=2,
    )

    assert report["schema_version"] == 1
    assert set(report["variants"]) == {
        "float32_exact",
        "float32_hnsw_sq8",
        "float16_exact",
        "float16_hnsw_sq8",
    }
    assert report["corpus"]["passages"] == 40
    assert report["corpus"]["queries"] == len(RETRIEVAL_TOPICS)
    assert report["corpus"]["digest"]
    assert report["lancedb_version"]
    assert report["thresholds"] == {"recall_at_k": 0.99, "rank_correlation": 0.95}
    # Flat float32 Lance search must reproduce the numpy float32 reference
    # exactly: it is the same computation on the same numbers.
    baseline = report["variants"]["float32_exact"]
    assert baseline.get("error") is None
    assert baseline["recall_at_k"] == pytest.approx(1.0)
    assert baseline["rank_correlation"] == pytest.approx(1.0)
    assert report["baseline_self_recall"] == pytest.approx(1.0)
    for name in ("float32_exact", "float16_exact"):
        variant = report["variants"][name]
        assert variant.get("error") is None, name
        assert variant["recall_at_k"] == pytest.approx(1.0), name
        assert variant["rank_correlation"] == pytest.approx(1.0), name
    # Every combination that ran reports the full cost picture; approximate
    # indexes are measured, not assumed identical to the exact reference.
    for name, variant in report["variants"].items():
        if variant.get("error") is not None:
            continue
        assert variant["physical_bytes"] > 0, name
        assert variant["post_optimize_bytes"] > 0, name
        assert variant["table_build_ms"] >= 0, name
        assert variant["index_build_ms"] >= 0, name
        assert variant["hybrid_latency_ms"]["count"] == len(RETRIEVAL_TOPICS) * 2, name
    # Gates evaluate the recorded thresholds, so the exact variants pass and
    # every variant's gate matches its own measured numbers.
    for name, gate in report["gates"].items():
        variant = report["variants"][name]
        if variant.get("error") is not None:
            assert gate["recall_ok"] is False, name
            assert gate["rank_ok"] is False, name
            continue
        assert gate["recall_ok"] == (variant["recall_at_k"] >= 0.99), name
        assert gate["rank_ok"] == (variant["rank_correlation"] >= 0.95), name


def test_precision_report_recomputes_float16_recall_independently(tmp_path: Path) -> None:
    """The published recall must be the arithmetic it claims (T3 convention)."""
    embedder = TopicPrecisionEmbedder()
    report = run_precision_benchmark(embedder, tmp_path, passages=40, top_k=5, iterations=1)

    corpus, queries = build_retrieval_corpus(passages=40)
    passages = np.asarray(embedder.embed_passages([p.content for p in corpus]), dtype=np.float32)
    query_vectors = np.asarray(
        [embedder.embed_query(query.text) for query in queries], dtype=np.float32
    )
    normalized = passages / np.linalg.norm(passages, axis=1, keepdims=True)
    expected = 0.0
    for query_vector in query_vectors:
        reference_order = np.argsort(-(query_vector @ normalized.T), kind="stable")[:5]
        reference = {corpus[index].chunk_id for index in reference_order}
        halved = passages.astype(np.float16).astype(np.float32)
        halved_normalized = halved / np.linalg.norm(halved, axis=1, keepdims=True)
        candidate_order = np.argsort(-(query_vector @ halved_normalized.T), kind="stable")[:5]
        candidate = {corpus[index].chunk_id for index in candidate_order}
        expected += len(reference & candidate) / 5
    expected /= len(query_vectors)

    assert report["variants"]["float16_exact"]["recall_at_k"] == pytest.approx(expected)


def test_directory_physical_bytes_ignores_symlinks(tmp_path: Path) -> None:
    measured = tmp_path / "measured"
    measured.mkdir()
    (measured / "real.bin").write_bytes(b"x" * 100)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"y" * 4096)
    (measured / "link.bin").symlink_to(outside)

    assert _directory_physical_bytes(measured) == 100
