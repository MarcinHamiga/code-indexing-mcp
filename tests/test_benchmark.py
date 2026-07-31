from pathlib import Path

import benchmark_index_memory

from code_indexing_mcp.benchmark import run_index_benchmark, write_benchmark_corpus
from code_indexing_mcp.models import IndexReport, ProjectInfo
from code_indexing_mcp.settings import IndexSettings


class BenchmarkApplication:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.force_calls: list[bool] = []

    def init_project(self, path: Path) -> ProjectInfo:
        assert path == self.root
        return ProjectInfo(id="benchmark-project", name="benchmark", root=path)

    def index_project(self, project: str, *, force: bool = False) -> IndexReport:
        assert project == "benchmark-project"
        self.force_calls.append(force)
        call = len(self.force_calls)
        return IndexReport(
            project_id=project,
            discovered_files=4,
            indexed_files=1 if call == 3 else 4,
            parsed_files=1 if call == 3 else 4,
            embedded_chunks=2 if call == 3 else 8,
            duration_ms=100,
            embedding_backend="cpu",
            embedding_batch_size=8,
        )


def test_benchmark_runs_cold_warm_incremental_and_forced_scenarios(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    write_benchmark_corpus(root, files=4, functions_per_file=2)
    app = BenchmarkApplication(root)

    payload = run_index_benchmark(app, root)

    assert app.force_calls == [True, True, False, True]
    assert list(payload["scenarios"]) == [
        "cold_start",
        "warm_index",
        "incremental_index",
        "forced_reindex",
    ]
    assert payload["scenarios"]["incremental_index"]["report"]["indexed_files"] == 1
    assert payload["scenarios"]["warm_index"]["chunks_per_second"] == 80.0
    assert "phase_1_incremental_marker" in (root / "module_0000.py").read_text()


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
