from pathlib import Path

from incode_mcp.benchmark import run_index_benchmark, write_benchmark_corpus
from incode_mcp.models import IndexReport, ProjectInfo


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
