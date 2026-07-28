import json
from pathlib import Path

import pytest

from incode_mcp import cli, daemon
from incode_mcp.application import Application
from incode_mcp.cli import main


def test_cli_initializes_and_lists_projects(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setenv("INCODE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("INCODE_CACHE_DIR", str(tmp_path / "cache"))

    assert main(["init", str(root)]) == 0
    init_result = json.loads(capsys.readouterr().out)
    assert init_result["name"] == "repo"

    assert main(["projects", "list"]) == 0
    projects = json.loads(capsys.readouterr().out)
    assert [project["id"] for project in projects] == [init_result["id"]]


def test_cli_runs_the_index_benchmark_with_machine_readable_output(
    tmp_path: Path, monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("INCODE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("INCODE_CACHE_DIR", str(tmp_path / "cache"))
    received: dict[str, object] = {}

    def fake_benchmark(**kwargs: object) -> dict[str, object]:
        received.update(kwargs)
        return {"schema_version": 1, "scenarios": {"cold_start": {}}}

    monkeypatch.setattr(cli, "run_index_benchmark_command", fake_benchmark)
    work_dir = tmp_path / "benchmark"

    assert (
        main(
            [
                "benchmark",
                "index",
                "--files",
                "3",
                "--functions-per-file",
                "2",
                "--batch-size",
                "8",
                "--work-dir",
                str(work_dir),
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out)["schema_version"] == 1
    assert received["files"] == 3
    assert received["functions_per_file"] == 2
    assert received["batch_size"] == 8
    assert received["work_dir"] == work_dir


def test_serve_falls_back_to_direct_when_local_sockets_are_unavailable(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """INCODE_BROKER=auto must not crash where AF_UNIX does not exist."""
    monkeypatch.setenv("INCODE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("INCODE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("INCODE_BROKER", raising=False)
    monkeypatch.setattr(daemon, "daemon_supported", lambda: False)

    def refuse(*_: object, **__: object) -> object:
        raise AssertionError("the daemon must not be started on an unsupported platform")

    monkeypatch.setattr(cli, "ensure_daemon", refuse)
    served: dict[str, object] = {}

    class FakeServer:
        def run(self, transport: str) -> None:
            served["transport"] = transport

    def fake_create_server(app: object) -> FakeServer:
        served["app"] = app
        return FakeServer()

    monkeypatch.setattr(cli, "create_server", fake_create_server)

    assert cli.main(["serve"]) == 0
    assert served["transport"] == "stdio"
    assert isinstance(served["app"], Application)


def test_serve_refuses_an_explicit_broker_opt_in_without_local_sockets(
    tmp_path: Path, monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("INCODE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("INCODE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("INCODE_BROKER", "on")
    monkeypatch.setattr(daemon, "daemon_supported", lambda: False)

    assert cli.main(["serve"]) == 2
    assert "INCODE_BROKER=off" in capsys.readouterr().err


def test_cli_reports_the_resolved_embedding_backend(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("INCODE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("INCODE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("INCODE_EMBED_ACCELERATOR", "cpu")

    assert main(["model", "status"]) == 0

    status = json.loads(capsys.readouterr().out)
    assert status["requested_accelerator"] == "cpu"
    assert status["resolved_accelerator"] == "cpu"
    assert status["execution_provider"] == "CPUExecutionProvider"
    assert status["probe_cache_state"] == "not-applicable"
    assert status["fallback_reason"] is None
    assert status["strict"] is False


def test_model_status_explains_an_accelerator_it_cannot_honour(
    tmp_path: Path, monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("INCODE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("INCODE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("INCODE_EMBED_ACCELERATOR", "cuda")

    assert main(["model", "status"]) == 0

    status = json.loads(capsys.readouterr().out)
    if "CUDAExecutionProvider" in status["available_providers"]:
        # Skipped rather than passed vacuously: on a CUDA host the request is
        # honoured and there is no unhonourable request left to explain.
        pytest.skip("this host offers CUDA, so the request is honoured")
    # Status reports the CPU it will really use and names the reason rather
    # than claiming the CUDA it cannot deliver.
    assert status["resolved_accelerator"] == "cpu"
    assert "CUDAExecutionProvider" in status["fallback_reason"]
