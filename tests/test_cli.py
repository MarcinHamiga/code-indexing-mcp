import io
import json
from pathlib import Path

import pytest

from code_indexing_mcp import cli, daemon
from code_indexing_mcp.application import Application
from code_indexing_mcp.cli import main


def test_cli_initializes_and_lists_projects(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setenv("CODE_INDEXING_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CODE_INDEXING_CACHE_DIR", str(tmp_path / "cache"))

    assert main(["init", str(root)]) == 0
    init_result = json.loads(capsys.readouterr().out)
    assert init_result["name"] == "repo"

    assert main(["projects", "list"]) == 0
    projects = json.loads(capsys.readouterr().out)
    assert [project["id"] for project in projects] == [init_result["id"]]


def test_cli_runs_the_index_benchmark_with_machine_readable_output(
    tmp_path: Path, monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CODE_INDEXING_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CODE_INDEXING_CACHE_DIR", str(tmp_path / "cache"))
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
    """CODE_INDEXING_BROKER=auto must not crash where AF_UNIX does not exist."""
    monkeypatch.setenv("CODE_INDEXING_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CODE_INDEXING_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CODE_INDEXING_BROKER", raising=False)
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
    monkeypatch.setenv("CODE_INDEXING_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CODE_INDEXING_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CODE_INDEXING_BROKER", "on")
    monkeypatch.setattr(daemon, "daemon_supported", lambda: False)

    assert cli.main(["serve"]) == 2
    assert "CODE_INDEXING_BROKER=off" in capsys.readouterr().err


def test_cli_reports_the_resolved_embedding_backend(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CODE_INDEXING_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CODE_INDEXING_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CODE_INDEXING_EMBED_ACCELERATOR", "cpu")

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
    monkeypatch.setenv("CODE_INDEXING_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CODE_INDEXING_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CODE_INDEXING_EMBED_ACCELERATOR", "cuda")

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


def test_configure_delegates_to_the_installer(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    calls = []

    def fake_configure_main(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return 0

    import code_indexing_mcp.installer.cli as installer_cli

    monkeypatch.setattr(installer_cli, "configure_main", fake_configure_main)
    code = main(["configure", "--install-dir", "/opt/ci-mcp", "--set", "CODE_INDEXING_OFFLINE=1"])
    assert code == 0
    assert calls == [
        {
            "install_dir": "/opt/ci-mcp",
            "accelerator": None,
            "harnesses": None,
            "settings": ["CODE_INDEXING_OFFLINE=1"],
            "unsets": [],
            "no_tui": False,
            "bin_dir": None,
            "no_launcher": False,
            "no_modify_path": False,
            "repair": False,
        }
    ]


def test_update_delegates_to_the_installer(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls = []

    def fake_update_main(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return 0

    import code_indexing_mcp.installer.update as installer_update

    monkeypatch.setattr(installer_update, "update_main", fake_update_main)
    code = main(["update", "--install-dir", "/opt/ci-mcp", "--skip-accelerator"])
    assert code == 0
    assert calls == [
        {
            "install_dir": "/opt/ci-mcp",
            "check": False,
            "skip_accelerator": True,
            "finalize": False,
            "previous_sha": None,
        }
    ]


def test_serve_path_does_not_import_textual_or_the_updater() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import code_indexing_mcp.cli, sys; "
            "print('textual' in sys.modules, "
            "'code_indexing_mcp.installer.update' in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False False"


def test_serve_does_not_spawn_git_on_a_development_checkout(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Without a managed install there is nothing to check, so nothing may run."""

    import subprocess

    monkeypatch.setenv("CODE_INDEXING_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CODE_INDEXING_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CODE_INDEXING_MCP_INSTALL_DIR", str(tmp_path / "absent"))
    monkeypatch.setattr(daemon, "daemon_supported", lambda: False)

    def refuse(*_: object, **__: object) -> object:
        raise AssertionError("the serve path must not spawn a subprocess")

    monkeypatch.setattr(subprocess, "run", refuse)

    class FakeServer:
        def run(self, transport: str) -> None:
            return None

    monkeypatch.setattr(cli, "create_server", lambda app: FakeServer())

    assert cli.main(["serve"]) == 0


def test_version_flag_prints_the_version_and_exits(capsys) -> None:  # type: ignore[no-untyped-def]
    from code_indexing_mcp import __version__

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    printed = capsys.readouterr().out.strip()
    assert printed.startswith("code-indexing-mcp ")
    assert __version__ in printed


def _fake_managed_install(tmp_path: Path, monkeypatch, remote_sha: str) -> None:  # type: ignore[no-untyped-def]
    """Make update_check believe this process runs from a managed install."""

    import sys
    import time

    from code_indexing_mcp import update_check

    install_dir = tmp_path / "install"
    git_dir = install_dir / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "refs" / "heads" / "main").write_text("a" * 40 + "\n")
    monkeypatch.setenv("CODE_INDEXING_MCP_INSTALL_DIR", str(install_dir))
    monkeypatch.setattr(sys, "prefix", str(install_dir / ".venv"))
    monkeypatch.setenv("CODE_INDEXING_DATA_DIR", str(tmp_path / "data"))
    cache = tmp_path / "cache"
    monkeypatch.setenv("CODE_INDEXING_CACHE_DIR", str(cache))
    # A fresh timestamp keeps the throttle closed, so no background refresh runs.
    update_check.write_cache(
        cache,
        update_check.UpdateStatus(
            checked_at=time.time(), local_sha="a" * 40, remote_sha=remote_sha
        ),
    )


def test_projects_list_reports_an_available_update_on_stderr(  # type: ignore[no-untyped-def]
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _fake_managed_install(tmp_path, monkeypatch, remote_sha="b" * 40)

    assert main(["projects", "list"]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == []
    assert "code-indexing-mcp update" in captured.err


def test_update_notice_is_silent_when_disabled(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    _fake_managed_install(tmp_path, monkeypatch, remote_sha="b" * 40)
    monkeypatch.setenv("CODE_INDEXING_UPDATE_CHECK", "off")

    assert main(["projects", "list"]) == 0

    assert "update is available" not in capsys.readouterr().err


class _TinyEmbedder:
    model_id = "test/tiny"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, float(len(text))]


def test_index_narrates_its_progress_on_stderr_and_keeps_stdout_json(  # type: ignore[no-untyped-def]
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A long index must say what it is doing without polluting the machine-readable output."""

    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    monkeypatch.setenv("CODE_INDEXING_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CODE_INDEXING_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        cli,
        "Application",
        lambda paths, cwd: Application(paths, embedder=_TinyEmbedder(), cwd=cwd),
    )

    assert main(["init", str(root)]) == 0
    capsys.readouterr()
    assert main(["index", str(root)]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out)["indexed_files"] == 1
    # Redirected stderr gets whole lines, and every phase gets one.
    assert "Scanning for changed files" in captured.err
    assert "Embedding" in captured.err
    assert "Committing the index" in captured.err


def test_a_terminal_gets_one_status_line_that_is_cleaned_up_afterwards() -> None:
    from code_indexing_mcp.progress import IndexProgress

    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = Terminal()
    printer = cli._ProgressPrinter(stream)

    printer(IndexProgress(project_id="abc", files_seen=1, phase="scanning"))
    printer(IndexProgress(project_id="abc", files_seen=2, phase="scanning"))
    printer.clear()

    written = stream.getvalue()
    assert written.count("\n") == 0, "a status line must not scroll the terminal"
    assert "Scanning 2 files" in written
    # Whatever the last line was, the cursor ends on a blank line so the JSON
    # report is not printed on top of it.
    assert written.rstrip("\r").endswith(" " * len("Scanning 2 files"))
