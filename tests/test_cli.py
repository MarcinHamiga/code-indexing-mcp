import json
from pathlib import Path

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
