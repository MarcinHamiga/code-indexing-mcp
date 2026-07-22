import json
from pathlib import Path

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
