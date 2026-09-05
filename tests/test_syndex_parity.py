"""Parity tests: `syndex` exposes the full `code-indexing-mcp` command surface."""

from __future__ import annotations

import argparse

import pytest

from code_indexing_mcp import cli
from code_indexing_mcp.tui import main as syndex_main


def _subcommand_names(prog: str) -> set[str]:
    parser = cli._parser(prog)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("CLI parser defines no subcommands")


def test_command_names_match_the_parser() -> None:
    assert set(cli.COMMAND_NAMES) == _subcommand_names("code-indexing-mcp")


def test_both_programs_share_the_same_subcommands() -> None:
    assert _subcommand_names("syndex") == _subcommand_names("code-indexing-mcp")


def test_bare_syndex_launches_the_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    import code_indexing_mcp.tui as tui_package

    received: list[str | None] = []
    monkeypatch.setattr(tui_package, "_launch_tui", lambda project: received.append(project) or 0)
    assert syndex_main([]) == 0
    assert received == [None]


def test_syndex_project_shorthand_launches_the_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    import code_indexing_mcp.tui as tui_package

    received: list[str | None] = []
    monkeypatch.setattr(tui_package, "_launch_tui", lambda project: received.append(project) or 0)
    assert syndex_main(["myproj"]) == 0
    assert received == ["myproj"]


def test_syndex_subcommand_delegates_to_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, object] = {}

    def fake_cli_main(argv: object = None, prog: str = "") -> int:
        received["argv"] = argv
        received["prog"] = prog
        return 0

    monkeypatch.setattr(cli, "main", fake_cli_main)
    assert syndex_main(["status", "myproj"]) == 0
    assert received == {"argv": ["status", "myproj"], "prog": "syndex"}


def test_syndex_flag_delegates_to_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, object] = {}

    def fake_cli_main(argv: object = None, prog: str = "") -> int:
        received["argv"] = argv
        received["prog"] = prog
        return 0

    monkeypatch.setattr(cli, "main", fake_cli_main)
    assert syndex_main(["--version"]) == 0
    assert received == {"argv": ["--version"], "prog": "syndex"}


def test_syndex_tui_subcommand_opens_the_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    import code_indexing_mcp.tui as tui_package

    received: list[str | None] = []
    monkeypatch.setattr(tui_package, "_launch_tui", lambda project: received.append(project) or 0)
    assert syndex_main(["tui", "myproj"]) == 0
    assert received == ["myproj"]


def test_syndex_tui_subcommand_without_project_opens_the_tui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import code_indexing_mcp.tui as tui_package

    received: list[str | None] = []
    monkeypatch.setattr(tui_package, "_launch_tui", lambda project: received.append(project) or 0)
    assert syndex_main(["tui"]) == 0
    assert received == [None]


def test_project_named_like_a_command_needs_the_tui_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import code_indexing_mcp.tui as tui_package

    received: list[str | None] = []
    monkeypatch.setattr(tui_package, "_launch_tui", lambda project: received.append(project) or 0)
    assert syndex_main(["tui", "status"]) == 0
    assert received == ["status"]


def test_bare_syndex_skips_the_cli_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    import code_indexing_mcp.tui as tui_package

    monkeypatch.delitem(sys.modules, "code_indexing_mcp.cli", raising=False)
    monkeypatch.setattr(tui_package, "_launch_tui", lambda project: 0)
    assert syndex_main([]) == 0
    assert "code_indexing_mcp.cli" not in sys.modules


def test_tui_package_exports_are_importable() -> None:
    pytest.importorskip("textual", reason="tui extra not installed")
    import code_indexing_mcp.tui as tui_package

    for name in tui_package.__all__:
        assert getattr(tui_package, name) is not None


def test_version_reports_the_invoked_program(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert capsys.readouterr().out.startswith("code-indexing-mcp ")
    with pytest.raises(SystemExit):
        cli.main(["--version"], prog="syndex")
    assert capsys.readouterr().out.startswith("syndex ")
