"""Configure preserves the live Syndex session and targets its own installation."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from subprocess import CompletedProcess

import pytest
from test_tui import _make_app, _wait_for_preview, _wait_for_workers
from textual.widgets import Button, Input, Select, Static

from code_indexing_mcp.installer.accelerator import server_executable


@pytest.fixture
def configure_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    import code_indexing_mcp.tui.app as app_module

    directory = tmp_path / "custom install"
    executable = server_executable(directory)
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(sys, "prefix", str(directory / ".venv"))
    monkeypatch.setenv("CODE_INDEXING_MCP_INSTALL_DIR", str(tmp_path / "other-install"))
    calls: list[list[str]] = []

    def run(command: list[str], *, check: bool) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0)

    monkeypatch.setattr(app_module.subprocess, "run", run)
    return directory, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["button", "key"])
async def test_configure_preserves_session(
    monkeypatch: pytest.MonkeyPatch,
    configure_process,
    trigger: str,  # type: ignore[no-untyped-def]
) -> None:
    import code_indexing_mcp.tui.app as app_module

    directory, calls = configure_process
    app = _make_app()
    monkeypatch.setattr(app, "suspend", nullcontext)
    monkeypatch.setattr(app_module, "create_tui_service", lambda **kwargs: app.service)
    async with app.run_test() as pilot:
        await _wait_for_workers(app, pilot)
        app.query_one("#project-select", Select).value = "proj-2"
        await pilot.pause()
        await _wait_for_workers(app, pilot)
        app.query_one("#query-input", Input).value = "main"
        app.action_submit_query()
        await _wait_for_preview(app, pilot)
        app.action_show_outline()
        await _wait_for_workers(app, pilot)
        await pilot.pause()
        selected = app.service.selected_project
        detail = app._capture_detail()
        history = list(app._history)
        hits = list(app._hits)
        if trigger == "button":
            await pilot.click("#configure-button")
        else:
            app.query_one("#configure-button", Button).focus()
            await pilot.press("c")
        await pilot.pause()
        assert calls == [
            [
                sys.executable,
                "-m",
                "code_indexing_mcp",
                "configure",
                "--install-dir",
                str(directory),
            ]
        ]
        assert app.return_code is None
        assert app.service.selected_project == selected
        assert app.query_one("#query-input", Input).value == "main"
        assert app._hits == hits
        assert app._capture_detail() == detail
        assert app._history == history


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [1, 130])
async def test_configure_failure_or_cancel_returns_to_live_app(
    monkeypatch: pytest.MonkeyPatch,
    configure_process,
    code: int,  # type: ignore[no-untyped-def]
) -> None:
    import code_indexing_mcp.tui.app as app_module

    app = _make_app()
    monkeypatch.setattr(app, "suspend", nullcontext)
    monkeypatch.setattr(
        app_module.subprocess, "run", lambda command, **kwargs: CompletedProcess(command, code)
    )
    async with app.run_test() as pilot:
        await _wait_for_workers(app, pilot)
        app.query_one("#configure-button", Button).focus()
        await pilot.pause()
        app.action_open_configure()
        assert app.return_code is None
        if code == 1:
            assert app.query_one("#error-panel").display
            assert "status 1" in str(app.query_one("#error-content", Static).render())
        else:
            assert "cancelled" in str(app.query_one("#status-bar", Static).render()).lower()


@pytest.mark.asyncio
async def test_unmanaged_install_does_not_configure_a_different_install(
    monkeypatch: pytest.MonkeyPatch,
    configure_process,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    _, calls = configure_process
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "unmanaged"))
    app = _make_app()
    async with app.run_test() as pilot:
        await _wait_for_workers(app, pilot)
        app.query_one("#configure-button", Button).focus()
        await pilot.pause()
        app.action_open_configure()
        assert app.return_code is None
        assert calls == []
        assert app.query_one("#error-panel").display
