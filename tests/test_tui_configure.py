"""Tests for reaching Configure from inside the syndex TUI."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input

from code_indexing_mcp.tui.app import CodeIndexingApp


class _StubService:
    """Just enough service for the app to mount without a real index."""

    selected_project = None

    def discover_current_project(self):  # type: ignore[no-untyped-def]
        return None

    def list_projects(self):  # type: ignore[no-untyped-def]
        return []

    def project_status(self, project=None):  # type: ignore[no-untyped-def]
        return None


async def _click(pilot, selector: str) -> None:  # type: ignore[no-untyped-def]
    await pilot.click(selector)
    await pilot.pause(0.4)


@pytest.mark.asyncio
async def test_configure_button_opens_the_wizard(monkeypatch: pytest.MonkeyPatch) -> None:
    app = CodeIndexingApp(service=_StubService())  # type: ignore[arg-type]
    calls: list[bool] = []
    monkeypatch.setattr(app, "action_open_configure", lambda: calls.append(True))
    async with app.run_test() as pilot:
        await _click(pilot, "#configure-button")
        assert calls == [True]
        assert app.return_code is None


@pytest.mark.asyncio
async def test_c_key_opens_configure_outside_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    app = CodeIndexingApp(service=_StubService())  # type: ignore[arg-type]
    calls: list[bool] = []
    monkeypatch.setattr(app, "action_open_configure", lambda: calls.append(True))
    async with app.run_test() as pilot:
        app.query_one("#configure-button", Button).focus()
        await pilot.press("c")
        await pilot.pause()
        assert calls == [True]
        assert app.return_code is None


@pytest.mark.asyncio
async def test_c_key_types_inside_the_search_field() -> None:
    """A bare letter belongs to the text being typed, like every other key."""

    app = CodeIndexingApp(service=_StubService())  # type: ignore[arg-type]
    async with app.run_test() as pilot:
        app.query_one("#query-input", Input).focus()
        await pilot.press("c")
        await pilot.pause()
        assert app.return_code is None
        assert "c" in app.query_one("#query-input", Input).value


class _FakeApp:
    """Stand-in for CodeIndexingApp with a scripted sequence of exit codes."""

    def __init__(self, codes: list[int], service=None):  # type: ignore[no-untyped-def]
        self._codes = codes
        self.return_code = 0

    def run(self) -> int:
        return self._codes.pop(0)


def test_launch_tui_keeps_the_selected_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import code_indexing_mcp.installer.cli as installer_cli
    import code_indexing_mcp.tui as tui_package
    import code_indexing_mcp.tui.app as tui_app
    import code_indexing_mcp.tui.service as tui_service

    selections: list[str] = []

    class Service:
        def select_project(self, project: str) -> None:
            selections.append(project)

    monkeypatch.setattr(tui_service, "create_tui_service", Service)
    fake = _FakeApp([0])
    monkeypatch.setattr(tui_app, "CodeIndexingApp", lambda service: fake)
    calls: list[dict] = []
    monkeypatch.setattr(
        installer_cli,
        "configure_main",
        lambda **kwargs: calls.append(kwargs) or 0,
    )
    assert tui_package._launch_tui("selected-project") == 0
    assert selections == ["selected-project"]
    assert calls == []  # Configure is handled in the mounted app.


def test_launch_tui_passes_normal_exits_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import code_indexing_mcp.installer.cli as installer_cli
    import code_indexing_mcp.tui as tui_package
    import code_indexing_mcp.tui.app as tui_app
    import code_indexing_mcp.tui.service as tui_service

    monkeypatch.setattr(tui_service, "create_tui_service", lambda: object())
    monkeypatch.setattr(tui_app, "CodeIndexingApp", lambda service: _FakeApp([5]))
    called = []
    monkeypatch.setattr(
        installer_cli, "configure_main", lambda **kwargs: called.append(kwargs) or 0
    )
    assert tui_package._launch_tui(None) == 5
    assert called == []
