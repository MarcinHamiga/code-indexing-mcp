"""Headless interaction tests for the Code Indexing MCP TUI application."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from test_tui_service import FakeApplication, _sample_hit, _sample_project
from textual.widgets import Input, Label, OptionList, Select, Static

from code_indexing_mcp.application import ApplicationLike
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.models import SearchHit, SearchResponse
from code_indexing_mcp.tui.app import CodeIndexingApp
from code_indexing_mcp.tui.service import TuiService


def _make_app(app: FakeApplication | None = None) -> CodeIndexingApp:
    fake_app = app or FakeApplication(
        projects=[
            _sample_project(project_id="proj-1", name="repo-alpha"),
            _sample_project(
                project_id="proj-2", name="repo-beta", root=Path("/workspace/repo-beta")
            ),
        ]
    )
    service = TuiService(cast(ApplicationLike, fake_app), cwd=Path("/workspace/test-repo"))
    return CodeIndexingApp(service=service)


@pytest.mark.asyncio
async def test_tui_app_mount_and_initial_discovery() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        header_title = app.query_one("#header-title", Label).render()
        assert "repo-alpha" in str(header_title)
        assert "proj-1" in str(header_title)

        header_status = app.query_one("#header-status", Label).render()
        assert "State: ready" in str(header_status)

        status_bar = app.query_one("#status-bar", Static).render()
        assert "Selected project: repo-alpha" in str(status_bar)

        proj_select = app.query_one("#project-select", Select)
        assert proj_select.value == "proj-1"


@pytest.mark.asyncio
async def test_tui_app_project_switch() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Run a search to populate results in the initial project
        query_input = app.query_one("#query-input", Input)
        query_input.value = "main"
        app.action_submit_query()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(app._hits) > 0

        proj_select = app.query_one("#project-select", Select)
        proj_select.value = "proj-2"
        await pilot.pause()

        header_title = app.query_one("#header-title", Label).render()
        assert "repo-beta" in str(header_title)
        assert "proj-2" in str(header_title)

        status_bar = app.query_one("#status-bar", Static).render()
        assert "Selected project: repo-beta" in str(status_bar)

        # Hits and detail pane should be cleared
        assert len(app._hits) == 0
        assert app.query_one("#results-list", OptionList).option_count == 0
        detail_text = str(app.query_one("#detail-content", Static).render())
        assert "Active project changed to repo-beta" in detail_text


@pytest.mark.asyncio
async def test_tui_app_search_semantic() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        query_input = app.query_one("#query-input", Input)
        query_input.value = "authentication"
        app.action_submit_query()

        await app.workers.wait_for_complete()
        await pilot.pause()

        results_title = app.query_one("#results-title", Label).render()
        assert "Results (1)" in str(results_title)

        results_list = app.query_one("#results-list", OptionList)
        assert results_list.option_count == 1

        status_bar = app.query_one("#status-bar", Static).render()
        assert "Found 1 hit(s)" in str(status_bar)


@pytest.mark.asyncio
async def test_tui_app_search_empty() -> None:
    fake_app = FakeApplication()
    fake_app.search_code = lambda *args, **kwargs: SearchResponse(query="nonexistent", hits=[])  # type: ignore[method-assign]
    app = _make_app(fake_app)

    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        query_input = app.query_one("#query-input", Input)
        query_input.value = "nonexistent"
        app.action_submit_query()

        await app.workers.wait_for_complete()
        await pilot.pause()

        results_title = app.query_one("#results-title", Label).render()
        assert "Results (0)" in str(results_title)

        results_list = app.query_one("#results-list", OptionList)
        assert results_list.option_count == 0

        status_bar = app.query_one("#status-bar", Static).render()
        assert "No results found for 'nonexistent'" in str(status_bar)


@pytest.mark.asyncio
async def test_tui_app_search_symbol() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        mode_select = app.query_one("#mode-select", Select)
        mode_select.value = "symbol"

        query_input = app.query_one("#query-input", Input)
        query_input.value = "main"
        app.action_submit_query()

        await app.workers.wait_for_complete()
        await pilot.pause()

        results_title = app.query_one("#results-title", Label).render()
        assert "Results (1)" in str(results_title)


@pytest.mark.asyncio
async def test_tui_app_preview_chunk_and_shortcuts() -> None:
    fake_app = FakeApplication()
    app = _make_app(fake_app)

    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Run a search to populate results
        query_input = app.query_one("#query-input", Input)
        query_input.value = "main"
        app.action_submit_query()
        await app.workers.wait_for_complete()
        await pilot.pause()

        # 1. Preview chunk via action_open_selected
        app.action_open_selected()
        await app.workers.wait_for_complete()
        await pilot.pause()

        detail_title = app.query_one("#detail-title", Label).render()
        assert "Preview: src/main.py:1-10" in str(detail_title)

        # 2. File outline via action_show_outline
        app.action_show_outline()
        await app.workers.wait_for_complete()
        await pilot.pause()

        detail_title = app.query_one("#detail-title", Label).render()
        assert "Outline: src/main.py (1 items)" in str(detail_title)

        # 3. References via action_show_references
        app.action_show_references()
        await app.workers.wait_for_complete()
        await pilot.pause()

        detail_title = app.query_one("#detail-title", Label).render()
        assert "References: src/main.py (1 hits)" in str(detail_title)

        # 4. Impact radius via action_show_impact
        app.action_show_impact()
        await app.workers.wait_for_complete()
        await pilot.pause()

        detail_title = app.query_one("#detail-title", Label).render()
        assert "Impact Radius: src/main.py (1 layers, 3 visited)" in str(detail_title)


@pytest.mark.asyncio
async def test_tui_app_index_trigger() -> None:
    fake_app = FakeApplication()
    app = _make_app(fake_app)

    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert not app._is_indexing
        app.action_trigger_index()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert not app._is_indexing
        status_bar = app.query_one("#status-bar", Static).render()
        assert "Indexing complete: 5 files indexed, 10 chunks embedded." in str(status_bar)
        assert len(fake_app.index_calls) == 1


@pytest.mark.asyncio
async def test_tui_app_error_surfaced_in_status_bar() -> None:
    fake_app = FakeApplication()

    def fail_search(*args: object, **kwargs: object) -> None:
        raise CodeIndexingError(ErrorCode.INDEX_INCOMPATIBLE, "Index corrupted")

    fake_app.search_code = fail_search  # type: ignore[method-assign]
    app = _make_app(fake_app)

    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        query_input = app.query_one("#query-input", Input)
        query_input.value = "test"
        app.action_submit_query()
        await app.workers.wait_for_complete()
        await pilot.pause()

        status_bar = app.query_one("#status-bar", Static)
        assert "Index corrupted" in str(status_bar.render())
        assert status_bar.has_class("error")


@pytest.mark.asyncio
async def test_tui_app_stale_response_discarded() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        # If request ID doesn't match current ID, render is ignored
        app._search_request_id = 10
        stale_hit: list[SearchHit] = [_sample_hit(symbol="old_symbol")]
        app._render_search_results(request_id=9, hits=stale_hit, query="old")

        results_list = app.query_one("#results-list", OptionList)
        assert results_list.option_count == 0


@pytest.mark.asyncio
async def test_tui_app_80x24_smoke_test() -> None:
    app = _make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        query_input = app.query_one("#query-input", Input)
        query_input.value = "query"
        app.action_submit_query()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.query_one("#results-list", OptionList).option_count == 1
        app.action_open_selected()
        await app.workers.wait_for_complete()
        await pilot.pause()

        detail_title = app.query_one("#detail-title", Label).render()
        assert "Preview: src/main.py" in str(detail_title)


@pytest.mark.asyncio
async def test_tui_app_navigation_and_quit() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Focus query input
        app.action_focus_query()
        await pilot.pause()
        assert isinstance(app.focused, Input)

        # Quit when input is focused should do nothing
        app.action_quit_app()
        assert app.return_code is None

        # Escape switches focus to results-list
        app.action_escape_action()
        await pilot.pause()
        assert isinstance(app.focused, OptionList)

        # Quit when input not focused exits app
        app.action_quit_app()
        await pilot.pause()
        assert app.return_code == 0


def test_tui_main_invalid_project_exits_with_error(capsys: pytest.CaptureFixture[str]) -> None:
    from code_indexing_mcp.tui import main as tui_main

    exit_code = tui_main(["nonexistent-project-xyz"])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Error:" in captured.err
