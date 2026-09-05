"""Headless interaction tests for the Code Indexing MCP TUI application."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from rich.text import Text
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
        assert "proj-1" in str(app.query_one("#header-title").tooltip)

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
        assert "proj-2" in str(app.query_one("#header-title").tooltip)

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
        assert "References: main · src/main.py (1 hits)" in str(detail_title)

        # 4. Impact radius via action_show_impact
        app.action_show_impact()
        await app.workers.wait_for_complete()
        await pilot.pause()

        detail_title = app.query_one("#detail-title", Label).render()
        assert "Impact Radius: main · src/main.py (1 layers, 3 visited)" in str(detail_title)


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


@pytest.mark.asyncio
async def test_empty_query_results_clear_previous_preview() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#query-input", Input).value = "main"
        app.action_submit_query()
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.action_open_selected()
        await app.workers.wait_for_complete()
        await pilot.pause()
        app._render_search_results(app._search_request_id, [], "missing")
        assert "Preview:" not in str(app.query_one("#detail-title", Label).render())
        assert "Try" in str(app.query_one("#detail-content", Static).render())


@pytest.mark.asyncio
async def test_project_switch_rejects_old_search_completion() -> None:
    import threading

    started = threading.Event()
    release = threading.Event()
    fake = FakeApplication(projects=[_sample_project(), _sample_project("proj-2", "second")])
    original = fake.search_code

    def delayed(*args: object, **kwargs: object) -> SearchResponse:
        started.set()
        release.wait(3)
        return original("old")

    fake.search_code = delayed  # type: ignore[method-assign]
    app = _make_app(fake)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#query-input", Input).value = "old"
        app.action_submit_query()
        await pilot.pause()
        assert started.is_set()
        app.query_one("#project-select", Select).value = "proj-2"
        await pilot.pause()
        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.query_one("#results-list", OptionList).option_count == 0
        assert "second" in str(app.query_one("#header-title", Label).render())


@pytest.mark.asyncio
async def test_new_search_rejects_old_detail_completion() -> None:
    import threading

    release = threading.Event()
    fake = FakeApplication()
    original = fake.get_chunk

    def delayed(chunk_id: str):
        release.wait(3)
        return original(chunk_id)

    fake.get_chunk = delayed  # type: ignore[method-assign]
    app = _make_app(fake)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app._render_search_results(app._search_request_id, [_sample_hit()], "old")
        app.action_open_selected()
        await pilot.pause()
        app.query_one("#query-input", Input).value = "new"
        app.action_submit_query()
        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "Preview:" not in str(app.query_one("#detail-title", Label).render())


@pytest.mark.asyncio
async def test_compact_layout_preserves_search_space_and_switches_panes() -> None:
    app = _make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        query = app.query_one("#query-input", Input)
        assert query.content_region.width >= 60
        assert app.focused is query
        assert app.query_one("#results-pane").display
        assert not app.query_one("#detail-pane").display
        await pilot.press("m", "a", "i", "n", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.query_one("#detail-pane").display
        assert not app.query_one("#results-pane").display
        await pilot.press("escape")
        assert app.query_one("#results-pane").display
        await pilot.resize_terminal(120, 32)
        await pilot.pause()
        assert app.query_one("#results-pane").display
        assert app.query_one("#detail-pane").display
        assert (
            app.query_one("#detail-pane").region.width > app.query_one("#results-pane").region.width
        )


@pytest.mark.asyncio
async def test_symbol_match_controls_are_explained_and_forwarded() -> None:
    fake = FakeApplication()
    app = _make_app(fake)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#mode-select", Select).value = "symbol"
        await pilot.pause()
        match = app.query_one("#match-select", Select)
        assert match.display
        match.value = "contains"
        app.query_one("#query-input", Input).value = "validate"
        app.action_submit_query()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert fake.symbol_calls[-1]["match"] == "contains"
        assert "symbol" in app.query_one("#query-input", Input).placeholder.lower()


@pytest.mark.asyncio
async def test_outline_navigation_and_back_restore_selected_entry(tmp_path: Path) -> None:
    from textual.widgets import Tabs

    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.py").write_text("def main():\n    pass\n")
    fake = FakeApplication(projects=[_sample_project(root=tmp_path)])
    app = _make_app(fake)
    async with app.run_test(size=(120, 32)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#query-input", Input).value = "main"
        app.action_submit_query()
        await app.workers.wait_for_complete()
        await pilot.pause(0.3)
        await app.workers.wait_for_complete()
        assert "Preview:" in str(app.query_one("#detail-title", Label).render())
        app.query_one("#detail-tabs", Tabs).active = "outline-tab"
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        entries = app.query_one("#detail-list", OptionList)
        assert entries.display and entries.option_count == 1
        entries.focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "Working tree:" in str(app.query_one("#detail-title", Label).render())
        await pilot.press("escape")
        await pilot.pause()
        assert "Outline:" in str(app.query_one("#detail-title", Label).render())
        assert entries.highlighted == 0
        assert app.focused is entries


@pytest.mark.asyncio
async def test_reference_and_impact_destinations_open_and_return(tmp_path: Path) -> None:
    from code_indexing_mcp.models import ImpactEdge, ImpactLayer

    (tmp_path / "src").mkdir()
    (tmp_path / "src/runner.py").write_text("\n" * 4 + "main()\n")
    fake = FakeApplication(projects=[_sample_project(root=tmp_path)])
    app = _make_app(fake)
    async with app.run_test(size=(120, 32)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#query-input", Input).value = "main"
        app.action_submit_query()
        await app.workers.wait_for_complete()
        await pilot.pause(0.3)
        await app.workers.wait_for_complete()
        app.action_show_references()
        await app.workers.wait_for_complete()
        await pilot.pause()
        entries = app.query_one("#detail-list", OptionList)
        entries.focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "runner.py" in str(app.query_one("#detail-title", Label).render())
        assert "main()" in app.query_one("#detail-content", Static).content.code
        await pilot.press("escape")
        assert "References:" in str(app.query_one("#detail-title", Label).render())
        assert entries.highlighted == 0

        impact = fake.impact_radius(app.service.to_selector(_sample_hit()))
        target = impact.selected.model_copy(update={"chunk_id": "chk-1"})
        impact = impact.model_copy(
            update={
                "layers": [
                    ImpactLayer(
                        depth=1,
                        edges=[ImpactEdge(source=impact.selected, target=target, kinds=["call"])],
                    )
                ]
            }
        )
        app._render_impact(impact)
        entries.focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "Preview:" in str(app.query_one("#detail-title", Label).render())


@pytest.mark.asyncio
async def test_lazy_index_progress_does_not_overwrite_error() -> None:
    import threading

    release = threading.Event()
    started = threading.Event()
    fake = FakeApplication()
    fake.statuses["proj-1"] = "stale"
    original = fake.index_project

    def delayed(*args: object, **kwargs: object):
        started.set()
        release.wait(4)
        return original("proj-1")

    fake.index_project = delayed  # type: ignore[method-assign]
    app = _make_app(fake)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#query-input", Input).value = "main"
        app.action_submit_query()
        await pilot.pause(0.4)
        assert started.is_set()
        progress = app.query_one("#progress-bar", Static)
        assert "Preparing index" in str(progress.render())
        app._show_error("A recoverable problem")
        await pilot.pause(0.4)
        assert "A recoverable problem" in str(app.query_one("#error-content", Static).render())
        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.query_one("#error-panel").display


@pytest.mark.asyncio
async def test_help_and_copy_location_are_available(tmp_path: Path) -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.press("ctrl+h")
        assert "Describe code" in str(app.screen.query_one("#help-content", Static).render())
        await pilot.press("escape")
        app.query_one("#query-input", Input).value = "main"
        app.action_submit_query()
        await app.workers.wait_for_complete()
        await pilot.pause(0.3)
        await app.workers.wait_for_complete()
        await pilot.press("y")
        assert app.clipboard == "src/main.py:10"


@pytest.mark.asyncio
async def test_impact_explains_incomplete_results() -> None:
    from code_indexing_mcp.models import ReferenceLimitation

    fake = FakeApplication()
    app = _make_app(fake)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        impact = fake.impact_radius(app.service.to_selector(_sample_hit())).model_copy(
            update={
                "budget_exhausted": True,
                "limitations": [
                    ReferenceLimitation(code="dynamic", explanation="Dynamic calls omitted")
                ],
            }
        )
        app._render_impact(impact)
        text = str(app.query_one("#detail-content", Static).render())
        assert "Dynamic calls omitted" in text
        assert "budget" in text.lower()
        assert "depth" in text.lower()


@pytest.mark.asyncio
async def test_help_can_remain_open_during_search_completion() -> None:
    import threading

    release = threading.Event()
    fake = FakeApplication()
    original = fake.search_code

    def delayed(*args: object, **kwargs: object) -> SearchResponse:
        release.wait(3)
        return original("main")

    fake.search_code = delayed  # type: ignore[method-assign]
    app = _make_app(fake)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#query-input", Input).value = "main"
        app.action_submit_query()
        await pilot.press("ctrl+h")
        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause(0.3)
        await app.workers.wait_for_complete()
        assert app.screen.query_one("#help-content", Static)
        await pilot.press("escape")
        assert app.query_one("#results-list", OptionList).option_count == 1


@pytest.mark.asyncio
async def test_editor_action_uses_selected_location(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from contextlib import nullcontext
    from subprocess import CompletedProcess

    import code_indexing_mcp.tui.app as app_module

    fake = FakeApplication(projects=[_sample_project(root=tmp_path)])
    app = _make_app(fake)
    calls: list[list[str]] = []
    monkeypatch.setenv("VISUAL", "code --wait")
    monkeypatch.setattr(app, "suspend", nullcontext)

    def run(command: list[str], *, check: bool) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0)

    monkeypatch.setattr(app_module.subprocess, "run", run)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#query-input", Input).value = "main"
        app.action_submit_query()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.press("e")
        assert calls == [["code", "--wait", "--goto", f"{tmp_path}/src/main.py:10"]]


@pytest.mark.asyncio
async def test_long_error_keeps_preview_and_footer_inside_screen() -> None:
    app = _make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#query-input", Input).value = "main"
        app.action_submit_query()
        await app.workers.wait_for_complete()
        app.action_open_selected()
        await app.workers.wait_for_complete()
        app._show_error("Cannot connect to daemon. " * 20)
        await pilot.pause()
        preview = app.query_one("#detail-scroll")
        errors = app.query_one("#error-panel")
        assert preview.region.bottom <= errors.region.y
        assert preview.region.height >= 2
        assert app.query_one("#context-help").region.bottom <= 23


@pytest.mark.asyncio
async def test_pane_titles_have_visible_text_rows() -> None:
    app = _make_app()
    async with app.run_test(size=(120, 32)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.query_one("#results-title", Label).content_region.height >= 1
        assert app.query_one("#detail-title", Label).content_region.height >= 1


@pytest.mark.asyncio
async def test_failed_navigation_preserves_target_and_history() -> None:
    from code_indexing_mcp.tui.navigation import SourceLocation

    app = _make_app()
    async with app.run_test(size=(120, 32)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#query-input", Input).value = "main"
        app.action_submit_query()
        await app.workers.wait_for_complete()
        await pilot.pause(0.3)
        await app.workers.wait_for_complete()
        app.action_show_outline()
        await app.workers.wait_for_complete()
        await pilot.pause()
        target = app._active_target
        history_count = len(app._history)
        app._set_detail_entries([(Text("Missing source"), SourceLocation("missing.py", 1))])
        app.query_one("#detail-list", OptionList).focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._active_target == target
        assert app._detail_mode == "outline"
        assert len(app._history) == history_count
        assert "Outline:" in str(app.query_one("#detail-title", Label).render())


@pytest.mark.asyncio
async def test_results_shortcuts_use_focused_result_after_drilldown(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/runner.py").write_text("\n" * 4 + "main()\n")
    fake = FakeApplication(projects=[_sample_project(root=tmp_path)])
    app = _make_app(fake)
    async with app.run_test(size=(120, 32)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#query-input", Input).value = "main"
        app.action_submit_query()
        await app.workers.wait_for_complete()
        await pilot.pause(0.3)
        await app.workers.wait_for_complete()
        app.action_show_references()
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#detail-list", OptionList).focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#results-list", OptionList).focus()
        await pilot.press("o")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "src/main.py" in str(app.query_one("#detail-title", Label).render())
        app.query_one("#results-list", OptionList).focus()
        await pilot.press("r")
        await app.workers.wait_for_complete()
        assert fake.references_calls[-1]["selector"].chunk_id == "chk-1"


@pytest.mark.asyncio
async def test_lazy_search_refreshes_header_state() -> None:
    fake = FakeApplication()
    fake.statuses["proj-1"] = "stale"
    app = _make_app(fake)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#query-input", Input).value = "main"
        app.action_submit_query()
        await app.workers.wait_for_complete()
        await pilot.pause(0.3)
        await app.workers.wait_for_complete()
        assert "State: ready" in str(app.query_one("#header-status", Label).render())


@pytest.mark.asyncio
async def test_error_retry_remains_bound_to_failed_action() -> None:
    app = _make_app()
    calls: list[str] = []
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app._show_error("Could not complete action", retry=lambda: calls.append("failed action"))
        await pilot.pause()
        await pilot.click("#retry-button")
        assert calls == ["failed action"]
