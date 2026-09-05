"""Regression tests for callbacks that can run while the TUI is shutting down."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_tui import _make_app
from test_tui_service import _sample_hit, _sample_project


def test_focus_callback_is_safe_after_main_screen_teardown() -> None:
    app = _make_app()
    event = SimpleNamespace(widget=SimpleNamespace(id="query-input"))

    # Textual removes the base screen before already queued focus events run.
    assert app.screen_stack == []
    app.on_descendant_focus(event)


def test_preview_callback_is_safe_after_main_screen_teardown() -> None:
    app = _make_app()
    app._hits = [_sample_hit()]
    app._highlighted_index = 0

    # A debounce timer may already have queued this callback when the app exits.
    assert app.screen_stack == []
    app._preview_highlighted()


@pytest.mark.asyncio
async def test_exit_cancels_delayed_callbacks_and_invalidates_requests() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._hits = [_sample_hit()]
        app._highlighted_index = 0
        app.on_option_list_option_highlighted(
            SimpleNamespace(option_list=SimpleNamespace(id="results-list"), option_index=0)
        )
        assert app._preview_timer is not None
        request_ids = (
            app._search_request_id,
            app._detail_request_id,
            app._project_request_id,
        )

        app.exit()

    assert app._preview_timer is None
    assert app._activity_timer is None
    assert (
        app._search_request_id,
        app._detail_request_id,
        app._project_request_id,
    ) == tuple(request_id + 1 for request_id in request_ids)


@pytest.mark.asyncio
async def test_project_callbacks_are_safe_after_partial_screen_teardown() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        project = app.service.selected_project
        assert project is not None
        status = app.service.project_status(project)
        request_id = app._project_request_id

        await app.query_one("#header-title").remove()
        await app.query_one("#status-bar").remove()
        assert app.screen_stack

        app._apply_project_status(request_id, project, status)
        app._set_status("late status update")
        app._project_error(request_id, "late project status failure")


def test_retry_button_callback_is_safe_after_main_screen_teardown() -> None:
    app = _make_app()
    project = _sample_project()
    app._error_retry = lambda: app._start_search("main", "semantic", project, "exact")
    event = SimpleNamespace(button=SimpleNamespace(id="retry-button"))

    assert app.screen_stack == []
    app.on_button_pressed(event)
