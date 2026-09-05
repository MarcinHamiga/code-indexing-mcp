"""Regression tests for callbacks that can run while the TUI is shutting down."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError
from types import SimpleNamespace

import pytest
from test_tui import _make_app
from test_tui_service import _sample_hit, _sample_project

from code_indexing_mcp.tui.navigation import SourceLocation


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


@pytest.mark.asyncio
async def test_queued_worker_callback_is_dropped_during_screen_teardown() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        queued: list[Callable[[], None]] = []

        def capture(callback: Callable[[], None]) -> None:
            queued.append(callback)

        app.call_from_thread = capture  # type: ignore[method-assign]
        app._detail_request_id = 1
        app._pending_detail = (SourceLocation("src/main.py", 1), "chunk", None)
        app._call_from_worker(app._apply_detail, 1, lambda value: None, None)
        assert len(queued) == 1

        await app.query_one("#detail-list").remove()
        await app.query_one("#detail-tabs").remove()
        assert app.screen_stack and app.is_running
        app._running = False

        queued[0]()

        assert app._pending_detail is not None
        app._running = True


def test_worker_callback_submitted_after_shutdown_is_ignored() -> None:
    app = _make_app()
    delivered = False

    def callback() -> None:
        nonlocal delivered
        delivered = True

    app._running = False
    app._call_from_worker(callback)

    assert not delivered


@pytest.mark.asyncio
async def test_actions_are_rejected_after_shutdown() -> None:
    app = _make_app()
    app._running = False

    assert await app.run_action("submit_query") is False


@pytest.mark.asyncio
async def test_actions_are_allowed_while_running() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()

        assert await app.run_action("focus_query") is True


def test_worker_callback_forwards_args_and_kwargs_while_running() -> None:
    app = _make_app()
    received: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def callback(*args: object, **kwargs: object) -> None:
        received.append((args, kwargs))

    def deliver(callback: Callable[[], None]) -> None:
        callback()

    app.call_from_thread = deliver  # type: ignore[method-assign]
    app._running = True
    app._call_from_worker(callback, 1, 2, label="live")

    assert received == [((1, 2), {"label": "live"})]


@pytest.mark.parametrize("error", [RuntimeError, CancelledError])
def test_live_worker_delivery_errors_are_not_swallowed(
    error: type[Exception],
) -> None:
    app = _make_app()

    def fail(callback: Callable[[], None]) -> None:
        raise error("delivery failed")

    app.call_from_thread = fail  # type: ignore[method-assign]
    app._running = True

    with pytest.raises(error, match="delivery failed"):
        app._call_from_worker(lambda: None)


@pytest.mark.parametrize("error", [RuntimeError, CancelledError])
def test_worker_submission_errors_are_ignored_after_shutdown(
    error: type[Exception],
) -> None:
    app = _make_app()

    def fail(callback: Callable[[], None]) -> None:
        app._running = False
        raise error("submission failed")

    app.call_from_thread = fail  # type: ignore[method-assign]
    app._running = True
    app._call_from_worker(lambda: None)
