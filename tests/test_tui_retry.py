"""Regression tests for operation-specific TUI retries."""

from __future__ import annotations

import asyncio
import threading

import pytest
from test_tui import _make_app
from test_tui_service import FakeApplication
from textual.widgets import Input


@pytest.mark.asyncio
async def test_search_retry_keeps_original_query_when_detail_starts() -> None:
    fake = FakeApplication()
    started = threading.Event()
    release = threading.Event()
    attempts = 0
    queries: list[str] = []
    original_search = fake.search_code

    def failing_search(*args: object, **kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        queries.append(str(args[0]))
        if attempts == 1:
            started.set()
            release.wait(timeout=5)
            raise RuntimeError("search unavailable")
        return original_search(*args, **kwargs)  # type: ignore[arg-type]

    fake.search_code = failing_search  # type: ignore[method-assign]
    app = _make_app(fake)

    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        app.query_one("#query-input", Input).value = "main"
        app.action_submit_query()
        assert await asyncio.to_thread(started.wait, 2)
        app.query_one("#query-input", Input).value = "changed later"
        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.click("#retry-button")
        await app.workers.wait_for_complete()

        assert attempts == 2
        assert queries == ["main", "main"]


@pytest.mark.asyncio
async def test_index_retry_stays_bound_when_preview_starts() -> None:
    started = threading.Event()
    release = threading.Event()
    fake = FakeApplication()

    def failing_index(*args: object, **kwargs: object) -> object:
        fake.index_calls.append({"project_id": "proj-1"})
        started.set()
        release.wait(timeout=5)
        raise RuntimeError("index unavailable")

    fake.index_project = failing_index  # type: ignore[method-assign]
    app = _make_app(fake)

    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#query-input", Input).value = "main"
        app.action_submit_query()
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.action_trigger_index()
        assert await asyncio.to_thread(started.wait, 2)
        app.action_open_selected()
        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.click("#retry-button")
        await app.workers.wait_for_complete()

        assert len(fake.index_calls) == 2
