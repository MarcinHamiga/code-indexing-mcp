"""Headless tests for the Textual installer wizard."""

from pathlib import Path

import pytest
from textual.pilot import Pilot

from incode_mcp.installer.tui.app import InstallerApp
from incode_mcp.installer.wizard import WizardState


async def click(pilot: Pilot, selector: str) -> None:
    """Click and wait out the button's active-effect timer (0.2s by default).

    Textual ignores a click landing while the button still has its -active
    class from the previous one, so pilot clicks must be paced.
    """

    await pilot.click(selector)
    await pilot.pause(0.4)


def _install_state(tmp_path: Path) -> WizardState:
    return WizardState.for_install(tmp_path, "https://example.invalid/repo.git", home=tmp_path)


def _reconfigure_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WizardState:
    import incode_mcp.installer.wizard as wizard

    monkeypatch.setattr(wizard.accelerator, "prepared_accelerator", lambda directory: None)
    return WizardState.for_reconfigure(tmp_path, home=tmp_path)


@pytest.mark.asyncio
async def test_install_wizard_walks_forward_and_back(tmp_path: Path) -> None:
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        assert app.current == "welcome"
        await click(pilot, "#next")
        assert app.current == "location"
        await click(pilot, "#next")
        assert app.current == "accelerator"
        await click(pilot, "#back")
        assert app.current == "location"
        assert app.done_code is None


@pytest.mark.asyncio
async def test_reconfigure_skips_the_location_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = InstallerApp(_reconfigure_state(tmp_path, monkeypatch))
    async with app.run_test() as pilot:
        await click(pilot, "#next")
        assert app.current == "accelerator"


@pytest.mark.asyncio
async def test_cancel_exits_with_130(tmp_path: Path) -> None:
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        await click(pilot, "#cancel")
    assert app.return_code == 130


@pytest.mark.asyncio
async def test_location_commit_updates_state(tmp_path: Path) -> None:
    from textual.widgets import Input

    state = _install_state(tmp_path)
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        await click(pilot, "#next")
        target = tmp_path / "custom"
        app.query_one("#install-dir", Input).value = str(target)
        await click(pilot, "#next")
        assert state.install_directory == target


@pytest.mark.asyncio
async def test_location_rejects_an_empty_directory(tmp_path: Path) -> None:
    from textual.widgets import Input

    state = _install_state(tmp_path)
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        await click(pilot, "#next")
        app.query_one("#install-dir", Input).value = "   "
        await click(pilot, "#next")
        assert app.current == "location"  # blocked


@pytest.mark.asyncio
async def test_accelerator_panel_shows_detection_and_commits_choice(tmp_path: Path) -> None:
    from textual.widgets import RadioButton, Static

    state = _install_state(tmp_path)
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        await click(pilot, "#next")
        await click(pilot, "#next")
        assert app.current == "accelerator"
        text = str(app.query_one("#detection", Static).render())
        assert "Platform:" in text
        app.query_one("#accel-cpu", RadioButton).toggle()
        await click(pilot, "#next")
        assert state.accelerator == "cpu"


@pytest.mark.asyncio
async def test_reconfigure_offers_keep_prepared_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from textual.widgets import RadioButton

    app = InstallerApp(_reconfigure_state(tmp_path, monkeypatch))
    async with app.run_test() as pilot:
        await click(pilot, "#next")
        assert app.current == "accelerator"
        assert app.query_one("#accel-keep", RadioButton).value is True
        await click(pilot, "#next")
        assert app.state.accelerator is None


@pytest.mark.asyncio
async def test_harnesses_panel_commits_checked_slugs(tmp_path: Path) -> None:
    from textual.widgets import Checkbox

    state = _install_state(tmp_path)
    state.harness_slugs = []
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        for _ in range(3):
            await click(pilot, "#next")
        assert app.current == "harnesses"
        app.query_one("#harness-kimi-code", Checkbox).toggle()
        app.query_one("#harness-codex", Checkbox).toggle()
        await click(pilot, "#next")
        assert state.harness_slugs == ["codex", "kimi-code"]
