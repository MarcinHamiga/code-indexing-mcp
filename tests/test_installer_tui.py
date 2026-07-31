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
