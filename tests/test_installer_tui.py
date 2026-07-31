"""Headless tests for the Textual installer wizard."""

from pathlib import Path

import pytest
from textual.pilot import Pilot
from textual.widgets import Static

from code_indexing_mcp.installer.tui.app import InstallerApp
from code_indexing_mcp.installer.wizard import WizardState


async def click(pilot: Pilot, selector: str) -> None:
    """Click and wait out the button's active-effect timer (0.2s by default).

    Textual ignores a click landing while the button still has its -active
    class from the previous one, so pilot clicks must be paced.
    """

    await pilot.click(selector)
    await pilot.pause(0.4)


def _prepare_checkout(directory: Path) -> Path:
    """Create the server command the Location panel requires to exist."""

    from code_indexing_mcp.installer.accelerator import server_executable

    command = server_executable(directory)
    command.parent.mkdir(parents=True, exist_ok=True)
    command.touch()
    return directory


def _install_state(tmp_path: Path) -> WizardState:
    return WizardState.for_install(_prepare_checkout(tmp_path), home=tmp_path)


def _reconfigure_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WizardState:
    import code_indexing_mcp.installer.wizard as wizard

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
    target = _prepare_checkout(tmp_path / "custom")
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        await click(pilot, "#next")
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
async def test_location_rejects_a_directory_without_an_installation(tmp_path: Path) -> None:
    from textual.widgets import Input, Label

    state = _install_state(tmp_path)
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        await click(pilot, "#next")
        app.query_one("#install-dir", Input).value = str(tmp_path / "not-an-install")
        await click(pilot, "#next")
        assert app.current == "location"  # blocked
        assert "No prepared installation" in str(app.query_one("#location-error", Label).render())
        assert state.install_directory == tmp_path  # unchanged


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


@pytest.mark.asyncio
async def test_settings_panels_validate_and_commit(tmp_path: Path) -> None:
    from textual.widgets import Input

    state = _install_state(tmp_path)
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        for _ in range(4):
            await click(pilot, "#next")
        assert app.current == "indexing"
        field = app.query_one("#f-CODE_INDEXING_INDEX_WAIT_SECONDS", Input)
        field.value = "99999999"
        await click(pilot, "#next")
        assert app.current == "indexing"  # blocked by validation
        field.value = "60"
        await click(pilot, "#next")
        assert app.current == "embedding"
        assert state.values["CODE_INDEXING_INDEX_WAIT_SECONDS"] == "60"


@pytest.mark.asyncio
async def test_settings_widgets_render_hand_written_values(tmp_path: Path) -> None:
    """Select raises on a value outside its options, so nothing may reach it raw."""

    from textual.widgets import Checkbox, Select

    state = _install_state(tmp_path)
    state.values["CODE_INDEXING_OFFLINE"] = "true"
    state.values["CODE_INDEXING_INDEX_MODE"] = "EAGER"
    state.values["CODE_INDEXING_BROKER"] = "nonsense"
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        for _ in range(4):
            await click(pilot, "#next")
        assert app.current == "indexing"
        assert app.query_one("#f-CODE_INDEXING_OFFLINE", Checkbox).value is True
        assert app.query_one("#f-CODE_INDEXING_INDEX_MODE", Select).value == "eager"
        assert app.query_one("#f-CODE_INDEXING_BROKER", Select).value == "auto"  # fell back
        await click(pilot, "#next")
        assert app.current == "embedding"
        assert state.values["CODE_INDEXING_OFFLINE"] == "1"
        assert state.values["CODE_INDEXING_INDEX_MODE"] == "eager"


@pytest.mark.asyncio
async def test_summary_lists_updates_and_target_files(tmp_path: Path) -> None:
    from textual.widgets import Static

    state = _install_state(tmp_path)
    state.values["CODE_INDEXING_OFFLINE"] = "1"
    state.harness_slugs = ["kimi-code"]
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        for _ in range(6):
            await click(pilot, "#next")
        assert app.current == "summary"
        text = str(app.query_one("#summary-body", Static).render())
        assert "CODE_INDEXING_OFFLINE" in text
        assert "mcp.json" in text
        assert "auto" in text  # accelerator choice


@pytest.mark.asyncio
async def test_summary_warns_about_accelerator_disk_cost(tmp_path: Path) -> None:
    from textual.widgets import Static

    state = _install_state(tmp_path)
    state.accelerator = "mlx"
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        for _ in range(6):
            await click(pilot, "#next")
        text = str(app.query_one("#summary-body", Static).render())
        assert "gigabytes" in text


def _fake_result(failures: tuple = ()):  # type: ignore[no-untyped-def]
    from code_indexing_mcp.installer.accelerator import AcceleratorPlan
    from code_indexing_mcp.installer.orchestrator import InstallResult

    return InstallResult(
        AcceleratorPlan("cpu", "CPU was requested"),
        (("kimi-code", Path("/home/u/.kimi-code/mcp.json")),),
        failures,
        (("kimi-code", "2 linked, 2 already installed"),),
    )


@pytest.mark.asyncio
async def test_progress_runs_pipeline_and_finishes_on_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import code_indexing_mcp.installer.tui.panels as panels
    from code_indexing_mcp.installer.orchestrator import StepEvent

    def fake_run_install(plan, on_event=lambda event: None, should_continue=lambda: True):  # type: ignore[no-untyped-def]
        on_event(StepEvent("accelerator", "started", "auto"))
        on_event(StepEvent("accelerator", "finished", "cpu (ok)"))
        return _fake_result()

    monkeypatch.setattr(panels, "run_install", fake_run_install)
    state = _install_state(tmp_path)
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        for _ in range(7):
            await click(pilot, "#next")
        await pilot.pause()
        assert app.current == "done"
        assert app.done_code == 0
        body = str(app.query_one("#done-body", Static).render())
        assert "mcp.json" in body


@pytest.mark.asyncio
async def test_done_reports_failures_with_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import code_indexing_mcp.installer.tui.panels as panels

    monkeypatch.setattr(
        panels,
        "run_install",
        lambda plan, on_event=None, should_continue=None: _fake_result((("codex", "broken"),)),
    )
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        for _ in range(7):
            await click(pilot, "#next")
        await pilot.pause()
        assert app.done_code == 1
        assert "codex" in str(app.query_one("#done-body", Static).render())


@pytest.mark.asyncio
async def test_pipeline_error_finishes_with_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import code_indexing_mcp.installer.tui.panels as panels
    from code_indexing_mcp.installer.config_files import InstallerError

    def explode(plan, on_event=None, should_continue=None):  # type: ignore[no-untyped-def]
        raise InstallerError("boom")

    monkeypatch.setattr(panels, "run_install", explode)
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        for _ in range(7):
            await click(pilot, "#next")
        await pilot.pause()
        assert app.done_code == 1
        assert "boom" in str(app.query_one("#done-body", Static).render())
