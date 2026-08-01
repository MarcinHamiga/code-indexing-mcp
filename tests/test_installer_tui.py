"""Headless tests for the Textual installer wizard."""

from pathlib import Path

import pytest
from textual.pilot import Pilot
from textual.widgets import Label, Static

from code_indexing_mcp.installer.tui.app import InstallerApp
from code_indexing_mcp.installer.tui.panels import PathPanel
from code_indexing_mcp.installer.wizard import WizardState


async def click(pilot: Pilot, selector: str) -> None:
    """Click and wait out the button's active-effect timer (0.2s by default).

    Textual ignores a click landing while the button still has its -active
    class from the previous one, so pilot clicks must be paced.
    """

    await pilot.click(selector)
    await pilot.pause(0.4)


async def advance_to(pilot: Pilot, app: InstallerApp, panel: str) -> None:
    """Click Next until ``panel`` is showing.

    Counting clicks would make every test in this file wrong the moment a step is
    added between two existing ones; naming the destination keeps them honest.
    """

    while app.current != panel:
        previous = app.current
        await click(pilot, "#next")
        if app.current == previous:
            raise AssertionError(f"navigation stopped on {previous!r} before reaching {panel!r}")


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
async def test_keyboard_navigates_the_wizard(tmp_path: Path) -> None:
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        await pilot.press("ctrl+n")
        assert app.current == "location"
        await pilot.press("ctrl+n")
        assert app.current == "accelerator"
        await pilot.press("ctrl+b")
        assert app.current == "location"
        await pilot.press("escape")
    assert app.return_code == 130


@pytest.mark.asyncio
async def test_keyboard_navigation_is_locked_while_installing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Escape must not abandon a run that is already writing to the user's files."""

    import code_indexing_mcp.installer.tui.panels as panels

    monkeypatch.setattr(
        panels,
        "run_install",
        lambda plan, on_event=None, should_continue=None: _fake_result(),
    )
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        await advance_to(pilot, app, "summary")
        await click(pilot, "#next")
        await pilot.pause()
        assert app.current == "done"
        await pilot.press("escape")
        await pilot.press("ctrl+b")
        assert app.current == "done"
        assert app.return_code is None  # still on the Done screen, not exited


@pytest.mark.asyncio
async def test_header_counts_the_steps_the_user_walks(tmp_path: Path) -> None:
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        assert app.sub_title == "Step 1 of 8 - Welcome"
        await click(pilot, "#next")
        assert app.sub_title == "Step 2 of 8 - Install location"


@pytest.mark.asyncio
async def test_reconfigure_drops_the_skipped_panel_from_the_step_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = InstallerApp(_reconfigure_state(tmp_path, monkeypatch))
    async with app.run_test():
        assert app.sub_title == "Step 1 of 7 - Welcome"


@pytest.mark.asyncio
async def test_panels_land_focus_on_their_first_control(tmp_path: Path) -> None:
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        await advance_to(pilot, app, "harnesses")
        focused = app.focused
        assert focused is not None
        # The first harness checkbox, not a nav button and not the Advanced input
        # hidden inside a collapsed section.
        assert focused.id == "harness-codex"


@pytest.mark.asyncio
async def test_summary_jumps_straight_back_to_a_named_panel(tmp_path: Path) -> None:
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        await advance_to(pilot, app, "summary")
        await click(pilot, "#jump-accelerator")
        assert app.current == "accelerator"


@pytest.mark.asyncio
async def test_reconfigure_skips_the_location_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = InstallerApp(_reconfigure_state(tmp_path, monkeypatch))
    async with app.run_test() as pilot:
        await click(pilot, "#next")
        assert app.current == "accelerator"


@pytest.mark.asyncio
async def test_reconfigure_points_at_repair_when_a_piece_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # for_reconfigure does not require a built checkout, so this state describes
    # an installation whose executable and launcher have both gone.
    app = InstallerApp(_reconfigure_state(tmp_path, monkeypatch))
    async with app.run_test():
        text = str(app.query_one("#welcome-repair", Static).render())
        assert "no server executable" in text


@pytest.mark.asyncio
async def test_a_healthy_install_says_nothing_about_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from textual.css.query import NoMatches

    state = _reconfigure_state(_prepare_checkout(tmp_path), monkeypatch)
    launcher = state.bin_directory / "code-indexing-mcp"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.touch()
    app = InstallerApp(state)
    async with app.run_test():
        with pytest.raises(NoMatches):
            app.query_one("#welcome-repair", Static)


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
        await advance_to(pilot, app, "harnesses")
        app.query_one("#harness-kimi-code", Checkbox).toggle()
        app.query_one("#harness-codex", Checkbox).toggle()
        await click(pilot, "#next")
        assert state.harness_slugs == ["codex", "kimi-code"]


@pytest.mark.asyncio
async def test_path_panel_commits_launcher_choices(tmp_path: Path) -> None:
    from textual.widgets import Checkbox, Input

    state = _install_state(tmp_path)
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        await advance_to(pilot, app, "path")
        assert app.query_one("#path-launcher", Checkbox).value is True
        app.query_one("#path-bin-dir", Input).value = str(tmp_path / "custom-bin")
        await click(pilot, "#next")
        assert app.current == "indexing"
        assert state.bin_directory == tmp_path / "custom-bin"
        assert state.install_launcher is True
        # commit() creates the directory so the pipeline is not the first to find
        # out it cannot be written.
        assert (tmp_path / "custom-bin").is_dir()


@pytest.mark.asyncio
async def test_path_panel_rejects_an_empty_directory(tmp_path: Path) -> None:
    from textual.widgets import Input, Label

    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        await advance_to(pilot, app, "path")
        app.query_one("#path-bin-dir", Input).value = "  "
        await click(pilot, "#next")
        assert app.current == "path"  # blocked
        assert "cannot be empty" in str(app.query_one("#path-error", Label).render())


@pytest.mark.asyncio
async def test_path_panel_allows_an_empty_directory_with_no_launcher(tmp_path: Path) -> None:
    """Nothing will be put anywhere, so there is nothing to demand a path for."""

    from textual.widgets import Checkbox, Input

    state = _install_state(tmp_path)
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        await advance_to(pilot, app, "path")
        app.query_one("#path-launcher", Checkbox).value = False
        app.query_one("#path-bin-dir", Input).value = ""
        await click(pilot, "#next")
        assert app.current == "indexing"
        assert state.install_launcher is False


@pytest.mark.asyncio
async def test_path_panel_disables_the_profile_edit_when_already_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from textual.widgets import Checkbox, Static

    import code_indexing_mcp.installer.shell_path as shell_path

    monkeypatch.setattr(shell_path, "is_on_path", lambda directory, **kwargs: True)
    state = _install_state(tmp_path)
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        await advance_to(pilot, app, "path")
        profile = app.query_one("#path-profile", Checkbox)
        assert profile.value is False and profile.disabled is True
        assert "already on PATH" in str(app.query_one("#path-status", Static).render())
        await click(pilot, "#next")
        # The box was unchecked by the panel for display, not by the user, so
        # their answer is left as it was rather than overwritten with False.
        assert state.modify_shell_profiles is True


@pytest.mark.asyncio
async def test_path_panel_restores_the_profile_choice_when_the_directory_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moving off an on-PATH directory must not leave the box silently unchecked."""

    from textual.widgets import Checkbox, Input

    import code_indexing_mcp.installer.shell_path as shell_path

    on_path = {str(tmp_path / ".local" / "bin")}
    monkeypatch.setattr(
        shell_path, "is_on_path", lambda directory, **kwargs: str(directory) in on_path
    )
    state = _install_state(tmp_path)
    state.bin_directory = tmp_path / ".local" / "bin"
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        await advance_to(pilot, app, "path")
        profile = app.query_one("#path-profile", Checkbox)
        assert profile.disabled is True

        app.query_one("#path-bin-dir", Input).value = str(tmp_path / "elsewhere")
        await pilot.pause(PathPanel.INSPECT_DEBOUNCE_SECONDS + 0.2)

        assert profile.disabled is False
        assert profile.value is True


@pytest.mark.asyncio
async def test_summary_lists_the_launcher_and_the_profile_it_will_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import code_indexing_mcp.installer.shell_path as shell_path

    profile = tmp_path / ".zshrc"
    profile.write_text("", encoding="utf-8")
    monkeypatch.setattr(shell_path, "is_on_path", lambda directory, **kwargs: False)
    monkeypatch.setattr(shell_path, "shell_profiles", lambda **kwargs: (profile,))
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        await advance_to(pilot, app, "summary")
        text = str(app.query_one("#summary-body", Static).render())
        assert "code-indexing-mcp" in text
        assert str(profile) in text


@pytest.mark.asyncio
async def test_settings_panels_validate_and_commit(tmp_path: Path) -> None:
    from textual.widgets import Input

    state = _install_state(tmp_path)
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        await advance_to(pilot, app, "indexing")
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
        await advance_to(pilot, app, "indexing")
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
        await advance_to(pilot, app, "summary")
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
        await advance_to(pilot, app, "summary")
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
        await advance_to(pilot, app, "summary")
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
        await advance_to(pilot, app, "summary")
        await click(pilot, "#next")
        await pilot.pause()
        # A failure holds the progress screen so the run can be retried; the
        # exit code is only decided once the user moves on.
        assert app.current == "progress"
        assert app.done_code is None
        await click(pilot, "#progress-continue")
        assert app.current == "done"
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
        await advance_to(pilot, app, "summary")
        await click(pilot, "#next")
        await pilot.pause()
        assert app.current == "progress"
        await click(pilot, "#progress-continue")
        assert app.done_code == 1
        assert "boom" in str(app.query_one("#done-body", Static).render())


@pytest.mark.asyncio
async def test_done_panel_tells_the_user_how_to_reach_the_new_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import code_indexing_mcp.installer.tui.panels as panels
    from code_indexing_mcp.installer.orchestrator import InstallResult
    from code_indexing_mcp.installer.shell_path import LauncherResult

    launcher = LauncherResult(tmp_path / "bin" / "code-indexing-mcp", "created", "points at it")
    profile = tmp_path / ".zshrc"
    monkeypatch.setattr(
        panels,
        "run_install",
        lambda plan, on_event=None, should_continue=None: InstallResult(
            None, (), (), (), launcher=launcher, profiles_updated=(profile,)
        ),
    )
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        await advance_to(pilot, app, "summary")
        await click(pilot, "#next")
        await pilot.pause()
        body = str(app.query_one("#done-body", Static).render())
        assert str(launcher.path) in body
        assert str(profile) in body
        # The PATH entry is not live in the shell the installer was started from.
        assert "exec" in body


@pytest.mark.asyncio
async def test_progress_shows_a_row_per_step_and_marks_what_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import code_indexing_mcp.installer.tui.panels as panels
    from code_indexing_mcp.installer.orchestrator import StepEvent

    def fake_run_install(plan, on_event=lambda event: None, should_continue=lambda: True):  # type: ignore[no-untyped-def]
        on_event(StepEvent("accelerator", "started", "auto"))
        on_event(StepEvent("accelerator", "finished", "cpu (ok)"))
        on_event(StepEvent("path", "skipped", "launcher not requested"))
        on_event(StepEvent("harnesses", "started", "kimi-code"))
        on_event(StepEvent("harnesses", "finished", "kimi-code: /x/mcp.json"))
        return _fake_result()

    monkeypatch.setattr(panels, "run_install", fake_run_install)
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        await advance_to(pilot, app, "summary")
        progress = app.query_one("#progress", panels.ProgressPanel)
        await click(pilot, "#next")
        await pilot.pause()
        rows = str(app.query_one("#progress-steps", Static).render())
        assert "Prepare the embedding accelerator" in rows
        assert progress._status["accelerator"] == "finished"
        assert progress._status["path"] == "skipped"
        assert progress._status["harnesses"] == "finished"
        # Nothing reported for these, so they never ran rather than silently passing.
        assert progress._status["verify"] == "skipped"
        # The raw event stream is still there, just folded away.
        assert app.query_one("#progress-details", panels.Collapsible).collapsed is True


@pytest.mark.asyncio
async def test_progress_keeps_the_worst_outcome_a_step_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A step that configures four clients and breaks on one must not read "ok"."""

    import code_indexing_mcp.installer.tui.panels as panels
    from code_indexing_mcp.installer.orchestrator import StepEvent

    def fake_run_install(plan, on_event=lambda event: None, should_continue=lambda: True):  # type: ignore[no-untyped-def]
        on_event(StepEvent("harnesses", "started", "codex, kimi-code"))
        on_event(StepEvent("harnesses", "failed", "codex: unwritable"))
        on_event(StepEvent("harnesses", "finished", "kimi-code: /x/mcp.json"))
        return _fake_result((("codex", "unwritable"),))

    monkeypatch.setattr(panels, "run_install", fake_run_install)
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        await advance_to(pilot, app, "summary")
        progress = app.query_one("#progress", panels.ProgressPanel)
        await click(pilot, "#next")
        await pilot.pause()
        assert progress._status["harnesses"] == "failed"


@pytest.mark.asyncio
async def test_progress_retry_reruns_only_the_failed_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import code_indexing_mcp.installer.tui.panels as panels

    plans: list = []

    def fake_run_install(plan, on_event=lambda event: None, should_continue=lambda: True):  # type: ignore[no-untyped-def]
        plans.append(plan)
        # Fails the first time, succeeds on the retry.
        return _fake_result((("codex", "unwritable"),) if len(plans) == 1 else ())

    monkeypatch.setattr(panels, "run_install", fake_run_install)
    state = _install_state(tmp_path)
    state.harness_slugs = ["codex", "kimi-code"]
    state.accelerator = "cpu"
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        await advance_to(pilot, app, "summary")
        await click(pilot, "#next")
        await pilot.pause()
        assert app.current == "progress"
        await click(pilot, "#progress-retry")
        await pilot.pause()

    assert len(plans) == 2
    assert plans[0].harness_slugs == ("codex", "kimi-code")
    # Only the client that failed, and no second accelerator build.
    assert plans[1].harness_slugs == ("codex",)
    assert plans[0].accelerator == "cpu"
    assert plans[1].accelerator is None
    assert app.done_code == 0


@pytest.mark.asyncio
async def test_done_panel_shows_checks_without_calling_a_warning_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import code_indexing_mcp.installer.tui.panels as panels
    from code_indexing_mcp.installer.orchestrator import InstallResult
    from code_indexing_mcp.installer.verify import Check

    checks = (
        Check("server executable", "ok", "/x/code-indexing-mcp"),
        Check("command on PATH", "warn", "resolves once you start a new shell"),
    )
    monkeypatch.setattr(
        panels,
        "run_install",
        lambda plan, on_event=None, should_continue=None: InstallResult(
            None, (), (), (), checks=checks
        ),
    )
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        await advance_to(pilot, app, "summary")
        await click(pilot, "#next")
        await pilot.pause()
        # A check that did not pass is worth saying out loud, but the install
        # itself succeeded and the exit code has to keep saying so.
        assert app.done_code == 0
        assert "with warnings" in str(app.query_one("#done-title", Label).render())
        body = str(app.query_one("#done-body", Static).render())
        assert "ok   - server executable" in body
        assert "warn - command on PATH" in body


@pytest.mark.asyncio
async def test_done_panel_gives_the_full_path_when_no_launcher_was_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import code_indexing_mcp.installer.tui.panels as panels
    from code_indexing_mcp.installer.orchestrator import InstallResult
    from code_indexing_mcp.installer.shell_path import LauncherResult

    monkeypatch.setattr(
        panels,
        "run_install",
        lambda plan, on_event=None, should_continue=None: InstallResult(
            None,
            (),
            (),
            (),
            launcher=LauncherResult(tmp_path / "bin" / "code-indexing-mcp", "failed", "nope"),
        ),
    )
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        await advance_to(pilot, app, "summary")
        await click(pilot, "#next")
        await pilot.pause()
        body = str(app.query_one("#done-body", Static).render())
        assert "no launcher was created" in body
        assert "Launcher NOT created" in body
