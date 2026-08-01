"""Tests for the shared install pipeline."""

from pathlib import Path

import pytest

from code_indexing_mcp.installer import accelerator, harnesses, shell_path
from code_indexing_mcp.installer.config_files import InstallerError
from code_indexing_mcp.installer.orchestrator import (
    InstallPlan,
    StepEvent,
    default_install_directory,
    run_install,
)


def _server(tmp_path: Path) -> Path:
    """A stand-in for the built server command, which run_install requires to exist."""

    command = tmp_path / "server"
    command.touch()
    return command


def _plan(**overrides: object) -> InstallPlan:
    values: dict[str, object] = {
        "install_directory": Path("/opt/ci-mcp"),
        "accelerator": "cpu",
        "harness_slugs": ("kimi-code",),
        "env_updates": {"CODE_INDEXING_OFFLINE": "1"},
        # Off unless a test asks for it: the launcher step writes outside tmp_path
        # by default, and no test should reach the developer's real ~/.local/bin.
        "install_launcher": False,
    }
    values.update(overrides)
    return InstallPlan(**values)  # type: ignore[arg-type]


def test_run_install_emits_step_events_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        accelerator,
        "configure_accelerator",
        lambda directory, requested, *, offline=False: (
            calls.append(("accel", requested))
            or accelerator.AcceleratorPlan("cpu", "CPU was requested")
        ),
    )
    monkeypatch.setattr(accelerator, "server_executable", lambda directory: _server(tmp_path))

    def fake_configure(slugs, command, *, env=None, **kwargs):
        calls.append(("harnesses", tuple(slugs), dict(env or {})))
        return [("kimi-code", tmp_path / "mcp.json")], []

    monkeypatch.setattr(harnesses, "configure_selected_harnesses", fake_configure)
    monkeypatch.setattr(
        harnesses,
        "install_skills",
        lambda slugs, directory: (
            calls.append(("skills", tuple(slugs)))
            or [("kimi-code", "1 linked, 3 already installed")]
        ),
    )
    events: list[StepEvent] = []

    result = run_install(_plan(), on_event=events.append)

    assert calls == [
        ("accel", "cpu"),
        ("harnesses", ("kimi-code",), {"CODE_INDEXING_OFFLINE": "1"}),
        ("skills", ("kimi-code",)),
    ]
    steps = [event.step for event in events]
    # The verify step emits one event per check, so its count is not fixed here;
    # what matters is that it comes last and that nothing precedes it out of order.
    assert steps[: steps.index("verify")] == [
        "accelerator",
        "accelerator",
        "path",
        "harnesses",
        "harnesses",
        "skills",
        "skills",
    ]
    assert set(steps[steps.index("verify") :]) == {"verify"}
    assert result.checks and result.checks[0].name == "server executable"
    assert events[0] == StepEvent("accelerator", "started", "cpu")
    assert result.failures == ()
    assert result.accelerator_plan is not None
    assert result.accelerator_plan.accelerator == "cpu"


def test_run_install_reports_unhonored_accelerator_as_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        accelerator,
        "configure_accelerator",
        lambda directory, requested, *, offline=False: accelerator.AcceleratorPlan(
            "cpu", "CUDA was requested but no driver", honored=False
        ),
    )
    monkeypatch.setattr(accelerator, "server_executable", lambda directory: _server(tmp_path))
    monkeypatch.setattr(harnesses, "configure_selected_harnesses", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(harnesses, "install_skills", lambda *args: [])
    events: list[StepEvent] = []

    run_install(_plan(accelerator="cuda"), on_event=events.append)

    accelerator_events = [event for event in events if event.step == "accelerator"]
    assert accelerator_events[-1].status == "warning"
    assert "no driver" in accelerator_events[-1].detail


def test_run_install_skips_accelerator_when_plan_keeps_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        accelerator,
        "configure_accelerator",
        lambda *args, **kwargs: pytest.fail("accelerator step must not run"),
    )
    monkeypatch.setattr(accelerator, "server_executable", lambda directory: _server(tmp_path))
    monkeypatch.setattr(harnesses, "configure_selected_harnesses", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(harnesses, "install_skills", lambda *args: [])
    events: list[StepEvent] = []

    result = run_install(_plan(accelerator=None), on_event=events.append)

    assert result.accelerator_plan is None
    assert events[0] == StepEvent("accelerator", "skipped", "keeping the prepared backend")


def test_run_install_stops_between_steps_when_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        accelerator,
        "configure_accelerator",
        lambda directory, requested, *, offline=False: accelerator.AcceleratorPlan("cpu", "ok"),
    )
    monkeypatch.setattr(
        harnesses,
        "configure_selected_harnesses",
        lambda *args, **kwargs: pytest.fail("must not run after cancellation"),
    )
    monkeypatch.setattr(
        harnesses,
        "install_skills",
        lambda *args: pytest.fail("must not run after cancellation"),
    )

    result = run_install(_plan(), should_continue=lambda: False)

    assert result.accelerator_plan is None
    assert result.configured == () and result.skills == ()


def test_run_install_refuses_to_configure_from_a_directory_without_an_installation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        accelerator,
        "configure_accelerator",
        lambda directory, requested, *, offline=False: accelerator.AcceleratorPlan("cpu", "ok"),
    )
    monkeypatch.setattr(
        harnesses,
        "configure_selected_harnesses",
        lambda *args, **kwargs: pytest.fail("must not write a nonexistent command path"),
    )

    with pytest.raises(InstallerError, match="No prepared installation"):
        run_install(_plan(install_directory=tmp_path / "missing"))


def test_run_install_allows_a_missing_command_when_nothing_is_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        accelerator,
        "configure_accelerator",
        lambda directory, requested, *, offline=False: accelerator.AcceleratorPlan("cpu", "ok"),
    )
    monkeypatch.setattr(harnesses, "configure_selected_harnesses", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(harnesses, "install_skills", lambda *args: [])

    result = run_install(_plan(install_directory=tmp_path / "missing", harness_slugs=()))

    assert result.configured == ()


def _quiet_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out every step except the launcher, which is what these tests exercise."""

    monkeypatch.setattr(
        accelerator,
        "configure_accelerator",
        lambda directory, requested, *, offline=False: accelerator.AcceleratorPlan("cpu", "ok"),
    )
    monkeypatch.setattr(harnesses, "configure_selected_harnesses", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(harnesses, "install_skills", lambda *args: [])


def _checkout(tmp_path: Path) -> Path:
    """A checkout whose venv holds a stand-in for the built server command."""

    directory = tmp_path / "checkout"
    command = accelerator.server_executable(directory)
    command.parent.mkdir(parents=True, exist_ok=True)
    command.touch(mode=0o755)
    return directory


def test_run_install_creates_the_launcher_and_adds_it_to_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _quiet_pipeline(monkeypatch)
    bin_directory = tmp_path / "bin"
    profile = tmp_path / ".zshrc"
    profile.write_text("# mine\n", encoding="utf-8")
    monkeypatch.setattr(shell_path, "is_on_path", lambda directory, **kwargs: False)
    monkeypatch.setattr(shell_path, "shell_profiles", lambda **kwargs: (profile,))
    events: list[StepEvent] = []

    result = run_install(
        _plan(
            install_directory=_checkout(tmp_path),
            harness_slugs=(),
            install_launcher=True,
            bin_directory=bin_directory,
        ),
        on_event=events.append,
    )

    launcher = shell_path.launcher_path(bin_directory)
    # A symlink on POSIX, a .cmd shim on Windows; both land at launcher_path.
    assert launcher.is_symlink() or launcher.is_file()
    assert result.launcher is not None and result.launcher.status == "created"
    assert result.profiles_updated == (profile,)
    text = profile.read_text(encoding="utf-8")
    assert text.startswith("# mine\n")
    # The block spells the directory relative to $HOME when it sits under it, so
    # ask the module whether the line names it rather than matching raw text.
    assert shell_path.BLOCK_START in text
    assert shell_path.profile_mentions_directory(text, bin_directory, Path.home())
    assert [event.step for event in events].count("path") == 3


def test_run_install_leaves_shell_profiles_alone_when_already_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _quiet_pipeline(monkeypatch)
    monkeypatch.setattr(shell_path, "is_on_path", lambda directory, **kwargs: True)
    monkeypatch.setattr(
        shell_path,
        "shell_profiles",
        lambda **kwargs: pytest.fail("must not look for profiles when the directory is on PATH"),
    )

    result = run_install(
        _plan(
            install_directory=_checkout(tmp_path),
            harness_slugs=(),
            install_launcher=True,
            bin_directory=tmp_path / "bin",
        )
    )

    assert result.launcher is not None and result.launcher.ok
    assert result.profiles_updated == ()


def test_run_install_reports_a_failed_launcher_without_stopping_the_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _quiet_pipeline(monkeypatch)
    monkeypatch.setattr(shell_path, "is_on_path", lambda directory, **kwargs: True)
    events: list[StepEvent] = []

    # A checkout with no built environment: the launcher has nothing to point at.
    result = run_install(
        _plan(
            install_directory=tmp_path / "never-built",
            harness_slugs=(),
            install_launcher=True,
            bin_directory=tmp_path / "bin",
        ),
        on_event=events.append,
    )

    assert result.launcher is not None and not result.launcher.ok
    assert any(event.step == "path" and event.status == "warning" for event in events)
    assert [event.step for event in events][-1] == "verify"


def test_default_install_directory_honours_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CODE_INDEXING_MCP_INSTALL_DIR", str(tmp_path / "custom"))
    assert default_install_directory() == tmp_path / "custom"
    monkeypatch.delenv("CODE_INDEXING_MCP_INSTALL_DIR")
    assert default_install_directory() == Path.home() / ".local" / "share" / "code-indexing-mcp"
