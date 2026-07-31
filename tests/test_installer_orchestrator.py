"""Tests for the shared install pipeline."""

from pathlib import Path

import pytest

from incode_mcp.installer import accelerator, harnesses
from incode_mcp.installer.config_files import InstallerError
from incode_mcp.installer.orchestrator import (
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
        "env_updates": {"INCODE_OFFLINE": "1"},
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
        ("harnesses", ("kimi-code",), {"INCODE_OFFLINE": "1"}),
        ("skills", ("kimi-code",)),
    ]
    assert [event.step for event in events] == [
        "accelerator",
        "accelerator",
        "harnesses",
        "harnesses",
        "skills",
        "skills",
    ]
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


def test_default_install_directory_honours_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CODE_INDEXING_MCP_INSTALL_DIR", str(tmp_path / "custom"))
    assert default_install_directory() == tmp_path / "custom"
    monkeypatch.delenv("CODE_INDEXING_MCP_INSTALL_DIR")
    assert default_install_directory() == Path.home() / ".local" / "share" / "code-indexing-mcp"
