"""Tests for the installer's module CLI."""

import sys
import types
from pathlib import Path

import pytest

from code_indexing_mcp.installer import orchestrator
from code_indexing_mcp.installer.cli import main, parse_settings
from code_indexing_mcp.installer.config_files import InstallerError
from code_indexing_mcp.installer.wizard import Prefill


def test_parse_settings_validates_and_normalizes() -> None:
    updates = parse_settings(
        ["CODE_INDEXING_INDEX_MODE=EAGER", "CODE_INDEXING_OFFLINE=yes"], ["CODE_INDEXING_BROKER"]
    )
    assert updates == {
        "CODE_INDEXING_INDEX_MODE": "eager",
        "CODE_INDEXING_OFFLINE": "1",
        "CODE_INDEXING_BROKER": None,
    }


@pytest.mark.parametrize(
    "pair",
    ["CODE_INDEXING_FROBNICATE=1", "CODE_INDEXING_INDEX_MODE=sometimes", "CODE_INDEXING_OFFLINE"],
)
def test_parse_settings_rejects_bad_input(pair: str) -> None:
    with pytest.raises(InstallerError):
        parse_settings([pair], [])


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch, recorded: list) -> None:
    def fake_run_install(plan, on_event=lambda event: None, should_continue=lambda: True):
        recorded.append(plan)
        return orchestrator.InstallResult(None, (), (), ())

    import code_indexing_mcp.installer.cli as cli

    monkeypatch.setattr(cli, "run_install", fake_run_install)


def test_main_runs_plan_without_prompting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recorded: list = []
    _stub_pipeline(monkeypatch, recorded)
    code = main(
        [
            "--install-dir",
            str(tmp_path),
            "--accelerator",
            "cpu",
            "--harnesses",
            "kimi-code",
            "--set",
            "CODE_INDEXING_OFFLINE=1",
            "--no-prompt",
        ]
    )
    assert code == 0
    (plan,) = recorded
    assert plan.accelerator == "cpu"
    assert plan.harness_slugs == ("kimi-code",)
    assert plan.env_updates == {"CODE_INDEXING_OFFLINE": "1"}


def test_main_defaults_to_auto_accelerator_on_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: list = []
    _stub_pipeline(monkeypatch, recorded)
    assert main(["--install-dir", str(tmp_path), "--no-prompt"]) == 0
    assert recorded[0].accelerator == "auto"
    assert recorded[0].harness_slugs == ()


def test_reconfigure_keeps_backend_and_prefills_harnesses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: list = []
    _stub_pipeline(monkeypatch, recorded)
    monkeypatch.setattr(
        "code_indexing_mcp.installer.cli.load_prefill",
        lambda: Prefill({}, ("kimi-code",), ()),
    )
    assert main(["--install-dir", str(tmp_path), "--reconfigure", "--no-prompt"]) == 0
    assert recorded[0].accelerator is None
    assert recorded[0].harness_slugs == ("kimi-code",)


def test_main_reports_installer_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_pipeline(monkeypatch, [])
    code = main(["--install-dir", str(tmp_path), "--set", "CODE_INDEXING_NOPE=1", "--no-prompt"])
    assert code == 1
    assert "CODE_INDEXING_NOPE" in capsys.readouterr().err


def _fake_tui_app(monkeypatch: pytest.MonkeyPatch) -> list:
    """Stand in for the Textual app and record the state it was handed."""

    calls: list = []

    class FakeApp:
        done_code = 0

        def __init__(self, state):  # type: ignore[no-untyped-def]
            calls.append(state)

        def run(self) -> None:
            return None

    fake_package = types.ModuleType("code_indexing_mcp.installer.tui")
    fake_app_module = types.ModuleType("code_indexing_mcp.installer.tui.app")
    fake_app_module.InstallerApp = FakeApp  # type: ignore[attr-defined]
    fake_package.app = fake_app_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "code_indexing_mcp.installer.tui", fake_package)
    monkeypatch.setitem(sys.modules, "code_indexing_mcp.installer.tui.app", fake_app_module)
    return calls


def test_main_tui_flag_delegates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _fake_tui_app(monkeypatch)

    code = main(["--install-dir", str(tmp_path), "--tui", "--set", "CODE_INDEXING_OFFLINE=1"])

    assert code == 0
    assert calls and calls[0].values.get("CODE_INDEXING_OFFLINE") == "1"


def test_main_tui_honours_scripted_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--tui alongside --harnesses/--accelerator/--unset seeds the wizard, never drops them."""

    calls = _fake_tui_app(monkeypatch)
    monkeypatch.setattr(
        "code_indexing_mcp.installer.cli.load_prefill",
        lambda: Prefill({"CODE_INDEXING_BROKER": "off"}, ("kimi-code",), ()),
    )
    monkeypatch.setattr(
        "code_indexing_mcp.installer.wizard.load_prefill",
        lambda **kwargs: Prefill({"CODE_INDEXING_BROKER": "off"}, ("kimi-code",), ()),
    )
    monkeypatch.setattr(
        "code_indexing_mcp.installer.wizard.accelerator.prepared_accelerator",
        lambda directory: "cpu",
    )

    code = main(
        [
            "--install-dir",
            str(tmp_path),
            "--reconfigure",
            "--tui",
            "--accelerator",
            "mlx",
            "--harnesses",
            "codex,claude-code",
            "--unset",
            "CODE_INDEXING_BROKER",
        ]
    )

    assert code == 0
    (state,) = calls
    assert state.accelerator == "mlx"
    assert state.harness_slugs == ["codex", "claude-code"]
    # Cleared, but still remembered as prefilled, so confirming deletes the key.
    assert "CODE_INDEXING_BROKER" not in state.values
    assert state.env_updates() == {"CODE_INDEXING_BROKER": None}


def test_configure_does_not_open_the_wizard_over_explicit_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from code_indexing_mcp.installer.cli import configure_main

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "code_indexing_mcp.installer.accelerator.server_executable",
        lambda directory: tmp_path / "server",
    )
    (tmp_path / "server").touch()
    recorded: list = []
    _stub_pipeline(monkeypatch, recorded)
    monkeypatch.setattr(
        "code_indexing_mcp.installer.cli._run_tui",
        lambda *args, **kwargs: pytest.fail("must not open the wizard"),
    )

    code = configure_main(
        install_dir=str(tmp_path),
        accelerator="cpu",
        harnesses="kimi-code",
        settings=[],
        unsets=[],
        no_tui=False,
    )

    assert code == 0
    assert recorded[0].harness_slugs == ("kimi-code",)
    assert recorded[0].accelerator == "cpu"


def test_main_reports_harness_failures_instead_of_claiming_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import code_indexing_mcp.installer.cli as cli

    monkeypatch.setattr(
        cli,
        "run_install",
        lambda plan, on_event=lambda event: None, should_continue=lambda: True: (
            orchestrator.InstallResult(None, (), (("codex", "unwritable"),), ())
        ),
    )

    code = main(["--install-dir", str(tmp_path), "--harnesses", "codex", "--no-prompt"])

    assert code == 1
    captured = capsys.readouterr()
    assert "Installation complete" not in captured.out
    assert "1 failed harness" in captured.err


def test_main_threads_the_launcher_flags_into_the_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: list = []
    _stub_pipeline(monkeypatch, recorded)

    code = main(
        [
            "--install-dir",
            str(tmp_path),
            "--bin-dir",
            str(tmp_path / "bin"),
            "--no-modify-path",
            "--no-prompt",
        ]
    )

    assert code == 0
    (plan,) = recorded
    assert plan.bin_directory == tmp_path / "bin"
    assert plan.install_launcher is True
    assert plan.modify_shell_profiles is False


def test_main_no_launcher_turns_the_step_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: list = []
    _stub_pipeline(monkeypatch, recorded)

    main(["--install-dir", str(tmp_path), "--no-launcher", "--no-prompt"])

    (plan,) = recorded
    assert plan.install_launcher is False
    assert plan.bin_directory is None  # resolved at run time


def test_main_reports_the_shell_profiles_it_edited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import code_indexing_mcp.installer.cli as cli

    profile = tmp_path / ".zshrc"
    monkeypatch.setattr(
        cli,
        "run_install",
        lambda plan, on_event=lambda event: None, should_continue=lambda: True: (
            orchestrator.InstallResult(None, (), (), (), profiles_updated=(profile,))
        ),
    )

    code = main(["--install-dir", str(tmp_path), "--no-prompt"])

    assert code == 0
    out = capsys.readouterr().out
    assert str(profile) in out
    assert "exec" in out


def test_configure_forwards_the_launcher_flags_without_suppressing_the_wizard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--bin-dir says where things go, not which steps to skip, so the wizard opens."""

    from code_indexing_mcp.installer.cli import configure_main

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "code_indexing_mcp.installer.accelerator.server_executable",
        lambda directory: tmp_path / "server",
    )
    (tmp_path / "server").touch()
    calls = _fake_tui_app(monkeypatch)
    monkeypatch.setattr(
        "code_indexing_mcp.installer.wizard.load_prefill", lambda **kwargs: Prefill({}, (), ())
    )
    monkeypatch.setattr(
        "code_indexing_mcp.installer.wizard.accelerator.prepared_accelerator",
        lambda directory: "cpu",
    )

    code = configure_main(
        install_dir=str(tmp_path),
        accelerator=None,
        harnesses=None,
        settings=[],
        unsets=[],
        no_tui=False,
        bin_dir=str(tmp_path / "bin"),
        no_modify_path=True,
    )

    assert code == 0
    (state,) = calls
    assert state.bin_directory == tmp_path / "bin"
    assert state.modify_shell_profiles is False


def test_repair_reapplies_the_current_configuration_without_rebuilding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: list = []
    _stub_pipeline(monkeypatch, recorded)
    monkeypatch.setattr(
        "code_indexing_mcp.installer.cli.load_prefill",
        lambda: Prefill({"CODE_INDEXING_BROKER": "off"}, ("kimi-code", "codex"), ()),
    )

    code = main(["--install-dir", str(tmp_path), "--repair"])

    assert code == 0
    (plan,) = recorded
    # The harnesses already configured, their current settings written back, and
    # explicitly no accelerator work.
    assert plan.harness_slugs == ("kimi-code", "codex")
    assert plan.env_updates == {"CODE_INDEXING_BROKER": "off"}
    assert plan.accelerator is None
    assert plan.install_launcher is True


def test_repair_never_opens_the_wizard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from code_indexing_mcp.installer.cli import configure_main

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "code_indexing_mcp.installer.accelerator.server_executable",
        lambda directory: tmp_path / "server",
    )
    (tmp_path / "server").touch()
    monkeypatch.setattr(
        "code_indexing_mcp.installer.cli._run_tui",
        lambda *args, **kwargs: pytest.fail("repair must not open the wizard"),
    )
    monkeypatch.setattr(
        "code_indexing_mcp.installer.cli.load_prefill", lambda: Prefill({}, ("kimi-code",), ())
    )
    recorded: list = []
    _stub_pipeline(monkeypatch, recorded)

    code = configure_main(
        install_dir=str(tmp_path),
        accelerator=None,
        harnesses=None,
        settings=[],
        unsets=[],
        no_tui=False,
        repair=True,
    )

    assert code == 0
    assert recorded[0].accelerator is None


def test_main_tui_without_textual_reports_the_fix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name.endswith("tui.app") or name == "textual" or name.startswith("textual."):
            raise ImportError("No module named 'textual'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    code = main(["--install-dir", str(tmp_path), "--tui"])
    assert code == 1
    assert "uv sync --extra cpu --extra tui" in capsys.readouterr().err
