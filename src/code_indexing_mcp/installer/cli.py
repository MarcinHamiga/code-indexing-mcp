"""Non-interactive installer entry shared by the bootstrap and ``configure``."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .accelerator import ACCELERATOR_CHOICES
from .config_files import InstallerError
from .daemon_control import stop_daemon
from .harnesses import HARNESS_CHOICES, grouped_choices, parse_harness_selection
from .orchestrator import (
    InstallPlan,
    InstallResult,
    StepEvent,
    default_install_directory,
    finalize_reconfigure,
    run_install,
)
from .settings_spec import BY_NAME, as_bool, normalize, validate
from .wizard import load_prefill


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="code_indexing_mcp.installer",
        description="Install, update, or reconfigure Code Indexing MCP.",
    )
    parser.add_argument("--install-dir", default=str(default_install_directory()))
    parser.add_argument(
        "--accelerator",
        choices=ACCELERATOR_CHOICES,
        default=None,
        help="accelerator to prepare; omit to keep the prepared backend",
    )
    parser.add_argument("--harnesses", help="comma-separated harness numbers/slugs or 'all'")
    parser.add_argument(
        "--set",
        dest="settings",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="set a managed CODE_INDEXING_* value; repeatable",
    )
    parser.add_argument(
        "--unset",
        dest="unsets",
        action="append",
        default=[],
        metavar="NAME",
        help="remove a managed CODE_INDEXING_* value from harness configs; repeatable",
    )
    parser.add_argument(
        "--bin-dir",
        default=None,
        help="directory for the code-indexing-mcp launcher (default: ~/.local/bin)",
    )
    parser.add_argument(
        "--no-launcher",
        action="store_true",
        help="do not create the code-indexing-mcp launcher",
    )
    parser.add_argument(
        "--no-modify-path",
        action="store_true",
        help="never edit a shell profile to put the launcher directory on PATH",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=as_bool(os.environ.get("CODE_INDEXING_OFFLINE", "")),
    )
    parser.add_argument("--tui", action="store_true", help="open the interactive wizard")
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="never prompt; a missing harness selection configures none",
    )
    parser.add_argument("--reconfigure", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--repair",
        action="store_true",
        help=(
            "re-apply the launcher, client entries, and skills for the harnesses already "
            "configured, keeping the prepared accelerator and every current setting"
        ),
    )
    return parser


def parse_settings(pairs: Sequence[str], unsets: Sequence[str]) -> dict[str, str | None]:
    updates: dict[str, str | None] = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        name = name.strip()
        if not separator:
            raise InstallerError(f"--set expects NAME=VALUE, got {pair!r}")
        setting = BY_NAME.get(name)
        if setting is None:
            options = ", ".join(sorted(BY_NAME))
            raise InstallerError(f"unknown setting {name!r}; managed settings: {options}")
        error = validate(setting, value)
        if error is not None:
            raise InstallerError(error)
        updates[name] = normalize(setting, value)
    for name in unsets:
        name = name.strip()
        if name not in BY_NAME:
            options = ", ".join(sorted(BY_NAME))
            raise InstallerError(f"unknown setting {name!r}; managed settings: {options}")
        updates[name] = None
    return updates


@dataclass(frozen=True)
class ConfigureRequest:
    """Typed input shared by the module CLI and ``configure`` entry point."""

    install_directory: Path
    accelerator: str | None
    harnesses: str | None
    settings: tuple[str, ...]
    unsets: tuple[str, ...]
    interactive: bool
    offline: bool
    bin_directory: Path | None
    no_launcher: bool
    no_modify_path: bool
    reconfigure: bool
    repair: bool


def _print_event(event: StepEvent) -> None:
    stream = sys.stderr if event.status in {"warning", "failed"} else sys.stdout
    print(f"[{event.step}] {event.status}: {event.detail}", file=stream)


def _restart_daemon_if_settings_changed(result: InstallResult) -> None:
    """Keep the CLI event sink for the shared reconfigure finalization."""

    finalize_reconfigure(result, on_event=_print_event, stop=stop_daemon)


def _prompt_harnesses(
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> list[str]:
    output_fn("Select the harnesses to configure:")
    # Numbers stay pinned to the flat HARNESS_CHOICES order so a saved number
    # keeps meaning the same harness; display groups by provider so each
    # provider header prints exactly once.
    numbers = {choice.slug: index for index, choice in enumerate(HARNESS_CHOICES, start=1)}
    for provider, choices in grouped_choices():
        output_fn(f"{provider}:")
        for choice in choices:
            output_fn(f"  {numbers[choice.slug]}. {choice.label}")
    return parse_harness_selection(
        input_fn("Enter comma-separated choices, 'all', or leave blank to skip: ")
    )


def _run_tui(
    request: ConfigureRequest,
    install_directory: Path,
    env_updates: dict[str, str | None],
) -> int:
    try:
        from .tui.app import InstallerApp  # lazy: Textual is an optional dependency
    except ImportError:
        print(
            "Error: the interactive wizard needs the tui extra; run "
            "`uv sync --extra cpu --extra tui` in the installation checkout, "
            "or re-run with --no-tui.",
            file=sys.stderr,
        )
        return 1
    from .wizard import WizardState

    preset = {name: value for name, value in env_updates.items() if value is not None}
    if request.reconfigure:
        state = WizardState.for_reconfigure(install_directory)
        state.values.update(preset)
        if request.accelerator is not None:
            state.accelerator = request.accelerator
    else:
        state = WizardState.for_install(
            install_directory,
            preset_values=preset,
            preset_accelerator=request.accelerator,
        )
    # An explicit --unset clears the field, which the wizard then reads as
    # "reset to default" and turns back into a deletion on confirmation.
    for name, value in env_updates.items():
        if value is None:
            state.values.pop(name, None)
    if request.harnesses is not None:
        state.harness_slugs = parse_harness_selection(request.harnesses)
    state.offline = request.offline
    if request.bin_directory:
        state.bin_directory = request.bin_directory
    state.install_launcher = not request.no_launcher
    state.modify_shell_profiles = not request.no_modify_path
    app = InstallerApp(state)
    app.run()
    return app.done_code if app.done_code is not None else 130


def _repair(request: ConfigureRequest) -> int:
    """Re-apply the launcher, client entries, and skills for an existing install."""

    prefill = load_prefill()
    selected = (
        parse_harness_selection(request.harnesses)
        if request.harnesses is not None
        else list(prefill.configured_slugs)
    )
    plan = InstallPlan(
        install_directory=request.install_directory,
        accelerator=None,
        harness_slugs=tuple(selected),
        # Repair fixes wiring around the existing configuration. It must not
        # rewrite values merely because they were used to prefill the wizard.
        env_updates={},
        offline=request.offline,
        bin_directory=request.bin_directory,
        install_launcher=not request.no_launcher,
        modify_shell_profiles=not request.no_modify_path,
    )
    result = run_install(plan, on_event=_print_event)
    if result.failures:
        print(
            f"Repair finished with {len(result.failures)} failure(s); see above.",
            file=sys.stderr,
        )
        return 1
    print("Repair complete.")
    return 0


def _run_request(request: ConfigureRequest) -> int:
    install_directory = request.install_directory
    try:
        env_updates = parse_settings(request.settings, request.unsets)
        if request.interactive:
            return _run_tui(request, install_directory, env_updates)
        if request.repair:
            # Repair re-runs the cheap steps against what is already configured.
            # Nothing is chosen anew, so it never opens the wizard and never
            # rebuilds an accelerator environment that already works.
            return _repair(request)
        if request.harnesses is not None:
            selected = parse_harness_selection(request.harnesses)
        elif request.reconfigure:
            selected = list(load_prefill().configured_slugs)
        elif not request.interactive or not sys.stdin.isatty():
            selected = []
        else:
            selected = _prompt_harnesses()
        accelerator = request.accelerator
        if accelerator is None and not request.reconfigure:
            accelerator = "auto"
        plan = InstallPlan(
            install_directory=install_directory,
            accelerator=accelerator,
            harness_slugs=tuple(selected),
            env_updates=env_updates,
            offline=request.offline,
            bin_directory=request.bin_directory,
            install_launcher=not request.no_launcher,
            modify_shell_profiles=not request.no_modify_path,
        )
        result = run_install(plan, on_event=_print_event)
        if request.reconfigure:
            _restart_daemon_if_settings_changed(result)
    except InstallerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Installation cancelled.", file=sys.stderr)
        return 130
    if not result.configured and not result.failures and not result.skills:
        print("No harness configuration selected.")
    if result.failures:
        print(
            f"Installation finished with {len(result.failures)} failed harness "
            "configuration(s); see the errors above.",
            file=sys.stderr,
        )
        return 1
    if result.warnings:
        print(
            f"Installation complete with {len(result.warnings)} check warning(s); "
            "see the [verify] lines above.",
            file=sys.stderr,
        )
    print("Installation complete. Restart configured clients to load the MCP server.")
    if result.profiles_updated:
        from .shell_path import activation_hint

        names = ", ".join(str(profile) for profile in result.profiles_updated)
        print(f"PATH was updated in {names}; start a new shell or run: ")
        print(f"  {activation_hint(result.profiles_updated)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _run_request(
        ConfigureRequest(
            install_directory=Path(args.install_dir).expanduser().resolve(),
            accelerator=args.accelerator,
            harnesses=args.harnesses,
            settings=tuple(args.settings),
            unsets=tuple(args.unsets),
            interactive=args.tui,
            offline=args.offline,
            bin_directory=Path(args.bin_dir).expanduser() if args.bin_dir else None,
            no_launcher=args.no_launcher,
            no_modify_path=args.no_modify_path,
            reconfigure=args.reconfigure,
            repair=args.repair,
        )
    )


def configure_main(
    *,
    install_dir: str | None,
    accelerator: str | None,
    harnesses: str | None,
    settings: Sequence[str],
    unsets: Sequence[str],
    no_tui: bool,
    bin_dir: str | None = None,
    no_launcher: bool = False,
    no_modify_path: bool = False,
    repair: bool = False,
) -> int:
    """Entry for ``code-indexing-mcp configure``: reconfigure an existing install."""

    install_directory = (
        Path(install_dir).expanduser().resolve()
        if install_dir
        else default_install_directory().resolve()
    )
    from .accelerator import server_executable

    if not server_executable(install_directory).is_file():
        print(f"Error: no installation found at {install_directory}", file=sys.stderr)
        return 1
    # Any flag that already says what to do is an instruction to apply it, not an
    # invitation to open a wizard over the top of it. The launcher flags are not
    # among them: they say where things go, not which steps to skip.
    scripted = bool(
        settings or unsets or harnesses is not None or accelerator is not None or repair
    )
    interactive = not no_tui and not scripted and sys.stdin.isatty()
    return _run_request(
        ConfigureRequest(
            install_directory=install_directory,
            accelerator=accelerator,
            harnesses=harnesses,
            settings=tuple(settings),
            unsets=tuple(unsets),
            interactive=interactive,
            offline=as_bool(os.environ.get("CODE_INDEXING_OFFLINE", "")),
            bin_directory=Path(bin_dir).expanduser() if bin_dir else None,
            no_launcher=no_launcher,
            no_modify_path=no_modify_path,
            reconfigure=True,
            repair=repair,
        )
    )
