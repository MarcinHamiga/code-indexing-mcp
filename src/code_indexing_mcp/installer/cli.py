"""Non-interactive installer entry shared by the bootstrap and ``configure``."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from .accelerator import ACCELERATOR_CHOICES
from .config_files import InstallerError
from .harnesses import HARNESS_CHOICES, parse_harness_selection
from .orchestrator import (
    InstallPlan,
    StepEvent,
    default_install_directory,
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


def _print_event(event: StepEvent) -> None:
    stream = sys.stderr if event.status in {"warning", "failed"} else sys.stdout
    print(f"[{event.step}] {event.status}: {event.detail}", file=stream)


def _prompt_harnesses(
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> list[str]:
    output_fn("Select the harnesses to configure:")
    for index, choice in enumerate(HARNESS_CHOICES, start=1):
        output_fn(f"  {index}. {choice.label}")
    return parse_harness_selection(
        input_fn("Enter comma-separated choices, 'all', or leave blank to skip: ")
    )


def _run_tui(
    args: argparse.Namespace,
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
    if args.reconfigure:
        state = WizardState.for_reconfigure(install_directory)
        state.values.update(preset)
        if args.accelerator is not None:
            state.accelerator = args.accelerator
    else:
        state = WizardState.for_install(
            install_directory,
            preset_values=preset,
            preset_accelerator=args.accelerator,
        )
    # An explicit --unset clears the field, which the wizard then reads as
    # "reset to default" and turns back into a deletion on confirmation.
    for name, value in env_updates.items():
        if value is None:
            state.values.pop(name, None)
    if args.harnesses is not None:
        state.harness_slugs = parse_harness_selection(args.harnesses)
    state.offline = args.offline
    app = InstallerApp(state)
    app.run()
    return app.done_code if app.done_code is not None else 130


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    install_directory = Path(args.install_dir).expanduser().resolve()
    try:
        env_updates = parse_settings(args.settings, args.unsets)
        if args.tui:
            return _run_tui(args, install_directory, env_updates)
        if args.harnesses is not None:
            selected = parse_harness_selection(args.harnesses)
        elif args.reconfigure:
            selected = list(load_prefill().configured_slugs)
        elif args.no_prompt or not sys.stdin.isatty():
            selected = []
        else:
            selected = _prompt_harnesses()
        accelerator = args.accelerator
        if accelerator is None and not args.reconfigure:
            accelerator = "auto"
        plan = InstallPlan(
            install_directory=install_directory,
            accelerator=accelerator,
            harness_slugs=tuple(selected),
            env_updates=env_updates,
            offline=args.offline,
        )
        result = run_install(plan, on_event=_print_event)
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
    print("Installation complete. Restart configured clients to load the MCP server.")
    return 0


def configure_main(
    *,
    install_dir: str | None,
    accelerator: str | None,
    harnesses: str | None,
    settings: Sequence[str],
    unsets: Sequence[str],
    no_tui: bool,
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
    argv = ["--install-dir", str(install_directory), "--reconfigure", "--no-prompt"]
    if accelerator is not None:
        argv += ["--accelerator", accelerator]
    if harnesses is not None:
        argv += ["--harnesses", harnesses]
    for pair in settings:
        argv += ["--set", pair]
    for name in unsets:
        argv += ["--unset", name]
    # Any flag that already says what to do is an instruction to apply it, not an
    # invitation to open a wizard over the top of it.
    scripted = bool(settings or unsets or harnesses is not None or accelerator is not None)
    if not no_tui and not scripted and sys.stdin.isatty():
        argv.remove("--no-prompt")
        argv.append("--tui")
    return main(argv)
