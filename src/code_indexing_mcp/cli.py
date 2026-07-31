"""Command-line interface for Code Indexing MCP administration and stdio serving."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .application import Application, RuntimePaths
from .benchmark import run_index_benchmark_command
from .daemon import (
    BrokerApplication,
    DaemonServer,
    daemon_status,
    ensure_daemon,
    require_daemon_support,
)
from .errors import CodeIndexingError
from .server import create_server
from .settings import IndexSettings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="code-indexing-mcp", description="Local MCP code indexer")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Run the stdio MCP server")
    serve.add_argument("--direct", action="store_true", help="Bypass the per-user daemon")
    init = commands.add_parser("init", help="Initialize a local project marker")
    init.add_argument("path", nargs="?")
    init.add_argument("--name")
    init.add_argument("--force-new-id", action="store_true")
    index = commands.add_parser("index", help="Incrementally index a project")
    index.add_argument("project", nargs="?")
    index.add_argument("--force", action="store_true")
    status = commands.add_parser("status", help="Show project index status")
    status.add_argument("project", nargs="?")
    projects = commands.add_parser("projects", help="Manage registered projects")
    project_commands = projects.add_subparsers(dest="projects_command", required=True)
    project_commands.add_parser("list")
    remove = project_commands.add_parser("remove")
    remove.add_argument("project")
    model = commands.add_parser("model", help="Manage the local embedding model")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    model_commands.add_parser("pull")
    model_commands.add_parser("status", help="Show the resolved embedding backend")
    benchmark = commands.add_parser("benchmark", help="Run reproducible local benchmarks")
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_command", required=True)
    benchmark_index = benchmark_commands.add_parser(
        "index", help="Measure cold, warm, incremental, and forced indexing"
    )
    benchmark_index.add_argument("--files", type=int, default=128)
    benchmark_index.add_argument("--functions-per-file", type=int, default=2)
    benchmark_index.add_argument("--batch-size", type=int, default=8)
    benchmark_index.add_argument("--work-dir", type=Path, default=None)
    daemon = commands.add_parser("daemon", help="Manage the shared indexing daemon")
    daemon_commands = daemon.add_subparsers(dest="daemon_command", required=True)
    daemon_commands.add_parser("run", help=argparse.SUPPRESS)
    daemon_commands.add_parser("status")
    daemon_commands.add_parser("stop")
    daemon_commands.add_parser("restart")
    configure = commands.add_parser(
        "configure", help="Reconfigure this installation (wizard, or scripted with --set)"
    )
    configure.add_argument("--install-dir", help="checkout location of the installation")
    configure.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "cuda", "mlx", "webgpu", "migraphx", "coreml"],
        default=None,
        help="prepare a different accelerator; omit to keep the prepared backend",
    )
    configure.add_argument("--harnesses", help="comma-separated harness slugs or 'all'")
    configure.add_argument(
        "--set",
        dest="settings",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="set a managed CODE_INDEXING_* value; repeatable",
    )
    configure.add_argument(
        "--unset",
        dest="unsets",
        action="append",
        default=[],
        metavar="NAME",
        help="remove a managed CODE_INDEXING_* value from harness configs; repeatable",
    )
    configure.add_argument(
        "--bin-dir",
        help="directory for the code-indexing-mcp launcher (default: ~/.local/bin)",
    )
    configure.add_argument(
        "--no-launcher",
        action="store_true",
        help="do not create or refresh the code-indexing-mcp launcher",
    )
    configure.add_argument(
        "--no-modify-path",
        action="store_true",
        help="never edit a shell profile to put the launcher directory on PATH",
    )
    configure.add_argument("--no-tui", action="store_true", help="apply without opening the wizard")
    configure.add_argument(
        "--repair",
        action="store_true",
        help="re-apply the launcher, client entries, and skills without changing any choice",
    )
    uninstall = commands.add_parser(
        "uninstall", help="Remove this installation's client entries, skills, and launcher"
    )
    uninstall.add_argument("--install-dir", help="checkout location of the installation")
    uninstall.add_argument(
        "--harnesses",
        help="comma-separated harness slugs or 'all'; omit to clear every configured one",
    )
    uninstall.add_argument("--bin-dir", help="directory holding the code-indexing-mcp launcher")
    uninstall.add_argument("--keep-launcher", action="store_true", help="leave the launcher alone")
    uninstall.add_argument(
        "--keep-path", action="store_true", help="leave the shell profile PATH block alone"
    )
    uninstall.add_argument(
        "--purge",
        action="store_true",
        help="also delete the index and cache directories (not recoverable)",
    )
    uninstall.add_argument(
        "--remove-checkout",
        action="store_true",
        help="also delete the installation checkout and its virtual environments",
    )
    uninstall.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    return parser


def _json(value: BaseModel | Sequence[BaseModel] | dict[str, Any]) -> str:
    if isinstance(value, BaseModel):
        payload: Any = value.model_dump(mode="json")
    elif isinstance(value, Sequence):
        payload = [item.model_dump(mode="json") for item in value]
    else:
        payload = value
    return json.dumps(payload, indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    paths = RuntimePaths.from_environment()
    try:
        if args.command == "daemon":
            if args.daemon_command == "run":
                DaemonServer(paths).serve()
                return 0
            if args.daemon_command == "status":
                print(_json(daemon_status(paths)))
                return 0
            if args.daemon_command == "stop":
                status = daemon_status(paths)
                if status["running"]:
                    BrokerApplication(paths).stop()
                print(_json({"stopped": bool(status["running"])}))
                return 0
            if args.daemon_command == "restart":
                if daemon_status(paths)["running"]:
                    BrokerApplication(paths).stop()
                    for _ in range(100):
                        if not daemon_status(paths)["running"]:
                            break
                        time.sleep(0.05)
                broker = ensure_daemon(paths)
                print(_json({"restarted": True, **broker.ping()}))
                return 0
        if args.command == "benchmark" and args.benchmark_command == "index":
            benchmark_result = run_index_benchmark_command(
                paths=paths,
                files=args.files,
                functions_per_file=args.functions_per_file,
                batch_size=args.batch_size,
                work_dir=args.work_dir,
            )
            print(_json(benchmark_result))
            return 0
        if args.command == "configure":
            # Lazy import: the installer package is never on the serve path, and
            # Textual is only imported when the wizard actually opens.
            from .installer.cli import configure_main

            return configure_main(
                install_dir=args.install_dir,
                accelerator=args.accelerator,
                harnesses=args.harnesses,
                settings=args.settings,
                unsets=args.unsets,
                no_tui=args.no_tui,
                bin_dir=args.bin_dir,
                no_launcher=args.no_launcher,
                no_modify_path=args.no_modify_path,
                repair=args.repair,
            )
        if args.command == "uninstall":
            # Lazy for the same reason as configure: never on the serve path.
            from .installer.uninstall import uninstall_main

            return uninstall_main(
                install_dir=args.install_dir,
                harnesses_selection=args.harnesses,
                bin_dir=args.bin_dir,
                keep_launcher=args.keep_launcher,
                keep_path=args.keep_path,
                purge=args.purge,
                remove_checkout=args.remove_checkout,
                assume_yes=args.yes,
                error_output=lambda line: print(line, file=sys.stderr),
            )
        settings = IndexSettings.from_environment()
        if args.command == "serve":
            use_daemon = not args.direct and settings.broker_mode != "off"
            if use_daemon:
                try:
                    require_daemon_support()
                except CodeIndexingError:
                    # An explicit opt-in cannot be silently downgraded, but the
                    # default "auto" serves directly rather than failing.
                    if settings.broker_mode == "on":
                        raise
                    logging.warning(
                        "Unix domain sockets are unavailable on this platform; "
                        "serving directly instead of via the shared daemon"
                    )
                    use_daemon = False
            app = ensure_daemon(paths) if use_daemon else Application(paths, cwd=Path.cwd())
            create_server(app).run(transport="stdio")
            return 0
        app = Application(paths, cwd=Path.cwd())
        result: BaseModel | Sequence[BaseModel] | dict[str, Any]
        if args.command == "init":
            result = app.init_project(args.path, args.name, args.force_new_id)
        elif args.command == "index":
            result = app.index_project(args.project, force=args.force)
        elif args.command == "status":
            result = app.project_status(args.project)
        elif args.command == "projects" and args.projects_command == "list":
            result = app.list_projects()
        elif args.command == "projects" and args.projects_command == "remove":
            result = app.remove_project(args.project)
        elif args.command == "model" and args.model_command == "pull":
            app.prepare_model()
            result = {"model": app.embedder.model_id, "prepared": True}
        elif args.command == "model" and args.model_command == "status":
            result = app.model_status()
        else:
            raise AssertionError("unreachable command")
        print(_json(result))
        return 0
    except CodeIndexingError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
