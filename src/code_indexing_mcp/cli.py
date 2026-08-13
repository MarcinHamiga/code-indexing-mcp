"""Command-line interface for Code Indexing MCP administration and stdio serving."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel

from . import __version__, update_check
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
from .progress import IndexProgress
from .server import create_server
from .settings import IndexSettings

# Commands a human runs and reads: the only place an update notice belongs.
_NOTIFY_COMMANDS = frozenset({"init", "index", "status", "projects", "model", "storage"})


class _VersionAction(argparse.Action):
    """Print the version, reading the revision only when the flag is used."""

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str = argparse.SUPPRESS,
        help: str | None = None,
    ) -> None:
        super().__init__(option_strings=list(option_strings), dest=dest, nargs=0, help=help)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        head = update_check.checkout_head(Path(__file__).resolve().parents[2])
        print(f"code-indexing-mcp {__version__} ({head[:7] if head else 'unknown'})")
        # Exits during parsing, so --version needs no subcommand despite required=True.
        parser.exit()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="code-indexing-mcp", description="Local MCP code indexer")
    parser.add_argument("--version", action=_VersionAction, help="show the version and exit")
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
    history = commands.add_parser(
        "history", help="Show a project's durable indexing history, newest first"
    )
    history.add_argument("project", nargs="?")
    history.add_argument("--limit", type=int, default=20)
    history.add_argument("--cursor", default=None)
    scan = commands.add_parser(
        "scan", help="Dry-run scan inspection: what an index run would find, without writing"
    )
    scan.add_argument("project", nargs="?")
    scan.add_argument("--outcome", choices=["eligible", "skipped"], default=None)
    scan.add_argument(
        "--reason",
        choices=["unsupported", "ignored", "symlink", "oversized", "unreadable"],
        default=None,
    )
    scan.add_argument("--limit", type=int, default=50)
    scan.add_argument("--cursor", default=None)
    storage = commands.add_parser(
        "storage", help="Inspect index storage statistics and maintenance"
    )
    storage_commands = storage.add_subparsers(dest="storage_command", required=True)
    storage_status = storage_commands.add_parser("status", help="Show storage statistics")
    storage_status.add_argument(
        "project", nargs="?", help="Project id, name, or path; omit for the whole installation"
    )
    storage_vacuum = storage_commands.add_parser(
        "vacuum", help="Compact tables and remove verified old versions (dry-run by default)"
    )
    storage_vacuum.add_argument(
        "project", nargs="?", help="Project id, name, or path; omit for the whole installation"
    )
    storage_vacuum.add_argument(
        "--execute",
        action="store_true",
        help="perform the cleanup; without it the command only estimates what could be reclaimed",
    )
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
    update = commands.add_parser("update", help="Update this installation to the latest main")
    update.add_argument("--install-dir", help="checkout location of the installation")
    update.add_argument(
        "--check",
        action="store_true",
        help="report whether an update is available; exit 0 up-to-date, 10 available, 1 unknown",
    )
    update.add_argument(
        "--skip-accelerator",
        action="store_true",
        help="do not rebuild a prepared accelerator whose locked runtime changed",
    )
    update.add_argument("--finalize", action="store_true", help=argparse.SUPPRESS)
    update.add_argument("--previous-sha", default=None, help=argparse.SUPPRESS)
    return parser


class _ProgressPrinter:
    """Render live indexing progress to *stream* without touching stdout.

    Indexing a large repository takes minutes, and printing nothing until the
    report lands makes a working command indistinguishable from a hung one. The
    JSON result stays alone on stdout, so piping the command is unaffected.
    """

    # A redirected stream has no cursor to rewrite, so it gets periodic lines
    # instead of a status line refreshed several times a second.
    LOG_INTERVAL_SECONDS = 5.0

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.interactive = stream.isatty()
        self._width = 0
        self._logged_at: float | None = None
        self._logged_phase: str | None = None

    def __call__(self, progress: IndexProgress) -> None:
        line = progress.describe()
        if self.interactive:
            self.stream.write("\r" + line.ljust(self._width))
            self._width = len(line)
            self.stream.flush()
            return
        now = time.monotonic()
        # A phase change is news whenever it happens: embedding a batch is where
        # a run spends minutes without a word, and the log should say so before
        # the wait rather than after it.
        if (
            progress.phase == self._logged_phase
            and self._logged_at is not None
            and now - self._logged_at < self.LOG_INTERVAL_SECONDS
        ):
            return
        self._logged_at = now
        self._logged_phase = progress.phase
        print(line, file=self.stream, flush=True)

    def clear(self) -> None:
        """Take the status line back down before anything else is printed."""

        if self.interactive and self._width:
            self.stream.write("\r" + " " * self._width + "\r")
            self.stream.flush()
            self._width = 0


def _update_notice(cache_directory: Path) -> str | None:
    """The notice, honouring the disable switch even when a cache lingers."""

    if update_check._disabled():
        return None
    return update_check.notice(cache_directory)


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
    refresh = (
        update_check.start_background_refresh(paths.cache)
        if args.command in _NOTIFY_COMMANDS
        else None
    )
    try:
        if args.command == "daemon":
            if args.daemon_command == "run":
                update_check.start_background_refresh(paths.cache)
                notice = _update_notice(paths.cache)
                if notice is not None:
                    logging.info(notice)
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
        if args.command == "update":
            # Lazy for the same reason as configure: never on the serve path.
            from .installer.update import update_main

            return update_main(
                install_dir=args.install_dir,
                check=args.check,
                skip_accelerator=args.skip_accelerator,
                finalize=args.finalize,
                previous_sha=args.previous_sha,
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
            update_check.start_background_refresh(paths.cache)
            # Logged rather than printed: stdout carries the MCP protocol.
            notice = _update_notice(paths.cache)
            if notice is not None:
                logging.info(notice)
            create_server(app).run(transport="stdio")
            return 0
        app = Application(paths, cwd=Path.cwd())
        result: BaseModel | Sequence[BaseModel] | dict[str, Any]
        if args.command == "init":
            result = app.init_project(args.path, args.name, args.force_new_id)
        elif args.command == "index":
            printer = _ProgressPrinter(sys.stderr)
            try:
                result = app.index_project(args.project, force=args.force, on_progress=printer)
            finally:
                printer.clear()
        elif args.command == "status":
            result = app.project_status(args.project)
        elif args.command == "history":
            result = app.index_history(args.project, cursor=args.cursor, limit=args.limit)
        elif args.command == "scan":
            result = app.inspect_scan(
                args.project,
                outcome=args.outcome,
                reason=args.reason,
                cursor=args.cursor,
                limit=args.limit,
            )
        elif args.command == "storage" and args.storage_command == "status":
            result = app.storage_status(args.project)
        elif args.command == "storage" and args.storage_command == "vacuum":
            # Cleanup needs the writer locks; a human asked for it, so wait them
            # out rather than skipping busy projects. The command still requires
            # the explicit --execute flag for any mutation.
            result = app.maintain_storage(
                args.project, dry_run=not args.execute, wait_for_lock=True
            )
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
        if refresh is not None:
            refresh.join(timeout=1.0)
        notice = _update_notice(paths.cache)
        if notice is not None:
            print(notice, file=sys.stderr)
        return 0
    except CodeIndexingError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
