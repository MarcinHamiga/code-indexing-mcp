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
from .errors import IncodeError
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
        settings = IndexSettings.from_environment()
        if args.command == "serve":
            use_daemon = not args.direct and settings.broker_mode != "off"
            if use_daemon:
                try:
                    require_daemon_support()
                except IncodeError:
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
        else:
            raise AssertionError("unreachable command")
        print(_json(result))
        return 0
    except IncodeError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
