"""Command-line interface for Incode administration and stdio serving."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .application import Application
from .errors import IncodeError
from .server import create_server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="incode", description="Local MCP code indexer")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="Run the stdio MCP server")
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
    app = Application.from_environment(cwd=Path.cwd())
    try:
        if args.command == "serve":
            create_server(app).run(transport="stdio")
            return 0
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
