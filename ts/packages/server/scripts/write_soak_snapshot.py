"""Write a Phase 9 soak snapshot with the Python build.

The migration plan's dual-run soak (§8) indexes the same real repositories
with both builds and diffs the chunk tables row by row plus compares search
rankings for a query set. This is the Python half: it produces the snapshot
artifact that ``write_soak_snapshot.ts`` mirrors and ``soak_compare.ts``
gates on. Run it *first* -- ``init_project`` writes the
``.ci-mcp/project.toml`` marker into each repository, and the project id in
that marker is what makes chunk ids comparable across the two builds.

Usage, from the repository root:

    uv run python ts/packages/server/scripts/write_soak_snapshot.py \
        --manifest soak.manifest.json --output soak-python.json

The manifest is the JSON shape ``src/soak.ts`` defines: repositories (path,
optional name) and queries. Indexing happens into a scratch data directory
removed afterwards (pass ``--data-dir`` to keep or reuse one), with execution
forced in-process and the daemon off, exactly like the benchmark commands.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from code_indexing_mcp import update_check
from code_indexing_mcp.application import Application, RuntimePaths
from code_indexing_mcp.settings import IndexSettings


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="soak manifest JSON")
    parser.add_argument("--output", required=True, help="snapshot JSON to write")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="reuse a specific scratch data directory instead of a temporary one",
    )
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_arguments(sys.argv[1:])
    manifest = json.loads(Path(arguments.manifest).read_text(encoding="utf-8"))
    if not manifest.get("queries"):
        raise SystemExit("the soak manifest needs at least one query")
    limit = int(manifest.get("limit", 8))
    if not 1 <= limit <= 50:
        raise SystemExit("the soak manifest limit must be from 1 to 50")

    temporary = (
        Path(arguments.data_dir)
        if arguments.data_dir is not None
        else Path(tempfile.mkdtemp(prefix="ci-mcp-py-soak-"))
    )
    settings = replace(
        IndexSettings.from_environment(),
        index_execution="in-process",
        broker_mode="off",
    )
    application = Application(
        RuntimePaths(data=temporary, cache=RuntimePaths.from_environment().cache),
        cwd=Path.cwd(),
        settings=settings,
    )
    try:
        repositories = []
        for repository in manifest["repositories"]:
            root = Path(repository["path"]).expanduser().resolve()
            name = repository.get("name") or root.name
            project = application.init_project(root)
            application.index_project(project.id)
            chunks = sorted(
                application.store.list_chunks([project.id]), key=lambda chunk: chunk.chunk_id
            )
            queries = []
            for query in manifest["queries"]:
                response = application.search_code(query, projects=[project.id], limit=limit)
                queries.append(
                    {
                        "query": query,
                        "hits": [
                            {"chunk_id": hit.chunk_id, "score": hit.score} for hit in response.hits
                        ],
                    }
                )
            print(f"{name}: {len(chunks)} chunks, {len(queries)} queries", file=sys.stderr)
            repositories.append(
                {
                    "name": name,
                    "path": str(root),
                    "project_id": project.id,
                    "chunk_count": len(chunks),
                    "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
                    "queries": queries,
                }
            )
        snapshot = {
            "schema_version": 1,
            "build": "python",
            "revision": update_check.checkout_head(Path(__file__).resolve().parents[4]),
            "model_id": application.embedder.model_id,
            "repositories": repositories,
        }
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}", file=sys.stderr)
    finally:
        if arguments.data_dir is None:
            shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
