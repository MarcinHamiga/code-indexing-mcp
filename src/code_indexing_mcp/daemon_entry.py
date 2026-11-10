"""Entry point of the per-user daemon process.

Deliberately tiny and free of heavy imports at module level. Spawned CPU
embedding workers (``multiprocessing`` "spawn") re-import the parent's
``__main__`` module under the name ``__mp_main__``; if that were
``code_indexing_mcp.cli`` every worker would import the whole CLI/server stack
(about 1,200 modules and 115 MB) before doing any work. The real imports happen
only under the ``__main__`` guard, which a worker never runs.
"""

from __future__ import annotations


def main() -> int:
    from .cli import main as cli_main

    return cli_main(["daemon", "run"])


if __name__ == "__main__":
    raise SystemExit(main())
