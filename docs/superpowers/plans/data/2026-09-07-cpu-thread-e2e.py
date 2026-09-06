"""Throwaway full indexing probe using three unmodified source modules."""

import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

from code_indexing_mcp.application import Application, RuntimePaths
from code_indexing_mcp.settings import IndexSettings


def main():
    source = Path.cwd() / "src/code_indexing_mcp"
    names = ("embedding.py", "token_batching.py", "calibration.py")
    print(
        json.dumps(
            {"corpus": {n: hashlib.sha256((source / n).read_bytes()).hexdigest() for n in names}}
        ),
        flush=True,
    )
    for threads in (2, 4, 4, 2, 2, 4):
        with tempfile.TemporaryDirectory(prefix="index-speed-e2e-") as temporary:
            work = Path(temporary)
            root = work / "corpus"
            root.mkdir()
            for name in names:
                shutil.copy2(source / name, root / name)
            cache = work / "cache"
            cache.mkdir()
            (cache / "models").symlink_to(
                "/home/marcinh/.cache/code-indexing-mcp/models", target_is_directory=True
            )
            settings = IndexSettings.from_environment(
                {
                    "CODE_INDEXING_EMBED_ACCELERATOR": "cpu",
                    "CODE_INDEXING_EMBED_THREADS": str(threads),
                    "CODE_INDEXING_EMBED_BATCH_SIZE": "1",
                    "CODE_INDEXING_EMBED_CALIBRATE": "0",
                    "CODE_INDEXING_OFFLINE": "1",
                    "CODE_INDEXING_INDEX_MEMORY_MB": "2048",
                    "CODE_INDEXING_BROKER": "off",
                    "CODE_INDEXING_AUTO_MAINTENANCE": "0",
                }
            )
            app = Application(
                RuntimePaths(data=work / "data", cache=cache), cwd=root, settings=settings
            )
            app.init_project(str(root))
            started = time.perf_counter()
            report = app.index_project(str(root))
            print(
                json.dumps(
                    {
                        "threads": threads,
                        "wall_seconds": time.perf_counter() - started,
                        "report": report.model_dump(mode="json"),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
