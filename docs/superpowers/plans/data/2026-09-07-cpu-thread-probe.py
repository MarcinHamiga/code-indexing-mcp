"""Throwaway analysis probe: real worker, cached model, no index writes."""

import json
import platform
import statistics
import time
from dataclasses import asdict
from importlib.metadata import version

import numpy as np

from code_indexing_mcp.calibration import calibration_candidates
from code_indexing_mcp.embedding import SegmentPlan
from code_indexing_mcp.embedding_worker import EmbeddingWorkerSession, WorkerConfig


def main():
    candidates = calibration_candidates()
    baseline = None
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "fastembed": version("fastembed"),
                "onnxruntime": version("onnxruntime"),
                "candidates": len(candidates),
                "characters": sum(len(c.prefix) + len(c.content) for c in candidates),
            }
        ),
        flush=True,
    )
    for threads, batch in [(2, 1), (4, 1), (8, 1), (2, 2), (4, 2), (8, 2), (2, 1)]:
        config = WorkerConfig(
            cache_directory="/home/marcinh/.cache/code-indexing-mcp/models",
            offline=True,
            threads=threads,
            enable_cpu_mem_arena=False,
            dimension=768,
        )
        with EmbeddingWorkerSession(config, configured_ceiling_bytes=2 * 1024**3) as session:
            session.initialize()
            plan = SegmentPlan(max_items=batch)
            session.plan_and_embed(candidates, plan)
            elapsed = []
            for _ in range(3):
                start = time.perf_counter()
                result = session.plan_and_embed(candidates, plan)
                elapsed.append(time.perf_counter() - start)
            vectors = np.stack(
                [np.frombuffer(s.vector, dtype="<f4") for item in result for s in item]
            )
            if baseline is None:
                baseline = vectors.copy()
            cosine = np.sum(vectors * baseline, axis=1) / (
                np.linalg.norm(vectors, axis=1) * np.linalg.norm(baseline, axis=1)
            )
            print(
                json.dumps(
                    {
                        "threads": threads,
                        "batch": batch,
                        "seconds": elapsed,
                        "median_seconds": statistics.median(elapsed),
                        "cosine_min": float(cosine.min()),
                        "telemetry": asdict(session.telemetry()),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
