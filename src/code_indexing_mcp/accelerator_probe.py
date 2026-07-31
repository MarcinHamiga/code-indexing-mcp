"""Prove an accelerator environment really embeds, from inside that environment.

The installer runs this with the accelerator environment's own interpreter, as
the last step before it writes a record that offers the backend to the server.
Hardware detection only nominates a backend: a machine can have the driver, the
wheels, and the provider listed and still fail to compile the graph or return
vectors an index could not use. So this loads the actual model through the
actual worker code path, embeds the same probe texts the runtime does, and
validates the vectors the same way.

It prints one JSON object and exits non-zero on failure, so the installer can
report exactly why a backend was refused rather than guessing.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .backends import (
    Accelerator,
    available_execution_providers,
    backend_for,
    parse_accelerator,
    platform_fingerprint,
    runtime_version,
)
from .embedding import (
    DEFAULT_DIMENSION,
    DEFAULT_MODEL,
    PROBE_TEXTS,
    resolve_session_providers,
    validate_probe_vectors,
)


def probe(
    accelerator: Accelerator, *, offline: bool, model_id: str, dimension: int
) -> dict[str, Any]:
    """Load the model on *accelerator* and embed the probe texts through it."""
    # Imported here so a broken accelerator runtime is reported as a probe
    # failure with its own message, rather than as an import error before the
    # argument parsing that decides how to report it.
    import numpy as np

    from .application import RuntimePaths
    from .embedding_worker import WorkerConfig, _load_model

    descriptor = backend_for(accelerator)
    if descriptor is None:
        raise ValueError(f"no backend is registered for {accelerator.value}")
    providers = available_execution_providers() if descriptor.publishes_execution_providers else ()
    if descriptor.provider_is_preregistered and descriptor.provider not in providers:
        raise RuntimeError(
            f"{descriptor.provider} is not offered by this environment's ONNX Runtime "
            f"({', '.join(providers)})"
        )
    config = WorkerConfig(
        cache_directory=str(RuntimePaths.from_environment().cache / "models"),
        offline=offline,
        threads=1,
        enable_cpu_mem_arena=False,
        dimension=dimension,
        model_id=model_id,
        providers=descriptor.providers,
        accelerator=accelerator.value,
    )
    model = _load_model(config)
    resolved = resolve_session_providers(model)
    if resolved:
        if descriptor.provider not in resolved:
            # ONNX Runtime drops a provider it cannot initialise and carries on
            # with the next one, so a session that quietly became a CPU session
            # must not be recorded as a working accelerator.
            raise RuntimeError(
                f"{descriptor.provider} was requested but the session runs on {', '.join(resolved)}"
            )
    elif descriptor.uses_direct_model:
        # The direct model reports the target its own session resolved, so
        # nothing at all means the session is broken. FastEmbed models are
        # different: resolution walks a private layout there, so an empty tuple
        # means "unknown" and stays tolerated rather than letting a FastEmbed
        # refactor fail the probe on a working CUDA environment.
        raise RuntimeError(
            f"the direct session reported no providers, so {descriptor.provider} cannot be verified"
        )
    providers = tuple(dict.fromkeys((*providers, *resolved)))
    vectors = [
        np.asarray(vector, dtype="<f4").tobytes()
        for vector in model.passage_embed(list(PROBE_TEXTS))
    ]
    validate_probe_vectors(vectors, dimension=dimension, count=len(PROBE_TEXTS))
    return {
        "ok": True,
        "accelerator": accelerator.value,
        "interpreter": sys.executable,
        "providers": list(providers),
        "resolved_providers": list(resolved),
        "runtime_version": runtime_version(descriptor.runtime),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform_fingerprint(),
        "device": descriptor.device,
        "dimension": dimension,
        "model_id": model_id,
        "detail": f"probed {len(PROBE_TEXTS)} passages on {descriptor.provider}",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m code_indexing_mcp.accelerator_probe",
        description="Verify that this environment can embed on a given accelerator",
    )
    parser.add_argument("--accelerator", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="fail rather than download a model that is not already cached",
    )
    arguments = parser.parse_args(argv)
    try:
        result = probe(
            parse_accelerator(arguments.accelerator),
            offline=arguments.offline,
            model_id=arguments.model,
            dimension=arguments.dimension,
        )
    except BaseException as exc:
        # The JSON line *is* this program's protocol, so nothing may escape as a
        # traceback: the installer needs a reason it can quote back to the user.
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":  # pragma: no cover - run as a subprocess by the installer
    raise SystemExit(main())
