from __future__ import annotations

import sys
from pathlib import Path

import pytest

from incode_mcp.embedding_worker import EmbeddingWorkerSession, WorkerConfig
from incode_mcp.errors import ErrorCode, IncodeError
from incode_mcp.worker_launcher import ExternalInterpreterLauncher

FIXTURES = Path(__file__).parent / "fixtures"
BODY = "external_worker_body:echo"


def _config(tmp_path: Path, **overrides: object) -> WorkerConfig:
    settings: dict[str, object] = {
        "cache_directory": str(tmp_path / "models"),
        "offline": True,
        "threads": 1,
        "enable_cpu_mem_arena": False,
        "dimension": 4,
        "providers": ("CUDAExecutionProvider", "CPUExecutionProvider"),
        "accelerator": "cuda",
    }
    settings.update(overrides)
    return WorkerConfig(**settings)  # type: ignore[arg-type]


def _launcher(**overrides: object) -> ExternalInterpreterLauncher:
    settings: dict[str, object] = {
        "executable": Path(sys.executable),
        "target": BODY,
        # The child imports the body itself, from an interpreter that knows
        # nothing about pytest's collection path.
        "extra_environment": {"PYTHONPATH": str(FIXTURES)},
    }
    settings.update(overrides)
    return ExternalInterpreterLauncher(**settings)  # type: ignore[arg-type]


def test_external_worker_receives_its_configuration_over_the_channel(tmp_path: Path) -> None:
    """The config reaches another interpreter intact, without sharing sys.path."""
    config = _config(tmp_path, model_id="jinaai/jina-embeddings-v2-base-code")
    launched = _launcher().launch(config)
    try:
        launched.connection.send(("identity", None))
        status, payload = launched.connection.recv()
        assert status == "identity"
        assert payload == ("jinaai/jina-embeddings-v2-base-code", "cuda")

        launched.connection.send(("initialize", None))
        status, (providers, dimension) = launched.connection.recv()
        assert status == "initialized"
        # Tuples survive the trip: the config is delivered over the connection
        # rather than through the JSON handshake.
        assert providers == ("CUDAExecutionProvider", "CPUExecutionProvider")
        assert dimension == 4
    finally:
        launched.connection.send(("stop", None))
        launched.process.join(timeout=5)
        launched.connection.close()
    assert launched.process.is_alive() is False


def test_a_session_drives_an_external_worker_through_the_usual_protocol(tmp_path: Path) -> None:
    session = EmbeddingWorkerSession(
        _config(tmp_path),
        effective_ceiling_bytes=2 * 1024**3,
        launcher=_launcher(),
    )
    with session:
        info = session.initialize()
        assert info.resolved_providers == ("CUDAExecutionProvider", "CPUExecutionProvider")
        assert info.dimension == 4
        assert session.report_memory() == 1
        pid = session.pid
    assert pid is not None
    assert session.pid is None


def test_a_missing_interpreter_is_reported_as_an_unavailable_backend(tmp_path: Path) -> None:
    launcher = _launcher(executable=tmp_path / "no-such-python")

    with pytest.raises(IncodeError) as failure:
        launcher.launch(_config(tmp_path))

    assert failure.value.code is ErrorCode.BACKEND_UNAVAILABLE


@pytest.mark.skipif(sys.platform.startswith("win"), reason="needs a POSIX shell script")
def test_an_environment_that_cannot_start_the_worker_is_reported_not_awaited(
    tmp_path: Path,
) -> None:
    """A child that dies before dialling back fails at once, not at the timeout."""
    broken = tmp_path / "broken-python"
    broken.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    broken.chmod(0o755)
    launcher = _launcher(executable=broken, timeout_seconds=30.0)

    with pytest.raises(IncodeError) as failure:
        launcher.launch(_config(tmp_path))

    assert failure.value.code is ErrorCode.BACKEND_UNAVAILABLE
    assert "exited with status 3" in str(failure.value)


def test_a_worker_that_dies_after_connecting_fails_its_request(tmp_path: Path) -> None:
    """The command protocol reports a dead external worker like a dead spawned one."""
    session = EmbeddingWorkerSession(
        _config(tmp_path),
        effective_ceiling_bytes=2 * 1024**3,
        launcher=_launcher(target="external_worker_body:missing_attribute"),
    )

    with session, pytest.raises(IncodeError) as failure:
        session.initialize()

    assert failure.value.code is ErrorCode.EMBEDDING_WORKER_FAILED


@pytest.mark.skipif(sys.platform.startswith("win"), reason="needs a POSIX shell script")
def test_a_child_that_never_connects_gives_up_at_the_timeout(tmp_path: Path) -> None:
    """A wedged environment fails the run's start, it does not hang it."""
    stalled = tmp_path / "stalled-python"
    stalled.write_text("#!/bin/sh\nexec sleep 30\n", encoding="utf-8")
    stalled.chmod(0o755)
    launcher = _launcher(executable=stalled, timeout_seconds=0.5)

    with pytest.raises(IncodeError) as failure:
        launcher.launch(_config(tmp_path))

    assert failure.value.code is ErrorCode.EMBEDDING_WORKER_FAILED
    assert "did not connect" in str(failure.value)
