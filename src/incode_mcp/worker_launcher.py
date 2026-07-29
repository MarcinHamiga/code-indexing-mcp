"""How an embedding worker is started, and which environment its code runs in.

The CPU worker is a child of this interpreter and keeps using ``multiprocessing``
spawn. An accelerator worker cannot: ``fastembed`` and ``fastembed-gpu`` install
the same module over two ONNX Runtime distributions that both own the
``onnxruntime`` import, so the accelerator lives in an environment of its own and
its worker has to be started from *that* interpreter.

``multiprocessing`` cannot cross that boundary even with ``set_executable``,
because ``spawn`` hands the child the parent's ``sys.path`` and the child
installs it verbatim -- the accelerator interpreter would start up and then
import the serving environment's CPU runtime. So an external worker is an
ordinary subprocess that dials back to a socket this process is listening on,
and everything above the handshake -- the command protocol, the memory ceiling,
the batch retries -- stays exactly as it is for both kinds of worker.

The channel is a Unix socket inside a private directory, or a loopback socket on
Windows, which has no filesystem permissions to lean on. Both ends authenticate
with the same challenge-response ``multiprocessing`` uses for its own
connections, and the key travels on the child's stdin rather than its argv,
where every other process on the machine could read it.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from multiprocessing.connection import Client, Connection, answer_challenge, deliver_challenge
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .errors import ErrorCode, IncodeError

if TYPE_CHECKING:  # pragma: no cover - imported for annotations only
    from .embedding_worker import WorkerConfig, WorkerTarget

logger = logging.getLogger(__name__)

# Covers interpreter start-up and the dial-back, not the model load: the child
# connects before it imports anything heavy, precisely so a slow model load
# cannot be mistaken for a worker that will never arrive.
HANDSHAKE_TIMEOUT_SECONDS = 30.0
# How often the wait for the dial-back looks up from the socket to check whether
# the child is still running.
HANDSHAKE_POLL_SECONDS = 0.1
DEFAULT_WORKER_TARGET = "incode_mcp.embedding_worker:_worker_main"


class WorkerProcess(Protocol):
    """The part of a child process a session needs, however it was started."""

    @property
    def pid(self) -> int | None: ...

    def is_alive(self) -> bool: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...


@dataclass(frozen=True)
class LaunchedWorker:
    """A started worker and the channel its commands travel over."""

    process: WorkerProcess
    # Connection on POSIX, PipeConnection on Windows for the spawned variant;
    # typeshed does not relate the two even though both carry the send/recv/
    # poll/close surface used here.
    connection: Any


class WorkerLauncher(Protocol):
    """Starts one worker process for a configuration."""

    def launch(self, config: WorkerConfig) -> LaunchedWorker: ...

    @property
    def description(self) -> str:
        """A short, local-only phrase naming where the worker runs."""


class SpawnLauncher:
    """Start the worker as a ``multiprocessing`` child of this interpreter."""

    def __init__(self, target: WorkerTarget) -> None:
        self._target = target

    @property
    def description(self) -> str:
        return "serving environment"

    def launch(self, config: WorkerConfig) -> LaunchedWorker:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=self._target,
            args=(child, config),
            name="incode-embedding-worker",
            daemon=True,
        )
        process.start()
        child.close()
        connection: Any = parent
        return LaunchedWorker(process=process, connection=connection)


class ExternalWorker:
    """A subprocess presented with the same surface as a spawned child."""

    def __init__(self, popen: subprocess.Popen[bytes]) -> None:
        self._popen = popen

    @property
    def pid(self) -> int | None:
        return self._popen.pid

    def is_alive(self) -> bool:
        return self._popen.poll() is None

    def terminate(self) -> None:
        self._popen.terminate()

    def kill(self) -> None:
        self._popen.kill()

    def join(self, timeout: float | None = None) -> None:
        try:
            self._popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # BaseProcess.join returns rather than raising when a child outlives
            # its timeout, and every caller here is written against that.
            return


class ExternalInterpreterLauncher:
    """Start the worker from another environment's Python interpreter.

    The interpreter is one the installer prepared and recorded. Nothing here
    resolves, downloads, or installs anything, which is what keeps package
    installation out of the request path.
    """

    def __init__(
        self,
        executable: Path,
        *,
        environment_name: str = "accelerator environment",
        timeout_seconds: float = HANDSHAKE_TIMEOUT_SECONDS,
        target: str = DEFAULT_WORKER_TARGET,
        extra_environment: Mapping[str, str] | None = None,
    ) -> None:
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self._environment_name = environment_name
        self._target = target
        self._extra_environment = dict(extra_environment or {})

    @property
    def description(self) -> str:
        return f"{self._environment_name} ({self.executable})"

    def launch(self, config: WorkerConfig) -> LaunchedWorker:
        channel = _Channel.open()
        try:
            popen = self._start(channel)
        except BaseException:
            channel.close()
            raise
        try:
            connection = channel.accept(popen, timeout_seconds=self.timeout_seconds)
        except BaseException:
            _reap(popen)
            raise
        finally:
            # The listening socket has done its job, and the accepted connection
            # depends on neither it nor the directory holding its path.
            channel.close()
        try:
            self._authenticate(connection, channel.authkey)
            connection.send(("configure", asdict(config)))
        except BaseException:
            connection.close()
            _reap(popen)
            raise
        return LaunchedWorker(process=ExternalWorker(popen), connection=connection)

    def _start(self, channel: _Channel) -> subprocess.Popen[bytes]:
        if not self.executable.is_file():
            raise IncodeError(
                ErrorCode.BACKEND_UNAVAILABLE,
                f"The recorded accelerator interpreter is missing: {self.executable}",
                interpreter=str(self.executable),
            )
        try:
            popen = subprocess.Popen(
                [str(self.executable), "-m", "incode_mcp.worker_launcher"],
                stdin=subprocess.PIPE,
                # This process may itself be speaking MCP over its own stdout, and
                # a child that printed there would corrupt that stream. stderr is
                # inherited so the child's tracebacks reach the same log as the
                # rest of the server's.
                stdout=subprocess.DEVNULL,
                env={**os.environ, **self._extra_environment},
            )
        except OSError as exc:
            raise IncodeError(
                ErrorCode.BACKEND_UNAVAILABLE,
                f"Could not start the accelerator interpreter: {exc}",
                interpreter=str(self.executable),
            ) from exc
        try:
            _write_handshake(popen, channel.handshake_payload(self._target))
        except OSError as exc:
            _reap(popen)
            raise IncodeError(
                ErrorCode.EMBEDDING_WORKER_FAILED,
                f"Could not hand the accelerator worker its connection details: {exc}",
            ) from exc
        return popen

    def _authenticate(self, connection: Any, authkey: bytes) -> None:
        """Prove both ends to each other, exactly as ``Listener.accept`` does."""
        try:
            deliver_challenge(connection, authkey)
            answer_challenge(connection, authkey)
        except (OSError, EOFError, ValueError) as exc:
            raise IncodeError(
                ErrorCode.EMBEDDING_WORKER_FAILED,
                f"The accelerator worker failed the connection handshake: {exc}",
            ) from exc


def _reap(popen: subprocess.Popen[bytes]) -> None:
    """Make sure a worker that never became usable does not outlive the attempt."""
    if popen.poll() is not None:
        return
    with suppress(OSError):
        popen.terminate()
    try:
        popen.wait(timeout=2)
    except subprocess.TimeoutExpired:
        with suppress(OSError):
            popen.kill()
        with suppress(subprocess.TimeoutExpired):
            popen.wait(timeout=2)


def _write_handshake(popen: subprocess.Popen[bytes], payload: str) -> None:
    assert popen.stdin is not None
    try:
        popen.stdin.write(f"{payload}\n".encode())
        popen.stdin.flush()
    finally:
        popen.stdin.close()


@dataclass(frozen=True)
class _Channel:
    """The listening socket an external worker dials back to."""

    server: socket.socket
    address: str | tuple[str, int]
    authkey: bytes
    directory: Path | None

    @classmethod
    def open(cls) -> _Channel:
        directory: Path | None = None
        if os.name == "nt":
            # Windows has no AF_UNIX story worth relying on here, so the channel
            # is a loopback socket on an ephemeral port. It is reachable by any
            # process on the machine, which is what the challenge-response is
            # for, and the key it protects never appears in an argument list.
            server = socket.socket(socket.AF_INET)
            server.bind(("127.0.0.1", 0))
            host, port = server.getsockname()[:2]
            address: str | tuple[str, int] = (str(host), int(port))
        else:
            directory = Path(tempfile.mkdtemp(prefix="incode-worker-"))
            server = socket.socket(socket.AF_UNIX)
            address = str(directory / "worker.sock")
            server.bind(address)
        server.listen(1)
        return cls(
            server=server,
            address=address,
            authkey=secrets.token_bytes(32),
            directory=directory,
        )

    def accept(self, popen: subprocess.Popen[bytes], *, timeout_seconds: float) -> Any:
        """Wait for the worker to dial back, or fail with why it never did."""
        deadline = time.monotonic() + timeout_seconds
        self.server.settimeout(HANDSHAKE_POLL_SECONDS)
        while True:
            try:
                client, _ = self.server.accept()
            except TimeoutError:
                exit_code = popen.poll()
                if exit_code is not None:
                    raise IncodeError(
                        ErrorCode.BACKEND_UNAVAILABLE,
                        f"The accelerator worker exited with status {exit_code} before it "
                        "could be reached; its environment is most likely incomplete",
                        exit_code=exit_code,
                    ) from None
                if time.monotonic() >= deadline:
                    raise IncodeError(
                        ErrorCode.EMBEDDING_WORKER_FAILED,
                        f"The accelerator worker did not connect within {timeout_seconds:.0f}s",
                    ) from None
                continue
            except OSError as exc:
                raise IncodeError(
                    ErrorCode.EMBEDDING_WORKER_FAILED,
                    f"Could not accept the accelerator worker connection: {exc}",
                ) from exc
            client.setblocking(True)
            # Connection is multiprocessing's socket-backed connection on every
            # platform, so the command protocol above it is identical either way.
            return Connection(client.detach())

    def handshake_payload(self, target: str) -> str:
        address = list(self.address) if isinstance(self.address, tuple) else self.address
        return json.dumps({"address": address, "authkey": self.authkey.hex(), "target": target})

    def close(self) -> None:
        try:
            self.server.close()
        finally:
            if self.directory is not None:
                shutil.rmtree(self.directory, ignore_errors=True)


# -- child side ------------------------------------------------------------


def _resolve_target(reference: str) -> WorkerTarget:
    """Import the ``module:function`` worker body the parent asked for."""
    from importlib import import_module

    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"Malformed worker target: {reference!r}")
    target: WorkerTarget = getattr(import_module(module_name), attribute)
    return target


def child_main(stream: Any = None) -> int:
    """Run as the worker: read the handshake, dial back, then serve commands.

    The heavy imports happen only once the connection exists, so the parent's
    dial-back timeout measures an interpreter starting rather than a model
    loading onto a device.
    """
    line = (sys.stdin if stream is None else stream).readline()
    if not line:
        return 2
    payload = json.loads(line)
    address = payload["address"]
    connection = Client(
        tuple(address) if isinstance(address, list) else address,
        authkey=bytes.fromhex(payload["authkey"]),
    )
    try:
        command, raw_config = connection.recv()
        if command != "configure":
            logger.error("Expected a configure command from the parent, got %r", command)
            return 2
        target = _resolve_target(str(payload.get("target") or DEFAULT_WORKER_TARGET))
        from .embedding_worker import WorkerConfig

        settings = dict(raw_config)
        settings["providers"] = tuple(settings.get("providers") or ())
        target(connection, WorkerConfig(**settings))
    finally:
        with suppress(OSError):
            connection.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(child_main())
