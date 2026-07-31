"""Accelerator planning, environment building, probing, and recording."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from .config_files import InstallerError, _atomic_write

# Accelerators this release can prepare, and the runtime extra each installs
# into an environment of its own. Core ML is still reached by explicit override
# inside the serving environment, so it needs no separate locked installation.
ACCELERATOR_EXTRAS = {
    "cuda": "cuda",
    "mlx": "mlx",
    "webgpu": "webgpu",
    "migraphx": "migraphx",
}
ACCELERATOR_CHOICES = ("auto", "cpu", "cuda", "mlx", "webgpu", "migraphx", "coreml")
ACCELERATOR_ENVIRONMENT_DIRECTORY = ".venv-accel"
# Bumped in lockstep with incode_mcp.accelerator_env.RECORD_SCHEMA_VERSION.
ACCELERATOR_RECORD_SCHEMA_VERSION = 1
# A cold probe downloads the embedding model before it can run an inference, so
# this has to cover a slow link as well as a slow device.
PROBE_TIMEOUT_SECONDS = 900

# The pinned CUDA support window for this release. onnxruntime-gpu 1.22-1.23
# builds against CUDA 12.x and cuDNN 9, and NVIDIA's minor-version compatibility
# makes the driver below the floor for every 12.x runtime. A driver under it is
# reported and left alone: the installer never touches system drivers.
MINIMUM_NVIDIA_DRIVER = {"linux": (525, 60), "win32": (527, 41)}
# Lower-cased `platform.machine()` values, matching the `platform_machine`
# markers on the cuda extra exactly. A machine name the markers would miss must
# be refused here rather than nominated: `uv sync --extra cuda` would resolve
# that extra to nothing and build an environment with no embedding runtime in
# it at all, which fails the probe for a reason that explains nothing.
CUDA_PLATFORMS = {"linux": {"x86_64"}, "win32": {"amd64"}}
# The native WebGPU plugin/core pair's published wheels. The plugin's macOS
# wheel is universal2, but ONNX Runtime 1.24.4 itself is arm64-only there and
# has a deployment target of 14.0. Linux and Windows publish x86-64 wheels.
WEBGPU_PLATFORMS = {
    "darwin": {"arm64"},
    "linux": {"x86_64"},
    "win32": {"amd64"},
}
MINIMUM_WEBGPU_MACOS = (14, 0)
# MLX's Metal backend needs Apple Silicon, and its published wheels start at
# macOS 14. It also ships CPU-only Linux and Windows wheels, which are excluded
# here: preparing a "Metal" environment with no Metal in it would pass its own
# probe and then lose to the CPU it really is.
MLX_PLATFORMS = {"darwin": {"arm64"}}
MINIMUM_MLX_MACOS = (14, 0)
# AMD publishes this ONNX Runtime/MIGraphX combination as a single wheel rather
# than on PyPI. Nomination stays exact so the installer never assembles an
# untested Python/ROCm pair around it.
MIGRAPHX_PLATFORM = ("linux", "x86_64")
MIGRAPHX_PYTHON_VERSION = "3.12"
MIGRAPHX_ROCM_VERSION = "7.2.1"

def _run_command(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            env=None if environment is None else {**os.environ, **environment},
        )
    except FileNotFoundError as exc:
        raise InstallerError(f"Required command was not found: {arguments[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        command = " ".join(arguments)
        message = f"Command failed: {command}"
        if detail:
            message = f"{message}\n{detail}"
        raise InstallerError(message) from exc

def server_executable(
    install_directory: Path,
    *,
    platform_name: str | None = None,
) -> Path:
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return install_directory / ".venv" / "Scripts" / "code-indexing-mcp.exe"
    return install_directory / ".venv" / "bin" / "code-indexing-mcp"


def environment_python(directory: Path, *, platform_name: str | None = None) -> Path:
    """Return the interpreter inside a virtual environment directory."""

    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return directory / "Scripts" / "python.exe"
    return directory / "bin" / "python"


def _uv_executable(uv_executable: str | None) -> str:
    uv = uv_executable or shutil.which("uv")
    if uv is None:
        raise InstallerError(
            "uv is required but was not found in PATH. Install it from https://docs.astral.sh/uv/"
        )
    return uv

class AcceleratorPlan(NamedTuple):
    """What the installer will prepare, and the reason it settled on that.

    ``accelerator`` is ``"cpu"`` when nothing will be prepared, which is an
    outcome rather than an error: every machine indexes on CPU.
    """

    accelerator: str
    reason: str
    driver_version: str = ""
    device_name: str = ""
    # False when the request could not be honoured, which is what decides
    # whether the outcome is reported as a problem. A CPU result is not by
    # itself a denial: ``--accelerator cpu`` asked for exactly this, ``auto``
    # finding no GPU is just what the machine is, and Core ML needs nothing
    # prepared at all.
    honored: bool = True
    # Hash of the lockfile and selected extra. A record without this exact
    # fingerprint describes an older resolved runtime and must be rebuilt.
    lock_fingerprint: str = ""

    @property
    def prepares_environment(self) -> bool:
        return self.accelerator in ACCELERATOR_EXTRAS


def _nvidia_smi_report() -> str | None:
    """Return nvidia-smi's driver/name line, or None when there is no driver."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        # A driver too broken to answer is a driver this installer will not
        # build on top of, and not a reason to fail the whole installation.
        return None
    return result.stdout if result.returncode == 0 else None


def _rocm_report() -> str | None:
    """Return the installed ROCm version and, when available, an AMD device."""

    version = ""
    for path in (Path("/opt/rocm/.info/version"), Path("/opt/rocm/.info/version-dev")):
        try:
            contents = path.read_text(encoding="utf-8")
        except OSError:
            continue
        match = re.search(r"\d+\.\d+(?:\.\d+)?", contents)
        if match is not None:
            version = match.group()
            break
    if not version:
        return None

    device = ""
    executable = shutil.which("rocminfo")
    if executable is not None:
        try:
            result = subprocess.run(
                [executable],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0:
            match = re.search(r"(?m)^\s*Marketing Name:\s*(.+?)\s*$", result.stdout)
            if match is not None and match.group(1).strip().lower() != "unknown":
                device = match.group(1).strip()
    return f"{version}, {device or 'AMD GPU'}"


def _driver_components(version: str) -> tuple[int, ...]:
    components: list[int] = []
    for part in version.strip().split("."):
        if not part.isdigit():
            break
        components.append(int(part))
    return tuple(components)


def _normalized_platform(platform_name: str) -> str:
    return "win32" if platform_name.startswith("win") else platform_name


def _webgpu_plan(
    *,
    platform_name: str,
    machine: str,
    platform_version: str,
    reason_prefix: str = "",
) -> AcceleratorPlan:
    supported = WEBGPU_PLATFORMS.get(platform_name)
    problem = ""
    if supported is None or machine not in supported:
        problem = f"no native WebGPU plugin wheel is published for {platform_name}/{machine}"
    elif platform_name == "darwin":
        components = _driver_components(platform_version)
        if not components or components < MINIMUM_WEBGPU_MACOS:
            problem = (
                f"the locked WebGPU plugin requires macOS "
                f"{'.'.join(str(part) for part in MINIMUM_WEBGPU_MACOS)} or newer"
            )

    if problem:
        prefix = f"{reason_prefix}; " if reason_prefix else "WebGPU was requested but "
        return AcceleratorPlan("cpu", f"{prefix}{problem}", honored=False)
    reason = (
        f"the locked WebGPU plugin is available for {platform_name}/{machine}"
        if not reason_prefix
        else f"{reason_prefix}; falling back to WebGPU with the locked plugin"
    )
    return AcceleratorPlan("webgpu", reason, honored=not reason_prefix)


def _mlx_problem(*, platform_name: str, machine: str, platform_version: str) -> str:
    """Return why MLX cannot be prepared here, or an empty string when it can."""

    supported = MLX_PLATFORMS.get(platform_name)
    if supported is None or machine not in supported:
        return f"MLX runs on Metal, and there is no Metal on {platform_name}/{machine}"
    components = _driver_components(platform_version)
    if not components or components < MINIMUM_MLX_MACOS:
        return (
            f"the locked MLX build requires macOS "
            f"{'.'.join(str(part) for part in MINIMUM_MLX_MACOS)} or newer"
        )
    return ""


def _mlx_plan(*, platform_name: str, machine: str, platform_version: str) -> AcceleratorPlan:
    problem = _mlx_problem(
        platform_name=platform_name, machine=machine, platform_version=platform_version
    )
    if problem:
        # MIGraphX degrades to WebGPU because both are cross-vendor GPU paths on
        # the same machine. A request for Metal where there is no Metal is not a
        # request for Vulkan or D3D12, so this reports CPU rather than
        # substituting one.
        return AcceleratorPlan("cpu", f"MLX was requested but {problem}", honored=False)
    return AcceleratorPlan(
        "mlx",
        f"the locked MLX build is available for macOS {platform_version} on {machine}",
        # Recorded as the driver version because it is what the probe result is
        # only valid for: Metal comes with the OS.
        driver_version=platform_version,
        device_name="Apple Silicon GPU",
    )


def plan_accelerator(
    requested: str,
    *,
    platform_name: str | None = None,
    machine: str | None = None,
    nvidia_report: Callable[[], str | None] = _nvidia_smi_report,
    rocm_report: Callable[[], str | None] = _rocm_report,
    python_version: str | None = None,
    platform_version: str | None = None,
) -> AcceleratorPlan:
    """Decide which accelerator, if any, this machine should have prepared.

    Detection only nominates: the environment still has to build and pass a real
    inference probe before anything offers the backend to the server.
    """

    platform_name = _normalized_platform((platform_name or sys.platform).lower())
    machine = (machine or platform.machine()).lower()
    python_version = python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
    if platform_version is None:
        platform_version = platform.mac_ver()[0] if platform_name == "darwin" else ""
    requested = requested.strip().lower()

    if requested == "cpu":
        return AcceleratorPlan("cpu", "CPU was requested")
    if requested == "coreml":
        # Not a denial: Core ML runs in the serving environment's own runtime,
        # so there is genuinely nothing for this installer to prepare.
        return AcceleratorPlan(
            "cpu",
            "Core ML needs no separate environment and stays manual-only: it lost to "
            "CPU on this model. Set INCODE_EMBED_ACCELERATOR=coreml to measure it",
        )
    if requested == "mlx":
        return _mlx_plan(
            platform_name=platform_name,
            machine=machine,
            platform_version=platform_version,
        )
    if requested == "webgpu":
        return _webgpu_plan(
            platform_name=platform_name,
            machine=machine,
            platform_version=platform_version,
        )
    if requested == "migraphx":
        problem = ""
        if (platform_name, machine) != MIGRAPHX_PLATFORM:
            problem = (
                f"the pinned MIGraphX wheel is published only for "
                f"{MIGRAPHX_PLATFORM[0]}/{MIGRAPHX_PLATFORM[1]}"
            )
        elif python_version != MIGRAPHX_PYTHON_VERSION:
            problem = (
                f"the pinned MIGraphX wheel requires Python {MIGRAPHX_PYTHON_VERSION}, "
                f"not {python_version}"
            )
        else:
            report = rocm_report()
            if not report or not report.strip():
                problem = "ROCm was not detected"
            else:
                first = report.strip().splitlines()[0]
                rocm_version, _, device_name = (part.strip() for part in first.partition(","))
                if rocm_version != MIGRAPHX_ROCM_VERSION:
                    problem = (
                        f"ROCm {rocm_version or 'unknown'} does not match the pinned "
                        f"{MIGRAPHX_ROCM_VERSION} runtime"
                    )
                else:
                    return AcceleratorPlan(
                        "migraphx",
                        f"ROCm {rocm_version} on {device_name or 'an AMD device'} matches "
                        "the pinned MIGraphX runtime",
                        driver_version=rocm_version,
                        device_name=device_name,
                    )
        return _webgpu_plan(
            platform_name=platform_name,
            machine=machine,
            platform_version=platform_version,
            reason_prefix=f"MIGraphX was requested but {problem}",
        )

    if requested != "cuda" and not _mlx_problem(
        platform_name=platform_name, machine=machine, platform_version=platform_version
    ):
        # `auto` on Apple Silicon: MLX passed the same correctness and 1.25x
        # performance gates CUDA did, so it is prepared without being asked for.
        # An unsupported Mac falls through to the CUDA path below, which reports
        # what every machine without a GPU this release can use reports.
        return _mlx_plan(
            platform_name=platform_name,
            machine=machine,
            platform_version=platform_version,
        )

    supported = CUDA_PLATFORMS.get(platform_name)
    # `auto` finding no CUDA is what most machines are, not a request denied.
    explicit = "CUDA was requested but " if requested == "cuda" else ""
    honored = not explicit
    if supported is None or machine not in supported:
        return AcceleratorPlan(
            "cpu",
            f"{explicit}no CUDA wheels are published for {platform_name}/{machine}",
            honored=honored,
        )
    report = nvidia_report()
    if not report or not report.strip():
        return AcceleratorPlan(
            "cpu",
            f"{explicit}no usable NVIDIA driver was detected (nvidia-smi reported nothing)",
            honored=honored,
        )
    first = report.strip().splitlines()[0]
    driver_version, _, device_name = (part.strip() for part in first.partition(","))
    floor = MINIMUM_NVIDIA_DRIVER[platform_name]
    components = _driver_components(driver_version)
    if not components or components < floor:
        return AcceleratorPlan(
            "cpu",
            f"{explicit}NVIDIA driver {driver_version or 'unknown'} is below the "
            f"{'.'.join(str(part) for part in floor)} this release's CUDA 12 runtime "
            "needs; the installer does not change drivers",
            driver_version=driver_version,
            device_name=device_name,
            honored=honored,
        )
    return AcceleratorPlan(
        "cuda",
        f"NVIDIA driver {driver_version} on {device_name or 'an NVIDIA device'} "
        "satisfies the pinned CUDA 12 runtime",
        driver_version=driver_version,
        device_name=device_name,
    )


def interpreter_version(python: Path) -> str:
    """Return the ``major.minor`` version of an interpreter."""

    result = _run_command([str(python), "-c", "import sys;print('%d.%d'%sys.version_info[:2])"])
    return result.stdout.strip()


def accelerator_lock_fingerprint(install_directory: Path, accelerator: str) -> str:
    """Hash the selected extra and the lockfile that resolved its environment."""

    lockfile = install_directory / "uv.lock"
    try:
        locked = lockfile.read_bytes()
    except OSError as exc:
        raise InstallerError(
            f"The accelerator lockfile cannot be read at {lockfile}: {exc}"
        ) from exc
    digest = hashlib.sha256()
    digest.update(accelerator.encode())
    digest.update(b"\0")
    digest.update(locked)
    return digest.hexdigest()


def runtime_record_path(python: Path) -> Path:
    """Ask the installed package where the server reads its accelerator record.

    The package is asked rather than told: it owns the filename, and it honours
    an ``INCODE_ACCEL_ENV`` override that a path assembled here would write
    straight past, leaving the record somewhere the server never looks.
    """

    result = _run_command(
        [
            str(python),
            "-c",
            "from incode_mcp.accelerator_env import record_path;"
            "from incode_mcp.application import RuntimePaths;"
            "print(record_path(RuntimePaths.from_environment().data))",
        ]
    )
    return Path(result.stdout.strip())


def accelerator_record_path(install_directory: Path, *, platform_name: str | None = None) -> Path:
    python = environment_python(install_directory / ".venv", platform_name=platform_name)
    return runtime_record_path(python)


def sync_accelerator_environment(
    install_directory: Path,
    extra: str,
    *,
    python_version: str,
    uv_executable: str | None = None,
    platform_name: str | None = None,
) -> Path:
    """Build the accelerator's own locked environment and return its interpreter.

    Whenever this runs it builds from empty, never over what is already there:
    an environment carrying leftovers from an earlier extra would resolve its
    ONNX Runtime to whichever distribution landed last, which is the exact
    failure the extras are separated to avoid. Deciding whether it needs to run
    at all is the caller's job -- see ``reusable_accelerator_environment``.
    """

    uv = _uv_executable(uv_executable)
    directory = install_directory / ACCELERATOR_ENVIRONMENT_DIRECTORY
    if directory.exists():
        try:
            shutil.rmtree(directory)
        except OSError as exc:
            # Not ignore_errors: building over a half-removed environment is the
            # ONNX Runtime collision this function exists to prevent, so the
            # removal has to succeed or the build has to stop. Stopping is said
            # in the installer's own vocabulary, though -- the caller degrades to
            # CPU on an InstallerError, and a raw OSError from a file the machine
            # merely had locked would take the whole installation down instead.
            raise InstallerError(
                f"Could not remove the existing accelerator environment at {directory}: {exc}"
            ) from exc
    _run_command(
        [
            uv,
            "sync",
            "--locked",
            "--no-default-groups",
            "--extra",
            extra,
            # Both ends of the worker channel speak multiprocessing's connection
            # protocol, so the accelerator interpreter has to match the server's.
            "--python",
            python_version,
        ],
        cwd=install_directory,
        environment={"UV_PROJECT_ENVIRONMENT": str(directory)},
    )
    python = environment_python(directory, platform_name=platform_name)
    if not python.is_file():
        raise InstallerError(f"The accelerator environment has no interpreter at {python}")
    return python


def probe_accelerator(python: Path, accelerator: str, *, offline: bool = False) -> dict[str, Any]:
    """Run a real inference in the accelerator environment and return its report."""

    arguments = [str(python), "-m", "incode_mcp.accelerator_probe", "--accelerator", accelerator]
    if offline:
        arguments.append("--offline")
    try:
        result = subprocess.run(
            arguments, capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired as exc:
        # Generous, because a cold probe downloads the model before it can embed
        # anything -- but bounded, because a driver that wedges initialising a
        # device wedges there forever, and the output is captured, so an
        # unbounded wait would look exactly like an installer that had hung.
        raise InstallerError(
            f"The accelerator probe did not finish within {PROBE_TIMEOUT_SECONDS // 60} minutes"
        ) from exc
    except OSError as exc:
        raise InstallerError(f"Could not run the accelerator probe: {exc}") from exc
    payload: Any = None
    for line in reversed(result.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        break
    if not isinstance(payload, dict):
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        raise InstallerError(
            "The accelerator probe returned no report"
            + (f": {detail[-1]}" if detail else f" (exit status {result.returncode})")
        )
    if not payload.get("ok"):
        raise InstallerError(f"The accelerator probe failed: {payload.get('error', 'unknown')}")
    return payload


def write_accelerator_record(path: Path, plan: AcceleratorPlan, probe: Mapping[str, Any]) -> None:
    """Record the verified environment where the server looks for one.

    The shape is read back by ``incode_mcp.accelerator_env``; the schema version
    is what keeps a record written here from being misread by a server that
    changed its mind about what these fields mean.
    """

    record = {
        "schema_version": ACCELERATOR_RECORD_SCHEMA_VERSION,
        "accelerator": plan.accelerator,
        "interpreter": str(probe["interpreter"]),
        "providers": list(probe["providers"]),
        "runtime_version": str(probe.get("runtime_version", "")),
        "lock_fingerprint": plan.lock_fingerprint,
        "driver_version": plan.driver_version,
        "device": str(probe.get("device", "")),
        "python_version": str(probe.get("python_version", "")),
        "recorded_at_ns": time.time_ns(),
        "detail": str(probe.get("detail", "")),
    }
    _atomic_write(path, json.dumps(record, indent=2, sort_keys=True) + "\n")


def clear_accelerator_record(path: Path) -> bool:
    """Drop any record, so an installation that fell back stops offering more."""

    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise InstallerError(
            f"Could not remove the stale accelerator record: {path}: {exc}"
        ) from exc
    return True


def reusable_accelerator_environment(
    path: Path, plan: AcceleratorPlan, *, python_version: str
) -> Path | None:
    """Return the interpreter an existing record still vouches for, if any.

    Rebuilding a multi-gigabyte environment and re-probing a device on every
    update is a lot of work to arrive back where the last run already was. The
    record is reused only when it describes this exact plan running on this
    exact interpreter; anything that moved -- the driver, the server's Python,
    the environment itself -- puts the full build and probe back.
    """

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    interpreter = Path(str(record.get("interpreter", "")))
    matches = (
        record.get("schema_version") == ACCELERATOR_RECORD_SCHEMA_VERSION
        and record.get("accelerator") == plan.accelerator
        and str(record.get("lock_fingerprint", "")) == plan.lock_fingerprint
        and str(record.get("driver_version", "")) == plan.driver_version
        and str(record.get("python_version", "")) == python_version
        and interpreter.is_file()
    )
    return interpreter if matches else None


def configure_accelerator(
    install_directory: Path,
    requested: str,
    *,
    uv_executable: str | None = None,
    platform_name: str | None = None,
    machine: str | None = None,
    nvidia_report: Callable[[], str | None] = _nvidia_smi_report,
    rocm_report: Callable[[], str | None] = _rocm_report,
    python_version: str | None = None,
    platform_version: str | None = None,
    offline: bool = False,
) -> AcceleratorPlan:
    """Prepare the planned accelerator, or leave the installation on CPU.

    Every failure below is a fall back to CPU with the reason attached, not an
    installation failure: an accelerator that cannot be built or cannot pass its
    probe costs speed, and refusing to install over it would cost the server.
    """

    serving_python = environment_python(install_directory / ".venv", platform_name=platform_name)
    detected_python_version: str | None = None
    planning_python_version = python_version
    planning_error: InstallerError | None = None
    if requested.strip().lower() == "migraphx" and planning_python_version is None:
        try:
            detected_python_version = interpreter_version(serving_python)
            planning_python_version = detected_python_version
        except InstallerError as exc:
            planning_error = exc
    if planning_error is None:
        plan = plan_accelerator(
            requested,
            platform_name=platform_name,
            machine=machine,
            nvidia_report=nvidia_report,
            rocm_report=rocm_report,
            python_version=planning_python_version,
            platform_version=platform_version,
        )
    else:
        plan = AcceleratorPlan(
            "cpu",
            f"MIGraphX was requested but the serving Python version could not be resolved: "
            f"{planning_error}",
            honored=False,
        )
    try:
        record = accelerator_record_path(install_directory, platform_name=platform_name)
    except InstallerError as exc:
        # Without the server's data directory there is nowhere to offer an
        # accelerator from, and nowhere a stale offer could be retracted from
        # either. Reporting that is the whole of what can be done about it; the
        # installation itself is fine and indexes on CPU.
        return AcceleratorPlan(
            "cpu",
            f"the server's runtime data directory could not be resolved: {exc}",
            honored=False,
        )
    if not plan.prepares_environment:
        clear_accelerator_record(record)
        # Once no record points at it, the environment is several gigabytes of
        # dead weight. A machine reinstalled as CPU-only should not keep paying
        # the disk for the GPU it used to have.
        shutil.rmtree(install_directory / ACCELERATOR_ENVIRONMENT_DIRECTORY, ignore_errors=True)
        return plan

    try:
        serving_python_version = detected_python_version or interpreter_version(serving_python)
        plan = plan._replace(
            lock_fingerprint=accelerator_lock_fingerprint(
                install_directory,
                ACCELERATOR_EXTRAS[plan.accelerator],
            )
        )
        reused = reusable_accelerator_environment(
            record,
            plan,
            python_version=serving_python_version,
        )
        if reused is not None:
            return plan._replace(reason=f"{plan.reason}; reusing the environment at {reused}")
        python = sync_accelerator_environment(
            install_directory,
            ACCELERATOR_EXTRAS[plan.accelerator],
            python_version=serving_python_version,
            uv_executable=uv_executable,
            platform_name=platform_name,
        )
        probe = probe_accelerator(python, plan.accelerator, offline=offline)
    except InstallerError as exc:
        # Nothing half-built may be left where the server could find it: the
        # record is what makes an environment reachable at all.
        clear_accelerator_record(record)
        shutil.rmtree(install_directory / ACCELERATOR_ENVIRONMENT_DIRECTORY, ignore_errors=True)
        return AcceleratorPlan(
            "cpu",
            f"{plan.accelerator} was detected but could not be prepared: {exc}",
            driver_version=plan.driver_version,
            device_name=plan.device_name,
            # Detection said this machine has the hardware and it still did not
            # come up. That is worth reporting however it was requested.
            honored=False,
        )
    write_accelerator_record(record, plan, probe)
    return plan._replace(reason=f"{plan.reason}; {probe.get('detail', 'probe passed')}")
