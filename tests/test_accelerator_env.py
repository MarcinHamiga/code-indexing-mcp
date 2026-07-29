from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

from incode_mcp.accelerator_env import (
    RECORD_FILENAME,
    RECORD_PATH_VARIABLE,
    AcceleratorEnvironment,
    apply_environment,
    clear_environment,
    load_environment,
    record_path,
    running_python_version,
    write_environment,
)
from incode_mcp.backends import ACCELERATOR_BACKENDS, CPU_BACKEND, Accelerator

CUDA_BACKEND = next(
    backend for backend in ACCELERATOR_BACKENDS if backend.accelerator is Accelerator.CUDA
)


def _record(tmp_path: Path, **overrides: object) -> AcceleratorEnvironment:
    interpreter = tmp_path / "venv-accel" / "python"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("", encoding="utf-8")
    settings: dict[str, object] = {
        "accelerator": Accelerator.CUDA,
        "interpreter": interpreter,
        "providers": ("CUDAExecutionProvider", "CPUExecutionProvider"),
        "runtime_version": "1.23.2",
        "driver_version": "550.54.14",
        "device": "cuda:0",
        "python_version": running_python_version(),
        "recorded_at_ns": 1,
        "detail": "probed 2 passages on CUDAExecutionProvider",
    }
    settings.update(overrides)
    return AcceleratorEnvironment(**settings)  # type: ignore[arg-type]


def test_a_record_round_trips_through_the_file_the_installer_writes(tmp_path: Path) -> None:
    data = tmp_path / "data"
    record = _record(tmp_path)

    write_environment(data / RECORD_FILENAME, record)
    status = load_environment(data)

    assert status.environment == record
    assert status.reason is None
    assert status.providers == ("CUDAExecutionProvider", "CPUExecutionProvider")


def test_no_record_at_all_is_a_cpu_installation_not_a_problem(tmp_path: Path) -> None:
    status = load_environment(tmp_path / "data")

    assert status.environment is None
    assert status.reason is None
    assert status.providers == ()


def test_a_record_naming_a_vanished_interpreter_is_refused_with_a_reason(
    tmp_path: Path,
) -> None:
    """The environment can be deleted long after the record was written."""
    data = tmp_path / "data"
    record = _record(tmp_path)
    write_environment(data / RECORD_FILENAME, record)
    record.interpreter.unlink()

    status = load_environment(data)

    assert status.environment is None
    assert "interpreter is gone" in (status.reason or "")


def test_a_record_built_for_another_python_is_refused(tmp_path: Path) -> None:
    """Both ends of the worker channel speak one Python's connection protocol."""
    data = tmp_path / "data"
    write_environment(data / RECORD_FILENAME, _record(tmp_path, python_version="3.7"))

    status = load_environment(data)

    assert status.environment is None
    assert "built for Python 3.7" in (status.reason or "")


def test_a_record_from_another_schema_version_is_ignored_rather_than_reinterpreted(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    path = data / RECORD_FILENAME
    write_environment(path, _record(tmp_path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")

    status = load_environment(data)

    assert status.environment is None
    assert "schema version" in (status.reason or "")


def test_an_unreadable_record_reports_why_rather_than_raising(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / RECORD_FILENAME).write_text("{not json", encoding="utf-8")

    status = load_environment(data)

    assert status.environment is None
    assert "unreadable" in (status.reason or "")


def test_a_record_without_providers_or_with_auto_is_unusable(tmp_path: Path) -> None:
    data = tmp_path / "data"
    path = data / RECORD_FILENAME
    write_environment(path, _record(tmp_path))
    payload = json.loads(path.read_text(encoding="utf-8"))

    payload["providers"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert "no execution providers" in (load_environment(data).reason or "")

    payload["providers"] = ["CUDAExecutionProvider"]
    payload["accelerator"] = "auto"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert "selection policy" in (load_environment(data).reason or "")


def test_an_explicit_record_path_overrides_the_data_directory(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    record = _record(tmp_path)
    write_environment(elsewhere / RECORD_FILENAME, record)
    environment = {RECORD_PATH_VARIABLE: str(elsewhere)}

    assert record_path(tmp_path / "data", environment) == elsewhere / RECORD_FILENAME
    assert load_environment(tmp_path / "data", environment).environment == record


def test_clearing_a_record_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / RECORD_FILENAME
    write_environment(path, _record(tmp_path))

    assert clear_environment(path) is True
    assert clear_environment(path) is False


def test_a_descriptor_is_described_only_by_its_own_environment(tmp_path: Path) -> None:
    """A CUDA record says nothing about what a CPU or Core ML session runs on."""
    record = _record(tmp_path)

    described = apply_environment(CUDA_BACKEND, record)
    assert described.driver_version == "550.54.14"
    assert described.runtime_version == "1.23.2"
    assert described.device == "cuda:0"

    assert apply_environment(CPU_BACKEND, record) == CPU_BACKEND


def test_the_recorded_interpreter_is_the_one_this_platform_would_run(tmp_path: Path) -> None:
    """The record is written by an installer on one machine and read on it."""
    record = _record(tmp_path, interpreter=Path(sys.executable))

    assert replace(record, detail="").verify() is None
