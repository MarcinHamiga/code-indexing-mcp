from __future__ import annotations

import pytest

from incode_mcp.backends import (
    CPU_BACKEND,
    CPU_PROVIDER,
    KNOWN_BACKENDS,
    Accelerator,
    BackendDescriptor,
    Precision,
    Stability,
    available_execution_providers,
    parse_accelerator,
    select_backend,
)
from incode_mcp.errors import ErrorCode, IncodeError

CUDA_PROVIDER = "CUDAExecutionProvider"
WEBGPU_PROVIDER = "WebGpuExecutionProvider"


def _descriptor(accelerator: Accelerator, provider: str, stability: Stability) -> BackendDescriptor:
    return BackendDescriptor(
        accelerator=accelerator,
        provider=provider,
        device="gpu",
        stability=stability,
        precision=Precision.FLOAT32,
    )


def _registry(*accelerators: BackendDescriptor) -> tuple[BackendDescriptor, ...]:
    return (CPU_BACKEND, *accelerators)


def test_cpu_is_always_selectable() -> None:
    selection = select_backend(Accelerator.CPU, available_providers=[CPU_PROVIDER])

    assert selection.accelerator is Accelerator.CPU
    assert selection.uses_accelerator is False
    assert selection.honored is True
    assert selection.fallback_reason is None


def test_auto_picks_the_first_automatic_backend_whose_provider_exists() -> None:
    registry = _registry(
        _descriptor(Accelerator.CUDA, CUDA_PROVIDER, Stability.AUTOMATIC),
        _descriptor(Accelerator.WEBGPU, WEBGPU_PROVIDER, Stability.AUTOMATIC),
    )

    selection = select_backend(
        Accelerator.AUTO,
        available_providers=[WEBGPU_PROVIDER, CUDA_PROVIDER, CPU_PROVIDER],
        registry=registry,
    )

    # Registry order decides, not the order the runtime happens to report.
    assert selection.accelerator is Accelerator.CUDA


def test_auto_skips_an_automatic_backend_the_runtime_does_not_offer() -> None:
    registry = _registry(
        _descriptor(Accelerator.CUDA, CUDA_PROVIDER, Stability.AUTOMATIC),
        _descriptor(Accelerator.WEBGPU, WEBGPU_PROVIDER, Stability.AUTOMATIC),
    )

    selection = select_backend(
        Accelerator.AUTO,
        available_providers=[WEBGPU_PROVIDER, CPU_PROVIDER],
        registry=registry,
    )

    assert selection.accelerator is Accelerator.WEBGPU


@pytest.mark.parametrize("stability", [Stability.EXPERIMENTAL, Stability.MANUAL])
def test_auto_never_picks_a_backend_below_automatic_stability(stability: Stability) -> None:
    registry = _registry(_descriptor(Accelerator.CUDA, CUDA_PROVIDER, stability))

    selection = select_backend(
        Accelerator.AUTO,
        available_providers=[CUDA_PROVIDER, CPU_PROVIDER],
        registry=registry,
    )

    assert selection.accelerator is Accelerator.CPU
    # Not a denied request: auto asked for the best qualifying backend and CPU
    # is the correct answer, so strict mode must not treat this as a failure.
    assert selection.honored is True
    assert selection.fallback_reason is not None
    selection.require_honored()


def test_an_explicit_request_overrides_stability() -> None:
    registry = _registry(_descriptor(Accelerator.CUDA, CUDA_PROVIDER, Stability.MANUAL))

    selection = select_backend(
        Accelerator.CUDA,
        available_providers=[CUDA_PROVIDER, CPU_PROVIDER],
        registry=registry,
    )

    assert selection.accelerator is Accelerator.CUDA
    assert selection.honored is True


def test_an_explicit_request_without_its_provider_falls_back_and_says_why() -> None:
    registry = _registry(_descriptor(Accelerator.CUDA, CUDA_PROVIDER, Stability.AUTOMATIC))

    selection = select_backend(
        Accelerator.CUDA, available_providers=[CPU_PROVIDER], registry=registry
    )

    assert selection.accelerator is Accelerator.CPU
    assert selection.honored is False
    assert CUDA_PROVIDER in (selection.fallback_reason or "")


def test_an_unregistered_explicit_request_falls_back_rather_than_raising() -> None:
    selection = select_backend(
        Accelerator.CUDA, available_providers=[CPU_PROVIDER], registry=(CPU_BACKEND,)
    )

    assert selection.accelerator is Accelerator.CPU
    assert selection.honored is False


def test_strict_mode_turns_a_denied_request_into_a_backend_error() -> None:
    registry = _registry(_descriptor(Accelerator.CUDA, CUDA_PROVIDER, Stability.AUTOMATIC))
    selection = select_backend(
        Accelerator.CUDA, available_providers=[CPU_PROVIDER], registry=registry
    )

    with pytest.raises(IncodeError) as caught:
        selection.require_honored()

    assert caught.value.code is ErrorCode.BACKEND_UNAVAILABLE
    assert caught.value.details["requested"] == "cuda"
    assert caught.value.details["resolved"] == "cpu"


def test_an_accelerator_keeps_cpu_behind_it_but_cpu_stands_alone() -> None:
    cuda = _descriptor(Accelerator.CUDA, CUDA_PROVIDER, Stability.AUTOMATIC)

    assert cuda.providers == (CUDA_PROVIDER, CPU_PROVIDER)
    # The CPU worker must pass no providers at all, so its own list stays bare.
    assert CPU_BACKEND.providers == (CPU_PROVIDER,)


def test_falling_back_records_the_new_backend_and_the_reason() -> None:
    registry = _registry(_descriptor(Accelerator.CUDA, CUDA_PROVIDER, Stability.AUTOMATIC))
    selection = select_backend(
        Accelerator.CUDA,
        available_providers=[CUDA_PROVIDER, CPU_PROVIDER],
        registry=registry,
    )

    degraded = selection.fell_back_to(CPU_BACKEND, "worker exited")

    assert degraded.accelerator is Accelerator.CPU
    assert degraded.requested is Accelerator.CUDA
    assert degraded.honored is False
    assert degraded.fallback_reason == "worker exited"
    # The original selection is untouched; a fallback is a new value.
    assert selection.accelerator is Accelerator.CUDA


def test_unknown_accelerator_names_are_a_configuration_error() -> None:
    with pytest.raises(IncodeError) as caught:
        parse_accelerator("tpu")

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION


@pytest.mark.parametrize("value", ["CUDA", " cpu ", "auto"])
def test_accelerator_names_are_parsed_leniently(value: str) -> None:
    assert parse_accelerator(value) in set(Accelerator)


def test_the_shipped_registry_promotes_nothing_to_automatic_yet() -> None:
    """Phase 2 ships the contract, not a promoted accelerator.

    Promotion is what Phase 3 does for CUDA once its correctness and throughput
    gates pass on real hardware. Until then ``auto`` must resolve to CPU even on
    a machine that has every provider installed.
    """
    automatic = [
        backend
        for backend in KNOWN_BACKENDS
        if not backend.is_cpu and backend.stability is Stability.AUTOMATIC
    ]

    assert automatic == []


def test_available_providers_always_include_cpu() -> None:
    assert CPU_PROVIDER in available_execution_providers()
