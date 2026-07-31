from __future__ import annotations

import pytest

from code_indexing_mcp.backends import (
    CPU_BACKEND,
    CPU_PROVIDER,
    KNOWN_BACKENDS,
    MLX_PROVIDER,
    Accelerator,
    BackendDescriptor,
    Precision,
    Runtime,
    Stability,
    available_execution_providers,
    backend_for,
    parse_accelerator,
    select_backend,
)
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode

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

    with pytest.raises(CodeIndexingError) as caught:
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
    with pytest.raises(CodeIndexingError) as caught:
        parse_accelerator("tpu")

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION


@pytest.mark.parametrize("value", ["CUDA", " cpu ", "auto"])
def test_accelerator_names_are_parsed_leniently(value: str) -> None:
    assert parse_accelerator(value) in set(Accelerator)


def test_only_backends_that_passed_their_gates_are_eligible_automatically() -> None:
    """CUDA and MLX are promoted; the rest still need an explicit override.

    The rest of the registry is reached only through CODE_INDEXING_EMBED_ACCELERATOR,
    which is how a backend earns the measurements its own promotion needs.
    """
    automatic = [
        backend.accelerator
        for backend in KNOWN_BACKENDS
        if not backend.is_cpu and backend.stability is Stability.AUTOMATIC
    ]

    assert automatic == [Accelerator.CUDA, Accelerator.MLX]


def test_auto_stays_on_cpu_where_no_accelerator_was_prepared() -> None:
    """Promotion makes CUDA eligible, not present: an unprepared machine is CPU."""
    selection = select_backend(Accelerator.AUTO, available_providers=[CPU_PROVIDER])

    assert selection.accelerator is Accelerator.CPU
    assert selection.honored is True
    assert "reinstall with --accelerator" in (selection.fallback_reason or "")


def test_available_providers_always_include_cpu() -> None:
    assert CPU_PROVIDER in available_execution_providers()


def test_mlx_is_registered_as_a_promoted_metal_backend() -> None:
    mlx = backend_for(Accelerator.MLX)

    assert mlx is not None
    assert mlx.runtime is Runtime.MLX
    assert mlx.provider == MLX_PROVIDER
    assert mlx.stability is Stability.AUTOMATIC
    assert mlx.uses_direct_model is True
    # MLX is not ONNX Runtime, so its target is never in a provider list the
    # runtime published before anything was loaded -- and its environment has no
    # provider list to publish at all.
    assert mlx.provider_is_preregistered is False
    assert mlx.runs_on_onnx is False
    assert mlx.publishes_execution_providers is False


def test_a_plugin_provider_is_still_an_onnx_runtime_backend() -> None:
    """Only the runtime is non-ONNX for MLX. WebGPU's provider merely arrives
    late, so its environment still has an ONNX provider list to report."""
    webgpu = backend_for(Accelerator.WEBGPU)

    assert webgpu is not None
    assert webgpu.runtime is Runtime.ONNX_PLUGIN
    assert webgpu.provider_is_preregistered is False
    assert webgpu.runs_on_onnx is True
    assert webgpu.publishes_execution_providers is True


def test_an_mlx_backend_has_no_onnx_cpu_provider_behind_it() -> None:
    """CPU sits behind an ONNX accelerator so an unpartitionable graph still runs.

    MLX has no graph partitioning and no ONNX session, so naming the ONNX CPU
    provider there would describe a fallback that does not exist.
    """
    mlx = backend_for(Accelerator.MLX)
    assert mlx is not None

    assert mlx.providers == (MLX_PROVIDER,)


def test_an_explicit_mlx_request_is_honoured_against_a_prepared_record() -> None:
    selection = select_backend(Accelerator.MLX, available_providers=[CPU_PROVIDER, MLX_PROVIDER])

    assert selection.accelerator is Accelerator.MLX
    assert selection.honored is True
    assert selection.uses_accelerator is True


def test_auto_selects_a_prepared_mlx_environment() -> None:
    selection = select_backend(Accelerator.AUTO, available_providers=[CPU_PROVIDER, MLX_PROVIDER])

    assert selection.accelerator is Accelerator.MLX
    assert selection.honored is True
    assert selection.fallback_reason is None
