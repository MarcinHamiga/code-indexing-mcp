# Phase 5 Adaptive Selection and Hardening: Shipped

## Outcome

Every backend is now measured rather than assumed. A verified accelerator is calibrated once per
configuration — batch size, warm throughput, and cold-load cost — CPU is calibrated beside it so the
comparison has both sides, and the run size above which starting the accelerator repays its model
load follows from the two. Microbatches are bounded by the configured memory as well as by item and
token count, and a size a ceiling overrun forced down is remembered instead of rediscovered.

This closes the long-term plan's Phase 5 and the Phase 3 release gate it left open: *small
incremental jobs remain on CPU whenever accelerator startup would make them slower.*

No index migration is required. Model identity, tokenizer, pooling, normalisation, and vector shape
are untouched; crossing over mid-run mixes CPU and accelerator vectors exactly as the existing
fallback always has, which is what the parity gates cover.

## Delivered

- Calibration through the ordinary `plan_and_embed` request rather than a protocol of its own, so it
  measures the IPC the real work pays and covers the FastEmbed, direct-ONNX, and MLX workers with one
  implementation. It sweeps a doubling ladder of batch sizes, stops as soon as a larger batch stops
  paying for the memory it holds, and treats a ceiling overrun as that size being unsafe rather than
  as the backend being broken.
- Throughput measured in characters per second — the only unit known in the parent before the work is
  handed over — against a deterministic, code-shaped corpus at two representative lengths.
- The measurement stored in the probe cache, which already keys by model artifact, accelerator,
  provider, runtime version, platform, device, and driver. The schema version rose so a record
  written before calibration is re-measured rather than read as a backend with a rate of zero.
- A workload crossover computed from both loads and both rates, and applied mid-run: a session that
  selected an accelerator embeds on CPU until the run passes the threshold, counting the request in
  hand so the one group large enough to justify the device is not itself sent to CPU. A deferral is
  not a fallback and does not pin the process to CPU.
- `max_token_product` derived from `INCODE_EMBED_MEMORY_MB` instead of the constant measured at
  2 GiB, floored at one longest window and capped at eight times the reference.
- A reduced batch size written back to the cache and reported as `"reduced"`, distinct from
  `"measured"`.
- `INCODE_EMBED_CROSSOVER=auto|off|<characters>` and `INCODE_EMBED_CALIBRATE=0|1`, both of which
  restore the previous behaviour exactly.
- `model status` reporting both rates, the cold-load cost, the crossover, and a recommended override
  when the numbers argue for one; `IndexReport` reporting `embedded_characters`,
  `embedding_crossover_characters`, and `embedding_selection_reason`.

## What measuring real hardware changed

The first formula charged the accelerator its whole cold-load cost against a CPU assumed to start
instantly. Calibrating an M4 Pro through the real workers showed that to be wrong in a way no unit
test would have caught: staying on CPU also spawns a worker and loads a model, and the converted MLX
weights are memory-mapped while the CPU ONNX graph is read and prepared, so the accelerator loads
*faster* than the fallback.

| Backend | Cold load | Calibrated `max_items` | Throughput |
| --- | ---: | ---: | ---: |
| CPU (`fastembed`, 2 threads) | 655 ms | 1 | 14,030 chars/s |
| MLX (Metal, converted weights) | 370 ms | 2 | 46,783 chars/s |

What has to be earned back is the *difference* between the two loads. It is negative here, so the
crossover on this machine is zero and MLX starts on the first chunk — which is the right answer, and
the opposite of what the original formula would have produced. The deferral machinery still applies
wherever the accelerator does cost more to start than CPU, which is the discrete-GPU shape: CUDA
loads a model into VRAM rather than mapping it into memory the CPU model is already in.

The calibration corpus ratio (3.3×) is not the end-to-end indexing ratio Phase 4B measured (1.52×).
It is not meant to be: it times embedding alone, on one corpus shape, where the index gate times
scan, parse, embed, and commit over a real repository. The crossover only needs the two rates to be
comparable with each other, which they are — both measured the same way, minutes apart, on the same
machine.

## Verification commands

```bash
uv lock --check
uv run --extra cpu pytest
uv run --extra cpu ruff check .
uv run --extra cpu ruff format --check .
uv run --extra cpu mypy src
uv run --extra cpu mypy scripts/benchmark_index_memory.py
git diff --check

uv run --extra mlx pytest tests/test_mlx_backend.py
```

## What is left in the long-term plan

- WebGPU and MIGraphX still need their own promotion evidence: WebGPU on Linux/Vulkan and Windows,
  MIGraphX on the pinned AMD/ROCm matrix.
- The crossover has been measured on one Apple Silicon machine. The case it exists for — an
  accelerator that costs meaningfully more to start than CPU — is the CUDA shape, and has not been
  measured on an NVIDIA machine. The formula and the deferral are exercised by unit tests; the
  numbers they would be fed there are not yet on record.
