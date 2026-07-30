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

## Acceptance evidence

Measured on an Apple M4 Pro running macOS 26.5.2, against a 151-file, 2,976-chunk corpus (this
project's own `src/incode_mcp`), through an installer-shaped locked MLX environment that passed the
real probe (`resolved_providers: ["MlxMetalBackend"]`, MLX 0.32.0, device `metal`).

What the first accelerated run measured and wrote to the probe cache:

| Backend | Cold load | Calibrated `max_items` | Throughput |
| --- | ---: | ---: | ---: |
| CPU (`fastembed`, 2 threads) | 538 ms | 1 | 13,786 chars/s |
| MLX (Metal, converted weights) | 216 ms | 4 | 46,783 chars/s |

Resulting crossover: **0 characters** — MLX loads faster than the CPU fallback, so there is nothing
to earn back and it starts on the first chunk.

| Run | Backend | Batch | Embed time |
| --- | --- | ---: | ---: |
| CPU only | `cpu` | 1 | 135.4 s |
| MLX, first run (measuring) | `mlx` | 1 | 53.3 s |
| MLX, calibrated | `mlx` | 4 | 46.7 s |
| `INCODE_EMBED_CROSSOVER=10000000` | `cpu` | 1 | 137.5 s |

The deferred run reported `embedding_backend: "cpu"`, `fallback_count: 0`, and
`embedding_selection_reason: "embedded 2125973 characters, below the 10000000-character crossover
for mlx"` — CPU by design, distinguishable from CPU after a failure. Calibration adopting a batch
size of 4 took MLX from 53.3 s to 46.7 s, 2.9× the CPU baseline.

Independently, a full CPU index of the same corpus ran at 15,672 chars/s against the 13,786 chars/s
calibration measured, so the synthetic corpus is representative of this one to within about 12%.

## What measuring real hardware changed

Two defects survived a green unit suite and were found by running the thing.

**The crossover formula charged the wrong load.** It billed the accelerator its whole cold-load cost
against a CPU assumed to start instantly. Staying on CPU also spawns a worker and loads a model, and
here the converted MLX weights are memory-mapped while the CPU ONNX graph is read and prepared — so
the accelerator loads *faster* than the fallback. What has to be earned back is the difference
between the two loads. The corrected formula puts the crossover at zero on this machine, the
opposite of what the original produced. The deferral machinery still applies wherever an accelerator
does cost more to start than CPU, which is the discrete-GPU shape: CUDA loads a model into VRAM
rather than mapping it into memory the CPU model is already in.

**A deferred run was packed for the wrong backend.** The calibrated batch size is adopted from the
accelerator's record and applied to the run, so a run that stayed on CPU was handed MLX's four items
to CPU's measured one. It overran the ceiling, halved, and retried its way through the corpus: 261 s
and ten retries for work the same machine does in 137 s with none. The plan is now repacked for
whichever worker is about to run it.

The calibration ratio (3.4×) is not the end-to-end indexing ratio Phase 4B measured (1.52×), and is
not meant to be: it times embedding alone on one corpus shape, where the index gate times scan,
parse, embed, and commit over a real repository. The crossover only needs the two rates to be
comparable with each other, which they are — measured the same way, seconds apart, on the same
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
- The crossover has been measured on one Apple Silicon machine, where it comes out at zero. The case
  it exists for — an accelerator that costs meaningfully more to start than CPU — is the CUDA shape,
  and has not been measured on an NVIDIA machine. The deferral itself is exercised end-to-end here by
  forcing a threshold, and by unit tests; the numbers a real NVIDIA machine would feed it are not yet
  on record.
