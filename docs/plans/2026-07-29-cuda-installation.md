# Phase 3: Locked Installation and CUDA — what shipped

Implements Phase 3 of [the acceleration plan](2026-07-28-hardware-accelerated-indexing.md), on top
of Phase 1 (`d350a2d`) and Phase 2 (`6ca8c38`).

## What the phase had to establish

An accelerator cannot live in the serving environment. `fastembed` and `fastembed-gpu` install the
same `fastembed` module over `onnxruntime` and `onnxruntime-gpu`, which both own the `onnxruntime`
import; an environment holding both runs whichever landed last. So the phase is really three
things: split the packaging, teach the runtime to run a worker in an environment that is not its
own, and give the installer a way to build and verify that environment without ever installing
anything at request time.

## Packaging

- `fastembed` left the unconditional dependencies. `cpu` and `cuda` are now mutually exclusive
  extras, declared conflicting in `[tool.uv]` so one lockfile carries both resolutions.
- The serving environment is pinned to `cpu` everywhere it is built: `install.py`, CI, and the
  README's manual setup. uv 0.11 has no `default-extras`, so a bare `uv sync` no longer produces a
  working install — `--extra cpu` is required and is now documented as such.
- The `cuda` extra pins `onnxruntime-gpu>=1.22,<1.24` rather than accepting `fastembed-gpu`'s much
  wider range, so the combination the installer probes is the combination the release was tested
  against. Its requirements are marked for 64-bit Linux and Windows, the only platforms with
  wheels, which keeps the universal lock resolvable on macOS.

## Runtime

- `worker_launcher.py` splits *how* a worker starts from *what* it runs. `SpawnLauncher` is the
  existing `multiprocessing` path and stays the default and the CPU fallback;
  `ExternalInterpreterLauncher` starts a subprocess from another environment's interpreter, which
  dials back to an authenticated local socket and then speaks the identical command protocol.
- `multiprocessing` cannot cross that boundary even with `set_executable`: `spawn` sends the
  parent's `sys.path` and the child installs it verbatim, so the accelerator interpreter would come
  up and then import the serving environment's CPU runtime. This is the reason the external path
  exists at all and is worth keeping in mind before anyone tries to simplify it back.
- The child dials back *before* importing anything heavy, so the handshake timeout measures an
  interpreter starting rather than a model loading onto a device. A child that exits first is
  reported with its exit status instead of being waited out.
- One deadline covers connecting *and* authenticating, and a peer that fails either is dropped so
  the wait can resume. On Windows the channel is a loopback port every local process can reach, so a
  stranger must cost the attempt one connection rather than the worker it was waiting for — and must
  not be able to hold the server open by connecting and then going quiet. The connection is read
  with `os.read`, which no socket timeout reaches, so a watchdog shuts the socket down instead.
- `accelerator_env.py` reads the installer's record and re-verifies every claim in it: a vanished
  interpreter or a server upgraded past the Python the environment was built for retires the record
  with a reason rather than repairing it, since repairing it would mean installing something.
- Selection now considers what the prepared environment reported, not just what this process can
  execute — but only for the one accelerator that environment was probed for, not for every provider
  its runtime happens to ship. A backend the serving environment already offers — an explicit Core
  ML on macOS — still runs in-process; only what the prepared environment adds needs its interpreter.
- CUDA is promoted to `AUTOMATIC`. Promotion makes it eligible, not present: `auto` still resolves
  to CPU wherever no record was prepared.

## Installer

- `--accelerator auto|cpu|cuda|webgpu|migraphx|coreml`, defaulting to `auto`.
- Detection is OS/architecture, then `nvidia-smi`, then a pinned driver floor (525.60 on Linux,
  527.41 on Windows, the CUDA 12.x minimum). A driver below it is reported and left alone.
- The accelerator environment is built from empty through `UV_PROJECT_ENVIRONMENT` +
  `uv sync --locked --extra cuda`, with `--python` matched to the serving interpreter, because both
  ends of the worker channel speak one Python's connection protocol. Building over an existing
  environment is never done: leftovers from another extra would resolve ONNX Runtime to whichever
  distribution landed last.
- A later run reuses what is there when the record still describes this exact combination —
  accelerator, driver, and the server's Python — and its interpreter is still present. Anything that
  moved puts the full build and probe back. Re-downloading CUDA and re-probing the device on every
  update would be a lot of work to arrive back where the last run already was.
- `code_indexing_mcp.accelerator_probe` runs a real inference *in that environment* through the same
  `_load_model` path the worker uses, and validates the vectors the same way. Only then is the
  record written. Detection nominates; the probe confirms. It is bounded: a cold probe downloads the
  model first, but a device that wedges initialising wedges forever, and the output is captured, so
  an unbounded wait would be indistinguishable from an installer that had hung.
- Every failure — detection, build, probe, even an unresolvable data directory — falls back to CPU
  with the reason attached and leaves nothing behind that the server could pick up. An installation
  never fails because acceleration was unavailable. Falling back to CPU also reclaims the
  environment: once no record points at it, it is gigabytes of dead weight.

## Deviations from the plan text

- **`webgpu` and `migraphx` extras were not defined.** FastEmbed cannot configure either provider;
  Phase 4 is what adds the direct ONNX backend that can. Defining the extras now would mean
  shipping an environment that installs a runtime nothing is able to select, and `onnxruntime-webgpu`
  and `onnxruntime-migraphx` conflict with the `onnxruntime` that `fastembed` itself pulls in. The
  extras belong in the phase that lands their backends; the installer names both accelerators today
  and reports that no locked installation exists for them yet.
- **The CUDA release gates are not verified here.** Cosine similarity ≥ 0.999 against CPU, ≥ 99%
  top-k overlap, and ≥ 1.25× end-to-end on a 1,000-chunk corpus all need an NVIDIA runner, which
  this machine is not. The machinery they gate is in place and the probe enforces the correctness
  floor (right width, finite, normalised, and the requested provider actually resolved) on every
  installation. The measured gates and the hardware CI matrix remain open work — see below.

## Still open

- A CUDA CI runner, the golden-vector comparison against CPU, and the throughput gate.
- Small-job routing: "incremental jobs stay on CPU when accelerator startup would make them
  slower" is Phase 5's crossover threshold, and nothing measures it yet.
- Batch calibration still reports `default`; `PassageBackendSession` is handed
  `calibrated_batch_size=0` deliberately so `model status` cannot claim a measurement that never
  ran.
