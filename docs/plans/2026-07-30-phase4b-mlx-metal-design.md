# Phase 4B: MLX Metal Backend Design

## Why this slice exists

The long-term plan made the Metal path conditional: "If WebGPU fails the Apple Silicon promotion
gates, implement an MLX backend as the designated Metal fallback; do not add PyTorch/MPS unless MLX
also fails model-parity requirements."

Phase 4A measured that gate on an Apple M4 Pro. A forced 1,000-chunk index took 19.498 s on CPU and
17.555 s on WebGPU — 1.11×, against a 1.25× requirement. WebGPU therefore stays experimental on
Apple Silicon, and this slice implements the named contingency. Nothing here promotes any backend:
it makes MLX installable, executable, and measurable through the machinery Phase 2-4A already
built, and records what the measurement said.

## Scope

- A project-owned MLX passage model that reproduces the CPU model's vectors on Metal.
- A locked `mlx` installer extra and Apple Silicon detection for `--accelerator mlx`.
- Worker routing, probe verification, and acceptance gates for a backend whose runtime is not ONNX
  Runtime at all.

Out of scope: automatic selection of MLX (it stays `EXPERIMENTAL` until its gates pass), any change
to the default model, tokenizer, pooling, normalization, vector dimension, or stored text, and any
PyTorch/MPS work — which the long-term plan only allows if MLX fails model parity.

## Chosen approach

### Weights come out of the ONNX artifact this project already ships

MLX cannot execute ONNX. Something has to supply the same parameters in a form MLX can hold, and
there were two candidates:

1. Download `model.safetensors` from the same Hugging Face repository.
2. Extract the initializers from the `onnx/model.onnx` artifact FastEmbed already uses.

This slice extracts from the ONNX artifact. The repository's Torch weights are published as
float16 (`torch_dtype: float16` in `config.json`) while every initializer in the ONNX artifact is
float32, so route 1 would compare a float16 model against a float32 reference and call the
difference "backend parity". Route 2 embeds *the same numbers* the CPU path embeds, needs no second
download, works offline from the snapshot the installer already resolved, and invalidates naturally
when the model revision changes.

Extraction is a one-time conversion into a float32 safetensors file under
`<cache>/models/mlx/<revision>-v<layout>.safetensors`. `mx.load` memory-maps it, so only the first
load pays for parsing a 640 MB protobuf, and the revision in the filename retires the conversion
when the snapshot moves. The conversion frees each initializer's `raw_data` as it goes, which keeps
its peak near one copy of the weights rather than two.

### The architecture is reproduced from the graph, not from a library

`jinaai/jina-embeddings-v2-base-code` is a JinaBERT v2: ALiBi instead of position embeddings,
GEGLU feed-forward, and post-norm on the query and key projections. The MLX implementation mirrors
the exported graph node for node:

```
token_type_ids = 0                        (the artifact has no token_type_ids input)
x  = LayerNorm(word[ids] + token_type[0], eps=1e-12)
bias = slopes * |i - j| + (1 - mask) * MASK_FILL
per layer:
  q = LayerNorm_q(x @ Wq + bq);  k = LayerNorm_k(x @ Wk + bk);  v = x @ Wv + bv
  c = softmax(q kᵀ / sqrt(head_dim) + bias) v
  h = x + LayerNorm(c @ Wo + bo + x)
  m = LayerNorm_1(h)
  up, gate = (m @ W_up).split(intermediate_size, axis=-1)
  x = LayerNorm_2(m + (up * gelu(gate)) @ W_down + b_down)
pooled = l2_normalize(mask_mean(x))
```

Two deliberate departures, both numerically inert:

- The graph adds `-3.4028235e38` to masked scores; adding a negative ALiBi bias to that overflows
  to `-inf` and risks `NaN` inside a fused attention kernel. This uses `-1e9`, which underflows to
  exactly zero through `exp` in float32 and cannot overflow.
- `mx.fast.scaled_dot_product_attention` scales the query before the product rather than dividing
  the scores after it.

The ALiBi slopes are read out of the graph rather than recomputed from the head count, so a model
whose slopes were not the textbook powers of two would be reproduced rather than silently replaced.

`config.json` is checked against the five architectural facts this implementation reproduces
(`model_type`, `position_embedding_type`, `feed_forward_type`, `hidden_act`, `emb_pooler`) and the
extraction refuses an artifact whose initializer names or shapes have moved. Guessing at a changed
architecture is how a backend produces plausible vectors that do not retrieve.

### A non-ONNX backend inside an ONNX-shaped contract

`BackendDescriptor.provider` has meant "ONNX execution provider" until now, and three places
assumed it: `providers` appends `CPUExecutionProvider` behind every accelerator, the probe requires
the provider to appear in `onnxruntime.get_available_providers()`, and both the probe and
`PassageBackendSession` verify that the loaded session kept the provider it was given.

Rather than special-case MLX at each site, descriptors now carry a `Runtime`
(`onnxruntime`, `onnxruntime-plugin`, `mlx`) and the existing accelerator-name special cases become
properties of it:

- `providers` no longer appends a CPU execution provider behind an MLX backend. There is no graph
  partitioning to fall back into; MLX either runs the layer or the load fails.
- The probe requires a pre-registered provider only for `Runtime.ONNX`, which is what the WebGPU
  plugin was already exempted from by name.
- `MlxMetalBackend` is what the MLX model reports as its resolved provider, so the record the
  installer writes, `auto`/explicit selection, `_runs_externally`, worker verification, and
  `model status` all keep working unchanged on a string that is honestly not an ONNX EP.

Everything else is reused as-is: the disposable external worker, the IPC protocol, packed float32
vectors, the memory ceiling, batch retries, run-level CPU degradation, strict mode, the probe cache,
and worker teardown.

## Packaging and compatibility

The `mlx` extra is mutually exclusive with `cpu`, `cuda`, `webgpu`, and `migraphx`:

- `mlx==0.32.0`, which pulls `mlx-metal` on Darwin
- `onnx>=1.22,<2` for the one-time weight conversion
- the direct-model dependencies (`huggingface-hub`, `numpy`, `tokenizers`)

Published wheels start at `macosx_14_0_arm64`, and MLX's Metal backend needs Apple Silicon, so the
installer nominates `mlx` only on macOS 14+ arm64. MLX also publishes CPU-only Linux and Windows
wheels; nominating those would install a "Metal" backend with no Metal in it, so the marker excludes
them. An unsupported `--accelerator mlx` request reports why and installs CPU — unlike MIGraphX,
which falls through to WebGPU, because a request for Metal on a machine that has no Metal is not a
request for Vulkan. `auto` still considers only CUDA.

The macOS version is recorded as the environment's `driver_version`, which puts it in the probe
cache key: an OS upgrade under a prepared environment retires the verdict recorded before it.

## Failure handling

- A config or artifact that does not match the reproduced architecture fails the load, which is the
  existing model-unavailable failure in the worker.
- A machine without a Metal device fails at model construction, which fails the installer probe and
  removes the half-built environment.
- A missing weight conversion is rebuilt from the snapshot already on disk. That is local file work,
  not a package installation, so it stays permitted in a worker; it is logged because it is slow.
- Anything that fails later follows the existing run-level CPU fallback, and strict mode forbids it.

## Testing and promotion

The MLX forward pass cannot be exercised from the serving environment, which installs the `cpu`
extra and therefore no MLX. Coverage is split accordingly:

- In the serving environment: worker routing to the MLX model without importing FastEmbed, probe
  behaviour for a non-ONNX runtime, the registry and selection entries, and installer nomination
  and rejection across the platform matrix.
- In the `mlx` extra environment (`uv run --extra mlx pytest tests/test_mlx_backend.py`): weight
  extraction from a synthesized ONNX graph, conversion caching and atomic replacement, config and
  artifact guards, and the whole forward pass against an independent NumPy implementation of the
  same graph.
- Across environments, opt-in on real hardware (`-m accelerator`): vector and ranking parity against
  the CPU model and a forced 1,000-chunk index against the 1.25× gate.

Promotion to `automatic` requires the same gates CUDA passed. This slice measures them and records
the result; it does not decide the outcome in advance.
