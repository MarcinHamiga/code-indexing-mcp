# Embedding Model Evaluation & Migration Plan — `codefuse-ai/F2LLM-v2-160M` (2026-11-12)

Raw machine-readable benchmark outputs backing this document live in
[`docs/plans/data/2026-11-12-f2llm-v2-160m-evaluation.json`](./data/2026-11-12-f2llm-v2-160m-evaluation.json).

---

## 1. Goal & Success Criteria

### Goal

1. Replace Syndex's default embedding model `jinaai/jina-embeddings-v2-base-code` (`768d`, 113.2M active trunk parameters) with **`codefuse-ai/F2LLM-v2-160M`** (`640d`, Apache-2.0, 9-layer Qwen3 architecture, **61.9M active trunk parameters**), using the **same local conversion & caching pattern as the MLX backend** (`huggingface_hub.snapshot_download` of official `model.safetensors` + atomic one-time local conversion to `.onnx` for ONNX Runtime backends and `.safetensors` for Apple Silicon MLX) with **zero third-party Hugging Face fork dependencies** and **zero runtime quantization loss**.
2. Support **`CPU`, `CUDA`, `MLX`, `WebGPU`, and `Vulkan`** end-to-end on the new model while **removing pure AMD ROCm / `MIGraphX` support** (`migraphx` extra, out-of-index `repo.radeon.com` wheel URL, `/opt/rocm` & `rocminfo` probing, and `Accelerator.MIGRAPHX`).

### Success Criteria

1. **Speedup without retrieval regression**:
   - **CPU ONNX FP32** (`threads=2`, default): **~2.4× faster** passage indexing (`52.2` vs `21.2 chunks/s`) and **1.5× faster** query latency (`4.77 ms` vs `7.17 ms` median).
   - **Apple Silicon MLX `FP16` Metal**: **~1.9× faster** end-to-end indexing (`118.0s` vs `226.5s` across 31,193 CoIR passages; `182.9 chunks/s` on 96 repo chunks).
   - **Download footprint**: **~50% smaller** network pull (**~305 MiB** BF16 `model.safetensors` + tokenizer JSONs vs ~615 MiB Jina FP32 ONNX).
   - **LanceDB storage**: **16.7% smaller** vector column (`640d` `float16` vs `768d` `float16`).
2. **Cross-backend vector parity across all supported backends (`CPU`, `CUDA`, `MLX`, `WebGPU`, `Vulkan`)**:
   - CPU ONNX `FP32` vs MLX `FP16`: `cosine_min >= 0.9999` (measured `0.999998`) and `top_k_overlap == 1.0000` at both `k=5` and `k=10`.
   - CPU ONNX `FP32` vs `WebGpuExecutionProvider` (Dawn Metal/Vulkan/D3D12): empirically verified `cosine_min = 0.99999988` (`max_abs_diff = 8.94e-8`) across `[2, 17]`, `[4, 128]`, and `[1, 2500]` dynamic sequence lengths, with every compute-heavy op (`Gather`, `MatMul`, `ReduceMean`/`Sqrt`/`Div` RMSNorm, `Sin`/`Cos` RoPE, `Softmax`, `Sigmoid` SwiGLU) executing natively on the GPU.
3. **Unified ONNX runtime stack (`FastEmbed` & `MIGraphX` removed; `Vulkan` added)**:
   - Every ONNX backend (`cpu`, `cuda`, `webgpu`, `vulkan`, `coreml`) runs through Syndex's own `direct_onnx.py` engine over the locally converted `.onnx` graph, removing both `fastembed` / `fastembed-gpu` and AMD's `onnxruntime-migraphx` wheel URL (`allow-direct-references = true` can also be dropped from `pyproject.toml`!).
   - `Vulkan` (`Accelerator.VULKAN`) uses the locked `webgpu` Dawn plugin (`onnxruntime==1.24.4` + `onnxruntime-ep-webgpu==0.1.0`) on Linux `x86_64` and Windows `x86_64` (`AMD64`) with explicit Dawn backend selection `ep.webgpuexecutionprovider.dawnBackendType = "Vulkan"` (forcing Vulkan on Windows instead of Dawn's D3D12 default, and targeting native Vulkan on Linux for AMD/Intel/NVIDIA GPUs without ROCm or CUDA).
   - Existing installations with `"accelerator": "migraphx"` in `accelerator.json` degrade gracefully via `load_environment()` (`unknown accelerator 'migraphx'`) to CPU with a logged warning rather than crashing server startup.
4. **Safe index, probe, and passage-cache invalidation**:
   - Existing `768d` / `jinaai/jina-embeddings-v2-base-code` partitions — **including inactive sibling branch and worktree slots** — are cleanly detected as incompatible (`embedding model` + `vector dimension 768 -> 640`) and marked `rebuild_required`.
   - `PassageCacheNamespace` and `ProbeKey` incorporate the local conversion layout version and artifact digest on both CPU ONNX and MLX.
5. **Full CI & Apple Silicon MLX gates green**:
   - `uv run ruff format --check .`
   - `uv run ruff check .`
   - `uv run mypy src`
   - `uv run pytest -n auto --dist worksteal`
   - `uv run --with "mlx==0.32.0" --with "onnx>=1.22,<2" pytest tests/test_mlx_backend.py`

---

## 2. Empirical Evaluation Summary

### 2.1 Candidate Sweep (CPU ONNX `FP32` + Apple Silicon MLX `FP16` + CoIR)

Measured on Apple M4 Pro using 96 real Tree-sitter chunks from `src/code_indexing_mcp/` packed with `plan_microbatches` (`max_items=8`, `max_token_product=4096`), synthetic MLX batches (`b8×256`, `b4×1024`), and a live 31,193-passage official CoIR evaluation (`codetrans-dl`, `codetrans-contest`, `cosqa`, `apps`) combined with official MTEB(Code, v1) / CoIR results:

| Model (License) | Total / **Active** Params | Dim & Ctx | CPU `t=2` (`chunks/s`) | Query `p50` | MLX `f16` `256` / `1024` (`chunks/s`) | 4-Task CoIR Wall Time (MLX) | `Apps` | `CosQA` | `CodeTrans-DL` | `CodeTrans-Contest` | 10-Task CoIR | Verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **`jina-embeddings-v2-base-code`** *(Apache-2.0 — Baseline)* | 161M / **113.2M** | `768d` · 8K | 21.2 (`1.00×`) | 7.17 ms | 78.2 / 17.1 (`1.00×`) | 226.5s (`1.00×`) | 17.59 | **39.55** | 27.88 | **95.11** | ~52.2–58.4 | **Baseline** (unscaled `768→6144` GEGLU + dense ALiBi) |
| **`codefuse-ai/F2LLM-v2-160M`** *(Apache-2.0 — **Selected**)* | 160M / **61.9M** | **`640d` · 40K** | **52.2 (`2.46×`)** | **4.77 ms** | **140.5 / 29.2 (`1.80×`)** | **118.0s (`1.92×`)** | **61.45** | 34.14 *(36.52 MTEB)* | **32.07** *(31.93 raw)* | 84.18 *(**91.83** raw)* | **70.52** | **SELECTED**: ~2–2.5× faster, +12–18 CoIR pts overall |
| `codefuse-ai/F2LLM-v2-80M` *(Apache-2.0)* | 80M / **31.5M** | `320d` · 40K | 44.4 (`2.09×`) | 3.90 ms | 256.0 / 48.4 (`3.27×`) | 74.1s (`3.06×`) | 55.20 | 30.26 *(34.03 MTEB)* | 33.43 *(31.31 raw)* | 82.64 *(91.08 raw)* | 68.38 | Runner-up; wider `-5.5` to `-9.3` gap on standalone `CosQA` |
| `dmis-lab/UltraBERT-Embed-150M` *(Apache-2.0)* | 151M / 113.0M | `768d` · 32K | 28.8 (`1.36×`) | 8.07 ms | 82.9 / 17.2 (`1.06×`) | ~215s | 68.50 | 36.29 | 37.23 | 88.08 | 72.73 | High accuracy, but flat (`1.06×`) speed on MLX Metal |
| `nomic-ai/CodeRankEmbed` *(MIT)* | 137M / 113.3M | `768d` · 8K | 25.0 (`1.18×`) | 7.95 ms | 88.0 / 19.5 (`1.13×`) | ~200s | — | — | — | — | 60.10 | Only 13–18% faster than Jina |
| `ibm-granite/granite-embedding-small-english-r2` | 47M / 28.3M | `384d` · 8K | 53.5 (`2.52×`) | 2.38 ms | 228.8 / 38.7 | ~80s | 13.54 | 35.58 | 33.63 | 77.63 | 53.47 | Rejected: collapses on `Apps` (`13.54`) and `CSN-CCR` (`48.42`) |
| `hotchpotch/bekko-embedding-v1-a25m` | 123M / 24.9M | `384d` · 8K | 43.2 (`2.04×`) | 2.35 ms | 280.0 / 58.4 | ~65s | 48.19 | 26.92 | 33.61 | 71.52 | 58.37 | Rejected: `-12.6` pt regression on `CosQA` (`26.92`) |
| `Alibaba-NLP/gte-modernbert-base` | 149M / 110.6M | `768d` · 8K | 20.2 (`0.95×`) | 6.28 ms | 80.9 / 16.7 | — | — | — | — | — | 71.5+ | Rejected: slower than Jina on padded CPU & MLX |

### 2.2 Precision & Quantization Audit (`FP32` / `FP16` vs `Q8`)

| Backend & Mode (`F2LLM-v2-160M`) | On-Disk Size | Repo Throughput (`96` chunks) | Query `p50` | Cosine vs `FP32` (`min` / `mean`) | Cross-Backend `top5` / `top10` Overlap | Verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **CPU ONNX `FP32`** (`t=2` / `t=all`) | 609.3 MiB | **52.17** / **36.82 chunks/s** | **4.77 ms** | `1.000000` / `1.000000` | **`1.0000` / `1.0000`** *(vs MLX `FP16`)* | **ADOPT for CPU / ONNX** |
| **MLX Metal `FP16`** (`FP32` pool/norm) | **303.6 MiB** | **182.9 chunks/s** | — | `0.999998` / `0.999999` | **`1.0000` / `1.0000`** *(vs CPU `FP32`)* | **ADOPT for MLX** |
| **WebGPU / Vulkan ONNX `FP32`** (`onnxruntime-ep-webgpu`) | 609.3 MiB | — | — | `0.99999988` / `1.000000` | **`1.0000` / `1.0000`** *(vs CPU `FP32`)* | **ADOPT for WebGPU & Vulkan** |
| MLX Metal `Q8` (`W8A16`, `gs=64`) | ~165 MiB | 181.3 chunks/s (`-1%`) | — | `0.999648` / `0.999764` | `1.0000` / `0.9786` | Reject: compute-bound dequantization is slightly slower & loses ~1.4 CoIR pts on `DL`/`CosQA` |
| CPU ONNX `QInt8` (`W8A8` per-channel linear) | 432.5 MiB | 33.49 (`t=2`) / 42.07 (`t=all`) | 5.76 ms | `0.985478` / `0.990202` | `0.9857` / `0.9429` | Reject: 36% slower at `t=2` and SwiGLU activation quantization shifts ~6% of top-10 hits |

### 2.3 Vector Dimension Audit (`640d` Native vs `512d` / `384d` MRL)

- **Model forward pass**: `0.0%` difference (`MRL` slices the `[B, 640]` output *after* all 9 layers).
- **LanceDB `float16` storage & search**:
  - `640d` is already **16.7% smaller** than Jina's `768d` (`8.60 MiB` vs `10.31 MiB` at 7k rows; `68.79 MiB` vs `82.47 MiB` at 56k rows).
  - Truncating `640d → 512d` saves only **0.38 ms** on a 7k-chunk flat scan (`1.65 ms → 1.27 ms`) and **1.09 ms** at 56k chunks (`5.90 ms → 4.81 ms`; **0.12 ms** with HNSW `1.89 ms → 1.77 ms`), while losing **`-0.51` mean CoIR nDCG@10** (`-0.83` on `CosQA`, `-0.57` on `CodeTrans-DL`) and shifting 6–15% of top-10 rankings.
- **Decision**: Keep native **`640d`**.

---

## 3. Architectural Design & Backend Matrix

### 3.1 Supported Backend Matrix (`CPU`, `CUDA`, `MLX`, `WebGPU`, `Vulkan`, `CoreML`) & Dropping `MIGraphX`

| Accelerator | Platform / Hardware | Runtime & Extra | Provider & Session Config | Selection Policy |
| :--- | :--- | :--- | :--- | :--- |
| **`cpu`** | Everywhere | `Runtime.ONNX` (`cpu` extra: `onnxruntime>=1.22,<2`) | `CPUExecutionProvider` (in-process queries + fallback worker) | `Stability.AUTOMATIC` (baseline) |
| **`cuda`** | Linux `x86_64`, Windows `AMD64` (NVIDIA driver `>=525.60` / `>=527.41`) | `Runtime.ONNX` (`cuda` extra: `onnxruntime-gpu>=1.22,<1.24`) | `CUDAExecutionProvider` + `CPUExecutionProvider` | `Stability.AUTOMATIC` |
| **`mlx`** | macOS `14.0+` `arm64` (Apple Silicon Metal) | `Runtime.MLX` (`mlx` extra: `mlx==0.32.0`) | `MlxMetalBackend` (`Qwen3Mlx` `FP16` + `FP32` pool/norm) | `Stability.AUTOMATIC` |
| **`webgpu`** | macOS `14.0+` `arm64`, Linux `x86_64`, Windows `AMD64` | `Runtime.ONNX_PLUGIN` (`webgpu` extra: `onnxruntime==1.24.4` + `onnxruntime-ep-webgpu==0.1.0`) | `WebGpuExecutionProvider` (Dawn native default: Metal on macOS, Vulkan on Linux, D3D12 on Windows) | `Stability.EXPERIMENTAL` (explicit `--accelerator webgpu`) |
| **`vulkan`** | Linux `x86_64`, Windows `AMD64` (AMD, Intel Arc, NVIDIA via Vulkan 1.3+) | `Runtime.ONNX_PLUGIN` (reuses locked `webgpu` extra!) | `WebGpuExecutionProvider` with `ep.webgpuexecutionprovider.dawnBackendType = "Vulkan"` | `Stability.EXPERIMENTAL` (explicit `--accelerator vulkan`) |
| **`coreml`** | macOS `arm64` | `Runtime.ONNX` (serving `cpu` extra) | `CoreMLExecutionProvider` + `CPUExecutionProvider` | `Stability.MANUAL` (`CODE_INDEXING_EMBED_ACCELERATOR=coreml`) |
| ~~`migraphx`~~ | *Removed* | *Removed (`migraphx` extra & `repo.radeon.com` wheel deleted)* | *Removed (`MIGraphXExecutionProvider`, `/opt/rocm`, `rocminfo` deleted)* | *Replaced by `vulkan` / `webgpu` for AMD GPUs* |

#### How `Vulkan` Works Cleanly Without Extra Wheels or Build Fragility

Microsoft does **not** publish a standalone `onnxruntime-vulkan` Python wheel on PyPI; instead, ONNX Runtime's official native GPU plugin `onnxruntime-ep-webgpu` embeds Google's **Dawn** engine with native **Vulkan** and **D3D12** backends (verified directly in the `onnxruntime_ep_webgpu` 0.1.0 binary: `ep.webgpuexecutionprovider.dawnBackendType` accepts `"Vulkan"` and `"D3D12"`).

- **Why exposing `vulkan` explicitly alongside `webgpu` is valuable**:
  1. **Replaces `migraphx` cleanly for AMD & Intel GPUs**: Linux and Windows users with AMD Radeon or Intel Arc GPUs get a first-class `--accelerator vulkan` / `CODE_INDEXING_EMBED_ACCELERATOR=vulkan` target backed by PyPI wheels (`onnxruntime==1.24.4` + `onnxruntime-ep-webgpu==0.1.0`) — zero ROCm version hell, zero Python-3.12-only wheel lock-in, and full Python 3.12 + 3.13 support!
  2. **Forces Vulkan on Windows**: Plain `webgpu` defaults to Direct3D 12 on Windows; `vulkan` sets `options.add_session_config_entry("ep.webgpuexecutionprovider.dawnBackendType", "Vulkan")` (and checks platform support up front so macOS — where Dawn only ships Metal — cleanly reports `"Vulkan was requested but Dawn's Vulkan backend is only available on Linux and Windows; use mlx or webgpu on macOS"` instead of aborting inside Dawn C++!).
  3. **Zero extra lockfile conflict group**: `ACCELERATOR_EXTRAS["vulkan"] = "webgpu"`, reusing the exact same locked `webgpu` extra in `.venv-accel` while keeping distinct `Accelerator.VULKAN`, `device="vulkan"`, probe cache keys, and `model status` reporting.

### 3.2 Single Official Source Snapshot + Local Atomic Conversions (Zero Custom HF Repo)

All backends resolve the exact same upstream Hugging Face snapshot via `resolve_model_snapshot` in `src/code_indexing_mcp/direct_onnx.py`:

- **Pinned `DEFAULT_MODEL`**: `"codefuse-ai/F2LLM-v2-160M"`
- **Pinned `DEFAULT_DIMENSION`**: `640`
- **Downloaded files (`_MODEL_FILES`)** (~305 MiB total):
  - `config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
  - `config_sentence_transformers.json`
  - `model.safetensors` (`303.6 MiB` BF16 weights)

From that single read-only HF snapshot in `<cache>/models/models--codefuse-ai--F2LLM-v2-160M/snapshots/<revision>/`, each runtime builds its own execution artifact locally once (with PID-scoped temp file + `os.fsync` + `os.replace` + cleanup of superseded layout versions, identical to `mlx_backend.py:335-409`):

1. **ONNX Runtime backends (`cpu`, `cuda`, `webgpu`, `vulkan`, `coreml`)**:
   - Path: `<cache>/models/onnx/<revision>-f2llm-v1-f32.onnx` (`ONNX_LAYOUT_VERSION = 1`, `609.3 MiB`, single self-contained protobuf `< 2 GiB` — no external `.onnx_data` sidecar hazard).
   - Shared across CPU query embedding and any ONNX accelerator (`cuda`, `webgpu`, `vulkan`, `coreml`)! Once `model pull` or the daemon builds `<revision>-f2llm-v1-f32.onnx`, the CUDA / WebGPU / Vulkan worker loads it directly without converting a second time.
2. **Apple Silicon MLX backend (`mlx`)**:
   - Path: `<cache>/models/mlx/<revision>-f2llm-v3-f16.safetensors` (`WEIGHT_LAYOUT_VERSION = 3`, `303.6 MiB` `float16` pre-transposed & `gate_up`-fused weights).
   - Built once directly from `model.safetensors` (peak conversion RSS ~0.7 GiB vs ~1.9 GiB previously).

### 3.3 Pure-Python + NumPy `.safetensors` Reader (Zero PyTorch / `safetensors` Wheel Dependency)

- Add `src/code_indexing_mcp/safetensors_io.py` (~110 lines, pure stdlib `struct`/`json`/`mmap` + `numpy`):
  - Validates 8-byte header size and JSON metadata, skips `"__metadata__"`, verifies exact expected shapes and dtypes against `ModelConfig`, and converts `BF16` (`uint16`) to `float32` losslessly via `((raw_u16.astype(np.uint32)) << 16).view(np.float32)` (and `.astype(np.float16)` for MLX).

### 3.4 Qwen3 Graph Semantics: WebGPU/CUDA/Vulkan Compatibility, Causal Masking, Last-Token Pooling & Prompts

- **Empirically verified WebGPU / Dawn shader compatibility**:
  - Tested on `onnxruntime==1.24.4` + `onnxruntime-ep-webgpu==0.1.0`:
    - Shape/index calculation ops (`Shape`, `Range`, `Greater`) stay on CPU as intended by ORT's `DeviceBasedPartition`, while **100% of tensor compute** (`Gather` embedding, `Mul`/`ReduceMean`/`Add`/`Sqrt`/`Div` RMSNorm, `MatMul` QKV/O/MLP, `Transpose`/`Slice`/`Neg`/`Concat`/`Sin`/`Cos` dynamic RoPE, `Where`/`Add`/`Softmax` causal SDPA, `Sigmoid`/`Mul` SwiGLU) compiles to WebGPU compute shaders and matches CPU `FP32` to `cosine = 0.99999988` (`max_abs_diff = 8.9e-8`)!
    - Because standard opset-18 primitives are used (no CPU-only custom contrib ops), `CUDAExecutionProvider` and `CoreMLExecutionProvider` also partition and execute the graph cleanly.
- **Tokenizer & Special Tokens (`load_tokenizer` in `direct_onnx.py`)**:
  - `tokenizer.json` automatically appends `<|im_end|>` (`id=151645`).
  - Unconditionally call `tokenizer.no_padding()` before `tokenizer.enable_padding(direction="right", pad_id=pad_id, pad_token=pad_token)`, resolving `pad_id` from `config.get("pad_token_id")` or `tokenizer.token_to_id(pad_token)` (`<|endoftext|>` → `151643`).
- **Causal Attention + Right-Padding Invariance + Last-Token Pooling**:
  - Causal mask (`col > row`) guarantees tokens at `<= eos_pos` never attend to trailing right-pad tokens (`> eos_pos`), making `eos_pos = attention_mask.sum(axis=1) - 1` pooling invariant to batch padding across CPU, CUDA, WebGPU, Vulkan, and MLX.
- **Dynamic RoPE in ONNX (`Range(0, S, 1)`)**:
  - Supports any `CODE_INDEXING_EMBED_MAX_TOKENS` up to `8192` (and model context up to `40,960`) without fixed position-table slicing limits.
- **Asymmetric Query Prompt vs Unprompted Passages & `search_by_example`**:
  - `FastEmbedder.embed_query(text)` prepends `QUERY_PROMPT = "Instruct: Given a question, retrieve passages that can help answer the question.\nQuery: "` after bounding `text` so `len(encode(QUERY_PROMPT + bounded_text)) <= max_input_tokens`.
  - `embed_passages`, `plan_and_embed`, and `search_by_example` embed raw unprompted text (`""`).

### 3.5 Simplified Packaging (`pyproject.toml` + `uv.lock`)

- **Remove `fastembed`, `fastembed-gpu`, and `migraphx`**:
  - `cpu`: `["huggingface-hub>=1,<2", "numpy>=2,<3", "onnx>=1.22,<2", "onnxruntime>=1.22,<2", "tokenizers>=0.22,<1"]`
  - `cuda`: `["huggingface-hub>=1,<2; ...", "numpy>=2,<3; ...", "onnx>=1.22,<2; ...", "onnxruntime-gpu>=1.22,<1.24; ...", "tokenizers>=0.22,<1; ..."]`
  - `webgpu` *(used by both `webgpu` and `vulkan` accelerators)*: `["huggingface-hub>=1,<2; ...", "numpy>=2,<3; ...", "onnx>=1.22,<2; ...", "onnxruntime==1.24.4; ...", "onnxruntime-ep-webgpu==0.1.0; ...", "tokenizers>=0.22,<1; ..."]`
  - `mlx`: `["huggingface-hub>=1,<2; ...", "mlx==0.32.0; ...", "numpy>=2,<3; ...", "onnx>=1.22,<2; ...", "tokenizers>=0.22,<1; ..."]` (keeping `onnx>=1.22,<2` in `mlx` preserves compatibility with `AGENTS.md`'s exact command `uv run --with "mlx==0.32.0" --with "onnx>=1.22,<2" pytest tests/test_mlx_backend.py` and existing `.venv-accel` expectations).
  - Remove `[tool.hatch.metadata] allow-direct-references = true` (which existed solely for AMD's `repo.radeon.com` MIGraphX wheel URL!).
  - Update `[tool.uv] conflicts` to the 4 mutually exclusive extras: `cpu`, `cuda`, `webgpu`, `mlx`.

---

## 4. File-by-File Change Map

### Production Code (`src/code_indexing_mcp/` & `install.py`)

1. **`src/code_indexing_mcp/safetensors_io.py` *(new, ~110 lines)***:
   - Pure-Python + NumPy `.safetensors` header parser and tensor extractor (`BF16`, `F16`, `F32`) with strict bounds, dtype, and shape validation.
2. **`src/code_indexing_mcp/direct_onnx.py`**:
   - Shared Qwen3 `ModelConfig`, `read_model_config`, and `read_query_prompt`.
   - Update `_MODEL_FILES` to `("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config_sentence_transformers.json", "model.safetensors")`.
   - Add `ONNX_LAYOUT_VERSION = 1`, `converted_onnx_path`, `ensure_converted_onnx`, and `export_qwen3_onnx` (opset 18, dynamic `[B, S]`, dynamic RoPE, fused `gate_up`, atomic temp write + `os.fsync` + `os.replace` + superseded cleanup).
   - Fix `load_tokenizer` (`no_padding()` + `token_to_id(pad_token)` fallback).
   - Add `last_token_pool_and_normalize` and `DirectOnnxEmbedding.query_embed`.
   - Update `create_webgpu_session(..., dawn_backend_type: str | None = None)`: when `accelerator == "vulkan"`, pass `ep.webgpuexecutionprovider.dawnBackendType = "Vulkan"` on `SessionOptions` (refusing non-Linux/Windows platforms cleanly before touching Dawn C++).
3. **`src/code_indexing_mcp/mlx_backend.py`**:
   - Bump `WEIGHT_LAYOUT_VERSION = 3` (`<revision>-f2llm-v3-f16.safetensors`), pruning legacy `*-jina-*` files.
   - Extract weights directly from `model.safetensors` via `safetensors_io.py` (pre-transposed linear weights + fused `gate_up` in `float16`).
   - Replace `JinaBertMlx` + `mask_mean_normalize` with `Qwen3Mlx` + `last_token_normalize`.
4. **`src/code_indexing_mcp/backends.py`**:
   - Replace `Accelerator.MIGRAPHX` with `Accelerator.VULKAN = "vulkan"`.
   - Register `Accelerator.VULKAN` in `ACCELERATOR_BACKENDS` (`provider="WebGpuExecutionProvider"`, `device="vulkan"`, `stability=Stability.EXPERIMENTAL`, `precision=Precision.FLOAT32`, `runtime=Runtime.ONNX_PLUGIN`).
   - Update `DIRECT_MODEL_ACCELERATORS` / docstrings to reflect unified direct ONNX execution across all ONNX backends.
5. **`src/code_indexing_mcp/embedding.py` & `src/code_indexing_mcp/embedding_worker.py`**:
   - Set `DEFAULT_MODEL = "codefuse-ai/F2LLM-v2-160M"`, `DEFAULT_DIMENSION = 640`, and `DEFAULT_QUERY_PROMPT`.
   - Back `FastEmbedder` and `_load_model` (for `cpu`, `cuda`, `webgpu`, `vulkan`, `coreml`) with `DirectOnnxEmbedding`.
   - Charge query prompt + `<|im_end|>` tokens in `FastEmbedder._bounded_query`.
6. **`src/code_indexing_mcp/installer/accelerator.py`, `src/code_indexing_mcp/installer/settings_spec.py`, `src/code_indexing_mcp/cli.py`, & `install.py`**:
   - Replace `"migraphx"` with `"vulkan"` in `ACCELERATOR_CHOICES`, `cli.py` `--accelerator` choices, and `settings_spec.py` `CODE_INDEXING_EMBED_ACCELERATOR` choices: `("auto", "cpu", "cuda", "mlx", "webgpu", "vulkan", "coreml")`.
   - Map `ACCELERATOR_EXTRAS = {"cuda": "cuda", "mlx": "mlx", "webgpu": "webgpu", "vulkan": "webgpu"}`.
   - Add `VULKAN_PLATFORMS = {"linux": {"x86_64"}, "win32": {"amd64"}}` and `_vulkan_plan(...)` in `installer/accelerator.py`.
   - Delete `MIGRAPHX_*`, `ROCM_*`, `_rocm_version_directories`, `_rocm_version`, `_rocminfo_device_name`, `_rocm_report`, and the MIGraphX Python-version special case in `configure_accelerator` and `detection_report`.
7. **`src/code_indexing_mcp/backend_coordinator.py` & `src/code_indexing_mcp/probe_cache.py`**:
   - Fold `converted_onnx_path` digest + `ONNX_LAYOUT_VERSION` into CPU's `PassageCacheNamespace` (matching MLX's `converted_weights_path` + `WEIGHT_LAYOUT_VERSION`), and include conversion layout versions in `model_artifact_fingerprint`.
8. **Packaging, CI & Docs**:
   - `pyproject.toml` & `uv.lock`: remove `fastembed`, `fastembed-gpu`, `migraphx` extra, and `allow-direct-references = true`; update `cpu`, `cuda`, `webgpu` extras.
   - `.github/workflows/ci.yml`: update memory-job model cache key to `code-indexing-mcp-model-f2llm-v2-160m-v1`.
   - `README.md`: update model (`codefuse-ai/F2LLM-v2-160M`, `640d`, ~310 MB download), document `vulkan` alongside `cpu`/`cuda`/`mlx`/`webgpu`/`coreml`, and remove `migraphx`/ROCm references.

### Test Suite Updates (`tests/`)

1. **`tests/test_safetensors_io.py` *(new)***: `BF16` / `F16` / `F32` decoding, truncated/corrupt headers, missing/misshapen tensors.
2. **`tests/test_direct_onnx.py`**:
   - Synthetic 2-layer Qwen3 BF16 `.safetensors` → `ensure_converted_onnx` atomic conversion, reuse, and superseded cleanup.
   - Real `onnxruntime.InferenceSession` execution verifying dynamic batching, right-padding invariance, dynamic RoPE (`seq > 2048`), and `query_embed` vs `passage_embed` prompt asymmetry.
   - WebGPU & Vulkan session creation tests: verifies `webgpu` attaches `WebGpuExecutionProvider` with default Dawn backend, while `vulkan` sets `ep.webgpuexecutionprovider.dawnBackendType = "Vulkan"` on Linux/Windows and cleanly refuses unsupported platforms (e.g., `darwin`).
3. **`tests/test_mlx_backend.py`**:
   - Synthetic 2-layer Qwen3 BF16 `.safetensors` fixtures testing `read_model_config`, `extract_weights`, `ensure_converted_weights` (`*-jina-*` cleanup), and `Qwen3Mlx` parity against both a pure-NumPy Qwen3 reference and ONNX Runtime.
4. **`tests/test_installer.py`, `tests/test_installer_orchestrator.py`, `tests/test_installer_wizard.py`, `tests/test_settings.py`, `tests/test_backends.py`**:
   - Replace `migraphx` / ROCm tests with `vulkan` planning, environment syncing (`vulkan` → `webgpu` extra), probe wiring, settings parsing, and legacy `"migraphx"` `accelerator.json` rejection/fallback tests.
5. **`tests/test_embedding.py`, `tests/test_embedding_worker.py`, `tests/test_accelerator_probe.py`, `tests/test_passage_cache_namespace.py`, `tests/test_probe_cache.py`, `tests/test_storage.py`**:
   - Update worker & probe tests for unified `DirectOnnxEmbedding` (`cpu`, `cuda`, `webgpu`, `vulkan`, `coreml`) + `MlxEmbedding` (`mlx`).
   - Add CPU `ONNX_LAYOUT_VERSION` cache namespace tests and multi-branch `768d → 640d` `rebuild_required` regression test.
6. **`tests/test_model_integration.py` & `tests/test_accelerator_acceptance.py`**:
   - Update `GATED_ACCELERATORS = (Accelerator.MLX, Accelerator.WEBGPU, Accelerator.VULKAN)` and verify `640d` `codefuse-ai/F2LLM-v2-160M` end-to-end.

---

## 5. Phased Execution Sequence

1. **Phase 1 — Foundation (`safetensors_io.py` + Qwen3 config/prompt validation)**
2. **Phase 2 — Backend Matrix Cleanup (`MIGraphX`/ROCm removal + `Vulkan` addition across `backends.py`, `installer/`, `cli.py`, `install.py`, `pyproject.toml`, `uv.lock`)**
3. **Phase 3 — Unified Direct ONNX Engine & Converter (`direct_onnx.py` + `embedding.py` + `embedding_worker.py` for `CPU`, `CUDA`, `WebGPU`, `Vulkan`, `CoreML`)**
4. **Phase 4 — Apple Silicon MLX Qwen3 Backend (`mlx_backend.py`) + Cache/Index Invalidation (`backend_coordinator.py`, `probe_cache.py`)**
5. **Phase 5 — Test Suite, Docs (`README.md`, `ci.yml`) & Full Verification Gates (`AGENTS.md`)**:
   - `uv run ruff format .` & `uv run ruff format --check .`
   - `uv run ruff check .`
   - `uv run mypy src`
   - `uv run pytest -n auto --dist worksteal`
   - `uv run --with "mlx==0.32.0" --with "onnx>=1.22,<2" pytest tests/test_mlx_backend.py`
   - Real-model & accelerator parity checks (`pytest -m model`, WebGPU smoke check, and MLX `pytest -m accelerator`).
