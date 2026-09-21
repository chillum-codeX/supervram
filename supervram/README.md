# SuperVRAM research prototype

SuperVRAM is an experimental SSD-backed heterogeneous tensor-storage project for Qwen3 MoE inference. It does not claim that SSD is physical VRAM.

## What is in this repository

1. A pinned llama.cpp tree at `third_party/llama.cpp` (`ce8caa6e60a03093351d6016a818720e0d46f0fb`).
2. A buildable patch that adds explicit Qwen3 MoE demand-paged expert storage:

   ```text
   --moe-expert-storage resident|mmap
   ```

   The `mmap` path keeps router/attention/dense tensors on their normal placements but maps gate/up/down expert stacks lazily and executes expert matrix multiplies on CPU. It is disabled by default.
3. A standalone policy prototype with an aligned SSD expert store, bounded asynchronous cache, eviction policies, predictors and JSONL tracing.
4. Hardware probing, GGUF expert packing, trace simulation, analytical roofline, benchmark and ablation scripts.
5. Exact integration maps and a target RTX 3090 protocol under `docs/`.

## Local validation

```bash
cd project
./entrypoint.sh                 # CPU checks only
SUPERVRAM_CUDA=1 ./entrypoint.sh  # also builds third_party/llama.cpp/build-3090 with CUDA
```

Overrides: `SUPERVRAM_PYTHON`, `CC`, `CXX`, `SUPERVRAM_JOBS`, `SUPERVRAM_CUDA_ARCH`, `SUPERVRAM_RESULTS`.

Files under `results/reproduced/` and `results/*.json` from the simulator and roofline are synthetic replay or analytical projection. Target measurements live under `results/rtx3090/` and carry `evidence_class: measured_rtx3090`.

## Core policy smoke test

```bash
PYTHONPATH=. python scripts/simulate_trace.py \
  --tokens 64 --layers 8 --experts 32 --top-k 4 \
  --cache-experts 8 --policy router-aware --predictor markov \
  --prefetch-depth 4 --output results/simulation.json
```

## Pack expert slices from a Qwen MoE GGUF

```bash
PYTHONPATH=. python scripts/pack_gguf_experts.py model.gguf experts.svram
```

The packer stores each `(layer, expert)` gate/up/down group as one aligned extent and records checksums and original tensor encodings.

## Patched llama.cpp CPU-mmap baseline

The patch is in `patches/0001-qwen3-moe-mmap-expert-storage.patch`. The working pinned tree already has it applied. Build with:

```bash
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build-supervram -G Ninja \
  -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build third_party/llama.cpp/build-supervram --target test-arg-parser test-llama-archs llama-bench
```

CUDA build for the RTX 3090 (UI assets disabled to avoid the network fetch):

```bash
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build-3090 -G Ninja \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF
cmake --build third_party/llama.cpp/build-3090 --target llama-server llama-bench llama-cli llama-perplexity test-backend-ops
```

The library API uses:

```cpp
llama_model_params params = llama_model_default_params();
params.moe_expert_storage = LLAMA_MOE_EXPERT_STORAGE_MMAP;
```

## Status of the GPU/NVMe design

A real bounded GPU expert cache is not integrated into llama.cpp yet. The exact remaining boundary is the existing selected-expert partial-copy logic in `ggml_backend_sched_compute_splits`, plus CUDA `MUL_MAT_ID` ID remapping. The standalone engine implements and tests the cache-control algorithms, but it does not substitute for a CUDA implementation.

See:

- `docs/IMPLEMENTATION_STATUS.md`
- `docs/LLAMA_CPP_CODE_MAP.md`
- `docs/RTX3090_PROTOCOL.md`
- `docs/METRICS_SCHEMA.md`
- `writer_handoff/`
