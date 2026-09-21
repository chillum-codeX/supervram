# SuperVRAM implementation status

## Implemented and tested now

- Pinned llama.cpp revision and exact Qwen3 MoE code map.
- A buildable llama.cpp patch with explicit `--moe-expert-storage mmap` for demand-paged Qwen3 MoE expert tensors.
- Disabled-by-default semantics; normal resident behavior is unchanged unless the option is selected.
- Architecture checks and file-backed/mmap requirements to avoid silent high-memory fallback.
- Standalone aligned expert tensor store and manifest.
- Bounded asynchronous expert-cache policy engine.
- LRU, LFU, weighted reload-aware and router-probability-aware eviction.
- No-prediction, history, router-probability, Markov, lightweight-online and oracle interfaces.
- JSONL tracing, cache and prediction metrics.
- Portable `pread`, mmap and O_DIRECT attempt/fallback in Python.
- Native C++ capability probe and asynchronous reader. cuFile/io_uring are capability slots; this CPU host cannot compile or validate their device paths.
- Hardware probe, deterministic trace replay, analytical transfer roofline, 720-run ablation plan and target-host llama-server harness.

## Important boundary

The integrated llama.cpp patch is an SSD-backed CPU expert baseline. It reuses the existing lazy GGUF mapping and CPU `MUL_MAT_ID` path. It does not implement bounded GPU expert slots or SSD-to-VRAM DMA.

The standalone cache/predictor engine validates algorithms and interfaces but is not wired into llama.cpp CUDA execution. A true bounded GPU cache requires fixed compact slots, logical-to-slot ID remapping, router-ready/transfer-ready events and backend-specific transfer streams at `ggml_backend_sched_compute_splits` / CUDA `MUL_MAT_ID`.

## Not validated here

- RTX 3090, CUDA, NVMe, GDS/cuFile, BAR1 or Resizable BAR behavior.
- Real Qwen3 MoE GGUF inference with the new mmap option.
- SSD throughput, tokens/s, energy, VRAM savings or prediction accuracy on real router traces.
- io_uring and cuFile data paths.
- Correctness of a compact GPU cache, because that cache is not yet integrated.

These items are explicitly pending the target machine and must not be inferred from synthetic results.
