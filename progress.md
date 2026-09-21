# SuperVRAM progress

## Status
- 2026-09-21: Workspace initialized; target RTX 3090/NVMe host unavailable in this environment. No target-hardware measurements claimed.
- 2026-09-21: Pinned llama.cpp at `ce8caa6e60a03093351d6016a818720e0d46f0fb` and mapped Qwen3 MoE loading, routing, scheduler and CUDA MMID paths.
- 2026-09-21: Added explicit disabled-by-default Qwen3 MoE mmap expert-storage patch and verified resident/mmap exact logits-checksum equality on a generated tiny model.
- 2026-09-21: Implemented aligned tensor store, bounded async cache, four eviction policies, six predictor modes, tracing, GGUF expert packer, native async reader and hardware capability probe.
- 2026-09-21: Added target benchmark harness, 720-run ablation manifest, analytical roofline, metrics schema and RTX 3090 execution protocol.
- 2026-09-21: `/home/novix/workspace/project/entrypoint.sh` completed successfully. Native CTest 1/1 and Python pytest 8/8 passed; llama parser and Qwen3 architecture tests passed.
- 2026-09-21 (this host): Ported scripts off `/home/novix`, CUDA-built llama.cpp (`build-3090`, `LLAMA_BUILD_UI=OFF`), measured RTX 3090 hardware probe + cuFile compat-mode bounce, and implemented a compact GPU expert cache in `ggml_backend_sched`.
- 2026-09-21: Correctness gate passed on this RTX 3090. Tiny synthetic Qwen3 MoE and real Qwen3-30B-A3B Q4_K_M/Q8_0 produced identical greedy token ids across resident/mmap/cache (Q8 vs `--cpu-moe`). Q4 cache logits FNV hashes matched full-GPU resident. llama-bench decode-only: Q4 cache 47.1 t/s vs mmap 33.8 vs full GPU 171; Q8 cache 24.6 vs cpu-moe/mmap ~23.1. Prompt batches with `n_used > n_slots` fail as designed.
- 2026-09-21: Cache-size x policy ablation (20 runs, 1 rep, 32-token cold decode; `results/rtx3090/ablations/`): hit rate plateaus ~81% (compulsory misses); LRU/LFU indistinguishable at n=1; Q4 18-28 t/s, Q8 10-20 t/s; Q8 at 20 GiB cache is a genuine CUDA OOM. Cache is not yet a demonstrated win over `--cpu-moe`.
- 2026-09-21: Regenerated `patches/0002-compact-gpu-expert-cache.patch` (19 files). The earlier copy was missing the `qwen3moe.cpp` lazy-read change for `cache` mode. Verified `0001` then `0002` on pristine `ce8caa6` reproduces the built tree byte for byte; `test-expert-cache` and pytest 8/8 pass.

## Active processes

None.

## Critical boundary

The compact GPU expert cache is now integrated at `ggml_backend_sched_compute_splits` with per-tensor slot buffers and id remapping. It is host to device copy (pageable mmap / cuFile compatibility mode), not SSD-to-VRAM DMA. Prompt-processing v1 requires `n_slots` at least the number of distinct routed experts or it errors. Prefetch, GDS direct I/O, and the full 720-run ablation matrix (prefetch/predictor axes, repetitions) are not done; only the 20-run cache-size x policy sweep exists.
