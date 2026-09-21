# SuperVRAM implementation status

## Implemented and tested now

- Pinned llama.cpp revision `ce8caa6e60a03093351d6016a818720e0d46f0fb` and Qwen3 MoE code map.
- `--moe-expert-storage resident|mmap|cache`, `--moe-expert-cache-size`, `--moe-expert-cache-policy lru|lfu`.
- Demand-paged Qwen3 MoE expert weights (`TENSOR_READ_LAZY` / lazy mode ON) stay in host/file mappings.
- Compact GPU expert cache in `ggml_backend_sched`: per-weight device slot buffer `{ne0,ne1,n_slots}`, LRU/LFU, logical-to-slot id remap, stats API.
- `svram-verify` greedy token-id / logits-hash gate.
- Hardware probe with a real `cuFileRead` into `cudaMalloc` (compatibility-mode host bounce on this machine).
- Standalone Python/C++ policy prototype, packer, simulator and roofline (synthetic / analytical).

## Measured on this RTX 3090 (`evidence_class: measured_rtx3090`)

See `results/rtx3090/SUMMARY.json`, `results/rtx3090/hardware-probe.json`, `results/rtx3090/verify/`, `results/rtx3090/baselines/`.

- GPU: GeForce RTX 3090 24 GiB, driver 595.91.07, CUDA 12.6, compute 8.6. BAR1 256 MiB. Idle PCIe gen2 x16 (max gen3). IOMMU off. `/home` is ext4 on WD SN550 NVMe. Sequential O_DIRECT about 2.0 GB/s.
- cuFile: driver opens, reads match pread, about 1.8 GB/s, `verdict: cufile_compat_mode_host_bounce`. `nvidia-fs` absent; this is not SSD-to-VRAM DMA.
- Tiny generated Qwen3 MoE: resident/mmap/cache token ids and logits hashes identical; cache hit rate 71% on a 2-token run.
- Qwen3-30B-A3B Q4_K_M: greedy 8 tokens identical across resident/mmap/cache. Resident vs cache logits hashes identical. Decode numbers: see the ablation section (the early short `svram-verify` timings were inflated by a timer bug).
- Qwen3-30B-A3B Q8_0 (exceeds VRAM): greedy 8 tokens identical across `--cpu-moe`, mmap, cache. Logits hashes differ (CPU vs CUDA kernels), as documented.
- llama-bench, 3 reps: Q4 pp512/tg128 resident 4273 / 171 t/s; `--n-cpu-moe 24` 591 / 57; mmap 341 / 34; cache decode-only tg128 47 t/s (8 GiB, 62 slots/tensor). Q8 `--n-cpu-moe 48` / mmap about 65 pp64 / 23 tg64; cache decode-only tg64 24.6 t/s (12 GiB, 53 slots/tensor).

## Cache-size ablation (`results/rtx3090/ablations-long/`, `evidence_class: measured_rtx3090`)

Scope: `svram-verify`, greedy decode of 256 tokens after a 7-token prompt, `-ub 1`, models warm in the page cache, **3 repetitions** per configuration (mean shown; min-max in `aggregate.json`, spreads are within about 3 %). Decode only: prompt batches with more distinct experts than slots are unsupported in v1. Cache-size x policy only: no prefetch or predictor axes, so this is not the 720-run matrix in `RTX3090_PROTOCOL.md`.

Baselines (same tool, same settings): Q4_K_M full GPU 195-200 t/s, mmap 30 t/s; Q8_0 `--cpu-moe` 22.7-24.4 t/s, mmap 23.9 t/s.

| Cache (GiB) | Q4 LRU / LFU decode t/s | Q4 hit rate | Q8 LRU / LFU decode t/s | Q8 hit rate |
|---|---|---|---|---|
| 4  | 18.3 / 17.1 | 72.1 / 69.3 % | 8.8 / 8.5 | 59.0 / 57.2 % |
| 8  | 42.4 / 36.9 | 91.3 / 89.4 % | 15.1 / 14.2 | 78.3 / 76.8 % |
| 12 | 60.9 / 60.6 | 95.1 / 95.0 % | 26.4 / 26.1 | 89.2 / 89.1 % |
| 16 | 63.0 / 63.1 | 95.4 / 95.4 % | 40.2 / 39.7 | 94.0 / 93.9 % |
| 20 | 63.5 / 63.0 | 95.4 / 95.4 % | failed (CUDA OOM) | - |

- **Q8_0 (does not fit in VRAM):** with a 16 GiB cache the cache path decodes at about 40 t/s against 22.7-24.4 t/s for `--cpu-moe`, roughly 1.7x, at a 94 % hit rate. Below 12 GiB it is slower than `--cpu-moe`. `llama-bench` with a 12 GiB cache (24.6 t/s) agrees with the 12 GiB row here (26.4 t/s).
- **Q4_K_M (fits in VRAM):** the cache plateaus at about 63 t/s, a third of full-GPU speed, with 95 % hits and zero evictions from 12 GiB up. The ceiling is therefore per-step overhead (per-layer routing-id readback and synchronization), not misses. This is an inference from the plateau, not a profiled result.
- **Policy:** LRU beats LFU when the cache is small (8 GiB Q4: 42.4 vs 36.9 t/s, ranges do not overlap); they are equal once the working set fits.
- Q8_0 at 20 GiB fails with a genuine `cudaMalloc` OOM (`results/rtx3090/ablations-long/q8/OOM-20480MiB-excerpt.txt`); the failures are in the manifests.
- **Correctness (256 tokens, `-ub 1`):** the Q4 cache output equals full-GPU resident execution bit for bit (256/256 logits hashes and tokens). CPU-expert paths (Q4 mmap, Q8 `--cpu-moe`) differ from CUDA from step 0 (different kernels), and tokens diverge at step 5 (Q4) / 16 (Q8). Q8 has no GPU-resident reference because it does not fit.
- **Timer bug:** the first `svram-verify` timings (including the earlier 32-token, 1-rep sweep in `results/rtx3090/ablations/`, `results/rtx3090/verify/` and `SUMMARY.json`) stopped the clock before the GPU finished, inflating GPU-heavy modes (resident showed 279-3600 t/s). Fixed in `src/svram_verify.cpp`. Those older `decode_tps` values are superseded; their hit rates and token/hash results are unaffected. `llama-bench` numbers were never affected.

## Remaining gaps

- Prompt ubatches that route more distinct experts than `n_slots` return a hard error (observed as llama-bench warmup `res = -3` at `-ub 16`). Use `-ub 1` or a larger cache for decode; prompt processing is not a v1 win.
- No next-token prefetch, no GDS copy backend, no llama-server `/metrics` cache stats (stats are in the server task JSON only), and only the 20-run cache-size x policy ablation above, not the 720-run matrix (no prefetch/predictor axes, no repetitions).
- Host to device copies are from mmap/pageable memory (driver-staged). Pinned staging ring not implemented.
- CUDA graphs were disabled (`GGML_CUDA_DISABLE_GRAPHS=1`) for bring-up.
