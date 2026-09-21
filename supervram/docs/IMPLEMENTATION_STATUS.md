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
- Qwen3-30B-A3B Q4_K_M: greedy 8 tokens identical across resident/mmap/cache. Resident vs cache logits hashes identical. Decode (short verify): resident 279 t/s, mmap 29 t/s, cold cache 17 t/s.
- Qwen3-30B-A3B Q8_0 (exceeds VRAM): greedy 8 tokens identical across `--cpu-moe`, mmap, cache. Logits hashes differ (CPU vs CUDA kernels), as documented.
- llama-bench, 3 reps: Q4 pp512/tg128 resident 4273 / 171 t/s; `--n-cpu-moe 24` 591 / 57; mmap 341 / 34; cache decode-only tg128 47 t/s (8 GiB, 62 slots/tensor). Q8 `--n-cpu-moe 48` / mmap about 65 pp64 / 23 tg64; cache decode-only tg64 24.6 t/s (12 GiB, 53 slots/tensor).

## Cache-size ablation (`results/rtx3090/ablations/`, `evidence_class: measured_rtx3090`)

Scope: 20 runs, **one repetition each**, `svram-verify` cold-start greedy decode of 32 tokens after a 7-token prompt, `-ub 1`. This is a cache-size x policy sweep only. It is **not** the 720-run matrix in `RTX3090_PROTOCOL.md`: no prefetch depths, no predictors, no pp512, no repetitions, so no variance estimate.

| Cache (GiB) | Q4_K_M LRU / LFU decode t/s | Q4 hit rate | Q8_0 LRU / LFU decode t/s | Q8 hit rate |
|---|---|---|---|---|
| 4  | 18.2 / 19.8 | 71.5 / 73.9 % | 10.1 / 10.1 | 64.8 / 64.4 % |
| 8  | 27.5 / 27.7 | 81.2 / 81.1 % | 15.5 / 15.5 | 76.5 / 76.5 % |
| 12 | 28.2 / 28.0 | 81.3 / 81.3 % | 19.8 / 20.1 | 81.0 / 81.2 % |
| 16 | 28.1 / 28.0 | 81.3 / 81.3 % | 20.4 / 20.6 | 81.6 / 81.6 % |
| 20 | 28.0 / 27.7 | 81.3 / 81.3 % | failed (CUDA OOM) | - |

- Hit rate plateaus at about 81 % once the cache holds every expert touched (0 evictions at >= 12 GiB on Q4). The remaining ~19 % are compulsory misses of a cold 32-token run, so this plateau says nothing about steady-state hit rate on long generations.
- LRU vs LFU differ by less than run-to-run noise would allow us to resolve with one repetition; no policy winner is claimed.
- Q8_0 at 20 GiB fails with a genuine `cudaMalloc` OOM in `ggml_backend_sched_expert_cache_layout` (20 GiB of slots plus dense/KV/compute buffers exceed 24 GiB). The failures are kept in the manifest and status files, not dropped.
- Decode is far below full residency on Q4 (28 vs 279 t/s) and, cold, stays slightly below `--cpu-moe` on Q8 even at 16 GiB (20.4 vs 21.6 t/s in the verify run). Warm `llama-bench` tg64 gave cache 24.6 vs `--cpu-moe` 23.1 t/s, a ~6 % edge from a 3-rep run. The cache is not yet a demonstrated win over `--cpu-moe`.

## Remaining gaps

- Prompt ubatches that route more distinct experts than `n_slots` return a hard error (observed as llama-bench warmup `res = -3` at `-ub 16`). Use `-ub 1` or a larger cache for decode; prompt processing is not a v1 win.
- No next-token prefetch, no GDS copy backend, no llama-server `/metrics` cache stats (stats are in the server task JSON only), and only the 20-run cache-size x policy ablation above, not the 720-run matrix (no prefetch/predictor axes, no repetitions).
- Host to device copies are from mmap/pageable memory (driver-staged). Pinned staging ring not implemented.
- CUDA graphs were disabled (`GGML_CUDA_DISABLE_GRAPHS=1`) for bring-up.
