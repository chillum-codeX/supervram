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

## Cold-SSD path: is the SSD really the backing tier? (`results/rtx3090/cold/`, `evidence_class: measured_rtx3090`)

Why this exists: the ablation above ran with both models fully resident in the Linux page cache (`fincore`: 30.3 GiB and 17.3 GiB resident), so it measured GPU + RAM, not GPU + SSD. ElasticVRAM notes sec. 35 says that must not be presented as SSD-backed. `scripts/cold_run.py` evicts the model file from the page cache (`posix_fadvise DONTNEED`), optionally caps RAM with a `systemd-run --user` cgroup (`MemoryMax`, `MemorySwapMax=0`), and records SSD bytes read (`/sys/block/nvme0n1/stat`), peak RAM and page-cache residency afterwards.

Q8_0 (30.3 GiB, does not fit VRAM), 16 GiB GPU cache, LRU, `-ub 1`, 256 tokens, cold start:

| Expert source | RAM cap | Decode t/s | SSD read | Expert data in RAM | Peak cgroup |
|---|---|---|---|---|---|
| mmap page faults (patch 0001/0002) | none | 2.0 (3.5 second half) | 22 GiB at ~0.12 GB/s | 22 GiB page cache | 23.6 GiB |
| mmap page faults | 4 GiB | 0.55 (only 32 tokens, cold-start dominated) | 14 GiB at ~0.12 GB/s | capped at 3.7 GiB | 4.0 GiB |
| **O_DIRECT + pinned staging (patch 0003, `SVRAM_DIRECT_IO=1`)** | 4 GiB | **13.8 / 13.9 / 13.9 (3 reps; second half 14.5)** | 29.7 GiB, none cached | 0 (1.3 GiB page cache is the dense non-expert weights, loaded once) | 1.7 GiB |

- The warm-page-cache Q8 number (about 40 t/s) is a GPU + RAM result. The honest GPU + SSD number on this machine today is about 14 t/s.
- Direct I/O correctness: Q4 cache with direct reads matches full-GPU resident bit for bit (64/64 tokens and logits hashes).
- Time is SSD-bound: 16.7 s of the 18.5 s decode is spent waiting for reads (`io_ms`). It reads about 119 MB per token (94 % hit rate, ~23 missed experts of ~5 MB), matching the arithmetic estimate.
- SSD ceiling: `scripts/ssd_expert_read_bench.py` measures the WD SN550 at about 2.4 GB/s for expert-sized (1.6 MB) O_DIRECT reads at queue depth >= 8 (1.6 GB/s at QD1). Bandwidth-only ceiling for Q8_0: about 21 t/s at 94 % hits, 42 t/s at 97 %, 62 t/s at 98 %, 125 t/s at 99 %. The direct path reaches about 1.8 GB/s during waits, roughly 70 % of that ceiling.
- Estimated parity (not measured): a native 48 GB card at 3090-class bandwidth would decode Q8_0 at roughly 100 t/s (about twice Q4's 195-200 t/s). 14 t/s is therefore about 0.14x, below the notes' 0.5x "basic viability" line. Reaching 0.5x needs a hit rate near 97.5 % or better, fewer bytes per expert, or more SSD bandwidth.

## Cache policy headroom and steady state (`results/rtx3090/traces/`, `results/rtx3090/cold/*long1536*`)

Method: `SVRAM_TRACE=<file>` (patch 0003) records the distinct experts each layer picks for every token. Eight varied prompts (code, science, story, translation, math, history, engineering, literature) x 384 tokens on Q8_0 (3,276 tokens, 48 layers) were replayed offline by `scripts/analyze_policies.py`, including Belady's offline-optimal policy as the upper bound. Hit rate is per distinct expert per (token, layer), like the runtime stats; per-layer pools of 71 slots = 16 GiB.

| Policy (71 slots/layer, each prompt from a cold cache) | Hit rate |
|---|---|
| LRU (current) | 96.28 % |
| LFU (current) / LFU with decay / SLRU | 96.08 / 96.36 / 96.29 % |
| **Optimal (Belady), per-layer pools** | **97.10 %** |
| Optimal, one shared pool across layers | 97.31 % |

- **Replacement policy is already within about 0.8 points of optimal.** Even a perfect policy cuts misses by only about 25 %. Smarter eviction cannot reach the ~97.5-98 % hit rate that 0.5x parity needs on this SSD. (A "pinned hot experts" variant looked better at 89 slots but gets its hot set loaded for free, so that comparison is not fair and is not claimed.)
- **A persistent cache matters more than the policy.** Replaying all eight prompts back to back with a warm cache gives 97.3 % (LRU). A real long cold-start run agrees: 1,536 tokens, 4 GiB RAM cap, direct I/O: 97.4 % hits and **26.3 t/s overall, 27.5 t/s from token 256 on** (first 256 tokens: 21.7 t/s). The 13.8 t/s of the short cold runs is mostly filling an empty cache.
- **Prompt matters:** decode speed on the eight prompts ranged 15-48 t/s (hit rate 94-97.7 %) with direct I/O; the single-prompt figures elsewhere in this document are not typical.
- **Sharing loads across tokens does not help.** Verifying k consecutive tokens together (best case of speculative decoding, all accepted) reduces missed experts per token only from 10.2 to 9.6 at k=8 (-6 %). Running k independent requests in lock-step is worse per token (23 misses/token at k=4, 36 at k=8, hit rate down to 87 %) because they evict each other's experts. Trace-based estimate, LRU, not measured end to end.
- **Time split in steady state (1,536-token run):** 42.6 s of 58.4 s decode is SSD wait (73 %), the other ~10 ms/token is per-layer synchronization and compute. Overlapping reads with compute could therefore gain up to about 1.35x here (the earlier "at most ~10 %" applied only to the cold 256-token runs).
- **SSD bandwidth is the binding limit.** At the steady-state 97.3 % hit rate a token needs about 51 MB from the SSD. That is at most 47 t/s at this drive's 2.4 GB/s, about 68 t/s at 3.5 GB/s and about 137 t/s at 7 GB/s (bandwidth-only arithmetic; a native-card estimate of ~100 t/s caps the useful range). The steady-state 27.5 t/s is about 0.27x of that unmeasured native estimate.

## Layer-batched reads and one sync per layer (patch 0003, `results/rtx3090/cold/*long1536-batched*`)

Finding: a cached `MUL_MAT_ID` is forced to be the first node of its own scheduler split, so gate, up and down of a layer are three consecutive splits. The first version therefore read each weight's missed experts separately (queue depth about 1, about 1 ms per 1.6 MB read) and synchronized the GPU stream once per weight. The scheduler now looks ahead across the following splits that share the same routing ids, plans all three weights together, issues one batch of O_DIRECT reads and synchronizes once per layer.

Same 1,536-token cold-start run as above (Q8_0, 16 GiB cache, direct I/O, 4 GiB RAM cap):

| | Overall t/s | Steady state (from token 256) | SSD wait | SSD read rate while waiting |
|---|---|---|---|---|
| per-weight reads | 26.3 | 27.5 | 42.6 s | 1.8 GB/s |
| **layer-batched reads** | **30.5 / 30.4 (2 reps)** | **31.7-31.9** | **36.1 s** | **2.1 GB/s** |

- Output is unchanged: tokens and logits hashes are identical to the per-weight version; Q4 with direct reads and with mmap still matches full-GPU execution bit for bit; `test-expert-cache` passes.
- 72 % of decode time is now SSD wait (36.1 of 50.4 s); the remaining ~9 ms/token is GPU compute and launch overhead. The read rate is within about 12 % of the drive's measured 2.4 GB/s ceiling.
- Estimated parity (native ~100 t/s is an unmeasured estimate): 31.8 t/s is about 0.32x.
- Overlapping the remaining reads with compute would need a prediction of the missed experts one layer ahead. Because the SSD wait (about 23 ms/token) exceeds the compute (about 9 ms/token), the best case is about 41 t/s, and only with a perfect predictor.

## Remaining gaps

- Prompt ubatches that route more distinct experts than `n_slots` return a hard error (observed as llama-bench warmup `res = -3` at `-ub 16`). Use `-ub 1` or a larger cache for decode; prompt processing is not a v1 win.
- No next-token prefetch, no GDS copy backend, no llama-server `/metrics` cache stats (stats are in the server task JSON only), and only the 20-run cache-size x policy ablation above, not the 720-run matrix (no prefetch/predictor axes, no repetitions).
- Direct I/O (`SVRAM_DIRECT_IO=1`, Linux only, env-var controlled, no CLI flag yet) uses one pinned staging buffer with a synchronize before each reuse: no double buffering and no prefetch, so reads never overlap compute (reads of one layer are batched, see above). The default path is still mmap.
- CUDA graphs were disabled (`GGML_CUDA_DISABLE_GRAPHS=1`) for bring-up.
