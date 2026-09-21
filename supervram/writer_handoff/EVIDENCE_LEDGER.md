# Evidence ledger

## Measured in the current environment (`measured_rtx3090`)

- Host: Intel Core i9-9920X, 24 threads, 125 GiB RAM, `/home` ext4 on WD SN550 NVMe.
- GPU: GeForce RTX 3090 24 GiB, driver 595.91.07, CUDA 12.6, BAR1 256 MiB, idle PCIe gen2 x16 (max gen3).
- `cuFileRead` into `cudaMalloc` works in compatibility mode (~1.8 GB/s, matches O_DIRECT pread+H2D). `nvidia-fs` is absent; this is host bounce, not SSD-to-VRAM DMA.
- Sequential O_DIRECT pread of a 2 GiB file on `/home`: ~1991 MiB/s.
- Native C++ reader test and Python cache/store/predictor tests: passed on this host as well.
- Tiny synthetic Qwen3 MoE: resident/mmap/cache greedy token ids and logits hashes identical.
- Qwen3-30B-A3B Q4_K_M: greedy 8-token ids identical across resident/mmap/cache; resident vs cache logits hashes identical.
- Qwen3-30B-A3B Q8_0: greedy 8-token ids identical across `--cpu-moe`, mmap, cache; logits hashes differ (CPU vs CUDA).
- llama-bench (3 reps): see `results/rtx3090/SUMMARY.json`. Cache decode-only Q4 tg128 47.1 t/s vs mmap 33.8 vs full GPU 171; Q8 cache tg64 24.6 vs cpu-moe/mmap ~23.1.

- Cache-size x policy ablation, 3 reps x 256-token decode (`results/rtx3090/ablations-long/`): Q8_0 with a 16 GiB cache decodes ~40 t/s vs 22.7-24.4 for `--cpu-moe` (94 % hits); Q4_K_M cache plateaus ~63 t/s vs 195-200 full GPU (per-step overhead); LRU >= LFU when cache-constrained; Q4 cache output bit-identical to full-GPU resident. Earlier `svram-verify` decode_tps values were inflated by a timer bug and are superseded.
- Cold-SSD path (`results/rtx3090/cold/`): warm-cache numbers were RAM-backed (models 100 % page-cache resident). Cold Q8_0, 16 GiB cache, 256 tokens: mmap page faults 2.0 t/s uncapped and 0.55 t/s under a 4 GiB RAM cap; direct O_DIRECT reads into 128 MiB pinned staging 13.8-13.9 t/s under the 4 GiB cap (3 reps), 1.7 GiB peak cgroup, no expert data in the page cache. SSD ceiling measured at ~2.4 GB/s. Estimated (not measured) parity vs a native 48 GB card: ~0.14x.
- Policy headroom (offline replay of 8 real Q8_0 traces): LRU 96.3 %, LFU-decay 96.4 %, SLRU 96.3 %, Belady optimum 97.1 % (per-layer) / 97.3 % (shared) at 16 GiB, so replacement policy has < 1 point of headroom. Warm, persistent cache: 97.3-97.4 %, measured 26.3 t/s overall / 27.5 t/s steady state over 1,536 tokens cold-started under a 4 GiB RAM cap. Sharing loads across k consecutive tokens saves only ~6 % of misses at k=8; lock-step batching of independent requests raises misses per token.
- Layer-batched direct reads (`results/rtx3090/cold/*batched*`): 1,536-token cold-start run, 4 GiB RAM cap: 26.3 -> 30.5 t/s overall (30.4 in a repeat), 27.5 -> 31.8 t/s steady state; SSD wait 42.6 -> 36.1 s (2.1 GB/s); output identical. Estimated ~0.32x of a native 48 GB card (unmeasured estimate).
- Smaller experts (`results/rtx3090/quality/`, `cold/q4-*`): Q4_K_M agrees with Q8_0's next token 96.0 % of the time (teacher-forced, 3,072 tokens, 8 prompts) at +2.1 % perplexity on Q8's text (reference is Q8, not ground truth). Ground-truth check on 30.7k tokens of human-written docs prose: paired PPL ratio Q4/Q8 = 1.020 (95% CI 1.015-1.025), Q4 worse in 48/60 chunks; single corpus, no F16 reference. Cold SSD path, same 55 % cached fraction: 41.4 t/s vs 31.8 t/s (-32 % bytes read). Q4 with everything cached: 126.8 t/s (0.64x of full-GPU resident). Equal-fraction parity vs the same model's native speed is about 0.21x (Q4) vs ~0.32x (Q8, estimated native).
- Larger model (`results/rtx3090/cold/big46g-*`): a synthetic 42.9 GiB variant (experts of layers 0-23 as F16 dequantized from Q8, rest Q8_0; throughput/capacity only, not a quality result). Cold start, direct I/O, 4 GiB RAM cap, 16 GiB cache: 6.6 t/s, 90.9 % hits (38 % of experts cached), 272 MB/token from the SSD at 2.2 GB/s (81 % of decode time), 1.69 GiB peak RAM. Q8_0 (55 % cached) gave 31.8 t/s. Estimated parity vs a native card ~0.08x (unmeasured). Drives: two PCIe 3.0 x4 NVMe (WD SN550 measured 2.4 GB/s; Intel unmounted NTFS, not readable without root); no faster drive available.
- Prefetch feasibility (simulation on 3,276 real routing tokens, Q8_0, 16 GiB): an oracle gains +4 % (1 layer ahead) to +30 % (a whole token ahead) on this drive; routing-history predictors are right 1.4-2 % of the time (12-13 % for the top 0.1 % most confident guesses), so wrong reads make prefetch slower than none (0.82-0.96x at 1.0-0.5 wrong reads per useful one). Not built. Model calibrated to within 7 % of the measured speed. Simulation, not an end-to-end measurement.
- Long context (`results/rtx3090/longctx/`, Q8_0, 32,000-token prompt): stock llama.cpp (25 layers' experts on CPU) prefill 22.6 s, decode 33.7 t/s; tiered 14 GiB VRAM cache + RAM prefill 34.7 s, decode 40.5 t/s (first 2,048 tokens; later decode is inflated by greedy looping); tiered 14 GiB cache + SSD with RAM capped at 4 GiB: prefill 123 s, decode 15.2 t/s, peak RAM 2.2 GiB. Earlier 'tiered decodes 25-35 % faster' claim withdrawn (used looping output). Single runs.
- Zero-copy expert tier, stage 1 (patch 0007): experts in pinned RAM, GPU kernel reads them in place through per-expert pointers. Exact (Q4_K_M, 64 tokens: tokens and logits hashes identical to full-GPU). All-from-RAM decode 6.09 t/s (Q4) and 3.99 t/s (Q8) = 6.7 and 7.8 GB/s effective PCIe reads vs an 11.3 GB/s ceiling. Projected ~60 t/s with a VRAM cache (unmeasured until stage 2).
- Zero-copy tier stage 2 (patch 0008, background promotion with score-based admission): exact vs full GPU (Q4, 384-512 tokens). Q4 8 GiB short context: 50.5 t/s vs 42.4 t/s for the earlier cache. Target workload (Q8_0, 32,000-token prompt, 14 GiB cache): prefill 33.4 s, decode 32.7 t/s over 2,048 tokens (21.8 -> 39.9 as the cache warms), i.e. NOT better than the earlier design (40.5 t/s, 85 s total) or plain llama.cpp (33.7 t/s, 83 s total); the ~60 t/s projection was not reached (promotion step ~5.4 ms/token and PCIe contention).

## Deterministic synthetic trace replay

`results/simulation-smoke.json` validates the software policies, not LLM throughput. Do not interpret those hit rates as Qwen routing behavior.

## Analytical projection

`results/roofline-projection.json` is not measured data.

## Overnight results (2026-09-22)

Source: `results/overnight/FINAL_RESULTS.md`, `bench/SUMMARY.tsv`, `ram4g/SUMMARY.txt`. Measured on the RTX 3090 host; forced identical 4,096-token
output for all systems. Exactness gates: Q4_K_M tokens + logits hashes identical to full-GPU. Bias mode is approximate (perplexity checked, no task benchmark).
The "48 GB-class" row is an all-hot proxy, not a real card.

## Still pending

- Prefetch and double-buffered staging (a single pinned staging buffer exists), GDS device DMA, llama-server `/metrics` cache stats.
- Prompt-batch cache (v1 errors when `n_used > n_slots`).
- Full 720-run ablation matrix (prefetch/predictor axes, >= 5 reps, pp512), energy, Nsight overlap traces.
