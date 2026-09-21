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

## Deterministic synthetic trace replay

`results/simulation-smoke.json` validates the software policies, not LLM throughput. Do not interpret those hit rates as Qwen routing behavior.

## Analytical projection

`results/roofline-projection.json` is not measured data.

## Still pending

- Prefetch, pinned staging, GDS device DMA, llama-server `/metrics` cache stats.
- Prompt-batch cache (v1 errors when `n_used > n_slots`).
- Full 720-run ablation matrix (prefetch/predictor axes, >= 5 reps, pp512), energy, Nsight overlap traces.
