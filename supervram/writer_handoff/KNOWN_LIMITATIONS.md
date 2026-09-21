# Known limitations and failure modes

1. The compact GPU cache copies host mmap pages to device slots. It is not SSD-to-VRAM DMA. cuFile on this machine is compatibility-mode host bounce.
2. Prompt ubatches can select up to `min(n_expert, n_tokens * top_k)` distinct experts. v1 fails loudly when that exceeds `n_slots` (llama-bench `-ub 16` returned decode `-3`). Decode with `-ub 1` stays within top-k.
3. Q4_K_M fits in 24 GiB, so cache cannot beat full GPU residency there; it is the correctness/overhead reference. Q8_0 (30 GiB) is the case where cache competes with `--cpu-moe` / mmap.
4. Cold-cache short runs copy several GiB and look slower than mmap; warmed llama-bench decode is the fairer comparison.
5. Linux page cache is not a hard RAM budget. mmap/cache minimize eager reads but do not cap system RAM.
6. BAR1 is 256 MiB; Resizable BAR is not in effect. Idle PCIe is gen2 x16.
7. CUDA graphs were disabled for bring-up (`GGML_CUDA_DISABLE_GRAPHS=1`).
8. No prefetch, no double-buffered staging ring (one pinned staging buffer, synchronized before reuse), no llama-server cache metrics.
9. Cross-framework throughput comparisons still require matching quantization, context, batch and output length.
10. Early `svram-verify` decode timings (before the `llama_synchronize` fix) were inflated; only the 256-token sweep in `results/rtx3090/ablations-long/` and `llama-bench` numbers are valid.
11. The ablation covers decode at `-ub 1` from a warm page cache with 3 repetitions and one prompt; long contexts and prompt processing are not covered.
12. Direct I/O is Linux-only (`--moe-expert-direct-io`, `--moe-expert-io-threads`, `--moe-expert-staging-mib`; the `SVRAM_*` environment variables are a fallback). `llama-bench` does not have the flags yet. The 13.9 t/s cold result is one model, one prompt, decode only, and about 0.14x of an estimated (unmeasured) native 48 GB card.
13. The dense non-expert weights (about 1.3 GiB for Q8_0) still pass through the page cache once at load time; expert data does not.
