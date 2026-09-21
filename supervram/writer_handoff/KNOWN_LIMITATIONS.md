# Known limitations and failure modes

1. The compact GPU cache copies host mmap pages to device slots. It is not SSD-to-VRAM DMA. cuFile on this machine is compatibility-mode host bounce.
2. Prompt ubatches can select up to `min(n_expert, n_tokens * top_k)` distinct experts. v1 fails loudly when that exceeds `n_slots` (llama-bench `-ub 16` returned decode `-3`). Decode with `-ub 1` stays within top-k.
3. Q4_K_M fits in 24 GiB, so cache cannot beat full GPU residency there; it is the correctness/overhead reference. Q8_0 (30 GiB) is the case where cache competes with `--cpu-moe` / mmap.
4. Cold-cache short runs copy several GiB and look slower than mmap; warmed llama-bench decode is the fairer comparison.
5. Linux page cache is not a hard RAM budget. mmap/cache minimize eager reads but do not cap system RAM.
6. BAR1 is 256 MiB; Resizable BAR is not in effect. Idle PCIe is gen2 x16.
7. CUDA graphs were disabled for bring-up (`GGML_CUDA_DISABLE_GRAPHS=1`).
8. No prefetch, no pinned staging ring, no llama-server cache metrics.
9. Cross-framework throughput comparisons still require matching quantization, context, batch and output length.
