# Known limitations and failure modes

1. The integrated llama.cpp patch provides a demand-paged CPU expert baseline, not a true bounded GPU cache.
2. The standalone cache engine is policy-accurate but not connected to ggml CUDA kernels.
3. Current llama.cpp stores each layer projection as one 3-D tensor. A compact GPU cache needs new fixed slots and logical-to-slot ID remapping; partial copies alone still reserve full tensor shape.
4. Router IDs become known inside the graph. Host-driven storage submission introduces a synchronization boundary unless a backend-specific asynchronous protocol is added.
5. Prompt batches can select up to `min(n_expert, n_tokens * top_k)` distinct experts, potentially defeating a small cache. Decode-only and prompt-processing modes need separate policies or fallback.
6. Linux page cache is not a hard RAM budget. The mmap baseline minimizes eager reads but cannot guarantee bounded system-RAM residency.
7. O_DIRECT requires aligned offsets, sizes and user buffers and may not work on all filesystems. The Python path falls back; strict experimental runs should reject fallback.
8. Seeing `libcufile` is not proof of direct SSD-to-device DMA. Compatibility mode, topology, filesystem, driver, CUDA and device restrictions must be recorded.
9. RTX 3090 behavior, consumer-GPU GDS support and BAR1/topology impacts are unresolved pending target hardware.
10. Real prediction quality is unknown until router traces are collected from Qwen3 MoE prompts.
11. SSD endurance and thermal throttling may distort sustained results.
12. CUDA graph capture may be incompatible with host decisions, callbacks, dynamic addresses or per-layer synchronization.
13. Quantized expert slices must preserve GGML block encoding and alignment. The packer preserves raw encoded bytes but no CUDA consumer is implemented.
14. Cross-framework throughput comparisons require matching quantization, context, batch, output length and quality; otherwise they are not fair.
