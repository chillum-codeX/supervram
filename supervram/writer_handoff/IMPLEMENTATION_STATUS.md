# SuperVRAM implementation status

See `docs/IMPLEMENTATION_STATUS.md` for the full write-up.

The compact GPU expert cache is integrated in `ggml_backend_sched` and was correctness-gated on this RTX 3090 against real Qwen3-30B-A3B GGUFs. Remaining work is prefetch, GDS, prompt-batch fallback, and the full ablation matrix. The repeated 256-token cache-size sweep in `results/rtx3090/ablations-long/` shows the cache beating `--cpu-moe` ~1.7x on Q8_0 at a 16 GiB cache (decode only).
