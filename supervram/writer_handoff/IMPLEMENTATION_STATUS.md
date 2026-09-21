# SuperVRAM implementation status

See `docs/IMPLEMENTATION_STATUS.md` for the full write-up.

The compact GPU expert cache is integrated in `ggml_backend_sched` and was correctness-gated on this RTX 3090 against real Qwen3-30B-A3B GGUFs. Remaining work is prefetch, GDS, prompt-batch fallback, and the full ablation matrix. A 20-run cache-size x policy sweep (1 rep) is in `results/rtx3090/ablations/`; it shows the cache does not yet clearly beat `--cpu-moe` on Q8_0.
