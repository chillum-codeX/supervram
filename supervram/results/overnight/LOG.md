# Overnight work log (autonomous session, user asleep)

Goal (user): on 24 GB VRAM + SSD + RAM, run Qwen3-30B Q8_0 with a 32,000-token input and 4,096-token output as fast as possible,
and compare against what a 48 GB-class GPU would do. Everything measured; estimates are labeled.

Rules I follow: exactness checks (tokens + logits hashes vs full-GPU) after every exact change; lossy options are separate and flagged
with quality measurements; save and commit as I go; never fabricate numbers.

## Entries (newest last)
Tuesday 22 September 2026 01:29:57 AM IST
- 01:29 start. State: zero-copy tier stage 2 committed (ed3b66f). Benchmark to beat: plain llama.cpp 83 s (23 s prefill + 33.7 t/s) and previous cache 85 s.
- 01:30 added sampling (--temp/--top-k/--top-p/--repeat-penalty/--seed) + --ctk/--ctv to svram-verify; scripts/bench32k.sh is THE benchmark (32k in / 4096 sampled out). Queue 1 running: stock, previous cache, zero-copy stage2, and a hot-loop VRAM-resident proxy (teacher-forced repeating tokens = all experts hot = 48GB-class proxy).
- 01:42 R0 reference (32k in / 4096 sampled out): stock 141s (22.3s + 34.4 t/s); prev cache 158s; zero-copy stage2 171s (29.7 t/s, hit 92.3% on sampled text); hot-loop proxy (all experts hot) steady ~70 t/s = measured VRAM-resident-class speed at 32k context. Gap to close: misses + promotion overhead. Building warm start, promote profiling, cache-aware routing, resident-reuse prefill.
- 01:56 built + verified exact (Q4, vs full GPU): warm start, promotion with persistent slot descriptors, resident-reuse (D2D) prefill. Found+fixed a real bug: warm start ran after the prompt's own promotions were still pending and overwrote slot state under them (debug switches SVRAM_PROMOTE_CHECK / _SYNC / _PROFILE added; consistency check now 0 errors). Cache-aware routing (--bias) compiled but not yet tested. Next: benchmark warm start on the 32k workload.
