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
- 02:06 FOUND: promotion step 6 ms/token was all 'decide+issue' (host enqueueing ~30 CUDA copies) -> moved to a helper thread. Kernel L2 prefetch for pinned-RAM reads: in-place read speed 7.8 -> ~10.6 GB/s (Q8 all-from-RAM 3.99 -> 5.43 t/s). Warm-before-prefill: resident experts reused via D2D in prefill (Q4 1500-token prefill 5.0 -> 2.8 s). All exact vs full GPU (Q4). Methodology fix: sampled texts differ between runs (hit rate 92.6 vs 96.7 at same cache) so all further system comparisons force ONE canonical natural 4096-token output (results/overnight/bench/canonical-out-4096.txt). Queue 3 running: F1 (zc+warm+prefetch+worker), F2 (+KV q8), F0 (hot proxy v2), F3 (previous cache, forced).
- 02:18 RESULT (forced canonical 4096-token output, identical routing, hit 94.5%): NEW zero-copy tier (warm+prefetch+worker, 14 GiB cache) 125 s total = prefill 23.0 s + decode 40.2 t/s; plain llama.cpp 141 s (22.3 s + 34.4 t/s); previous cache 143 s; KV q8_0 WORSE (31.1 t/s, dropped); all-hot proxy v2: 84 s (decode ~75 t/s steady). Promotion cost 6.1 ms -> 0.01 ms/token. Remaining gap = ~17-21 misses/token x ~0.5 ms. Next: cache-aware routing (--bias) quality + speed.
