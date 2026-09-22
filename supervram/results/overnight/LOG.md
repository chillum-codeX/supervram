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
- 02:24 CACHE-AWARE ROUTING (--bias B; selection-only bias toward VRAM-resident experts): Q8, 14 GiB cache, short ctx, 2048 forced tokens of human docs prose: bias 0.02 -> hit rate 94.6% -> 98.9%, decode 42.1 -> 70.2 t/s, paired perplexity change x0.998 (+/-0.5%, 0.3 SE from zero), next-token accuracy 52.1 -> 51.9%; bias 0.05 costs +2.1% ppl (2.7 SE). Stress tests queued: 32k context, code + python-docs domains, generation self-consistency, and the 32k speed benchmark.
- 02:39 bias quality: teacher-forced ppl unchanged up to bias 0.02 in 4 domains incl. 32k context (docs 32k: 0.02 -> ppl x1.0002, hit 93.8->98.6%, decode 31.3->50.2 t/s; 0.03 costs +0.9%). BUT free generation gets more repetitive with bias (1 prompt: repeated 8-grams 3.1% -> 5.6% (0.02) -> 19.4% (0.05); distinct-2 84.7 -> 75.5 -> 67.9). So bias is an approximate mode with a real diversity cost; multi-domain generation test queued (queue5) to pick a safe strength. Exact mode (bias 0) is the primary result.
- 02:46 FULL BENCHMARK with cache-aware routing (32k in / 4096 forced out, identical text): bias 0.01 -> 96 s (23.0 s + 55.9 t/s, hit 98.4%); 0.02 -> 88 s (62.8 t/s, hit 99.2%); 0.03 -> 86 s (65.6 t/s, hit 99.5%); exact (bias 0) 125 s; plain llama.cpp 141 s; all-hot proxy 84 s. Built multiplicative bias (--bias-mul) and bias for the classic slot cache (SSD design). Generation-diversity test running (queue5).
- 02:59 generation-diversity test, 3 domains x bias {0,0.005,0.01,0.02}, 768 tokens, 1 seed (noisy): no detectable degradation up to bias 0.01; at 0.02 docs 4.1->7.2% repeated 8-grams and pydoc 0->2.8% (code unchanged). Recommended safe bias 0.01 (96 s), aggressive 0.02 (88 s). Now testing multiplicative bias (relative margin).
- 03:03 multiplicative bias (32k docs, teacher-forced): mul 0.25 hit 96.2% ppl x0.9998; 0.5 97.4% x1.0067; 1.0 98.4% x1.0036 (decode 48.1 t/s) - about the same frontier as additive (0.02: 98.6%, x1.0002); keep additive as recommended. Queued: RAM-poor SSD test with bias (queue6), baseline fairness (stock n-cpu-moe 23/27) + repeat runs (queue7).
- 03:15 RAM-POOR (4 GB RAM cap, peak 2.2 GiB, SSD, classic slot cache + direct I/O; 32k prompt + 2048 forced tokens): bias 0: prefill 122.8 s + 14.5 t/s = 264 s; bias 0.01: 29.5 t/s = 192 s; bias 0.02: 38.7 t/s = 176 s (hit 94.9 -> 98.2 -> 99.0%). Prefill (SSD streaming, 8 passes over 30 GiB) is now the bottleneck; testing bigger prompt batch / smaller cache.
- 03:25 repeat runs (variance ~1-2%): exact 123 s (run1 125), bias 0.01 95 s (96), bias 0.02 87 s (88). Baseline fairness: stock n-cpu-moe 27 = 147 s (33.1 t/s); 23 OOMs; best fitting split so far 25 (141 s); 24 queued.
- 04:10 classic-cache (SSD design) warm start + D2D prefill reuse built and exact vs full GPU (Q4, 1500-token prompt, direct and mmap: tokens+hashes identical; warm_start_ms>0 confirmed after fixing a build break that had hidden stale-binary results). RAM-POOR 4 GB cap, 32k in / 2048 forced out, bias 0.02, cache 14 GiB, ub 4096: prefill 122.8 -> 80.2 s, decode 38.7 -> 42.6 t/s (no warm-up dip), total 176 -> 128 s. Queue10 running more configs (ub 8192/11 GiB; bias 0; bias 0.01).
- 05:00 RAM-poor full workload measured: 4 GB RAM cap + SSD, bias 0.02, 32k in / 4096 forced out: prefill 80.1 s + 46.9 t/s = 167 s total (plain llama.cpp with plenty of RAM: 142 s).
- 05:20 spot-checked the two caveats against real evidence rather than just stats. (1) Coding task (palindrome fn) + math word problem, greedy decode, bias 0/0.01/0.02: 3 of 4 biased runs token-identical to unbiased; the 4th differed by one cosmetic character (- vs unicode minus), arithmetic/answer unchanged; generated code executes and all asserts pass at every bias level. (2) No Q8_0 32k full-GPU reference is possible (30 GiB model does not fit in 24 GB regardless of caching), so tested the mechanism instead: exact mode (bias 0) with a 14 GiB cache vs a 6 GiB cache on the same 32k prompt gives token- and logits-hash-identical output, i.e. the cache is invisible to the result as claimed. Results: results/overnight/realtask/.
- 05:45 built a live monitoring dashboard (user request): monitor/server.py (stdlib Python, no deps) + monitor/index.html.
  Instrumented svram-verify with a --progress JSON-lines stream (per prefill chunk and per decode step: cache hits/misses/
  evictions/bytes_h2d/bytes_ssd/hit_rate, warm timing) - purely additional I/O, does not touch the compute path. Server polls
  nvidia-smi/proc for VRAM/RAM/SSD and tails the stream; computes a bottleneck label (SSD read / PCIe host-copy / GPU compute /
  balanced). Verified end-to-end against two real runs: a plenty-of-RAM zero-copy bias-0.02 run (VRAM 22.7 GiB, hit rate 98.9%)
  and a 4 GB RAM-capped SSD run via cold_run.py (RAM panel correctly showed the 4 GiB systemd-run cap, SSD read 1.0-1.6 GB/s,
  bottleneck correctly flagged SSD READ during prefill). Screenshots confirm both light and dark themes render cleanly.
- 23:10 root-caused and closed out the "cache exactness regression" alarm from earlier. It was a TEST METHODOLOGY bug, not a
  code bug: comparing --storage cache against a --storage resident baseline run at different --n-batch/--n-ubatch picks up
  ordinary CPU-fallback (batches > GGML_SCHED_EXPERT_CACHE_MAX_BATCH=4 correctly excluded from the cache and run on CPU, by
  design) and batch-size-dependent GPU kernel selection -- confirmed resident-vs-resident alone diverges across batch sizes.
  Apples-to-apples (same batch size both sides): classic cache, warm start (direct-io and mmap), and zero-copy with/without
  bias are all bit-identical (tokens + logits hashes) to full-GPU resident. Along the way found and fixed a real, separate
  bug: my own earlier "use slots_view not slots_persist" fix (aimed at a theoretical sync gap) broke warm start outright
  (crash: read_direct is shared by live-miss handling, where slots_view exists, and warm start, which runs before any graph
  exists, where it does not) -- corrected to prefer slots_view only when bound. Also wired the expert-cache lazy-read flags
  and cache-aware-routing bias hook into qwen35moe.cpp (Qwen3.6-35B-A3B's architecture), mirroring qwen3moe.cpp. Resuming the
  Qwen3.6-35B-A3B download/bring-up now.
- 23:45 Qwen3.6-35B-A3B-Q4_K_M (ggml-org GGUF, 20.4 GB, 256 experts/8+1 active, hybrid gated-delta-net+attention) downloaded and
  wired into the expert cache: extended the qwen3moe-only architecture gate in llama.cpp to also allow qwen35moe. Loads and
  runs correctly out of the box (coherent output on the very first try). CLASSIC SLOT CACHE verified exact (matched-batch
  methodology): tokens and logits hashes bit-identical to full-GPU resident. ZERO-COPY TIER: exactness FAILS even at properly
  matched batch size (ub=1) -- diverges from step 0. Checked and ruled out: tensor naming collisions (gate/up/down/shexp all
  cleanly distinct in this GGUF), mixed quantization types (all three expert tensors are uniformly Q4_K), and a hardcoded
  128-expert assumption in the pointer-table sizing/kernel index width (stride is correctly computed as max n_expert = 256,
  channel_x is uint32_t). Root cause not yet found. Classic cache is a complete, verified way to run this model under
  SuperVRAM today; zero-copy (the fastest tier, used for the 48-GB-class comparison) needs further debugging before it can be
  used on this architecture. Recommend running the benchmark on classic cache for this model, or holding zero-copy for a
  follow-up session.
