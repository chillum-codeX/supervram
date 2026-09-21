# SuperVRAM / ElasticVRAM: what was built and what was measured, in plain language

Machine: RTX 3090 (24 GB), WD SN550 NVMe (about 2.4 GB/s), 125 GB RAM, Qwen3-30B-A3B models.
All numbers below come from files under `results/rtx3090/`; the detailed tables are in `docs/IMPLEMENTATION_STATUS.md`.

## 1. The goal

Let a 24 GB GPU run models much larger than 24 GB by treating an NVMe SSD as a cold storage tier, and get speed close to a real
high-memory GPU (for example a 48 GB card). The SSD is not made as fast as VRAM. Instead the runtime keeps the experts a token needs in
VRAM and fetches the rest ahead of time. Mixture-of-experts (MoE) models make this plausible: Qwen3-30B-A3B has 128 experts per layer
and uses only 8 per token.

The notes set the rules for a valid result:
1. System RAM does not count as model capacity (staging of about 1 GB is allowed).
2. The Linux page cache must not fake the result (cold runs, direct I/O, prove where the data lives).
3. Success is measured against a native larger-memory GPU: 0.5x viable, 0.8x strong, 0.95x near-native.
4. The model should reach 35-48 GB or more.

## 2. Where we started, and what I got wrong along the way

The repo already had a policy simulator, a llama.cpp patch that runs experts on the CPU from a memory-mapped file, and a plan to add a
GPU expert cache. I finished that plan first, without checking it against your document. That produced numbers that looked good and were
not valid. The mistakes, all found and fixed:

- **Warm numbers were RAM-backed.** Both models sat entirely in the Linux page cache, so the early "40 tokens/s" was GPU + RAM, not
  GPU + SSD. This is exactly what rule 2 forbids.
- **A timing bug.** The benchmark tool stopped its clock before the GPU finished, so full-GPU runs showed thousands of tokens/s and
  every earlier decode number was inflated. Fixed, and the sweeps were rerun.
- **A missing line in a patch**, so the saved patch did not reproduce the tested build. Fixed and verified: applying patches 0001, 0002,
  0003, 0004 to the pinned llama.cpp rebuilds the tested tree exactly.
- **Two wrong statements I made and later corrected:** that the Q4 cache "plateaus at 63 t/s because of per-step overhead", and that
  overlapping reads with compute could gain only about 10%.
- **A measurement race** that once reported 47 GiB of RAM use (it sampled the desktop app, not the test process). Fixed.

## 3. What was built

A bounded GPU cache of expert weights inside llama.cpp's scheduler, in four patches:
- **0001 (existing):** demand-paged experts on the CPU.
- **0002:** the GPU expert cache. Each layer gets a fixed number of GPU "slots". For every token the router says which experts are
  needed; hits are used in place, misses are copied into a slot (evicting the least recently used expert), and the routing ids are
  remapped to slot numbers. Output is bit-identical to running the same weights fully on the GPU.
- **0003:** the SSD path. A miss is read straight from the model file with O_DIRECT (bypassing the page cache), several reads at once,
  into a 128 MiB pinned staging buffer, then copied to the GPU. A layer's three weights (gate, up, down) are read as one batch with one
  GPU sync per layer. Also an opt-in routing trace.
- **0004:** real command-line options for all of it: `--moe-expert-direct-io`, `--moe-expert-io-threads`, `--moe-expert-staging-mib`,
  `--moe-expert-trace` (they work in `llama-server`, `llama-completion` and the other llama.cpp tools; the old `SVRAM_*`
  environment variables still work as a fallback).

## 4. Results in order

Q8_0 (30 GiB, does not fit in VRAM), 16 GiB cache, cold start unless noted:

| Step | Decode | Notes |
|---|---|---|
| Model warm in RAM | about 40 t/s | not a valid GPU+SSD result |
| Page-fault reads, cold, no RAM limit | 2.0 t/s | 22 GiB of the model ended up in RAM |
| Page-fault reads, 4 GB RAM cap | 0.55 t/s | reads ran at 0.12 GB/s |
| Direct reads, 4 GB cap, 256 tokens | 13.8 t/s | mostly filling an empty cache |
| Direct reads, long session (1,536 tokens) | 27.5 t/s | 97.4% hit rate |
| Plus one batched read and one sync per layer | 31.8 t/s | reads at 2.1 of 2.4 GB/s |

## 5. What each investigation showed

- **The SSD is the wall.** Expert-sized reads reach about 2.4 GB/s (1.6 GB/s one at a time, 0.12 GB/s by page faults). At a 97% hit
  rate a token needs about 51 MB from the SSD, so this drive caps Q8_0 near 47 t/s in theory. In practice 72% of decode time is
  waiting on the SSD.
- **Eviction policy is not the lever.** Replaying 3,276 tokens of real routing: LRU 96.3% hits, LFU with decay 96.4%, the best
  possible (an oracle that knows the future) 97.1%. A perfect policy removes only about a quarter of the misses.
- **A warm, persistent cache matters more:** 97.3-97.4% in a long session versus 94-96% for a fresh prompt.
- **Sharing loads across tokens does not help.** Verifying several consecutive tokens at once (speculative decoding) saves only about
  6% of misses; decoding several requests together makes hit rates worse (simulated on the real traces).
- **Smaller experts help, at a cost.** Q4_K_M experts read 32% fewer bytes and decode 41.4 vs 31.8 t/s at the same cached fraction.
  Quality: Q4 picks Q8's next token 96.0% of the time, and on 30.7k tokens of human-written text its perplexity is 1.020x Q8's
  (95% CI 1.015-1.025).
- **A bigger model hurts fast.** A synthetic 43 GiB model (half the layers' experts as F16, no download) with the same 16 GiB cache
  holds only 38% of experts: hit rate 90.9%, 272 MB/token from the SSD, 6.6 t/s.

## 6. Scorecard against your rules

| Criterion | Result |
|---|---|
| Model larger than 24 GB runs correctly | **Yes** (30 GiB and 43 GiB; Q4 output bit-exact against full GPU) |
| RAM not used as model capacity | **Yes for expert data.** Peak RAM about 1.7 GiB under a 4 GB cap: 0.2 GiB heap, a 128 MiB staging buffer and 1.3 GiB of dense weights read once at load. That is above the notes' "about 1 GB staging" example |
| Page cache not faking the result | **Yes** on the direct path; the original path failed this |
| 0.5x of a native card | **No.** About 0.32x at 30 GiB and about 0.08x at 43 GiB, against native speeds I estimated, not measured |
| 35-48 GB model | **Partly.** A synthetic 43 GiB model, not a real one |

## 7. What is not proven

- No native 48 GB card here, so every parity figure is an estimate (about 100 t/s for Q8_0, about 78 for the 43 GiB model).
- One prompt for the long runs; decode only; the cache does not handle long prompt batches (it errors when a batch needs more experts
  than slots).
- Prefetch is not built; simulation shows it would lose speed with history-based prediction and gain at most about 4-11% with an ideal early-router predictor.
- The 43 GiB model is synthetic: throughput and capacity only, not quality.
- Quality is measured on one text domain and against a Q8 reference, with no full-precision baseline or task benchmark.
- The second NVMe (Intel, PCIe 3.0 x4, unmounted, Windows-style NTFS) was not read: it needs root, and I did not use it. There is no
  faster (PCIe 4) drive here, so the faster-drive test was not run.

## 8. What would change the outcome

1. **A faster drive or several in parallel.** The measured cost per token is fixed, so throughput scales with SSD bandwidth: about 47
   t/s at this drive's 2.4 GB/s, about 68 t/s at 3.5 GB/s and about 137 t/s at 7 GB/s for Q8_0 (bandwidth-only arithmetic). This is
   the biggest lever and it is hardware.
2. **Fewer bytes per expert** (Q4-style formats): about 30% faster for about 2% perplexity.
3. **Cache capacity:** the model's routing working set has to fit; below about half the experts cached, throughput collapses.
4. **Prediction-based prefetch:** tested by simulation and **not built**. A perfect predictor would gain +4% to +30% here, but predictions from routing history are right only 1-2% of the time, so wrong reads would make decoding slower (details in `IMPLEMENTATION_STATUS.md`). Only a model-router-based early predictor could work, for about 4-11%.

## 9. Where everything lives

Code: `patches/0001..0004`, `src/svram_verify.cpp`. Tools: `scripts/cold_run.py`, `ssd_expert_read_bench.py`, `analyze_policies.py`,
`compare_quality.py`, `aggregate_ablations.py`. Results: `results/rtx3090/` (cold, ablations-long, traces, quality). Status tables:
`docs/IMPLEMENTATION_STATUS.md`; evidence log: `writer_handoff/EVIDENCE_LEDGER.md`; running log: `../progress.md`.
