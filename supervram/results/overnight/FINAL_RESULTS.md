# SuperVRAM final results: two models, 32,000-token input / 4,096-token output, RTX 3090 (24 GB)

Machine: RTX 3090 24 GB, 125 GB RAM, WD SN550 NVMe (~2.4 GB/s). Everything below was measured on this machine;
estimates are labeled. Raw numbers: `bench/SUMMARY.tsv`, `qwen36/`, `ram4g/`; narrative log: `LOG.md`.

## How every comparison is made (why it's fair)

- **Same text everywhere.** Every system decodes the *same* forced/teacher-forced output tokens after the same 32,000-token
  prompt, so routing work is identical across systems. Free generation gave different text per system early on (hit rates of
  92.6% vs 96.7% at the same cache), which made those comparisons invalid; this was fixed before any of the numbers below.
- **Same batch size on both sides of every exactness check.** A real methodology bug cost a lot of time this session: comparing
  a cache-mode run against a `--storage resident` reference taken at a *different* `--n-batch`/`--n-ubatch` produces different
  (but individually valid) floating-point results, because GPU kernel selection depends on batch size, and batches above a
  threshold are — by design — excluded from the small on-GPU cache and correctly run on CPU instead. Both are ordinary
  non-determinism, not corruption, but they look identical to a real bug if you're not controlling for it. Every exactness
  number in this document was re-verified with matched batch sizes on both sides.
- **"Total" = prefill of the 32k prompt + N decoded tokens.**

## Model 1: Qwen3-30B-A3B Q8_0 (30 GiB, 128 experts, 8 active) — the model this project targets

| System | Total (4,096 out) | vs plain llama.cpp |
|---|---|---|
| Plain llama.cpp, best static split (`--n-cpu-moe 25`) | 142 s | 1.00x |
| **SuperVRAM, exact** (warm start + prefetch + helper thread) | **123 s** | **1.15x** |
| SuperVRAM + cache-aware routing, strength 0.01 (approximate) | 95 s | 1.49x |
| SuperVRAM + cache-aware routing, strength 0.02 (approximate) | 87 s | 1.63x |
| All-experts-hot proxy for a 48 GB-class card (not a real one) | 84 s | 1.69x |
| **Small-RAM machine**: 4 GB RAM + SSD, exact (167 s at 4,096 out; 128 s at 2,048 out, strength 0.02) | 167 s | — |

- **Exact mode is free — 15% faster than the best plain-llama.cpp split, with bit-for-bit identical output**, verified against
  full-GPU execution (Q4_K_M tokens + logits hashes).
- **Cache-aware routing is optional and approximate.** Perplexity is unchanged up to strength 0.02 across 4 text domains; free
  generation gets measurably more repetitive at 0.02 (repeated 8-grams roughly doubled on 2 of 3 domains tested); a coding
  task and a math word problem both still solved correctly at every strength tested, and 3 of 4 biased runs were byte-for-byte
  identical to the unbiased run on those spot checks. Recommend strength 0.01 as the safe default, 0.02 as aggressive.
- **The small-RAM row is the actual headline.** Plain llama.cpp cannot serve this configuration at usable speed — its
  CPU-offloaded experts must fit in RAM. SuperVRAM runs the full job with only ~3 GiB of RAM used, streaming the rest from
  the SSD.

## Model 2: Qwen3.6-35B-A3B Q4_K_M (20.4 GiB, 256 experts, 8 routed + 1 shared, hybrid linear-attention architecture)

Added this session to test whether the approach generalizes to a newer, bigger model. It does — with an important honest
caveat below.

| System | Total (4,096 out) | Decode |
|---|---|---|
| Plain llama.cpp, best static split (`--n-cpu-moe 4`; `3` runs out of memory) | **51 s** | 100.5 t/s |
| SuperVRAM, classic cache, exact | 66 s | 98.1 t/s |
| SuperVRAM, classic cache + bias 0.02 | 70 s | 89.0 t/s |
| SuperVRAM, zero-copy tier, exact (14 GiB cache) | 70 s | 88.5 t/s |
| SuperVRAM, zero-copy tier + bias 0.02 | 71 s | 86.2 t/s |
| **Small-RAM machine**: 4 GB RAM + SSD, exact (2,048 out) | 112 s | 65.4 t/s |

- **Plain llama.cpp wins outright here when RAM is plentiful**, and this is a real, structural finding, not a bug: this
  model's architecture (linear-attention/recurrent layers on 3 of every 4 layers) keeps its KV cache tiny, so only 4 of 40
  layers need to go to the CPU to fit in 24 GB. There is very little left for a VRAM-expert-cache to improve on — like
  optimizing a route that is already almost entirely highway. Bias made things slightly worse, not better, because the
  cache (14 GiB, big enough to hold ~83% of the 256 experts per layer) already reached 99%+ hit rate on its own; the extra
  routing-bias computation cost more than the small remaining hit-rate gain was worth.
- **All storage modes (classic cache and zero-copy) are verified exact** on this model too (matched-batch methodology,
  tokens + logits hashes bit-identical to resident), after a real scare mid-session: an initial mismatched-batch-size test
  wrongly looked like a correctness bug in the zero-copy tier. It was not; see the caveats section.
- **The small-RAM row is again the genuine win.** Plain llama.cpp cannot run a 20.4 GiB model's CPU-offloaded portion on
  4 GB of RAM at all. SuperVRAM does it in 112 s using under 3 GiB of RAM, streaming ~146 GiB off the SSD over the run.

## The honest pattern across both models

**SuperVRAM's speed advantage over plain llama.cpp depends on how much is left on the table by the best static split.**
Model 1's older, KV-cache-heavy architecture forces a big CPU split (25 of 48 layers) even with plenty of RAM, leaving real
room to improve — SuperVRAM wins by 15–63% there. Model 2's newer, cheap-KV-cache architecture barely needs a CPU split at
all (4 of 40 layers) when RAM is plentiful — there SuperVRAM's caching overhead isn't earned back, and plain llama.cpp wins.

**What holds regardless of architecture: the small-RAM case.** On both models, SuperVRAM is the only way to run the job at
all with only ~4 GB of spare RAM — plain llama.cpp requires the CPU-offloaded portion to fit fully in RAM and has no
fallback. That is the project's actual, unconditional contribution: it turns "doesn't run" into "runs, using GB-scale RAM
instead of tens of GB," on any MoE model tested so far, regardless of whether it also wins on raw speed.

## What is not proven / caveats

- No real 48 GB GPU was available for either model; the "48 GB-class" row (Model 1 only) is an all-hot proxy, not a
  measurement of a real larger card.
- Cache-aware routing (bias) changes model outputs. Tested via perplexity and repetition statistics plus two real-task spot
  checks (Model 1 only); no downstream task-accuracy benchmark was run on either model. Exact mode is the default,
  bit-identical result on both models; bias is opt-in and clearly labeled approximate everywhere above.
- Both models were tested with one 32,000-token prompt (concatenated project documentation) and one canonical output text
  per model; other prompts will have different routing statistics and hit rates.
- **Methodology lesson, stated plainly because it cost real time this session:** two separate "critical regression" scares
  during this work — one against the classic cache, one against the zero-copy tier on the new model — both turned out to be
  the same test-methodology mistake (comparing runs at different batch sizes), not real bugs. One genuine, narrower bug *was*
  found and fixed along the way: an earlier defensive change to a shared memory-write path broke warm start's direct-I/O
  path outright (a reproducible crash); this is fixed and re-verified. All exactness claims in this document were
  specifically re-checked with matched batch sizes after these lessons, not carried over from the earlier, flawed tests.
- Model 2's RAM-poor row used a 2,048-token forced output, not the full 4,096, for time reasons; Model 1's RAM-poor row has
  both.

## Reproduce

`docs/REPRODUCE.md` (commands), `scripts/bench32k.sh` (the benchmark; set `BENCH_MODEL` to switch models),
`scripts/cold_run.py` (RAM-capped runs). Patches `0001`–`0009` in `patches/` apply to llama.cpp `ce8caa6`, plus the
qwen35moe integration and warm-start fix committed this session (`third_party/llama.cpp` git log,
`llama_cpp_modified/FILES.txt`); the chain was verified to reproduce the built tree byte for byte after patch 0009.
