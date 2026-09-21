# Overnight results: 32,000-token input, 4,096-token output, Qwen3-30B-A3B Q8_0 (30 GiB) on an RTX 3090 (24 GB)

Machine: RTX 3090 24 GB, 125 GB RAM, WD SN550 NVMe (about 2.4 GB/s). Everything below was measured on this machine;
estimates are labeled. Raw numbers: `bench/SUMMARY.tsv`, `ram4g/SUMMARY.txt`, narrative in `LOG.md`.

## How the comparison is made (why it is fair)

- **Same text everywhere.** Free generation gives each system different text and different routing (hit rates of 92.6% vs 96.7% at the
  same cache), which made earlier comparisons unfair. Every system below decodes the *same* natural 4,096-token output
  (`bench/canonical-out-4096.txt`, teacher-forced) after the *same* 32,000-token prompt, so the routing work is identical.
- **"Total" = prefill of the 32k prompt + 4,096 decoded tokens.** Decode tokens/s is averaged over the whole 4,096.
- **The "48 GB-class" line is a proxy, not a real 48 GB card.** It is this GPU running the same kernels with every token
  repeating, so all experts are hot (no cache misses, no promotion work). It is the speed a card that holds every needed expert
  would reach at 32k context with this software. A real 48 GB card would also hold the whole 30 GiB model and could prefill faster;
  I have no such card, so its prefill is not measured.

## Headline: plenty of RAM (24 GB VRAM + RAM)

| System | Prefill 32k | Decode 4,096 | Total | vs plain llama.cpp |
|---|---|---|---|---|
| Plain llama.cpp, best static split (`--n-cpu-moe 25`; 24: 144 s, 27: 147 s, 23 out of memory) | 22.4 s | 34.4 t/s | 142 s | 1.00x |
| Previous slot cache (14 GiB) | 34.6 s | 37.8 t/s | 143 s | 0.99x |
| **New zero-copy tier, exact** (warm start + prefetch + helper thread) | 22.7 s | 40.9 t/s | **123 s** | **1.15x** |
| New tier + cache-aware routing, bias 0.01 (approximate) | 22.7 s | 56.9 t/s | **95 s** | **1.49x** |
| New tier + cache-aware routing, bias 0.02 (approximate) | 22.8 s | 63.6 t/s | **87 s** | **1.63x** |
| All-experts-hot proxy ("48 GB-class") | 23.0 s | 67.1 t/s (about 75 steady) | 84 s | 1.69x |

Reading it:
- **Exact mode** (bit-identical outputs to full-GPU execution, verified on Q4_K_M) is 15% faster than the best plain llama.cpp
  split. It does not reach the proxy: about 5% of expert accesses miss the VRAM cache and are read from pinned RAM.
- **Cache-aware routing** nudges the router toward experts already in VRAM (selection only; the mixing weights are unchanged).
  With bias 0.02 the hit rate goes from 94.5% to 99.2% and total time is within 4% of the all-hot proxy (87 s vs 84 s),
  decode 63.6 t/s vs about 75 steady. This is an **approximate mode**: outputs differ from the unbiased model.
- Repeat runs vary by about 1-2% (exact 125/123 s, bias 0.01 96/95 s, bias 0.02 88/87 s).

## Quality cost of cache-aware routing (measured, not assumed)

- **Teacher-forced perplexity** (same text scored by the biased and unbiased model, paired): no measurable change up to bias 0.02 in 4 domains
  including the 32k context (docs at 32k: perplexity x1.0002 at 0.02; hit 93.8% -> 98.6%). Bias 0.03 costs +0.9%; bias 0.05 costs +2.1% (2.7 standard errors).
- **Free generation gets more repetitive with strong bias.** 3 domains x bias {0, 0.005, 0.01, 0.02}, 768 tokens, 1 seed each (noisy): no
  detectable degradation up to 0.01; at 0.02 the share of repeated 8-grams rose from 4.1% to 7.2% (docs) and 0% to 2.8% (python docs); code unchanged.
  On one earlier prompt bias 0.05 pushed repeated 8-grams from 3.1% to 19.4%.
- A multiplicative variant (`--moe-expert-bias-mul`) lands on about the same speed/quality frontier; additive is the recommendation.
- **Recommendation:** exact mode as the default; bias 0.01 as the "safe fast" setting; bias 0.02 as aggressive. Neither has been checked
  on a downstream task benchmark, only perplexity and repetition statistics, and diversity was one seed per cell.

## Headline: small RAM (24 GB VRAM + 4 GB RAM cap + SSD)

The regime that plain llama.cpp cannot serve (its experts must live in RAM; page-faulting them from the SSD ran at 0.55 t/s in
earlier cold tests). Classic slot cache + direct I/O, warm start + device-to-device prefill reuse. RAM peak 2.2-2.6 GiB, no expert
data in the page cache. Here the decode covers **2,048 forced tokens** (not 4,096), so totals are for 32k in / 2,048 out:

| Configuration | Prefill 32k | Decode | Total (2,048 out) | SSD reads |
|---|---|---|---|---|
| Before tonight's work, exact | 122.8 s | 14.5 t/s | 264 s | - |
| Before, bias 0.02 | 122.8 s | 38.7 t/s | 176 s | - |
| Warm start + prefill reuse, exact | 79.9 s | 14.9 t/s | 217 s | 299 GiB |
| Warm start + prefill reuse, bias 0.01 | 80.2 s | 31.5 t/s | 145 s | 175 GiB |
| **Warm start + prefill reuse, bias 0.02** (cache 14 GiB, ub 4096) | **80.2 s** | **42.6 t/s** | **128 s** | 148 GiB |
| Same as the bias-0.02 row but 4,096 forced out | 80.1 s | 46.9 t/s | **167 s (4,096 out)** | - |
| Same, bias 0.02, cache 11 GiB, ub 8192 | **52.1 s** | 29.5 t/s | **122 s** | 141 GiB |

Measured with the full 4,096 forced output tokens (bias 0.02, cache 14 GiB, ub 4096): prefill 80.1 s + decode 46.9 t/s (42.9 -> 51.5 t/s by
position as the cache warms) = **167 s total**, about 1.18x the 142 s of plain llama.cpp on a machine with plenty of RAM, on a machine that has
only 4 GB of free RAM plus an SSD. Plain llama.cpp cannot run that workload at usable speed there; this works because of the streaming reads.
Prefill is the SSD-bound part: it streams the 30 GiB model through the GPU once per prompt batch, and a bigger batch (ub 8192) trades cache
size for fewer passes.

## What is not proven / caveats

- No real 48 GB GPU was available: the "48 GB-class" row is an all-hot proxy, and parity ratios against a native card are estimates.
- Cache-aware routing changes model outputs. Perplexity is unchanged up to 0.02, but generation diversity drops at 0.02 and no
  task-level (accuracy) evaluation was run. Exact mode is the default result.
- Single 32k prompt (concatenated project documentation); other prompts have different routing statistics. The warm profile
  (`warm-profile-q8.txt`) was built from routing traces of other prompts, not from the benchmark prompt itself.
- RAM-poor: the table rows are 2,048 output tokens; one bias-0.02 run was repeated at 4,096 (167 s). The exact-mode and 11 GiB rows were not run at 4,096.
- Exactness (tokens and logits hashes identical to full-GPU execution) was verified on Q4_K_M with short prompts and 1,500-token prompts, not on Q8_0
  at 32k (no Q8 full-GPU reference fits in 24 GB). The Q8 runs use the same code paths.
- The synthetic 43 GiB model and the "SSD is the wall" analysis from the previous day are unchanged; a bigger model than VRAM+RAM
  still falls to about 6-7 t/s on this single 2.4 GB/s SSD.

## Reproduce

`docs/REPRODUCE.md` (commands), `scripts/bench32k.sh` (the benchmark), `scripts/cold_run.py` (RAM-capped runs), patches `0001`-`0009`
in `patches/` apply to llama.cpp `ce8caa6`; the chain was verified to reproduce the built tree byte for byte.
