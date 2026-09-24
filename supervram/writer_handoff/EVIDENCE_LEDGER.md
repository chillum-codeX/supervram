# Evidence ledger

## Measured in the current environment (`measured_rtx3090`)

- Host: Intel Core i9-9920X, 24 threads, 125 GiB RAM, `/home` ext4 on WD SN550 NVMe.
- GPU: GeForce RTX 3090 24 GiB, driver 595.91.07, CUDA 12.6, BAR1 256 MiB, idle PCIe gen2 x16 (max gen3).
- `cuFileRead` into `cudaMalloc` works in compatibility mode (~1.8 GB/s, matches O_DIRECT pread+H2D). `nvidia-fs` is absent; this is host bounce, not SSD-to-VRAM DMA.
- Sequential O_DIRECT pread of a 2 GiB file on `/home`: ~1991 MiB/s.
- Native C++ reader test and Python cache/store/predictor tests: passed on this host as well.
- Tiny synthetic Qwen3 MoE: resident/mmap/cache greedy token ids and logits hashes identical.
- Qwen3-30B-A3B Q4_K_M: greedy 8-token ids identical across resident/mmap/cache; resident vs cache logits hashes identical.
- Qwen3-30B-A3B Q8_0: greedy 8-token ids identical across `--cpu-moe`, mmap, cache; logits hashes differ (CPU vs CUDA).
- llama-bench (3 reps): see `results/rtx3090/SUMMARY.json`. Cache decode-only Q4 tg128 47.1 t/s vs mmap 33.8 vs full GPU 171; Q8 cache tg64 24.6 vs cpu-moe/mmap ~23.1.

- Cache-size x policy ablation, 3 reps x 256-token decode (`results/rtx3090/ablations-long/`): Q8_0 with a 16 GiB cache decodes ~40 t/s vs 22.7-24.4 for `--cpu-moe` (94 % hits); Q4_K_M cache plateaus ~63 t/s vs 195-200 full GPU (per-step overhead); LRU >= LFU when cache-constrained; Q4 cache output bit-identical to full-GPU resident. Earlier `svram-verify` decode_tps values were inflated by a timer bug and are superseded.
- Cold-SSD path (`results/rtx3090/cold/`): warm-cache numbers were RAM-backed (models 100 % page-cache resident). Cold Q8_0, 16 GiB cache, 256 tokens: mmap page faults 2.0 t/s uncapped and 0.55 t/s under a 4 GiB RAM cap; direct O_DIRECT reads into 128 MiB pinned staging 13.8-13.9 t/s under the 4 GiB cap (3 reps), 1.7 GiB peak cgroup, no expert data in the page cache. SSD ceiling measured at ~2.4 GB/s. Estimated (not measured) parity vs a native 48 GB card: ~0.14x.
- Policy headroom (offline replay of 8 real Q8_0 traces): LRU 96.3 %, LFU-decay 96.4 %, SLRU 96.3 %, Belady optimum 97.1 % (per-layer) / 97.3 % (shared) at 16 GiB, so replacement policy has < 1 point of headroom. Warm, persistent cache: 97.3-97.4 %, measured 26.3 t/s overall / 27.5 t/s steady state over 1,536 tokens cold-started under a 4 GiB RAM cap. Sharing loads across k consecutive tokens saves only ~6 % of misses at k=8; lock-step batching of independent requests raises misses per token.
- Layer-batched direct reads (`results/rtx3090/cold/*batched*`): 1,536-token cold-start run, 4 GiB RAM cap: 26.3 -> 30.5 t/s overall (30.4 in a repeat), 27.5 -> 31.8 t/s steady state; SSD wait 42.6 -> 36.1 s (2.1 GB/s); output identical. Estimated ~0.32x of a native 48 GB card (unmeasured estimate).
- Smaller experts (`results/rtx3090/quality/`, `cold/q4-*`): Q4_K_M agrees with Q8_0's next token 96.0 % of the time (teacher-forced, 3,072 tokens, 8 prompts) at +2.1 % perplexity on Q8's text (reference is Q8, not ground truth). Ground-truth check on 30.7k tokens of human-written docs prose: paired PPL ratio Q4/Q8 = 1.020 (95% CI 1.015-1.025), Q4 worse in 48/60 chunks; single corpus, no F16 reference. Cold SSD path, same 55 % cached fraction: 41.4 t/s vs 31.8 t/s (-32 % bytes read). Q4 with everything cached: 126.8 t/s (0.64x of full-GPU resident). Equal-fraction parity vs the same model's native speed is about 0.21x (Q4) vs ~0.32x (Q8, estimated native).
- Larger model (`results/rtx3090/cold/big46g-*`): a synthetic 42.9 GiB variant (experts of layers 0-23 as F16 dequantized from Q8, rest Q8_0; throughput/capacity only, not a quality result). Cold start, direct I/O, 4 GiB RAM cap, 16 GiB cache: 6.6 t/s, 90.9 % hits (38 % of experts cached), 272 MB/token from the SSD at 2.2 GB/s (81 % of decode time), 1.69 GiB peak RAM. Q8_0 (55 % cached) gave 31.8 t/s. Estimated parity vs a native card ~0.08x (unmeasured). Drives: two PCIe 3.0 x4 NVMe (WD SN550 measured 2.4 GB/s; Intel unmounted NTFS, not readable without root); no faster drive available.
- Prefetch feasibility (simulation on 3,276 real routing tokens, Q8_0, 16 GiB): an oracle gains +4 % (1 layer ahead) to +30 % (a whole token ahead) on this drive; routing-history predictors are right 1.4-2 % of the time (12-13 % for the top 0.1 % most confident guesses), so wrong reads make prefetch slower than none (0.82-0.96x at 1.0-0.5 wrong reads per useful one). Not built. Model calibrated to within 7 % of the measured speed. Simulation, not an end-to-end measurement.
- Long context (`results/rtx3090/longctx/`, Q8_0, 32,000-token prompt): stock llama.cpp (25 layers' experts on CPU) prefill 22.6 s, decode 33.7 t/s; tiered 14 GiB VRAM cache + RAM prefill 34.7 s, decode 40.5 t/s (first 2,048 tokens; later decode is inflated by greedy looping); tiered 14 GiB cache + SSD with RAM capped at 4 GiB: prefill 123 s, decode 15.2 t/s, peak RAM 2.2 GiB. Earlier 'tiered decodes 25-35 % faster' claim withdrawn (used looping output). Single runs.
- Zero-copy expert tier, stage 1 (patch 0007): experts in pinned RAM, GPU kernel reads them in place through per-expert pointers. Exact (Q4_K_M, 64 tokens: tokens and logits hashes identical to full-GPU). All-from-RAM decode 6.09 t/s (Q4) and 3.99 t/s (Q8) = 6.7 and 7.8 GB/s effective PCIe reads vs an 11.3 GB/s ceiling. Projected ~60 t/s with a VRAM cache (unmeasured until stage 2).
- Zero-copy tier stage 2 (patch 0008, background promotion with score-based admission): exact vs full GPU (Q4, 384-512 tokens). Q4 8 GiB short context: 50.5 t/s vs 42.4 t/s for the earlier cache. Target workload (Q8_0, 32,000-token prompt, 14 GiB cache): prefill 33.4 s, decode 32.7 t/s over 2,048 tokens (21.8 -> 39.9 as the cache warms), i.e. NOT better than the earlier design (40.5 t/s, 85 s total) or plain llama.cpp (33.7 t/s, 83 s total); the ~60 t/s projection was not reached (promotion step ~5.4 ms/token and PCIe contention).

## Deterministic synthetic trace replay

`results/simulation-smoke.json` validates the software policies, not LLM throughput. Do not interpret those hit rates as Qwen routing behavior.

- Adaptive speculative scheduler (ASS, `PLAN_ADAPTIVE.md`, `supervram/scheduler.py`): on a
  deterministic, intentionally weak-predictor synthetic trace (`tests/test_scheduler.py`'s
  `_weak_trace`, near-zero locality so history-based prediction has little real signal), gated
  prefetch's `waste_bytes` is strictly lower than blind unconditional prefetch at equal
  `--prefetch-depth` in that specific test scenario (e.g. locality ~0.05: 287,768,576
  vs 321,126,400 bytes; locality ~0.15: 295,763,968 vs 320,864,256). At artificially high locality
  (>= 0.5, not representative of the 1.4-2 % real routing-history accuracy measured above) the
  margin narrows or disappears -- consistent with the gate mattering exactly where the earlier
  prefetch-feasibility measurement found blind prefetch to be a net loss. Software-policy
  validation only, not LLM throughput.
- **Full 3,600-run scaled ablation matrix** (`scripts/run_ablations.py --mode simulate`; a first,
  unscaled attempt against the matrix's original multi-GiB cache axis and the harness's tiny 16
  MiB default trace produced zero evictions in all 3,600 runs -- a null result, not a win, since
  discarded): 12-layer/32-expert/256 KiB-expert trace (96 MiB working set), cache sizes as
  fractions of that working set (0.1x-1.5x, guaranteeing real pressure), locality 0.2. Pre-Phase-E
  (`results/ablation-simulate-ass-scaled/`): of 2,304 comparable rows, modeled throughput was
  worse for ASS in 65 (2.8 %, worst -33.5 %), 63 of them `predictor=oracle`. **This corrected an
  earlier claim (and a test docstring) that ASS's modeled throughput is never worse than the
  baseline's "universally" -- that was true only for the one scenario it was checked against.**
- **Phase E fix and re-verification** (`results/ablation-simulate-ass-scaled-phaseE/`): the first
  hypothesis for the oracle regression -- slow EMA calibration convergence -- was tested and found
  wrong (a bias-corrected faster-converging EMA made the 480-row oracle-only subset marginally
  *worse*, 70/384 vs 63/384). The actual cause: `plan_window`/`record_and_observe` each call the
  base predictor's `predict()` independently, and overlapping lookahead windows mean the same real
  access gets predicted multiple times (194 calls measured for a 40-access trace). Harmless for
  stateless predictors, but `OraclePredictor.predict()` mutates a queue on every call, so redundant
  calls silently exhausted it early. Fixed with a per-(token, layer) prediction cache
  (`AdaptiveSpeculativeScheduler._predict_once`) so the base predictor is called at most once per
  real access -- verified the oracle's call count dropped to an exact 1:1 match with trace length.
  Re-running the full 3,600-run matrix: modeled throughput worse for ASS in only 12/2,304 (0.5 %,
  down from 65/2.8 %), 8 of them oracle (down from 63) and 4 history (noise-level); worst
  regression fell from -33.5 % to -13.2 %. `waste_bytes` results were essentially unchanged (708
  better / 1,175 tied / 421 worse), as expected since this fix targeted throughput specifically.

## Analytical projection

`results/roofline-projection.json` is not measured data.

- `scripts/cost_model.py`'s roofline projection of the same ASS replay counters (`compute_s` vs
  `blocking_io_s`, `wall_s = max` of the two): for that one weak-predictor scenario, modeled
  throughput stayed at or above the blind-prefetch baseline's across a drive-speed sweep, and the
  gap widened over a middle range of simulated drive speeds (2.4 -> 0.5 GB/s: parity -> ~10 %
  ahead in one run), then narrowed back toward parity at very slow simulated speeds because the
  gate correctly stops firing anything once no candidate's read fits the fixed per-window compute
  budget (verified: 0/3584 fired at 0.1 and 0.02 GB/s) -- a safe floor, not a broken result. **Not
  a universal property**: the full scaled ablation matrix (above) found modeled throughput worse
  for ASS in a small minority of rows (12/2,304 post-Phase-E-fix, down from 65/2,304), mostly
  concentrated in `predictor=oracle`. No drive-contention term is modeled (a wasted prefetch is treated as
  free beyond its own bytes), which is optimistic and, if anything, understates ASS's real
  overhead. Not measured hardware throughput.

## Overnight results (2026-09-22)

Source: `results/overnight/FINAL_RESULTS.md`, `bench/SUMMARY.tsv`, `ram4g/SUMMARY.txt`. Measured on the RTX 3090 host; forced identical 4,096-token
output for all systems. Exactness gates: Q4_K_M tokens + logits hashes identical to full-GPU. Bias mode is approximate (perplexity checked, no task benchmark).
The "48 GB-class" row is an all-hot proxy, not a real card.

## Popularity-tiered expert precision (2026-09-24)

Source: `docs/SPEC_IDEA1_POPULARITY_TIERED_PRECISION.md` rev 2, `scripts/popularity_to_layers.py`,
`scripts/build_tiered_model.sh`, `results/tiered/`. Measured on the RTX 3090 host.
`evidence_class: measured_rtx3090` throughout except where noted.

Built `Qwen3-30B-A3B-tiered-popk24.gguf` (24,061 MiB, down from the Q8_0 source's 30,973 MiB):
24 layers kept at Q8_0, 24 downgraded to Q4_K, selected by an entropy-based popularity signal
over the 8-prompt/3,276-token routing trace set (`results/rtx3090/traces/`) -- NOT the literal
activation-count formula in spec rev 1/2 section 3.1, which is degenerate (~8.0 for every layer)
on this model's fixed-top-8 router; see spec section 0 correction 4.

- **Gate 1 (byte accounting):** PASS. 25.23 GB total vs 32.48 GB Q8_0 source. Every expert tensor
  at its expected precision (0 mismatches across 144 tensors), verified via `gguf-py` against the
  built file.
- **Gate 2 (gate exactness):** PASS. All 48 router (`ffn_gate_inp`) tensors bit-exact (SHA-256)
  between the tiered model and the Q8_0 source.
- **Gate 3 (quality):** PASS. Teacher-forced against Q8_0 on the same 8 prompts / 3,072 tokens
  used for the existing Q4_K_M quality comparison: **96.97% top-1 agreement**, **perplexity ratio
  1.0073** (+0.73% vs Q8_0) -- both comfortably inside the <5%/>95% gates, and well inside
  full-Q4_K_M's already-accepted +2% PPL penalty.
- **Gate 4 (throughput, SSD-bound cold regime):** PASS, large margin. Same benchmark as the
  existing Q8_0 cold-cache baseline (`results/rtx3090/cold/q8-cache16g-direct-cap4g-long1536.json`):
  16 GiB cache, 4 GiB RAM cap, evicted page cache, direct I/O, 1,536-token decode, identical
  prompt. 3 repetitions, tightly reproducible:

  | | decode t/s (all) | decode t/s (2nd half) | hit rate | SSD bytes read |
  |---|---|---|---|---|
  | Q8_0 baseline | 26.30 | 26.44 | 97.4% | 76.69 GB |
  | tiered (rep 1) | 82.38 | 107.47 | 99.1% | 20.04 GB |
  | tiered (rep 2) | 83.67 | 110.64 | 99.1% | 20.04 GB |
  | tiered (rep 3) | 83.71 | 110.19 | 99.1% | 20.04 GB |

  **~3.1-3.2x decode throughput, ~74% less SSD traffic, higher hit rate**, matching the mechanism
  predicted in spec section 4.5: shrinking the cold layers to Q4_K raised the shared cache slot
  budget for every tensor (13,392 slots vs the baseline's 10,224), so more of the *whole* model's
  working set fits in the 16 GiB cache, not just the downgraded layers' own footprint.

### Popularity-based vs arbitrary layer selection (2026-09-24, same day)

Closes the "not yet done" gap above: built a byte-matched, layer-count-matched **control**
model, `Qwen3-30B-A3B-arbitrary-q8x24-q4x24.gguf` -- identical K=24 Q8_0/Q4_K split, same
24,061.40 MiB output size, but layers 0-23 kept hot and 24-47 downgraded (the same boundary
`make_synthetic_model.sh` uses), instead of the entropy-based popularity selection. Same 8
quality prompts, same cold-cache throughput benchmark (3 reps), same reference tokens.

| | decode t/s (all, mean of 3) | stdev | hit rate | SSD bytes | top-1 agreement | PPL ratio |
|---|---|---|---|---|---|---|
| Q8_0 baseline | 26.30 | -- | 97.4% | 76.69 GB | -- | -- |
| arbitrary split (layers 0-23) | 65.85 | 1.49 | 98.9% | 27.19 GB | 97.98% | 1.0071 |
| popularity split (entropy, K=24) | 83.25 | 0.76 | 99.1% | 20.04 GB | 96.97% | 1.0073 |

**Throughput: popularity selection wins clearly.** 83.25 vs 65.85 t/s, a **1.26x** speedup over
the arbitrary split with the *same* total model size and *same* cache slot budget (13,392 slots
for both -- the difference is entirely which 24 layers were chosen). Both configurations beat
the Q8_0 baseline by a wide margin (3.17x and 2.50x respectively), confirming the core "downgrade
cold layers" idea works regardless of selection method, but the popularity signal is what
delivers the *extra* margin the spec's hypothesis predicted.

**Quality: a wash, not a win.** Top-1 agreement and PPL ratio are statistically indistinguishable
between the two splits (differences under 1 percentage point / 0.0002 PPL-ratio, likely within
prompt-to-prompt noise on an 8-prompt set) -- if anything the arbitrary split is marginally
*better* on quality. The spec's per-layer entropy hypothesis (`popularity_to_layers.py`'s
`compute_popularity` docstring) is **not supported** by this quality data; it should not be
cited as a quality-improving mechanism. What the data does support is a throughput-selection
mechanism: whichever way the popularity signal is choosing layers, it produces a mix that the
runtime's LRU expert cache serves more efficiently for this specific decode workload than an
arbitrary contiguous block does. The precise causal reason (temporal locality of the one
benchmark prompt's expert access pattern vs. the trace-aggregate entropy statistic used to pick
layers) has not been isolated -- flagged as a follow-up, not resolved here.

Artifacts: `results/tiered/layers.summary.json` (popularity ranking, byte accounting),
`results/tiered/layers.tensor-type-file`, `results/tiered/quality/` (per-prompt logprobs +
`summary-tiered-vs-q8.json`), `results/tiered/cold/` (3 cold-run reps) -- popularity split.
`results/tiered/arbitrary/` -- arbitrary-split control (tensor-type-file, quality/, cold/).

## ASS confidence-gated speculative prefetch: real C++ port (2026-09-25)

Source: `docs/ASS_CPP_PORT_PROGRESS.md`, `third_party/llama.cpp/ggml/src/ggml-backend-expert-scheduler.h`,
`ggml/src/ggml-backend.cpp` (`ggml_backend_sched_expert_cache_spec_*`), `results/ass/cold/`.
Measured on the RTX 3090 host, `evidence_class: measured_rtx3090` throughout. This closes the
gap flagged in `docs/PRIOR_ART_AND_NEXT_STEPS.md` section 5: the `AdaptiveSpeculativeScheduler`
existed only as a Python simulation before this; it is now a real, working, measured C++ feature
in the classic/direct-io expert cache tier, gated behind `--moe-expert-scheduler {off,ass}`
(default off, verified bit-exact no-op).

**Correctness (bit-exact, with prefetch genuinely active, not the trivial off-vs-off case):**
256-token greedy decode, 4 GiB cache (deliberately small, to maximize eviction-path exercise),
`off` vs `ass` diffed via `scripts/compare_verify.py`: 23,574 speculative reads issued / 23,571
committed during the `ass` run, and **tokens and logits hashes identical across all 256 steps**.
Confirms the additive-only design (prefetch only ever warms extra cache slots; it never changes
which expert gets computed for a step the model actually needs) held up under real, heavy
exercise. `test-expert-cache`'s full suite (3 pre-existing cache-bookkeeping tests + 4 new tests
verifying the ported confidence-gate math is bit-exact against the real Python
`ConfidencePredictor`/`plan_window` reference) also passes.

**Throughput, real target regime — a genuine regression, reported honestly.** Same protocol as
every other cold-SSD row in this ledger: 16 GiB cache, 4 GiB RAM cap, evicted page cache, direct
I/O, 1,536-token decode, `Qwen3-30B-A3B-Q8_0.gguf`, 3 reps each:

| | decode t/s (mean of 3) | stdev | hit rate (required path) | SSD bytes read | spec issued / committed / wasted |
|---|---|---|---|---|---|
| `--moe-expert-scheduler off` | 30.68 | 0.30 | 97.4% | 72.74 GB | 0 / 0 / 0 |
| `--moe-expert-scheduler ass` | 26.84 | 0.24 | 97.4% | 92.35 GB | 11,970 / 11,802 / 168 |

**~12.5% slower, ~27% more SSD traffic, and no improvement in the required path's own hit
rate** — if anything very slightly worse (required-path hits dropped by ~597, misses rose by the
same amount, out of ~1.79M accesses: `ass`'s eviction-based admission policy occasionally evicted
an expert that turned out to be needed again shortly after, a real and measured cost of allowing
speculative fills to evict, not just fill empty slots). The speculative mechanism itself works as
designed (98.6% of issued reads committed successfully, confirming the background worker +
re-verified commit logic is sound) — the extra ~19.6 GB of SSD reads is almost entirely
*genuinely wasted* bandwidth: candidates that were fetched but not turned into a required-path
hit before being evicted again or superseded.

**Root cause, diagnosed from the data, not guessed:** the ported confidence-gate's budget model
(faithfully reproducing `supervram/scheduler.py`'s Python design) measures its spending budget in
*compute time* (`compute_ms_per_layer`), implicitly assuming an overlap regime where I/O can be
hidden behind spare GPU compute (`docs/PLAN_ADAPTIVE.md` section 2.3's "overlap model"). But
SuperVRAM's actual target regime — the one every benchmark in this ledger measures — is
**SSD-bandwidth-bound**: 72-81% of decode time is literally SSD wait (see the evidence table at
the top of this file). In that regime there is no meaningful compute-time slack to hide I/O
behind; the scarce resource is SSD bandwidth itself, and every speculative byte directly competes
with the required path's own bytes on the same physical drive, regardless of which thread or
queue issues the read. A budget denominated in compute-time headroom structurally cannot see this
contention, so the gate keeps firing (in fact fires often: `default_confidence=0.5` clears the
`gate_threshold=0.3` floor by default for any never-observed candidate, and Qwen3-30B-A3B's
routing is high-entropy enough — see the tiered-precision entropy numbers above — that many
candidates never build up enough track record to be confidently refused) even though, in this
regime, firing is essentially always a net loss.

**Implication for future work, not attempted here:** the fix this diagnosis points to is
denominating the gate's budget in *spare SSD bandwidth* (bytes/sec beyond what the required path
is already consuming), not compute-time — a materially different design, not a parameter tweak,
and out of scope for this port per the approved plan's "smallest viable port first" principle.
Whether that redesign would actually help is an open, testable question, not assumed here.

Artifacts: `results/ass/cold/` (3 reps each, `off` and `ass`), `docs/ASS_CPP_PORT_PROGRESS.md`
(full design writeup, the bugs found while porting and while wiring into the real cache, and the
Phase 4 correctness methodology).

## Still pending

- Confidence-gated speculative prefetch **is now ported and measured** (see the "ASS confidence-gated speculative prefetch" section above) -- real, bit-exact, but a measured ~12.5% throughput regression in the SSD-bandwidth-bound target regime, root-caused to the gate's compute-time budget model not accounting for SSD bandwidth contention. Not a net win as shipped; the bandwidth-budget redesign that diagnosis points to is unbuilt. Double-buffered staging (a single pinned staging buffer exists), GDS device DMA, and llama-server `/metrics` cache stats remain unbuilt.
- Prompt-batch cache (v1 errors when `n_used > n_slots`).
- Full 720-run ablation matrix (prefetch/predictor axes, >= 5 reps, pp512), energy, Nsight overlap traces. The simulation-mode ablation harness (`scripts/run_ablations.py`) now has scheduler x gate x lookahead axes for ASS, but a full run has not been executed on this host, only a 15-run smoke test.
