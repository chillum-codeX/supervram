# SuperVRAM implementation status

## Implemented and tested now

- Pinned llama.cpp revision `ce8caa6e60a03093351d6016a818720e0d46f0fb` and Qwen3 MoE code map.
- `--moe-expert-storage resident|mmap|cache`, `--moe-expert-cache-size`, `--moe-expert-cache-policy lru|lfu`, and (patch 0004) `--moe-expert-direct-io`, `--moe-expert-io-threads N`, `--moe-expert-staging-mib N`, `--moe-expert-trace FILE`. With `--moe-expert-direct-io` only the dense weights (815 MiB of a 17.3 GiB Q4_K_M file) end up in the page cache; on the mmap path 9.5 GiB do (standard `llama-completion`, both runs after evicting the file).
- Demand-paged Qwen3 MoE expert weights (`TENSOR_READ_LAZY` / lazy mode ON) stay in host/file mappings.
- Compact GPU expert cache in `ggml_backend_sched`: per-weight device slot buffer `{ne0,ne1,n_slots}`, LRU/LFU, logical-to-slot id remap, stats API.
- `svram-verify` greedy token-id / logits-hash gate.
- Hardware probe with a real `cuFileRead` into `cudaMalloc` (compatibility-mode host bounce on this machine).
- Standalone Python/C++ policy prototype, packer, simulator and roofline (synthetic / analytical).

## Measured on this RTX 3090 (`evidence_class: measured_rtx3090`)

See `results/rtx3090/SUMMARY.json`, `results/rtx3090/hardware-probe.json`, `results/rtx3090/verify/`, `results/rtx3090/baselines/`.

- GPU: GeForce RTX 3090 24 GiB, driver 595.91.07, CUDA 12.6, compute 8.6. BAR1 256 MiB. Idle PCIe gen2 x16 (max gen3). IOMMU off. `/home` is ext4 on WD SN550 NVMe. Sequential O_DIRECT about 2.0 GB/s.
- cuFile: driver opens, reads match pread, about 1.8 GB/s, `verdict: cufile_compat_mode_host_bounce`. `nvidia-fs` absent; this is not SSD-to-VRAM DMA.
- Tiny generated Qwen3 MoE: resident/mmap/cache token ids and logits hashes identical; cache hit rate 71% on a 2-token run.
- Qwen3-30B-A3B Q4_K_M: greedy 8 tokens identical across resident/mmap/cache. Resident vs cache logits hashes identical. Decode numbers: see the ablation section (the early short `svram-verify` timings were inflated by a timer bug).
- Qwen3-30B-A3B Q8_0 (exceeds VRAM): greedy 8 tokens identical across `--cpu-moe`, mmap, cache. Logits hashes differ (CPU vs CUDA kernels), as documented.
- llama-bench, 3 reps: Q4 pp512/tg128 resident 4273 / 171 t/s; `--n-cpu-moe 24` 591 / 57; mmap 341 / 34; cache decode-only tg128 47 t/s (8 GiB, 62 slots/tensor). Q8 `--n-cpu-moe 48` / mmap about 65 pp64 / 23 tg64; cache decode-only tg64 24.6 t/s (12 GiB, 53 slots/tensor).

## Cache-size ablation (`results/rtx3090/ablations-long/`, `evidence_class: measured_rtx3090`)

Scope: `svram-verify`, greedy decode of 256 tokens after a 7-token prompt, `-ub 1`, models warm in the page cache, **3 repetitions** per configuration (mean shown; min-max in `aggregate.json`, spreads are within about 3 %). Decode only: prompt batches with more distinct experts than slots are unsupported in v1. Cache-size x policy only: no prefetch or predictor axes, so this is not the 720-run matrix in `RTX3090_PROTOCOL.md`.

Baselines (same tool, same settings): Q4_K_M full GPU 195-200 t/s, mmap 30 t/s; Q8_0 `--cpu-moe` 22.7-24.4 t/s, mmap 23.9 t/s.

| Cache (GiB) | Q4 LRU / LFU decode t/s | Q4 hit rate | Q8 LRU / LFU decode t/s | Q8 hit rate |
|---|---|---|---|---|
| 4  | 18.3 / 17.1 | 72.1 / 69.3 % | 8.8 / 8.5 | 59.0 / 57.2 % |
| 8  | 42.4 / 36.9 | 91.3 / 89.4 % | 15.1 / 14.2 | 78.3 / 76.8 % |
| 12 | 60.9 / 60.6 | 95.1 / 95.0 % | 26.4 / 26.1 | 89.2 / 89.1 % |
| 16 | 63.0 / 63.1 | 95.4 / 95.4 % | 40.2 / 39.7 | 94.0 / 93.9 % |
| 20 | 63.5 / 63.0 | 95.4 / 95.4 % | failed (CUDA OOM) | - |

- **Q8_0 (does not fit in VRAM):** with a 16 GiB cache the cache path decodes at about 40 t/s against 22.7-24.4 t/s for `--cpu-moe`, roughly 1.7x, at a 94 % hit rate. Below 12 GiB it is slower than `--cpu-moe`. `llama-bench` with a 12 GiB cache (24.6 t/s) agrees with the 12 GiB row here (26.4 t/s).
- **Q4_K_M (fits in VRAM):** in this 256-token sweep the cache plateaus at about 63 t/s with 95 % hits and zero evictions from 12 GiB up. **Superseded:** that plateau included cache fill and predates the one-sync-per-layer change; see the Q4_K_M section below (about 106-127 t/s warm).
- **Policy:** LRU beats LFU when the cache is small (8 GiB Q4: 42.4 vs 36.9 t/s, ranges do not overlap); they are equal once the working set fits.
- Q8_0 at 20 GiB fails with a genuine `cudaMalloc` OOM (`results/rtx3090/ablations-long/q8/OOM-20480MiB-excerpt.txt`); the failures are in the manifests.
- **Correctness (256 tokens, `-ub 1`):** the Q4 cache output equals full-GPU resident execution bit for bit (256/256 logits hashes and tokens). CPU-expert paths (Q4 mmap, Q8 `--cpu-moe`) differ from CUDA from step 0 (different kernels), and tokens diverge at step 5 (Q4) / 16 (Q8). Q8 has no GPU-resident reference because it does not fit.
- **Timer bug:** the first `svram-verify` timings (including the earlier 32-token, 1-rep sweep in `results/rtx3090/ablations/`, `results/rtx3090/verify/` and `SUMMARY.json`) stopped the clock before the GPU finished, inflating GPU-heavy modes (resident showed 279-3600 t/s). Fixed in `src/svram_verify.cpp`. Those older `decode_tps` values are superseded; their hit rates and token/hash results are unaffected. `llama-bench` numbers were never affected.

## Cold-SSD path: is the SSD really the backing tier? (`results/rtx3090/cold/`, `evidence_class: measured_rtx3090`)

Why this exists: the ablation above ran with both models fully resident in the Linux page cache (`fincore`: 30.3 GiB and 17.3 GiB resident), so it measured GPU + RAM, not GPU + SSD. ElasticVRAM notes sec. 35 says that must not be presented as SSD-backed. `scripts/cold_run.py` evicts the model file from the page cache (`posix_fadvise DONTNEED`), optionally caps RAM with a `systemd-run --user` cgroup (`MemoryMax`, `MemorySwapMax=0`), and records SSD bytes read (`/sys/block/nvme0n1/stat`), peak RAM and page-cache residency afterwards.

Q8_0 (30.3 GiB, does not fit VRAM), 16 GiB GPU cache, LRU, `-ub 1`, 256 tokens, cold start:

| Expert source | RAM cap | Decode t/s | SSD read | Expert data in RAM | Peak cgroup |
|---|---|---|---|---|---|
| mmap page faults (patch 0001/0002) | none | 2.0 (3.5 second half) | 22 GiB at ~0.12 GB/s | 22 GiB page cache | 23.6 GiB |
| mmap page faults | 4 GiB | 0.55 (only 32 tokens, cold-start dominated) | 14 GiB at ~0.12 GB/s | capped at 3.7 GiB | 4.0 GiB |
| **O_DIRECT + pinned staging (patch 0003, `SVRAM_DIRECT_IO=1`)** | 4 GiB | **13.8 / 13.9 / 13.9 (3 reps; second half 14.5)** | 29.7 GiB, none cached | 0 (1.3 GiB page cache is the dense non-expert weights, loaded once) | 1.7 GiB |

- The warm-page-cache Q8 number (about 40 t/s) is a GPU + RAM result. The honest GPU + SSD number on this machine today is about 14 t/s.
- Direct I/O correctness: Q4 cache with direct reads matches full-GPU resident bit for bit (64/64 tokens and logits hashes).
- Time is SSD-bound: 16.7 s of the 18.5 s decode is spent waiting for reads (`io_ms`). It reads about 119 MB per token (94 % hit rate, ~23 missed experts of ~5 MB), matching the arithmetic estimate.
- SSD ceiling: `scripts/ssd_expert_read_bench.py` measures the WD SN550 at about 2.4 GB/s for expert-sized (1.6 MB) O_DIRECT reads at queue depth >= 8 (1.6 GB/s at QD1). Bandwidth-only ceiling for Q8_0: about 21 t/s at 94 % hits, 42 t/s at 97 %, 62 t/s at 98 %, 125 t/s at 99 %. The direct path reaches about 1.8 GB/s during waits, roughly 70 % of that ceiling.
- Estimated parity (not measured): a native 48 GB card at 3090-class bandwidth would decode Q8_0 at roughly 100 t/s (about twice Q4's 195-200 t/s). 14 t/s is therefore about 0.14x, below the notes' 0.5x "basic viability" line. Reaching 0.5x needs a hit rate near 97.5 % or better, fewer bytes per expert, or more SSD bandwidth.

## Cache policy headroom and steady state (`results/rtx3090/traces/`, `results/rtx3090/cold/*long1536*`)

Method: `SVRAM_TRACE=<file>` (patch 0003) records the distinct experts each layer picks for every token. Eight varied prompts (code, science, story, translation, math, history, engineering, literature) x 384 tokens on Q8_0 (3,276 tokens, 48 layers) were replayed offline by `scripts/analyze_policies.py`, including Belady's offline-optimal policy as the upper bound. Hit rate is per distinct expert per (token, layer), like the runtime stats; per-layer pools of 71 slots = 16 GiB.

| Policy (71 slots/layer, each prompt from a cold cache) | Hit rate |
|---|---|
| LRU (current) | 96.28 % |
| LFU (current) / LFU with decay / SLRU | 96.08 / 96.36 / 96.29 % |
| **Optimal (Belady), per-layer pools** | **97.10 %** |
| Optimal, one shared pool across layers | 97.31 % |

- **Replacement policy is already within about 0.8 points of optimal.** Even a perfect policy cuts misses by only about 25 %. Smarter eviction cannot reach the ~97.5-98 % hit rate that 0.5x parity needs on this SSD. (A "pinned hot experts" variant looked better at 89 slots but gets its hot set loaded for free, so that comparison is not fair and is not claimed.)
- **A persistent cache matters more than the policy.** Replaying all eight prompts back to back with a warm cache gives 97.3 % (LRU). A real long cold-start run agrees: 1,536 tokens, 4 GiB RAM cap, direct I/O: 97.4 % hits and **26.3 t/s overall, 27.5 t/s from token 256 on** (first 256 tokens: 21.7 t/s). The 13.8 t/s of the short cold runs is mostly filling an empty cache.
- **Prompt matters:** decode speed on the eight prompts ranged 15-48 t/s (hit rate 94-97.7 %) with direct I/O; the single-prompt figures elsewhere in this document are not typical.
- **Sharing loads across tokens does not help.** Verifying k consecutive tokens together (best case of speculative decoding, all accepted) reduces missed experts per token only from 10.2 to 9.6 at k=8 (-6 %). Running k independent requests in lock-step is worse per token (23 misses/token at k=4, 36 at k=8, hit rate down to 87 %) because they evict each other's experts. Trace-based estimate, LRU, not measured end to end.
- **Time split in steady state (1,536-token run):** 42.6 s of 58.4 s decode is SSD wait (73 %), the other ~10 ms/token is per-layer synchronization and compute. Overlapping reads with compute could therefore gain up to about 1.35x here (the earlier "at most ~10 %" applied only to the cold 256-token runs).
- **SSD bandwidth is the binding limit.** At the steady-state 97.3 % hit rate a token needs about 51 MB from the SSD. That is at most 47 t/s at this drive's 2.4 GB/s, about 68 t/s at 3.5 GB/s and about 137 t/s at 7 GB/s (bandwidth-only arithmetic; a native-card estimate of ~100 t/s caps the useful range). The steady-state 27.5 t/s is about 0.27x of that unmeasured native estimate.

## Layer-batched reads and one sync per layer (patch 0003, `results/rtx3090/cold/*long1536-batched*`)

Finding: a cached `MUL_MAT_ID` is forced to be the first node of its own scheduler split, so gate, up and down of a layer are three consecutive splits. The first version therefore read each weight's missed experts separately (queue depth about 1, about 1 ms per 1.6 MB read) and synchronized the GPU stream once per weight. The scheduler now looks ahead across the following splits that share the same routing ids, plans all three weights together, issues one batch of O_DIRECT reads and synchronizes once per layer.

Same 1,536-token cold-start run as above (Q8_0, 16 GiB cache, direct I/O, 4 GiB RAM cap):

| | Overall t/s | Steady state (from token 256) | SSD wait | SSD read rate while waiting |
|---|---|---|---|---|
| per-weight reads | 26.3 | 27.5 | 42.6 s | 1.8 GB/s |
| **layer-batched reads** | **30.5 / 30.4 (2 reps)** | **31.7-31.9** | **36.1 s** | **2.1 GB/s** |

- Output is unchanged: tokens and logits hashes are identical to the per-weight version; Q4 with direct reads and with mmap still matches full-GPU execution bit for bit; `test-expert-cache` passes.
- 72 % of decode time is now SSD wait (36.1 of 50.4 s); the remaining ~9 ms/token is GPU compute and launch overhead. The read rate is within about 12 % of the drive's measured 2.4 GB/s ceiling.
- Estimated parity (native ~100 t/s is an unmeasured estimate): 31.8 t/s is about 0.32x.
- Overlapping the remaining reads with compute would need a prediction of the missed experts one layer ahead. Because the SSD wait (about 23 ms/token) exceeds the compute (about 9 ms/token), the best case is about 41 t/s, and only with a perfect predictor.

## Smaller experts: Q4_K_M vs Q8_0 through the SSD path (`results/rtx3090/quality/`, `results/rtx3090/cold/q4-*`)

Quality (teacher-forced, `scripts/compare_quality.py`): Q8_0 generated 384 tokens for each of 8 varied prompts; Q4_K_M was fed exactly those tokens (`svram-verify --force-tokens`) and we recorded its own argmax and the probability it gives Q8's tokens. The tool is checked: Q8 scoring its own tokens gives 100 % agreement and identical log-probabilities.

| 3,072 tokens, 8 prompts | Q4_K_M vs Q8_0 |
|---|---|
| Top-1 agreement with Q8's next token | **96.0 %** (per prompt 93.0-97.7 %) |
| Mean log-prob of Q8's tokens: Q8 / Q4 | -0.183 / -0.204 nats |
| Perplexity ratio on Q8's text (Q4 / Q8) | **1.021** (+2.1 %) |

Caveats: the reference is Q8_0, not ground truth, and the text is Q8-generated (which favors Q8). Stock Q4_K_M also quantizes the dense weights, so this is not an experts-only change. No benchmark accuracy or ground-truth perplexity has been measured.

Ground-truth check (`results/rtx3090/quality/ground-truth-ppl-q4-vs-q8.json`, `ppl-*.log`): `llama-perplexity`, n_ctx 512, 60 chunks (about 30.7k tokens) of human-written text: the prose of llama.cpp's `docs/*.md` (255,802 characters, code blocks and tables removed; a technical-documentation domain, not a standard benchmark like WikiText). Q8_0 (experts on the CPU for this scoring run) PPL 9.073 +/- 0.224; Q4_K_M PPL 9.253 +/- 0.231. The per-model error bars are wide because both models scored the same tokens; the paired per-chunk difference is 0.0196 +/- 0.0026 nats/token, i.e. **Q4_K_M perplexity is 1.020x Q8_0's (95 % CI 1.015-1.025), worse in 48 of 60 chunks**. This agrees with the +2.1 % from the teacher-forced test and does not rely on Q8 as the reference. Limits: one corpus and domain, no full-precision (F16/BF16) reference, no downstream-task accuracy.

Speed, same cold-start protocol (evicted, direct I/O, 4 GiB RAM cap, 1,536 tokens, same prompt), steady state from token 256:

| Experts | GPU cache | Fraction of experts cached | Hit rate | SSD read | Decode t/s |
|---|---|---|---|---|---|
| Q8_0 | 16 GiB | ~55 % | 97.4 % | 76.7 GB | 31.8 |
| **Q4_K_M** | 9 GiB | ~55 % | 97.0 % | 52.2 GB | **41.4** |
| Q4_K_M | 16 GiB (everything fits) | ~92 % | 99.2 % | 14.1 GB (read once) | **126.8** |

- **Smaller experts read 32 % fewer bytes at the same cached fraction and decode about 30 % faster (41.4 vs 31.8 t/s).** The bandwidth-only argument (fewer bytes per miss) holds; the hit rate is about the same.
- **Parity does not improve at equal cached fraction.** Q4_K_M running fully in VRAM is 195-200 t/s, so 41.4 t/s is about 0.21x of the same model's native speed; Q8_0 was about 0.32x of an unmeasured native estimate. Quantizing raises absolute speed and lets more of the model fit; it does not by itself close the ratio while ~45 % of the experts come from a 2.4 GB/s SSD.
- **Correction:** earlier text said the Q4 cache "plateaus at about 63 t/s ... per-step overhead". That figure was a 256-token average including cache fill and predates the one-sync-per-layer change. Warm, with everything resident, the current build decodes about 106 t/s (mmap path, tokens 512+) to 127 t/s (direct path), i.e. about 0.64x of full-GPU residency (195-200 t/s); the remaining cache-path cost is about 2.7 ms/token.

## Larger model: 43 GiB (46 GB) through the cold SSD path (`results/rtx3090/cold/big46g-*`)

ElasticVRAM notes Experiment 3/8 ask for a model in the 35-48 GB range. No such Qwen3 MoE is on this machine and nothing was downloaded, so a **synthetic** one was built locally: `llama-quantize` from the Q8_0 GGUF with the experts of layers 0-23 stored as F16 (dequantized Q8 values) and layers 24-47 plus all dense weights left as Q8_0 (`Qwen3-30B-A3B-synthetic-F16x24-Q8x24.gguf`, 42.9 GiB). Its size and read pattern match a real model of that size; **its quality is only Q8-level, and it is not a real F16 model**, so use it for throughput and capacity, not quality.

Cold start, direct I/O, 4 GiB RAM cap, 16 GiB GPU cache, `-ub 1`, 1,536 tokens, same prompt as the earlier long runs:

| Model | Size | Cache slots per tensor (of 128) | Hit rate | SSD read per token | Decode t/s | SSD wait |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B Q8_0 | 30.3 GiB | 71 (55 %) | 97.4 % | 50 MB | 31.8 (steady) | 72 % |
| **synthetic F16x24 + Q8x24** | **42.9 GiB** | **49 (38 %)** | **90.9 %** | **272 MB** | **6.6** (6.1-7.0) | **81 %** (2.2 GB/s) |

- Peak RAM (capped cgroup) 1.69 GiB; only 1.33 GiB of the model file is in the page cache (dense weights). No expert data is held in RAM.
- Correctness: cache output is bit-identical (64 tokens, tokens and logits hashes) across 8 vs 16 GiB caches and direct vs mmap reads; the text is coherent and factually right. CPU-expert and GPU paths diverge at token 3 (F16 rounding), as with Q8.
- Going from 30 to 43 GiB with the same 16 GiB cache drops the cached fraction from 55 % to 38 %, misses rise 3.5x and bytes per miss 1.5x, so SSD traffic per token grows 5.4x and decode falls 4.8x. The SSD is saturated (2.2 GB/s) in both cases.
- Estimated parity (unmeasured; a native card would read about 4.1 GB per token here vs about 3.2 GB for Q8, so roughly 78 t/s): about 0.08x. Parity falls quickly as the model outgrows the cache: about 0.32x at 30 GiB, about 0.08x at 43 GiB.
- Consequence for the notes' hypothesis: "the active working set fits in VRAM" holds only partly. The routing working set of this model at 16 GiB is well above the cache once the cached fraction drops below about 55 %, so hit rate, not policy or overlap, is what decides throughput.

Measurement fix: `scripts/cold_run.py` sampled the launching session's cgroup (47 GiB) before `systemd-run` moved the process into the capped scope, once producing a bogus peak; it now ignores samples from outside the scope. The 43 GiB run was repeated with the fixed script (6.60 vs 6.61 t/s).

## Drives on this machine

| Device | Model | Link | State | Measured |
|---|---|---|---|---|
| nvme0n1 | WD SN550 1 TB (`/home`, `/`, swap) | PCIe 3.0 x4 | mounted ext4 | ~2.4 GB/s for expert-sized O_DIRECT reads |
| nvme1n1 | Intel SSDPEKNW010T8 1 TB | PCIe 3.0 x4 | not mounted, NTFS (looks like a Windows drive), `root:disk` only | not measured (no read permission without root) |
| sda | Seagate 2 TB HDD | SATA | NTFS | not tested (rotational) |

There is no PCIe 4 drive here, so the faster-drive test could not be run. The Intel drive could only add bandwidth in parallel; whether it does depends on how both are wired to the CPU. A read-only benchmark (`scripts/ssd_expert_read_bench.py` now accepts block devices and several targets) would answer that, but needs root to open `/dev/nvme1n1`; it was not run.

## Prefetch feasibility: measured before building (`scripts/simulate_prefetch.py`, `scripts/prefetch_predictor_eval.py`)

Prefetching means reading the experts a layer will miss while the GPU is still computing earlier layers. Before touching the model graph, two questions were tested on the 3,276-token Q8_0 routing traces (16 GiB cache, per-layer LRU).

**1. How much could any prefetch gain?** A timing model of the layer pipeline (GPU 0.19 ms/layer, one SSD serving missed experts at 2.4 ms each; it reproduces the measured speed within 7 %: 29.6 vs 31.8 t/s) with an oracle that knows every miss D layers ahead and wastes no reads:

| Lookahead D (layers) | Speedup, this drive (2.4 GB/s) | at ~3.5 GB/s | at ~7 GB/s |
|---|---|---|---|
| 1 | +4 % | +5 % | +8 % |
| 4 | +11 % | +15 % | +23 % |
| 8 | +17 % | +23 % | +31 % |
| 48 (a whole token ahead) | +30 % | +37 % | +43 % |

A missed expert takes about 2.4 ms to read but a whole layer computes in about 0.19 ms, so a short lookahead hides almost nothing; the reads must start many layers early.

**2. Can routing history predict the misses?** An online cross-layer co-occurrence predictor (which experts a layer picks, given the experts chosen D layers earlier for the same token), evaluated only on real misses:

| Lookahead | Top-1 guess is right | Most confident 0.1 % of guesses right | Always prefetching top-4: recall / wrong reads per useful read |
|---|---|---|---|
| 1 layer | 2.0 % | 12.5 % | 34 % / 69 |
| 8 layers | 1.4 % | 13.0 % | 28 % / 95 |

Misses are the rarely used experts (about 0.2 per layer per token), so they are close to unpredictable from routing history. Because the SSD is already the bottleneck, every wrong read costs a full 2.4 ms: in the timing model, a predictor with 50 % recall and just 0.5 wrong reads per useful one is already **slower** than no prefetch (0.96x), and one with 1.0 wrong per useful is 0.82x. No confidence threshold reached even a 1-in-2 hit rate.

**Conclusion:** history-based prefetch would lose speed. The only credible predictor is the model's own router applied early to the hidden state (a "pre-gated" design), which needs extra graph nodes and host read-backs in `qwen3moe`; even a perfect version would gain about 4-11 % at 1-4 layers of lookahead on this drive (about 8-23 % on a 7 GB/s drive). It was therefore not built. Prefetch becomes more attractive only with a much faster SSD.

## Long context: 32,000-token prompt + 4,096-token output, VRAM + RAM + SSD (`results/rtx3090/longctx/`)

Goal (clarified): run the same weights at the maximum context using any combination of VRAM, RAM and SSD, for people without a high-end GPU. So RAM is a legitimate tier; the earlier "RAM must not hold the model" rule now applies only to the low-RAM row below. Q8_0 (30 GiB) on the 24 GB RTX 3090, prompt = first 32,000 tokens of the docs prose plus a one-line instruction, context 36,352 tokens (KV cache 3.4 GiB of VRAM), greedy decoding with end-of-text ignored (`--ignore-eos`). `scripts/run_longctx_compare.sh`, `scripts/summarize_longctx.py`.

**Patch 0005** makes this possible: (1) prompt batches larger than 4 tokens step aside from the slot cache and use llama.cpp's regular whole-expert offload (a big batch touches nearly every expert, so a small cache cannot serve it, and the cache errors when a batch needs more experts than slots); (2) with `--moe-expert-direct-io` that whole-expert copy is streamed from the model file with large parallel O_DIRECT reads into double-buffered pinned staging, so it runs at SSD speed instead of at page-fault speed.

| Setup (same Q8_0 weights, 32,000 in) | Prefill | Decode, first 2,048 output tokens | Prompt + 2,048 tokens | RAM used |
|---|---|---|---|---|
| Stock llama.cpp, experts of 25 layers on the CPU (23 layers' experts in VRAM), batch 4096 | **22.6 s** (1,418 t/s) | 33.7 t/s | **83 s** | model in RAM (30 GiB) |
| Tiered: 14 GiB VRAM expert cache + RAM, batch 4096 | 34.7 s (922 t/s) | **40.5 t/s** | 85 s | model in RAM (30 GiB) |
| Tiered: 12 GiB VRAM cache + RAM | 34.7 s | 31.3 t/s | 100 s | model in RAM |
| **Tiered: 14 GiB VRAM cache + SSD, RAM capped at 4 GiB** (direct I/O) | 123 s (261 t/s) | 15.2 t/s | 257 s | **2.2 GiB peak**, 1.45 GiB of the file in the page cache |

- Prefill scales with batch size for both systems (stock: 56.5 s at 1024, 34.2 s at 2048, 22.5 s at 4096; tiered: 97.9 / 56.6 / 34.6 s). Whole-expert copies over PCIe (about 9.6 GB/s) dominate the tiered system's prefill because its VRAM holds cache slots, not resident layers.
- **Caveat on decode speed: repetition artifact.** Forcing greedy decoding to continue past the natural end of text makes the output degenerate into loops from about token 2,048 (80 % then 99.8 % repeated 8-grams; 2 distinct tokens in a 60-token window at token 3,500). Looping text is trivially cacheable, so tiered decode over all 4,096 tokens reads 42.5 / 51.0 t/s (12 / 14 GiB) and reaches 63-76 t/s in the last thousand tokens. Those figures are inflated and are not claimed; the table uses the first 2,048 tokens (6 % repeated 8-grams or less). Stock llama.cpp is unaffected (33.6 t/s over all 4,096).
- **Correction to an earlier statement:** I first said the tiered system decodes 25-35 % faster than stock; that used the inflated segments. On natural text the best tiered setting is about 20 % faster on decode and about even on the whole request. **When the model fits in VRAM + RAM, plain llama.cpp with a static layer split is already good; the tiered system roughly matches it there.**
- **Where the new work matters:** machines with too little RAM for the model (or models bigger than VRAM + RAM). With 4 GiB of RAM the 30 GiB model still handles a 32,000-token prompt in 2 minutes and decodes at 15 t/s. Before patch 0005 the same setting was projected at about 25 minutes of prefill (0.17 GB/s page-fault reads) and, before the batch bypass, about 45 minutes (6-token batches). Streaming vs mmap output is bit-identical (3,000-token prompt: tokens and logits hashes match).
- The 4 GiB decode rate (15.2 t/s) is below the 31.8 t/s of the earlier short-context run because the cache is 14 GiB (61 slots) instead of 16 GiB (71 slots) to leave room for the KV cache and the prefill batch buffers, the cache starts empty after a prompt (prompt batches do not fill it), and attention over 32,000 keys costs extra. Hit rate 95.2 %.
- Single runs, one prompt, Q8_0 only. Expected next lever for mid-size RAM (for example 16 GB): an explicit RAM tier between the VRAM cache and the SSD; not built yet.

## Zero-copy expert tier, stage 1 (patch 0007) - measured on this machine

**Idea.** Today every layer stops the GPU, sends the routing result to the CPU, and copies the missed experts over PCIe (about 10-11 ms of a 25 ms decode token is spent in that host loading path). New design: all experts live in pinned host RAM; the GPU kernel gets a per-expert pointer table and reads a missed expert straight from RAM over PCIe, with no copy and no per-layer CPU round trip; hot experts are promoted into VRAM slots in the background (stage 2). Correctness (any expert can always run from RAM) is decoupled from performance (VRAM residency).

**Stage 1 (built, patch 0007, 95 lines).** `mmvq` reads each expert's weights through `fusion.x_ptrs[expert_id]` (and `gate_ptrs` for the fused gate/up kernel); in `--moe-expert-zerocopy` mode the scheduler keeps the router's original ids, attaches the table as an extra source of the `MUL_MAT_ID` node, skips the per-weight forced split and the load step, and initializes the table with each expert's pinned-RAM address. Experts are placed in pinned RAM with `-ot "\.ffn_(up|down|gate)_exps\.=CUDA_Host"` (now accepted) or `svram-verify --pinned-moe`. Single-token decode only; batches of 2+ tokens use the regular whole-expert path.

| Check | Result |
|---|---|
| Exactness: Q4_K_M, all experts in pinned RAM and read in place by the GPU kernel vs ordinary all-in-VRAM execution, 64 tokens | tokens **identical**, logits hashes **identical** |
| Unit tests (`test-expert-cache`), patch chain 0001-0007 vs built tree | pass / identical |
| Decode with all experts read from RAM, Q4_K_M | 6.09 t/s = 164 ms/token, effective **6.7 GB/s** over PCIe |
| Decode with all experts read from RAM, Q8_0 | 3.99 t/s = 250 ms/token, effective **7.8 GB/s** over PCIe |
| Ceilings measured with standalone CUDA programs (`scripts/microbench/`) | pinned copy 11.8-12.1 GB/s, kernel reading pinned RAM 10.8-11.5 GB/s, pageable copy 6.6-8.3 GB/s, at least 60 GiB pinnable |

- The real `mmvq` kernel therefore reaches 60-70 % of the PCIe ceiling when reading from RAM (its loads are narrower than the 16-byte loads of the microbenchmark); a wider-load variant could recover part of that.
- Also observed: with the experts in pinned RAM but the flag off, the scheduler runs them on the CPU (25 t/s, GPU 6-28 % busy, PCIe about 30 MB/s): the CPU streams experts from RAM at about 48 GB/s. This is the stock `--cpu-moe` behavior, not part of this work.
- **Projection (not yet measured):** with a 14 GiB VRAM cache at the 97 % hit rate seen in long-context runs, about 10 missed experts per token at 7.8 GB/s cost about 7 ms, on top of about 9 ms of compute (including the 32k-token KV read), for roughly 60 t/s decode, against 40.5 t/s (current tiered) and 33.7 t/s (plain llama.cpp). Stage 2 (background promotion into VRAM) is needed to confirm this.

## Zero-copy expert tier, stage 2 (patch 0008): background promotion into VRAM

**Built.** The `mmvq` kernel counts each expert's use (one atomic per expert per call). At every token boundary the scheduler (1) reads all counters with one copy, (2) commits the promotions whose copies finished (flips their table entries to the VRAM slot only after the copy is done, so the kernel never reads a half-copied expert), and (3) copies the most used experts that were served from RAM into VRAM slots on a second CUDA stream, overlapped with the next token's compute. Admission control: a candidate replaces the lowest-scoring resident expert only if its decayed use score is higher (`SVRAM_SCORE_DECAY`, default 0.97, best 0.99; `SVRAM_PROMOTE_PER_STEP`, default 48).

**Exactness.** Q4_K_M, 384-512 tokens, promotion active in every configuration tried: tokens and logits hashes identical to full-GPU execution (no races).

**Q4_K_M, 8 GiB cache, short context (steady state, tokens 256-512):** promote every miss 12/step: 14.6 t/s (cache still filling); 48/step: 39.8 t/s at 87 % hits; with admission control and decay 0.99: **50.5 t/s** (hit rate 86.7 %, copy traffic 24.8 GB vs 36-39 GB), against 42.4 t/s for the earlier VRAM-cache design at the same size.

**The target workload (Q8_0, 32,000-token prompt, 14 GiB cache, first 2,048 output tokens, `results/rtx3090/longctx/zerocopy-cache14g-*`):**

| System | Prefill | Decode | Prompt + 2,048 tokens |
|---|---|---|---|
| Plain llama.cpp, static split (25 layers' experts on CPU) | 22.6 s | 33.7 t/s | **83 s** |
| Previous design: VRAM cache + RAM, sync per layer | 34.7 s | **40.5 t/s** | 85 s |
| Zero-copy tier, stage 2 | 33.4 s | 32.7 t/s (0-512: 21.8, 512-1024: 37.7, 1024-2048: 39.9) | 96 s |

- **The projection of about 60 t/s was not reached.** Steady state (39.9 t/s) only matches the previous design; the cold cache costs the first 512 tokens (21.8 t/s), so the whole request is slower (96 s vs 83-85 s).
- Cache at the end: 94.5 % hit rate (62 slots per layer), 44 MB/token promoted, and the promotion step itself costs about **5.4 ms per token of host time**, out of a 25 ms token in steady state. Misses read in place at 7.8 GB/s and promotion copies share the same PCIe link, which is at or near saturation (read misses plus about 44 MB of copies per token).
- Where the projected time went: about 9 ms compute + about 6 ms miss reads = 15 ms was the model; the measured 25 ms adds about 5 ms promotion step and link contention.
- Identified next steps, none built: (1) take the promotion step off the critical path (helper thread; the blocking counter read and commit sync are its cost), (2) warm start (populate the cache from a usage profile before the first token), (3) a higher admission threshold to cut copy traffic, (4) a wider-load kernel variant to lift in-place reads from 7.8 toward 11 GB/s.
- Single run, one prompt, Q8_0.

## Remaining gaps

- Prompt ubatches that route more distinct experts than `n_slots` return a hard error (observed as llama-bench warmup `res = -3` at `-ub 16`). Use `-ub 1` or a larger cache for decode; prompt processing is not a v1 win.
- No prefetch (measured to be a net loss with history-based prediction and worth at most about 4-11 % with an ideal early-router predictor, see above), no GDS copy backend, no llama-server `/metrics` cache stats (stats are in the server task JSON only), and only the 20-run cache-size x policy ablation above, not the 720-run matrix (no prefetch/predictor axes, no repetitions).
- Direct I/O (`--moe-expert-direct-io`, Linux only; the `SVRAM_*` environment variables remain as a fallback) uses one pinned staging buffer with a synchronize before each reuse: no double buffering and no prefetch, so reads never overlap compute (reads of one layer are batched, see above). The default path is still mmap.
- CUDA graphs were disabled (`GGML_CUDA_DISABLE_GRAPHS=1`) for bring-up.
