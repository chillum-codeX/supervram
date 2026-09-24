# Prior art, honest novelty assessment, and a plan for the unproven parts

Date: 2026-09-24. Scope: a literature/prior-art check against three things built this session --
the VRAM+RAM+SSD expert cache (shipped, measured), the AdaptiveSpeculativeScheduler / ASS
(Python simulation only), and the popularity-tiered-precision spec (unbuilt). Searched live on
2026-09-24; not exhaustive, but covers the closest matches found.

**TL;DR:** none of the three ideas are unprecedented. The core cache lands in an active,
crowded space with academic systems claiming much larger speedups than our measured 1.7x
(different baselines/hardware, not apples-to-apples) and at least one llama.cpp-native
competitor. ASS's core idea (confidence-gated speculative expert prefetch) has a near-identical,
independently-published paper (SPICE) that appeared within days of when we built ASS, and it has
real multi-GPU hardware validation that ours does not. The tiered-precision spec's core idea
(importance/popularity-aware mixed-precision expert quantization) is an active academic subfield
with multiple 2025-2026 papers, one of which contradicts an assumption our spec leans on. What is
real and defensible: the core cache is *shipped, working code*, bit-exact-verified, beating
llama.cpp's own built-in technique by a measured amount on real consumer hardware -- that's not
nothing, it's just not "solved a problem nobody has solved."

---

## 1. Core VRAM+RAM+SSD expert cache (what's already built and shipped)

### Academic systems (peer-reviewed or preprint, not llama.cpp-specific)

| System | Venue/status | Claimed result | Notes |
|---|---|---|---|
| [MoE-Infinity](https://arxiv.org/abs/2401.14361) (Edinburgh) | 2024 preprint | 4-20x latency reduction, >8x cost reduction vs SOTA serving | Activation-aware expert prefetching + caching, sequence-level activation tracing. Targets *serving* (batched, datacenter), not single-user local inference. |
| [Fiddler](https://arxiv.org/abs/2402.07033) (UW/Tsinghua) | ICLR'25 | 8.2-10.1x over prior offloading work (Eliseev & Mazur 2023), 19.4-22.5x over DeepSpeed-MII | CPU-GPU orchestration; uses CPU *compute* (not just RAM as backing store) to reduce data movement. Different mechanism than a pure VRAM cache. |
| [Pre-gated MoE](https://arxiv.org/abs/2308.12066) (MSR-affiliated) | 2023-2024 | Throughput improvement + large VRAM reduction (no single headline multiplier found) | Predicts next block's expert selection to overlap migration with compute -- closer in spirit to our earlier "prefetch" investigation (which we measured as a net loss with history-based prediction) than to the cache itself. |

These are **not apples-to-apples comparisons** to our 1.7x: different frameworks (not llama.cpp),
different hardware classes, and largely optimized for *batched serving throughput*, not
single-request local decode on one consumer GPU. But they establish that "MoE expert offloading
beats naive baselines by a large multiple" is an active, published result, not new territory, and
that the *ceiling* other systems have found (4-20x) is well above what we've measured (1.7x over
`--cpu-moe`). Whether that gap is closable on a single 3090 with a real workload is untested here.

### llama.cpp-native prior art (the actually comparable set)

- **[`JigSawPT/moe-autopilot`](https://github.com/JigSawPT/moe-autopilot)** -- the closest thing
  found to our own cache, and worth taking seriously despite its size. Measured on a real consumer
  GPU (RTX 5090): **+26% decode (Qwen3-Coder-Next 80B), +31.6% (gpt-oss-120B), +10.3%
  (Qwen3.6-35B)**. Architecturally different from ours: it's a *static, load-time* hot/cold split
  from an offline profiler with **no runtime eviction policy** (the hot list is fixed until
  restart), versus our dynamic LRU/LFU cache that adapts as routing shifts. Its own numbers show a
  real tradeoff we don't have to make: ~10% *prefill* regression from the resident duplicate
  copies, break-even only past a ~2.3:1 output:prompt ratio. Early-stage: 2 stars, 9 commits, a
  separate fork (`aipc-hardening`), not upstreamed. Notably, the author also discloses heavy AI
  assistance building it -- we are not the only ones doing this with an LLM's help right now.
- **[`ggml-org/llama.cpp` issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757)** --
  an **open, unimplemented** feature request describing almost exactly our architecture: a
  three-tier cache (GPU VRAM slots / pinned CPU RAM / SSD-mmap) with a **pluggable eviction
  policy**, explicitly recommending SLRU as the default over plain LRU (matches our own policy
  ablation work finding LRU/LFU/SLRU all within ~1 point of each other and of Belady-optimal -- so
  the choice barely matters, which the issue doesn't seem to know yet). A cited proof-of-concept
  on an RTX PRO 2000 (8GB) reached 12-14 t/s vs 0.5-1 t/s uncached. This means **the llama.cpp
  community has been asking, unanswered, for the exact system we built and verified.** That's a
  much more specific and defensible claim than "novel": *"implements and ships the two-tier
  VRAM/RAM/SSD MoE expert cache with pluggable eviction that llama.cpp issue #20757 requests, with
  bit-exact correctness verification and measured throughput numbers on real hardware."*
- **[`dvmazur/mixtral-offloading`](https://github.com/dvmazur/mixtral-offloading)** -- an
  established community project (not llama.cpp) that already combines an LRU expert cache *and*
  mixed quantization (HQQ) in one system -- i.e., it already does a version of both our core cache
  *and* our unbuilt tiered-precision idea, together, for Mixtral specifically.

**Verdict for this piece:** real, working, correctly-verified engineering, sitting in a space with
both stronger academic results (different setting) and at least one llama.cpp-native alternative
with real numbers on comparable hardware. The most honest, checkable claim is the narrow one above
-- built and shipped the specific thing an open llama.cpp issue is asking for -- not "beat
everyone" or "solved an unsolved problem."

---

## 2. Confidence-gated speculative expert prefetch (relevant to the ASS scheduler, Phases A-E)

- **[SPICE](https://arxiv.org/abs/2608.21240)** ("Speculative Prefetching with Low-Rank Expert
  Surrogates and Heterogeneous Orchestration for MoE Inference Acceleration") -- **submitted 21
  Aug 2026, revised 7 Sep 2026, accepted ASP-DAC 2027.** This is close to a direct hit on ASS's
  core idea: a confidence-aware adaptive lookahead algorithm that prefetches high-confidence
  experts and falls back (a low-rank expert surrogate, or exact residual work on CPU) when
  confidence is low, instead of firing a wasted read -- the same "gate the speculative read on
  confidence" principle ASS's `AdaptiveSpeculativeScheduler` implements. **Measured on real
  hardware**, DeepSeek-V2-Lite and Qwen2-57B-A14B, **"across diverse GPU platforms," up to 3.12x
  speedup in TPOT**. This was published essentially the same week this session built and verified
  ASS in a Python simulator -- independent convergence on the same idea, not derivative in either
  direction, but SPICE got real multi-model, multi-GPU validation and we have none yet.
- **[SpecPrefetch](https://arxiv.org/pdf/2607.24787)**, **[SpecMD](https://arxiv.org/html/2602.03921v1)**
  ("A Comprehensive Study On Speculative Expert Prefetching") -- more work in the same space,
  suggesting this is an active subfield with multiple groups working on it concurrently in 2026,
  not a niche idea.
- **[Fate](https://arxiv.org/html/2502.12224v2)** (cross-layer gate signal for edge MoE inference)
  -- closer to the "cache-aware routing bias" feature already in the shipped cache (patch 0009)
  than to ASS specifically, but confirms cross-layer signal exploitation is also an active area.

**Verdict for this piece:** the core idea is sound -- multiple independent groups landed on
"gate speculative expert prefetch by confidence" as the fix for blind prefetch being a net loss,
which is exactly what our own measured evidence (1.4-2% predictor accuracy, 0.82-0.96x slowdown
from blind prefetch) also found necessary. But it is **not novel at the idea level**, and SPICE
specifically is ahead of us on the one thing that actually matters for a real claim: hardware
validation. ASS is a well-tested *simulator*, not a measured result.

---

## 3. Popularity/importance-aware mixed-precision expert quantization (relevant to SPEC_IDEA1)

- **[GEMQ](https://arxiv.org/pdf/2605.23078)** (Global Expert-level Mixed-precision Quantization)
  -- global linear-programming bit allocation from expert importance, with router fine-tuning.
- **[MODE](https://arxiv.org/html/2606.17118v1)** -- modality-decomposed expert-level
  mixed-precision for multimodal MoE.
- **[Colla-Q](https://arxiv.org/pdf/2609.18131)**, **[EAC-MoE](https://arxiv.org/pdf/2508.01625)**,
  **[QuantMoE-Bench](https://arxiv.org/pdf/2406.08155)** -- further 2024-2026 work in the same
  family: allocate bits per expert based on some importance signal, not routing frequency alone.
- **A specific, directly relevant finding**: multiple of these papers note that **"frequency
  reflects how often an expert is selected but doesn't fully align with how much quantization
  degrades model performance, and experts with similar frequencies can exhibit vastly different
  quantization sensitivities."** This is exactly the assumption `SPEC_IDEA1_POPULARITY_TIERED_
  PRECISION.md` leans on (§10 lists "quality of *mixed* precision is not measured anywhere in the
  ledger... a prediction, not a measurement" as a risk, but doesn't flag that popularity/frequency
  specifically -- as opposed to some other importance signal -- may be the wrong proxy). This is
  new, published evidence to fold into that spec's own risk section before building it.

**Verdict for this piece:** also not novel at the idea level -- it's an active academic subfield
with several 2025-2026 papers. Our spec's version is additionally *unbuilt* (zero measurements
exist) and has two real engineering gaps found by direct code inspection (per-expert vs per-layer
granularity given how GGUF actually stores expert tensors, and the runtime cache's actual
slot-budgeting formula) that the academic papers don't have to solve, because they're generally
framework-native rather than constrained by GGUF's on-disk tensor layout.

---

## 4. Honest novelty verdict

Nothing built or specced this session is conceptually unprecedented. All three ideas have active,
often very recent (2025-2026), published or in-progress prior art, and in two of the three cases
(core cache vs. moe-autopilot/#20757; ASS vs. SPICE) there is a directly comparable llama.cpp- or
MoE-specific competitor. The one piece that is real, checkable, and shipped -- the core cache --
is genuinely useful and correctly verified, but its honest framing is *"implemented and shipped a
specific, community-requested llama.cpp feature, with rigorous correctness verification and a
measured speedup over the tool's own existing technique,"* not *"solved a problem nobody has
solved."* The other two pieces (ASS, tiered-precision) are reasonable, well-motivated research
directions that converge with what other groups are actively doing -- but they are not yet
results. Publishing either as a finished achievement right now would not survive someone in this
space actually checking.

---

## 5. Plan: taking ASS from Python simulation to a real, measured result

Current state: Phases A-E built and unit-tested (44 tests) in `supervram/` (pure Python), verified
via a 3,600-run *simulated* ablation matrix. Zero lines run against a real model or real GPU.
SPICE (above) already has multi-GPU, multi-model hardware numbers for a similar idea -- so simply
re-deriving the same result on real hardware, later than SPICE, is not itself the goal; the goal
is an honest, working local feature, not a paper.

1. **Port the gate into the real cache path, minimally first.** `ContextManager`-style Python
   objects can't run inside `ggml_backend_sched`'s C++ decode loop. Smallest viable port:
   `AdaptiveSpeculativeScheduler`'s gate logic (confidence lookup + budget-pooled selection) as a
   small C++ class living next to the existing `ggml_backend_sched_expert_cache` (already found
   and read this session, `ggml-backend.cpp`), fed by the *existing* per-token routing IDs the
   cache already resolves -- do not attempt the full window-planner/predictor abstraction from the
   Python version on the first pass. `PLAN_ADAPTIVE.md` section 9's TODO-9 already sketches this
   (`--moe-expert-scheduler {off,ass}`, gate decisions as async pinned-staging reads); treat that
   as the real spec now, not aspirational.
2. **Calibrate confidence from real routing traces, not synthetic ones.** We already have real
   routing traces from `SVRAM_TRACE` on the actual Qwen3-30B-A3B model (used for the policy-headroom
   ablation). Replay those through the *C++* gate (once ported) before trusting it on live decode --
   this directly closes KNOWN_LIMITATIONS item 15 ("confidence gate has not been calibrated on real
   Qwen routing data").
3. **Measure the same four things the cold-SSD protocol already measures for the classic cache**:
   steady-state decode t/s, SSD read rate/bytes, hit rate, and correctness (token + logits hash vs
   a non-gated baseline on the same trace). Reuse `scripts/cold_run.py`'s existing methodology so
   the number is comparable to every other row already in `EVIDENCE_LEDGER.md`, not a new ad hoc
   protocol.
4. **Benchmark against `--scheduler off` on the SAME real hardware, not just the simulator.** The
   simulated result (waste_bytes down, throughput up in ~38% of cases, worse in ~0.5%, oracle
   warm-up cost understood) is a hypothesis about what should happen on hardware, not a
   confirmation. If real-hardware gains are smaller or absent, that is itself a valid, reportable
   result (matches this project's own history of honestly reporting "we tried it, it didn't help" --
   see the prefetch-feasibility section that motivated ASS in the first place).
5. **Only after that: compare directly against SPICE's numbers**, same models if feasible
   (DeepSeek-V2-Lite / Qwen2-57B-A14B aren't downloaded here, but Qwen3-30B-A3B and Qwen3.6-35B-A3B
   are close analogs), same reporting convention (TPOT), so a claim like "matches/beats SPICE on
   this specific hardware class" is actually checkable rather than asserted.

**Effort estimate honestly**: step 1 is the large one -- it's C++ inside an already-complex
scheduler, previous patches to this file (0001-0009) each took real, multi-session effort. Steps
2-4 are mostly reusing existing tooling. Do not skip straight to step 5.

---

## 6. Plan: taking SPEC_IDEA1 (tiered precision) from spec to a real, measured result

Current state: a detailed, evidence-grounded spec, zero code, zero measurements. Two real
mechanism gaps already found by direct inspection (not yet fixed in the spec itself):

1. **Resolve the granularity question first, before writing any code.** GGUF stores all experts
   for one (layer, projection) as a single 3D tensor at one quantization type -- confirmed this
   session via `gguf_dump` on the real Qwen3-30B-A3B file. `llama-quantize --tensor-type-file`
   can only target whole tensors, i.e. whole layers, not individual experts. Two honest paths:
   - **(a) Layer-granularity v1** (small change to the spec, no new C++): rewrite SPEC_IDEA1 §5.2
     to select *layers* by an aggregated per-layer popularity score (sum or max of that layer's
     expert weights), not individual (L,e) pairs. This is implementable exactly as scoped ("no C++
     changes required for v1") and is the version actually worth building first.
   - **(b) True per-expert granularity** (large change): would need either a GGUF format extension
     or a runtime that reassembles a logically-split tensor from separately-quantized pieces --
     out of scope until (a) is measured and shown to be worth the larger investment.
2. **Verify the cache's real slot-budgeting behavior before promising a specific win mechanism.**
   Already read the actual code (`ggml_backend_sched_expert_cache_layout`): the slot *count* per
   tensor is a shared global budget (`capacity_bytes / sum_expert_size`) applied uniformly, not
   independently byte-proportional per tensor. Update SPEC_IDEA1 §7.1's tripwire language to match
   this mechanism (more total cached experts *in aggregate* when some layers shrink, not "cold
   layers specifically get more room") so the measurement step doesn't chase the wrong effect.
3. **Fold in the published counter-evidence on frequency-as-importance-proxy (section 3 above)
   before trusting the tier-assignment policy.** Cheapest test, no new tooling: for a handful of
   experts already flagged as "rarely used" by `make_warm_profile.py`, run the *existing*
   teacher-forced quality comparison (`compare_quality.py`) with just those specific experts
   forced to Q4 one at a time, and check whether quantization sensitivity actually tracks usage
   frequency on this model family, or whether (per GEMQ/MODE/Colla-Q) some rare experts are
   disproportionately sensitive. This is a half-day check that determines whether the whole
   popularity-based tiering premise holds before building the full pipeline.
4. **Then follow SPEC_IDEA1's own §4/§6/§7 as written** (popularity_to_tiers.py,
   build_tiered_model.sh, the four-section measurement plan) -- that part of the spec is sound and
   doesn't need revision, only steps 1-3 need to land first.
5. **Compare the measured result against GEMQ/Colla-Q's reported PPL-vs-bytes tradeoffs** once a
   real tiered file exists, the same way step 5 above proposes for ASS vs SPICE -- an honest,
   checkable "how does our llama.cpp-native, GGUF-constrained version compare to the
   framework-native academic ones" statement, not an assumed win.

**Effort estimate honestly**: step 1(a) and step 3 are cheap (a few hours to a day each, mostly
reusing existing scripts). Step 2 is a documentation fix, not new work. The real cost is in
step 4's actual pipeline + the RTX 3090 measurement runs themselves (quantize, verify, benchmark --
each of those has historically taken real wall-clock time in this project, per
`writer_handoff/EVIDENCE_LEDGER.md`'s own timeline).
