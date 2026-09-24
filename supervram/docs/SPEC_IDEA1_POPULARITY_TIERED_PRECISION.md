# Spec — Idea 1: Popularity-tiered expert precision

**Status:** draft for handoff to a coding agent. Self-contained; the agent should not need to
read other files except the four references at the end.
**Goal:** cut SSD-bound decode time by storing the *rarely routed* MoE experts at a lower
precision than the frequently routed ones, in a single mixed-precision GGUF. This is the
highest-confidence, lowest-cost throughput lever in the project's evidence base.

---

## 1. Why this, grounded in the project's own measured numbers

The regime we optimize is: model larger than VRAM, low RAM (4 GiB), SSD is the backing tier,
decode is the metric. In that regime the evidence says decode time is dominated by **bytes
pulled from the SSD**, and the three levers are (a) miss rate, (b) bytes per miss,
(c) SSD bandwidth.

Relevant measured facts (from `writer_handoff/EVIDENCE_LEDGER.md` and
`docs/IMPLEMENTATION_STATUS.md`, all `evidence_class: measured_rtx3090`):

1. **SSD wait is 72–81% of decode time.** Q8_0 steady state: 36.1 s SSD wait of 50.4 s (72%).
   43 GiB synthetic: 81%. Throughput is bandwidth-bound, not policy-bound.
2. **Replacement policy is not the lever.** LRU 96.28%, Belady optimum 97.10% per-layer /
   97.31% shared at 16 GiB. < 1 point of headroom. Stop optimizing eviction.
3. **Lower precision is a real, measured lever.** Q4_K_M reads **32% fewer bytes** than Q8_0 at
   the same cached fraction and decodes **41.4 vs 31.8 t/s** (+30%) at 97% hits. Full-Q4 costs
   only **+2.1% perplexity** vs Q8 (ground-truth, 30.7k tokens of human prose, 95% CI
   1.015–1.025) and **96.0% top-1 agreement** with Q8's tokens.
4. **Misses are the cold experts.** "Misses are the rarely used experts (~0.2 per layer per
   token), so they are close to unpredictable from routing history." The long tail of experts —
   the ones that dominate *miss-bytes* — are the ones that rarely fire.
5. **Mixed-precision GGUF already builds and runs.** `make_synthetic_model.sh` produced a
   42.9 GiB model where layers 0–23 experts are F16 and everything else Q8_0, via
   `llama-quantize --allow-requantize --tensor-type-file <regex>`. It ran, gated bit-exact,
   and produced coherent output. So per-tensor mixed precision is a *proven* mechanism, not a
   hypothesis.
6. **Popularity is already computed and used.** `make_warm_profile.py` emits per-(layer,
   expert) routing share from `--trace` / `SVRAM_TRACE` traces, and the warm-start +
   admission-control features already consume a popularity ranking. We are adding precision
   as *one more property derived from the same ranking* — it composes, it doesn't fight.

**The key insight this spec exploits:** today, *every* expert in the file is the same
precision, so the ~45% of experts that rarely fire still cost full Q8 bytes when they are
missed. Since quality tolerance is *highest* exactly for the experts that rarely fire (they
rarely contribute to the output), downgrading *them* buys the largest byte savings per unit of
quality risk. This is the natural, principled extension of the already-measured "Q4 is 32%
cheaper at +2% PPL" result, applied selectively instead of uniformly.

---

## 2. What this spec does and does NOT do

**Does:**
- Produce a popularity profile of experts from routing traces (reuse existing tooling).
- Decide a per-expert precision tier (hot = Q8_0, cold = Q4_K_M) from that profile using an
  explicit, tunable policy.
- Build a single mixed-precision GGUF where the selected "cold" experts are Q4_K_M and all
  others (including *all* non-expert weights, all gate/up/down of hot experts, and the router)
  stay Q8_0.
- Verify: correctness gate, byte-size accounting (bytes saved, where), quality gate
  (teacher-forced + ground-truth PPL vs both Q8_0 and full-Q4_K_M), and the SSD-bound decode
  benchmark (the actual metric).

**Does NOT (in this draft):**
- Change the runtime cache, scheduler, or kernel. The mixed-precision file is a *drop-in
  GGUF*; the existing cache path reads it unchanged. No C++ changes are required for v1.
- Do signal-based prefetch (Idea 3) or parallel-drive sharding (Idea 2). Those are separate
  specs; this one composes with them later.
- Touch the gate/router matrices or attention — they stay Q8_0 (see §5).

**v1 scope decision:** two precision tiers (Q8_0 hot / Q4_K_M cold). A three-tier variant
(Q8 / Q4 / Q3) is the natural follow-up (§9) once the two-tier quality/byte tradeoff is
measured and understood. Keep v1 to two tiers so the ablation is clean.

---

## 3. Definitions

- **Expert weight tensor** (Qwen3-30B-A3B): the MoE expert MLP weights, GGUF names of the form
  `blk.<L>.ffn_<gate|up|down>_exps.weight`, `<L>` in 0..47, 128 experts per layer, 48 layers.
  (Confirmed by the regex in `make_synthetic_model.sh`:
  `blk\.([0-9]|1[0-9]|2[23])\.ffn_(gate|up|down)_exps\.weight=f16`.)
- **Non-expert weights:** everything else — attention (`blk.L.attn_*`), `ffn_gate`/router
  projection, `ffn_down` (the shared/down proj if present), norms, embeddings, output head.
  These are small in aggregate (~1.3 GiB dense for Q8_0 per the ledger) and stay Q8_0.
- **Popularity / usage weight w(L,e):** fraction of tokens whose routing selected expert `e`
  at layer `L`. Already produced by `make_warm_profile.py` (third column of its output).
- **Hot set H:** the set of (L,e) experts assigned to Q8_0. **Cold set C:** the rest, assigned
  to Q4_K_M. H ∪ C = all experts, H ∩ C = ∅.
- **Cached fraction (bytes):** bytes of expert weights that fit in the 16 GiB cache, as a
  fraction of total expert bytes. (Distinct from "fraction of experts," which is what the
  43 GiB analysis used — see §8, this is exactly the assumption to re-test.)

---

## 4. Deliverables (the coding agent's concrete output)

1. **`scripts/popularity_to_tiers.py`** — reads one or more warm profiles (or traces) and
   emits (a) a ranked per-expert usage table, (b) a chosen (L,e) → tier assignment under a
   given policy, (c) a `--tensor-type-file` mapping file, and (d) a JSON summary of the
   decision (how many experts per layer hot/cold, expected byte reduction).
2. **`scripts/build_tiered_model.sh`** — runs `llama-quantize --allow-requantize
   --tensor-type-file` to produce the mixed GGUF, modeled directly on
   `make_synthetic_model.sh`.
3. **A `results/tiered-precision/` run** with the four measurement sections in §7: size
   accounting, exactness gate, quality (teacher-forced + ground-truth PPL), and the SSD-bound
   decode benchmark against the two existing baselines.
4. **Updated docs** (agent should append, following the project's convention):
   `docs/IMPLEMENTATION_STATUS.md` (new section "Popularity-tiered precision"),
   `writer_handoff/EVIDENCE_LEDGER.md` (new measured rows with `evidence_class` labels),
   `writer_handoff/KNOWN_LIMITATIONS.md` if any new caveat.

The agent should *not* hand-edit `EVIDENCE_LEDGER.md` claims beyond the measured numbers it
actually produced — the project is strict about separating measured vs simulated vs estimated.

---

## 5. Design

### 5.1 Tensor selection: what may be downgraded, and what must not

**Downgrade to Q4_K_M (the cold set C):** a *contiguous per-expert triple*
`(gate, up, down)_exps` for the chosen (L,e). These three are a single expert's MLP and must
move together — do not split an expert's gate/up/down across precisions, that breaks the
expert's internal consistency and adds no benefit.

**Keep Q8_0 (the hot set H ∪ non-experts):**
- All non-expert weights (attention, norms, embedding, output head, router).
- **The router / gate projection explicitly stays Q8_0.** It decides which experts fire and is
  small; a quantized router could systematically shift popularity in a way that invalidates the
  very profile we built from. Keeping it exact is cheap and removes a confound. (If the agent
  wants, it may *additionally* produce a router-quantized variant as a sensitivity row, but the
  primary artifact keeps the router at Q8.)
- All experts in the hot set H (their gate/up/down).

### 5.2 Tier-assignment policy (the tunable core)

Input: `w(L,e)` for all experts, from one or more warm profiles. The agent should support these
policies, implemented as selectable modes, with the *default* being the one that maximizes
decode throughput subject to a quality budget:

- **`--mode topk`** (default candidate): H = the top-K experts *globally by usage weight* (or
  top-K *per layer* — support both via `--per-layer`), C = the rest. K is the primary knob.
- **`--mode threshold`**: C = experts with `w(L,e) < T`, H = the rest. `T` is the knob.
- **`--mode bytes`**: given a target total byte budget B (≈ the 16 GiB cache), fill it with
  the most-used experts at Q8 and downgrade the rest to Q4 — this directly optimizes "maximize
  cached fraction at fixed bytes," which is the quantity the 43 GiB analysis showed matters.

Default to a configuration that mirrors the *existing* Q8_0 setup's cached byte fraction so the
A/B comparison in §7 is apples-to-apples: i.e., choose K (or T) such that the hot-set byte
footprint ≈ the bytes the current 71-slot/layer cache holds. That isolates "did precision
help" from "did we just change cache size."

**Why "per (L,e)" not "per layer":** popularity is highly skewed per the trace evidence (misses
concentrate in rare experts). A whole-layer tiering (like the synthetic F16x24 model) is coarse
and would downgrade experts that are hot *within their layer*. Per-expert tiering is the point
of this idea; the existing tooling (`make_warm_profile.py`) already gives per-(L,e) weights, so
the granularity is free.

### 5.3 Building the tensor-type-file

`make_synthetic_model.sh` writes one line `regex=<type>` per line. For per-expert tiering we
need many lines (one per cold (L,e), plus the complement at Q8). `llama-quantize
--tensor-type-file` accepts multiple `regex=type` lines and later lines override earlier ones
(verify this precedence in the quantize source before relying on it — see §10). Two clean ways
to express it, prefer the one that is unambiguous:

- **Explicit cold list:** emit one line per cold (L,e) triple as
  `blk\.<L>\.ffn_(gate|up|down)_exps\.weight=Q4_K_M`, and a final catch-all line
  `blk\.[0-9]{1,2}\.ffn_(gate|up|down)_exps\.weight=Q8_0`.
  Order: cold-specific lines first, catch-all Q8 last (or confirm override order and rely on it).
- **If override order is "last match wins":** emit the Q8 catch-all first, then the cold lines.

The agent MUST print the resolved regex→type mapping and, ideally, a count of tensors matched by
each line, so a wrong regex (a silent all-Q8 or all-Q4 file) is caught immediately by the size
accounting in §7.1 rather than discovered later.

### 5.4 The file that gets built (v1)

`<base>-tiered-Q8hot-Q4cold.gguf` from the existing
`Qwen3-30B-A3B-Q8_0.gguf` (the 30 GiB model is the primary target — it is the one where the
cache competes with `--cpu-moe`/mmap and where SSD-bound decode is measured). Optionally also
the 43 GiB synthetic model (§8) to test whether tiering rescues the 0.08× parity.

---

## 6. Concrete build pipeline (reference commands)

The agent should implement these as the scripts in §4, but here is the shape, so the intent is
unambiguous:

```bash
# 1. Profiles already exist (from prior trace collection). If not, collect first:
#    run svram-verify with --trace / SVRAM_TRACE over the 8 standard prompts (see
#    scripts/prompts.txt), 384 tokens each, Q8_0, 16 GiB cache — the same setup that
#    produced results/rtx3090/traces/. (Reusing existing trace files is fine and cheaper.)

# 2. Decide tiers (default: match the current hot-set byte footprint).
python scripts/popularity_to_tiers.py \
    --profiles results/rtx3090/traces/profile.txt \
    --mode bytes --byte-budget-mib 16384 \
    --out-tier results/tiered-precision/tiers.json \
    --out-tensorfile results/tiered-precision/tensor-types.txt

# 3. Build the mixed GGUF (mirrors make_synthetic_model.sh).
bash scripts/build_tiered_model.sh \
    "$HOME/models/Qwen3-30B-A3B-Q8_0.gguf" \
    "$HOME/models/Qwen3-30B-A3B-tiered-Q8hot-Q4cold.gguf" \
    results/tiered-precision/tensor-types.txt

# 4. Measure (sections in §7).
```

`make_warm_profile.py` output format is exactly: `<layer> <expert> <weight>` per line, weight =
share of tokens that used the expert. `popularity_to_tiers.py` should accept this format
directly (it is the project's existing profile format — do not invent a new one).

---

## 7. Measurement plan (this is the part that proves the idea)

Follow the project's existing protocols (`scripts/cold_run.py`, `scripts/compare_quality.py`,
`scripts/run_ground_truth_ppl.sh`) so the numbers are comparable to the ledger. Run everything
on the RTX 3090 host. Four sections, in order:

### 7.1 Size / byte accounting (gate the build)
- Report total file size, and the bytes in expert tensors that are Q8 vs Q4 vs F16.
- Report the **cached-fraction-bytes at 16 GiB** for: (a) current Q8_0 file, (b) the new
  tiered file, (c) full-Q4_K_M file. The tiered file should have *more* experts/bytes cached at
  the same 16 GiB than Q8_0, and more than full-Q4 if the hot set is a superset of what Q4
  caches. If it does not, the tiering is not doing anything — stop and fix §5.2.
- **This catches a silently-wrong GGUF** (all-Q8 → size ≈ 30 GiB; all-Q4 → size ≈ 17 GiB)
  before any expensive benchmark.

### 7.2 Exactness gate (correctness, not speed)
- Use `svram-verify` with the tiered file, greedy, `-ub 1`, and confirm the output is
  *deterministic and coherent*. Because the file is mixed precision, it will **not** be
  bit-identical to pure Q8_0 (different arithmetic) — that is expected, like the Q4 case.
  The gate here is: (a) it runs without OOM/crash, (b) greedy tokens are stable across two
  runs (determinism), (c) output text is coherent. Bit-identical-to-something is only required
  against *itself*, not against Q8_0.
- Confirm the runtime cache path reads it unchanged (no C++ edits) and that hit-rate / slot
  accounting behaves sanely (slots are byte-sized; a Q4 expert occupies fewer bytes, so more fit
  — verify the slot count actually increases, which is the whole point).

### 7.3 Quality (the load-bearing cost)
Two measurements, both already scripted in the project:
1. **Teacher-forced** (`scripts/compare_quality.py`): tiered-file-as-candidate vs Q8_0-as-ref,
   3,072 tokens / 8 prompts, same protocol. Report top-1 agreement and perplexity ratio.
   **Target: between full-Q4 (96.0% agreement, +2.1% PPL) and Q8 (100%, +0%).** Expect closer
   to Q8 because the downgraded experts are the rarely-used ones.
2. **Ground-truth PPL** (`scripts/run_ground_truth_ppl.sh`): tiered vs Q8_0 vs full-Q4_K_M on
   the same 60 chunks of human prose. **Target: perplexity ratio tiered/Q8 strictly below
   full-Q4/Q8 (1.020).** If tiered lands within CI of Q8, the quality cost is effectively free.

### 7.4 The actual metric: SSD-bound decode (the win)
Same protocol as the ledger's "Smaller experts" row so it is directly comparable:
cold start (evict page cache), O_DIRECT, 4 GiB RAM cap, 1,536 tokens, same prompt, 16 GiB cache,
`-ub 1`. Report for the tiered file vs Q8_0 vs full-Q4_K_M:
- steady-state decode t/s (from token 256),
- SSD read rate (GB/s) and total SSD bytes over the run,
- hit rate and **cached-fraction-bytes**,
- estimated parity vs native (~0.32× for Q8_0 is the reference point).

**Success criterion for v1:** at the *same* cached-fraction-bytes as the Q8_0 baseline, the
tiered file decodes **strictly faster than Q8_0** and **no slower than full-Q4_K_M**, with
quality (7.3) at or better than full-Q4. A plausible target is >41.4 t/s (full-Q4's number)
because the hot set stays Q8 while still fitting more bytes in cache than Q8_0. If the tiered
file is *slower* than full-Q4, the byte budget wasn't actually smaller and §5.2 is wrong.

### 7.5 (Optional, high value) 43 GiB model
Run 7.1 + 7.4 on the synthetic 43 GiB model with a tiered file sized to the same 16 GiB cache.
Hypothesis: because the cold experts now cost ~half the bytes, the same 16 GiB cache covers a
much larger *byte* fraction, and the 6.6 t/s / 0.08× parity improves toward ~0.15–0.2×. This is
the cleanest test of whether "working set doesn't fit" is a *byte* problem (fixable by
precision) or a *count* problem (not fixable by precision). Label it clearly as using the
synthetic model (throughput/capacity only, not quality).

---

## 8. The existing assumption this spec is testing

`IMPLEMENTATION_STATUS.md` (43 GiB section): "Parity falls quickly as the model outgrows the
cache: about 0.32x at 30 GiB, about 0.08x at 43 GiB," and "'the active working set fits in
VRAM' holds only partly... once the cached *fraction* drops below about 55%."

That conclusion was measured with a *uniform-precision* file, where cached fraction is counted
in **experts** (71 of 128). This spec re-frames the constraint as **bytes**: the cache holds a
fixed number of *bytes*, so a mixed-precision file with a Q4 cold tail has the same byte budget
cover *more* of the routing byte-distribution. If the dominant cost is truly bytes-from-SSD
(which the 72–81% SSD-wait numbers say it is), then reducing per-miss bytes should improve
parity even as the *expert-count* fraction stays the same. §7.5 is the direct test of this
reframe.

---

## 9. Follow-ups (out of scope for v1, but the design should not block them)

1. **Three tiers** (Q8 / Q4 / Q3_K) once the two-tier byte/quality tradeoff is understood — the
   `bytes` mode already generalizes; just add a third precision level and a second budget.
2. **Popularity as the *unifying* axis**: the same ranking already drives warm-start and
   admission control. A future spec could co-tune precision + placement + routing-bias from one
   popularity profile (the routing-bias feature, overnight `bias 0.02` 123s→87s, is a separate
   but synergistic lever).
3. **Adaptive / online**: if popularity drifts, re-quantize is expensive, so this is a *batch*
   artifact, not a runtime one. Fine for v1. A runtime variant would need in-place precision
   selection, which is a bigger design — do not scope it here.
4. **Composes with Idea 2 (parallel NVMe)** and **Idea 3 (signal prefetch)**: fewer bytes per
   miss (this) × more bandwidth (2) × fewer misses (3). The §7.4 number should be reported so it
   can be combined multiplicatively later.

---

## 10. Risks and load-bearing uncertainties (flag these in the results writeup)

- **`--tensor-type-file` override order / multi-line precedence** is assumed, not verified in
  this spec. The agent MUST confirm in `llama-quantize`'s source how multiple matching regex
  lines resolve (last-wins vs first-wins), and MUST use §7.1 size accounting as the tripwire.
  *Load-bearing: a wrong assumption here produces a silently-wrong file.*
- **Quality of *mixed* precision is not measured anywhere in the ledger.** The +2.1% PPL is for
  *full* Q4. The mixed case should be better, but this is a **prediction, not a measurement** —
  §7.3 is the experiment that turns it into evidence. Do not claim a specific PPL number until
  §7.3 is run.
- **Popularity is prompt-dependent.** The ledger notes decode speed ranged 15–48 t/s across the
  8 prompts. The tier assignment should be built from *all* 8 prompts' profiles (or a held-out
  split: train tiers on some prompts, test on others) to avoid overfitting the tiers to the
  benchmark prompts. *If the agent uses the same 8 prompts to both build tiers and benchmark,
  the result is optimistic; note this.*
- **Slot/byte accounting in the runtime cache**: the cache is described as "slots" — confirm
  whether a slot is a fixed expert-count or a fixed byte size. If slots are a fixed *count*,
  a Q4 expert in a "slot" wastes half the slot's bytes, and the win is smaller than §7.1
  suggests. The agent should read the cache sizing in the C++ (`ggml_backend_sched` expert
  cache) to confirm the cache is byte-budgeted, not count-budgeted. *Load-bearing for the size
  of the win.*
- **`--allow-requantize` from Q8_0 to Q4_K_M for individual tensors** is the same operation the
  synthetic model used (Q8→F16), so it should work, but the agent should confirm the quantize
  path accepts a *target* type per tensor when the base type is Q8_0 (the synthetic script uses
  target `Q8_0` as the file-level arg and per-tensor `f16`; here the per-tensor type is `Q4_K_M`).

---

## 11. Acceptance checklist (the coding agent is done when)

- [ ] `popularity_to_tiers.py` produces a tier assignment + a valid `tensor-types.txt` + a JSON
      summary, under all three modes, and prints the resolved regex→type mapping with per-line
      tensor counts.
- [ ] `build_tiered_model.sh` produces a GGUF whose §7.1 size/byte accounting shows the hot set
      at Q8 and the cold set at Q4 (not accidentally all-one-or-the-other).
- [ ] §7.2 exactness gate passes (deterministic, coherent, no OOM, slot count increased).
- [ ] §7.3 quality: tiered top-1 agreement and ground-truth PPL both **strictly better than
      full-Q4_K_M** and at-or-better-than Q8 within CI.
- [ ] §7.4: at the *same* cached-fraction-bytes as Q8_0, tiered decode is **faster than Q8_0**
      and **no slower than full-Q4**, on the cold/O_DIRECT/4GiB/16GiB protocol.
- [ ] Results written to `results/tiered-precision/` with the four sections' raw numbers, and
      the three docs updated with `evidence_class: measured_rtx3090` labels and a clear
      "prediction vs measurement" distinction per §10.
- [ ] The prompt-dependence caveat (build tiers on a held-out prompt split if possible) is
      stated in the writeup.

## 12. References (read these if you need more context — the spec above should be enough)

- `writer_handoff/EVIDENCE_LEDGER.md` — the measured numbers cited throughout.
- `writer_handoff/KNOWN_LIMITATIONS.md` — failure modes and evidence-label conventions.
- `docs/IMPLEMENTATION_STATUS.md` — the "Smaller experts", "43 GiB", and cache sections.
- `scripts/make_warm_profile.py`, `scripts/make_synthetic_model.sh`,
  `scripts/compare_quality.py`, `scripts/run_ground_truth_ppl.sh`, `scripts/cold_run.py` —
  the existing tooling this spec extends or reuses.
