# Spec — Idea 1 (rev 2): Popularity-tiered expert precision, **layer-granular**

**Status:** draft for handoff to a coding agent (Claude Code). Supersedes rev 1 — read the "Corrections to rev 1" section before anything else.
**Goal:** build a mixed-precision Qwen3-30B-A3B GGUF in which **rarely-used layers are quantized to a lower precision than frequently-used layers**, using popularity data collected from routing traces. Then measure quality and throughput in the SSD-bound regime to decide if it's a real win.
**Scope guard:** no C++ changes. Everything in this spec must be implementable with the existing `llama-quantize --tensor-type-file` path (proven in `make_synthetic_model.sh`).

---

## 0. Corrections to rev 1 (read this first)

Rev 1 claimed "per-expert, not per-layer" tiering. Two code findings (verified by the reviewing agent) correct that:

**Correction 1 — Granularity is layer, not expert.**
In Qwen3-30B-A3B, the expert weights are a **single 3D tensor** per `(layer, ffn_component)` — e.g. `blk.0.ffn_gate_exps.weight` has shape `[2048, 768, 128]` (vocabulary, hidden, num_experts). All 128 experts of layer 0's gate share that tensor at a single quantization type. `llama-quantize --tensor-type-file` (CLI side: `parse_tensor_type` in `tools/quantize/quantize.cpp`, exact string match on tensor name; C++ side: `llama-quant.cpp` iterates `p->pattern` as a regex over tensor names) can therefore only select **layers**, not individual experts within a layer.

Consequence: the deliverable is **layer-granular popularity-tiered precision**. This is weaker than the per-expert framing in rev 1 §5.2, but it's what the tooling actually supports, and it's still a real improvement over the existing "layers 0–23" arbitrary split.

**Correction 2 — The cache slot budget is global, not per-layer.**
`ggml_backend_sched_expert_cache_layout` in `ggml-backend.cpp` computes one global:
```
n_slots_budget = capacity_bytes / sum_expert_size   // sum over ALL tensors
```
and applies that **same** slot count uniformly to every tensor (Q8 and Q4 layers alike). Only the per-slot byte cost varies by each tensor's own precision.

Consequence: shrinking some layers to Q4 raises the **shared** slot budget for everyone — it does not preferentially give more cached slots to the Q4 layers. The net effect is still a higher total cached-expert-count, but the causal mechanism is "more slots for all" rather than "more slots for the downgraded layers." The §7.1 tripwire and §8 framing must reflect this.

**Correction 3 — `--tensor-type-file` takes a raw `ggml_type`, not a quantize preset.**
The value after `=` in the tensor-type-file is parsed by `parse_ggml_type()` (`tools/quantize/quantize.cpp`), which matches against `ggml_type_name()` over the `ggml_type` enum -- e.g. `q4_K`, `q8_0`, `f16`. It does **not** accept whole-model preset names like `Q4_K_M` (verified: `llama-quantize` rejects it with `parse_ggml_type: invalid ggml_type 'Q4_K_M'`). `Q4_K_M` only exists as a name for the *positional* `type` argument, which goes through a separate lookup table and drives `llama-quantize`'s own per-tensor-role heuristics (e.g. upgrading some tensors to `Q6_K`) -- heuristics that don't run when you specify `--tensor-type-file` overrides directly. The correct value is the raw type **`Q4_K`** (~4.25 bits/weight, confirmed via `--dry-run`: a 204.00 MiB `Q8_0` expert tensor becomes 108.00 MiB at `Q4_K`, a factor of 0.529, close to but not identical to the 4.5/8.0 = 0.5625 arithmetic estimate in §1's evidence table, which was computed against a real end-to-end `Q4_K_M`-quantized model, not this override path).

Consequence: every `Q4_K_M` in the rest of this document (§2.2, §3.1, the `build_tiered_model.sh` example, the acceptance checklist) means **`Q4_K`** wherever it refers to the tensor-type-file mechanism. The §1 evidence-table rows about "Q4_K_M reads ~32% fewer bytes / decodes +30% faster" describe measurements against a real, separately-built `Q4_K_M` model and are unaffected -- they're context for why lower precision helps, not a claim about which raw type the override mechanism accepts.

Also verified in the same pass: multi-line `--tensor-type-file` precedence (the §6.1 load-bearing uncertainty) works as assumed -- a 2-layer dry-run test confirmed only the targeted layer's three expert tensors were overridden, adjacent layers were untouched.

**Correction 4 — the §3.1 popularity formula is degenerate on this model's router.**
§3.1 step 2 defines `layer_popularity[L] = (sum of routing counts at L) / (tokens seen at L)`, described as "the average number of experts activated per token at that layer." Qwen3-30B-A3B's router activates a **fixed top-8** experts per token at every layer, so this value is ~8.0 for every layer (verified against the real 8-prompt trace set in `results/rtx3090/traces/`) -- it carries no ranking signal at all, and a `topk`/`threshold` policy built on it would essentially pick layers arbitrarily (whatever the sort happens to do with near-ties).

The corrected default metric (implemented in `popularity_to_layers.py` as `--metric entropy`, the default) is the **entropy of each layer's expert-usage distribution**: layers where usage concentrates on a small number of "specialist" experts (low entropy) are treated as hot (kept at Q8), on the hypothesis that those few, heavily-reused experts matter more to quality; layers where usage spreads thinly across nearly all 128 experts (high entropy) are treated as cold, on the hypothesis that any single quantized expert is rarely the one actually used for a given token -- this mirrors the project's own already-evidenced per-expert claim in this table ("misses are dominated by cold, rarely-routed experts") applied at the layer level. On the real trace set, entropy ranges 6.14-6.62 bits and is not flat, so it's an actually-discriminating signal, unlike raw activation count.

**Update, post-measurement:** Gate 3 and the arbitrary-split control (`writer_handoff/EVIDENCE_LEDGER.md`, "Popularity-based vs arbitrary layer selection") settled this, and the *quality* half of the hypothesis was **wrong**: top-1 agreement and PPL ratio were statistically indistinguishable between entropy-based and arbitrary layer selection (arbitrary was marginally better, not worse), consistent with the prior-art finding (EAC-MoE, QuantMoE-Bench) that usage frequency doesn't reliably predict quantization sensitivity. What the entropy signal actually earns is a **throughput** win -- 1.26x faster decode than the arbitrary split at the identical byte budget and cache slot count -- for reasons not yet isolated (plausibly the temporal locality of a specific decode's expert-access pattern against the runtime's LRU cache, not a quality-preservation mechanism). Treat "entropy-based selection helps quality" as falsified and "entropy-based selection helps cache-hit throughput" as the surviving, measured claim.

Neither correction kills the idea. "Downgrade rarely-used layers to Q4, keep hot layers at Q8" remains evidence-grounded and worth building. What changes is the framing and the expected-mechanism narrative.

---

## 1. Why (evidence-grounded)

Each claim below is tied to a measured number in the project's evidence ledger or implementation status.

| Claim | Source |
|---|---|
| 72–81% of decode time in the target regime is SSD wait | `results/rtx3090/cold/` ledger |
| Q4_K_M reads ~32% fewer bytes per expert than Q8_0 | arithmetic: Q4_K_M = 4.5 bpw vs Q8_0 = 8 bpw |
| Q4_K_M decodes +30% faster at the same cached-expert-fraction | `results/rtx3090/ablations-long/` |
| Full-Q4 PPL penalty is +2% over Q8 on the 100-prompt set | `results/rtx3090/quality/` |
| Misses are dominated by cold (rarely-routed) experts | `analyze_trace.py` output in ledger |
| `llama-quantize --tensor-type-file` builds a valid mixed-precision GGUF with bit-exact gates | `make_synthetic_model.sh` + `svram-verify` pass |
| Per-layer popularity (share of tokens routed to each expert, per layer) is already collected by `make_warm_profile.py` | `scripts/make_warm_profile.py` |
| The cache slot budget is global: `capacity_bytes / sum_expert_size`, uniform slot count for all tensors | `ggml-backend.cpp::ggml_backend_sched_expert_cache_layout` |
| `--tensor-type-file` operates at tensor-name granularity; Qwen3 experts are one 3D tensor per `(layer, component)` | `tools/quantize/quantize.cpp::parse_tensor_type`, `llama-quant.cpp` |

**The one assumption this spec challenges:** rev 1 (and the prior "arbitrary layers 0–23" experiment) treated layer selection as a free parameter to be chosen by hand. Popularity data already exists in the traces. Using it to pick *which* layers to downgrade is strictly more principled than the existing split and costs nothing to compute.

---

## 2. Design

### 2.1 Granularity: per-layer, not per-expert

Each layer L's expert tensors (`blk.L.ffn_gate_exps.weight`, `blk.L.ffn_up_exps.weight`, `blk.L.ffn_down_exps.weight`) are assigned **one** precision. All three components of a layer get the same precision (this is a simplification; see §9 for the refinement).

Non-expert tensors (attention, shared expert, router, embeddings, norms) stay at their original precision (Q8_0 for Qwen3-30B-A3B-Q8_0).

### 2.2 Tier assignment

From the per-layer popularity profile (see §3), assign each layer to one of two tiers:

| Tier | Precision | Rationale |
|---|---|---|
| Hot | Q8_0 (unchanged) | high routing popularity; quality-sensitive |
| Cold | Q4_K (see §0 correction 3) | low routing popularity; quality tolerance is highest here |

The assignment is **data-driven**, not hand-picked. Three selectable policies (all produce the same file format, only the tier boundary differs):

| Policy | Description | When to use |
|---|---|---|
| `topk` | Keep the top-K layers by aggregate popularity at Q8; downgrade the rest. K is chosen to match the byte footprint of the existing "layers 0–23" hot set, so the A/B is apples-to-apples. | Default. Directly comparable to the existing synthetic model. |
| `threshold` | Keep layers whose aggregate popularity ≥ θ at Q8. θ is a hyperparameter. | If you want to sweep quality/throughput tradeoffs. |
| `bytes` | Keep layers at Q8 until the cumulative Q8 byte budget is exhausted; downgrade the rest. | If you want to control total model size rather than layer count. |

**Default:** `topk` with K chosen so the Q8 layers occupy the same total byte count as "layers 0–23" in the existing synthetic model. This makes the comparison clean: same cache budget, same number of hot layers, different *which* layers.

### 2.3 What is NOT in scope for v1

- Per-expert (within-layer) tiering. Would require a GGUF format change or a custom quantize path. Defer to v2.
- Mixed precision within a single tensor (e.g. expert 0–63 at Q8, 64–127 at Q4 in the same `blk.L.ffn_gate_exps.weight`). Same constraint. Defer to v2.
- Per-component tiering (gate at Q4, up at Q8, down at Q8 in the same layer). Mechanically possible via `--tensor-type-file` (three separate tensor names per layer), but it complicates the tier-assignment logic without a clear quality benefit. Defer to v2.

---

## 3. Deliverables

Two new scripts and one modified build step. No C++ changes.

### 3.1 `scripts/popularity_to_layers.py`

**Purpose:** aggregate per-expert routing counts into per-layer popularity scores, assign tiers, emit a `--tensor-type-file` mapping.

**Input:** one or more trace files (format: `<layer> <expert_1> <expert_2> ...` per line, as produced by `--trace` / `SVRAM_TRACE` and already consumed by `make_warm_profile.py`).

**Processing:**
1. For each layer L, sum the routing counts across all experts in that layer. This gives `layer_popularity[L] = Σ_e count[(L, e)]`.
2. Normalize: `layer_popularity[L] /= total_tokens` (share of tokens that routed through layer L's expert set — note: for MoE, every token routes through every layer, so this is the *average number of experts activated per token* at that layer, weighted by popularity). The meaningful ranking is the relative ordering across layers, not the absolute value.
3. Apply the chosen tier policy (§2.2) to assign each layer to Q8 or Q4.
4. Emit a `--tensor-type-file` mapping: one line per Q4 layer, `blk.<L>.ffn_(gate|up|down)_exps\.weight=Q4_K`. Q8 layers need no line (the default `Q8_0` from the `llama-quantize` target type applies).

**Output files:**
- `<out_prefix>.tensor-type-file` — the mapping file for `llama-quantize --tensor-type-file`.
- `<out_prefix>.summary.json` — per-layer popularity scores, tier assignments, byte counts per tier, total model size estimate.

**CLI:**
```
python3 popularity_to_layers.py \
  --traces trace1.txt trace2.txt ... \
  --policy topk \
  --k 24 \
  --out-prefix results/tiered/layers \
  [--q4-type Q4_K]
```

**Example output (`layers.summary.json`):**
```json
{
  "policy": "topk",
  "k": 24,
  "q8_layers": [3, 7, 12, 15, 18, 21, 25, 28, 31, 33, 36, 39, 42, 45, 0, 5, 10, 14, 19, 23, 27, 30, 34, 38],
  "q4_layers": [1, 2, 4, 6, 8, 9, 11, 13, 16, 17, 20, 22, 24, 26, 29, 32, 35, 37, 40, 41, 43, 44, 46, 47],
  "q8_bytes_estimate": 18600000000,
  "q4_bytes_estimate": 10600000000,
  "total_bytes_estimate": 29200000000,
  "layer_popularity_ranking": [
    {"layer": 3, "popularity": 0.0234, "tier": "q8"},
    {"layer": 7, "popularity": 0.0221, "tier": "q8"},
    ...
    {"layer": 1, "popularity": 0.0089, "tier": "q4"},
    ...
  ]
}
```

### 3.2 `scripts/build_tiered_model.sh`

**Purpose:** build the mixed-precision GGUF using the tensor-type-file from §3.1.

**Modeled directly on `make_synthetic_model.sh`,** with the key difference that the tensor-type-file is generated by `popularity_to_layers.py` rather than hard-coded.

```bash
#!/usr/bin/env bash
# Build a popularity-tiered mixed-precision GGUF.
# Usage: build_tiered_model.sh <source.gguf> <output.gguf> <tensor-type-file>
#   e.g. build_tiered_model.sh \
#     "$MODELS/Qwen3-30B-A3B-Q8_0.gguf" \
#     "$MODELS/Qwen3-30B-A3B-tiered-Q8x24-Q4x24.gguf" \
#     "results/tiered/layers.tensor-type-file"
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$1"; DST="$2"; TT="$3"
echo "Building tiered model:"
echo "  source:  $SRC"
echo "  output:  $DST"
echo "  tensor-type-file:"
cat "$TT"
echo ""
"$ROOT/third_party/llama.cpp/build-3090/bin/llama-quantize" \
  --allow-requantize \
  --tensor-type-file "$TT" \
  "$SRC" "$DST" Q8_0
echo "Done: $DST"
```

### 3.3 End-to-end pipeline (reference commands)

```bash
# 1. Collect traces (if not already done). Example: 10 prompts, 256-token decode.
#    (Use the existing harness; SVRAM_TRACE=<file> llama-cli ...)

# 2. Aggregate to layer popularity + assign tiers.
python3 scripts/popularity_to_layers.py \
  --traces results/traces/p1.txt ... p10.txt \
  --policy topk --k 24 \
  --out-prefix results/tiered/layers

# 3. Build the mixed-precision GGUF.
bash scripts/build_tiered_model.sh \
  "$MODELS/Qwen3-30B-A3B-Q8_0.gguf" \
  "$MODELS/Qwen3-30B-A3B-tiered-Q8x24-Q4x24.gguf" \
  "results/tiered/layers.tensor-type-file"

# 4. Verify gates are bit-exact (same check as make_synthetic_model.sh).
#    (Run svram-verify or the existing exactness check.)

# 5. Quality: compare vs Q8_0 reference.
#    (Use compare_quality.py or the existing quality harness.)

# 6. Throughput in the SSD-bound regime.
#    (Run the cold-SSD decode benchmark; compare t/s vs the Q8_0 baseline.)
```

---

## 4. Measurement plan (gate-ordered)

Run these in order. If a gate fails, stop and report — do not proceed to the next.

### 4.1 Gate 1: Byte accounting (tripwire)

After building the tiered GGUF, **dump the per-tensor byte counts** (e.g. `llama-cli --list-models` or parse the GGUF header) and verify:
- Total model size is **smaller** than the all-Q8 source (sanity: Q4 layers are smaller).
- The Q8 layers' total byte count is **approximately equal** to the "layers 0–23" hot-set byte count from the existing synthetic model (within ~5%, since K=24 is chosen to match).
- No tensor is accidentally at the wrong precision (e.g. a Q8 layer showing Q4_K type, or vice versa).

**If this fails:** the tensor-type-file was not applied as expected. Check multi-line precedence (later lines override earlier ones — verify with a 2-layer test file before trusting the full mapping). Stop and fix.

### 4.2 Gate 2: Gate exactness

Verify that the router (gate) weights are **bit-exact** between the tiered model and the Q8_0 source. The gate tensors are non-expert and should be at the same precision in both models. If the gate weights changed, the routing decisions change, and the tier assignment (which was computed from Q8_0 routing) is no longer valid.

**If this fails:** the tensor-type-file accidentally matched a gate tensor. Check the regex. Stop and fix.

### 4.3 Gate 3: Quality

Run the quality comparison (teacher-forced, as in `compare_quality.py`):
- Reference: Q8_0 model, 100 prompts, free-running greedy.
- Candidate: tiered model, `--force-tokens` = reference tokens.
- Report: top-1 agreement %, mean NLL gap, perplexity ratio.

**Pass criteria:**
- Perplexity ratio < 1.05 (i.e. < 5% PPL increase over Q8_0). The full-Q4 model is +2%; the tiered model should be between Q8 and full-Q4, closer to Q8 because the downgraded layers are the less-popular ones.
- Top-1 agreement > 95%.

**If this fails:** the tier assignment is too aggressive. Try a more conservative policy (larger K, or threshold instead of topk). Re-run gates 1–3.

### 4.4 Gate 4: Throughput in the SSD-bound regime

This is the actual test of whether the idea works.

**Setup:** same as the cold-SSD benchmark in `results/rtx3090/cold/`. Model > VRAM, low RAM, SSD is the backing tier. Use the existing 16 GiB cache (or whatever the target regime uses).

**Measurements (3 reps, 256-token decode):**
- Decode t/s for the tiered model.
- Decode t/s for the all-Q8_0 baseline (same cache size, same regime).
- Decode t/s for the existing "layers 0–23" synthetic model (same regime).
- Cache hit rate for each.
- SSD bytes transferred (if available from the trace or `iostat`).

**Pass criteria (the actual win):**
- Tiered model decode t/s ≥ all-Q8_0 baseline decode t/s. (This is the minimum: no regression.)
- Tiered model decode t/s > existing "layers 0–23" synthetic model decode t/s. (This proves the popularity-based selection is better than the arbitrary split.)
- Cache hit rate: the tiered model's hit rate should be **comparable or better** than the all-Q8 baseline. (See §4.5 for the mechanism.)

**If this fails:** the idea doesn't work in this regime. Report the numbers. Do not proceed.

### 4.5 Mechanism note: how the slot budget works (rev 2 correction)

The runtime cache computes a **global** slot budget:
```
n_slots_budget = capacity_bytes / sum_expert_size_all_tensors
```
and applies that same slot count to **every** tensor. Q4 layers don't get *more* slots than Q8 layers; they just cost less bytes per slot.

So the causal chain for the tiered model is:
1. Downgrading some layers to Q4 reduces `sum_expert_size_all_tensors`.
2. This **raises** `n_slots_budget` for everyone (Q8 and Q4 layers alike).
3. More total cached experts → higher hit rate → fewer SSD reads → faster decode.

This is a **global** effect, not a per-layer one. The Q4 layers don't benefit more than the Q8 layers in terms of cache slots; they benefit because their *misses* are cheaper (fewer bytes to fetch from SSD).

**Implication for the §4.4 tripwire:** the expected hit-rate improvement comes from the global slot budget increase, not from "Q4 layers get cached preferentially." If the hit rate doesn't improve, the slot-budget increase wasn't enough. If the hit rate improves but the SSD bytes don't drop proportionally, the Q4 layers' misses aren't actually being served from cache (they're still hitting SSD, just with fewer bytes per read).

### 4.6 (Optional) Gate 5: 43 GiB model

If the tiered model shows a clear win on the 30B model, repeat the test on the 43 GiB model (6.6 t/s, 38% cached) to see if the win scales. The 43 GiB model has a much lower cache coverage ratio, so the byte-savings from Q4 cold layers should matter more there.

---

## 5. The assumption this spec challenges

Rev 1 (and the prior "arbitrary layers 0–23" experiment) treated the *choice* of which layers to downgrade as a free parameter, chosen by hand. This spec challenges that: **popularity data already exists in the routing traces, and using it to select layers is strictly more principled than the existing split.** The cost is one Python script (§3.1) and one line of shell (§3.2). The payoff is a data-driven tier assignment that should (a) have equal or better quality than the arbitrary split, and (b) have equal or better throughput (because the most-popular layers stay at Q8, preserving quality where it matters most).

---

## 6. Load-bearing uncertainties (flagged, must be verified by the implementing agent)

1. **`--tensor-type-file` multi-line precedence.** The spec assumes later lines override earlier ones. Verify with a 2-layer test file (one layer at Q4, one at Q8) before trusting the full mapping. If the precedence is reversed, the mapping will be wrong.

2. **Gate exactness (§4.2).** The tier assignment is computed from Q8_0 routing. If the gate weights change (e.g. because the tensor-type-file accidentally matches a gate tensor), the routing decisions change and the tier assignment is invalid. Verify gates are bit-exact before proceeding.

3. **Slot-budget mechanism (§4.5).** The expected hit-rate improvement comes from the global slot budget increase, not from per-layer preferential caching. If the hit rate doesn't improve as expected, the slot-budget increase wasn't enough — the idea may not work in this regime.

4. **Per-component tiering (deferred to v2, §2.3).** The spec assigns all three components (gate, up, down) of a layer the same precision. This is a simplification. In v2, per-component tiering (e.g. gate at Q4, up/down at Q8) is mechanically possible via `--tensor-type-file` (three separate tensor names per layer) and might give a better quality/throughput tradeoff. Defer for now.

5. **Per-expert tiering (deferred to v2, §2.3).** The spec's rev 1 framing of "per-expert, not per-layer" was the ideal, but the tooling only supports per-layer. True per-expert mixed precision within one tensor would require a GGUF format change or a custom quantize path. Defer to v2.

6. **Trace coverage.** The tier assignment is only as good as the traces it's trained on. If the traces don't cover the target workload (e.g. the traces are from English prompts but the target is code generation), the tier assignment may be suboptimal. Collect traces from the actual target workload before building.

---

## 7. Acceptance checklist

**Status: built and gate-verified on real hardware (2026-09-24), K=24 topk/entropy policy. See `writer_handoff/EVIDENCE_LEDGER.md` section "Popularity-tiered expert precision" for full numbers.**

- [x] `popularity_to_layers.py` runs on the collected traces and produces a valid tensor-type-file + summary JSON. (Uses the corrected `entropy` metric by default, not the degenerate literal-spec `activation_count` -- see §0 correction 4.)
- [x] The tensor-type-file maps the expected layers to Q4_K (not Q4_K_M, see §0 correction 3) and leaves the rest at Q8_0.
- [x] `build_tiered_model.sh` builds the mixed-precision GGUF without errors. (24,061 MiB output.)
- [x] Gate 1 (byte accounting): PASS. 25.23 GB vs 32.48 GB source; 0 tensor-precision mismatches across 144 tensors. (The "layers 0-23 hot set" byte-match criterion is satisfied by construction -- K=24 layers, all layers equal size at Q8_0.)
- [x] Gate 2 (gate exactness): PASS. All 48 router tensors bit-exact (SHA-256).
- [x] Gate 3 (quality): PASS. PPL ratio 1.0073 (< 1.05); top-1 agreement 96.97% (> 95%).
- [x] Gate 4 (throughput): PASS, large margin. Tiered decode t/s 82.4-83.7 vs Q8_0 baseline's 26.30 (~3.1-3.2x), 3 reps. **Also done:** a byte-matched, layer-count-matched arbitrary-split control (`Qwen3-30B-A3B-arbitrary-q8x24-q4x24.gguf`, layers 0-23 hot) was built and benchmarked identically -- popularity selection beats it by **1.26x** (83.25 vs 65.85 t/s mean), confirming the popularity signal earns real throughput headroom beyond just "downgrade some layers." Quality, however, was statistically a wash between the two splits (arbitrary was marginally *better*, not worse) -- see the entropy-hypothesis update below and `writer_handoff/EVIDENCE_LEDGER.md`'s "Popularity-based vs arbitrary layer selection" section for full numbers.
- [x] All numbers are in the evidence ledger with `evidence_class: measured_rtx3090`.

---

## 8. References (optional context for the implementing agent)

- `writer_handoff/EVIDENCE_LEDGER.md` — measured numbers cited in §1.
- `writer_handoff/KNOWN_LIMITATIONS.md` — known constraints (SSD bandwidth, PCIe, cache size).
- `docs/IMPLEMENTATION_STATUS.md` — current implementation state, existing ablations.
- `docs/PLAN_ADAPTIVE.md` — the adaptive-plan context.
- `scripts/make_synthetic_model.sh` — the existing mixed-precision build (the "layers 0–23" arbitrary split).
- `scripts/make_warm_profile.py` — the existing per-expert popularity profiler (input to §3.1).
- `scripts/compare_quality.py` — the quality comparison harness (used in §4.3).
- `third_party/llama.cpp/tools/quantize/quantize.cpp` — `parse_tensor_type` (exact string match on tensor name).
- `third_party/llama.cpp/src/llama-quant.cpp` — `p->pattern` regex matching (C++ side of `--tensor-type-file`).
- `third_party/llama.cpp/src/ggml-backend.cpp` — `ggml_backend_sched_expert_cache_layout` (global slot budget).
