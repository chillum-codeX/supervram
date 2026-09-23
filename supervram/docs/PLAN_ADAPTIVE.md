# SuperVRAM — Game-Changing Improvement: Adaptive Prefetch + Cost-Aware Scheduling

Status: **Phases A-E built and verified in the Python reference harness** (see
`docs/IMPLEMENTATION_STATUS.md`'s "Adaptive speculative prefetch scheduler" section,
`writer_handoff/KNOWN_LIMITATIONS.md` items 14-21, and `writer_handoff/EVIDENCE_LEDGER.md`'s
deterministic-replay and analytical-projection sections for the actual numbers, caveats, and the
formula/state bugs found and fixed while verifying this -- including a Phase E one where the first
diagnosis of a real regression was itself wrong and had to be caught by testing, not assumed).
Section 6's C++/llama.cpp port (v2) is still not built -- the real cache
(`writer_handoff/KNOWN_LIMITATIONS.md` item 8) still has no prefetch.

**The "TODO — Remaining Work" section below is stale and describes work that is already done.**
It appeared in this file mid-session, written (by a different process/session, not this one)
against an exact-signature spec (`step_cost`/`aggregate_costs` functions, `--gate
{off,auto,fixed}`, `--predictor-base`, a separate `--mode ass`) that assumed nothing past the
original Phase A had been built yet. That assumption was wrong by the time it was written: Phases
B-D (and now E) already cover this same scope with different, already-shipped, already-tested
interfaces (`scripts/cost_model.py::project_throughput`, `--scheduler {off,ass}` on the existing
"simulate" mode, `AdaptiveSpeculativeScheduler` wrapping whatever `--predictor` was already
selected rather than a separate `--predictor-base`). Treat the actual interfaces as documented in
`docs/IMPLEMENTATION_STATUS.md` and the source, not the TODO section's proposed signatures --
nobody has reconciled the two, and this note exists so that gap is not silently missed.
Scope: Python reference/simulation package (`supervram/`) + the scripts harness. The C++/llama.cpp
integration is a later port; this plan makes the algorithm correct and *measurably better* in the
deterministic trace-replay harness first, where it can be validated without hardware.

---

## 0. What we are trying to change

Today's system already:
- keeps hot experts in VRAM, pages cold experts through NVMe (3-tier: SSD / pinned-RAM / VRAM),
- evicts with LRU/LFU/weighted/router-aware,
- prefetches with a predictor (history / router-prob / Markov / lightweight / oracle).

Measured reality (see `writer_handoff/KNOWN_LIMITATIONS.md`, `EVIDENCE_LEDGER.md`):
- **SSD bandwidth is the wall.** On the big model, 81% of decode time is waiting on the drive (~2.2 GB/s).
- **Replacement policy has < 1 point of headroom.** LRU ≈ LFU ≈ SLRU ≈ Belady (96.3–97.3%). So a better
  *victim* choice is NOT where the win is.
- **Prediction is weak.** Routing-history predictors are right only 1.4–2% of the time (top 0.1% most
  confident: 12–13%). Wrong reads make prefetch *slower* than none (0.82–0.96×). Oracle gains +4%…+30%,
  but nobody can reach oracle.

**The core insight this plan exploits:** the bottleneck is not "which expert to keep" (nearly optimal) —
it is **"do we issue the SSD read early enough, and only when we're confident it will be used, and does
the read overlap with useful GPU work instead of stalling the kernel."**

Current code already prefetches *speculatively and unconditionally* for every predicted expert, with a
fixed `prefetch_depth`, and the cache blocks (`get` → `future.result()`) whenever a needed expert is not
resident. There is no:
1. **confidence gate** (fire a read only if P(use) × benefit − P(waste) × cost > 0),
2. **lookahead-window scheduling** (issue reads as far ahead as the trace is predictable, amortizing
   latency over D layers/tokens of compute),
3. **overlap accounting** (the harness doesn't model GPU work vs SSD time, so "faster" is unmeasurable),
4. **cost-aware admission/eviction** (we should treat each byte by its *reload cost*, and protect
   experts that are cheap to keep but expensive to fetch).

This plan builds exactly those four, and makes the harness able to *prove* the speedup.

---

## 1. Guiding principles

- **No new third-party deps.** Pure stdlib (as now). The harness stays deterministic.
- **Additive, not a rewrite.** New classes sit beside `policies.py` / `predictors.py`; the old ones keep
  working so we can A/B in the same harness.
- **Measure, don't guess.** Every new knob must show a number in the trace-replay harness (hit rate,
  waste, and — new — modeled end-to-end latency).
- **Deterministic replay** stays the default path so results are reproducible; an `--async-replay`
  wall-clock mode already exists for the async executor.
- **Oracle remains the ceiling.** We report how close the new scheme gets to oracle (the honest metric).

---

## 2. The algorithm (the "game changer")

Name it **AdaptiveSpeculativeScheduler (ASS)** — a *confidence-gated, cost-aware, lookahead* prefetch
scheduler. Four cooperating pieces:

### 2.1 Confidence-gated issuance (kill the wasted reads)
For a candidate expert `e` at layer `L`, predicted D steps ahead, compute:
- `P_use(e)`  — probability the expert will actually be selected before it would be evicted (from the
  predictor, calibrated to history).
- `T_read(e)` — SSD read cost (bytes / drive_throughput). Known from `TensorStore` extent length +
  a drive bandwidth constant.
- `T_evict_window(e)` — how many steps of compute remain before `e` would be needed *or* evicted.

**Fire the read iff** `P_use(e) · T_read(e) < T_compute_budget − cost_of_a_waste(e)`.
In words: issue a read only when the *expected* stall saved exceeds the expected cost of a miss.
This directly attacks the measured "wrong reads make prefetch slower" problem by turning a *blind*
prefetch into an *expected-value* decision. A cheap, near-certain read fires; a big, doubtful one waits.

Implementation: `predictors.py` gains a `ConfidencePredictor` that wraps any base predictor and returns
`(expert, confidence)` pairs; a calibrated threshold (EMA of measured precision) auto-raises/lowers the
gate. `--gate {off,auto,fixed}` + `--gate-threshold` in the harness.

### 2.2 Lookahead-window scheduling (amortize latency)
Instead of prefetching `top-k` for *one* layer, plan across a **window of D layers/tokens**:
- The harness knows the full access order (deterministic replay). We expose a `planner` interface that,
  given the *next D accesses*, returns the set of reads to issue *now*.
- In production the "next D accesses" come from the router probabilities of the in-flight token batch
  (we already receive `probabilities` per access). So the planner is the same code path.
- This is the mechanism that converts SSD latency (hundreds of µs–ms) into *free* time hidden behind
  D layers of compute. It is the single biggest lever for the 81%-SSD-wall.

Implementation: `engine.py` gains `plan_window(accesses_window) -> list[read]`. `cache.py` gains a
scheduled issue path distinct from the current reactive `get()`.

### 2.3 Overlap model in the harness (make speed measurable)
Current harness counts bytes/latency but not the *wall-clock overlap* between GPU work and SSD. Add a
**roofline-style cost model** to the replay:
- `compute_cost(step)` — GPU work for one layer/token (a constant or from model params).
- `io_cost(step)` — bytes read this step × (1/drive_BW).
- `wall_time(step)` = `max(compute_cost, io_cost_serialized) + io_cost_overlapped_fraction`.
- A "naive" baseline serializes; ASS overlaps reads issued D ahead.

This turns the harness output from "hit rate 97%" into **"modeled decode X t/s vs baseline Y t/s"** —
the number that matters. It is an *analytical* projection (clearly labeled, matching
`results/roofline-projection.json`'s evidence class), not a hardware claim.

Implementation: new `scripts/cost_model.py` + a `--cost-model` flag in `simulate_trace.py` that emits
`modeled_tps_baseline` and `modeled_tps_ass` into the JSON result.

### 2.4 Cost-aware admission + protection (the eviction side)
Even though replacement policy has little headroom, *admission* matters: with a fixed slot budget,
ASS should **protect** a small set of *expensive-to-fetch, near-certain* experts (never evict them while
speculative reads are in flight) and **evict** cheap/uncertain ones first.
- Extend `WeightedPolicy` with a `reload_cost` term (already partially present as `reload_ms`) and a
  `pinned` set supplied by the scheduler.
- New `make_policy("adaptive")` = cost-aware + scheduler-protected.

---

## 3. Concrete code changes (file by file)

All under `project/supervram/`. Additive; existing files keep passing their tests.

| File | Change | Kind |
|---|---|---|
| `supervram/types.py` | add `Confidence` dataclass, `ReadDecision` (key, p_use, t_read, fire), `CostModelParams` | new types |
| `supervram/predictors.py` | add `ConfidencePredictor` (wraps base, emits calibrated `(expert, conf)`); keep all existing predictors | additive |
| `supervram/policies.py` | add `AdaptivePolicy(WeightedPolicy)` with `reload_cost` + `pinned` protection; `make_policy("adaptive")` | additive |
| `supervram/cache.py` | add `plan_window(window)`, `issue(ReadDecision)`, overlap counters in `CacheStats` (`overlapped_bytes`, `waste_bytes`); keep `get/prefetch` | additive |
| `supervram/engine.py` | add ASS orchestrator: `plan_window` → gate → issue → overlap accounting; `metrics()` gains scheduler stats | additive |
| `supervram/scheduler.py` | **new** file: `AdaptiveSpeculativeScheduler` — the gate + window planner + cost model glue | new |
| `scripts/cost_model.py` | **new** file: roofline cost model used by the harness to emit modeled t/s | new |
| `scripts/simulate_trace.py` | add flags `--scheduler {off,ass}`, `--gate`, `--gate-threshold`, `--lookahead-d`, `--drive-gbps`, `--cost-model`; emit new metrics | additive |
| `scripts/run_ablations.py` | add ASS axes to the matrix (scheduler × gate × lookahead) | additive |
| `tests/test_scheduler.py` | **new**: unit tests — gate fires only above threshold; window issues the right set; overlap model monotonic in drive BW; adaptive policy protects pinned expert; oracle-gap reported | new |
| `tests/test_supervram.py` | keep as-is (must still pass) | unchanged |
| `supervram/__init__.py` | export new public names (`AdaptivePolicy`, `AdaptiveSpeculativeScheduler`, `ConfidencePredictor`) | additive |

No C++ changes in v1. (Later: port `scheduler.py` logic into the `ggml_backend_sched` expert cache in
`llama_cpp_modified/`, behind `SVRAM_SCHEDULER=ass`.)

---

## 4. Algorithm specification (pseudo)

```
# per access step t, we may see the next D accesses (deterministic replay gives us this;
# in prod the router probabilities of the in-flight batch approximate it)
window = accesses[t : t + D]

for (layer, experts, probs) in window:
    candidates = predict(layer, probs)           # base predictor
    for e, p_raw in candidates:
        conf  = calibrate(p_raw)                 # ConfidencePredictor
        t_read = extent_len(e) / drive_bw
        budget = compute_time(window)            # 2.3
        decision = ReadDecision(e, p_use=conf, t_read=t_read,
                                fire = conf * t_read <= budget - waste_cost(e))
        if decision.fire and e not resident and e not in_flight:
            cache.issue(decision)                # schedule SSD read now (2.1)

# while GPU does window's compute:
#   in-flight reads complete and are admitted; overlap = min(io, compute) (2.3)

# admission/eviction:
#   protect pinned = {e: p_use high AND t_read high}
#   evict cheapest uncertain first (2.4)
```

Calibration (EMA): `conf = alpha * conf + (1-alpha) * (was_right ? 1 : 0)` per (layer,expert), so the
gate self-tunes to the model's real predictability and never wastes the drive on low-confidence guesses.

---

## 5. Success criteria (must all be met to call it "done")

1. **Correctness unchanged.** `tests/test_supervram.py` still passes; new `tests/test_scheduler.py`
   passes (pytest green).
2. **Gate reduces waste.** In the harness, with the existing weak predictors, ASS's `waste_bytes`
   (wrong reads) is strictly lower than blind prefetch at equal `--prefetch-depth`.
3. **Modelled speedup.** With `--cost-model`, ASS reports `modeled_tps_ass >= modeled_tps_baseline`,
   and the gap widens as `--drive-gbps` drops (i.e., the gain is *because* of overlap, not magic).
4. **Closer to oracle.** Report `modeled_tps_ass` vs `modeled_tps_oracle`; ASS must be meaningfully
   closer to oracle than the current best (markov + router-aware) in the same run.
5. **No regressions.** `--scheduler off` reproduces current numbers exactly (byte-for-byte metrics).

---

## 6. Out of scope (v1)

- C++/llama.cpp port (v2). GDS device DMA. Prompt-batch cache. Energy/Nsight.
- Learned (neural) predictors — keep it online/analytic and dependency-free; oracle is the reference.
- Multi-drive aggregation (the box has a 2nd NVMe) — could be a later bandwidth knob.

---

## 7. Open questions (answer before coding, or default as noted)

- Default `--lookahead-d`: start at `D = 4` layers (matches the `prefetch_predictor_eval.py` sweep that
  found 1 layer = +4%, whole-token = +30% for oracle). **Default: D=4.**
- Drive BW constant for the cost model: **default 2.4 GB/s** (measured `~1991 MiB/s` O_DIRECT ceiling,
  conservative vs 2.4).
- Gate default: **`auto`** (EMA calibration), threshold floor 0.3.
- Should `plan_window` be allowed to *reorder* reads (larger first to saturate the drive)? **Yes** — sort
  in-flight reads by size desc, matching the measured "layer-batched direct reads" win (26.3→30.5 t/s).

---

## 8. Rollout

1. **Phase A — types + predictor + policy** (`types.py`, `predictors.py`, `policies.py`) + unit tests.
   Deliverable: `make_policy("adaptive")` and `ConfidencePredictor` importable and tested.
2. **Phase B — scheduler + cache window path** (`scheduler.py`, `cache.py`, `engine.py`) + tests.
   Deliverable: `--scheduler ass` runs in `simulate_trace.py`, reduces `waste_bytes` vs baseline.
3. **Phase C — cost model + harness + ablations** (`cost_model.py`, `simulate_trace.py`, `run_ablations.py`)
   + tests. Deliverable: modeled t/s reported; ablation CSV shows the win and the oracle gap.
4. **Phase D — docs + handoff** (update `docs/IMPLEMENTATION_STATUS.md`, `KNOWN_LIMITATIONS.md`,
   `EVIDENCE_LEDGER.md` with the new numbers and evidence class).

Each phase is independently shippable and tested; nothing is a big-bang.

---

## TODO — Remaining Work

This section lists every piece that is **not yet built**. Each item is self-contained, has an explicit
start-to-finish definition, and can be done in any order. The harness (`scripts/cost_model.py`,
`scripts/simulate_trace.py`, `scripts/run_ablations.py`) and the test file (`tests/test_scheduler.py`)
are the only missing pieces; everything in `supervram/` is already written and tested against the
existing `tests/test_supervram.py` suite.

### TODO-1: `scripts/cost_model.py` — roofline overlap cost model

**Why:** the harness currently reports "hit rate 97%" but not modeled throughput. Without a cost model,
we cannot prove that ASS's overlap actually helps — only that it wastes fewer bytes. The cost model
turns `bytes_read`, `overlapped_bytes`, `waste_bytes`, and `compute_cost` into a **modeled decode
tokens/second** figure so we can say "ASS is X% faster than baseline" in the harness output.

**What it must do:**

- **Inputs (all per-step):**
  - `compute_cost_ns` — GPU compute time per layer/token (a constant; default `0.19 ms` per
    `CostModelParams`, derived from the project's own `compute_ms_per_layer` evidence).
  - `io_bytes` — total bytes read this step (from `cache.stats.bytes_read`).
  - `overlapped_bytes` — bytes that were already resident before the GPU needed them (from
    `cache.stats.overlapped_bytes` — these contribute zero stall time).
  - `waste_bytes` — bytes read for experts that were never used (from `cache.stats.waste_bytes` —
    these contribute full stall time with no benefit).
  - `drive_throughput_gbps` — drive ceiling (default `2.4 GB/s` from `CostModelParams`; users can
    override via `--drive-gbps`).

- **Computation per step:**
  1. `io_total_ns = ns_for_drive_bytes(io_bytes - overlapped_bytes, drive_throughput_gbps)` — only the
     bytes that were NOT already overlapped cost time.
  2. `waste_stall_ns = ns_for_drive_bytes(waste_bytes, drive_throughput_gbps)` — wrong reads add pure
     latency.
  3. `effective_compute_ns = compute_cost_ns + waste_stall_ns` — waste adds to the wall-clock time
     the GPU "waits" for the host.
  4. `step_wall_ns = max(compute_cost_ns, io_total_ns) + waste_stall_ns` — the serial overlap model:
     SSD latency and GPU compute run in parallel, but waste is added on top (it cannot be overlapped
     because it was never issued early enough to matter).

- **Aggregation:**
  - `total_compute_ns = sum(compute_cost_ns)` over all steps.
  - `total_io_ns = sum(max(compute_cost_ns, io_total_ns))` — the parallel wall-clock.
  - `total_waste_ns = sum(waste_stall_ns)`.
  - `total_wall_ns = total_compute_ns + total_io_ns + total_waste_ns`.
  - `modeled_tps = total_tokens / (total_wall_ns / 1e9)`.

- **Output:** a dict matching `CostModelMetrics` but at the aggregate level:
  ```
  {
    "total_compute_ns": int,
    "total_io_ns": int,
    "total_waste_ns": int,
    "total_wall_ns": int,
    "total_bytes_read": int,
    "total_overlapped_bytes": int,
    "total_waste_bytes": int,
    "modeled_tps": float,
    "evidence_class": "analytical_projection"
  }
  ```

- **Function signatures (exact):**
  ```python
  def step_cost(
      compute_cost_ns: int,
      io_bytes: int,
      overlapped_bytes: int,
      waste_bytes: int,
      drive_throughput_gbps: float,
  ) -> CostModelMetrics: ...

  def aggregate_costs(
      per_step_costs: Sequence[CostModelMetrics],
      total_tokens: int,
  ) -> dict: ...  # returns the dict above
  ```

- **Edge cases:**
  - `overlapped_bytes > io_bytes` (should not happen, but cap to `io_bytes`).
  - `drive_throughput_gbps == 0` (return `inf` or `0` for tps, documented).
  - `io_bytes == 0` (step is pure compute, no IO stall).

- **Tests (in `tests/test_scheduler.py`):**
  - `test_step_cost_pure_compute` — 0 bytes read → `modeled_tps = 1 / compute_ms_per_layer`.
  - `test_step_cost_pure_io` — no compute, only IO → `modeled_tps = drive_throughput / bytes_per_token`.
  - `test_step_cost_with_waste` — adding waste_bytes increases `total_wall_ns` and decreases `modeled_tps`.
  - `test_step_cost_overlapped_reduces_stall` — if all bytes are overlapped, `io_total_ns == 0`, no
    stall from IO.
  - `test_aggregate_monotonic_in_drive_bw` — doubling `drive_throughput_gbps` never decreases `modeled_tps`
    (proves the model is well-behaved).
  - `test_aggregate_modeled_tps_formula` — verify `total_tokens / (total_wall_ns / 1e9)` matches.

---

### TODO-2: `scripts/simulate_trace.py` — harness flags for ASS + cost model

**Why:** the current harness runs blind prefetch (`--prefetch-depth`) or nothing. It has no way to
invoke `AdaptiveSpeculativeScheduler`, pass its parameters, or report the cost-model output.

**What it must add:**

- **New CLI flags:**
  ```
  --scheduler {off,ass}     # default "off" for baseline; "ass" enables AdaptiveSpeculativeScheduler
  --gate {off,auto,fixed}   # default "auto" (EMA calibration); "off" = fire all; "fixed" = use --gate-threshold
  --gate-threshold float    # default 0.3; floor confidence to fire
  --lookahead-d int         # default 4; window size for plan_window
  --drive-gbps float        # default 2.4; drive throughput for cost model
  --cost-model              # flag: compute and emit modeled throughput
  --predictor-base name     # the base predictor the ASS wraps (e.g., "markov", "router-probability")
  ```

- **Logic changes (in `main()`):**
  1. If `--scheduler ass`:
     - Create `ConfidencePredictor(base=make_predictor(args.predictor_base))` — this wraps the chosen
       base predictor (default `markov` if not specified).
     - Create `AdaptiveSpeculativeScheduler(confidence_predictor, extent_length=..., params=...)`.
     - Use `engine.process_window(window)` instead of `engine.process(access)`.
     - `window = accesses[i : i + lookahead_d]`; emit `process_window(window, scheduler)` for each
       step, where `i` increments by 1 (sliding window).
     - Call `scheduler.record_and_observe(access)` after resolving each step (mirrors the
       `predict → compare → record` rhythm of the plan's pseudo-code).

  2. If `--cost-model` (either scheduler):
     - Collect `cache.stats.bytes_read`, `cache.stats.overlapped_bytes`, `cache.stats.waste_bytes`
       per step.
     - After the replay loop, call `cost_model.aggregate_costs(per_step_costs, total_tokens)` and
       add the result to the output JSON under `"cost_model"`.

  3. JSON output additions (at the top level of the result dict, alongside `"metrics"`):
     ```json
     {
       "scheduler": "off" | "ass",
       "scheduler_config": { "gate": "auto", "threshold": 0.3, "lookahead_d": 4 },
       "cost_model": { ... }  // only if --cost-model; omitted otherwise
     }
     ```

- **Default predictor-base:** if `--predictor-base` is not specified and `--scheduler ass`, default to
  `"markov"` (the strongest non-oracle predictor in the current harness). If `--scheduler off`, ignore
  this flag.

- **Backwards compatibility:** `--scheduler off` must reproduce **byte-for-byte identical** metrics
  to the existing harness behavior (same hit rate, same bytes_read, same everything). This is a success
  criterion (section 5.5 of the plan).

- **Tests (in `tests/test_scheduler.py`):**
  - `test_scheduler_off_matches_baseline` — run the harness with `--scheduler off` and `--scheduler ass`
    (with a trivial no-op scheduler) and assert metrics are identical.
  - `test_scheduler_ass_emits_cost_model` — run with `--cost-model` and verify the JSON has a `"cost_model"`
    key with the right structure (modeled_tps, evidence_class, etc.).
  - `test_scheduler_ass_with_markov` — a short trace, `--scheduler ass --predictor-base markov`; assert
    `waste_bytes` < baseline and `modeled_tps` > baseline (or at least not worse).
  - `test_scheduler_ass_with_router` — same, but `--predictor-base router-probability`; assert the gate
    actually suppresses some reads (fire_rate < 1.0).

---

### TODO-3: `scripts/run_ablations.py` — ASS axes in the ablation matrix

**Why:** the current ablation matrix sweeps `cache_gib × prefetch × policy × predictor` (720 combinations
in simulation mode). To prove ASS is a real win, we need an ablation axis that sweeps:
`scheduler(off/ass) × gate(auto/fixed) × lookahead-d(2,4,8) × cost-model(on/off)`.

**What it must add:**

- **New constants (near the top of the file, after the existing ones):**
  ```python
  ASS_SCHEDULERS = ["off", "ass"]
  ASS_GATES = ["auto", "fixed"]
  ASS_LOOKAHEAD_D = [2, 4, 8]
  ASS_COST_MODEL = [True, False]
  ```

- **New mode (or a new axis in the existing "simulate" mode):**
  Add an `"ass"` mode to `--mode`:
  ```
  --mode {plan, simulate, target, ass}
  ```
  - `"plan"`: print the planned combinations (dry run).
  - `"sim" or "simulate"`: existing axes (unchanged).
  - `"target"`: real-hardware `svram-verify` sweeps (unchanged).
  - `"ass"`: new ASS axes.

- **Command construction for ASS runs:**
  ```python
  command = [
      sys.executable,
      str(Path(__file__).with_name("simulate_trace.py")),
      "--scheduler", scheduler,        # "off" or "ass"
      "--gate", gate,                  # "auto" or "fixed"
      "--gate-threshold", "0.3",       # fixed for all runs
      "--lookahead-d", str(lookahead_d),
      "--cost-model",                  # present if cost_model is True
      "--predictor-base", "markov",    # fixed baseline
      "--drive-gbps", "2.4",           # fixed
      "--cache-bytes", str(cache_bytes),
      "--expert-bytes", str(expert_bytes),
      "--prefetch-depth", str(prefetch_depth),  # only for scheduler=off; 0 for ass
      "--policy", policy,
      "--output", str(result_path),
  ]
  ```

- **Record fields for the manifest:**
  Each ASS run record in the manifest must include:
  ```python
  {
      "run_id": str,
      "scheduler": "off" | "ass",
      "gate": "auto" | "fixed",
      "lookahead_d": int,
      "cache_gib": int,
      "prefetch_depth": int,
      "policy": str,
      "cost_model": bool,
      "modeled_tps": float | null,    # from cost_model output, null if --cost-model not set
      "waste_bytes": int,             # from scheduler stats or cache stats
      "fire_rate": float | null,      # from scheduler stats
      "hit_rate": float,              # from cache stats
      ...
  }
  ```

- **CSV output:** extend `manifest.csv` fields to include the new axes (same CSV, more columns).

- **Dry-run / plan mode:** `--mode plan --mode ass` should print every planned command and its
  arguments so the user can review before running.

- **Tests (in `tests/test_scheduler.py`):**
  - `test_ablation_ass_mode_plans_correctly` — `--mode ass --max-runs 3` → verify the manifest has
    the right number of runs with the right fields.
  - `test_ablation_ass_mode_runs_all_combinations` — `--mode ass --max-runs 0` → verify the manifest
    has `len(ASS_SCHEDULERS) * len(ASS_GATES) * len(ASS_LOOKAHEAD_D) * len(ASS_COST_MODEL) *
    len(CACHE_GIB) * len(PREFETCH) * len(POLICIES) * len(PREDICTORS)` runs (or fewer if max-runs caps it).
  - `test_ablation_ass_mode_cost_model_emitted` — verify at least one run has `"cost_model"` in its
    output JSON (i.e., the `--cost-model` flag was passed).

---

### TODO-4: `tests/test_scheduler.py` — full unit test suite

**Why:** every new class and function must be tested in isolation before the harness integration.
These tests are deterministic, fast, and use in-memory mock stores (no temp files needed for most
of them). The existing `tests/test_supervram.py` must still pass.

**File structure:** one file, `tests/test_scheduler.py`, grouped by component.

#### Group A: `ConfidencePredictor` tests (predictors.py)

| Test name | What it asserts |
|---|---|
| `test_confidence_predictor_wraps_base` | `ConfidencePredictor(HistoryPredictor()).predict(...)` returns the same list as the unwrapped base. |
| `test_confidence_predictor_predict_confident` | `predict_confident(layer, top_k, probs)` returns a list of `Confidence` objects, one per predicted expert. |
| `test_confidence_predictor_ema_calibration` | Simulate 100 accesses where expert 0 is correct 80% of the time; after calibration, `confidence_of(layer, 0) ≈ 0.8`. |
| `test_confidence_predictor_ema_calibration_wrong` | Simulate 100 accesses where expert 1 is correct 20% of the time; after calibration, `confidence_of(layer, 1) ≈ 0.2`. |
| `test_confidence_predictor_default_confidence` | Before any calibration, `confidence_of(layer, any_expert) == default_confidence` (0.5 by default). |
| `test_confidence_predictor_alpha_bounds` | `alpha < 0` or `alpha > 1` raises `ValueError`. |
| `test_confidence_predictor_observe_cleans_pending` | After `observe`, `_pending[layer]` is empty (no stale predictions left behind). |

#### Group B: `AdaptivePolicy` tests (policies.py)

| Test name | What it asserts |
|---|---|
| `test_adaptive_policy_returns_non_pinned_victims` | With 3 entries, 2 pinned, 1 not pinned, `victims(...)` returns only the unpinned one. |
| `test_adaptive_policy_no_victims_when_all_pinned` | All entries pinned → `victims` raises `MemoryError` (nothing to free). |
| `test_adaptive_policy_pin_unpin` | `pin([k1])`, then `pin([k2])` → both pinned; `unpin([k1])` → k1 unpinned, k2 still pinned. |
| `test_adaptive_policy_evicts_lowest_value_first` | With 3 entries having different `value()` scores, `victims(...)` returns them in ascending value order until `bytes_needed` is met. |
| `test_adaptive_policy_is_a_weighted_policy` | `isinstance(AdaptivePolicy(), WeightedPolicy)` is `True`; all `WeightedPolicy` fields work (probability_weight, frequency_weight, etc.). |

#### Group C: `AdaptiveSpeculativeScheduler` tests (scheduler.py)

| Test name | What it asserts |
|---|---|
| `test_scheduler_wraps_or_creates_confidence_predictor` | Passing a plain `Predictor` → `scheduler.confidence` is a `ConfidencePredictor` wrapping it. Passing a `ConfidencePredictor` → `scheduler.confidence` is that exact instance (no double-wrap). |
| `test_scheduler_plan_window_empty` | `plan_window([])` → `[]`. |
| `test_scheduler_plan_window_below_threshold` | All candidates have confidence < `gate_threshold` → `len([d for d in decisions if d.fire]) == 0`. |
| `test_scheduler_plan_window_fires_above_threshold` | All candidates have confidence > `gate_threshold` and budget is infinite → `all(d.fire for d in decisions)`. |
| `test_scheduler_plan_window_budget_exhaustion` | Budget is tight (e.g., `compute_ms_per_layer=0.0001`, window len=4) → only 1–2 reads fire; rest gated. |
| `test_scheduler_plan_window_sorts_by_confidence_then_size` | Pool sorted `(conf desc, read_size desc)`; decisions appear in that order. |
| `test_scheduler_plan_window_pool_limited_to_top_k` | Window of 4 accesses, each with top_k=4 candidates → `len(pool) == top_k` (4), not `4 * 4 = 16`. |
| `test_scheduler_record_and_observe_updates_calibration` | Before observe: `confidence_of == default`; after 100 correct observes: `confidence_of > 0.7`. |
| `test_scheduler_t_read_seconds` | `t_read_seconds(key)` returns `extent_length(key) / (drive_gbps * 1e9)`. Verify with a known extent length and drive BW. |
| `test_scheduler_t_read_seconds_missing_key` | Unknown `ExpertKey` → `t_read_seconds` returns `0.0` (no KeyError). |

#### Group D: `Cache.issue()` / `CacheStats` tests (cache.py)

| Test name | What it asserts |
|---|---|
| `test_issue_fires_when_true` | `cache.issue(ReadDecision(key, ..., fire=True))` → `cache.stats.prefetches += 1`, key enters inflight. |
| `test_issue_ignores_when_false` | `cache.issue(ReadDecision(key, ..., fire=False))` → `cache.stats.prefetches` unchanged, key never inflight. |
| `test_issue_noop_when_already_inflight` | Same key prefetched twice → second `issue` is a no-op (same as `prefetch` dedup logic). |
| `test_stats_overlapped_bytes_on_hit` | After `cache.get(key)` where key was previously overlapped, `cache.stats.overlapped_bytes += key.size`. |
| `test_stats_waste_bytes_on_eviction_of_prefetch` | After an eviction of a key that was in `prefetched`, `cache.stats.waste_bytes += key.size`. |

#### Group E: `SuperVRAM.process_window()` tests (engine.py)

| Test name | What it asserts |
|---|---|
| `test_process_window_resolves_window_zero` | `process_window([access, ...], scheduler)` resolves `access` through the cache exactly like `process(access)` (same payloads, same prediction stats). |
| `test_process_window_scheduler_record_called` | After `process_window`, `scheduler.record_and_observe(window[0])` has been called (verify via a mock or by checking calibration state). |
| `test_process_window_scheduler_plan_called` | After `process_window`, `scheduler.plan_window(window[1:])` has been called and its decisions issued (verify via a mock or by checking `cache.stats.prefetches`). |
| `test_process_window_empty_returns_empty` | `process_window([], scheduler)` → `[]`. |
| `test_metrics_includes_scheduler_stats` | After `process_window`, `engine.metrics()["scheduler"]` has `fired`, `gated`, `fire_rate`. Without ASS, `metrics()["scheduler"]` is absent. |
| `test_process_vs_process_window_equivalence_off` | With `--scheduler off` (or no scheduler), `process(access)` and `process_window([access], scheduler)` produce identical `metrics()`. |

#### Group F: `CostModel` tests (cost_model.py — in the same file for now, moves later)

| Test name | What it asserts |
|---|---|
| `test_step_cost_pure_compute` | `io_bytes=0` → `io_total_ns=0`, `total_wall_ns = compute_cost_ns`, `modeled_tps = tokens / (compute_ms_per_layer)`. |
| `test_step_cost_pure_io` | `compute_cost_ns=0`, only `io_bytes` → `total_wall_ns = io_total_ns`, `modeled_tps = drive_throughput_gbps / bytes_per_token`. |
| `test_step_cost_with_waste_increases_stall` | Adding `waste_bytes > 0` increases `total_wall_ns` and decreases `modeled_tps` vs the same run with `waste_bytes=0`. |
| `test_step_cost_overlapped_reduces_stall` | `overlapped_bytes = io_bytes` → `io_total_ns = 0`, `modeled_tps` equals pure-compute tps (no IO stall at all). |
| `test_aggregate_monotonic_in_drive_bw` | Doubling `drive_throughput_gbps` never decreases `modeled_tps` (proves the model is well-behaved). |
| `test_aggregate_modeled_tps_formula` | `modeled_tps == total_tokens / (total_wall_ns / 1e9)` within floating-point tolerance. |

---

### TODO-5: Update harness integration in `scripts/simulate_trace.py` (full function signatures)

**This is the "glue" — the actual function signatures and call sites in the harness, not just flags.**
Once TODO-2 is done, the harness wiring is trivial. This section is here to make the interface
explicit so there is no ambiguity.

**In `main()` (the only place that needs changes):**

```python
# --- New imports (add to existing imports) ---
from supervram.predictors import ConfidencePredictor
from supervram.scheduler import AdaptiveSpeclicativeScheduler
from scripts.cost_model import step_cost, aggregate_costs

# --- New argument parsing (add after existing parser) ---
parser.add_argument("--scheduler", default="off", choices=["off", "ass"])
parser.add_argument("--gate", default="auto", choices=["off", "auto", "fixed"])
parser.add_argument("--gate-threshold", type=float, default=0.3)
parser.add_argument("--lookahead-d", type=int, default=4)
parser.add_argument("--drive-gbps", type=float, default=2.4)
parser.add_argument("--cost-model", action="store_true")
parser.add_argument("--predictor-base", default=None)

# --- New logic block (before the replay loop, after predictor creation) ---
accesses = generate_trace(...)  # unchanged
predictor = make_predictor(args.predictor, ...)  # unchanged

# --- ASS setup (new) ---
if args.scheduler == "ass":
    base_predictor_name = args.predictor_base or "markov"
    base_predictor = make_predictor(base_predictor_name)
    confidence_predictor = ConfidencePredictor(base_predictor, alpha=0.9)
    extent_length_fn = lambda key: args.expert_bytes  # all experts same size in synthetic
    scheduler = AdaptiveSpeculativeScheduler(
        predictor=confidence_predictor,
        extent_length=extent_length_fn,
        params=CostModelParams(
            drive_gbps=args.drive_gbps,
            lookahead_d=args.lookahead_d,
            gate_threshold=args.gate_threshold,
        ),
        top_k=args.prefetch_depth if args.prefetch_depth else 4,
    )
else:
    scheduler = None

# --- Replay loop (replaced) ---
per_step_costs = []
start = time.perf_counter_ns()
with TensorStore(store_path) as store, ExpertCache(...) as cache:
    if args.scheduler == "ass":
        engine = SuperVRAM(cache, predictor, args.prefetch_depth)
        for i in range(0, len(accesses), 1):  # slide by 1, not by prefetch_depth
            window = accesses[i : i + args.lookahead_d]
            engine.process_window(window, scheduler)
            # collect per-step cost data for the cost model
            step_io = ...  # read from cache.stats at this point
            step_cost_record = step_cost(
                compute_cost_ns=int(params.compute_ms_per_layer * 1e6),
                io_bytes=step_io,
                overlapped_bytes=cache.stats.overlapped_bytes,
                waste_bytes=cache.stats.waste_bytes,
                drive_throughput_gbps=args.drive_gbps,
            )
            per_step_costs.append(step_cost_record)
            cache.drain()  # drain inflight before advancing window
    else:
        # existing process() path (unchanged)
        for access in accesses:
            engine.process(access)
            cache.drain()

# --- Cost model output (new) ---
cost_model_result = None
if args.cost_model:
    total_tokens = len(accesses)
    cost_model_result = aggregate_costs(per_step_costs, total_tokens)

# --- Result dict (new fields) ---
result = {
    ...
    "scheduler": args.scheduler,
    "scheduler_config": {
        "gate": args.gate,
        "threshold": args.gate_threshold,
        "lookahead_d": args.lookahead_d,
    },
    "cost_model": cost_model_result,
}
```

---

### TODO-6: Update `scripts/run_ablations.py` (full function signatures)

**The existing `run_ablations.py` has `mode="plan" | "simulate" | "target"`.** Add `"ass"` mode.

```python
# --- New constants (after existing ones) ---
ASS_SCHEDULERS = ["off", "ass"]
ASS_GATES = ["auto", "fixed"]
ASS_LOOKAHEAD_D = [2, 4, 8]
ASS_COST_MODEL = [True, False]

# --- New mode handler (in main(), after the existing "target" branch) ---
if args.mode == "ass":
    # Build the ASS axes
    combinations = list(itertools.product(
        ASS_SCHEDULERS, ASS_GATES, ASS_LOOKAHEAD_D,
        ASS_COST_MODEL, CACHE_GIB, PREFETCH, POLICIES, PREDICTORS
    ))
    if args.max_runs:
        combinations = combinations[:args.max_runs]
    
    for index, (scheduler, gate, lookahead_d, cost_model,
                cache_gib, depth, policy, predictor) in enumerate(combinations):
        run_id = f"ass-run-{index:04d}-{scheduler}-g{gate}-d{lookahead_d}-cm{cost_model}-c{cache_gib}-{policy}-{predictor}"
        result_path = args.output_dir / f"{run_id}.json"
        cache_bytes = cache_gib * 1024**3
        expert_bytes = 64 * 1024
        
        command = [
            sys.executable,
            str(Path(__file__).with_name("simulate_trace.py")),
            "--scheduler", scheduler,
            "--gate", gate,
            "--gate-threshold", "0.3",
            "--lookahead-d", str(lookahead_d),
            "--cost-model" if cost_model else "",  # present only when True
            "--predictor-base", predictor,  # use predictor as base for ASS
            "--drive-gbps", "2.4",
            "--cache-bytes", str(cache_bytes),
            "--expert-bytes", str(expert_bytes),
            "--prefetch-depth", str(depth) if scheduler == "off" else "0",
            "--policy", policy,
            "--output", str(result_path),
        ]
        # remove empty strings
        command = [c for c in command if c]
        
        record = run(command, args.output_dir / f"{run_id}.status.json", args.mode == "plan")
        
        # Parse cost model output if present
        parsed = {}
        if cost_model and record.get("status") == "ok":
            output_text = record.get("stdout", "")
            try:
                import json
                cost_output = json.loads(output_text.split("---")[-1]) if "---" in output_text else json.loads(output_text)
                parsed["modeled_tps"] = cost_output.get("cost_model", {}).get("modeled_tps")
                parsed["total_wall_ns"] = cost_output.get("cost_model", {}).get("total_wall_ns")
            except:
                parsed["modeled_tps"] = None
        
        record["evidence_class"] = "synthetic_ass_ablation"
        record.update(parsed)
        record["scheduler"] = scheduler
        record["gate"] = gate
        record["lookahead_d"] = lookahead_d
        record["cost_model"] = cost_model
        
        manifest.append({"run_id": run_id, "scheduler": scheduler, "gate": gate,
                         "lookahead_d": lookahead_d, "cost_model": cost_model,
                         "cache_gib": cache_gib, "prefetch_depth": depth,
                         "policy": policy, "predictor": predictor, **record})
```

---

### TODO-7: Integration test — end-to-end harness run

**Why:** after all the individual pieces above, we need a single test (or manual command) that proves
the whole pipeline works: generate a trace → run with `--scheduler ass` → emit cost model → compare to
baseline.

**Test name:** `test_end_to_end_ass_pipeline` (in `tests/test_scheduler.py`, or as a standalone
script that pytest calls).

**What it does:**
1. Generates a synthetic trace (same as `simulate_trace.py` does): 64 tokens, 8 layers, 32 experts,
   top-k 4, locality 0.8.
2. Creates a synthetic store (same as `simulate_trace.py` does): 64 KB per expert.
3. Runs the harness twice:
   - Run A: `--scheduler off --predictor markov --cost-model`
   - Run B: `--scheduler ass --predictor-base markov --cost-model`
4. Asserts:
   - Both runs produce a `"cost_model"` key in the output.
   - Run B's `modeled_tps` ≥ Run A's `modeled_tps` (or at least not strictly worse; if the predictor
     is weak, the gate should suppress enough waste to compensate).
   - Run B's `waste_bytes` ≤ Run A's `waste_bytes`.
   - Run B's `scheduler.fire_rate` is between 0.0 and 1.0 (not all fire, not none fire — the gate
     is doing something).

**Alternative (faster) version:** mock the `TensorStore` to return tiny blobs (1 byte each), use a
1-layer, 4-expert, 16-token trace. Assert the same things with deterministic results.

---

### TODO-8: Documentation update — `docs/IMPLEMENTATION_STATUS.md`, `writer_handoff/` files

**Why:** the plan lives in `PLAN_ADAPTIVE.md`, but the project's "source of truth" is
`IMPLEMENTATION_STATUS.md`, `KNOWN_LIMITATIONS.md`, and `EVIDENCE_LEDGER.md`. After the harness
produces actual numbers, these files must be updated so future contributors know the real status.

**What to add to each file:**

#### `docs/IMPLEMENTATION_STATUS.md` — new section "Adaptive speculative prefetch scheduler"
```markdown
## Adaptive speculative prefetch scheduler (v1)

Status: **Python reference harness complete; C++/llama.cpp port pending.**

Built:
- `supervram/scheduler.py`: `AdaptiveSpeculativeScheduler` (confidence gate, window planner,
  observe loop).
- `supervram/predictors.py`: `ConfidencePredictor` (EMA calibration wrapper).
- `supervram/policies.py`: `AdaptivePolicy` (pin protection + reload-cost eviction).
- `supervram/cache.py`: `issue()` scheduled read path, `overlapped_bytes` / `waste_bytes` stats.
- `supervram/engine.py`: `process_window()` ASS orchestration.
- `scripts/cost_model.py`: roofline overlap cost model (modeled t/s).
- `scripts/simulate_trace.py`: `--scheduler {off,ass}`, `--cost-model`, `--lookahead-d`, etc.
- `tests/test_scheduler.py`: full unit test suite (20+ tests).

Known results (harness, synthetic traces):
- Gate suppresses X% of speculative reads vs blind prefetch.
- ASS modeled t/s: Y vs baseline Z (X% improvement).
- Oracle gap: ASS is A% closer to oracle than the best existing predictor.

Remaining:
- C++/llama.cpp integration (behind `SVRAM_SCHEDULER=ass`).
- Real-hardware validation on Qwen3-30B-A3B.
- Multi-predictor sweep (markov, lightweight, router-probability as base for ASS).
```

#### `writer_handoff/KNOWN_LIMITATIONS.md` — add items 14–18
```markdown
14. Adaptive speculative prefetch scheduler (`--scheduler ass`) is implemented in the Python harness
    but not in the C++/llama.cpp cache (the real cache still has no prefetch).
15. The confidence gate (EMA calibration) has not been calibrated on real Qwen routing data;
    defaults use synthetic traces (may not generalize).
16. The cost model is an analytical projection, not measured throughput; real GPU/SSD overlap
    may differ.
17. ASS has not been tested on multi-request concurrent batches (only single-request sliding window).
18. `process_window()` drains inflight before advancing the window by 1; this means each step
    waits for all speculative reads from the previous window to complete. In production, a true
    double-buffered staging ring would allow fully overlapped reads.
```

#### `writer_handoff/EVIDENCE_LEDGER.md` — new section "Adaptive speculative prefetch scheduler — harness results"
```markdown
## Adaptive speculative prefetch scheduler — harness results

Evidence class: `synthetic_ass_ablation`

Run: `scripts/run_ablations.py --mode ass --output-dir results/ass-ablation/`
Trace: 64 tokens, 8 layers, 32 experts, top-k 4, locality 0.8, seed 7.

| scheduler | gate | lookahead_d | modeled_tps | waste_bytes | fire_rate | hit_rate |
|-----------|------|-------------|-------------|-------------|-----------|----------|
| off       | -    | -           | X.XX        | YYY         | -         | Z.ZZ%    |
| ass       | auto | 4           | X.XX        | YYY         | 0.XX      | Z.ZZ%    |
| ass       | auto | 8           | X.XX        | YYY         | 0.XX      | Z.ZZ%    |
| ass       | fixed| 0.3         | X.XX        | YYY         | 0.XX      | Z.ZZ%    |

(Replace X.XX, YYY, Z.ZZ with actual numbers after running.)

Caveats:
- Synthetic traces, not real Qwen routing.
- Cost model is analytical; real hardware may differ.
- One model configuration per run; no multi-model sweep.
```

---

### TODO-9: C++ / llama.cpp port (v2 — out of scope for this plan, but listed for completeness)

**Why:** the Python harness proves the algorithm works. The real win requires it in `llama.cpp`.

**What it would involve (not built in v1, documented for later):**
1. Port `AdaptiveSpeculativeScheduler` logic into `ggml_backend_sched`'s expert cache path.
2. Add `--moe-expert-scheduler {off,ass}` CLI flag (parallel to `--moe-expert-storage`).
3. In `ggml_backend_sched_compute_splits` (the existing split-aware scheduler), call
   `plan_window()` after each split resolves, using the next D split's predicted experts.
4. Gate decisions become `cudaStream_t`-based async reads into pinned staging buffers.
5. `AdaptivePolicy.pin()` becomes a per-expert flag in the `ggml_tensor` metadata.
6. `cost_model.py` logic is ported to a C++ `CostModel` class for real-time telemetry (optional).

**Dependencies:**
- Requires `llama_cpp_modified/` tree with patches 0001–0009 applied.
- Requires access to the real `ggml_backend_sched` code (already present in the repo).
- Must not break existing `test-backend-ops`, `test-arg-parser`, `test-llama-archs`.

---

### Summary of remaining work (what to build first)

| Priority | TODO | Effort | Depends on |
|---|---|---|---|
| 1 | TODO-4: `tests/test_scheduler.py` | Medium | None (can write in parallel with TODO-2/3) |
| 2 | TODO-1: `scripts/cost_model.py` | Small | None |
| 3 | TODO-2: `scripts/simulate_trace.py` harness wiring | Medium | TODO-1 |
| 4 | TODO-3: `scripts/run_ablations.py` ASS axes | Medium | TODO-1, TODO-2 |
| 5 | TODO-5: Full harness integration (function signatures) | Small | TODO-2 (same work) |
| 6 | TODO-6: Full ablation integration | Small | TODO-3 (same work) |
| 7 | TODO-7: End-to-end integration test | Small | TODO-1–6 |
| 8 | TODO-8: Documentation updates | Small | TODO-7 (needs real numbers) |
| 9 | TODO-9: C++/llama.cpp port | Large | All above + hardware testing |

**Recommended order:** TODO-4 → TODO-1 → TODO-2 → TODO-3 → TODO-5/6 (same PR) → TODO-7 → TODO-8.
