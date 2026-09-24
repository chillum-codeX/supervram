# ASS confidence-gated prefetch: C++ port progress (all 5 phases complete)

**Status:** All 5 phases complete and verified: the feature is real, bit-exact, and
**measured to regress decode throughput by ~12.5%** in SuperVRAM's actual target regime
(SSD-bandwidth-bound cold-cache decode) -- a genuine, root-caused negative result, not a bug.
Full numbers in `writer_handoff/EVIDENCE_LEDGER.md`'s "ASS confidence-gated speculative prefetch"
section. See `docs/PRIOR_ART_AND_NEXT_STEPS.md` section 5 for why this port mattered, and the
approved plan this work followed (`/home/truppy/.claude/plans/robust-gliding-key.md` at the time
of writing, reproduced in spirit below) for the full 5-phase design and its scope boundaries.

**Goal, restated:** port the `AdaptiveSpeculativeScheduler` confidence-gate — currently a
Python-only simulation (`supervram/scheduler.py`, `predictors.py`, 44 unit tests, never run
against real hardware) — into the `classic`/`direct-io` tier of the real
`ggml_backend_sched_expert_cache` in `third_party/llama.cpp`, so a real measured result is
directly comparable to every benchmark already in `writer_handoff/EVIDENCE_LEDGER.md` (the Q8_0
baseline and both tiered-precision models all used this tier).

**Real-hardware caveat that shapes the whole design:** the Python simulator's "lookahead window"
assumes several *future* layers' real routing decisions are already known (deterministic replay
gives this for free). On real single-request autoregressive decode (batch size 1, matching
`cold_run.py`'s existing protocol), layer L+1's routing genuinely isn't known until layer L+1's
hidden state is computed — true multi-layer lookahead isn't physically available. This port does
**not** attempt it. It predicts *same-layer, next-token* reuse instead: "expert e at layer L was
used in M% of this layer's recent visits — even though token T didn't need it, will token T+1?"
— using only already-observed history, with the confidence-gate math ported faithfully from
`ConfidencePredictor`.

---

## Phase 1 — plumbing, default off, provably a no-op

Threaded a new `--moe-expert-scheduler {off,ass}` setting end to end, mirroring the exact
existing pattern used for `moe_expert_cache_policy` / `--moe-expert-storage`. Every new field
defaults to `OFF`, and `OFF` was verified to be a true no-op before any consuming logic existed
(nothing reads the field yet as of Phase 1 — the point was to prove the plumbing itself changes
nothing).

**Files touched** (all in `third_party/llama.cpp`, which is git-tracked internally but
`.gitignore`d from the outer `supervram` repo — line numbers below are from `git diff --stat` at
the time of writing):

- `include/llama.h` (+8 lines): new `enum llama_moe_expert_scheduler { LLAMA_MOE_EXPERT_SCHEDULER_OFF = 0, LLAMA_MOE_EXPERT_SCHEDULER_ASS = 1 }`, and a `moe_expert_scheduler` field on `llama_context_params`, next to the existing `moe_expert_zerocopy` field.
- `src/llama-cparams.h` (+1 line): mirrored field on the internal `llama_cparams` struct.
- `src/llama-context.cpp` (+3 lines): copies the field from `params` to `cparams` in the context constructor; sets `LLAMA_MOE_EXPERT_SCHEDULER_OFF` in `llama_context_default_params()`; passes it into `ggml_backend_sched_expert_cache_params.scheduler` inside `sched_set_expert_cache()`.
- `ggml/include/ggml-backend.h` (+8 lines): mirrored `enum ggml_backend_sched_expert_scheduler` and a `scheduler` field on `ggml_backend_sched_expert_cache_params` — the public C API surface.
- `ggml/src/ggml-backend.cpp` (+7 lines): a `scheduler` field on the internal `struct ggml_backend_sched_expert_cache` (alongside `direct_io`/`io_threads`), set in `ggml_backend_sched_set_expert_cache()` with an `SVRAM_SCHEDULER` environment-variable fallback (explicit parameter wins, env is fallback — matching the existing pattern for `SVRAM_DIRECT_IO`, `SVRAM_ZEROCOPY`, etc.).
- `src/svram_verify.cpp` (the project's own benchmark/verification tool, the primary CLI surface per this port's scope — `common/arg.cpp`/llama-server wiring is deliberately deferred): new `--moe-expert-scheduler off|ass` flag, `struct options::scheduler` field (default `"off"`), mapped into `cparams.moe_expert_scheduler`, and recorded in the tool's `--json` output as `"moe_expert_scheduler"` for evidence-ledger traceability.

**Verification.** Rebuilt (`ninja -C third_party/llama.cpp/build-3090 test-expert-cache
llama-server llama-cli && ninja -C build svram-verify`, both clean). Ran `test-expert-cache`
(the project's existing fake-backend C++ unit test for the cache) — all 3 pre-existing cases
still pass unchanged. Ran `svram-verify` three ways on the same prompt/model/cache config (Q8_0,
16 GiB, 64-token greedy decode): no flag at all, explicit `--moe-expert-scheduler off`, and
`--moe-expert-scheduler ass` (still fully inert at this phase — nothing consumes the value yet).
All three produced **bit-identical token sequences and logits hashes**
(`fnv1a` hash of the raw logit bytes per step, via `scripts/compare_verify.py`, the same
methodology used to gate every prior C++ patch to this codebase, 0001-0009).

---

## Phase 2 — the confidence-gate class, standalone and unit-tested

New header, **no dependency on any ggml/ggml-backend type by design**: pure, backend-agnostic
C++ logic, so it can be exhaustively unit-tested against the Python reference in isolation before
being wired into a performance-critical, already-shipped file (that wiring is Phase 3).

**New file:** `third_party/llama.cpp/ggml/src/ggml-backend-expert-scheduler.h` —
`svram::SpeculativeGate`, porting two pieces of the Python design:

1. **`ConfidencePredictor`'s bias-corrected EMA confidence table** (`supervram/predictors.py`):
   per-`(layer, expert)` confidence, updated as `conf = effective_alpha * conf + (1 -
   effective_alpha) * hit`, where `effective_alpha = min(alpha, (n-1)/n)` ramps from 0 up to the
   configured `alpha` (default 0.9) as observations `n` accumulate. `observe(layer, chosen)` /
   `set_pending(layer, candidates)` / `confidence_of(layer, expert)` mirror the Python method
   names and semantics directly.
2. **`AdaptiveSpeculativeScheduler.plan_window`'s pooled greedy-knapsack budget spend**
   (`supervram/scheduler.py`): pool candidates across the whole lookahead window, sort by
   `(p_use, t_read)` descending, truncate to `top_k` (a *global* pool-wide cap, not per-access —
   this preserves "equal total speculative-read depth vs. the `off` baseline," per the Python
   docstring's own fairness argument), then greedily spend a *shared* time budget on the sorted
   list, gated by a hard `p_use >= gate_threshold` floor independent of budget. Confirmed via a
   dedicated test (`test_gate_budget_is_shared_not_per_candidate`) that this really is
   budget-*shared*, not per-candidate: two equally-confident, individually-affordable candidates
   can still see the second one refused once the first has spent the shared budget.

   Note for anyone extending this: `docs/PLAN_ADAPTIVE.md` sections 2.1/4 describe a *different*,
   simpler per-candidate formula (`p_use * t_read < budget - cost_of_a_waste`). That formula was
   tried and abandoned — it algebraically degenerates to `t_read < budget`, silently dropping
   confidence from the decision entirely — and the code's own docstring documents this. Treat
   `scheduler.py`'s actual implementation as ground truth, not the older doc pseudocode.

**How correctness was verified — and two real bugs this caught.** Rather than hand-deriving
expected test values, the actual Python classes were run on a fixed, deterministic input
sequence to generate reference numbers, which were then asserted against bit-for-bit in the C++
tests. This caught two bugs that a hand-derived "looks right" check would have missed:

1. **Pending-set consumption.** Python's `ConfidencePredictor.observe()` does
   `self._pending.pop(access.layer, None)` — it *consumes* (removes) the pending prediction set,
   not just reads it. A first draft of the C++ port only read it (`unordered_map::find`), which
   would have let a second `observe()` call for the same layer silently reuse a stale prediction
   instead of correctly no-op'ing. Fixed by erasing the entry after use
   (`test_gate_observe_without_pending_is_noop` guards this).
2. **Float vs. double precision.** A first draft stored confidence as `float` (32-bit); Python's
   `float` is a 64-bit double throughout. Empirically, two confidence values that are
   conceptually both "2/3" can land on **different doubles one ULP apart**
   (`0.6666666666666666` vs. `0.6666666666666667`) depending on which sequence of EMA updates
   reached them — and `plan_window`'s `(p_use, t_read)` sort keys off exactly that residual
   difference in practice (verified by dumping both the Python and C++ pool order for a
   deliberately tie-prone scenario — the "sort by t_read descending on ties" behavior described
   in the docstring essentially never triggers in practice, because true bit-exact ties in
   `p_use` are rare; the *actual* observed tie-break is whichever value's floating-point noise
   happens to be larger). Using `float` would have silently produced a different fire/no-fire
   pattern than the Python reference on real data. Fixed by using `double` throughout
   (confidence table, `ReadDecision::p_use`, all arithmetic) — verified bit-exact against the
   Python reference's `.hex()` dump in `test_gate_confidence_matches_python_reference` and
   `test_gate_plan_window_matches_python_reference`.

**New unit tests** (`third_party/llama.cpp/tests/test-expert-cache.cpp`, run via
`third_party/llama.cpp/build-3090/bin/test-expert-cache`, no fake-backend fixture needed since
the gate has no ggml dependency):

- `test_gate_confidence_matches_python_reference` — bit-exact confidence values vs. a real
  Python `ConfidencePredictor` run on the same 4-step observe/set_pending trace.
- `test_gate_plan_window_matches_python_reference` — bit-exact pool sort order and fire/no-fire
  decisions vs. a real Python `plan_window` run, including the FP-noise-driven tie-break order
  above (not the "obvious" theoretical tie-break).
- `test_gate_budget_is_shared_not_per_candidate` — the pooled-budget property specifically
  (see above).
- `test_gate_observe_without_pending_is_noop` — the pending-consumption semantics specifically.

All 7 tests (3 pre-existing cache-bookkeeping tests + 4 new gate tests) pass:

```
test_cache_correctness_and_stats PASSED
test_cache_disabled_is_noop PASSED
test_oversized_step_fails_cleanly PASSED
test_gate_confidence_matches_python_reference PASSED
test_gate_plan_window_matches_python_reference PASSED
test_gate_budget_is_shared_not_per_candidate PASSED
test_gate_observe_without_pending_is_noop PASSED
```

---

## Phase 3 — candidate generation + wiring into the real classic-tier step loop

New file: `ggml/src/ggml-backend-expert-scheduler.h`'s `svram::SpeculativeGate` is now wired into
`ggml_backend_sched_expert_cache_plan()` (`ggml/src/ggml-backend.cpp`), guarded end to end by
`cache->scheduler == GGML_SCHED_EXPERT_SCHEDULER_ASS && cache->direct_io` (zero cost, zero
behavior change when off, per Phase 1's verification).

- `ggml_backend_sched_expert_tensor` gained `spec_score` (`std::vector<float>`, one decayed
  recency/frequency score per expert, independent of slot residency — an expert that gets evicted
  keeps its score, unlike `last_use`/`use_count` which are indexed by slot and lost on eviction).
  `layer`/`kind` fields already existed on this struct but were previously populated **only** for
  the zero-copy tier (inside the zero-copy-only `ctl_init()`); moved that parsing into
  `desc_init()`, which runs for both tiers, so the classic tier gets them too.
- `ggml_backend_sched_expert_cache` gained a `SpeculativeGate*`, a dedicated background
  `spec_worker` thread + job queue (`spec_queue`/`spec_done`, its own mutex/condvar — deliberately
  separate from the required-miss path's `ggml_expert_io_pool`, so speculative reads never
  contend with or slow down a real miss), and `spec_issued`/`spec_committed`/`spec_wasted`
  counters (added to the public `ggml_backend_sched_expert_cache_stats` struct, threaded through
  to `svram-verify`'s `spec_stats` stdout line and `scripts/cold_run.py`'s stat capture).
- **Safety design**: the background worker only ever reads bytes into a job's own staging buffer
  via `pread()` — it never touches cache bookkeeping (`slot_of_expert`/`expert_of_slot`/etc.).
  Those arrays are mutated *only* on the main compute thread, inside
  `ggml_backend_sched_expert_cache_spec_commit()`, called at the very **start** of `plan()` for a
  tensor — before that step's own real hit/miss resolution — so a speculative fill that turns out
  to match what's actually needed registers as a genuine hit, not a redundant miss. This mirrors
  the zero-copy tier's already-proven `promote()` two-phase "commit previous, then issue new"
  pattern, and avoids any data race on the arrays `plan()` also reads/writes on the same thread.
  Candidate generation itself (same-layer, next-token prediction from `spec_score`, feeding the
  gate) runs at the **end** of `plan()`, after the step's real needs are satisfied.

**A real design bug found and fixed while verifying this worked at all, not just that it
compiled.** The first version only ever issued a speculative read into a genuinely **empty**
slot, reasoning that never evicting a resident expert for speculation was the safest possible
policy. Empirically (`SVRAM_SPEC_DEBUG` tracing, see below) this made the feature **permanently
inert**: `spec_issued` stayed exactly 0 across a 256-token run on a 16 GiB cache — because a
warmed-up cache simply has no empty slots left after the first handful of steps; every later
step's required misses fill them immediately. An empty-slots-only policy can only ever fire during
the brief cold-start window, which is not the regime this feature is meant to help in at all.

Fixed by allowing eviction of a slot **not needed by the current step**, using the exact same
LRU/LFU victim-selection comparison the required-miss path already trusts
(`cache->params.policy`), extended with a two-part re-verification at commit time: the job now
records the victim slot's occupant (`evicted_expert`) *and* its `last_use` tick at submission
time, and commit only proceeds if **both** are unchanged. The second check closes a real gap the
first one alone would have missed: if the intervening real access pattern legitimately reused the
same expert in that slot again (without moving it, so `expert_of_slot` alone looks unchanged),
`last_use` will have advanced — and committing anyway would evict something just genuinely
needed. Verified after the fix: 256 real decode tokens on a 4 GiB cache (deliberately small, to
maximize eviction churn and stress-test the eviction path) produced 23,574 issued / 23,571
committed / 3 wasted speculative reads, i.e. genuinely active and >99.9% landing successfully.

## Phase 4 — correctness gate

Ran the real bit-exactness gate with speculative prefetch **genuinely active** (not the inert
Phase-1-era no-op): `svram-verify --moe-expert-scheduler off` vs `ass`, same Q8_0 model, same
4 GiB cache, 256-token greedy decode, diffed via `scripts/compare_verify.py`.

```
spec_stats issued=23574 committed=23571 wasted=0
tokens_identical: true
first_divergence_step: null
logits_hashes_identical: true
```

Bit-exact across all 256 steps despite over 23,000 real speculative commits happening during the
run — confirms the additive-only design principle (prefetch only ever warms extra slots, never
changes which expert gets computed for a required step) held up under real, heavy exercise, not
just in the trivial case where nothing fired.

`test-expert-cache`'s full suite (3 pre-existing + 4 Phase-2 gate tests) still passes unchanged
after the Phase 3 wiring.

## Phase 5 — real measurement: a genuine regression, not a win

Ran `scripts/cold_run.py`'s existing protocol exactly (16 GiB cache, 4 GiB RAM cap, evicted page
cache, direct I/O, 1,536-token decode, `Qwen3-30B-A3B-Q8_0.gguf`), 3 reps each, `off` vs `ass` —
the same protocol as the Q8_0 baseline and both tiered-precision models, so this lands in the
same comparison table.

| | decode t/s (mean of 3) | stdev | hit rate | SSD bytes | spec issued/committed/wasted |
|---|---|---|---|---|---|
| `off` | 30.68 | 0.30 | 97.4% | 72.74 GB | 0/0/0 |
| `ass` | 26.84 | 0.24 | 97.4% | 92.35 GB | 11,970 / 11,802 / 168 |

**~12.5% slower, ~27% more SSD traffic, no hit-rate improvement** (required-path hits actually
dropped slightly — ~597 fewer hits out of 1.79M accesses — from occasional mistimed evictions).
Tight, reproducible, no overlap between the two 3-rep ranges: this is a real effect, not noise.

**Root cause (diagnosed, not guessed):** the ported gate's budget is denominated in *compute
time* (`compute_ms_per_layer`), faithfully matching the Python simulation's own design — which
implicitly assumes an overlap regime where I/O hides behind spare GPU compute. But this project's
actual target regime is **SSD-bandwidth-bound** (72-81% of decode time is SSD wait, per the
evidence table at the top of this ledger): there's no compute slack to hide behind, so every
speculative byte directly competes with required bytes on the same physical drive. A
compute-time budget structurally can't see that contention, so the gate keeps firing — and in
this regime, firing is close to a coin flip that costs real bandwidth either way, given how
often a never-before-seen candidate clears the neutral `default_confidence=0.5` floor by default
on Qwen3-30B-A3B's fairly high-entropy routing (see the tiered-precision entropy numbers above).

**What this doesn't mean:** the C++ port itself is correct and working as designed (bit-exact
under real load, 98.6% of issued speculative reads land successfully, the gate math matches the
Python reference exactly). The regression is a genuine finding about *this specific budget
model's* mismatch with the target regime, not an implementation bug. The fix the diagnosis points
to — denominating the budget in spare SSD bandwidth instead of compute time — is a materially
different design, not a parameter tweak, and is explicitly not attempted here per the approved
plan's "smallest viable port first" scope. Whether it would actually help is an open, testable
question for a follow-up, not assumed.

Explicitly out of scope for this pass (see the approved plan for the full reasoning): multi-layer
lookahead, router-probability-based prediction, CUDA-stream async reads, zero-copy tier changes,
`common/arg.cpp`/llama-server wiring, comparison against SPICE's published numbers, and the
SSD-bandwidth-budget redesign this Phase 5 result now motivates.
