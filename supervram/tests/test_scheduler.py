from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cost_model import project_throughput  # noqa: E402

from supervram import (
    AdaptivePolicy,
    AdaptiveSpeculativeScheduler,
    Confidence,
    ConfidencePredictor,
    CostModelParams,
    ExpertCache,
    ExpertKey,
    ReadDecision,
    SuperVRAM,
    TensorStore,
    TensorStoreWriter,
    make_policy,
    make_predictor,
)
from supervram.predictors import HistoryPredictor, Predictor
from supervram.types import Access, CacheEntry

# Phase A (PLAN_ADAPTIVE.md): types + predictor + policy. Phase B: scheduler + cache window path.
# Phase C: overlap cost model + ablation-matrix coverage.


class _StubPredictor(Predictor):
    """Always predicts the same fixed candidates, ignoring history/probabilities -- isolates
    ConfidencePredictor's own calibration logic from a real predictor's adaptive behavior."""

    def __init__(self, always_predict: list[int]):
        self._candidates = always_predict

    def predict(self, layer: int, top_k: int, probabilities=None) -> list[int]:
        return list(self._candidates)[:top_k]


def test_confidence_predictor_calibrates_toward_one_for_a_consistent_hit() -> None:
    predictor = ConfidencePredictor(_StubPredictor([5]), alpha=0.5)
    for _ in range(20):
        predictor.predict_confident(layer=0, top_k=1)
        predictor.observe(Access(token=0, layer=0, experts=(5,)))
    assert predictor.confidence_of(0, 5) > 0.99


def test_confidence_predictor_converges_fast_for_an_always_right_predictor() -> None:
    """PLAN_ADAPTIVE.md Phase E: a fixed alpha=0.9 EMA needs ~44 observations to pull a perfect
    predictor's confidence near 1.0, which is exactly what caused ASS's throughput regressions
    against predictor=oracle in the full ablation matrix (63/65 cases, see
    docs/IMPLEMENTATION_STATUS.md). The bias-corrected ramp (effective alpha starts at 0, a plain
    running mean, and only reaches the configured alpha once enough observations accumulate) must
    reach a high-confidence verdict in a handful of observations, not dozens."""
    predictor = ConfidencePredictor(_StubPredictor([5]), alpha=0.9)
    for _ in range(3):
        predictor.predict_confident(layer=0, top_k=1)
        predictor.observe(Access(token=0, layer=0, experts=(5,)))
    assert predictor.confidence_of(0, 5) > 0.9


def test_confidence_predictor_first_observation_is_a_plain_running_mean() -> None:
    """n=1: effective alpha is 0, so confidence is exactly the single observed outcome, not a
    blend with the neutral default -- the mechanism this test locks in."""
    predictor = ConfidencePredictor(_StubPredictor([5]), alpha=0.9, default_confidence=0.5)
    predictor.predict_confident(layer=0, top_k=1)
    predictor.observe(Access(token=0, layer=0, experts=(5,)))
    assert predictor.confidence_of(0, 5) == 1.0


def test_confidence_predictor_still_settles_low_for_a_consistently_wrong_predictor_fast() -> None:
    predictor = ConfidencePredictor(_StubPredictor([5]), alpha=0.9, default_confidence=0.5)
    for _ in range(3):
        predictor.predict_confident(layer=0, top_k=1)
        predictor.observe(Access(token=0, layer=0, experts=(9,)))
    assert predictor.confidence_of(0, 5) < 0.1


def test_confidence_predictor_calibrates_toward_zero_for_a_consistent_miss() -> None:
    predictor = ConfidencePredictor(_StubPredictor([5]), alpha=0.5, default_confidence=0.5)
    for _ in range(20):
        predictor.predict_confident(layer=0, top_k=1)
        predictor.observe(Access(token=0, layer=0, experts=(9,)))
    assert predictor.confidence_of(0, 5) < 0.01


def test_confidence_predictor_default_for_unobserved_expert() -> None:
    predictor = ConfidencePredictor(HistoryPredictor(), default_confidence=0.42)
    assert predictor.confidence_of(layer=3, expert=7) == 0.42


def test_confidence_predictor_passthrough_predict_matches_base() -> None:
    base = HistoryPredictor()
    base.observe(Access(token=0, layer=1, experts=(2, 3)))
    predictor = ConfidencePredictor(base)
    assert predictor.predict(1, 2) == base.predict(1, 2)


def test_confidence_predictor_rejects_invalid_alpha() -> None:
    try:
        ConfidencePredictor(HistoryPredictor(), alpha=1.5)
    except ValueError:
        return
    raise AssertionError("expected ValueError for alpha outside [0, 1]")


def _entry(layer: int, expert: int, *, probability: float, now_ns: int) -> CacheEntry:
    return CacheEntry(
        key=ExpertKey(layer, expert),
        size=1024,
        loaded_ns=now_ns,
        last_access_ns=now_ns,
        accesses=1,
        predicted_probability=probability,
    )


def test_adaptive_policy_never_evicts_a_pinned_key_even_if_lowest_value() -> None:
    now = time.time_ns()
    cheap_but_pinned = ExpertKey(0, 0)
    expensive_unpinned = ExpertKey(0, 1)
    entries = {
        cheap_but_pinned: _entry(0, 0, probability=0.0, now_ns=now),  # would rank first for eviction
        expensive_unpinned: _entry(0, 1, probability=1.0, now_ns=now),
    }
    policy = AdaptivePolicy()
    policy.pin([cheap_but_pinned])
    victims = policy.victims(entries, bytes_needed=1024, now_ns=now)
    assert cheap_but_pinned not in victims
    assert victims == [expensive_unpinned]


def test_adaptive_policy_unpin_restores_eviction_eligibility() -> None:
    now = time.time_ns()
    key = ExpertKey(0, 0)
    entries = {key: _entry(0, 0, probability=0.0, now_ns=now)}
    policy = AdaptivePolicy()
    policy.pin([key])
    assert policy.victims(entries, bytes_needed=1024, now_ns=now) == []
    policy.unpin([key])
    assert policy.victims(entries, bytes_needed=1024, now_ns=now) == [key]


def test_adaptive_policy_matches_weighted_policy_when_nothing_pinned() -> None:
    now = time.time_ns()
    entries = {
        ExpertKey(0, 0): _entry(0, 0, probability=0.1, now_ns=now),
        ExpertKey(0, 1): _entry(0, 1, probability=0.9, now_ns=now),
    }
    adaptive = AdaptivePolicy()
    from supervram.policies import WeightedPolicy

    weighted = WeightedPolicy()
    assert adaptive.victims(entries, 1024, now) == weighted.victims(entries, 1024, now)


def test_make_policy_adaptive_returns_adaptive_policy() -> None:
    policy = make_policy("adaptive")
    assert isinstance(policy, AdaptivePolicy)


def test_read_decision_holds_gate_fields() -> None:
    decision = ReadDecision(key=ExpertKey(2, 5), p_use=0.8, t_read=1.2, fire=True)
    assert decision.key == ExpertKey(2, 5)
    assert decision.fire is True


def test_confidence_dataclass_fields() -> None:
    confidence = Confidence(expert=5, probability=0.75)
    assert confidence.expert == 5
    assert confidence.probability == 0.75


def test_cost_model_params_defaults_match_measured_evidence() -> None:
    params = CostModelParams()
    assert params.drive_gbps == 2.4
    assert params.lookahead_d == 4


# --- Phase B: scheduler + cache window path -------------------------------------------------


def test_scheduler_t_read_seconds_uses_extent_length_and_drive_bandwidth() -> None:
    params = CostModelParams(drive_gbps=1.0)  # 1e9 bytes/sec, so 1e9 bytes -> 1.0 second
    scheduler = AdaptiveSpeculativeScheduler(_StubPredictor([]), lambda key: 1_000_000_000, params)
    assert scheduler.t_read_seconds(ExpertKey(0, 0)) == 1.0


def test_scheduler_t_read_seconds_missing_key_is_zero() -> None:
    def missing(key: ExpertKey) -> int:
        raise KeyError(key)

    scheduler = AdaptiveSpeculativeScheduler(_StubPredictor([]), missing)
    assert scheduler.t_read_seconds(ExpertKey(0, 0)) == 0.0


def test_plan_window_empty_window_returns_no_decisions() -> None:
    scheduler = AdaptiveSpeculativeScheduler(_StubPredictor([1]), lambda key: 4096)
    assert scheduler.plan_window([]) == []


def test_plan_window_respects_gate_threshold_floor() -> None:
    # A huge compute budget would let the read fire on cost grounds alone; the confidence floor
    # must still refuse a candidate that has never earned any calibrated trust.
    params = CostModelParams(gate_threshold=0.6, compute_ms_per_layer=1_000_000.0)
    scheduler = AdaptiveSpeculativeScheduler(_StubPredictor([5]), lambda key: 1, params, top_k=1)
    decisions = scheduler.plan_window([Access(token=0, layer=0, experts=(), probabilities={})])
    assert len(decisions) == 1
    assert decisions[0].fire is False
    assert decisions[0].p_use < params.gate_threshold


def test_plan_window_fires_once_confidence_is_calibrated_above_floor() -> None:
    params = CostModelParams(gate_threshold=0.3, compute_ms_per_layer=1_000_000.0)
    scheduler = AdaptiveSpeculativeScheduler(_StubPredictor([5]), lambda key: 1, params, top_k=1, alpha=0.5)
    for _ in range(10):
        scheduler.record_and_observe(Access(token=0, layer=0, experts=(5,), probabilities={}))
    decisions = scheduler.plan_window([Access(token=0, layer=0, experts=(), probabilities={})])
    assert decisions[0].fire is True
    assert decisions[0].p_use > 0.9


def test_plan_window_limits_total_candidates_to_top_k_across_whole_window() -> None:
    # top_k=2 must mean 2 candidates total across the window, not 2 per window entry -- matching
    # the baseline's prefetch_depth for a fair "equal depth" comparison (PLAN_ADAPTIVE.md 5.2).
    scheduler = AdaptiveSpeculativeScheduler(_StubPredictor([1, 2, 3]), lambda key: 4096, top_k=2)
    window = [Access(token=t, layer=t, experts=(), probabilities={}) for t in range(3)]
    decisions = scheduler.plan_window(window)
    assert len(decisions) == 2


def test_scheduler_stats_track_considered_fired_and_gated() -> None:
    params = CostModelParams(gate_threshold=0.6, compute_ms_per_layer=1_000_000.0)
    scheduler = AdaptiveSpeculativeScheduler(_StubPredictor([5]), lambda key: 1, params, top_k=1)
    scheduler.plan_window([Access(token=0, layer=0, experts=(), probabilities={})])
    assert scheduler.stats.considered == 1
    assert scheduler.stats.fired == 0
    assert scheduler.stats.gated == 1
    assert scheduler.stats.fire_rate == 0.0


def _build_store(path: Path, layers: int, experts: int, size: int) -> None:
    with TensorStoreWriter(path, alignment=4096) as writer:
        for layer in range(layers):
            for expert in range(experts):
                writer.add(ExpertKey(layer, expert), [bytes([(layer * experts + expert) % 251]) * size])


def test_cache_issue_is_noop_when_decision_does_not_fire(tmp_path: Path) -> None:
    path = tmp_path / "experts.svram"
    _build_store(path, layers=1, experts=2, size=4096)
    with TensorStore(path) as store, ExpertCache(store, 2 * 4096, make_policy("lru")) as cache:
        cache.issue(ReadDecision(key=ExpertKey(0, 0), p_use=0.9, t_read=0.0, fire=False))
        cache.drain()
        assert cache.stats.prefetches == 0
        assert ExpertKey(0, 0) not in cache.entries


def test_cache_issue_prefetches_when_decision_fires(tmp_path: Path) -> None:
    path = tmp_path / "experts.svram"
    _build_store(path, layers=1, experts=2, size=4096)
    with TensorStore(path) as store, ExpertCache(store, 2 * 4096, make_policy("lru")) as cache:
        cache.issue(ReadDecision(key=ExpertKey(0, 0), p_use=0.9, t_read=0.0, fire=True))
        cache.drain()
        assert cache.stats.prefetches == 1
        assert ExpertKey(0, 0) in cache.entries


def test_cache_stats_overlapped_bytes_tracks_a_useful_prefetch_hit(tmp_path: Path) -> None:
    path = tmp_path / "experts.svram"
    _build_store(path, layers=1, experts=1, size=4096)
    with TensorStore(path) as store, ExpertCache(store, 4096, make_policy("lru")) as cache:
        cache.prefetch(ExpertKey(0, 0))
        cache.drain()
        cache.get(ExpertKey(0, 0))
        assert cache.stats.overlapped_bytes == 4096
        assert cache.stats.waste_bytes == 0


def test_cache_stats_waste_bytes_tracks_an_evicted_unused_prefetch(tmp_path: Path) -> None:
    path = tmp_path / "experts.svram"
    _build_store(path, layers=1, experts=2, size=4096)
    with TensorStore(path) as store, ExpertCache(store, 4096, make_policy("lru")) as cache:  # room for exactly one expert
        cache.prefetch(ExpertKey(0, 0))
        cache.drain()
        cache.get(ExpertKey(0, 1))  # evicts the still-unused prefetch of expert 0
        assert cache.stats.waste_bytes == 4096
        assert cache.stats.overlapped_bytes == 0


def _weak_trace(tokens: int, layers: int, experts: int, top_k: int, seed: int) -> list[Access]:
    """A deterministic, near-unpredictable (locality ~0) access trace: each layer's experts are
    an independent random draw every time, so a history-based predictor gets essentially no
    signal -- this is the "existing weak predictors" regime PLAN_ADAPTIVE.md section 5.2's
    success criterion is stated for (the real measured routing-history accuracy is 1.4-2%,
    writer_handoff/EVIDENCE_LEDGER.md), unlike a high-locality trace where blind history
    prediction is accidentally strong and not representative of the real evidence."""
    rng = random.Random(seed)
    accesses = []
    for token in range(tokens):
        for layer in range(layers):
            selected = tuple(rng.sample(range(experts), top_k))
            accesses.append(Access(token, layer, selected, {e: 1.0 / top_k for e in selected}))
    return accesses


def _run_baseline(accesses: list[Access], store_path: Path, layers: int, experts: int, size: int, cache_experts: int, prefetch_depth: int):
    with TensorStore(store_path) as store, ExpertCache(store, cache_experts * size, make_policy("lru"), deterministic=True) as cache:
        engine = SuperVRAM(cache, make_predictor("history"), prefetch_depth)
        for access in accesses:
            engine.process(access)
            cache.drain()
        return cache.stats


def _run_ass(accesses: list[Access], store_path: Path, layers: int, experts: int, size: int, cache_experts: int, prefetch_depth: int, lookahead_d: int, params: "CostModelParams | None" = None):
    with TensorStore(store_path) as store, ExpertCache(store, cache_experts * size, make_policy("lru"), deterministic=True) as cache:
        engine = SuperVRAM(cache, make_predictor("history"), prefetch_depth)
        scheduler = AdaptiveSpeculativeScheduler(make_predictor("history"), lambda key: store.extents[key].length, params, top_k=prefetch_depth)
        for index, access in enumerate(accesses):
            window = accesses[index : index + lookahead_d]
            engine.process_window(window, scheduler)
            cache.drain()
        return cache.stats, scheduler.stats


def test_ass_scheduler_reduces_waste_bytes_vs_blind_prefetch_with_weak_predictor(tmp_path: Path) -> None:
    """PLAN_ADAPTIVE.md section 5, success criterion 2: "Gate reduces waste. In the harness, with
    the existing weak predictors, ASS's waste_bytes (wrong reads) is strictly lower than blind
    prefetch at equal --prefetch-depth." """
    layers, experts, size, cache_experts, depth = 6, 24, 65536, 12, 4
    path = tmp_path / "experts.svram"
    _build_store(path, layers=layers, experts=experts, size=size)
    accesses = _weak_trace(tokens=150, layers=layers, experts=experts, top_k=4, seed=11)

    baseline_stats = _run_baseline(accesses, path, layers, experts, size, cache_experts, depth)
    ass_stats, scheduler_stats = _run_ass(accesses, path, layers, experts, size, cache_experts, depth, lookahead_d=4)

    assert ass_stats.waste_bytes < baseline_stats.waste_bytes
    # the gate must actually be gating something, not firing on everything it considers
    assert scheduler_stats.gated > 0


def test_engine_process_window_reports_scheduler_stats_in_metrics(tmp_path: Path) -> None:
    path = tmp_path / "experts.svram"
    _build_store(path, layers=2, experts=4, size=4096)
    with TensorStore(path) as store, ExpertCache(store, 2 * 4096, make_policy("lru"), deterministic=True) as cache:
        engine = SuperVRAM(cache, make_predictor("history"), prefetch_depth=2)
        scheduler = AdaptiveSpeculativeScheduler(make_predictor("history"), lambda key: store.extents[key].length, top_k=2)
        accesses = [Access(0, 0, (0, 1), {0: 0.6, 1: 0.4}), Access(0, 1, (2, 3), {2: 0.6, 3: 0.4})]
        for index, access in enumerate(accesses):
            engine.process_window(accesses[index:], scheduler)
        metrics = engine.metrics()
        assert "scheduler" in metrics
        assert metrics["scheduler"]["considered"] == scheduler.stats.considered


def test_engine_metrics_omits_scheduler_block_when_only_process_used(tmp_path: Path) -> None:
    path = tmp_path / "experts.svram"
    _build_store(path, layers=1, experts=2, size=4096)
    with TensorStore(path) as store, ExpertCache(store, 4096, make_policy("lru"), deterministic=True) as cache:
        engine = SuperVRAM(cache, make_predictor("history"), prefetch_depth=1)
        engine.process(Access(0, 0, (0,), {0: 1.0}))
        assert "scheduler" not in engine.metrics()


# --- Phase C: overlap cost model + ablation-matrix coverage ----------------------------------


def test_cost_model_compute_bound_regime_uses_compute_time() -> None:
    # 100 blocking bytes on a very fast drive is negligible I/O -- compute should dominate.
    metrics = {"bytes_read": 100, "overlapped_bytes": 0, "waste_bytes": 0}
    params = CostModelParams(drive_gbps=1000.0, compute_ms_per_layer=1.0)
    result = project_throughput(metrics, params, tokens=10, steps=10)
    assert abs(result.wall_s - result.compute_s) < 1e-9
    assert result.compute_s > result.blocking_io_s


def test_cost_model_io_bound_regime_uses_blocking_io_time() -> None:
    metrics = {"bytes_read": 10_000_000_000, "overlapped_bytes": 0, "waste_bytes": 0}
    params = CostModelParams(drive_gbps=1.0, compute_ms_per_layer=0.001)
    result = project_throughput(metrics, params, tokens=10, steps=10)
    assert abs(result.wall_s - result.blocking_io_s) < 1e-9
    assert result.blocking_io_s > result.compute_s


def test_cost_model_overlapped_and_wasted_bytes_are_excluded_from_blocking() -> None:
    metrics = {"bytes_read": 1000, "overlapped_bytes": 300, "waste_bytes": 200}
    params = CostModelParams(drive_gbps=1.0)
    result = project_throughput(metrics, params, tokens=1, steps=1)
    assert result.blocking_bytes == 500


def test_cost_model_zero_steps_and_zero_bytes_yields_zero_wall_time_and_tps() -> None:
    metrics = {"bytes_read": 0, "overlapped_bytes": 0, "waste_bytes": 0}
    result = project_throughput(metrics, CostModelParams(), tokens=0, steps=0)
    assert result.wall_s == 0.0
    assert result.modeled_tokens_per_second == 0.0


def test_ass_modeled_throughput_never_worse_than_baseline_blocking_bytes(tmp_path: Path) -> None:
    """PLAN_ADAPTIVE.md section 5, success criterion 3 (modelled speedup), for a realistic weak
    predictor (this project's own measured routing-history accuracy is 1.4-2 %, not the ceiling).

    CORRECTION: an earlier version of this docstring claimed this holds "universally, not just at
    a hand-picked drive speed." The full 3,600-run scaled ablation matrix
    (results/ablation-simulate-ass-scaled/, see docs/IMPLEMENTATION_STATUS.md) disproved that:
    with `predictor=oracle`, ASS's modeled throughput is *worse* than blind prefetch in 63/384
    comparisons (up to -33.5 %), because the scheduler's own confidence calibration starts at a
    neutral default and needs a few observations to warm up to the oracle's actual (perfect)
    accuracy -- during that warm-up it under-trusts a predictor that was already reliable from
    turn one, refusing prefetches blind prefetch would have fired correctly. For every *non*-
    oracle predictor in the same matrix (the realistic case this plan targets), the story holds:
    2/1920 comparisons regressed throughput (0.10 %), essentially noise, against 384/1920 (20 %)
    that reduced waste and 535/1920 (27.9 %) unaffected either way. This test's own scenario below
    is one specific weak-predictor trace, not a claim that spans predictor quality -- see
    writer_handoff/KNOWN_LIMITATIONS.md for the oracle-warm-up caveat."""
    layers, experts, size, cache_experts, depth = 6, 24, 65536, 12, 4
    path = tmp_path / "experts.svram"
    _build_store(path, layers=layers, experts=experts, size=size)
    accesses = _weak_trace(tokens=150, layers=layers, experts=experts, top_k=4, seed=11)
    tokens = 150

    baseline_stats = _run_baseline(accesses, path, layers, experts, size, cache_experts, depth)
    baseline_metrics = {"bytes_read": baseline_stats.bytes_read, "overlapped_bytes": baseline_stats.overlapped_bytes, "waste_bytes": baseline_stats.waste_bytes}
    baseline_cost = project_throughput(baseline_metrics, CostModelParams(), tokens=tokens, steps=tokens * layers)

    ass_stats, _ = _run_ass(accesses, path, layers, experts, size, cache_experts, depth, lookahead_d=4)
    ass_metrics = {"bytes_read": ass_stats.bytes_read, "overlapped_bytes": ass_stats.overlapped_bytes, "waste_bytes": ass_stats.waste_bytes}
    ass_cost = project_throughput(ass_metrics, CostModelParams(), tokens=tokens, steps=tokens * layers)

    assert ass_cost.blocking_bytes <= baseline_cost.blocking_bytes
    assert ass_cost.modeled_tokens_per_second >= baseline_cost.modeled_tokens_per_second


def test_ass_modeled_gap_over_baseline_widens_on_a_slower_drive(tmp_path: Path) -> None:
    """Success criterion 3's other half: "the gap widens as --drive-gbps drops." Verified between
    a fast default drive and a slower-but-still-usable one (not the pathological near-zero-
    bandwidth regime, where the gate correctly stops firing altogether and both strategies
    converge back toward "no prefetch" -- a real, safe floor, not a broken result, but not the
    transitional regime this criterion is about either)."""
    layers, experts, size, cache_experts, depth = 6, 24, 65536, 12, 4
    path = tmp_path / "experts.svram"
    _build_store(path, layers=layers, experts=experts, size=size)
    accesses = _weak_trace(tokens=150, layers=layers, experts=experts, top_k=4, seed=11)
    tokens = 150

    def gap_at(drive_gbps: float) -> float:
        params = CostModelParams(drive_gbps=drive_gbps)
        baseline_stats = _run_baseline(accesses, path, layers, experts, size, cache_experts, depth)
        baseline_metrics = {"bytes_read": baseline_stats.bytes_read, "overlapped_bytes": baseline_stats.overlapped_bytes, "waste_bytes": baseline_stats.waste_bytes}
        baseline_cost = project_throughput(baseline_metrics, params, tokens=tokens, steps=tokens * layers)
        ass_stats, _ = _run_ass(accesses, path, layers, experts, size, cache_experts, depth, lookahead_d=4, params=params)
        ass_metrics = {"bytes_read": ass_stats.bytes_read, "overlapped_bytes": ass_stats.overlapped_bytes, "waste_bytes": ass_stats.waste_bytes}
        ass_cost = project_throughput(ass_metrics, params, tokens=tokens, steps=tokens * layers)
        return ass_cost.modeled_tokens_per_second / baseline_cost.modeled_tokens_per_second if baseline_cost.modeled_tokens_per_second else 1.0

    fast_gap = gap_at(2.4)
    slower_gap = gap_at(0.5)
    assert slower_gap >= fast_gap


def test_oracle_predictor_is_a_ceiling_over_ass_and_baseline(tmp_path: Path) -> None:
    """Success criterion 4: ASS should be judged by how close it gets to the oracle predictor's
    ceiling. This checks the ceiling property directly: an oracle-driven run (perfect foresight,
    no wrong reads at all) must model at least as fast as both ASS and the naive baseline on the
    same trace -- it cannot be beaten, only approached."""
    from supervram.predictors import OraclePredictor

    layers, experts, size, cache_experts, depth = 6, 24, 65536, 12, 4
    path = tmp_path / "experts.svram"
    _build_store(path, layers=layers, experts=experts, size=size)
    accesses = _weak_trace(tokens=150, layers=layers, experts=experts, top_k=4, seed=11)
    tokens = 150
    params = CostModelParams()

    baseline_stats = _run_baseline(accesses, path, layers, experts, size, cache_experts, depth)
    ass_stats, _ = _run_ass(accesses, path, layers, experts, size, cache_experts, depth, lookahead_d=4, params=params)

    with TensorStore(path) as store, ExpertCache(store, cache_experts * size, make_policy("lru"), deterministic=True) as cache:
        oracle_engine = SuperVRAM(cache, OraclePredictor(accesses[layers:]), depth)
        oracle_scheduler = AdaptiveSpeculativeScheduler(OraclePredictor(accesses[layers:]), lambda key: store.extents[key].length, params, top_k=depth)
        for index, access in enumerate(accesses):
            window = accesses[index : index + 4]
            oracle_engine.process_window(window, oracle_scheduler)
            cache.drain()
        oracle_stats = cache.stats

    def tps(stats) -> float:
        metrics = {"bytes_read": stats.bytes_read, "overlapped_bytes": stats.overlapped_bytes, "waste_bytes": stats.waste_bytes}
        return project_throughput(metrics, params, tokens=tokens, steps=tokens * layers).modeled_tokens_per_second

    oracle_tps, ass_tps, baseline_tps = tps(oracle_stats), tps(ass_stats), tps(baseline_stats)
    assert oracle_tps >= ass_tps
    assert oracle_tps >= baseline_tps
