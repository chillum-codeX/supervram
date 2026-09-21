from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time

import pytest

from supervram import Access, ExpertCache, ExpertKey, SuperVRAM, TensorStore, TensorStoreWriter, make_policy, make_predictor
from supervram.policies import WeightedPolicy
from supervram.types import CacheEntry


def build_store(path: Path, layers: int = 2, experts: int = 4, size: int = 4096) -> None:
    with TensorStoreWriter(path, alignment=4096) as writer:
        for layer in range(layers):
            for expert in range(experts):
                byte = (layer * experts + expert) % 256
                writer.add(ExpertKey(layer, expert), [bytes([byte]) * size])


def test_store_round_trip_and_alignment(tmp_path: Path) -> None:
    path = tmp_path / "experts.svram"
    build_store(path, layers=1, experts=3, size=123)
    manifest = json.loads(path.with_suffix(".svram.json").read_text())
    assert all(item["offset"] % 4096 == 0 for item in manifest["extents"])
    with TensorStore(path, use_mmap=True) as store:
        assert store.read(ExpertKey(0, 2), verify=True) == bytes([2]) * 123


def test_lru_cache_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "experts.svram"
    build_store(path, layers=1, experts=4)
    with TensorStore(path) as store, ExpertCache(store, 2 * 4096, make_policy("lru"), queue_depth=2) as cache:
        cache.get(ExpertKey(0, 0))
        cache.get(ExpertKey(0, 1))
        cache.get(ExpertKey(0, 0))
        cache.get(ExpertKey(0, 2))
        assert cache.used_bytes <= cache.capacity_bytes
        assert ExpertKey(0, 0) in cache.entries
        assert ExpertKey(0, 1) not in cache.entries
        assert cache.stats.evictions == 1


def test_prefetch_becomes_hit(tmp_path: Path) -> None:
    path = tmp_path / "experts.svram"
    build_store(path, layers=1, experts=2)
    with TensorStore(path) as store, ExpertCache(store, 2 * 4096, make_policy("lru")) as cache:
        cache.prefetch(ExpertKey(0, 1))
        cache.get(ExpertKey(0, 1))
        assert cache.stats.prefetches == 1
        assert cache.stats.useful_prefetches == 1


def test_markov_predictor_and_engine(tmp_path: Path) -> None:
    path = tmp_path / "experts.svram"
    build_store(path, layers=1, experts=4)
    predictor = make_predictor("markov")
    with TensorStore(path) as store, ExpertCache(store, 3 * 4096, make_policy("router-aware")) as cache:
        engine = SuperVRAM(cache, predictor, prefetch_depth=2)
        sequence = [(0, 1), (2, 3), (0, 1), (2, 3), (0, 1)]
        for token, experts in enumerate(sequence):
            engine.process(Access(token, 0, experts, {expert: 0.5 for expert in experts}))
        metrics = engine.metrics()
        assert metrics["cache"]["accesses"] == 10
        assert metrics["prediction"]["correct"] > 0


def test_weighted_policy_values_probability_and_reload() -> None:
    now = time.monotonic_ns()
    policy = WeightedPolicy()
    cold = CacheEntry(ExpertKey(0, 0), 1024, now, now, accesses=1, predicted_probability=0.0, transfer_cost_ns=1_000)
    valuable = CacheEntry(ExpertKey(0, 1), 1024, now, now, accesses=1, predicted_probability=0.9, transfer_cost_ns=10_000_000)
    victims = policy.victims({cold.key: cold, valuable.key: valuable}, 1024, now)
    assert victims == [cold.key]


def test_router_probability_survives_other_layer_update(tmp_path: Path) -> None:
    path = tmp_path / "experts.svram"
    build_store(path, layers=2, experts=2)
    with TensorStore(path) as store, ExpertCache(store, 3 * 4096, make_policy("router-aware"), deterministic=True) as cache:
        cache.get(ExpertKey(0, 0))
        cache.update_probabilities({ExpertKey(0, 0): 0.9})
        cache.update_probabilities({ExpertKey(1, 0): 0.8})
        assert cache.entries[ExpertKey(0, 0)].predicted_probability == 0.9


def test_failed_load_can_retry(tmp_path: Path) -> None:
    path = tmp_path / "experts.svram"
    build_store(path, layers=1, experts=1)
    with TensorStore(path) as store:
        original = store.read
        attempts = 0
        def flaky(key, verify=False):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("transient")
            return original(key, verify)
        store.read = flaky
        with ExpertCache(store, 4096, make_policy("lru"), deterministic=True) as cache:
            with pytest.raises(OSError):
                cache.get(ExpertKey(0, 0))
            cache.get(ExpertKey(0, 0))
            assert attempts == 2


def test_oversized_expert_fails(tmp_path: Path) -> None:
    path = tmp_path / "experts.svram"
    build_store(path, layers=1, experts=1, size=8192)
    with TensorStore(path) as store, ExpertCache(store, 4096, make_policy("lru")) as cache:
        with pytest.raises(MemoryError):
            cache.get(ExpertKey(0, 0))
