from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .types import CacheEntry, ExpertKey


class CachePolicy(ABC):
    @abstractmethod
    def victims(self, entries: Mapping[ExpertKey, CacheEntry], bytes_needed: int, now_ns: int) -> list[ExpertKey]:
        raise NotImplementedError


class LRU(CachePolicy):
    def victims(self, entries: Mapping[ExpertKey, CacheEntry], bytes_needed: int, now_ns: int) -> list[ExpertKey]:
        return _take_until(entries, sorted(entries, key=lambda k: entries[k].last_access_ns), bytes_needed)


class LFU(CachePolicy):
    def victims(self, entries: Mapping[ExpertKey, CacheEntry], bytes_needed: int, now_ns: int) -> list[ExpertKey]:
        order = sorted(entries, key=lambda k: (entries[k].accesses, entries[k].last_access_ns))
        return _take_until(entries, order, bytes_needed)


@dataclass
class WeightedPolicy(CachePolicy):
    probability_weight: float = 4.0
    frequency_weight: float = 1.0
    reload_weight: float = 1.0
    recency_weight: float = 1.0
    size_weight: float = 1.0
    half_life_ns: int = 1_000_000_000

    def value(self, entry: CacheEntry, now_ns: int) -> float:
        age = max(0, now_ns - entry.last_access_ns)
        recency = math.exp(-math.log(2.0) * age / max(1, self.half_life_ns))
        frequency = math.log1p(entry.accesses)
        reload_ms = entry.transfer_cost_ns / 1e6
        numerator = (
            self.probability_weight * entry.predicted_probability
            + self.frequency_weight * frequency
            + self.reload_weight * reload_ms
            + self.recency_weight * recency
        )
        return numerator / max(1.0, (entry.size / (1024 * 1024)) ** self.size_weight)

    def victims(self, entries: Mapping[ExpertKey, CacheEntry], bytes_needed: int, now_ns: int) -> list[ExpertKey]:
        order = sorted(entries, key=lambda key: self.value(entries[key], now_ns))
        return _take_until(entries, order, bytes_needed)


class RouterAwarePolicy(WeightedPolicy):
    def update_probabilities(self, entries: Mapping[ExpertKey, CacheEntry], probabilities: Mapping[ExpertKey, float]) -> None:
        for key, probability in probabilities.items():
            if key in entries:
                entries[key].predicted_probability = float(probability)


@dataclass
class AdaptivePolicy(WeightedPolicy):
    """Cost-aware eviction (inherited from WeightedPolicy's reload_weight/transfer_cost_ns term)
    plus scheduler-directed pin protection (PLAN_ADAPTIVE.md section 2.4): experts the scheduler
    judges expensive-to-fetch and near-certain to be needed soon are never chosen as eviction
    victims while pinned, even if their weighted score would otherwise rank lowest. The scheduler
    (built in a later phase) is expected to keep the pinned set small -- if everything is pinned,
    there is nothing left to evict, which is the caller's responsibility to avoid.
    """

    pinned: frozenset[ExpertKey] = field(default_factory=frozenset)

    def pin(self, keys: Iterable[ExpertKey]) -> None:
        self.pinned = frozenset(self.pinned) | frozenset(keys)

    def unpin(self, keys: Iterable[ExpertKey]) -> None:
        self.pinned = frozenset(self.pinned) - frozenset(keys)

    def victims(self, entries: Mapping[ExpertKey, CacheEntry], bytes_needed: int, now_ns: int) -> list[ExpertKey]:
        evictable = {key: entry for key, entry in entries.items() if key not in self.pinned}
        order = sorted(evictable, key=lambda key: self.value(evictable[key], now_ns))
        return _take_until(evictable, order, bytes_needed)


def make_policy(name: str) -> CachePolicy:
    normalized = name.lower().replace("_", "-")
    if normalized == "lru":
        return LRU()
    if normalized == "lfu":
        return LFU()
    if normalized in {"weighted", "reload-aware"}:
        return WeightedPolicy()
    if normalized in {"router", "router-aware", "probability-aware"}:
        return RouterAwarePolicy()
    if normalized == "adaptive":
        return AdaptivePolicy()
    raise ValueError(f"unknown cache policy: {name}")


def _take_until(entries: Mapping[ExpertKey, CacheEntry], order: Iterable[ExpertKey], bytes_needed: int) -> list[ExpertKey]:
    result: list[ExpertKey] = []
    freed = 0
    for key in order:
        result.append(key)
        freed += entries[key].size
        if freed >= bytes_needed:
            break
    return result
