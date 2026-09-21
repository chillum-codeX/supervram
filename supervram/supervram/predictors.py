from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from pathlib import Path
import json
from typing import Iterable, Mapping, Sequence

from .types import Access, ExpertKey


class Predictor(ABC):
    @abstractmethod
    def predict(self, layer: int, top_k: int, probabilities: Mapping[int, float] | None = None) -> list[int]:
        raise NotImplementedError

    def observe(self, access: Access) -> None:
        pass


class NoPredictor(Predictor):
    def predict(self, layer: int, top_k: int, probabilities: Mapping[int, float] | None = None) -> list[int]:
        return []


class HistoryPredictor(Predictor):
    def __init__(self, window: int = 128):
        self.history: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=window))

    def observe(self, access: Access) -> None:
        self.history[access.layer].extend(access.experts)

    def predict(self, layer: int, top_k: int, probabilities: Mapping[int, float] | None = None) -> list[int]:
        counts = Counter(self.history[layer])
        return [expert for expert, _ in counts.most_common(top_k)]


class RouterProbabilityPredictor(Predictor):
    def predict(self, layer: int, top_k: int, probabilities: Mapping[int, float] | None = None) -> list[int]:
        return [expert for expert, _ in sorted((probabilities or {}).items(), key=lambda item: item[1], reverse=True)[:top_k]]


class MarkovPredictor(Predictor):
    def __init__(self):
        self.transitions: dict[int, dict[tuple[int, ...], Counter[tuple[int, ...]]]] = defaultdict(lambda: defaultdict(Counter))
        self.previous: dict[int, tuple[int, ...]] = {}

    def observe(self, access: Access) -> None:
        current = tuple(sorted(access.experts))
        previous = self.previous.get(access.layer)
        if previous is not None:
            self.transitions[access.layer][previous][current] += 1
        self.previous[access.layer] = current

    def predict(self, layer: int, top_k: int, probabilities: Mapping[int, float] | None = None) -> list[int]:
        previous = self.previous.get(layer)
        if previous is None:
            return []
        candidates = self.transitions[layer].get(previous)
        if not candidates:
            return []
        expert_counts: Counter[int] = Counter()
        for next_set, count in candidates.items():
            for expert in next_set:
                expert_counts[expert] += count
        return [expert for expert, _ in expert_counts.most_common(top_k)]


class LightweightPredictor(Predictor):
    """Online feature-weight predictor without external ML dependencies."""

    def __init__(self, learning_rate: float = 0.05):
        self.learning_rate = learning_rate
        self.weights: dict[tuple[int, int], float] = defaultdict(float)
        self.frequency: dict[tuple[int, int], int] = defaultdict(int)

    def observe(self, access: Access) -> None:
        chosen = set(access.experts)
        candidates = set(chosen) | set(access.probabilities)
        for expert in candidates:
            key = (access.layer, expert)
            router = float(access.probabilities.get(expert, 0.0))
            target = 1.0 if expert in chosen else 0.0
            prediction = 1.0 / (1.0 + pow(2.718281828, -(self.weights[key] + router)))
            self.weights[key] += self.learning_rate * (target - prediction)
            if expert in chosen:
                self.frequency[key] += 1

    def predict(self, layer: int, top_k: int, probabilities: Mapping[int, float] | None = None) -> list[int]:
        experts = {expert for candidate_layer, expert in self.weights if candidate_layer == layer} | set(probabilities or {})
        scored = []
        for expert in experts:
            key = (layer, expert)
            score = self.weights[key] + float((probabilities or {}).get(expert, 0.0)) + 0.05 * self.frequency[key]
            scored.append((expert, score))
        return [expert for expert, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]]


class OraclePredictor(Predictor):
    def __init__(self, accesses: Sequence[Access]):
        self.by_layer: dict[int, deque[tuple[int, ...]]] = defaultdict(deque)
        for access in accesses:
            self.by_layer[access.layer].append(access.experts)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "OraclePredictor":
        accesses = []
        with Path(path).open() as handle:
            for line in handle:
                item = json.loads(line)
                if item.get("event") == "expert_access":
                    accesses.append(Access(item["token"], item["layer"], tuple(item["experts"]), item.get("probabilities", {})))
        return cls(accesses)

    def predict(self, layer: int, top_k: int, probabilities: Mapping[int, float] | None = None) -> list[int]:
        if not self.by_layer[layer]:
            return []
        return list(self.by_layer[layer].popleft())[:top_k]


def make_predictor(name: str, oracle_trace: str | None = None) -> Predictor:
    normalized = name.lower().replace("_", "-")
    if normalized in {"none", "no-prediction"}:
        return NoPredictor()
    if normalized in {"history", "frequency"}:
        return HistoryPredictor()
    if normalized in {"router", "router-probability"}:
        return RouterProbabilityPredictor()
    if normalized == "markov":
        return MarkovPredictor()
    if normalized in {"lightweight", "online"}:
        return LightweightPredictor()
    if normalized == "oracle":
        if not oracle_trace:
            raise ValueError("oracle predictor requires a trace")
        return OraclePredictor.from_jsonl(oracle_trace)
    raise ValueError(f"unknown predictor: {name}")
