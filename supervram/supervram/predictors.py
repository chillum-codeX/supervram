from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from pathlib import Path
import json
from typing import Iterable, Mapping, Sequence

from .types import Access, Confidence, ExpertKey


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


class ConfidencePredictor:
    """Wraps a base predictor and calibrates a per-(layer, expert) confidence score from
    observed precision, so a downstream gate can fire only on near-certain guesses instead of
    blindly prefetching everything (PLAN_ADAPTIVE.md section 2.1 -- this is the fix for the
    measured result that history-based prefetch is a net loss: wrong reads cost a full SSD
    round-trip, so a predictor with no confidence signal can't tell a good guess from a bad one).

    Calibration is a bias-corrected EMA of correctness per (layer, expert): `conf = a*conf +
    (1-a)*hit`, seeded at `default_confidence` for anything not yet observed, where the *effective*
    weight `a` starts at 0 (the first observation fully determines confidence, i.e. a plain
    running mean) and ramps up to the configured `alpha` as observations accumulate --
    `a_n = min(alpha, (n-1)/n)` at the n-th observation of a given key.

    PLAN_ADAPTIVE.md Phase E: a fixed `alpha` (e.g. 0.9) needs about 1/(1-alpha) observations (44
    at alpha=0.9) to pull confidence from the neutral default up near 1.0, even for a predictor
    that is *always* right. The full 3,600-run scaled ablation matrix
    (results/ablation-simulate-ass-scaled/) found this costs real modeled throughput specifically
    against `predictor=oracle` (63/65 throughput regressions, worst -33.5%): the gate spends many
    steps under-trusting a predictor that was correct from the start, refusing prefetches blind
    prefetch would have fired correctly. Ramping the weight up with the observation count gives a
    perfect predictor near-certain confidence after only a handful of hits (n=1: conf=hit; n=2:
    conf=mean of both; ... converges to the fixed alpha once `n >= 1/(1-alpha)`), while a genuinely
    noisy predictor still settles near its true hit rate once enough samples accumulate -- the
    early volatility (a single early hit or miss swings confidence hard) is the accepted cost of
    not waiting ~44 steps to trust a predictor that never needed to prove itself that long.
    """

    def __init__(self, base: Predictor, alpha: float = 0.9, default_confidence: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.base = base
        self.alpha = alpha
        self.default_confidence = default_confidence
        self._confidence: dict[tuple[int, int], float] = {}
        self._counts: dict[tuple[int, int], int] = {}
        self._pending: dict[int, set[int]] = defaultdict(set)

    def confidence_of(self, layer: int, expert: int) -> float:
        return self._confidence.get((layer, expert), self.default_confidence)

    def predict(self, layer: int, top_k: int, probabilities: Mapping[int, float] | None = None) -> list[int]:
        """Predictor-compatible passthrough, so a ConfidencePredictor can be used anywhere a
        plain Predictor is expected."""
        return self.base.predict(layer, top_k, probabilities)

    def predict_confident(self, layer: int, top_k: int, probabilities: Mapping[int, float] | None = None) -> list[Confidence]:
        candidates = self.base.predict(layer, top_k, probabilities)
        return self.set_pending(layer, candidates)

    def set_pending(self, layer: int, candidates: list[int]) -> list[Confidence]:
        """Like predict_confident, but takes an already-computed candidate list instead of
        calling the base predictor again. For a caller (AdaptiveSpeculativeScheduler) that must
        not invoke the base predictor's predict() more than once per real access -- a stateful
        predictor like OraclePredictor mutates on every predict() call (it pops a queue), and
        calling it once per lookahead-window revisit instead of once per real access silently
        exhausts it early (PLAN_ADAPTIVE.md Phase E)."""
        self._pending[layer] = set(candidates)
        return [Confidence(expert, self.confidence_of(layer, expert)) for expert in candidates]

    def observe(self, access: Access) -> None:
        chosen = set(access.experts)
        pending = self._pending.pop(access.layer, None)
        if pending:
            for expert in pending:
                key = (access.layer, expert)
                hit = 1.0 if expert in chosen else 0.0
                count = self._counts.get(key, 0) + 1
                self._counts[key] = count
                effective_alpha = min(self.alpha, (count - 1) / count)
                previous = self._confidence.get(key, self.default_confidence)
                self._confidence[key] = effective_alpha * previous + (1.0 - effective_alpha) * hit
        self.base.observe(access)


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
