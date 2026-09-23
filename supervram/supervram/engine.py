from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .cache import ExpertCache
from .predictors import Predictor
from .scheduler import AdaptiveSpeculativeScheduler
from .types import Access, ExpertKey


@dataclass
class PredictionStats:
    opportunities: int = 0
    predicted: int = 0
    correct: int = 0

    @property
    def precision(self) -> float:
        return self.correct / self.predicted if self.predicted else 0.0

    @property
    def coverage(self) -> float:
        return self.correct / self.opportunities if self.opportunities else 0.0


class SuperVRAM:
    def __init__(self, cache: ExpertCache, predictor: Predictor, prefetch_depth: int = 0):
        self.cache = cache
        self.predictor = predictor
        self.prefetch_depth = prefetch_depth
        self.prediction_stats = PredictionStats()
        self._last_predictions: dict[int, set[int]] = {}
        self._scheduler: AdaptiveSpeculativeScheduler | None = None

    def _resolve(self, access: Access) -> list[object]:
        """Shared by process() and process_window(): score this access against whatever was
        predicted for this layer last time, resolve its experts through the cache, and let the
        (engine-owned) predictor observe the outcome. Prefetch issuance is the part that differs
        between the two callers and stays out of this method."""
        selected = set(access.experts)
        previous = self._last_predictions.pop(access.layer, set())
        self.prediction_stats.opportunities += len(selected)
        self.prediction_stats.predicted += len(previous)
        self.prediction_stats.correct += len(selected & previous)

        probabilities = {ExpertKey(access.layer, expert): probability for expert, probability in access.probabilities.items()}
        self.cache.update_probabilities(probabilities)
        payloads = [self.cache.get(ExpertKey(access.layer, expert)) for expert in access.experts]
        self.predictor.observe(access)
        return payloads

    def process(self, access: Access) -> list[object]:
        """Baseline path: predicts this layer's next visit and prefetches every candidate
        unconditionally (the measured net-loss behavior PLAN_ADAPTIVE.md replaces -- kept as the
        `--scheduler off` control so ASS's gains can be compared against it, not just asserted)."""
        payloads = self._resolve(access)
        predicted = self.predictor.predict(access.layer, self.prefetch_depth, access.probabilities)
        self._last_predictions[access.layer] = set(predicted)
        for expert in predicted:
            self.cache.prefetch(ExpertKey(access.layer, expert))
        return payloads

    def process_window(self, window: Sequence[Access], scheduler: AdaptiveSpeculativeScheduler) -> list[object]:
        """ASS-scheduled path (PLAN_ADAPTIVE.md section 2): window[0] is the access happening
        now and is resolved exactly like process(). The scheduler only plans for window[1:] --
        the genuinely *future* accesses -- not window[0] itself: by the time plan_window would
        run, window[0]'s experts are already known from the synchronous resolve above, so gating
        speculative reads for it would just be re-deciding on ground truth we already have (pure
        waste, not lookahead). The scheduler owns its own predictor/confidence state (kept
        separate from self.predictor so the same access stream is never observed twice into one
        stateful predictor's history)."""
        if not window:
            return []
        access = window[0]
        payloads = self._resolve(access)
        predicted = self.predictor.predict(access.layer, self.prefetch_depth, access.probabilities)
        self._last_predictions[access.layer] = set(predicted)
        self._scheduler = scheduler
        scheduler.record_and_observe(access)
        for decision in scheduler.plan_window(window[1:]):
            self.cache.issue(decision)
        return payloads

    def metrics(self) -> dict:
        result = {
            "cache": asdict(self.cache.stats) | {"hit_rate": self.cache.stats.hit_rate},
            "prediction": asdict(self.prediction_stats)
            | {"precision": self.prediction_stats.precision, "coverage": self.prediction_stats.coverage},
            "cache_used_bytes": self.cache.used_bytes,
            "cache_capacity_bytes": self.cache.capacity_bytes,
        }
        if self._scheduler is not None:
            result["scheduler"] = asdict(self._scheduler.stats) | {
                "gated": self._scheduler.stats.gated,
                "fire_rate": self._scheduler.stats.fire_rate,
            }
        return result
