from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .cache import ExpertCache
from .predictors import Predictor
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

    def process(self, access: Access) -> list[object]:
        selected = set(access.experts)
        previous = self._last_predictions.pop(access.layer, set())
        self.prediction_stats.opportunities += len(selected)
        self.prediction_stats.predicted += len(previous)
        self.prediction_stats.correct += len(selected & previous)

        probabilities = {ExpertKey(access.layer, expert): probability for expert, probability in access.probabilities.items()}
        self.cache.update_probabilities(probabilities)
        payloads = [self.cache.get(ExpertKey(access.layer, expert)) for expert in access.experts]
        self.predictor.observe(access)

        predicted = self.predictor.predict(access.layer, self.prefetch_depth, access.probabilities)
        self._last_predictions[access.layer] = set(predicted)
        for expert in predicted:
            self.cache.prefetch(ExpertKey(access.layer, expert))
        return payloads

    def metrics(self) -> dict:
        return {
            "cache": asdict(self.cache.stats) | {"hit_rate": self.cache.stats.hit_rate},
            "prediction": asdict(self.prediction_stats)
            | {"precision": self.prediction_stats.precision, "coverage": self.prediction_stats.coverage},
            "cache_used_bytes": self.cache.used_bytes,
            "cache_capacity_bytes": self.cache.capacity_bytes,
        }
