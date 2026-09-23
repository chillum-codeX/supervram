from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .predictors import ConfidencePredictor, Predictor
from .types import Access, CostModelParams, ExpertKey, ReadDecision


@dataclass
class SchedulerStats:
    considered: int = 0
    fired: int = 0

    @property
    def gated(self) -> int:
        return self.considered - self.fired

    @property
    def fire_rate(self) -> float:
        return self.fired / self.considered if self.considered else 0.0


class AdaptiveSpeculativeScheduler:
    """Confidence-gated, cost-aware, lookahead prefetch scheduler (PLAN_ADAPTIVE.md section 2).

    The measured baseline (writer_handoff/EVIDENCE_LEDGER.md) is that blind prefetch is a net
    loss: routing-history predictors are right only 1.4-2% of the time, and every wrong read
    costs a full SSD round-trip, so unconditional prefetching at 0.5-1.0 wrong reads per useful
    one is *slower* than no prefetch at all (0.82-0.96x). This scheduler turns "prefetch every
    predicted candidate" into "prefetch only when it earns its share of a shared budget":

    1. A `gate_threshold` floor (default 0.3, PLAN_ADAPTIVE.md section 7) refuses any candidate
       below that calibrated confidence outright, regardless of budget.
    2. The survivors share one compute-time budget for the whole lookahead window (section 2.2's
       "amortize latency over D layers of compute"), spent on the most-confident candidates first
       (ties broken by read size descending, per section 7's "sort in-flight reads by size desc to
       saturate the drive") until the budget runs out; anything left over is gated off.

    An earlier version of this gate computed `p_use * t_read < budget - (1 - p_use) * t_read` per
    candidate independently. That formula is a trap: the `p_use` terms algebraically cancel to
    `t_read < budget`, so confidence stopped affecting anything past the floor check, and firing
    only tracked "is this one read cheap," not "is this read worth its share of a scarce shared
    budget" -- verified against the real harness (a first --scheduler ass run had *higher*
    waste_bytes than the --scheduler off baseline it's supposed to beat) before landing on the
    budget-pooled version below.
    """

    def __init__(
        self,
        predictor: Predictor | ConfidencePredictor,
        extent_length: Callable[[ExpertKey], int],
        params: CostModelParams | None = None,
        top_k: int = 4,
        alpha: float = 0.9,
    ):
        self.confidence = predictor if isinstance(predictor, ConfidencePredictor) else ConfidencePredictor(predictor, alpha=alpha)
        self.extent_length = extent_length
        self.params = params or CostModelParams()
        self.top_k = top_k
        self.stats = SchedulerStats()
        # Caches the base predictor's candidates per real access (token, layer), not per call --
        # see _predict_once.
        self._predict_cache: dict[tuple[int, int], list[int]] = {}

    def _predict_once(self, access: Access) -> list[int]:
        """Calls the base predictor's predict() at most once per real access, however many
        overlapping lookahead windows revisit it. Without this, a stateful predictor like
        OraclePredictor -- whose predict() mutates an internal queue via popleft() -- gets called
        many times per real access (once per window-slide it appears in, plus once more from
        record_and_observe), silently draining its queue early and making it go blind for the
        rest of the run. Verified empirically: a 40-access trace produced 194 predict() calls
        before this cache existed (PLAN_ADAPTIVE.md Phase E). Evicted once the access is resolved
        (record_and_observe), so this never holds more than ~lookahead_d entries at once."""
        key = (access.token, access.layer)
        cached = self._predict_cache.get(key)
        if cached is None:
            cached = self.confidence.base.predict(access.layer, self.top_k, access.probabilities)
            self._predict_cache[key] = cached
        return cached

    def t_read_seconds(self, key: ExpertKey) -> float:
        try:
            length = self.extent_length(key)
        except KeyError:
            return 0.0
        drive_bytes_per_second = self.params.drive_gbps * 1e9
        return length / drive_bytes_per_second if drive_bytes_per_second else 0.0

    def plan_window(self, window: Sequence[Access]) -> list[ReadDecision]:
        """Read-only gating over the next D accesses (section 2.2's lookahead window). Candidates
        come from the *base* predictor (`predict(layer, probs)`, per the plan's own pseudocode in
        section 4) -- the same call `record_and_observe` uses to seed calibration -- not straight
        from the router's raw probabilities, so every candidate gated here is one the calibration
        EMA actually has (or will have) an opinion on. Using the raw router score instead would
        gate a different set of experts than the ones being calibrated, leaving most candidates
        stuck at the neutral default confidence and defeating the gate. Confidence is calibrated,
        not the raw router score, so a router that's confident but historically wrong still gets
        suppressed. This method does not mutate calibration state; call `record_and_observe` once
        per access as it actually resolves to keep the calibration EMA up to date.
        """
        if not window:
            return []
        budget_seconds = (self.params.compute_ms_per_layer / 1000.0) * len(window)
        pool: list[tuple[ExpertKey, float, float]] = []
        for access in window:
            for expert in self._predict_once(access):
                key = ExpertKey(access.layer, expert)
                p_use = self.confidence.confidence_of(access.layer, expert)
                pool.append((key, p_use, self.t_read_seconds(key)))

        # A fair comparison against the `--scheduler off` baseline (PLAN_ADAPTIVE.md section 5.2:
        # "at equal prefetch-depth") needs an equal total speculative-read budget, not an equal
        # per-access one: pooling every window entry's own top_k would consider len(window)x more
        # candidates per step than the baseline's single-layer top_k. Keep only the top_k
        # most-confident candidates across the *whole* window -- same total depth as the
        # baseline, just spent more intelligently across the lookahead window instead of blindly
        # on one layer.
        pool.sort(key=lambda c: (c[1], c[2]), reverse=True)
        pool = pool[: self.top_k]

        # Of that pool, spend the window's shared compute-time budget on the most-confident
        # candidates first (largest read first among ties, to saturate the drive per section 7),
        # refusing anything once the budget runs out -- this is what makes firing depend on both
        # confidence AND a scarce shared resource, instead of each candidate being judged alone.
        remaining_budget = budget_seconds
        decisions: list[ReadDecision] = []
        for key, p_use, t_read in pool:
            fire = p_use >= self.params.gate_threshold and t_read <= remaining_budget
            if fire:
                remaining_budget -= t_read
            decisions.append(ReadDecision(key, p_use, t_read, fire))
            self.stats.considered += 1
            if fire:
                self.stats.fired += 1
        return decisions

    def record_and_observe(self, access: Access) -> None:
        """Call once per access as it is actually resolved, mirroring SuperVRAM.process()'s own
        predict-then-compare-next-time rhythm: first score this access's outcome against whatever
        was predicted the last time this layer was visited (updates the calibration EMA), then
        record a fresh prediction for this layer's next visit. Uses the same per-access cache as
        plan_window (_predict_once) rather than calling the base predictor again -- see there."""
        self.confidence.observe(access)
        self.confidence.set_pending(access.layer, self._predict_once(access))
        self._predict_cache.pop((access.token, access.layer), None)
