from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


@dataclass(frozen=True, order=True)
class ExpertKey:
    layer: int
    expert: int

    def __str__(self) -> str:
        return f"L{self.layer}:E{self.expert}"


@dataclass(frozen=True)
class TensorExtent:
    key: ExpertKey
    offset: int
    length: int
    checksum: str = ""
    tensors: tuple[str, ...] = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


class Residency(str, Enum):
    COLD = "nvme"
    LOADING = "loading"
    RESIDENT = "vram"


@dataclass
class Access:
    token: int
    layer: int
    experts: tuple[int, ...]
    probabilities: Mapping[int, float] = field(default_factory=dict)


@dataclass
class CacheEntry:
    key: ExpertKey
    size: int
    loaded_ns: int
    last_access_ns: int
    accesses: int = 0
    predicted_probability: float = 0.0
    transfer_cost_ns: int = 0
    dirty: bool = False


@dataclass(frozen=True)
class Confidence:
    """A predicted expert paired with a calibrated probability that it will actually be used,
    as produced by ConfidencePredictor (see predictors.py)."""

    expert: int
    probability: float


@dataclass(frozen=True)
class ReadDecision:
    """The gate's verdict for one candidate speculative read (PLAN_ADAPTIVE.md section 2.1)."""

    key: ExpertKey
    p_use: float
    t_read: float
    fire: bool


@dataclass
class CostModelParams:
    """Constants for the roofline-style overlap cost model (PLAN_ADAPTIVE.md sections 2.2/2.3).

    Defaults are taken from this project's own measured evidence
    (docs/IMPLEMENTATION_STATUS.md, writer_handoff/EVIDENCE_LEDGER.md), not guessed: ~0.19 ms/layer
    of GPU compute, a 2.4 GB/s drive ceiling, and the D=4 lookahead window that the
    prefetch_predictor_eval.py sweep found already captures most of the oracle's 1-layer gain.
    """

    drive_gbps: float = 2.4
    compute_ms_per_layer: float = 0.19
    lookahead_d: int = 4
    gate_threshold: float = 0.3
