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
