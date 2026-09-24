from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Lifecycle(str, Enum):
    PINNED = "pinned"
    ACTIVE = "active"
    STALE = "stale"
    ARCHIVAL = "archival"


@dataclass(frozen=True)
class ClassifiedItem:
    item: dict[str, Any]
    state: Lifecycle
    score: float
    age: int


def classify(item: dict[str, Any], current_turn: int, stale_after: int, archival_after: int) -> ClassifiedItem:
    metadata = item.get("metadata", {}) or {}
    turn = int(metadata.get("turn", current_turn))
    age = max(0, current_turn - turn)
    if metadata.get("pinned") or item.get("role") in {"system", "developer"}:
        state = Lifecycle.PINNED
    elif metadata.get("active") or age <= stale_after:
        state = Lifecycle.ACTIVE
    elif age <= archival_after:
        state = Lifecycle.STALE
    else:
        state = Lifecycle.ARCHIVAL
    importance = float(metadata.get("importance", 1.0))
    if item.get("is_error") or metadata.get("error"):
        importance += 4.0
    if item.get("role") == "user":
        importance += 3.0
    if metadata.get("modified_files"):
        importance += 2.0
    score = importance * 10.0 - age
    return ClassifiedItem(item, state, score, age)
