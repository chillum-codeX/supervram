from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SuperVRAMConfig:
    enabled: bool = False
    store_path: Path | None = None
    cache_bytes: int = 18 * 1024**3
    transfer_backend: str = "auto"
    queue_depth: int = 4
    prefetch_depth: int = 4
    policy: str = "router-aware"
    predictor: str = "router-probability"
    pinned_staging_bytes: int = 512 * 1024**2
    direct_io_alignment: int = 4096
    strict: bool = False
    trace_path: Path | None = None
    oracle_trace: Path | None = None

    def validate(self) -> None:
        if self.enabled and self.store_path is None:
            raise ValueError("store_path is required when SuperVRAM is enabled")
        if self.cache_bytes <= 0:
            raise ValueError("cache_bytes must be positive")
        if self.queue_depth <= 0:
            raise ValueError("queue_depth must be positive")
        if self.prefetch_depth < 0:
            raise ValueError("prefetch_depth must be non-negative")
        if self.pinned_staging_bytes < 0:
            raise ValueError("pinned_staging_bytes must be non-negative")
        if self.direct_io_alignment <= 0 or self.direct_io_alignment & (self.direct_io_alignment - 1):
            raise ValueError("direct_io_alignment must be a positive power of two")
