from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._handle = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", buffering=1)

    def emit(self, event: str, **fields: Any) -> None:
        if self._handle is None:
            return
        record = {"timestamp_ns": time.monotonic_ns(), "event": event, **fields}
        with self._lock:
            self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
