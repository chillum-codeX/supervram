from __future__ import annotations

import concurrent.futures
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping

from .policies import CachePolicy, RouterAwarePolicy
from .store import TensorStore
from .trace import TraceWriter
from .types import CacheEntry, ExpertKey, ReadDecision


@dataclass
class UploadHandle:
    payload: object
    wait_ready: Callable[[], None] | None = None
    release: Callable[[], None] | None = None


@dataclass
class CacheStats:
    accesses: int = 0
    hits: int = 0
    misses: int = 0
    prefetches: int = 0
    useful_prefetches: int = 0
    evictions: int = 0
    bytes_read: int = 0
    read_ns: int = 0
    wait_ns: int = 0
    overlapped_bytes: int = 0
    waste_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0


class ExpertCache:
    """Bounded expert cache with asynchronous SSD reads.

    The payload callback is the device integration boundary. CUDA integrations return
    an UploadHandle whose readiness callback inserts the required stream/event wait.
    """

    def __init__(
        self,
        store: TensorStore,
        capacity_bytes: int,
        policy: CachePolicy,
        queue_depth: int = 4,
        trace: TraceWriter | None = None,
        upload: Callable[[ExpertKey, bytes], object | UploadHandle] | None = None,
        deterministic: bool = False,
    ):
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.store = store
        self.capacity_bytes = capacity_bytes
        self.policy = policy
        self.trace = trace or TraceWriter(None)
        self.upload = upload or (lambda key, payload: payload)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, queue_depth), thread_name_prefix="svram-io")
        self.entries: dict[ExpertKey, CacheEntry] = {}
        self.payloads: dict[ExpertKey, object] = {}
        self.inflight: dict[ExpertKey, concurrent.futures.Future] = {}
        self.prefetched: set[ExpertKey] = set()
        self.used_bytes = 0
        self.reserved_bytes = 0
        self.stats = CacheStats()
        self.deterministic = deterministic
        self._logical_ns = 0
        self._lock = threading.RLock()

    def _now(self) -> int:
        if not self.deterministic:
            return time.monotonic_ns()
        with self._lock:
            self._logical_ns += 1
            return self._logical_ns

    def update_probabilities(self, probabilities: Mapping[ExpertKey, float]) -> None:
        with self._lock:
            if isinstance(self.policy, RouterAwarePolicy):
                self.policy.update_probabilities(self.entries, probabilities)
            else:
                for key, entry in self.entries.items():
                    entry.predicted_probability = float(probabilities.get(key, 0.0))

    def prefetch(self, key: ExpertKey) -> None:
        with self._lock:
            if key in self.entries or key in self.inflight:
                return
            self.stats.prefetches += 1
            self.prefetched.add(key)
            self.inflight[key] = self.executor.submit(self._load, key, True)

    def issue(self, decision: ReadDecision) -> None:
        """Scheduled-read path for AdaptiveSpeculativeScheduler (PLAN_ADAPTIVE.md section 2.2),
        distinct from the reactive get()/blind prefetch(). A non-firing decision is a deliberate
        no-op: the entire point of the gate is to skip doubtful reads rather than issue them and
        pay for the waste, so `issue()` on a gated-off decision does nothing."""
        if decision.fire:
            self.prefetch(decision.key)

    def get(self, key: ExpertKey) -> object:
        request_ns = self._now()
        with self._lock:
            self.stats.accesses += 1
            if key in self.entries:
                self.stats.hits += 1
                entry = self.entries[key]
                entry.accesses += 1
                entry.last_access_ns = request_ns
                if key in self.prefetched:
                    self.stats.useful_prefetches += 1
                    self.stats.overlapped_bytes += entry.size
                    self.prefetched.discard(key)
                self.trace.emit("cache_hit", layer=key.layer, expert=key.expert)
                cached = self.payloads[key]
                return cached.payload if isinstance(cached, UploadHandle) else cached
            self.stats.misses += 1
            future = self.inflight.get(key)
            if future is None:
                future = self.executor.submit(self._load, key, False)
                self.inflight[key] = future
        result = future.result()
        wait_ns = 0 if self.deterministic else self._now() - request_ns
        with self._lock:
            self.stats.wait_ns += wait_ns
            if key in self.prefetched:
                self.stats.useful_prefetches += 1
                entry = self.entries.get(key)
                if entry is not None:
                    self.stats.overlapped_bytes += entry.size
                self.prefetched.discard(key)
            self.trace.emit("cache_miss", layer=key.layer, expert=key.expert, wait_ns=wait_ns)
        return result

    def _load(self, key: ExpertKey, prefetch: bool) -> object:
        reserved = 0
        try:
            start_ns = self._now()
            data = self.store.read(key)
            read_ns = len(data) if self.deterministic else self._now() - start_ns
            size = len(data)
            now_ns = self._now()
            with self._lock:
                if size > self.capacity_bytes:
                    raise MemoryError(f"expert {key} is larger than cache capacity")
                self._evict_for(size, now_ns)
                self.reserved_bytes += size
                reserved = size
            uploaded = self.upload(key, data)
            handle = uploaded if isinstance(uploaded, UploadHandle) else UploadHandle(uploaded)
            if handle.wait_ready is not None:
                handle.wait_ready()
            payload = handle
            with self._lock:
                if key in self.entries:
                    if handle.release is not None:
                        handle.release()
                    cached = self.payloads[key]
                    return cached.payload if isinstance(cached, UploadHandle) else cached
                self.reserved_bytes -= reserved
                reserved = 0
                self.entries[key] = CacheEntry(key, size, now_ns, now_ns, accesses=0, transfer_cost_ns=read_ns)
                self.payloads[key] = payload
                self.used_bytes += size
                self.stats.bytes_read += size
                self.stats.read_ns += read_ns
                self.trace.emit("expert_load", layer=key.layer, expert=key.expert, bytes=size, read_ns=read_ns, prefetch=prefetch)
            return payload
        finally:
            with self._lock:
                if reserved:
                    self.reserved_bytes -= reserved
                self.inflight.pop(key, None)
                if key not in self.entries:
                    self.prefetched.discard(key)

    def _evict_for(self, size: int, now_ns: int) -> None:
        excess = self.used_bytes + self.reserved_bytes + size - self.capacity_bytes
        if excess <= 0:
            return
        victims = self.policy.victims(self.entries, excess, now_ns)
        freed = 0
        for victim in victims:
            entry = self.entries.pop(victim)
            payload = self.payloads.pop(victim, None)
            if isinstance(payload, UploadHandle) and payload.release is not None:
                payload.release()
            if victim in self.prefetched:
                self.stats.waste_bytes += entry.size
            self.prefetched.discard(victim)
            self.used_bytes -= entry.size
            freed += entry.size
            self.stats.evictions += 1
            self.trace.emit("eviction", layer=victim.layer, expert=victim.expert, bytes=entry.size)
        if freed < excess:
            raise MemoryError("cache policy could not free enough space")

    def drain(self) -> None:
        while True:
            with self._lock:
                futures = list(self.inflight.values())
            if not futures:
                return
            for future in futures:
                future.result()

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "ExpertCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
