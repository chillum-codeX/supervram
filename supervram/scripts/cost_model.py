from __future__ import annotations

"""Roofline-style analytical cost model (PLAN_ADAPTIVE.md section 2.3).

Turns the cache's byte counters into a *modeled* decode throughput, so the harness can report
"modeled decode X t/s vs baseline Y t/s" instead of only "hit rate 97%" -- the number success
criterion 3 actually asks for. This is explicitly an analytical projection, matching this
project's evidence-class convention (see results/roofline-projection.json): it is not a hardware
measurement, and simulate_trace.py's deterministic replay mode does not produce real wall-clock
timings to model from in the first place (CacheStats.wait_ns/read_ns are logical byte-count
proxies in that mode, not nanoseconds -- see cache.py's `_now()`).

Model, in wall-clock seconds for the whole run:
    compute_s          = compute_ms_per_layer/1000 * steps   (steps = one layer visited once)
    blocking_bytes      = bytes that were NOT hidden behind compute: bytes_read minus whatever
                          was overlapped (a prefetch that completed before it was needed) minus
                          whatever was wasted (a prefetch that was never needed at all, so it
                          never sat on any request's critical path either)
    blocking_io_s       = blocking_bytes / (drive_gbps * 1e9)
    wall_s              = max(compute_s, blocking_io_s)        -- the roofline: compute and I/O
                          for the *blocking* portion cannot both be hidden; overlapped bytes are
                          free by construction, so they don't add to either term
    modeled_tokens_per_second = tokens / wall_s

Known simplification, stated rather than hidden: this v1 model has no drive-contention term --
wasted prefetch bytes still consume real drive bandwidth that a concurrent blocking read would
have to queue behind, which would push blocking_io_s up. Treating waste as fully free of cost is
optimistic for ASS and, if anything, understates its own overhead relative to a lower-waste
scheduler, so it does not manufacture ASS's advantage.
"""

from dataclasses import asdict, dataclass


@dataclass
class CostModelResult:
    evidence_class: str = "analytical_roofline_projection"
    warning: str = "Modeled projection from replay byte-counters, not a hardware measurement."
    steps: int = 0
    tokens: int = 0
    compute_s: float = 0.0
    blocking_bytes: int = 0
    blocking_io_s: float = 0.0
    wall_s: float = 0.0
    modeled_tokens_per_second: float = 0.0


def project_throughput(cache_metrics: dict, params, tokens: int, steps: int) -> CostModelResult:
    """`cache_metrics` is the `metrics()["cache"]` dict simulate_trace.py already produces
    (a plain dict, so this has no import-time dependency on the supervram package's internal
    dataclass shape). `params` is a `supervram.types.CostModelParams`. `steps` is the number of
    *layer visits* (tokens * layers) -- deliberately NOT `cache_metrics["accesses"]`, which counts
    one cache.get() per expert (top_k of them per layer visit, all resolved in one GPU compute
    step): using accesses as steps overcounts compute by a factor of top_k and makes the model
    permanently compute-bound, hiding any I/O-side difference between strategies entirely (caught
    empirically: an --scheduler off vs ass comparison reported identical modeled_tokens_per_second
    despite ass having real, lower blocking_bytes, until this was fixed).
    """
    bytes_read = int(cache_metrics.get("bytes_read", 0))
    overlapped_bytes = int(cache_metrics.get("overlapped_bytes", 0))
    waste_bytes = int(cache_metrics.get("waste_bytes", 0))
    blocking_bytes = max(0, bytes_read - overlapped_bytes - waste_bytes)

    compute_s = (params.compute_ms_per_layer / 1000.0) * steps
    drive_bytes_per_second = params.drive_gbps * 1e9
    blocking_io_s = blocking_bytes / drive_bytes_per_second if drive_bytes_per_second else 0.0
    wall_s = max(compute_s, blocking_io_s)
    tps = (tokens / wall_s) if wall_s > 0 else 0.0

    return CostModelResult(
        steps=steps,
        tokens=tokens,
        compute_s=compute_s,
        blocking_bytes=blocking_bytes,
        blocking_io_s=blocking_io_s,
        wall_s=wall_s,
        modeled_tokens_per_second=tps,
    )


def project_throughput_dict(cache_metrics: dict, params, tokens: int, steps: int) -> dict:
    return asdict(project_throughput(cache_metrics, params, tokens, steps))
