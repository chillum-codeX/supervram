# Metrics and evidence schema

All JSON artifacts include `schema_version` and an `evidence_class`.

Allowed evidence classes:

- `measured_environment_probe`: measured properties of the current host.
- `deterministic_synthetic_trace_replay`: software-policy validation using generated accesses.
- `analytical_projection`: formula-based scenario, never a benchmark.
- `published_external`: copied from a cited external source by the writer.
- `target_hardware_measurement`: produced only on the specified target host.
- `pending_target_measurement`: planned field with no observation.

Target benchmark records contain:

- Configuration: git revision, build flags, model/quantization hash, command line, cache/predictor/backend settings.
- Latency: request, TTFT proxy, prompt time, decode time, TPOT, p50/p95/p99.
- Throughput: prompt and decode tokens/s.
- Memory: VRAM, process RSS, page cache and pinned staging budget.
- Transfer: backend, read count, bytes, read latency, queue latency, H2D latency and overlap.
- Storage: NVMe bandwidth/IOPS/await/queue depth from system tools.
- Prediction: opportunities, top-k predictions, correct predictions, precision and coverage.
- Cache: accesses, hits, misses, useful prefetches, evictions and occupancy.
- Utilization: GPU utilization, GPU idle/overlap proxy and CPU utilization.
- Energy: sampled watts and integrated joules/token when the required sensors exist.

Unavailable counters remain JSON `null` and receive an `unavailable_reason`; they are never imputed.
