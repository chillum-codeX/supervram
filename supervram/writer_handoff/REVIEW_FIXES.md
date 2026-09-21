# Review fixes applied

The implementation review found several harness and core issues. The following were addressed before handoff:

- Added exception-safe cache inflight cleanup and retry behavior.
- Added capacity reservation before upload and accounted for reserved bytes.
- Preserved router-aware probabilities across layers.
- Added synchronized O_DIRECT fallback descriptor without closing the shared descriptor.
- Added native reader alignment validation and shared-state capture for async reads.
- Added deterministic replay mode with logical clock, single worker and drain before metrics.
- Disabled invalid ablation `target` mode until a compact GPU-cache CLI exists.
- Changed ablation runner status files so child simulation metrics are not overwritten.
- Converted requested cache GiB into actual byte capacities and records effective capacity.
- Added `llama-bench --moe-expert-storage resident|mmap` parsing, propagation, instance equality and result fields.
- Reset speculative draft model storage to resident when target expert mmap mode is used.
- Fixed seeded Qwen3 fixture generation in the reproducibility entrypoint.
- Correctness smoke now compares a complete-logit byte hash in addition to a scalar checksum.
- Reopened and checksum-verified every packed expert extent in the entrypoint.
- Renamed non-streaming HTTP response first-byte metric so it is not misreported as TTFT.
- Hardware direct-I/O probe now uses a unique temporary path and does not delete pre-existing files.

Still unresolved by design:

- A true CUDA compact expert cache and cuFile device-DMA path are not implemented or measured without the target host.
- The target HTTP harness remains a host measurement harness and does not self-prove RTX 3090 identity; it records an unverified evidence class.
- The Python O_DIRECT path is a conservative fallback implementation; target-host cuFile behavior requires a real device-buffer test.
