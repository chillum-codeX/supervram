# SuperVRAM progress

## Status
- 2026-09-21: Workspace initialized; target RTX 3090/NVMe host unavailable in this environment. No target-hardware measurements claimed.
- 2026-09-21: Pinned llama.cpp at `ce8caa6e60a03093351d6016a818720e0d46f0fb` and mapped Qwen3 MoE loading, routing, scheduler and CUDA MMID paths.
- 2026-09-21: Added explicit disabled-by-default Qwen3 MoE mmap expert-storage patch and verified resident/mmap exact logits-checksum equality on a generated tiny model.
- 2026-09-21: Implemented aligned tensor store, bounded async cache, four eviction policies, six predictor modes, tracing, GGUF expert packer, native async reader and hardware capability probe.
- 2026-09-21: Added target benchmark harness, 720-run ablation manifest, analytical roofline, metrics schema and RTX 3090 execution protocol.
- 2026-09-21: `/home/novix/workspace/project/entrypoint.sh` completed successfully. Native CTest 1/1 and Python pytest 8/8 passed; llama parser and Qwen3 architecture tests passed.

## Active processes

None.

## Critical boundary

The integrated llama.cpp path is a demand-paged CPU expert baseline. The compact GPU expert cache remains a source-level design plus independently tested policy prototype and requires RTX 3090/CUDA implementation and validation at the scheduler/backend boundary.
