# Evidence ledger

## Measured in the current environment

- Host is an 8-vCPU AMD EPYC container with no visible NVIDIA driver/GPU.
- Workspace filesystem is NFS, not local NVMe.
- `libcufile` is absent; GDS candidate status is false.
- Native C++ reader test: passed.
- Python cache/store/predictor tests: 8/8 passed.
- Patched llama.cpp `test-arg-parser`: passed.
- Patched llama.cpp Qwen3 MoE synthetic architecture fixture generation: passed.
- Resident and mmap-backed synthetic Qwen3 MoE one-token CPU inference produced an identical complete-logit FNV-1a hash and checksum under the fixed entrypoint seed.
- Expert packer extracted and checksum-verified four expert records from the tiny generated GGUF.
- `llama-server` target build was blocked by an unrelated UI asset packaging failure after core libraries and server objects compiled; non-server targets compiled.

## Deterministic synthetic trace replay

`results/simulation-smoke.json` validates the software policies, not LLM throughput. Its exact run used 16 tokens, 4 layers, 16 experts, top-2 routing, 4 KiB synthetic experts and an 8-expert cache. It observed a 0.195 cache hit rate and Markov prediction precision 0.56/coverage 0.21875. These values must not be interpreted as Qwen routing behavior.

## Analytical projection

`results/roofline-projection.json` computes transfer lower bounds from explicit assumed dimensions, quantization overhead, NVMe/PCIe bandwidth, fixed read latency and compute time. It is not measured data.

## Published external

None collected by NovixCodeAgent. Literature claims belong to NovixWriterAgent and must be cited there.

## Pending RTX 3090 target measurements

- cuFile/GDS device-buffer operation and compatibility-mode status.
- BAR1/Resizable BAR, ACS/IOMMU and PCIe topology implications.
- Local NVMe bandwidth/IOPS/latency.
- Real Qwen3 MoE correctness and model/context limits.
- Prompt/decode throughput, TTFT, TPOT, latency percentiles.
- VRAM/RAM/page-cache use, CPU/GPU utilization and transfer overlap.
- Cache hit rate, router prediction quality and bytes/reads per token.
- Power and joules/token.
- All cache-size, prefetch, replacement and predictor ablations.

No pending value may be filled from the simulation or roofline artifacts.
