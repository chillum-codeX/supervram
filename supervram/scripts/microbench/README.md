# Micro-benchmarks

Standalone CUDA programs used to size the design of the zero-copy expert tier (build with `nvcc -O2`).

- `h2d_bench.cu`: host-to-device copy speed for expert-sized chunks: pageable vs pinned memory, and how much RAM can be pinned.
  Measured here: pageable 6.6-8.3 GB/s, pinned 11.8-12.1 GB/s, at least 60 GiB pinnable; `cudaHostRegister` on a file mapping is not supported.
- `zerocopy_read_bench.cu`: a GPU kernel streaming 5-64 MiB straight from pinned host RAM (UVA) vs VRAM. Measured: 10.8-11.5 GB/s from RAM
  (full PCIe 3.0 x16 rate, even with 82 blocks), 390-670 GB/s from VRAM.
