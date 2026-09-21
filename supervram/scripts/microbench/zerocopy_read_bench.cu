// Can a GPU kernel read expert weights straight from pinned host RAM fast enough? (zero-copy expert tier)
// The kernel streams `bytes` with 16-byte loads and sums them (a stand-in for a quantized matrix-vector product).
// Build: nvcc -O2 -o /tmp/zc_bench zerocopy_read_bench.cu
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__global__ void stream_sum(const uint4 * __restrict__ p, size_t n16, unsigned * out) {
    unsigned s = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x * blockDim.x) { uint4 v = p[i]; s += v.x ^ v.y ^ v.z ^ v.w; }
    if (s == 0xdeadbeef) out[0] = s;
}
static void bench(const char * label, const void * dptr, size_t bytes, int blocks, int reps, unsigned * out) {
    for (int i = 0; i < 5; i++) stream_sum<<<blocks, 256>>>((const uint4 *) dptr, bytes / 16, out);
    cudaDeviceSynchronize();
    double t = now();
    for (int i = 0; i < reps; i++) { stream_sum<<<blocks, 256>>>((const uint4 *) dptr, bytes / 16, out); cudaDeviceSynchronize(); }   // sync each: worst case, as one miss
    t = (now() - t) / reps;
    printf("  %-30s %6.2f MiB read, %4d blocks: %7.1f us  = %5.1f GB/s\n", label, bytes / 1048576.0, blocks, t * 1e6, bytes / t / 1e9);
}
int main() {
    unsigned * out; cudaMalloc(&out, 64);
    const size_t big = 1ull << 30;
    void * host; if (cudaHostAlloc(&host, big, cudaHostAllocMapped) != cudaSuccess) { printf("mapped alloc failed\n"); return 1; }
    memset(host, 1, big);
    void * hdev; cudaHostGetDevicePointer(&hdev, host, 0);
    void * vram; cudaMalloc(&vram, big); cudaMemset(vram, 1, big);
    printf("GPU kernel reading pinned host RAM over PCIe (zero-copy) vs VRAM\n");
    for (size_t mb : {5, 16, 64}) {
        size_t bytes = mb * 1048576ull; if (mb == 5) bytes = 5 * 1048576ull;
        for (int blocks : {82, 328, 1312}) bench("zero-copy (pinned host RAM)", hdev, bytes, blocks, 50, out);
        bench("VRAM", vram, bytes, 1312, 50, out);
    }
    // ten different experts (5 MB each) read in ONE kernel launch, like a layer's misses batched
    printf("  (for comparison a cudaMemcpy of 5 MB from pinned RAM took ~450 us = 11 GB/s in the previous test)\n");
    return 0;
}
