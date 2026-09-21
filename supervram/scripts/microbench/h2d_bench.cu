// Host-to-device copy speed for expert-sized transfers: pageable vs pinned vs registered memory.
// Build: nvcc -O2 -o /tmp/h2d_bench h2d_bench.cu     Run: /tmp/h2d_bench [model.gguf]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("  %s failed: %s\n", #x, cudaGetErrorString(e)); return false; } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// copy `n` random chunks of `chunk` bytes from host buffer to a device buffer, syncing every `group` chunks
static bool run(const char * label, const uint8_t * host, size_t host_bytes, size_t chunk, int n, int group) {
    uint8_t * dev; CK(cudaMalloc(&dev, 64u << 20)); cudaStream_t st; CK(cudaStreamCreate(&st));
    srand(1); std::vector<size_t> off(n); for (auto & o : off) o = ((size_t) rand() * 4096ull) % (host_bytes - chunk) & ~4095ull;
    for (int i = 0; i < 20; i++) cudaMemcpyAsync(dev, host + off[i], chunk, cudaMemcpyHostToDevice, st); cudaStreamSynchronize(st);
    double t = now();
    for (int i = 0; i < n; i++) { cudaMemcpyAsync(dev + (i % 32) * chunk, host + off[i], chunk, cudaMemcpyHostToDevice, st); if ((i + 1) % group == 0) cudaStreamSynchronize(st); }
    cudaStreamSynchronize(st); t = now() - t;
    printf("  %-34s chunk %5.2f MiB, sync every %2d: %6.2f GB/s  (%.0f us per chunk)\n", label, chunk / 1048576.0, group, n * (double) chunk / t / 1e9, 1e6 * t / n);
    cudaFree(dev); cudaStreamDestroy(st); return true;
}

int main(int argc, char ** argv) {
    const size_t big = 4ull << 30, chunk = 1671168;   // one Q8_0 expert slice (1.59 MiB)
    printf("Host->device copies, PCIe link of this GPU\n");
    uint8_t * pageable = (uint8_t *) malloc(big); memset(pageable, 1, big);
    uint8_t * pinned = nullptr;
    if (cudaHostAlloc((void **) &pinned, big, cudaHostAllocDefault) != cudaSuccess) { printf("  cudaHostAlloc 4 GiB failed\n"); pinned = nullptr; } else memset(pinned, 1, big);
    for (int g : {1, 3, 24}) {
        run("pageable (malloc)", pageable, big, chunk, 2000, g);
        if (pinned) run("pinned (cudaHostAlloc)", pinned, big, chunk, 2000, g);
    }
    run("pageable, 3 chunks = 4.8 MiB", pageable, big, 3 * chunk, 700, 1);
    if (pinned) run("pinned,   3 chunks = 4.8 MiB", pinned, big, 3 * chunk, 700, 1);
    // page-locking an existing (file-backed) mapping, as the model mmap is
    if (argc > 1) {
        int fd = open(argv[1], O_RDONLY); size_t len = 4ull << 30;
        uint8_t * map = (uint8_t *) mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        for (size_t i = 0; i < len; i += 4096) (void) *(volatile uint8_t *) (map + i);   // make resident
        run("mmap'd file, not registered", map, len, chunk, 2000, 3);
        cudaError_t e = cudaHostRegister(map, len, cudaHostRegisterReadOnly);
        printf("  cudaHostRegister(4 GiB file mapping): %s\n", cudaGetErrorString(e));
        if (e == cudaSuccess) run("mmap'd file, registered (pinned)", map, len, chunk, 2000, 3);
    }
    // how much can be page-locked?
    size_t total = 0; std::vector<void *> blocks; void * p;
    while (total < (60ull << 30) && cudaHostAlloc(&p, 1ull << 30, cudaHostAllocDefault) == cudaSuccess) { blocks.push_back(p); total += 1ull << 30; }
    printf("  pinned memory obtainable with cudaHostAlloc: at least %zu GiB (stopped at %s)\n", total >> 30, total >= (60ull << 30) ? "60 GiB test cap" : "allocation failure");
    return 0;
}
