// SuperVRAM cuFile / GPUDirect Storage probe.
//
// Performs an actual cuFileRead into a cudaMalloc buffer and reports whether the
// nvidia-fs kernel driver is present (true peer-to-peer DMA) or cuFile fell back to
// compatibility mode (bounce through host memory). Also measures sustained read
// bandwidth of cuFile vs. O_DIRECT pread + cudaMemcpyAsync from pinned memory.
//
// Output: one JSON object on stdout.

#include <cuda_runtime.h>
#include <cufile.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct json_writer {
    std::string out;
    bool first = true;
    void key(const char * k) {
        if (!first) out += ",";
        first = false;
        out += "\"";
        out += k;
        out += "\":";
    }
    void kv(const char * k, const std::string & v) { key(k); out += "\"" + v + "\""; }
    void kv(const char * k, bool v) { key(k); out += v ? "true" : "false"; }
    void kv(const char * k, double v) { key(k); char b[64]; snprintf(b, sizeof(b), "%.6g", v); out += b; }
    void kv(const char * k, long long v) { key(k); out += std::to_string(v); }
};

std::string cufile_err(CUfileError_t e) {
    char b[128];
    snprintf(b, sizeof(b), "cufile_err=%d cuda_err=%d", (int) e.err, (int) e.cu_err);
    return b;
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool file_exists(const char * p) {
    struct stat st;
    return ::stat(p, &st) == 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file> [bytes_to_read (default 256MiB)] [chunk (default 4MiB)]\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];
    const size_t total_req = argc > 2 ? strtoull(argv[2], nullptr, 10) : (256ull << 20);
    const size_t chunk = argc > 3 ? strtoull(argv[3], nullptr, 10) : (4ull << 20);

    json_writer j;
    j.out += "{";
    j.kv("schema_version", 1LL);
    j.kv("evidence_class", std::string("measured_environment_probe"));
    j.kv("file", std::string(path));

    struct stat st;
    if (::stat(path, &st) != 0) {
        j.kv("error", std::string("stat failed: ") + strerror(errno));
        j.out += "}\n";
        fputs(j.out.c_str(), stdout);
        return 1;
    }
    const size_t total = std::min<size_t>(total_req, (size_t) st.st_size / chunk * chunk);
    j.kv("bytes", (long long) total);
    j.kv("chunk_bytes", (long long) chunk);

    // nvidia-fs kernel module presence is the discriminator between true GDS and compat mode
    const bool nvidia_fs_proc = file_exists("/proc/driver/nvidia-fs/stats");
    j.kv("nvidia_fs_kernel_module", nvidia_fs_proc);

    int dev_count = 0;
    cudaError_t ce = cudaGetDeviceCount(&dev_count);
    j.kv("cuda_device_count", (long long) (ce == cudaSuccess ? dev_count : 0));
    if (ce != cudaSuccess || dev_count == 0) {
        j.kv("error", std::string("no CUDA device: ") + cudaGetErrorString(ce));
        j.out += "}\n";
        fputs(j.out.c_str(), stdout);
        return 1;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    j.kv("cuda_device_name", std::string(prop.name));

    CUfileError_t status = cuFileDriverOpen();
    j.kv("cufile_driver_open_ok", status.err == CU_FILE_SUCCESS);
    if (status.err != CU_FILE_SUCCESS) {
        j.kv("cufile_driver_open_error", cufile_err(status));
    }

    bool compat_mode = true;
    if (status.err == CU_FILE_SUCCESS) {
        CUfileDrvProps_t props{};
        CUfileError_t ps = cuFileDriverGetProperties(&props);
        if (ps.err == CU_FILE_SUCCESS) {
            j.kv("nvfs_major", (long long) props.nvfs.major_version);
            j.kv("nvfs_minor", (long long) props.nvfs.minor_version);
            j.kv("nvfs_dstatusflags", (long long) props.nvfs.dstatusflags);
            j.kv("nvfs_dcontrolflags", (long long) props.nvfs.dcontrolflags);
            j.kv("nvfs_max_direct_io_size_kb", (long long) props.nvfs.max_direct_io_size);
            j.kv("nvfs_nvme_supported", (props.nvfs.dstatusflags & (1u << CU_FILE_NVME_SUPPORTED)) != 0);
            j.kv("nvfs_compat_mode_allowed", (props.nvfs.dcontrolflags & (1u << CU_FILE_ALLOW_COMPAT_MODE)) != 0);
            j.kv("max_device_cache_size_kb", (long long) props.max_device_cache_size);
            j.kv("max_device_pinned_mem_size_kb", (long long) props.max_device_pinned_mem_size);
            // If the nvidia-fs driver is missing, libcufile silently uses compat mode (host bounce).
            compat_mode = !nvidia_fs_proc || props.nvfs.major_version == 0;
        } else {
            j.kv("cufile_get_properties_error", cufile_err(ps));
        }
    }
    j.kv("cufile_compat_mode_inferred", compat_mode);

    // --- pread + pinned H2D baseline ---
    double pread_gbps = 0.0;
    {
        int fd = ::open(path, O_RDONLY | O_DIRECT);
        bool direct = fd >= 0;
        if (fd < 0) fd = ::open(path, O_RDONLY);
        j.kv("baseline_o_direct", direct);
        void * pinned = nullptr;
        void * dbuf = nullptr;
        cudaMallocHost(&pinned, chunk);
        cudaMalloc(&dbuf, chunk);
        cudaStream_t s;
        cudaStreamCreate(&s);
        const double t0 = now_s();
        size_t done = 0;
        bool ok = true;
        while (done < total) {
            ssize_t n = ::pread(fd, pinned, chunk, (off_t) done);
            if (n != (ssize_t) chunk) { ok = false; break; }
            cudaMemcpyAsync(dbuf, pinned, chunk, cudaMemcpyHostToDevice, s);
            cudaStreamSynchronize(s);
            done += chunk;
        }
        const double dt = now_s() - t0;
        pread_gbps = ok && dt > 0 ? (double) done / dt / 1e9 : 0.0;
        j.kv("baseline_pread_h2d_ok", ok);
        j.kv("baseline_pread_h2d_gbps", pread_gbps);
        cudaStreamDestroy(s);
        cudaFree(dbuf);
        cudaFreeHost(pinned);
        ::close(fd);
    }

    // --- cuFile read into device memory ---
    if (status.err == CU_FILE_SUCCESS) {
        int fd = ::open(path, O_RDONLY | O_DIRECT);
        if (fd < 0) {
            j.kv("cufile_read_error", std::string("O_DIRECT open failed: ") + strerror(errno));
        } else {
            CUfileDescr_t descr{};
            descr.handle.fd = fd;
            descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
            CUfileHandle_t fh;
            CUfileError_t hs = cuFileHandleRegister(&fh, &descr);
            if (hs.err != CU_FILE_SUCCESS) {
                j.kv("cufile_handle_register_error", cufile_err(hs));
            } else {
                void * dbuf = nullptr;
                cudaMalloc(&dbuf, chunk);
                CUfileError_t bs = cuFileBufRegister(dbuf, chunk, 0);
                j.kv("cufile_buf_register_ok", bs.err == CU_FILE_SUCCESS);
                const double t0 = now_s();
                size_t done = 0;
                bool ok = true;
                std::string err;
                while (done < total) {
                    ssize_t n = cuFileRead(fh, dbuf, chunk, (off_t) done, 0);
                    if (n != (ssize_t) chunk) {
                        ok = false;
                        err = "cuFileRead returned " + std::to_string((long long) n);
                        break;
                    }
                    done += chunk;
                }
                cudaDeviceSynchronize();
                const double dt = now_s() - t0;
                j.kv("cufile_read_ok", ok);
                if (!ok) j.kv("cufile_read_error", err);
                j.kv("cufile_read_gbps", ok && dt > 0 ? (double) done / dt / 1e9 : 0.0);
                // sanity: compare first chunk bytes against host read
                if (ok) {
                    void * aligned = nullptr;
                    if (posix_memalign(&aligned, 4096, chunk) == 0) {
                        std::vector<unsigned char> dev(chunk);
                        ssize_t hn = ::pread(fd, aligned, chunk, (off_t) (done - chunk));
                        cuFileRead(fh, dbuf, chunk, (off_t) (done - chunk), 0);
                        cudaMemcpy(dev.data(), dbuf, chunk, cudaMemcpyDeviceToHost);
                        j.kv("cufile_data_matches_pread", hn == (ssize_t) chunk && memcmp(aligned, dev.data(), chunk) == 0);
                        free(aligned);
                    }
                }
                if (bs.err == CU_FILE_SUCCESS) cuFileBufDeregister(dbuf);
                cudaFree(dbuf);
                cuFileHandleDeregister(fh);
            }
            ::close(fd);
        }
        cuFileDriverClose();
    }

    j.kv("verdict", std::string(
        status.err != CU_FILE_SUCCESS ? "cufile_unavailable" :
        compat_mode ? "cufile_compat_mode_host_bounce" : "cufile_gds_p2p_candidate"));
    j.out += "}\n";
    fputs(j.out.c_str(), stdout);
    return 0;
}
