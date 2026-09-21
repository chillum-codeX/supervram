#include "supervram_native.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace supervram {

namespace {

bool can_open_direct(const std::string & path) {
#ifdef O_DIRECT
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd >= 0) {
        ::close(fd);
        return true;
    }
#else
    (void) path;
#endif
    return false;
}

bool library_available(const char * name) {
    void * handle = ::dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        return false;
    }
    ::dlclose(handle);
    return true;
}

} // namespace

const char * backend_name(backend_kind value) {
    switch (value) {
        case backend_kind::automatic:    return "auto";
        case backend_kind::cufile:       return "cufile";
        case backend_kind::io_uring:     return "io_uring";
        case backend_kind::direct_pread: return "direct_pread";
        case backend_kind::pread:        return "pread";
    }
    return "unknown";
}

backend_kind parse_backend(const std::string & value) {
    if (value == "auto") return backend_kind::automatic;
    if (value == "cufile") return backend_kind::cufile;
    if (value == "io_uring") return backend_kind::io_uring;
    if (value == "direct" || value == "direct_pread") return backend_kind::direct_pread;
    if (value == "pread") return backend_kind::pread;
    throw std::invalid_argument("unknown backend: " + value);
}

capability_report probe(const std::string & path, backend_kind requested) {
    capability_report report;
    report.cuda_visible = library_available("libcuda.so.1");
    report.cufile_library = library_available("libcufile.so.0") || library_available("libcufile.so");
#ifdef SUPERVRAM_HAVE_IO_URING
    report.io_uring_headers = true;
#endif
    report.direct_io_open = can_open_direct(path);

    backend_kind selected = requested;
    if (requested == backend_kind::automatic) {
#ifdef SUPERVRAM_HAVE_CUFILE
        if (report.cuda_visible && report.cufile_library) {
            selected = backend_kind::cufile;
        } else
#endif
#ifdef SUPERVRAM_HAVE_IO_URING_RUNTIME
        if (report.io_uring_headers && report.direct_io_open) {
            selected = backend_kind::io_uring;
        } else
#endif
        if (report.direct_io_open) {
            selected = backend_kind::direct_pread;
        } else {
            selected = backend_kind::pread;
        }
    }
    if (selected == backend_kind::cufile) {
#if defined(SUPERVRAM_HAVE_CUFILE)
        if (!(report.cuda_visible && report.cufile_library)) {
            report.rejection_reasons.emplace_back("cuFile requires visible CUDA driver and libcufile");
            selected = backend_kind::pread;
        }
#else
        report.rejection_reasons.emplace_back("cuFile runtime is capability-probed but this build has no cuFile device-read implementation");
        selected = backend_kind::pread;
#endif
    }
    if (selected == backend_kind::io_uring) {
#if !defined(SUPERVRAM_HAVE_IO_URING_RUNTIME)
        report.rejection_reasons.emplace_back("io_uring runtime was not built");
        selected = report.direct_io_open ? backend_kind::direct_pread : backend_kind::pread;
#endif
    }
    if (selected == backend_kind::direct_pread && !report.direct_io_open) {
        report.rejection_reasons.emplace_back("O_DIRECT open failed");
        selected = backend_kind::pread;
    }
    if (selected == backend_kind::cufile) {
        report.rejection_reasons.emplace_back("cuFile runtime compatibility still requires target-host gdscheck and an actual device-buffer read");
    }
    report.selected_backend = backend_name(selected);
    return report;
}

struct reader::impl {
    std::string path;
    backend_kind selected;
    size_t alignment;
    int fd = -1;
    capability_report report;

    impl(std::string path_in, backend_kind requested, size_t alignment_in) : path(std::move(path_in)), alignment(alignment_in), report(probe(path, requested)) {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0 || alignment % sizeof(void *) != 0) {
            throw std::invalid_argument("alignment must be a nonzero power-of-two multiple of pointer size");
        }
        selected = parse_backend(report.selected_backend);
        int flags = O_RDONLY;
#ifdef O_DIRECT
        if (selected == backend_kind::direct_pread || selected == backend_kind::io_uring) {
            flags |= O_DIRECT;
        }
#endif
        fd = ::open(path.c_str(), flags);
        if (fd < 0 && flags != O_RDONLY) {
            selected = backend_kind::pread;
            report.selected_backend = backend_name(selected);
            report.rejection_reasons.emplace_back("direct open failed at reader construction");
            fd = ::open(path.c_str(), O_RDONLY);
        }
        if (fd < 0) {
            throw std::runtime_error("cannot open tensor store: " + path + ": " + std::strerror(errno));
        }
    }

    ~impl() {
        if (fd >= 0) {
            ::close(fd);
        }
    }
};

reader::reader(std::string path, backend_kind requested, size_t alignment) : pimpl(std::make_shared<impl>(std::move(path), requested, alignment)) {}
reader::~reader() = default;
reader::reader(reader &&) noexcept = default;
reader & reader::operator=(reader &&) noexcept = default;

read_result reader::read(const extent & item) const {
    if ((pimpl->selected == backend_kind::direct_pread || pimpl->selected == backend_kind::io_uring) &&
            (item.offset % pimpl->alignment != 0 || item.length % pimpl->alignment != 0)) {
        throw std::invalid_argument("direct I/O extent must be aligned");
    }
    const auto begin = std::chrono::steady_clock::now();
    read_result result;
    result.item = item;
    result.backend = backend_name(pimpl->selected);
    result.bytes.resize(item.length);
    void * aligned = nullptr;
    std::byte * target = result.bytes.data();
    const bool direct = pimpl->selected == backend_kind::direct_pread || pimpl->selected == backend_kind::io_uring;
    if (direct) {
        if (::posix_memalign(&aligned, pimpl->alignment, item.length) != 0) {
            throw std::bad_alloc();
        }
        target = static_cast<std::byte *>(aligned);
    }
    size_t done = 0;
    while (done < result.bytes.size()) {
        const ssize_t count = ::pread(pimpl->fd, target + done, result.bytes.size() - done, item.offset + done);
        if (count < 0) {
            std::free(aligned);
            throw std::runtime_error("pread failed: " + std::string(std::strerror(errno)));
        }
        if (count == 0) {
            std::free(aligned);
            throw std::runtime_error("short read at end of tensor store");
        }
        done += static_cast<size_t>(count);
    }
    if (direct) {
        std::memcpy(result.bytes.data(), aligned, item.length);
        std::free(aligned);
    }
    result.elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
    return result;
}

std::future<read_result> reader::read_async(extent item) const {
    const auto state = pimpl;
    return std::async(std::launch::async, [state, item] {
        reader temporary("/dev/null", backend_kind::pread);
        temporary.pimpl = state;
        return temporary.read(item);
    });
}

const capability_report & reader::capabilities() const {
    return pimpl->report;
}

} // namespace supervram
