#pragma once

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace supervram {

enum class backend_kind {
    automatic,
    cufile,
    io_uring,
    direct_pread,
    pread,
};

struct extent {
    int32_t layer;
    int32_t expert;
    uint64_t offset;
    uint64_t length;
};

struct capability_report {
    bool cuda_visible = false;
    bool cufile_library = false;
    bool io_uring_headers = false;
    bool direct_io_open = false;
    std::string selected_backend;
    std::vector<std::string> rejection_reasons;
};

struct read_result {
    extent item;
    std::vector<std::byte> bytes;
    uint64_t elapsed_ns = 0;
    std::string backend;
};

class reader {
public:
    reader(std::string path, backend_kind requested, size_t alignment = 4096);
    ~reader();
    reader(const reader &) = delete;
    reader & operator=(const reader &) = delete;
    reader(reader &&) noexcept;
    reader & operator=(reader &&) noexcept;

    read_result read(const extent & item) const;
    std::future<read_result> read_async(extent item) const;
    const capability_report & capabilities() const;

private:
    struct impl;
    std::shared_ptr<impl> pimpl;
};

capability_report probe(const std::string & path, backend_kind requested = backend_kind::automatic);
backend_kind parse_backend(const std::string & value);
const char * backend_name(backend_kind value);

} // namespace supervram
