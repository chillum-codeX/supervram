#include "supervram_native.h"

#include <cassert>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>

int main() {
    const auto path = std::filesystem::temp_directory_path() / "supervram-native-test.bin";
    {
        std::ofstream out(path, std::ios::binary);
        std::string block(8192, 'x');
        out.write(block.data(), block.size());
    }

    supervram::reader input(path.string(), supervram::backend_kind::pread);
    const supervram::extent first{0, 0, 0, 4096};
    auto result = input.read(first);
    assert(result.bytes.size() == 4096);
    assert(result.backend == "pread");
    assert(static_cast<char>(result.bytes[0]) == 'x');

    const supervram::extent second{0, 1, 4096, 4096};
    auto future = input.read_async(second);
    auto async_result = future.get();
    assert(async_result.bytes.size() == 4096);
    assert(async_result.item.expert == 1);

    const auto report = supervram::probe(path.string());
    assert(!report.selected_backend.empty());

    auto lifetime_future = [&path]() {
        supervram::reader temporary(path.string(), supervram::backend_kind::pread);
        return temporary.read_async({0, 2, 0, 4096});
    }();
    assert(lifetime_future.get().bytes.size() == 4096);

    bool invalid_alignment = false;
    try {
        supervram::reader invalid(path.string(), supervram::backend_kind::pread, 0);
    } catch (const std::invalid_argument &) {
        invalid_alignment = true;
    }
    assert(invalid_alignment);

    std::filesystem::remove(path);
    return 0;
}
