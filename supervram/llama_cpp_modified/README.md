# Modified llama.cpp files (readable copy)

The pinned llama.cpp checkout lives in `../third_party/llama.cpp/` (its own git repo, ignored by this repository).
This folder is a plain copy of the 26 files that patches `0001` to `0009` change or add, with their paths
preserved, so the complete C++ changes are visible and tracked here.

- Base: llama.cpp `ce8caa6e60a03093351d6016a818720e0d46f0fb`.
- The source of truth for applying the changes is `../patches/000{1..9}-*.patch` (apply in order to the pinned commit).
- Refresh this copy after editing the checkout: `../scripts/sync_llama_cpp_modified.sh`.
- Main new code: `ggml/src/ggml-backend.cpp` (expert cache, direct I/O reads, routing trace),
  `src/llama-model-loader.cpp` (file-source registration), `tests/test-expert-cache.cpp` (unit test).
