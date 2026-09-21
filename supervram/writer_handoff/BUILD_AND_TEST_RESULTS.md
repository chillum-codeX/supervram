# Build and test results

Date: 2026-09-21.

## Toolchain

- llama.cpp revision: `ce8caa6e60a03093351d6016a818720e0d46f0fb`.
- Conda-forge GCC/G++ 15.3, CMake 4.4.3, Ninja 1.13.2.
- Python 3.14.7, pytest 9.1.1.
- CUDA unavailable.

## Passing checks

```text
SuperVRAM native CTest: 1/1 passed
SuperVRAM Python pytest: 8/8 passed
llama.cpp test-arg-parser: all tests OK
llama.cpp test-llama-archs --arch qwen3moe: generated tiny GGUF successfully
resident-vs-mmap one-token logits checksum: exactly equal
GGUF expert pack/read/checksum smoke test: passed
```

## Build status

- Patched `libllama`, `llama-common`, `test-arg-parser`, `test-llama-archs`, and `llama-bench` compiled successfully.
- `llama-server` did not finish because its UI asset step downloaded a `latest` bundle whose manifest named a missing CSS file. This is external asset packaging, not a compiler failure in the patch. Core server object compilation had progressed before the asset target failed.
- CUDA, cuFile and io_uring device paths were not built in this environment.

## Logs

- `llama-tests-build.log`
- `test-arg-parser.log`
- `test-llama-archs.log`
- `llama-resident-smoke.log`
- `llama-mmap-smoke.log`
- `llama-build.log`
- `results/hardware-probe.json`
- `results/simulation-smoke.json` (deterministic replay mode; wall-clock asynchronous replay is separately opt-in)
- `results/roofline-projection.json`
