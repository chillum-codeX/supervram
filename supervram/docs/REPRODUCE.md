# Reproducing the results

Everything lives under `/home/truppy/Downloads/workspace/project/supervram/`. Models are expected in `~/models`
(override with `SUPERVRAM_MODELS=/path`): `Qwen3-30B-A3B-Q4_K_M.gguf`, `Qwen3-30B-A3B-Q8_0.gguf`.

## Code map

| What | Where |
|---|---|
| Patches to llama.cpp (apply 0001 to 0007 in order to `ce8caa6`) | `patches/` |
| The same 23 modified/added llama.cpp files, readable and tracked | `llama_cpp_modified/` (refresh: `scripts/sync_llama_cpp_modified.sh`) |
| The working llama.cpp checkout and CUDA build | `third_party/llama.cpp/`, `third_party/llama.cpp/build-3090/` |
| Test/measurement tool (`svram-verify`) | `src/svram_verify.cpp`, built to `build/svram-verify` |
| Scripts | `scripts/` |
| Results | `results/rtx3090/` |

## Build

```bash
cd project && SUPERVRAM_CUDA=1 ./entrypoint.sh          # CPU checks + CUDA build of the patched llama.cpp
cd supervram/build && ninja svram-verify
```

## Runtime options

Command-line flags (any llama.cpp tool that takes the usual model options, plus `svram-verify` with the short names in brackets):

| Flag | Meaning |
|---|---|
| `--moe-expert-storage cache` | GPU expert cache; also `--moe-expert-cache-size MiB`, `--moe-expert-cache-policy lru\|lfu` |
| `--moe-expert-direct-io` [`--direct-io`] | read cache misses with O_DIRECT into pinned staging (Linux); default is the mmap path |
| `--moe-expert-io-threads N` [`--io-threads`] (8), `--moe-expert-staging-mib N` [`--staging-mib`] (128) | direct-I/O parallelism and staging size |
| `--moe-expert-trace FILE` [`--trace`] | record the distinct experts each layer selects per token |

The environment variables `SVRAM_DIRECT_IO=1`, `SVRAM_IO_THREADS`, `SVRAM_STAGING_MIB`, `SVRAM_TRACE=<file>` still work as a fallback (the scripts use them); an explicit flag wins.

## Experiments

| Result | Command |
|---|---|
| Cold start under a RAM cap (evict, cgroup cap, SSD bytes, peak RAM) | `SVRAM_DIRECT_IO=1 python3 scripts/cold_run.py --model ~/models/Qwen3-30B-A3B-Q8_0.gguf --svram-verify build/svram-verify --ram-cap 4G --name NAME --out-dir results/rtx3090/cold -- --storage cache --cache-mib 16384 --n-ctx 2048 --n-predict 1536 --n-ubatch 1 --prompt "Compare the advantages and disadvantages of microservice architecture versus a monolith for a startup with five engineers."` |
| SSD ceiling for expert-sized reads | `python3 scripts/ssd_expert_read_bench.py ~/models/Qwen3-30B-A3B-Q8_0.gguf` (accepts block devices and several targets) |
| Cache size x policy sweep, 3 reps | `scripts/run_ablation_sweep.sh`, then `python3 scripts/aggregate_ablations.py results/rtx3090/ablations-long` |
| Routing traces + Q4-vs-Q8 teacher-forced quality | `scripts/run_traces_and_quality.sh` (prompts in `scripts/prompts.txt`), then `analyze_policies.py` and `compare_quality.py` |
| 32,000-token prompt + 4,096-token output, three ways to use VRAM/RAM/SSD | `scripts/run_longctx_compare.sh NAME --storage resident --n-cpu-moe 25` (stock), `... --storage cache --cache-mib 14336` (tiered, RAM), and `cold_run.py --ram-cap 4G ... -- --storage cache --cache-mib 14336 --direct-io --n-ubatch 4096 --prompt-file ... --prompt-tokens 32000` (RAM-poor); summarize with `scripts/summarize_longctx.py` |
| Prefetch upper bound and predictor accuracy (simulation on the traces) | `python3 scripts/simulate_prefetch.py results/rtx3090/traces/q8-p*.trace [--r 1.65]` and `python3 scripts/prefetch_predictor_eval.py results/rtx3090/traces/q8-p*.trace` |
| Zero-copy expert tier (stage 1): exactness and all-from-RAM speed | `build/svram-verify --model ~/models/Qwen3-30B-A3B-Q4_K_M.gguf --storage resident --pinned-moe --zerocopy --cache-mib 512 --n-ubatch 1 --n-predict 64` vs `--storage resident` (tokens and logits hashes must match); CUDA micro-benchmarks in `scripts/microbench/` |
| Ground-truth perplexity, Q4_K_M vs Q8_0 | `scripts/run_ground_truth_ppl.sh` |
| 43 GiB synthetic test model | `scripts/make_synthetic_model.sh`, then `cold_run.py` with `--model ~/models/Qwen3-30B-A3B-synthetic-F16x24-Q8x24.gguf` |

Correctness checks: `third_party/llama.cpp/build-3090/bin/test-expert-cache`, `python3 -m pytest -q tests`, and comparing
`svram-verify --storage resident` against `--storage cache` (tokens and logits hashes must match for Q4_K_M).
