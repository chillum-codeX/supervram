# Reproducing the results

Everything lives under `/home/truppy/Downloads/workspace/project/supervram/`. Models are expected in `~/models`
(override with `SUPERVRAM_MODELS=/path`): `Qwen3-30B-A3B-Q4_K_M.gguf`, `Qwen3-30B-A3B-Q8_0.gguf`.

## Code map

| What | Where |
|---|---|
| Patches to llama.cpp (apply 0001, 0002, 0003 in order to `ce8caa6`) | `patches/` |
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

## Runtime switches (environment variables)

| Variable | Meaning |
|---|---|
| `SVRAM_DIRECT_IO=1` | read expert cache misses with O_DIRECT into pinned staging (Linux); default is the mmap path |
| `SVRAM_IO_THREADS` (8), `SVRAM_STAGING_MIB` (128) | direct-I/O parallelism and staging size |
| `SVRAM_TRACE=<file>` | record the distinct experts each layer selects per token |

## Experiments

| Result | Command |
|---|---|
| Cold start under a RAM cap (evict, cgroup cap, SSD bytes, peak RAM) | `SVRAM_DIRECT_IO=1 python3 scripts/cold_run.py --model ~/models/Qwen3-30B-A3B-Q8_0.gguf --svram-verify build/svram-verify --ram-cap 4G --name NAME --out-dir results/rtx3090/cold -- --storage cache --cache-mib 16384 --n-ctx 2048 --n-predict 1536 --n-ubatch 1 --prompt "Compare the advantages and disadvantages of microservice architecture versus a monolith for a startup with five engineers."` |
| SSD ceiling for expert-sized reads | `python3 scripts/ssd_expert_read_bench.py ~/models/Qwen3-30B-A3B-Q8_0.gguf` (accepts block devices and several targets) |
| Cache size x policy sweep, 3 reps | `scripts/run_ablation_sweep.sh`, then `python3 scripts/aggregate_ablations.py results/rtx3090/ablations-long` |
| Routing traces + Q4-vs-Q8 teacher-forced quality | `scripts/run_traces_and_quality.sh` (prompts in `scripts/prompts.txt`), then `analyze_policies.py` and `compare_quality.py` |
| Prefetch upper bound and predictor accuracy (simulation on the traces) | `python3 scripts/simulate_prefetch.py results/rtx3090/traces/q8-p*.trace [--r 1.65]` and `python3 scripts/prefetch_predictor_eval.py results/rtx3090/traces/q8-p*.trace` |
| Ground-truth perplexity, Q4_K_M vs Q8_0 | `scripts/run_ground_truth_ppl.sh` |
| 43 GiB synthetic test model | `scripts/make_synthetic_model.sh`, then `cold_run.py` with `--model ~/models/Qwen3-30B-A3B-synthetic-F16x24-Q8x24.gguf` |

Correctness checks: `third_party/llama.cpp/build-3090/bin/test-expert-cache`, `python3 -m pytest -q tests`, and comparing
`svram-verify --storage resident` against `--storage cache` (tokens and logits hashes must match for Q4_K_M).
