#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/novix/workspace/project/supervram
LLAMA="$ROOT/third_party/llama.cpp"
RESULTS="$ROOT/results/reproduced"
ENV=/home/novix/micromamba
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$ROOT"
mkdir -p "$RESULTS"

if [[ ! -d "$LLAMA/.git" ]]; then
  "$ROOT/scripts/setup_llama_cpp.sh"
fi

cmake -S "$ROOT" -B "$ROOT/build" -G Ninja \
  -DCMAKE_CXX_COMPILER="$ENV/bin/x86_64-conda-linux-gnu-c++"
cmake --build "$ROOT/build" -j 8
ctest --test-dir "$ROOT/build" --output-on-failure
"$ENV/bin/python" -m pytest "$ROOT/tests" -v

"$ENV/bin/python" "$ROOT/scripts/probe_hardware.py" \
  --path "$ROOT" --output "$RESULTS/hardware-probe.json" > "$RESULTS/hardware-probe.stdout.json"
"$ENV/bin/python" "$ROOT/scripts/simulate_trace.py" \
  --tokens 16 --layers 4 --experts 16 --top-k 2 --expert-bytes 4096 \
  --cache-experts 8 --policy router-aware --predictor markov --prefetch-depth 2 \
  --output "$RESULTS/simulation-smoke.json" --trace "$RESULTS/simulation-trace.jsonl" \
  > "$RESULTS/simulation-smoke.stdout.json"
"$ENV/bin/python" "$ROOT/scripts/analyze_trace.py" "$RESULTS/simulation-trace.jsonl" \
  --json "$RESULTS/simulation-trace-summary.json" --csv "$RESULTS/simulation-expert-events.csv"
"$ENV/bin/python" "$ROOT/scripts/roofline.py" --output "$RESULTS/roofline-projection.json" \
  > "$RESULTS/roofline-projection.stdout.json"
"$ENV/bin/python" "$ROOT/scripts/run_ablations.py" --mode plan --output-dir "$RESULTS/ablation-plan"

cmake -S "$LLAMA" -B "$LLAMA/build-supervram" -G Ninja \
  -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DCMAKE_C_COMPILER="$ENV/bin/x86_64-conda-linux-gnu-cc" \
  -DCMAKE_CXX_COMPILER="$ENV/bin/x86_64-conda-linux-gnu-c++"
cmake --build "$LLAMA/build-supervram" --target test-arg-parser test-llama-archs llama-bench -j 8
"$LLAMA/build-supervram/bin/test-arg-parser" > "$RESULTS/test-arg-parser.log" 2>&1
mkdir -p "$RESULTS/arch-test"
"$LLAMA/build-supervram/bin/test-llama-archs" --arch qwen3moe --seed 20260921 --out "$RESULTS/arch-test" > "$RESULTS/test-llama-archs.log" 2>&1

"$ENV/bin/x86_64-conda-linux-gnu-c++" -std=c++17 "$ROOT/src/llama_mmap_smoke.cpp" \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" -L"$LLAMA/build-supervram/bin" \
  -Wl,-rpath,"$LLAMA/build-supervram/bin" -lllama -lggml -lggml-base -lggml-cpu \
  -o "$ROOT/build/llama_mmap_smoke"
MODEL="$RESULTS/arch-test/qwen3moe-moe.gguf"
"$ROOT/build/llama_mmap_smoke" "$MODEL" resident > "$RESULTS/llama-resident-smoke.log" 2>&1
"$ROOT/build/llama_mmap_smoke" "$MODEL" mmap > "$RESULTS/llama-mmap-smoke.log" 2>&1
MODEL="$MODEL" RESULTS="$RESULTS" "$ENV/bin/python" - <<'PY'
import os, re
from pathlib import Path
results = Path(os.environ["RESULTS"])
resident_text = (results / "llama-resident-smoke.log").read_text()
mmap_text = (results / "llama-mmap-smoke.log").read_text()
checksum_pattern = r"logits_checksum=([^\s]+)"
hash_pattern = r"logits_hash=([^\s]+)"
resident = float(re.search(checksum_pattern, resident_text).group(1))
mmap = float(re.search(checksum_pattern, mmap_text).group(1))
resident_hash = re.search(hash_pattern, resident_text).group(1)
mmap_hash = re.search(hash_pattern, mmap_text).group(1)
assert resident_hash == mmap_hash, (resident_hash, mmap_hash)
print(f"resident/mmap exact logits hash match: {resident_hash}; checksums {resident} / {mmap}")
PY

PYTHONPATH="$ROOT:$LLAMA/gguf-py" "$ENV/bin/python" "$ROOT/scripts/pack_gguf_experts.py" \
  "$MODEL" "$RESULTS/qwen3moe-experts.svram" > "$RESULTS/pack-smoke.log"
PYTHONPATH="$ROOT" STORE="$RESULTS/qwen3moe-experts.svram" "$ENV/bin/python" - <<'PY'
import os
from supervram import TensorStore
with TensorStore(os.environ["STORE"]) as store:
    for key in store.extents:
        store.read(key, verify=True)
    print(f"verified {len(store.extents)} packed expert extents")
PY

printf '\nSuperVRAM reproducibility checks complete.\n'
printf 'Results: %s\n' "$RESULTS"
