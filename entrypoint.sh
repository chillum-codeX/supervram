#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/supervram" && pwd)"
LLAMA="${SUPERVRAM_LLAMA_DIR:-$ROOT/third_party/llama.cpp}"
RESULTS="${SUPERVRAM_RESULTS:-$ROOT/results/reproduced}"
PYTHON="${SUPERVRAM_PYTHON:-python3}"
CXX="${CXX:-c++}"
CC="${CC:-cc}"
JOBS="${SUPERVRAM_JOBS:-$(nproc)}"
CUDA="${SUPERVRAM_CUDA:-0}"
GENERATOR=()
if command -v ninja > /dev/null 2>&1; then
    GENERATOR=(-G Ninja)
fi
export PYTHONPATH="$ROOT"
mkdir -p "$RESULTS"

if [[ ! -d "$LLAMA/.git" ]]; then
  "$ROOT/scripts/setup_llama_cpp.sh"
fi

cmake -S "$ROOT" -B "$ROOT/build" "${GENERATOR[@]}" -DCMAKE_CXX_COMPILER="$CXX"
cmake --build "$ROOT/build" -j "$JOBS"
ctest --test-dir "$ROOT/build" --output-on-failure
"$PYTHON" -m pytest "$ROOT/tests" -v

"$PYTHON" "$ROOT/scripts/probe_hardware.py" \
  --path "$ROOT" --output "$RESULTS/hardware-probe.json" > "$RESULTS/hardware-probe.stdout.json"
"$PYTHON" "$ROOT/scripts/simulate_trace.py" \
  --tokens 16 --layers 4 --experts 16 --top-k 2 --expert-bytes 4096 \
  --cache-experts 8 --policy router-aware --predictor markov --prefetch-depth 2 \
  --output "$RESULTS/simulation-smoke.json" --trace "$RESULTS/simulation-trace.jsonl" \
  > "$RESULTS/simulation-smoke.stdout.json"
"$PYTHON" "$ROOT/scripts/analyze_trace.py" "$RESULTS/simulation-trace.jsonl" \
  --json "$RESULTS/simulation-trace-summary.json" --csv "$RESULTS/simulation-expert-events.csv"
"$PYTHON" "$ROOT/scripts/roofline.py" --output "$RESULTS/roofline-projection.json" \
  > "$RESULTS/roofline-projection.stdout.json"
"$PYTHON" "$ROOT/scripts/run_ablations.py" --mode plan --output-dir "$RESULTS/ablation-plan"

# CPU build of the patched tree: used for the resident/mmap logits equality gate on a tiny model
BUILD_CPU="$LLAMA/build-supervram"
cmake -S "$LLAMA" -B "$BUILD_CPU" "${GENERATOR[@]}" \
  -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
cmake --build "$BUILD_CPU" --target test-arg-parser test-llama-archs llama-bench -j "$JOBS"
"$BUILD_CPU/bin/test-arg-parser" > "$RESULTS/test-arg-parser.log" 2>&1
mkdir -p "$RESULTS/arch-test"
"$BUILD_CPU/bin/test-llama-archs" --arch qwen3moe --seed 20260921 --out "$RESULTS/arch-test" > "$RESULTS/test-llama-archs.log" 2>&1

"$CXX" -std=c++17 "$ROOT/src/llama_mmap_smoke.cpp" \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" -L"$BUILD_CPU/bin" \
  -Wl,-rpath,"$BUILD_CPU/bin" -lllama -lggml -lggml-base -lggml-cpu \
  -o "$ROOT/build/llama_mmap_smoke"
MODEL="$RESULTS/arch-test/qwen3moe-moe.gguf"
"$ROOT/build/llama_mmap_smoke" "$MODEL" resident > "$RESULTS/llama-resident-smoke.log" 2>&1
"$ROOT/build/llama_mmap_smoke" "$MODEL" mmap > "$RESULTS/llama-mmap-smoke.log" 2>&1
MODEL="$MODEL" RESULTS="$RESULTS" "$PYTHON" - <<'PY'
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

PYTHONPATH="$ROOT:$LLAMA/gguf-py" "$PYTHON" "$ROOT/scripts/pack_gguf_experts.py" \
  "$MODEL" "$RESULTS/qwen3moe-experts.svram" > "$RESULTS/pack-smoke.log"
PYTHONPATH="$ROOT" STORE="$RESULTS/qwen3moe-experts.svram" "$PYTHON" - <<'PY'
import os
from supervram import TensorStore
with TensorStore(os.environ["STORE"]) as store:
    for key in store.extents:
        store.read(key, verify=True)
    print(f"verified {len(store.extents)} packed expert extents")
PY

if [[ "$CUDA" == "1" ]]; then
  BUILD_CUDA="$LLAMA/build-3090"
  cmake -S "$LLAMA" -B "$BUILD_CUDA" "${GENERATOR[@]}" \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${SUPERVRAM_CUDA_ARCH:-86}" \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
  cmake --build "$BUILD_CUDA" --target llama-server llama-bench llama-cli llama-perplexity \
    test-arg-parser test-llama-archs test-backend-ops -j "$JOBS"
  printf 'CUDA build ready at %s\n' "$BUILD_CUDA"
fi

printf '\nSuperVRAM reproducibility checks complete.\n'
printf 'Results: %s\n' "$RESULTS"
