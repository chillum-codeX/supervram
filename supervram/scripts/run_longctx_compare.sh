#!/usr/bin/env bash
# 32,000-token prompt + 4,096-token output on the same Q8_0 weights, several ways to use VRAM + RAM (+ SSD):
#   stock  : standard llama.cpp static split, experts of the first N layers on the CPU (RAM), the rest on the GPU
#   tiered : GPU expert cache (VRAM) filled from RAM (or the SSD with --direct-io); big prompt batches use whole-expert offload
# Usage: run_longctx_compare.sh <name> <svram-verify storage args...>   e.g.  run_longctx_compare.sh tiered14g --storage cache --cache-mib 14336
# The prompt is the first 32,000 tokens of the docs prose (scripts/make_corpus.py) plus a short instruction.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; MODELS="${SUPERVRAM_MODELS:-$HOME/models}"
NAME="$1"; shift
CORPUS="${LONGCTX_CORPUS:-/tmp/corpus-32k.txt}"; [ -f "$CORPUS" ] || python3 "$ROOT/scripts/make_corpus.py" "$ROOT/third_party/llama.cpp" "$CORPUS"
UB="${LONGCTX_UBATCH:-4096}"; OUT="$ROOT/results/rtx3090/longctx"; mkdir -p "$OUT"
"$ROOT/build/svram-verify" --model "$MODELS/Qwen3-30B-A3B-Q8_0.gguf" "$@" --n-ctx 36352 --n-batch "$UB" --n-ubatch "$UB" --threads 12 \
  --prompt-file "$CORPUS" --prompt-tokens 32000 --prompt-suffix $'\n\n---\nWrite a detailed, structured summary of the documentation above.\n' \
  --n-predict "${LONGCTX_NPREDICT:-4096}" --ignore-eos --json "$OUT/$NAME-ctx32000-out4096.json" > "$OUT/$NAME.out" 2> "$OUT/$NAME.err"
python3 "$ROOT/scripts/summarize_longctx.py" "$OUT/$NAME-ctx32000-out4096.json"
