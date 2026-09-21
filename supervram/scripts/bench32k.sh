#!/usr/bin/env bash
# THE benchmark: Qwen3-30B-A3B Q8_0, 32,000-token prompt (docs prose + instruction), 4,096 sampled output tokens (temp 0.8, top-k 40,
# top-p 0.95, repeat penalty 1.1, seed 1: natural text, no greedy loops), context 36,352.
# Usage: bench32k.sh NAME [svram-verify args...]     env: BENCH_UBATCH (4096), BENCH_NPREDICT (4096), BENCH_FORCE (token file: teacher-forced output)
# Results: results/overnight/bench/NAME.json (+ .out), summary via scripts/summarize_bench.py
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; MODELS="${SUPERVRAM_MODELS:-$HOME/models}"
NAME="$1"; shift
CORPUS="${LONGCTX_CORPUS:-/tmp/claude-1000/dio/corpus-32k.txt}"; [ -f "$CORPUS" ] || python3 "$ROOT/scripts/make_corpus.py" "$ROOT/third_party/llama.cpp" "$CORPUS"
UB="${BENCH_UBATCH:-4096}"; OUT="$ROOT/results/overnight/bench"; mkdir -p "$OUT"
EXTRA=()
if [ -n "${BENCH_FORCE:-}" ]; then EXTRA+=(--force-tokens "$BENCH_FORCE"); else EXTRA+=(--temp 0.8 --top-k 40 --top-p 0.95 --repeat-penalty 1.1 --seed 1); fi
"$ROOT/build/svram-verify" --model "${BENCH_MODEL:-$MODELS/Qwen3-30B-A3B-Q8_0.gguf}" "$@" --n-ctx 36352 --n-batch "$UB" --n-ubatch "$UB" --threads 12 \
  --prompt-file "$CORPUS" --prompt-tokens 32000 --prompt-suffix $'\n\n---\nWrite a detailed, structured summary of the documentation above.\n' \
  --n-predict "${BENCH_NPREDICT:-4096}" --ignore-eos "${EXTRA[@]}" --json "$OUT/$NAME.json" > "$OUT/$NAME.out" 2> "$OUT/$NAME.err"
python3 "$ROOT/scripts/summarize_bench.py" "$OUT/$NAME.json" "$NAME"
