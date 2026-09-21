#!/usr/bin/env bash
# Ground-truth perplexity of Q4_K_M vs Q8_0 on human-written prose (60 chunks x 512 tokens). Q8_0 keeps its experts on the CPU
# for this scoring run (it does not fit in VRAM). Logs go to results/rtx3090/quality/ppl-*.log.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; MODELS="${SUPERVRAM_MODELS:-$HOME/models}"; BIN="$ROOT/third_party/llama.cpp/build-3090/bin"
CORPUS="$(mktemp --suffix=.txt)"; python3 "$ROOT/scripts/make_corpus.py" "$ROOT/third_party/llama.cpp" "$CORPUS"
"$BIN/llama-perplexity" -m "$MODELS/Qwen3-30B-A3B-Q4_K_M.gguf" -f "$CORPUS" -c 512 --chunks 60 -ngl 99 -t 12 > "$ROOT/results/rtx3090/quality/ppl-q4_k_m.log" 2>&1
"$BIN/llama-perplexity" -m "$MODELS/Qwen3-30B-A3B-Q8_0.gguf"   -f "$CORPUS" -c 512 --chunks 60 -ngl 99 --cpu-moe -t 12 > "$ROOT/results/rtx3090/quality/ppl-q8_0.log" 2>&1
echo "PPL: $(grep 'Final estimate' "$ROOT/results/rtx3090/quality/ppl-q4_k_m.log") | $(grep 'Final estimate' "$ROOT/results/rtx3090/quality/ppl-q8_0.log")"
