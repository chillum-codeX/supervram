#!/usr/bin/env bash
# Quality cost of cache-aware routing (--bias): teacher-force 2048 tokens of human-written docs prose after a 512-token context and
# compare the mean log-probability of the true text (and the top-1 agreement) between bias 0 and bias B. Needs the pinned-RAM tier.
# Usage: bias_quality.sh "0 0.005 0.01 0.02 0.05"
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; MODELS="${SUPERVRAM_MODELS:-$HOME/models}"; OUT="$ROOT/results/overnight/bias"; mkdir -p "$OUT"
CORPUS="${LONGCTX_CORPUS:-/tmp/claude-1000/dio/corpus-32k.txt}"
V="$ROOT/build/svram-verify"; M="${BIAS_MODEL:-$MODELS/Qwen3-30B-A3B-Q8_0.gguf}"
# tokenize once: the first 2560 corpus tokens; context = first 512, forced text = the next 2048
"$V" --model "$MODELS/Qwen3-30B-A3B-Q4_K_M.gguf" --storage resident --n-predict 1 --n-ctx 4096 --prompt-file "$CORPUS" --prompt-tokens 2560 --dump-prompt-tokens "$OUT/corpus-tokens.txt" --n-ubatch 512 --n-batch 512 > /dev/null 2>&1 || true
python3 - "$OUT" <<'PY'
import sys
t = open(sys.argv[1] + "/corpus-tokens.txt").read().split()
open(sys.argv[1] + "/forced-2048.txt", "w").write(" ".join(t[512:2560]))
open(sys.argv[1] + "/context-512.txt", "w").write(" ".join(t[:512]))
print("forced", len(t[512:2560]), "tokens after a", len(t[:512]), "token context")
PY
for b in $1; do
  "$V" --model "$M" --storage resident --pinned-moe --zerocopy --cache-mib "${BIAS_CACHE_MIB:-8192}" --warm "$ROOT/results/overnight/warm-profile-q8.txt" --bias "$b" \
     --n-ctx 4096 --n-batch 512 --n-ubatch 512 --prompt-file "$CORPUS" --prompt-tokens 512 --force-tokens "$OUT/forced-2048.txt" --n-predict 2048 --json "$OUT/bias-$b.json" > "$OUT/bias-$b.out" 2> "$OUT/bias-$b.err"
  python3 "$ROOT/scripts/bias_quality_summary.py" "$OUT" "$b"
done
