#!/usr/bin/env bash
# Quality of cache-aware routing on human-written text: the model reads CTX tokens of CORPUS, then the next NFORCE true tokens are
# teacher-forced; we compare log-probabilities and top-1 agreement between bias 0 and each bias. Pinned-RAM tier, 14 GiB cache.
# Usage: bias_eval.sh NAME CORPUS_FILE CTX NFORCE "0 0.01 0.02"
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; MODELS="${SUPERVRAM_MODELS:-$HOME/models}"; V="$ROOT/build/svram-verify"
NAME="$1"; CORPUS="$2"; CTX="$3"; NF="$4"; BETAS="$5"; OUT="$ROOT/results/overnight/bias/$NAME"; mkdir -p "$OUT"
M="${BIAS_MODEL:-$MODELS/Qwen3-30B-A3B-Q8_0.gguf}"
"$V" --model "$MODELS/Qwen3-30B-A3B-Q4_K_M.gguf" --storage resident --n-ctx 64 --prompt-file "$CORPUS" --prompt-tokens $((CTX+NF)) --dump-prompt-tokens "$OUT/tokens.txt" --tokenize-only > /dev/null 2>&1
python3 - "$OUT" "$CTX" "$NF" <<'PY'
import sys
out, ctx, nf = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); t = open(out + "/tokens.txt").read().split()
open(out + "/forced.txt", "w").write(" ".join(t[ctx:ctx + nf])); print(len(t), "tokens; context", ctx, "forced", len(t[ctx:ctx + nf]))
PY
UB=512; [ "$CTX" -ge 4096 ] && UB=4096
for b in $BETAS; do
  "$V" --model "$M" --storage resident --pinned-moe --zerocopy --cache-mib 14336 --warm "$ROOT/results/overnight/warm-profile-q8.txt" --bias "$b" ${BIAS_MUL:+--bias-mul} \
     --n-ctx $((CTX+NF+64)) --n-batch $UB --n-ubatch $UB --prompt-file "$CORPUS" --prompt-tokens "$CTX" --force-tokens "$OUT/forced.txt" --n-predict "$NF" \
     --json "$OUT/bias-$b.json" > "$OUT/bias-$b.out" 2> "$OUT/bias-$b.err"
  python3 "$ROOT/scripts/bias_quality_summary.py" "$OUT" "$b"
done
