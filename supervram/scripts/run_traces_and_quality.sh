#!/usr/bin/env bash
# For each prompt in scripts/prompts.txt: Q8_0 generates 384 tokens through the direct-I/O cache while recording the routing trace
# (results/rtx3090/traces/q8-p<i>.trace) and its tokens (results/rtx3090/quality/q8-p<i>.json); Q4_K_M then scores those exact tokens
# (teacher forcing). Afterwards:
#   python3 scripts/analyze_policies.py results/rtx3090/traces/q8-p*.trace --json results/rtx3090/traces/policy-study.json
#   python3 scripts/compare_quality.py results/rtx3090/quality --json results/rtx3090/quality/summary-q4-vs-q8.json
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; MODELS="${SUPERVRAM_MODELS:-$HOME/models}"; V="$ROOT/build/svram-verify"
Q8="$MODELS/Qwen3-30B-A3B-Q8_0.gguf"; Q4="$MODELS/Qwen3-30B-A3B-Q4_K_M.gguf"; TMP="$(mktemp -d)"
mkdir -p "$ROOT/results/rtx3090/traces" "$ROOT/results/rtx3090/quality"
i=0
while IFS= read -r line; do
  i=$((i+1))
  SVRAM_DIRECT_IO=1 SVRAM_TRACE="$ROOT/results/rtx3090/traces/q8-p$i.trace" "$V" --model "$Q8" --storage cache --cache-mib 16384 \
    --n-predict 384 --n-ubatch 1 --prompt "$line" --json "$ROOT/results/rtx3090/quality/q8-p$i.json" > /dev/null 2>&1
  python3 -c "import json,sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['tokens'])))" "$ROOT/results/rtx3090/quality/q8-p$i.json" > "$TMP/p$i.tokens"
  "$V" --model "$Q4" --storage resident --n-predict 384 --n-ubatch 1 --prompt "$line" --force-tokens "$TMP/p$i.tokens" \
    --json "$ROOT/results/rtx3090/quality/q4-on-q8-p$i.json" > /dev/null 2>&1
  echo "prompt $i done"
done < "$ROOT/scripts/prompts.txt"
