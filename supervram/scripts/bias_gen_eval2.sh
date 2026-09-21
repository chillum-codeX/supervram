#!/usr/bin/env bash
# Multi-domain version of bias_gen_eval.sh. Usage: bias_gen_eval2.sh NAME CORPUS "betas" N
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; MODELS="${SUPERVRAM_MODELS:-$HOME/models}"; V="$ROOT/build/svram-verify"
NAME="$1"; CORPUS="$2"; BETAS="$3"; N="$4"; OUT="$ROOT/results/overnight/bias/gen-$NAME"; mkdir -p "$OUT"; M="$MODELS/Qwen3-30B-A3B-Q8_0.gguf"
COMMON=(--model "$M" --storage resident --pinned-moe --zerocopy --cache-mib 14336 --warm "$ROOT/results/overnight/warm-profile-q8.txt" --n-ctx $((512+N+64)) --n-batch 512 --n-ubatch 512 --prompt-file "$CORPUS" --prompt-tokens 512 --n-predict "$N" --ignore-eos)
for b in $BETAS; do
  "$V" "${COMMON[@]}" --bias "$b" ${BIAS_MUL:+--bias-mul} --temp 0.8 --top-k 40 --top-p 0.95 --repeat-penalty 1.1 --seed 1 --json "$OUT/gen-$b.json" > /dev/null 2>&1
  python3 -c "import json; print(' '.join(map(str, json.load(open('$OUT/gen-$b.json'))['tokens'])))" > "$OUT/gen-$b.tokens"
  "$V" "${COMMON[@]}" --bias 0 --force-tokens "$OUT/gen-$b.tokens" --json "$OUT/score-$b.json" > /dev/null 2>&1
  python3 - "$OUT" "$b" "$NAME" <<'PY'
import json, sys
out, b, name = sys.argv[1], sys.argv[2], sys.argv[3]
g = json.load(open(f"{out}/gen-{b}.json")); sc = json.load(open(f"{out}/score-{b}.json"))
lp = sum(sc["chosen_logprobs"]) / len(sc["chosen_logprobs"]); t = g["tokens"]
rep = 100 * (1 - len({tuple(t[i:i+8]) for i in range(len(t) - 8)}) / (len(t) - 8))
d2 = len({tuple(t[i:i+2]) for i in range(len(t) - 1)}) / (len(t) - 1)
line = f"GEN {name} bias {b}: unbiased-model log-prob {lp:.4f} | repeated 8-grams {rep:.1f}% | distinct 2-grams {100*d2:.1f}%"
print(line); open(f"{out}/../gen-SUMMARY.txt", "a").write(line + "\n")
PY
done
