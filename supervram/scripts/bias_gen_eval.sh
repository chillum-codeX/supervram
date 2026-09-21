#!/usr/bin/env bash
# Does cache-aware routing degrade GENERATED text? For each bias: generate N sampled tokens (temp 0.8, seed 1) from a fixed 512-token
# prompt, then score every generated sequence with the UNBIASED model (teacher forced): mean log-probability per token (lower = the
# unbiased model finds the text less natural), plus repeated 8-grams. The unbiased model's own samples are the baseline.
# Usage: bias_gen_eval.sh "0 0.02 0.05" [N]
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; MODELS="${SUPERVRAM_MODELS:-$HOME/models}"; V="$ROOT/build/svram-verify"; OUT="$ROOT/results/overnight/bias/gen"; mkdir -p "$OUT"
M="$MODELS/Qwen3-30B-A3B-Q8_0.gguf"; CORPUS="${LONGCTX_CORPUS:-/tmp/claude-1000/dio/corpus-32k.txt}"; N="${2:-1024}"
COMMON=(--model "$M" --storage resident --pinned-moe --zerocopy --cache-mib 14336 --warm "$ROOT/results/overnight/warm-profile-q8.txt" --n-ctx $((512+N+64)) --n-batch 512 --n-ubatch 512 --prompt-file "$CORPUS" --prompt-tokens 512 --n-predict "$N" --ignore-eos)
for b in $1; do
  "$V" "${COMMON[@]}" --bias "$b" ${BIAS_MUL:+--bias-mul} --temp 0.8 --top-k 40 --top-p 0.95 --repeat-penalty 1.1 --seed 1 --json "$OUT/gen-$b.json" > /dev/null 2>&1
  python3 -c "import json,sys; print(' '.join(map(str, json.load(open('$OUT/gen-$b.json'))['tokens'])))" > "$OUT/gen-$b.tokens"
  "$V" "${COMMON[@]}" --bias 0 --force-tokens "$OUT/gen-$b.tokens" --json "$OUT/score-$b.json" > /dev/null 2>&1
  python3 - "$OUT" "$b" <<'PY'
import json, sys
out, b = sys.argv[1], sys.argv[2]
g = json.load(open(f"{out}/gen-{b}.json")); sc = json.load(open(f"{out}/score-{b}.json"))
lp = sum(sc["chosen_logprobs"]) / len(sc["chosen_logprobs"]); t = g["tokens"]
rep = 100 * (1 - len({tuple(t[i:i+8]) for i in range(len(t) - 8)}) / (len(t) - 8))
d2 = len({tuple(t[i:i+2]) for i in range(len(t) - 1)}) / (len(t) - 1)
line = f"generated with bias {b}: mean log-prob of the text under the UNBIASED model {lp:.4f} | repeated 8-grams {rep:.1f}% | distinct 2-grams {100*d2:.1f}%"
print(line); open(f"{out}/SUMMARY.txt", "a").write(line + "\n")
PY
done
