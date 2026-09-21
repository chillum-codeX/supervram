#!/usr/bin/env bash
# Repeated cache-size x policy sweep (3 reps x 256-token decode) plus same-settings baselines -> results/rtx3090/ablations-long
# Then: python3 scripts/aggregate_ablations.py results/rtx3090/ablations-long
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; MODELS="${SUPERVRAM_MODELS:-$HOME/models}"
OUT="$ROOT/results/rtx3090/ablations-long"; V="$ROOT/build/svram-verify"; N=256
declare -A MODEL=( [q4]="$MODELS/Qwen3-30B-A3B-Q4_K_M.gguf" [q8]="$MODELS/Qwen3-30B-A3B-Q8_0.gguf" )
mkdir -p "$OUT"
for rep in 1 2 3; do
  for m in q4 q8; do
    d="$OUT/$m/rep$rep"; mkdir -p "$d/baselines"
    python3 "$ROOT/scripts/run_ablations.py" --mode target --output-dir "$d" --model "${MODEL[$m]}" --svram-verify "$V" --n-predict $N --n-ubatch 1
    if [ $m = q4 ]; then
      "$V" --model "${MODEL[$m]}" --storage resident --n-predict $N --json "$d/baselines/resident.json" > "$d/baselines/resident.out" 2> "$d/baselines/resident.err"
      "$V" --model "${MODEL[$m]}" --storage mmap     --n-predict $N --json "$d/baselines/mmap.json"     > "$d/baselines/mmap.out"     2> "$d/baselines/mmap.err"
    else
      "$V" --model "${MODEL[$m]}" --storage resident --cpu-moe --n-predict $N --json "$d/baselines/cpu-moe.json" > "$d/baselines/cpu-moe.out" 2> "$d/baselines/cpu-moe.err"
      "$V" --model "${MODEL[$m]}" --storage mmap     --n-predict $N --json "$d/baselines/mmap.json"    > "$d/baselines/mmap.out"    2> "$d/baselines/mmap.err"
    fi
  done
done
echo DONE > "$OUT/DONE"
