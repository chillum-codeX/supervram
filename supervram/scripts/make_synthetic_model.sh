#!/usr/bin/env bash
# Build the synthetic 42.9 GiB test model: experts of layers 0-23 as F16 (dequantized from Q8_0), everything else Q8_0.
# Throughput/capacity testing only - NOT a real F16 model and no better than Q8 in quality. Writes ~46 GB.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; MODELS="${SUPERVRAM_MODELS:-$HOME/models}"
TT="$(mktemp)"; printf '%s\n' 'blk\.([0-9]|1[0-9]|2[0-3])\.ffn_(gate|up|down)_exps\.weight=f16' > "$TT"
"$ROOT/third_party/llama.cpp/build-3090/bin/llama-quantize" --allow-requantize --tensor-type-file "$TT" \
  "$MODELS/Qwen3-30B-A3B-Q8_0.gguf" "$MODELS/Qwen3-30B-A3B-synthetic-F16x24-Q8x24.gguf" Q8_0
