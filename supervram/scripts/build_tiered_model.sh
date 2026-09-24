#!/usr/bin/env bash
# Build a popularity-tiered mixed-precision GGUF: the tensor-type-file from
# popularity_to_layers.py drives which layers get downgraded to Q4_K_M; everything else
# (including all non-expert weights) stays at the target type (Q8_0).
# Usage: build_tiered_model.sh <source.gguf> <output.gguf> <tensor-type-file>
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$1"; DST="$2"; TT="$3"
echo "Building tiered model:"
echo "  source:  $SRC"
echo "  output:  $DST"
echo "  tensor-type-file:"
cat "$TT"
echo ""
"$ROOT/third_party/llama.cpp/build-3090/bin/llama-quantize" \
  --allow-requantize \
  --tensor-type-file "$TT" \
  "$SRC" "$DST" Q8_0
echo "Done: $DST"
