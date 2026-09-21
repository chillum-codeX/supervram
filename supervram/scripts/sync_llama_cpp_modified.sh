#!/usr/bin/env bash
# Refresh llama_cpp_modified/ from the pinned llama.cpp checkout (files listed in llama_cpp_modified/FILES.txt).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
while read -r f; do
  mkdir -p "$ROOT/llama_cpp_modified/$(dirname "$f")"
  cp "$ROOT/third_party/llama.cpp/$f" "$ROOT/llama_cpp_modified/$f"
done < "$ROOT/llama_cpp_modified/FILES.txt"
echo "synced $(wc -l < "$ROOT/llama_cpp_modified/FILES.txt") files"
