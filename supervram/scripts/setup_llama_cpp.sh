#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${SUPERVRAM_LLAMA_DIR:-$ROOT/third_party/llama.cpp}"
REV=ce8caa6e60a03093351d6016a818720e0d46f0fb
PATCHES=("$ROOT"/patches/*.patch)

if [[ ! -d "$DEST/.git" ]]; then
    mkdir -p "$(dirname "$DEST")"
    git clone https://github.com/ggml-org/llama.cpp.git "$DEST"
fi

git -C "$DEST" fetch origin "$REV"
git -C "$DEST" checkout --detach "$REV"
git -C "$DEST" reset --hard "$REV"
git -C "$DEST" clean -fdx

for patch in "${PATCHES[@]}"; do
    git -C "$DEST" apply --check "$patch"
    git -C "$DEST" apply "$patch"
    printf 'applied %s\n' "$(basename "$patch")"
done
printf 'llama.cpp %s with SuperVRAM patches is ready at %s\n' "$REV" "$DEST"
