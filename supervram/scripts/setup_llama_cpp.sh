#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/novix/workspace/project/supervram
DEST="$ROOT/third_party/llama.cpp"
REV=ce8caa6e60a03093351d6016a818720e0d46f0fb
PATCH="$ROOT/patches/0001-qwen3-moe-mmap-expert-storage.patch"

if [[ ! -d "$DEST/.git" ]]; then
    mkdir -p "$(dirname "$DEST")"
    git clone https://github.com/ggml-org/llama.cpp.git "$DEST"
fi

git -C "$DEST" fetch origin "$REV"
git -C "$DEST" checkout --detach "$REV"
git -C "$DEST" reset --hard "$REV"
git -C "$DEST" clean -fdx

git -C "$DEST" apply --check "$PATCH"
git -C "$DEST" apply "$PATCH"
printf 'llama.cpp %s with SuperVRAM patch is ready at %s\n' "$REV" "$DEST"
