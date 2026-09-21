#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
LLAMA_GGUF = ROOT / "third_party" / "llama.cpp" / "gguf-py"
sys.path.insert(0, str(LLAMA_GGUF))

from gguf import GGUFReader  # noqa: E402

from supervram import ExpertKey, TensorStoreWriter  # noqa: E402

PATTERN = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack contiguous Qwen MoE expert slices from GGUF")
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    reader = GGUFReader(args.model, "r")
    tensors: dict[int, dict[str, object]] = {}
    ignored = 0
    for tensor in reader.tensors:
        match = PATTERN.match(tensor.name)
        if not match:
            ignored += 1
            continue
        layer = int(match.group(1))
        projection = match.group(2)
        tensors.setdefault(layer, {})[projection] = tensor
    if not tensors:
        raise SystemExit("no Qwen-style ffn gate/up/down expert tensors found")

    details = []
    with TensorStoreWriter(args.output, args.alignment) as writer:
        for layer, projections in sorted(tensors.items()):
            missing = {"gate", "up", "down"} - projections.keys()
            if missing:
                raise ValueError(f"layer {layer} missing projections: {sorted(missing)}")
            expert_counts = {int(tensor.shape[-1]) for tensor in projections.values()}
            if len(expert_counts) != 1:
                raise ValueError(f"layer {layer} has inconsistent expert counts")
            n_experts = expert_counts.pop()
            raw = {name: memoryview(tensor.data).cast("B") for name, tensor in projections.items()}
            for name, blob in raw.items():
                if len(blob) % n_experts:
                    raise ValueError(f"{projections[name].name} byte size is not divisible by expert count")
            bytes_per = {name: len(blob) // n_experts for name, blob in raw.items()}
            for expert in range(n_experts):
                names = (projections["gate"].name, projections["up"].name, projections["down"].name)
                blobs = []
                for name in ("gate", "up", "down"):
                    size = bytes_per[name]
                    blobs.append(bytes(raw[name][expert * size : (expert + 1) * size]))
                extent = writer.add(ExpertKey(layer, expert), blobs, names)
                details.append({
                    "layer": layer,
                    "expert": expert,
                    "offset": extent.offset,
                    "length": extent.length,
                    "projection_bytes": bytes_per,
                    "tensor_types": {name: projections[name].tensor_type.name for name in projections},
                })
    metadata = {
        "schema_version": 1,
        "source_model": str(args.model.resolve()),
        "source_size": args.model.stat().st_size,
        "layers": len(tensors),
        "experts": len(details),
        "ignored_tensors": ignored,
        "extents": details,
        "warning": "Packed bytes preserve the GGUF tensor encoding. GPU consumers must honor each ggml quantization layout.",
    }
    metadata_path = args.metadata or args.output.with_suffix(args.output.suffix + ".pack.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: metadata[key] for key in ("source_model", "layers", "experts", "ignored_tensors")}, indent=2))


if __name__ == "__main__":
    main()
