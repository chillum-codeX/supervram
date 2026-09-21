#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analytical SuperVRAM transfer roofline (projected, not measured)")
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--embedding", type=int, default=2048)
    parser.add_argument("--expert-hidden", type=int, default=768)
    parser.add_argument("--bits-per-weight", type=float, default=4.5, help="Effective quantized bits including block metadata")
    parser.add_argument("--nvme-gbps", type=float, default=7.0)
    parser.add_argument("--pcie-gbps", type=float, default=24.0, help="Effective PCIe transfer bandwidth")
    parser.add_argument("--fixed-read-us", type=float, default=80.0)
    parser.add_argument("--reads-per-expert", type=int, default=1)
    parser.add_argument("--compute-ms-token", type=float, default=40.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    params_per_expert = 3 * args.embedding * args.expert_hidden
    expert_bytes = params_per_expert * args.bits_per_weight / 8.0
    total_expert_bytes = expert_bytes * args.layers * args.experts
    active_bytes = expert_bytes * args.layers * args.top_k
    bottleneck_gbps = min(args.nvme_gbps, args.pcie_gbps)
    results = []
    for hit_rate in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        miss_experts = args.layers * args.top_k * (1.0 - hit_rate)
        bytes_token = active_bytes * (1.0 - hit_rate)
        bandwidth_ms = bytes_token / (bottleneck_gbps * 1e9) * 1e3
        latency_ms = miss_experts * args.reads_per_expert * args.fixed_read_us / 1000.0
        serialized_ms = bandwidth_ms + latency_ms
        hidden_ms = min(serialized_ms, args.compute_ms_token)
        residual_ms = max(0.0, serialized_ms - hidden_ms)
        token_ms = args.compute_ms_token + residual_ms
        results.append({
            "hit_rate": hit_rate,
            "miss_experts_per_token": miss_experts,
            "bytes_per_token": bytes_token,
            "bandwidth_time_ms": bandwidth_ms,
            "read_latency_time_ms": latency_ms,
            "serialized_io_ms": serialized_ms,
            "max_hideable_ms": hidden_ms,
            "projected_token_ms": token_ms,
            "projected_tokens_s": 1000.0 / token_ms,
        })
    output = {
        "schema_version": 1,
        "evidence_class": "analytical_projection",
        "warning": "Values are projections from explicit parameters, not hardware measurements.",
        "parameters": vars(args),
        "derived": {
            "parameters_per_expert": params_per_expert,
            "expert_bytes": expert_bytes,
            "all_experts_bytes": total_expert_bytes,
            "active_expert_bytes_per_token_without_cache": active_bytes,
            "transfer_bottleneck_gbps": bottleneck_gbps,
        },
        "scenarios": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
