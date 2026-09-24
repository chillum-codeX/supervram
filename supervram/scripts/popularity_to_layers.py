#!/usr/bin/env python3
"""Aggregate per-expert routing traces into per-layer popularity, assign each layer to a
precision tier (Q8_0 hot / Q4_K cold), and emit a llama-quantize --tensor-type-file mapping.

Note: --tensor-type-file takes a raw ggml_type name (matched against ggml_type_name(), e.g.
"Q4_K"), not a whole-model quantize preset like "Q4_K_M" -- those go through a different parser
(the main positional `type` argument) and are not valid here (verified: llama-quantize rejects
"Q4_K_M" with "parse_ggml_type: invalid ggml_type"). Q4_K is what Q4_K_M itself uses for most
tensors uniformly (~4.25 bits/weight); Q4_K_M's extra per-tensor upgrades to Q6_K don't apply to
MoE expert tensors in the whole-model pipeline either, so plain Q4_K is the correct, closest
target for a manual per-tensor override.

See docs/SPEC_IDEA1_POPULARITY_TIERED_PRECISION.md section 3.1 for the design.

Trace line format (as produced by --trace / SVRAM_TRACE, one line per token per layer):
    <layer> <expert_1> <expert_2> ...

Usage:
    python3 popularity_to_layers.py --traces trace1.txt trace2.txt ... \
        --policy topk --k 24 --out-prefix results/tiered/layers
"""
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

_EXPERT_TENSOR_RE = re.compile(r"^blk\.(\d+)\.ffn_(?:gate|up|down)_exps\.weight$")
_Q4_OVER_Q8_BPW = 108.0 / 204.0  # measured Q4_K/Q8_0 ratio for one Qwen3-30B-A3B expert tensor (llama-quantize --dry-run)


def _llama_cpp_gguf_py_path():
    return Path(__file__).resolve().parent.parent / "third_party" / "llama.cpp" / "gguf-py"


def load_layer_expert_counts(trace_paths):
    """Returns ({layer: {expert: count}}, {layer: tokens_seen}), aggregated across all given
    traces."""
    count = defaultdict(lambda: defaultdict(int))
    tokens = defaultdict(int)
    for path in trace_paths:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                layer = int(parts[0])
                tokens[layer] += 1
                for e in parts[1:]:
                    count[layer][int(e)] += 1
    if not count:
        raise ValueError(f"no trace lines found in {trace_paths}")
    return count, tokens


def layer_diagnostics(layer_expert_counts):
    """Per-layer entropy (bits) and top-1 share of the expert-usage distribution, plus the
    count of distinct experts that fired at all. Qwen3-30B-A3B's router always activates a
    fixed top-8 experts per token, so raw activation-count is identical (~8.0) for every layer
    and carries no ranking signal -- these distributional stats are what actually vary."""
    diag = {}
    for layer, counts in layer_expert_counts.items():
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        diag[layer] = {
            "entropy_bits": entropy,
            "top1_share": max(probs),
            "distinct_experts_used": len(counts),
        }
    return diag


def compute_popularity(layer_expert_counts, tokens, metric):
    """Returns {layer: score}, where a HIGHER score means the layer should be preferred for the
    Q8_0 (hot) tier.

    metric == "activation_count": literal reading of the original spec (sum of routing counts /
    tokens seen). Kept for reproducibility, but is degenerate (near-constant ~8.0 for every
    layer) on a fixed-top-k router like Qwen3-30B-A3B's -- do not use this to actually pick tiers.

    metric == "entropy" (default, corrected): score = -entropy_bits(layer). Low-entropy layers
    route most of their traffic through a small set of "specialist" experts, used consistently
    across many tokens -- those experts' precision matters more, so such layers are treated as
    hot. High-entropy layers spread traffic thinly across nearly all 128 experts, so any single
    quantized expert is rarely the one actually used -- this mirrors the project's own
    already-evidenced per-expert claim ("quality tolerance is highest for experts that rarely
    fire"), applied at the layer level. This has NOT been empirically validated against the
    quality gate (SPEC section 4.3) -- treat the direction as a hypothesis the gate must confirm,
    not a settled fact.
    """
    if metric == "activation_count":
        return {l: sum(c.values()) / tokens[l] for l, c in layer_expert_counts.items()}
    diag = layer_diagnostics(layer_expert_counts)
    return {l: -d["entropy_bits"] for l, d in diag.items()}


def assign_tiers_topk(popularity, k):
    ranked = sorted(popularity, key=lambda l: popularity[l], reverse=True)
    q8 = set(ranked[:k])
    return q8


def assign_tiers_threshold(popularity, theta):
    q8 = {l for l, p in popularity.items() if p >= theta}
    return q8


def assign_tiers_bytes(popularity, bytes_per_layer, byte_budget):
    ranked = sorted(popularity, key=lambda l: popularity[l], reverse=True)
    q8 = set()
    spent = 0
    for l in ranked:
        cost = bytes_per_layer.get(l, 0)
        if spent + cost > byte_budget:
            break
        q8.add(l)
        spent += cost
    return q8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", nargs="+", required=True, help="one or more --trace / SVRAM_TRACE output files")
    ap.add_argument("--policy", choices=["topk", "threshold", "bytes"], default="topk")
    ap.add_argument("--metric", choices=["entropy", "activation_count"], default="entropy",
                     help="entropy (default): concentration of each layer's expert-usage distribution -- "
                          "the corrected signal, since activation_count is degenerate on a fixed-top-k "
                          "router (see compute_popularity docstring). activation_count: literal spec "
                          "reading, kept for reproducibility/debugging only.")
    ap.add_argument("--k", type=int, default=24, help="topk policy: number of layers to keep at Q8_0")
    ap.add_argument("--theta", type=float, default=None, help="threshold policy: minimum popularity to keep at Q8_0")
    ap.add_argument("--byte-budget", type=int, default=None, help="bytes policy: total Q8 byte budget")
    ap.add_argument("--bytes-per-layer", type=int, default=None,
                     help="bytes policy: assumed uniform per-layer expert-tensor byte cost at Q8_0 "
                          "(pass real per-layer sizes via --bytes-per-layer-json for a non-uniform model)")
    ap.add_argument("--bytes-per-layer-json", default=None,
                     help="bytes policy: JSON file mapping layer index (str) -> Q8_0 expert-tensor byte size")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--q4-type", default="Q4_K",
                     help="raw ggml_type for cold layers (default Q4_K; NOT Q4_K_M -- that's a "
                          "whole-model preset name, invalid here, see module docstring)")
    ap.add_argument("--gguf", default=None,
                     help="source Q8_0 GGUF to read real per-layer expert-tensor byte sizes from "
                          "(for the summary's byte accounting; Q4 bytes are estimated via a "
                          "measured Q4_K/Q8_0 size ratio since no Q4 file is quantized yet)")
    args = ap.parse_args()

    layer_q8_bytes = {}
    if args.gguf:
        sys.path.insert(0, str(_llama_cpp_gguf_py_path()))
        from gguf.gguf_reader import GGUFReader
        reader = GGUFReader(args.gguf)
        for t in reader.tensors:
            m = _EXPERT_TENSOR_RE.match(t.name)
            if m:
                layer_q8_bytes[int(m.group(1))] = layer_q8_bytes.get(int(m.group(1)), 0) + int(t.n_bytes)

    layer_expert_counts, tokens = load_layer_expert_counts(args.traces)
    diag = layer_diagnostics(layer_expert_counts)
    popularity = compute_popularity(layer_expert_counts, tokens, args.metric)
    total_layers = len(popularity)

    if args.policy == "topk":
        if args.k > total_layers:
            sys.exit(f"error: --k {args.k} exceeds the {total_layers} layers found in the traces")
        q8_layers = assign_tiers_topk(popularity, args.k)
    elif args.policy == "threshold":
        if args.theta is None:
            sys.exit("error: --policy threshold requires --theta")
        q8_layers = assign_tiers_threshold(popularity, args.theta)
    else:  # bytes
        if args.bytes_per_layer_json:
            with open(args.bytes_per_layer_json) as f:
                raw = json.load(f)
            bytes_per_layer = {int(k): v for k, v in raw.items()}
        elif args.bytes_per_layer is not None:
            bytes_per_layer = {l: args.bytes_per_layer for l in popularity}
        elif layer_q8_bytes:
            bytes_per_layer = layer_q8_bytes
        else:
            sys.exit("error: --policy bytes requires --bytes-per-layer, --bytes-per-layer-json, or --gguf")
        if args.byte_budget is None:
            sys.exit("error: --policy bytes requires --byte-budget")
        q8_layers = assign_tiers_bytes(popularity, bytes_per_layer, args.byte_budget)

    q4_layers = sorted(set(popularity) - q8_layers)
    q8_layers = sorted(q8_layers)

    tt_path = f"{args.out_prefix}.tensor-type-file"
    with open(tt_path, "w") as f:
        for layer in q4_layers:
            f.write(f"blk\\.{layer}\\.ffn_(gate|up|down)_exps\\.weight={args.q4_type}\n")

    ranking = sorted(
        ({"layer": l, "popularity": round(popularity[l], 6), "tier": "q8" if l in q8_layers else "q4",
          "entropy_bits": round(diag[l]["entropy_bits"], 4), "top1_share": round(diag[l]["top1_share"], 4),
          "distinct_experts_used": diag[l]["distinct_experts_used"]}
         for l in popularity),
        key=lambda r: r["popularity"], reverse=True,
    )
    summary = {
        "policy": args.policy,
        "metric": args.metric,
        "k": args.k if args.policy == "topk" else None,
        "theta": args.theta if args.policy == "threshold" else None,
        "traces": args.traces,
        "total_layers": total_layers,
        "q8_layers": q8_layers,
        "q4_layers": q4_layers,
        "q4_type": args.q4_type,
        "layer_popularity_ranking": ranking,
        "metric_note": ("activation_count is near-constant (~8.0) for every layer on this fixed-top-8 "
                         "router and carries no ranking signal; entropy is the corrected, actually-"
                         "discriminating metric used by default -- see compute_popularity() docstring "
                         "for the (unvalidated) hypothesis behind its tiering direction."),
    }
    if layer_q8_bytes:
        q8_bytes = sum(layer_q8_bytes.get(l, 0) for l in q8_layers)
        q4_bytes_at_q8 = sum(layer_q8_bytes.get(l, 0) for l in q4_layers)
        q4_bytes_estimate = round(q4_bytes_at_q8 * _Q4_OVER_Q8_BPW)
        summary["byte_accounting"] = {
            "source_gguf": args.gguf,
            "q8_layers_bytes_at_q8": q8_bytes,
            "q4_layers_bytes_at_q8": q4_bytes_at_q8,
            "q4_layers_bytes_estimate_at_q4": q4_bytes_estimate,
            "total_expert_bytes_estimate": q8_bytes + q4_bytes_estimate,
            "total_expert_bytes_if_all_q8": q8_bytes + q4_bytes_at_q8,
            "note": "q4 estimate uses the measured Q4_K/Q8_0 per-tensor size ratio from a single-layer --dry-run, not the full built GGUF; verify against the actual built GGUF (gate 1).",
        }
    summary_path = f"{args.out_prefix}.summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{total_layers} layers total: {len(q8_layers)} Q8_0 (hot), {len(q4_layers)} {args.q4_type} (cold)")
    print(f"wrote {tt_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
