#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import tempfile
import time

from supervram import Access, ExpertCache, ExpertKey, SuperVRAM, TensorStore, TensorStoreWriter, make_policy, make_predictor
from supervram.predictors import OraclePredictor
from supervram.scheduler import AdaptiveSpeculativeScheduler
from supervram.trace import TraceWriter
from supervram.types import CostModelParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cost_model import project_throughput_dict  # noqa: E402


def generate_trace(tokens: int, layers: int, experts: int, top_k: int, seed: int, locality: float) -> list[Access]:
    rng = random.Random(seed)
    previous = {layer: tuple(rng.sample(range(experts), top_k)) for layer in range(layers)}
    accesses = []
    for token in range(tokens):
        for layer in range(layers):
            if rng.random() < locality:
                selected = list(previous[layer])
                if rng.random() < 0.35:
                    selected[rng.randrange(top_k)] = rng.randrange(experts)
                    selected = list(dict.fromkeys(selected))
                    while len(selected) < top_k:
                        candidate = rng.randrange(experts)
                        if candidate not in selected:
                            selected.append(candidate)
            else:
                selected = rng.sample(range(experts), top_k)
            previous[layer] = tuple(selected)
            raw = {expert: rng.uniform(0.5, 1.0) for expert in selected}
            for expert in rng.sample([x for x in range(experts) if x not in selected], min(top_k, experts - top_k)):
                raw[expert] = rng.uniform(0.0, 0.49)
            total = sum(raw.values())
            accesses.append(Access(token, layer, tuple(selected), {k: v / total for k, v in raw.items()}))
    return accesses


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic SuperVRAM trace-replay simulator")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--expert-bytes", type=int, default=64 * 1024)
    parser.add_argument("--cache-experts", type=int, default=8)
    parser.add_argument("--cache-bytes", type=int)
    parser.add_argument("--async-replay", action="store_true", help="Use wall-clock asynchronous completion; default replay is deterministic")
    parser.add_argument("--policy", default="router-aware")
    parser.add_argument("--predictor", default="markov")
    parser.add_argument("--prefetch-depth", type=int, default=4)
    parser.add_argument("--locality", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scheduler", choices=["off", "ass"], default="off",
                         help="off: process() blind prefetch (baseline). ass: AdaptiveSpeculativeScheduler "
                              "confidence-gated, cost-aware, lookahead prefetch (PLAN_ADAPTIVE.md)")
    parser.add_argument("--gate-threshold", type=float, default=CostModelParams().gate_threshold,
                         help="minimum calibrated confidence required to fire a speculative read")
    parser.add_argument("--lookahead-d", type=int, default=CostModelParams().lookahead_d,
                         help="how many upcoming accesses the scheduler plans reads across per step")
    parser.add_argument("--drive-gbps", type=float, default=CostModelParams().drive_gbps,
                         help="drive bandwidth constant used to estimate a candidate read's cost")
    parser.add_argument("--cost-model", action="store_true",
                         help="emit an analytical modeled-throughput projection (PLAN_ADAPTIVE.md section 2.3) "
                              "alongside the raw cache/scheduler counters -- not a hardware measurement")
    parser.add_argument("--store")
    parser.add_argument("--trace")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    accesses = generate_trace(args.tokens, args.layers, args.experts, args.top_k, args.seed, args.locality)
    if args.predictor == "oracle":
        predictor = OraclePredictor(accesses[args.layers:])
    else:
        predictor = make_predictor(args.predictor)

    temporary = None
    if args.store:
        store_path = Path(args.store)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="supervram-")
        store_path = Path(temporary.name) / "experts.svram"
    if not store_path.exists():
        with TensorStoreWriter(store_path) as writer:
            for layer in range(args.layers):
                for expert in range(args.experts):
                    pattern = bytes([(layer * args.experts + expert) % 251])
                    writer.add(ExpertKey(layer, expert), [pattern * args.expert_bytes])

    trace = TraceWriter(args.trace)
    start = time.perf_counter_ns()
    with TensorStore(store_path) as store, ExpertCache(
        store,
        args.cache_bytes or args.cache_experts * args.expert_bytes,
        make_policy(args.policy),
        queue_depth=max(1, args.prefetch_depth) if args.async_replay else 1,
        trace=trace,
        deterministic=not args.async_replay,
    ) as cache:
        engine = SuperVRAM(cache, predictor, args.prefetch_depth)
        params = CostModelParams(drive_gbps=args.drive_gbps, lookahead_d=args.lookahead_d, gate_threshold=args.gate_threshold)
        scheduler = None
        if args.scheduler == "ass":
            # Independent predictor instance, not the same object as `predictor` above: the
            # scheduler calibrates its own confidence from the same access stream, and sharing a
            # single stateful predictor between engine._resolve() and the scheduler would observe
            # every access twice into one history, corrupting it.
            if args.predictor == "oracle":
                scheduler_predictor = OraclePredictor(accesses[args.layers:])
            else:
                scheduler_predictor = make_predictor(args.predictor)
            scheduler = AdaptiveSpeculativeScheduler(scheduler_predictor, lambda key: store.extents[key].length, params, top_k=args.prefetch_depth)
        for index, access in enumerate(accesses):
            if scheduler is not None:
                window = accesses[index : index + args.lookahead_d]
                engine.process_window(window, scheduler)
            else:
                engine.process(access)
            if not args.async_replay:
                cache.drain()
        cache.drain()
        elapsed_ns = time.perf_counter_ns() - start
        metrics = engine.metrics()
        result = {
            "schema_version": 1,
            "evidence_class": "synthetic_async_trace_replay" if args.async_replay else "deterministic_synthetic_trace_replay",
            "warning": "This is a software-policy validation result, not an RTX 3090 inference benchmark.",
            "parameters": vars(args),
            "accesses": len(accesses),
            "elapsed_ns": elapsed_ns,
            "metrics": metrics,
        }
        if args.cost_model:
            result["cost_model"] = project_throughput_dict(metrics["cache"], params, args.tokens, args.tokens * args.layers)
    trace.close()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    if temporary:
        temporary.cleanup()


if __name__ == "__main__":
    main()
