#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from pathlib import Path
import subprocess
import sys

# Simulation-mode axes (standalone Python policy prototype; see scripts/simulate_trace.py).
PREFETCH = [0, 1, 2, 4, 8]
POLICIES = ["lru", "lfu", "weighted", "router-aware"]
PREDICTORS = ["none", "history", "router-probability", "markov", "lightweight", "oracle"]

# The synthetic trace's shape, fixed across the whole matrix (previously left at
# simulate_trace.py's own tiny defaults: 32 experts x 8 layers x 64 KiB = 16 MiB total, which a
# multi-GiB cache axis could never put under any real pressure -- every run had zero evictions,
# so nothing in the matrix, ASS or otherwise, could show a difference). What actually needs to be
# "properly scaled" here is the *ratio* between cache size and working-set size (see
# CACHE_FRACTIONS below), not the trace's absolute size -- this is pure byte-counting simulation,
# not a real model, so a bigger trace buys nothing but slower runs. A first attempt at
# layers=24/experts=64/top_k=8/tokens=120 (chasing literal Q8_0-model proportions) made each run
# ~13x more work than the original tiny-trace default and would have taken the full 3,600-run
# matrix ~3 hours; this is sized down to keep meaningful pressure while staying fast.
TRACE_LAYERS = 12
TRACE_EXPERTS = 32
TRACE_TOP_K = 4
TRACE_EXPERT_BYTES = 256 * 1024
TRACE_TOKENS = 80
# Default locality (0.8) makes history-based prediction accidentally strong and does not
# represent this project's own measured routing-history accuracy (1.4-2 %, EVIDENCE_LEDGER.md) --
# a lower, weak-predictor-representative locality is what actually exercises the gate.
TRACE_LOCALITY = 0.2
WORKING_SET_BYTES = TRACE_LAYERS * TRACE_EXPERTS * TRACE_EXPERT_BYTES
STORE_FILENAME = "shared-experts.svram"

# Cache sizes as fractions of the trace's total working set, not absolute GiB: an absolute axis
# is only meaningful relative to how much data there is to cache, and it needs to span "much
# smaller than the working set" (forces eviction) through "as large as the working set" (nothing
# ever evicted) to say anything about policy or scheduler differences at all.
CACHE_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]

# ASS axes (PLAN_ADAPTIVE.md Phase C, "scheduler x gate x lookahead"). Kept as a small addendum
# to each base (cache/prefetch/policy/predictor) combination rather than a full cross product
# with it -- one extra "off" row plus one row per (gate, lookahead) pair for "ass" -- so this
# stays additive without exploding the existing matrix's already-large run count.
GATE_THRESHOLD = [0.3, 0.5]
LOOKAHEAD_D = [2, 4]

# Target-mode axes: the integrated ggml_backend_sched expert cache only implements on-demand
# LRU/LFU (no prefetch/predictors - see docs/IMPLEMENTATION_STATUS.md), so this matrix is
# intentionally smaller than the simulation-mode one above.
TARGET_CACHE_MIB = [4096, 8192, 12288, 16384, 20480]
TARGET_POLICIES = ["lru", "lfu"]


def run(command: list[str], status_output: Path, dry_run: bool) -> dict:
    if dry_run:
        return {"status": "planned", "command": command}
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    record = {"status": "ok" if result.returncode == 0 else "failed", "returncode": result.returncode, "command": command, "stdout": result.stdout, "stderr": result.stderr}
    status_output.write_text(json.dumps(record, indent=2) + "\n")
    return record


def read_simulation_result(result_path: Path) -> dict:
    """Pulls the numbers the ablation CSV needs to show the ASS win and the oracle gap
    (PLAN_ADAPTIVE.md Phase C deliverable) out of one simulate_trace.py --cost-model JSON file."""
    if not result_path.exists():
        return {}
    try:
        data = json.loads(result_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    cost = data.get("cost_model") or {}
    cache = data.get("metrics", {}).get("cache", {})
    scheduler = data.get("metrics", {}).get("scheduler") or {}
    return {
        "modeled_tokens_per_second": cost.get("modeled_tokens_per_second"),
        "waste_bytes": cache.get("waste_bytes"),
        "hit_rate": cache.get("hit_rate"),
        "scheduler_fire_rate": scheduler.get("fire_rate"),
    }


def parse_svram_verify_stdout(stdout: str) -> dict:
    """Extract the mode= summary line and cache_stats line printed by svram-verify."""
    parsed: dict = {}
    mode_match = re.search(r"^mode=(\S+) ngl=(\S+) cpu_moe=(\S+) n_cpu_moe=(\S+) cache_mib=(\S+) prompt_tokens=(\S+) generated=(\S+) load_ms=(\S+) prompt_ms=(\S+) decode_tps=(\S+)$", stdout, re.MULTILINE)
    if mode_match:
        parsed["decode_tps"] = float(mode_match.group(10))
        parsed["load_ms"] = float(mode_match.group(8))
        parsed["prompt_ms"] = float(mode_match.group(9))
    stats_match = re.search(r"^cache_stats accesses=(\d+) hits=(\d+) misses=(\d+) evictions=(\d+) hit_rate=(\S+) bytes_h2d=(\d+) n_slots=(\d+) host_ms=(\S+)$", stdout, re.MULTILINE)
    if stats_match:
        parsed["cache_accesses"] = int(stats_match.group(1))
        parsed["cache_hits"] = int(stats_match.group(2))
        parsed["cache_misses"] = int(stats_match.group(3))
        parsed["cache_evictions"] = int(stats_match.group(4))
        parsed["cache_hit_rate_percent"] = float(stats_match.group(5))
        parsed["cache_bytes_h2d"] = int(stats_match.group(6))
        parsed["cache_n_slots"] = int(stats_match.group(7))
        parsed["cache_host_ms"] = float(stats_match.group(8))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="SuperVRAM requested ablation matrix")
    parser.add_argument("--mode", choices=["plan", "simulate", "target"], default="plan")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", help="GGUF model for target mode")
    parser.add_argument("--svram-verify", default="svram-verify", help="path to the svram-verify binary (target mode)")
    parser.add_argument("--n-predict", type=int, default=32, help="decode steps per target-mode run")
    parser.add_argument("--n-ubatch", type=int, default=1, help="decode batch size per target-mode run (keep small: distinct experts/step must stay <= slots)")
    parser.add_argument("--max-runs", type=int, default=0, help="0 runs all combinations")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    if args.mode == "target":
        if not args.model:
            parser.error("--model is required in target mode")
        combinations = list(itertools.product(TARGET_CACHE_MIB, TARGET_POLICIES))
        if args.max_runs:
            combinations = combinations[: args.max_runs]
        for index, (cache_mib, policy) in enumerate(combinations):
            run_id = f"run-{index:04d}-c{cache_mib}mib-{policy}"
            result_path = args.output_dir / f"{run_id}.json"
            command = [
                args.svram_verify,
                "--model", args.model,
                "--storage", "cache",
                "--cache-mib", str(cache_mib),
                "--cache-policy", policy,
                "--n-predict", str(args.n_predict),
                "--n-ubatch", str(args.n_ubatch),
                "--json", str(result_path),
            ]
            record = run(command, args.output_dir / f"{run_id}.status.json", False)
            record["evidence_class"] = "measured_rtx3090"
            record.update(parse_svram_verify_stdout(record.get("stdout", "")))
            manifest.append({"run_id": run_id, "cache_mib": cache_mib, "policy": policy, **record})
    else:
        # Built once, then reused by every run via --store: simulate_trace.py skips rebuilding a
        # store file that already exists, and building a few-hundred-MiB file (rather than the
        # old per-run temp store) 3,600 times would be wasteful disk I/O for identical content.
        shared_store = args.output_dir / STORE_FILENAME

        base_combinations = list(itertools.product(CACHE_FRACTIONS, PREFETCH, POLICIES, PREDICTORS))
        # Each base combination gets one "off" row plus one "ass" row per (gate, lookahead) pair
        # -- an addendum, not a full cross product, per the comment on GATE_THRESHOLD above.
        scheduler_variants: list[tuple[str, float | None, int | None]] = [("off", None, None)]
        scheduler_variants += [("ass", gate, lookahead) for gate, lookahead in itertools.product(GATE_THRESHOLD, LOOKAHEAD_D)]
        combinations = [(fraction, depth, policy, predictor, scheduler, gate, lookahead)
                         for (fraction, depth, policy, predictor) in base_combinations
                         for (scheduler, gate, lookahead) in scheduler_variants]
        if args.max_runs:
            combinations = combinations[: args.max_runs]
        for index, (fraction, depth, policy, predictor, scheduler, gate, lookahead) in enumerate(combinations):
            sched_tag = "off" if scheduler == "off" else f"ass-g{gate}-d{lookahead}"
            run_id = f"run-{index:04d}-c{fraction}-p{depth}-{policy}-{predictor}-{sched_tag}"
            result_path = args.output_dir / f"{run_id}.json"
            cache_bytes = max(1, round(fraction * WORKING_SET_BYTES))
            command = [
                sys.executable,
                str(Path(__file__).with_name("simulate_trace.py")),
                "--layers", str(TRACE_LAYERS),
                "--experts", str(TRACE_EXPERTS),
                "--top-k", str(TRACE_TOP_K),
                "--expert-bytes", str(TRACE_EXPERT_BYTES),
                "--tokens", str(TRACE_TOKENS),
                "--locality", str(TRACE_LOCALITY),
                "--cache-bytes", str(cache_bytes),
                "--prefetch-depth", str(depth),
                "--policy", policy,
                "--predictor", predictor,
                "--scheduler", scheduler,
                "--cost-model",
                "--store", str(shared_store),
                "--output", str(result_path),
            ]
            if scheduler == "ass":
                command += ["--gate-threshold", str(gate), "--lookahead-d", str(lookahead)]
            record = run(command, args.output_dir / f"{run_id}.status.json", args.mode == "plan")
            if record.get("status") == "ok":
                record.update(read_simulation_result(result_path))
            manifest.append({
                "run_id": run_id, "cache_fraction": fraction, "cache_bytes": cache_bytes,
                "working_set_bytes": WORKING_SET_BYTES,
                "prefetch_depth": depth, "policy": policy, "predictor": predictor,
                "scheduler": scheduler, "gate_threshold": gate, "lookahead_d": lookahead,
                **record,
            })

    (args.output_dir / "manifest.json").write_text(json.dumps({"schema_version": 1, "mode": args.mode, "runs": manifest}, indent=2) + "\n")
    fieldnames = (
        ["run_id", "cache_mib", "policy", "status", "decode_tps", "cache_hit_rate_percent", "cache_accesses", "cache_hits", "cache_misses", "cache_evictions"]
        if args.mode == "target"
        else ["run_id", "cache_fraction", "cache_bytes", "prefetch_depth", "policy", "predictor", "scheduler", "gate_threshold", "lookahead_d", "status", "modeled_tokens_per_second", "waste_bytes", "hit_rate", "scheduler_fire_rate"]
    )
    with (args.output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)
    print(f"wrote {len(manifest)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
