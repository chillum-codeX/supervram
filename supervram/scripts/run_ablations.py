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
CACHE_GIB = [4, 8, 12, 16, 20, 22]
PREFETCH = [0, 1, 2, 4, 8]
POLICIES = ["lru", "lfu", "weighted", "router-aware"]
PREDICTORS = ["none", "history", "router-probability", "markov", "lightweight", "oracle"]

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
        combinations = list(itertools.product(CACHE_GIB, PREFETCH, POLICIES, PREDICTORS))
        if args.max_runs:
            combinations = combinations[: args.max_runs]
        for index, (cache_gib, depth, policy, predictor) in enumerate(combinations):
            run_id = f"run-{index:04d}-c{cache_gib}-p{depth}-{policy}-{predictor}"
            result_path = args.output_dir / f"{run_id}.json"
            expert_bytes = 64 * 1024
            cache_bytes = cache_gib * 1024**3
            command = [
                sys.executable,
                str(Path(__file__).with_name("simulate_trace.py")),
                "--cache-bytes", str(cache_bytes),
                "--expert-bytes", str(expert_bytes),
                "--prefetch-depth", str(depth),
                "--policy", policy,
                "--predictor", predictor,
                "--output", str(result_path),
            ]
            record = run(command, args.output_dir / f"{run_id}.status.json", args.mode == "plan")
            manifest.append({"run_id": run_id, "cache_gib": cache_gib, "effective_cache_bytes": cache_gib * 1024**3, "prefetch_depth": depth, "policy": policy, "predictor": predictor, **record})

    (args.output_dir / "manifest.json").write_text(json.dumps({"schema_version": 1, "mode": args.mode, "runs": manifest}, indent=2) + "\n")
    fieldnames = ["run_id", "cache_mib", "policy", "status", "decode_tps", "cache_hit_rate_percent", "cache_accesses", "cache_hits", "cache_misses", "cache_evictions"] if args.mode == "target" else ["run_id", "cache_gib", "prefetch_depth", "policy", "predictor", "status"]
    with (args.output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)
    print(f"wrote {len(manifest)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
