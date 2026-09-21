#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

CACHE_GIB = [4, 8, 12, 16, 20, 22]
PREFETCH = [0, 1, 2, 4, 8]
POLICIES = ["lru", "lfu", "weighted", "router-aware"]
PREDICTORS = ["none", "history", "router-probability", "markov", "lightweight", "oracle"]


def run(command: list[str], status_output: Path, dry_run: bool) -> dict:
    if dry_run:
        return {"status": "planned", "command": command}
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    record = {"status": "ok" if result.returncode == 0 else "failed", "returncode": result.returncode, "command": command, "stdout": result.stdout, "stderr": result.stderr}
    status_output.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="SuperVRAM requested ablation matrix")
    parser.add_argument("--mode", choices=["plan", "simulate", "target"], default="plan")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", help="GGUF model for target mode")
    parser.add_argument("--llama-server", default="llama-server")
    parser.add_argument("--max-runs", type=int, default=0, help="0 runs all combinations")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "target":
        parser.error("target mode is disabled until the compact GPU-cache CLI is integrated; use benchmark_target.py for resident/mmap baselines")
    combinations = list(itertools.product(CACHE_GIB, PREFETCH, POLICIES, PREDICTORS))
    if args.max_runs:
        combinations = combinations[: args.max_runs]
    manifest = []
    for index, (cache_gib, depth, policy, predictor) in enumerate(combinations):
        run_id = f"run-{index:04d}-c{cache_gib}-p{depth}-{policy}-{predictor}"
        result_path = args.output_dir / f"{run_id}.json"
        if args.mode == "target":
            if not args.model:
                parser.error("--model is required in target mode")
            command = [
                args.llama_server,
                "-m", args.model,
                "--n-gpu-layers", "all",
                "--svram-store", os.environ.get("SVRAM_STORE", "experts.svram"),
                "--svram-cache", f"{cache_gib}G",
                "--svram-prefetch", str(depth),
                "--svram-policy", policy,
                "--svram-predictor", predictor,
                "--svram-metrics", str(result_path),
            ]
        else:
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
    with (args.output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "cache_gib", "prefetch_depth", "policy", "predictor", "status"])
        writer.writeheader()
        writer.writerows({key: item[key] for key in writer.fieldnames} for item in manifest)
    print(f"wrote {len(manifest)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
