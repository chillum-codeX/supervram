#!/usr/bin/env python3
"""Aggregate repeated target-mode ablation runs (results/rtx3090/ablations-long/<model>/rep*/)."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summary: dict = {}
    for model_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        cache: dict = defaultdict(lambda: {"tps": [], "hit": [], "failed": 0})
        baselines: dict = defaultdict(list)
        for rep in sorted(model_dir.glob("rep*")):
            with (rep / "manifest.csv").open() as handle:
                for row in csv.DictReader(handle):
                    entry = cache[(int(row["cache_mib"]), row["policy"])]
                    if row["status"] == "ok":
                        entry["tps"].append(float(row["decode_tps"]))
                        entry["hit"].append(float(row["cache_hit_rate_percent"]))
                    else:
                        entry["failed"] += 1
            for path in sorted((rep / "baselines").glob("*.json")):
                baselines[path.stem].append(json.loads(path.read_text())["decode_tps"])

        def stat(values: list[float]) -> dict:
            if not values:
                return {"n": 0}
            return {"n": len(values), "mean": round(statistics.fmean(values), 2), "min": round(min(values), 2), "max": round(max(values), 2)}

        summary[model_dir.name] = {
            "cache": {f"{mib}-{policy}": {"decode_tps": stat(v["tps"]), "hit_rate_percent": stat(v["hit"]), "failed_runs": v["failed"]} for (mib, policy), v in sorted(cache.items())},
            "baselines_decode_tps": {name: stat(values) for name, values in sorted(baselines.items())},
        }
    (args.root / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n")
    for model, data in summary.items():
        print(f"== {model}")
        for name, s in data["baselines_decode_tps"].items():
            print(f"  baseline {name:10s} {s}")
        for key, v in data["cache"].items():
            t, h = v["decode_tps"], v["hit_rate_percent"]
            print(f"  cache {key:12s} tps {t.get('mean')} [{t.get('min')}-{t.get('max')}] hit {h.get('mean')}% failed={v['failed_runs']}")


if __name__ == "__main__":
    main()
