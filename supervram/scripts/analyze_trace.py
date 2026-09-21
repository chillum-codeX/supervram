#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def percentile(values: list[int], p: float) -> int | None:
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * p)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SuperVRAM JSONL traces")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    counts = Counter()
    bytes_by_event = Counter()
    latency_by_event: dict[str, list[int]] = defaultdict(list)
    per_expert = Counter()
    with args.trace.open() as handle:
        for line in handle:
            item = json.loads(line)
            event = item["event"]
            counts[event] += 1
            if "bytes" in item:
                bytes_by_event[event] += int(item["bytes"])
            for key, value in item.items():
                if key.endswith("_ns") and key != "timestamp_ns":
                    latency_by_event[f"{event}.{key}"].append(int(value))
            if "layer" in item and "expert" in item:
                per_expert[(int(item["layer"]), int(item["expert"]))] += 1
    output = {
        "schema_version": 1,
        "evidence_class": "trace_aggregation",
        "event_counts": counts,
        "bytes_by_event": bytes_by_event,
        "latencies": {
            key: {"count": len(values), "p50_ns": percentile(values, 0.5), "p95_ns": percentile(values, 0.95), "p99_ns": percentile(values, 0.99), "sum_ns": sum(values)}
            for key, values in latency_by_event.items()
        },
        "per_expert_events": [{"layer": layer, "expert": expert, "events": count} for (layer, expert), count in per_expert.most_common()],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(output, indent=2, default=dict) + "\n")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["layer", "expert", "events"])
            writer.writeheader()
            writer.writerows(output["per_expert_events"])
    print(json.dumps({"event_counts": counts, "latencies": output["latencies"]}, indent=2, default=dict))


if __name__ == "__main__":
    main()
