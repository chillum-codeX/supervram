#!/usr/bin/env python3
"""Build a warm-start usage profile ("<layer> <expert> <weight>" per line, weight = share of tokens that used the expert)
from routing traces recorded with --trace / SVRAM_TRACE. Usage: make_warm_profile.py OUT trace1 [trace2 ...]"""
import sys
from collections import defaultdict
out, traces = sys.argv[1], sys.argv[2:]
count = defaultdict(int); tokens = defaultdict(int)
for t in traces:
    for line in open(t):
        p = line.split()
        if not p: continue
        layer = int(p[0]); tokens[layer] += 1
        for e in p[1:]: count[(layer, int(e))] += 1
with open(out, "w") as f:
    for (layer, e), c in sorted(count.items()): f.write(f"{layer} {e} {c / tokens[layer]:.5f}\n")
print(f"wrote {len(count)} (layer, expert) entries from {sum(tokens.values()) // max(1, len(tokens))} tokens per layer to {out}")
