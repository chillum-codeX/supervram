#!/usr/bin/env python3
"""One-line summary of a bench32k run: prefill, decode (overall and by output position), total time. Appends to results/overnight/bench/SUMMARY.tsv"""
import json, sys, os
f, name = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else os.path.basename(sys.argv[1]))
d = json.load(open(f)); s = d["step_ms"]; n = len(s); pre = sum(d["prefill_chunk_ms"]) / 1000
rate = lambda a, b: 1000 * (min(b, n) - a) / sum(s[a:min(b, n)]) if a < n else float("nan")
tot = pre + sum(s) / 1000
line = f"{name}\tprefill_s={pre:.1f}\tdecode_tps={1000*n/sum(s):.1f}\tn_out={n}\tby_pos(0-512,512-1024,1024-2048,2048-4096)={rate(0,512):.1f},{rate(512,1024):.1f},{rate(1024,2048):.1f},{rate(2048,4096):.1f}\ttotal_s={tot:.0f}"
print(line)
with open(os.path.join(os.path.dirname(f), "SUMMARY.tsv"), "a") as o: o.write(line + "\n")
