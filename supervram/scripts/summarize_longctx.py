#!/usr/bin/env python3
"""Summarize a long-context svram-verify JSON: prefill speed and decode speed over the output (per segment)."""
import json, sys
d = json.load(open(sys.argv[1])); c = d["prefill_chunk_ms"]; s = d["step_ms"]
pre = sum(c) / 1000
print(f"{sys.argv[1].split('/')[-1]}")
print(f"  prompt {d['prompt_tokens']} tokens: prefill {pre:.1f} s = {d['prompt_tokens'] / pre:.0f} tokens/s"
      f" (first 4k tokens {4096 / (sum(c[:max(1, int(4096 / (d['prompt_tokens'] / len(c))))]) / 1000):.0f} t/s ... last chunk {1000 * (d['prompt_tokens'] / len(c)) / c[-2]:.0f} t/s)")
n = len(s); print(f"  output {len(d['tokens'])} tokens: overall decode {1000 * n / sum(s):.1f} t/s; total wall for prompt+output {pre + sum(s) / 1000:.0f} s")
first = min(2048, n)
print(f"  first {first} output tokens (before greedy decoding degenerates into loops): {1000 * first / sum(s[:first]):.1f} t/s; prompt + {first} tokens = {pre + sum(s[:first]) / 1000:.0f} s")
segs = [(0, 512), (512, 1024), (1024, 2048), (2048, 3072), (3072, n)]
print("  decode by output position: " + " | ".join(f"{a}-{min(b, n)}: {1000 * (min(b, n) - a) / sum(s[a:min(b, n)]):.1f} t/s" for a, b in segs if a < n))
