#!/usr/bin/env python3
"""Compare bias B against bias 0 on the same forced human text: mean log-prob of the true tokens, perplexity ratio, top-1 agreement."""
import json, math, sys
out, b = sys.argv[1], sys.argv[2]
ref = json.load(open(f"{out}/bias-0.json")); cur = json.load(open(f"{out}/bias-{b}.json"))
n = min(len(ref["chosen_logprobs"]), len(cur["chosen_logprobs"]))
lr = sum(ref["chosen_logprobs"][:n]) / n; lc = sum(cur["chosen_logprobs"][:n]) / n
agree = sum(1 for i in range(n) if ref["argmax_tokens"][i] == cur["argmax_tokens"][i]) / n
truth = sum(1 for i in range(n) if cur["argmax_tokens"][i] == ref["tokens"][i]) / n
stats = [l for l in open(f"{out}/bias-{b}.out") if l.startswith("cache_stats")]
hr = stats[0].split("hit_rate=")[1].split()[0] if stats else "?"
s = cur["step_ms"]; tps = 1000 * len(s) / sum(s)
line = f"bias {b}: mean logprob of the true text {lc:.4f} (bias 0: {lr:.4f}), perplexity x{math.exp(lr - lc):.4f}, top-1 agreement with bias 0 {100*agree:.1f}%, matches the true next token {100*truth:.1f}% (bias 0: {100*sum(1 for i in range(n) if ref['argmax_tokens'][i]==ref['tokens'][i])/n:.1f}%), cache hit rate {hr}%, decode {tps:.1f} t/s"
print(line); open(f"{out}/SUMMARY.txt", "a").write(line + "\n")
