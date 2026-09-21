#!/usr/bin/env python3
"""Teacher-forced quality comparison: how well does a cheaper model track the reference model's tokens?

Reference run (free-running greedy, e.g. Q8_0): results/.../<ref>-p<i>.json
Candidate run (--force-tokens = reference tokens): results/.../<cand>-on-<ref>-p<i>.json
Reports per prompt and overall: top-1 agreement (candidate's argmax == reference's token), the mean
log-probability each model gives the reference's tokens, and the resulting gap (mean NLL increase).
"""
import argparse, json, math
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("dir", type=Path); ap.add_argument("--ref", default="q8"); ap.add_argument("--cand", default="q4")
ap.add_argument("--json", type=Path)
a = ap.parse_args()
rows = []; tot = {"n": 0, "agree": 0, "ref_lp": 0.0, "cand_lp": 0.0}
for i in range(1, 100):
    r = a.dir / f"{a.ref}-p{i}.json"; c = a.dir / f"{a.cand}-on-{a.ref}-p{i}.json"
    if not (r.exists() and c.exists()):
        break
    R, C = json.load(open(r)), json.load(open(c))
    n = min(len(R["tokens"]), len(C["tokens"]))
    assert R["tokens"][:n] == C["tokens"][:n], "candidate was not fed the reference tokens"
    agree = sum(1 for k in range(n) if C["argmax_tokens"][k] == R["tokens"][k])
    ref_lp = sum(R["chosen_logprobs"][:n]) / n; cand_lp = sum(C["chosen_logprobs"][:n]) / n
    rows.append({"prompt": i, "tokens": n, "top1_agreement_pct": round(100.0 * agree / n, 2),
                 "ref_mean_logprob": round(ref_lp, 4), "cand_mean_logprob": round(cand_lp, 4), "nll_gap": round(ref_lp - cand_lp, 4)})
    tot["n"] += n; tot["agree"] += agree; tot["ref_lp"] += ref_lp * n; tot["cand_lp"] += cand_lp * n
    print(rows[-1])
o = {"top1_agreement_pct": round(100.0 * tot["agree"] / tot["n"], 2), "tokens": tot["n"],
     "ref_mean_logprob": round(tot["ref_lp"] / tot["n"], 4), "cand_mean_logprob": round(tot["cand_lp"] / tot["n"], 4)}
o["nll_gap"] = round(o["ref_mean_logprob"] - o["cand_mean_logprob"], 4)
o["perplexity_ratio_cand_over_ref"] = round(math.exp(o["nll_gap"]), 4)
print("\nOVERALL", o)
if a.json: json.dump({"per_prompt": rows, "overall": o}, open(a.json, "w"), indent=2)
