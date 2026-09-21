#!/usr/bin/env python3
"""Replay real expert-routing traces (SVRAM_TRACE) against cache policies, including the offline optimum.

Trace line: "<layer> <distinct expert ids used by that layer for one token>". One token = layers 0..47 in order.
Each layer has its own slot pool (like the implementation: gate/up/down move together, so one pool per layer).
Hit rate counts distinct experts per (token, layer), same as the runtime stats.

Policies: lru, lfu (as implemented: count resets on insert), lfu-decay, slru, belady-layer (optimal with the same
per-layer capacity = upper bound for any policy on this partitioning), belady-global (optimal with one shared pool
over all layers = upper bound if slots could move between layers), pinned+lru (hot experts profiled on other
prompts stay resident).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_trace(path: Path):
    """-> list[token] of list[(layer, tuple(experts))]"""
    tokens, cur, last = [], [], -1
    for line in path.read_text().splitlines():
        parts = line.split()
        layer = int(parts[0])
        if layer <= last and cur:
            tokens.append(cur)
            cur = []
        cur.append((layer, tuple(int(x) for x in parts[1:])))
        last = layer
    if cur:
        tokens.append(cur)
    return tokens


def per_layer(tokens):
    seq = defaultdict(list)
    for tok in tokens:
        for layer, ex in tok:
            seq[layer].append(ex)
    return seq


class Result:
    def __init__(self):
        self.hits = self.total = 0

    def rate(self):
        return 100.0 * self.hits / max(1, self.total)


def sim_lru(seq, cap):
    r = Result(); last = {}
    for t, ex in enumerate(seq, 1):
        need = set(ex); r.total += len(ex)
        for e in ex:
            if e in last:
                r.hits += 1; last[e] = t
        for e in sorted(need):
            if e in last:
                continue
            if len(last) >= cap:
                v = min((k for k in last if k not in need), key=last.get)
                del last[v]
            last[e] = t
    return r


def sim_lfu(seq, cap, decay_every=0):
    r = Result(); cnt = {}; last = {}
    for t, ex in enumerate(seq, 1):
        if decay_every and t % decay_every == 0:
            for k in cnt:
                cnt[k] //= 2
        need = set(ex); r.total += len(ex)
        for e in ex:
            if e in cnt:
                r.hits += 1; cnt[e] += 1; last[e] = t
        for e in sorted(need):
            if e in cnt:
                continue
            if len(cnt) >= cap:
                v = min((k for k in cnt if k not in need), key=lambda k: (cnt[k], last[k]))
                del cnt[v]; del last[v]
            cnt[e] = 1; last[e] = t
    return r


def sim_slru(seq, cap, protected_frac=0.5):
    """segmented LRU: new -> probation; a hit promotes to protected; evict from probation first."""
    r = Result(); prot, prob = {}, {}; pcap = max(1, int(cap * protected_frac))
    for t, ex in enumerate(seq, 1):
        need = set(ex); r.total += len(ex)
        for e in ex:
            if e in prob:
                r.hits += 1; del prob[e]; prot[e] = t
            elif e in prot:
                r.hits += 1; prot[e] = t
        while len(prot) > pcap:  # demote least-recent protected into probation
            v = min(prot, key=prot.get); prob[v] = prot.pop(v)
        for e in sorted(need):
            if e in prob or e in prot:
                continue
            while len(prob) + len(prot) >= cap:
                cand = [k for k in prob if k not in need] or [k for k in prot if k not in need]
                pool = prob if cand and cand[0] in prob else prot
                v = min(cand, key=lambda k: pool[k]); del pool[v]
            prob[e] = t
    return r


def next_uses(seqs):
    """for each (key) the sorted list of times it is used; keys are hashables"""
    uses = defaultdict(list)
    for t, ex in enumerate(seqs):
        for e in ex:
            uses[e].append(t)
    return uses


def sim_belady_layer(seq, cap):
    import bisect
    r = Result(); uses = next_uses(seq); res = set()
    for t, ex in enumerate(seq):
        need = set(ex); r.total += len(ex)
        for e in ex:
            if e in res:
                r.hits += 1
        for e in sorted(need):
            if e in res:
                continue
            if len(res) >= cap:
                def nxt(k):
                    u = uses[k]; i = bisect.bisect_right(u, t)
                    return u[i] if i < len(u) else 1 << 60
                v = max((k for k in res if k not in need), key=nxt); res.remove(v)
            res.add(e)
    return r


def sim_belady_global(tokens, cap_total):
    import bisect
    r = Result(); uses = defaultdict(list); order = 0
    stamps = []
    for tok in tokens:
        for layer, ex in tok:
            for e in ex:
                uses[(layer, e)].append(order)
            stamps.append(order); order += 1
    res = set(); order = 0
    for tok in tokens:
        for layer, ex in tok:
            need = {(layer, e) for e in ex}; r.total += len(ex)
            for k in need:
                if k in res:
                    r.hits += 1
            for k in sorted(need):
                if k in res:
                    continue
                if len(res) >= cap_total:
                    def nxt(x):
                        u = uses[x]; i = bisect.bisect_right(u, order)
                        return u[i] if i < len(u) else 1 << 60
                    v = max((x for x in res if x not in need), key=nxt); res.remove(v)
                res.add(k)
            order += 1
    return r


def sim_pinned_lru(seq, cap, hot):
    """`hot` experts stay resident (profiled elsewhere); the remaining slots run LRU"""
    hot = list(hot)[: cap // 2]
    r = Result(); rest = cap - len(hot); last = {}
    hotset = set(hot)
    for t, ex in enumerate(seq, 1):
        need = set(ex); r.total += len(ex)
        for e in ex:
            if e in hotset:
                r.hits += 1
            elif e in last:
                r.hits += 1; last[e] = t
        for e in sorted(need - hotset):
            if e in last:
                continue
            if len(last) >= rest:
                v = min((k for k in last if k not in need), key=last.get); del last[v]
            last[e] = t
    return r


def amortization(traces, cap):
    """Misses per token when k tokens share one expert load (LRU, per-layer pool, cold start each session).
    'chain': k consecutive tokens of one sequence verified together (speculative decoding, all accepted).
    'batch': k independent requests decoded in lock-step (union of their experts per layer)."""
    out = {}
    for k in (1, 2, 4, 8):
        res = {}
        for mode in ("chain", "batch"):
            groups = []  # list of per-layer-union lists
            if mode == "chain":
                for tokens in traces:
                    for i in range(0, len(tokens) - k + 1, k):
                        groups.append(tokens[i:i + k])
            else:
                for i in range(0, len(traces), k):
                    sub = traces[i:i + k]
                    if len(sub) < k:
                        continue
                    n = min(len(t) for t in sub)
                    for j in range(n):
                        groups.append([t[j] for t in sub])
            if not groups:
                continue
            # one continuous session per mode: shared cache across groups
            seqs = defaultdict(list)
            n_tok = 0
            for g in groups:
                n_tok += len(g)
                for layer in range(len(g[0])):
                    u = set()
                    for tok in g:
                        u.update(tok[layer][1])
                    seqs[g[0][layer][0]].append(tuple(sorted(u)))
            miss = tot = 0
            for seq in seqs.values():
                r = sim_lru(seq, cap); miss += r.total - r.hits; tot += r.total
            res[mode] = {"misses_per_token": round(miss / n_tok, 2), "hit_rate": round(100.0 * (tot - miss) / tot, 2)}
        out[k] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--caps", default="40,54,71,89", help="slots per layer (71 = 16 GiB on Q8_0)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    caps = [int(c) for c in args.caps.split(",")]
    traces = [load_trace(p) for p in args.traces]
    layers = sorted({l for tok in traces[0] for l, _ in tok})
    n_layers = len(layers)
    half = len(traces) // 2
    # popularity profile from the first half of the traces, evaluated on the second half
    pop = defaultdict(lambda: defaultdict(int))
    for tokens in traces[:half]:
        for layer, seq in per_layer(tokens).items():
            for ex in seq:
                for e in ex:
                    pop[layer][e] += 1
    out = {}
    print(f"{len(traces)} traces, {sum(len(t) for t in traces)} tokens, {n_layers} layers")
    for cap in caps:
        row = defaultdict(lambda: [0, 0])
        def add(name, r):
            row[name][0] += r.hits; row[name][1] += r.total
        for tokens in traces:
            seqs = per_layer(tokens)
            for layer, seq in seqs.items():
                add("lru", sim_lru(seq, cap)); add("lfu", sim_lfu(seq, cap)); add("lfu-decay", sim_lfu(seq, cap, 64))
                add("slru", sim_slru(seq, cap)); add("belady-layer", sim_belady_layer(seq, cap))
            add("belady-global", sim_belady_global(tokens, cap * n_layers))
        for tokens in traces[half:]:
            for layer, seq in per_layer(tokens).items():
                hot = [e for e, _ in sorted(pop[layer].items(), key=lambda kv: -kv[1])]
                add("pinned+lru (profiled on other prompts)", sim_pinned_lru(seq, cap, hot))
        out[cap] = {k: round(100.0 * v[0] / v[1], 2) for k, v in row.items()}
        print(f"\nslots/layer={cap} ({cap / 128:.0%} of experts)")
        for k, v in out[cap].items():
            print(f"  {k:42s} hit {v:6.2f}%  miss {100 - v:5.2f}%")
    # steady state: all traces concatenated as one long session
    long = [tok for tokens in traces for tok in tokens]
    cap = 71 if 71 in caps else caps[len(caps) // 2]
    seqs = per_layer(long)
    sess = {}
    for name, fn in (("lru", lambda s: sim_lru(s, cap)), ("lfu", lambda s: sim_lfu(s, cap)), ("lfu-decay", lambda s: sim_lfu(s, cap, 64)),
                     ("slru", lambda s: sim_slru(s, cap)), ("belady-layer", lambda s: sim_belady_layer(s, cap))):
        h = t = 0
        for seq in seqs.values():
            r = fn(seq); h += r.hits; t += r.total
        sess[name] = round(100.0 * h / t, 2)
    print(f"\ncontinuous session (all prompts back to back), slots/layer={cap}:", sess)
    out["continuous_session"] = {"cap": cap, **sess}
    am = amortization(traces, cap)
    print(f"\nmisses per token at slots/layer={cap} when k tokens share one load (Q8_0 expert ~5.0 MB; 2.4 GB/s SSD ceiling):")
    for k, res in am.items():
        line = "  k=%d " % k
        for mode, v in res.items():
            mb = v["misses_per_token"] * 5.01
            line += f"| {mode}: {v['misses_per_token']:5.1f} misses/token = {mb:5.0f} MB/token -> <= {2400 / mb:5.1f} t/s (hit {v['hit_rate']}%) "
        print(line)
    out["amortization"] = {"cap": cap, **{str(k): v for k, v in am.items()}}
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
