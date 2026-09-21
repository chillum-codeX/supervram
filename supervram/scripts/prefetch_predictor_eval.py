#!/usr/bin/env python3
"""How well can routing history predict *cache misses* D layers ahead? (input for scripts/simulate_prefetch.py)

Predictor: online cross-layer co-occurrence. For layer L and lookahead D, count how often expert e is chosen at layer L given
that expert a was chosen at layer L-D for the same token; score(e) = sum over the experts chosen at L-D of P(e | a). The
top-K non-resident experts are "prefetched". Reports recall of the real misses and wrong reads per useful read (`waste`).
Cache: per-layer LRU with --slots slots over the concatenated traces (a warm session); counts are learned online.
"""
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path


def load_tokens(path):
    tokens, cur, last = [], [], -1
    for line in Path(path).read_text().splitlines():
        parts = line.split(); layer = int(parts[0])
        if layer <= last and cur:
            tokens.append(cur); cur = []
        cur.append((layer, [int(x) for x in parts[1:]])); last = layer
    if cur:
        tokens.append(cur)
    return tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+"); ap.add_argument("--slots", type=int, default=71)
    ap.add_argument("--depths", default="1,4,8"); ap.add_argument("--ks", default="1,2,3,4")
    a = ap.parse_args()
    tokens = [t for p in a.traces for t in load_tokens(p)]
    n_layers = len(tokens[0]); E = 128
    for D in [int(x) for x in a.depths.split(",")]:
        C = np.zeros((n_layers, E, E), dtype=np.float32); N = np.zeros((n_layers, E), dtype=np.float32)
        freq = np.zeros((n_layers, E), dtype=np.float32)
        cache = defaultdict(dict)
        conf = []                                                           # (top-1 score, was it a real miss)
        total_misses = 0
        stats = {k: [0, 0, 0] for k in [int(x) for x in a.ks.split(",")]}   # useful, wrong, total misses
        base_stats = {k: [0, 0] for k in stats}                             # frequency-prior baseline: useful, wrong
        for t, tok in enumerate(tokens, 1):
            chosen = {layer: ex for layer, ex in tok}
            for layer, ex in tok:
                resident = cache[layer]
                need = set(ex)
                misses = [e for e in need if e not in resident]
                if layer >= D and t > 200:   # skip the learning phase
                    src = chosen[layer - D]
                    score = np.zeros(E, dtype=np.float32)
                    for s in src:
                        score += C[layer, s] / max(N[layer, s], 1.0)
                    res_mask = np.zeros(E, dtype=bool); res_mask[list(resident)] = True
                    sc = np.where(res_mask, -1e9, score); fr = np.where(res_mask, -1e9, freq[layer])
                    total_misses += len(misses)
                    j = int(np.argmax(sc)); conf.append((float(sc[j]), j in set(misses)))
                    for k in stats:
                        top = set(np.argsort(-sc)[:k].tolist()); topf = set(np.argsort(-fr)[:k].tolist())
                        ms = set(misses)
                        stats[k][0] += len(top & ms); stats[k][1] += len(top - ms); stats[k][2] += len(ms)
                        base_stats[k][0] += len(topf & ms); base_stats[k][1] += len(topf - ms)
                for e in need:
                    if e in resident:
                        resident[e] = t
                for e in sorted(misses):
                    if len(resident) >= a.slots:
                        v = min((x for x in resident if x not in need), key=resident.get); del resident[v]
                    resident[e] = t
            for layer, ex in tok:
                freq[layer, ex] += 1
                if layer >= D:
                    for s in chosen[layer - D]:
                        N[layer, s] += 1
                        C[layer, s, ex] += 1
        print(f"lookahead D={D} layers")
        conf.sort(key=lambda x: -x[0]); useful = wrong = 0; marks = {}
        for score, ok in conf:
            useful += ok; wrong += (not ok)
            for w in (0.1, 0.3, 0.5, 1.0):
                if useful and wrong / useful <= w:
                    marks[w] = (useful, wrong, score)
        n = len(conf)
        print("  precision of the most confident top-1 predictions: " + " | ".join(
            f"top {100 * f:g}%: {100 * sum(ok for _, ok in conf[:max(1, int(n * f))]) / max(1, int(n * f)):4.1f}% right" for f in (0.001, 0.01, 0.05, 0.2, 1.0)))
        line = "  confidence-thresholded top-1 (fire only when confident): "
        for w in (0.1, 0.3, 0.5, 1.0):
            u, wr, sc0 = marks.get(w, (0, 0, 0.0))
            line += f"| waste<={w}: recall {100 * u / max(total_misses, 1):4.1f}% "
        print(line)
        for k, (u, w, m) in stats.items():
            bu, bw = base_stats[k]
            print(f"  top-{k}: recall of misses {100 * u / max(m, 1):5.1f}%  wrong reads per useful read {w / max(u, 1):5.1f}   | frequency-only baseline: recall {100 * bu / max(m, 1):5.1f}%  waste {bw / max(bu, 1):5.1f}")


if __name__ == "__main__":
    main()
