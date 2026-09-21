#!/usr/bin/env python3
"""Upper bound for prefetching: replay real routing traces through a timing model of the GPU layer pipeline and one SSD.

Model (calibrated on the measured Q8_0 run: ~9.3 ms/token non-I/O, ~2.1 GB/s read rate, ~5 MB per expert):
  - the GPU computes layers strictly in order, `c` ms per layer;
  - the SSD serves reads one expert-miss at a time, `r` ms each (FIFO);
  - a layer can start computing only when its missed experts have arrived;
  - baseline (no prefetch): the reads of layer j are issued when layer j-1 finishes;
  - prefetch with lookahead D: a fraction `recall` of layer j's misses is issued when layer j-D starts (D counted in
    layers, crossing token boundaries), together with `waste` extra reads per prefetched miss (wrong guesses cost SSD time).
Cache contents/misses come from per-layer LRU with `--slots` slots over the concatenated traces (a warm session).
An oracle (recall 1, no waste) bounds what any predictor could achieve.
"""
import argparse
from collections import defaultdict
from pathlib import Path


def load_tokens(path):
    tokens, cur, last = [], [], -1
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        layer = int(parts[0])
        if layer <= last and cur:
            tokens.append(cur); cur = []
        cur.append((layer, tuple(int(x) for x in parts[1:])))
        last = layer
    if cur:
        tokens.append(cur)
    return tokens


def miss_counts(tokens, slots):
    """per (token, layer): number of distinct experts missing in a per-layer LRU cache"""
    last = defaultdict(dict)
    out = []
    for t, tok in enumerate(tokens, 1):
        row = []
        for layer, ex in tok:
            cache = last[layer]
            need = set(ex)
            miss = [e for e in sorted(need) if e not in cache]
            for e in need:
                if e in cache:
                    cache[e] = t
            for e in miss:
                if len(cache) >= slots:
                    v = min((k for k in cache if k not in need), key=cache.get)
                    del cache[v]
                cache[e] = t
            row.append(len(miss))
        out.append(row)
    return out


def simulate(misses, c, r, D, recall, waste):
    flat = [m for row in misses for m in row]
    n = len(flat)
    start = [0.0] * n
    end = [0.0] * n
    io_free = 0.0
    prev_end = 0.0
    for j, m in enumerate(flat):
        if D == 0:
            pre = 0.0
        else:
            pre = m * recall
        dem = m - pre
        ready = prev_end
        if pre > 0:
            release = start[j - D] if j - D >= 0 else 0.0
            done = max(io_free, release) + (pre + pre * waste) * r
            io_free = done
            ready = max(ready, done)
        if dem > 0:
            done = max(io_free, prev_end) + dem * r
            io_free = done
            ready = max(ready, done)
        start[j] = ready
        end[j] = ready + c
        prev_end = end[j]
    return end[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--slots", type=int, default=71)
    ap.add_argument("--c", type=float, default=9.3 / 48, help="GPU ms per layer")
    ap.add_argument("--r", type=float, default=2.4, help="SSD ms per missed expert (3 tensors, ~5 MB)")
    args = ap.parse_args()
    tokens = [tok for p in args.traces for tok in load_tokens(p)]
    misses = miss_counts(tokens, args.slots)
    n_tok = len(tokens)
    mpt = sum(map(sum, misses)) / n_tok
    base = simulate(misses, args.c, args.r, 0, 0, 0)
    print(f"{n_tok} tokens, {mpt:.1f} missed experts/token, compute {args.c * 48:.1f} ms/token, SSD {mpt * args.r:.1f} ms/token")
    print(f"baseline (no prefetch): {1000 * n_tok / base:.1f} t/s   (lower bound on SSD-bound speed: {1000 / (mpt * args.r):.1f} t/s)\n")
    print("ORACLE (every miss known D layers ahead, no wasted reads) -> speedup over baseline")
    for D in (1, 2, 4, 8, 16, 32, 48, 96):
        t = simulate(misses, args.c, args.r, D, 1.0, 0.0)
        print(f"  D={D:3d}: {1000 * n_tok / t:5.1f} t/s   x{base / t:.3f}")
    print("\nREALISTIC predictors: D=8 layers ahead; recall = share of misses fetched early; waste = extra wrong reads per useful one")
    for recall in (0.3, 0.5, 0.8):
        row = []
        for waste in (0.0, 0.5, 1.0, 2.0):
            t = simulate(misses, args.c, args.r, 8, recall, waste)
            row.append(f"waste {waste:.1f}: x{base / t:.3f}")
        print(f"  recall {recall:.1f} | " + " | ".join(row))


if __name__ == "__main__":
    main()
