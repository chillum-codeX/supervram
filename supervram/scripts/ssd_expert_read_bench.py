#!/usr/bin/env python3
"""Measure what the NVMe can deliver for expert-sized O_DIRECT reads at various queue depths.

Targets may be files or raw block devices (e.g. /dev/nvme1n1, read-only; needs read permission). With several targets the
threads are spread round-robin over them, so the aggregate shows whether drives add up (or share a PCIe/DMI bottleneck).

Reads random expert-sized chunks (default 1,671,168 B = one Q8_0 Qwen3-30B-A3B gate/up/down expert slice)
from a GGUF with O_DIRECT (bypasses the page cache), aligned to 4 KiB. Gives the SSD ceiling for
"SSD off the critical path" and the resulting tokens/s bound for a given miss volume per token.
"""
import argparse, json, mmap, os, random, threading, time

def worker(fd, offsets, size, buf, lat):
    for off in offsets:
        a = off & ~4095
        n = ((off + size + 4095) & ~4095) - a
        t = time.perf_counter()
        os.preadv(fd, [memoryview(buf)[:n]], a)
        lat.append(time.perf_counter() - t)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", nargs="+", help="file(s) or block device(s) to read"); p.add_argument("--chunk", type=int, default=1671168)
    p.add_argument("--reads", type=int, default=400, help="reads per thread")
    p.add_argument("--depths", default="1,2,4,8,16,32"); p.add_argument("--json")
    a = p.parse_args()
    fds = [os.open(m, os.O_RDONLY | os.O_DIRECT) for m in a.models]
    sizes = [os.lseek(fd, 0, os.SEEK_END) for fd in fds]
    rng = random.Random(1); out = []
    for depth in [int(x) for x in a.depths.split(",")]:
        bufs = [mmap.mmap(-1, a.chunk + 8192) for _ in range(depth)]
        offs = [[rng.randrange(0, sizes[i % len(fds)] - a.chunk - 8192) for _ in range(a.reads)] for i in range(depth)]
        lats = [[] for _ in range(depth)]
        ts = [threading.Thread(target=worker, args=(fds[i % len(fds)], offs[i], a.chunk, bufs[i], lats[i])) for i in range(depth)]
        t0 = time.perf_counter(); [t.start() for t in ts]; [t.join() for t in ts]; dt = time.perf_counter() - t0
        allv = sorted(x for l in lats for x in l); n = len(allv)
        r = {"queue_depth": depth, "GB_per_s": round(depth * a.reads * a.chunk / dt / 1e9, 3), "read_latency_ms_p50": round(allv[n // 2] * 1e3, 2), "read_latency_ms_p99": round(allv[int(n * .99)] * 1e3, 2)}
        out.append(r); print(r, flush=True)
    if a.json: json.dump({"targets": a.models, "chunk_bytes": a.chunk, "odirect": True, "results": out}, open(a.json, "w"), indent=2)

main()
