# SuperVRAM live monitor

A small local dashboard that shows, while a benchmark runs: VRAM used (RTX 3090), RAM used (including the
actual `systemd-run` cap during a RAM-poor run), SSD read throughput against its measured ceiling, the
expert cache hit rate, and a "what is the bottleneck right now" indicator, updated twice a second.

No third-party dependencies (Python stdlib + a static HTML/canvas page); nothing is sent off this machine.

## Run it

```bash
python3 monitor/server.py --port 8787
```

or, inside this app, `preview_start` with name `svram-monitor` (configured in `../.claude/launch.json`).
Then open http://localhost:8787.

## Feed it a run

Any `svram-verify` (or `cold_run.py -- svram-verify ...`) invocation that adds:

```
--progress /tmp/claude-1000/svram-live.jsonl
```

shows up live: model/config, prefill and decode progress, tokens/s, cache hits/misses/evictions,
bytes moved from RAM vs from SSD, and warm-start timing. `svram-verify` truncates that file at the start
of every run, so starting a new run automatically replaces whatever was on screen. The path is
configurable with `SVRAM_LIVE_PROGRESS`; the SSD device to watch (`nvme1n1`, the WD drive that holds the
models — never `nvme0n1`, the Intel drive) and its measured ceiling are configurable with
`SVRAM_SSD_DEVICE` / `SVRAM_SSD_CEILING_GBPS`.

## How the bottleneck label is decided

Heuristic, from the live signals, not a profiler:
- **SSD READ** — SSD read rate is a large fraction of its measured ceiling (prefill: >60%, decode misses: >50%).
- **HOST→GPU COPY (PCIe)** — cache hit rate is low and the GPU isn't busy: time is going into copying misses from RAM.
- **GPU COMPUTE** — GPU utilization is high and the cache hit rate is already high: close to the best case for this cache size.
- **BALANCED / LOADING / IDLE** — none of the above dominates, warm start / model load is in progress, or nothing is running.

## What it reads

`nvidia-smi` (VRAM, utilization, power, temperature), `/proc/meminfo` and `/proc/<pid>/status` (RAM),
`/proc/diskstats` for the configured NVMe device (SSD read/write bytes/sec), and the `--progress`
JSON-lines file (per-token cache stats, straight from `ggml_backend_sched_expert_cache_stats` inside the
running process). If `cold_run.py` wrapped the run in a `systemd-run --scope` RAM cap, the monitor reads
that scope's `memory.current`/`memory.max` instead of process RSS, so the RAM panel shows the real cap
and how close the run is to hitting it.
