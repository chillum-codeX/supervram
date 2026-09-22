#!/usr/bin/env python3
"""SuperVRAM live monitor: a small stdlib-only HTTP server that samples the GPU, RAM, SSD and the
running svram-verify process (via its --progress JSON-lines file) and serves a live dashboard.

No third-party dependencies. Run: python3 monitor/server.py [--port 8787]
Point any svram-verify / cold_run.py invocation's --progress at $SVRAM_LIVE_PROGRESS
(default /tmp/claude-1000/svram-live.jsonl) to show up here live.
"""
import http.server
import json
import os
import re
import socketserver
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROGRESS_PATH = Path(os.environ.get("SVRAM_LIVE_PROGRESS", "/tmp/claude-1000/svram-live.jsonl"))
SSD_DEVICE = os.environ.get("SVRAM_SSD_DEVICE", "nvme1n1")   # the WD drive that holds the models; never nvme0n1 (Intel, off limits)
SSD_CEILING_BPS = float(os.environ.get("SVRAM_SSD_CEILING_GBPS", "2.4")) * 1e9  # measured ceiling, scripts/ssd_expert_read_bench.py
SAMPLE_S = 0.6
HISTORY_LEN = 200  # ~2 minutes at 0.6s

state_lock = threading.Lock()
history = deque(maxlen=HISTORY_LEN)
latest = {}


def sh(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=2.0)
    except Exception:
        return ""


def read_gpu():
    out = sh(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
              "--format=csv,noheader,nounits"])
    line = out.strip().splitlines()[0] if out.strip() else ""
    if not line:
        return {"vram_used_mib": 0, "vram_total_mib": 24576, "util_pct": 0, "power_w": 0, "temp_c": 0, "ok": False}
    parts = [p.strip() for p in line.split(",")]
    try:
        return {
            "vram_used_mib": float(parts[0]), "vram_total_mib": float(parts[1]),
            "util_pct": float(parts[2]), "power_w": float(parts[3]), "temp_c": float(parts[4]), "ok": True,
        }
    except Exception:
        return {"vram_used_mib": 0, "vram_total_mib": 24576, "util_pct": 0, "power_w": 0, "temp_c": 0, "ok": False}


def read_ram():
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            info[k] = int(v.strip().split()[0]) * 1024  # kB -> bytes
    except Exception:
        pass
    total = info.get("MemTotal", 0)
    avail = info.get("MemAvailable", 0)
    return {"used_bytes": max(total - avail, 0), "total_bytes": total, "cached_bytes": info.get("Cached", 0)}


def find_svram_pid():
    out = sh(["pgrep", "-x", "svram-verify"])
    pids = [p for p in out.split() if p.isdigit()]
    return int(pids[0]) if pids else None


def read_rss(pid):
    try:
        return int(Path(f"/proc/{pid}/status").read_text().split("VmRSS:")[1].split()[0]) * 1024
    except Exception:
        return None


def read_cgroup(pid):
    """If svram-verify is running inside a systemd-run --scope (cold_run.py's RAM cap), that cgroup
    is process-specific and its memory.current/memory.max are the right numbers to show. Otherwise a
    cgroup covers the whole desktop session (every app, not just this run), so fall back to this
    process's own RSS instead of that much-too-broad number."""
    if pid is None:
        return None
    rss = read_rss(pid)
    try:
        rel = Path(f"/proc/{pid}/cgroup").read_text().strip().split("::")[-1]
        if "/run-" in rel:
            cur = int(Path(f"/sys/fs/cgroup{rel}/memory.current").read_text())
            max_raw = Path(f"/sys/fs/cgroup{rel}/memory.max").read_text().strip()
            cap = None if max_raw == "max" else int(max_raw)
            return {"capped": True, "current_bytes": cur, "max_bytes": cap}
    except Exception:
        pass
    return {"capped": False, "current_bytes": rss if rss is not None else 0, "max_bytes": None} if rss is not None else None


_diskstats_prev = {"t": 0.0, "read_sectors": 0, "write_sectors": 0}


def read_ssd_rate():
    now = time.time()
    try:
        line = next(l for l in Path("/proc/diskstats").read_text().splitlines() if f" {SSD_DEVICE} " in l)
        f = line.split()
        read_sectors, write_sectors = int(f[5]), int(f[9])
    except Exception:
        return {"read_bps": 0.0, "write_bps": 0.0, "device": SSD_DEVICE, "ok": False}
    dt = now - _diskstats_prev["t"]
    read_bps = write_bps = 0.0
    if _diskstats_prev["t"] and dt > 0:
        read_bps = max(0.0, (read_sectors - _diskstats_prev["read_sectors"]) * 512 / dt)
        write_bps = max(0.0, (write_sectors - _diskstats_prev["write_sectors"]) * 512 / dt)
    _diskstats_prev.update(t=now, read_sectors=read_sectors, write_sectors=write_sectors)
    return {"read_bps": read_bps, "write_bps": write_bps, "device": SSD_DEVICE, "ok": True}


_progress_pos = {"path": None, "offset": 0, "events": deque(maxlen=4000)}


def tail_progress():
    """Incrementally read new lines from the live progress file. Resets when the file shrinks
    (svram-verify truncates it at the start of every run)."""
    p = PROGRESS_PATH
    if not p.exists():
        return
    try:
        size = p.stat().st_size
        if _progress_pos["path"] != str(p) or size < _progress_pos["offset"]:
            _progress_pos["path"] = str(p)
            _progress_pos["offset"] = 0
            _progress_pos["events"].clear()
        with p.open("r") as f:
            f.seek(_progress_pos["offset"])
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    _progress_pos["events"].append(json.loads(line))
                except Exception:
                    continue
            _progress_pos["offset"] = f.tell()
    except Exception:
        pass


def run_summary():
    """Reduce the tailed event list to a single current-run snapshot."""
    events = list(_progress_pos["events"])
    if not events:
        return {"active": False}
    start = next((e for e in events if e.get("phase") == "start"), events[0])
    last = events[-1]
    prefill = [e for e in events if e.get("phase") == "prefill"]
    decode = [e for e in events if e.get("phase") == "decode"]
    warm = [e for e in events if e.get("phase") == "warm_done"]
    done = last.get("phase") == "done"
    phase = last.get("phase", "?")
    d10 = decode[-10:]
    return {
        "active": True,
        "done": done,
        "phase": phase,
        "config": {k: start.get(k) for k in ("model", "storage", "zerocopy", "bias", "bias_mul",
                                              "cache_mib", "direct_io", "n_ctx", "prompt_tokens", "n_predict")},
        "model_name": Path(start.get("model", "")).name,
        "warm_ms": warm[-1]["warm_ms"] if warm else None,
        "prefill_pos": prefill[-1]["pos"] if prefill else 0,
        "prefill_total": prefill[-1]["total"] if prefill else start.get("prompt_tokens", 0),
        "prefill_elapsed_s": prefill[-1]["elapsed_s"] if prefill else 0,
        "decode_step": decode[-1]["step"] + 1 if decode else 0,
        "decode_total": start.get("n_predict", 0),
        "tps_inst": decode[-1]["tps_inst"] if decode else 0,
        "tps_avg": decode[-1]["tps_avg"] if decode else 0,
        "tps_recent": sum(e["tps_inst"] for e in d10) / len(d10) if d10 else 0,
        "hit_rate": decode[-1].get("hit_rate") if decode else None,
        "hits": decode[-1].get("hits") if decode else None,
        "misses": decode[-1].get("misses") if decode else None,
        "evictions": decode[-1].get("evictions") if decode else None,
        "bytes_h2d": decode[-1].get("bytes_h2d") if decode else None,
        "bytes_ssd": decode[-1].get("bytes_ssd") if decode else None,
        "n_slots": decode[-1].get("n_slots") if decode else None,
        "step_history": [e["tps_inst"] for e in decode[-120:]],
        "final": last if done else None,
    }


def bottleneck(gpu, ssd, run):
    ssd_ratio = ssd["read_bps"] / SSD_CEILING_BPS if SSD_CEILING_BPS else 0
    phase = run.get("phase") if run.get("active") else None
    if not run.get("active") or phase in (None, "done"):
        return {"label": "IDLE", "detail": "no run in progress", "level": "idle"}
    if phase in ("start", "warm", "warm_done"):
        return {"label": "LOADING / WARM START", "detail": "filling VRAM slots before the prompt", "level": "info"}
    if phase == "prefill_done":
        return {"label": "PREFILL DONE", "detail": "about to start decoding", "level": "info"}
    if phase == "prefill":
        if ssd_ratio > 0.6:
            return {"label": "SSD READ", "detail": f"prefill streaming from SSD at {ssd_ratio*100:.0f}% of its measured ceiling", "level": "hot"}
        if gpu["util_pct"] > 80:
            return {"label": "GPU COMPUTE", "detail": "prefill is compute-bound (prompt batch mostly resident)", "level": "ok"}
        return {"label": "PREFILL", "detail": "reading prompt / copying resident experts", "level": "info"}
    if phase == "decode":
        hit_rate = run.get("hit_rate") or 0
        if ssd_ratio > 0.5:
            return {"label": "SSD READ", "detail": f"expert misses streaming from SSD at {ssd_ratio*100:.0f}% of ceiling", "level": "hot"}
        if hit_rate < 90 and gpu["util_pct"] < 70:
            return {"label": "HOST→GPU COPY (PCIe)", "detail": f"cache hit rate {hit_rate:.1f}%: misses are being copied from RAM", "level": "warn"}
        if gpu["util_pct"] > 80 and hit_rate >= 90:
            return {"label": "GPU COMPUTE", "detail": f"cache hit rate {hit_rate:.1f}%: mostly resident, GPU-bound (near best case)", "level": "ok"}
        return {"label": "BALANCED", "detail": f"cache hit rate {hit_rate:.1f}%, GPU util {gpu['util_pct']:.0f}%", "level": "info"}
    return {"label": phase.upper(), "detail": "", "level": "info"}


def sampler_loop():
    while True:
        gpu = read_gpu()
        ram = read_ram()
        ssd = read_ssd_rate()
        tail_progress()
        run = run_summary()
        pid = find_svram_pid()
        cgroup = read_cgroup(pid)
        bn = bottleneck(gpu, ssd, run)
        snapshot = {
            "ts": time.time(), "gpu": gpu, "ram": ram, "ssd": ssd,
            "cgroup": cgroup, "run": run, "bottleneck": bn, "svram_pid": pid,
        }
        with state_lock:
            latest.clear()
            latest.update(snapshot)
            history.append({
                "ts": snapshot["ts"], "vram_used_mib": gpu["vram_used_mib"], "gpu_util": gpu["util_pct"],
                "ssd_read_bps": ssd["read_bps"], "ram_used_bytes": (cgroup or {}).get("current_bytes", ram["used_bytes"]),
                "tps_inst": run.get("tps_inst", 0) if run.get("active") else 0,
            })
        time.sleep(SAMPLE_S)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/api/stats"):
            with state_lock:
                payload = dict(latest)
                payload["history"] = list(history)
                payload["ssd_ceiling_bps"] = SSD_CEILING_BPS
                payload["progress_path"] = str(PROGRESS_PATH)
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    t = threading.Thread(target=sampler_loop, daemon=True)
    t.start()
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("0.0.0.0", args.port), Handler) as httpd:
        print(f"SuperVRAM monitor on http://localhost:{args.port}  (watching {PROGRESS_PATH})")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
