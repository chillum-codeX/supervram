#!/usr/bin/env python3
"""Run svram-verify with the model evicted from the page cache and (optionally) a hard RAM cap.

Purpose: prove where the data lives (ElasticVRAM notes sec. 35). Records SSD bytes actually read,
peak process/cgroup memory, and how much of the model file is page-cache resident afterwards.
No root needed: eviction uses posix_fadvise(DONTNEED); the cap is a systemd --user scope with
MemoryMax and MemorySwapMax=0 (swap must not act as hidden capacity).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def disk_of(path: str) -> str:
    source = subprocess.check_output(["findmnt", "-no", "SOURCE", "--target", path], text=True).strip()
    return subprocess.check_output(["lsblk", "-no", "PKNAME", source], text=True).strip().splitlines()[0]


def sectors_read(disk: str) -> int:
    return int(Path(f"/sys/block/{disk}/stat").read_text().split()[2])


def resident_bytes(path: str) -> int:
    out = subprocess.check_output(["fincore", "-b", "-n", "-o", "RES", path], text=True)
    return int(out.split()[0])


def evict(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def status_kib(pid: int, key: str) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def cgroup_current(pid: int, require_scope: bool) -> int:
    """memory.current of the process's cgroup. With a RAM cap the process starts in its parent's cgroup and is moved
    into the systemd-run scope a moment later; samples taken before the move belong to the parent (the whole desktop
    session) and are ignored."""
    try:
        rel = Path(f"/proc/{pid}/cgroup").read_text().strip().split("::")[-1]
        if require_scope and "/run-" not in rel:
            return 0
        return int(Path(f"/sys/fs/cgroup{rel}/memory.current").read_text())
    except (OSError, ValueError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--svram-verify", default="svram-verify")
    parser.add_argument("--ram-cap", default="", help="e.g. 6G; empty = uncapped")
    parser.add_argument("--no-evict", action="store_true", help="leave the model in the page cache (warm run)")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("verify_args", nargs=argparse.REMAINDER, help="args for svram-verify after --")
    args = parser.parse_args()
    extra = args.verify_args[1:] if args.verify_args[:1] == ["--"] else args.verify_args

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.name}.verify.json"
    disk = disk_of(args.model)
    if not args.no_evict:
        evict(args.model)
    resident_before = resident_bytes(args.model)

    command = [args.svram_verify, "--model", args.model, "--json", str(json_path), *extra]
    if args.ram_cap:
        command = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={args.ram_cap}", "-p", "MemorySwapMax=0", "--", *command]

    peaks = {"rss_anon_kib": 0, "rss_file_kib": 0, "cgroup_bytes": 0}
    stop = threading.Event()

    def poll(pid: int) -> None:
        while not stop.is_set():
            peaks["rss_anon_kib"] = max(peaks["rss_anon_kib"], status_kib(pid, "RssAnon"))
            peaks["rss_file_kib"] = max(peaks["rss_file_kib"], status_kib(pid, "RssFile"))
            peaks["cgroup_bytes"] = max(peaks["cgroup_bytes"], cgroup_current(pid, bool(args.ram_cap)))
            time.sleep(0.25)

    read_before = sectors_read(disk)
    t0 = time.time()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    poller = threading.Thread(target=poll, args=(proc.pid,), daemon=True)
    poller.start()
    stdout, stderr = proc.communicate()
    stop.set()
    poller.join()
    wall = time.time() - t0
    read_after = sectors_read(disk)

    record = {
        "name": args.name,
        "evidence_class": "measured_rtx3090",
        "command": command,
        "returncode": proc.returncode,
        "disk": disk,
        "ram_cap": args.ram_cap or None,
        "evicted_before": not args.no_evict,
        "model_page_cache_resident_gib_before": round(resident_before / 2**30, 2),
        "model_page_cache_resident_gib_after": round(resident_bytes(args.model) / 2**30, 2),
        "ssd_read_gib": round((read_after - read_before) * 512 / 2**30, 2),
        "wall_s": round(wall, 1),
        "peak_rss_anon_gib": round(peaks["rss_anon_kib"] / 2**20, 2),
        "peak_rss_file_gib": round(peaks["rss_file_kib"] / 2**20, 2),
        "peak_cgroup_gib": round(peaks["cgroup_bytes"] / 2**30, 2),
        "stdout_tail": stdout.strip().splitlines()[:3],
        "cache_stats": [l for l in stdout.splitlines() if l.startswith(("cache_stats", "direct_io_stats"))],
        "env": {k: os.environ[k] for k in os.environ if k.startswith("SVRAM_")},
        "stderr_tail": stderr.strip().splitlines()[-4:] if proc.returncode else [],
    }
    if json_path.exists():
        data = json.loads(json_path.read_text())
        steps = data.get("step_ms", [])
        record["decode_tps_all"] = data.get("decode_tps")
        if len(steps) >= 64:
            tail = steps[len(steps) // 2:]
            record["decode_tps_second_half"] = round(1000.0 * len(tail) / sum(tail), 2)
        record["tokens_generated"] = len(data.get("tokens", []))
    (args.out_dir / f"{args.name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("command", "stdout_tail")}, indent=1))
    sys.exit(0 if proc.returncode == 0 else 1)


if __name__ == "__main__":
    main()
