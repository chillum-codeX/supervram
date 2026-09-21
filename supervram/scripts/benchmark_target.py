#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import threading
import time
import urllib.request


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


class Monitor:
    def __init__(self, interval: float):
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples: list[dict] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            stamp = time.time_ns()
            sample = {"timestamp_ns": stamp}
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader,nounits"],
                    text=True, capture_output=True, timeout=3, check=False,
                )
                if out.returncode == 0:
                    values = [x.strip() for x in out.stdout.split(",")]
                    sample["gpu"] = {
                        "utilization_percent": float(values[0]),
                        "memory_mib": float(values[1]),
                        "power_w": float(values[2]),
                        "pcie_generation": int(values[3]),
                        "pcie_width": int(values[4]),
                    }
            except Exception as exc:
                sample["gpu_error"] = str(exc)
            try:
                with open("/proc/meminfo") as handle:
                    wanted = {"MemAvailable", "Cached", "Dirty", "Writeback"}
                    sample["meminfo_kib"] = {line.split(":")[0]: int(line.split()[1]) for line in handle if line.split(":")[0] in wanted}
            except OSError as exc:
                sample["meminfo_error"] = str(exc)
            self.samples.append(sample)
            self.stop_event.wait(self.interval)


def post_json(url: str, body: dict) -> tuple[dict, float, float]:
    encoded = json.dumps(body).encode()
    request = urllib.request.Request(url, data=encoded, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first_byte = None
    with urllib.request.urlopen(request, timeout=600) as response:
        first_byte = time.perf_counter()
        data = response.read()
    end = time.perf_counter()
    return json.loads(data), (first_byte - start) * 1000, (end - start) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-host llama-server benchmark with resource sampling")
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--prompt", default="Explain heterogeneous memory in one concise paragraph.")
    parser.add_argument("--prompt-tokens", type=int, default=0)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--cold-cache", action="store_true")
    args = parser.parse_args()
    if args.cold_cache and os.geteuid() != 0:
        parser.error("--cold-cache requires root to drop Linux page cache")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.server, "-m", args.model, "--port", str(args.port), "--no-webui", *args.server_arg]
    log_path = args.output.with_suffix(".server.log")
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.time() + 600
            health_url = f"http://127.0.0.1:{args.port}/health"
            while time.time() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited with {process.returncode}; see {log_path}")
                try:
                    with urllib.request.urlopen(health_url, timeout=2) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(1)
            else:
                raise TimeoutError("server did not become healthy")

            monitor = Monitor(0.1)
            monitor.start()
            records = []
            total_runs = args.warmup + args.repetitions
            for iteration in range(total_runs):
                if args.cold_cache:
                    subprocess.run(["sync"], check=True)
                    Path("/proc/sys/vm/drop_caches").write_text("3\n")
                body = {"prompt": args.prompt, "n_predict": args.decode_tokens, "temperature": 0, "stream": False, "cache_prompt": False}
                response, response_first_byte_ms, elapsed_ms = post_json(f"http://127.0.0.1:{args.port}/completion", body)
                timings = response.get("timings", {})
                record = {
                    "iteration": iteration,
                    "warmup": iteration < args.warmup,
                    "ttft_proxy_ms": response_first_byte_ms,
                    "request_ms": elapsed_ms,
                    "prompt_tokens": timings.get("prompt_n"),
                    "prompt_ms": timings.get("prompt_ms"),
                    "prompt_tokens_s": timings.get("prompt_per_second"),
                    "decode_tokens": timings.get("predicted_n"),
                    "decode_ms": timings.get("predicted_ms"),
                    "decode_tokens_s": timings.get("predicted_per_second"),
                }
                if record["decode_tokens"] and record["decode_ms"]:
                    record["tpot_ms"] = record["decode_ms"] / record["decode_tokens"]
                records.append(record)
            monitor.stop()
        finally:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    measured = [record for record in records if not record["warmup"]]
    request_ms = [record["request_ms"] for record in measured]
    decode_tps = [record["decode_tokens_s"] for record in measured if record["decode_tokens_s"] is not None]
    tpot = [record["tpot_ms"] for record in measured if record.get("tpot_ms") is not None]
    powers = [sample["gpu"]["power_w"] for sample in monitor.samples if "gpu" in sample]
    output = {
        "schema_version": 1,
        "evidence_class": "host_measurement_unverified_target",
        "warning": "This harness does not prove RTX 3090/NVMe/GDS target identity; pair it with a validated hardware probe before reclassifying evidence.",
        "command": command,
        "records": records,
        "resource_samples": monitor.samples,
        "summary": {
            "request_ms_p50": percentile(request_ms, 0.50),
            "request_ms_p95": percentile(request_ms, 0.95),
            "request_ms_p99": percentile(request_ms, 0.99),
            "decode_tokens_s_mean": statistics.mean(decode_tps) if decode_tps else None,
            "tpot_ms_p50": percentile(tpot, 0.50),
            "tpot_ms_p95": percentile(tpot, 0.95),
            "mean_gpu_power_w": statistics.mean(powers) if powers else None,
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
