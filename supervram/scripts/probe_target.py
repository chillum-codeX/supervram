#!/usr/bin/env python3
"""Target-host probe for the RTX 3090 protocol (docs/RTX3090_PROTOCOL.md, section 1).

Aggregates the generic capability probe (probe_hardware.py), NVIDIA topology/BAR1/PCIe state,
IOMMU groups, storage mount options, a short NVMe read bandwidth test and the real cuFile probe
into one JSON document with evidence_class "measured_rtx3090".
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def run(command: list[str], timeout: int = 120) -> dict:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
        return {"available": True, "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}


def nvidia_query(fields: str) -> dict:
    out = run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if out.get("returncode") == 0:
        values = [v.strip() for v in out["stdout"].split(",")]
        return dict(zip(fields.split(","), values))
    return {"error": out}


def read_bandwidth(path: Path, size_mib: int) -> dict:
    """Sequential O_DIRECT read of a freshly written file; falls back to buffered read."""
    result: dict = {"size_mib": size_mib}
    fd, name = tempfile.mkstemp(prefix=".supervram-bw-", dir=path)
    os.close(fd)
    tmp = Path(name)
    try:
        chunk = os.urandom(1 << 20)
        with tmp.open("wb") as handle:
            for _ in range(size_mib):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if shutil.which("fio"):
            fio = run([
                "fio", "--name=svram", f"--filename={tmp}", "--rw=read", "--bs=1M", "--direct=1", "--ioengine=libaio",
                "--iodepth=16", "--numjobs=1", "--runtime=10", "--time_based=0", "--output-format=json",
            ], timeout=120)
            if fio.get("returncode") == 0:
                data = json.loads(fio["stdout"])
                job = data["jobs"][0]["read"]
                result["fio_seq_1m_qd16_mbps"] = job["bw"] / 1024.0
            fio_rand = run([
                "fio", "--name=svram-rand", f"--filename={tmp}", "--rw=randread", "--bs=4k", "--direct=1", "--ioengine=libaio",
                "--iodepth=32", "--numjobs=4", "--runtime=8", "--time_based", "--group_reporting", "--output-format=json",
            ], timeout=120)
            if fio_rand.get("returncode") == 0:
                data = json.loads(fio_rand["stdout"])
                job = data["jobs"][0]["read"]
                result["fio_rand_4k_qd32x4_iops"] = job["iops"]
                result["fio_rand_4k_mean_lat_us"] = job.get("lat_ns", {}).get("mean", 0) / 1000.0
            result["tool"] = "fio"
        # single-threaded O_DIRECT pread 4 MiB
        flags = os.O_RDONLY | getattr(os, "O_DIRECT", 0)
        try:
            rfd = os.open(tmp, flags)
            direct = True
        except OSError:
            rfd = os.open(tmp, os.O_RDONLY)
            direct = False
        try:
            import mmap
            buf = mmap.mmap(-1, 4 << 20)
            start = time.perf_counter()
            total = 0
            while True:
                n = os.readv(rfd, [buf])
                if n <= 0:
                    break
                total += n
            elapsed = time.perf_counter() - start
            result["pread_4m_single_thread_mbps"] = total / elapsed / (1 << 20)
            result["pread_o_direct"] = direct
        finally:
            os.close(rfd)
    finally:
        tmp.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="RTX 3090 target probe")
    parser.add_argument("--path", type=Path, required=True, help="directory on the NVMe that holds the models")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cufile-probe", type=Path, help="path to supervram_cufile_probe binary")
    parser.add_argument("--bw-size-mib", type=int, default=2048)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    generic = run([sys.executable, str(root / "scripts" / "probe_hardware.py"), "--path", str(args.path)])
    generic_json = json.loads(generic["stdout"]) if generic.get("returncode") == 0 else {"error": generic}

    gpu = nvidia_query("name,driver_version,memory.total,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,compute_cap,power.limit,clocks.max.sm,clocks.max.mem")
    bar1 = run(["bash", "-c", "nvidia-smi -q -d MEMORY | grep -A3 'BAR1' | grep Total | head -1"])
    gpu["bar1_total"] = bar1.get("stdout", "").split(":")[-1].strip() if bar1.get("returncode") == 0 else None
    topo = run(["nvidia-smi", "topo", "-m"])
    smi_q = run(["nvidia-smi", "-q", "-d", "MEMORY,PCIE"])
    lspci = run(["lspci", "-tv"])
    lspci_nvme = run(["bash", "-c", "lspci -nn | grep -i -E 'nvme|non-volatile|vga|3d controller'"])
    iommu_root = Path("/sys/kernel/iommu_groups")
    iommu = {"enabled": iommu_root.exists() and any(iommu_root.iterdir()), "groups": len(list(iommu_root.iterdir())) if iommu_root.exists() else 0}
    kernel_cmdline = Path("/proc/cmdline").read_text().strip() if Path("/proc/cmdline").exists() else None
    resizable_bar = None
    rebar = run(["bash", "-c", "nvidia-smi -q | grep -i -A2 'resizable' || true"])
    if rebar.get("returncode") == 0 and rebar["stdout"]:
        resizable_bar = rebar["stdout"]
    nvidia_fs = Path("/proc/driver/nvidia-fs").exists()
    gdscheck = shutil.which("gdscheck") or shutil.which("gdscheck.py")
    gdscheck_out = run([gdscheck, "-p"]) if gdscheck else {"available": False}
    lsblk = run(["lsblk", "-o", "NAME,MOUNTPOINT,FSTYPE,SIZE,ROTA,TRAN,MODEL", "-J"])
    cpu = run(["bash", "-c", "lscpu | grep -E 'Model name|^CPU\\(s\\)|NUMA|Thread|L3'"])
    mem = run(["bash", "-c", "grep -E 'MemTotal|MemAvailable|HugePages_Total' /proc/meminfo"])
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    thp = Path("/sys/kernel/mm/transparent_hugepage/enabled")

    bandwidth = read_bandwidth(args.path, args.bw_size_mib)

    cufile = None
    if args.cufile_probe and args.cufile_probe.exists():
        fd, name = tempfile.mkstemp(prefix=".supervram-cufile-", dir=args.path)
        os.close(fd)
        tmp = Path(name)
        try:
            with tmp.open("wb") as handle:
                chunk = os.urandom(1 << 20)
                for _ in range(1024):
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            out = run([str(args.cufile_probe), str(tmp), str(1024 << 20), str(4 << 20)], timeout=300)
            cufile = json.loads(out["stdout"]) if out.get("returncode") == 0 and out["stdout"] else {"error": out}
        finally:
            tmp.unlink(missing_ok=True)

    output = {
        "schema_version": 1,
        "evidence_class": "measured_rtx3090",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target_path": str(args.path.resolve()),
        "gpu": gpu,
        "nvidia_topology": topo.get("stdout"),
        "nvidia_smi_memory_pcie": smi_q.get("stdout"),
        "resizable_bar_raw": resizable_bar,
        "pci_tree": lspci.get("stdout"),
        "pci_nvme_gpu": lspci_nvme.get("stdout"),
        "iommu": iommu,
        "kernel_cmdline": kernel_cmdline,
        "nvidia_fs_kernel_module": nvidia_fs,
        "gdscheck": gdscheck_out,
        "block_devices": json.loads(lsblk["stdout"]) if lsblk.get("returncode") == 0 else lsblk,
        "cpu": cpu.get("stdout"),
        "memory": mem.get("stdout"),
        "cpu_governor": governor.read_text().strip() if governor.exists() else None,
        "transparent_hugepages": thp.read_text().strip() if thp.exists() else None,
        "storage_bandwidth": bandwidth,
        "cufile_probe": cufile,
        "generic_probe": generic_json,
        "notes": [
            "cuFile in compatibility mode bounces through host memory; it is not SSD-to-VRAM DMA.",
            "Bandwidth numbers are short single-file tests and may be affected by concurrent I/O.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    summary = {
        "gpu": gpu.get("name"),
        "driver": gpu.get("driver_version"),
        "bar1": gpu.get("bar1_total"),
        "pcie": f"gen{gpu.get('pcie.link.gen.current')} x{gpu.get('pcie.link.width.current')}",
        "nvidia_fs": nvidia_fs,
        "cufile_verdict": (cufile or {}).get("verdict"),
        "storage": bandwidth,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
