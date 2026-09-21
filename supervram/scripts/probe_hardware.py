#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile


def run(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
        return {"available": True, "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def mount_info(target: Path) -> dict:
    best: tuple[int, dict] | None = None
    text = read("/proc/self/mountinfo") or ""
    target = target.resolve()
    for line in text.splitlines():
        left, _, right = line.partition(" - ")
        fields = left.split()
        if len(fields) < 6:
            continue
        mountpoint = Path(fields[4])
        try:
            target.relative_to(mountpoint)
        except ValueError:
            continue
        info = {"mountpoint": str(mountpoint), "mount_options": fields[5], "filesystem": right.split()[0] if right else None, "source": right.split()[1] if len(right.split()) > 1 else None}
        score = len(str(mountpoint))
        if best is None or score > best[0]:
            best = (score, info)
    return best[1] if best else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe SuperVRAM transfer-path capabilities")
    parser.add_argument("--path", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    nvidia = run(["nvidia-smi", "--query-gpu=index,name,pci.bus_id,driver_version,memory.total,bar1.memory.total", "--format=csv,noheader,nounits"])
    topology = run(["nvidia-smi", "topo", "-m"])
    cufile_lib = ctypes.util.find_library("cufile")
    cufile_cli = shutil.which("gdscheck") or shutil.which("gdscheck.py")
    iommu_groups = Path("/sys/kernel/iommu_groups")
    pci_devices = []
    for device in sorted(Path("/sys/bus/pci/devices").glob("*")):
        cls = read(str(device / "class"))
        if cls and (cls.startswith("0x0108") or cls.startswith("0x0300") or cls.startswith("0x0302")):
            pci_devices.append({
                "bdf": device.name,
                "class": cls,
                "vendor": read(str(device / "vendor")),
                "device": read(str(device / "device")),
                "numa_node": read(str(device / "numa_node")),
                "iommu_group": os.path.basename(os.path.realpath(device / "iommu_group")) if (device / "iommu_group").exists() else None,
            })

    mount = mount_info(args.path)
    fs_supported_hint = mount.get("filesystem") in {"ext4", "xfs"}
    direct_io_test = {"supported": False, "error": None}
    test_path = None
    if hasattr(os, "O_DIRECT"):
        try:
            fd_tmp, name = tempfile.mkstemp(prefix=".supervram-direct-io-probe-", dir=args.path)
            os.close(fd_tmp)
            test_path = Path(name)
            fd = os.open(test_path, os.O_WRONLY | os.O_DIRECT)
            os.close(fd)
            direct_io_test["supported"] = True
            direct_io_test["note"] = "O_DIRECT open succeeded; aligned data I/O was not validated"
        except OSError as exc:
            direct_io_test["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if test_path is not None:
                try:
                    test_path.unlink()
                except OSError:
                    pass

    gds_possible = bool(nvidia.get("available") and nvidia.get("returncode") == 0 and cufile_lib and fs_supported_hint)
    reasons = []
    if not nvidia.get("available") or nvidia.get("returncode") != 0:
        reasons.append("no NVIDIA GPU/driver visible")
    if not cufile_lib:
        reasons.append("libcufile not found")
    if not fs_supported_hint:
        reasons.append(f"filesystem {mount.get('filesystem')} is not an ext4/xfs GDS candidate")

    output = {
        "schema_version": 1,
        "evidence_class": "measured_environment_probe",
        "host": {"node": platform.node(), "kernel": platform.release(), "machine": platform.machine()},
        "target_path": str(args.path.resolve()),
        "mount": mount,
        "nvidia_smi": nvidia,
        "nvidia_topology": topology,
        "libcufile": cufile_lib,
        "gdscheck": cufile_cli,
        "iommu_enabled": iommu_groups.exists() and any(iommu_groups.iterdir()),
        "pci_devices": pci_devices,
        "direct_io_open_test": direct_io_test,
        "gds_candidate": gds_possible,
        "gds_rejection_reasons": reasons,
        "recommended_backend": "cufile" if gds_possible else ("pinned" if nvidia.get("returncode") == 0 else "pread"),
        "notes": [
            "A candidate result is not proof that cuFile device DMA works; run gdscheck and the transfer microbenchmark on the target host.",
            "Resizable BAR and BAR1 size do not independently establish GDS support.",
            "ACS/IOMMU and PCIe root-complex topology must be checked on the target host.",
        ],
    }
    rendered = json.dumps(output, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
