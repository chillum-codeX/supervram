from __future__ import annotations

import hashlib
import json
import mmap
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO, Iterable

from .types import ExpertKey, TensorExtent


MANIFEST_VERSION = 1


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & ~(alignment - 1)


class TensorStoreWriter:
    def __init__(self, path: str | Path, alignment: int = 4096):
        self.path = Path(path)
        self.alignment = alignment
        self._file: BinaryIO | None = None
        self._extents: list[TensorExtent] = []

    def __enter__(self) -> "TensorStoreWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("wb")
        return self

    def add(self, key: ExpertKey, blobs: Iterable[bytes], tensor_names: Iterable[str] | None = None) -> TensorExtent:
        if self._file is None:
            raise RuntimeError("store is not open")
        offset = align_up(self._file.tell(), self.alignment)
        if offset != self._file.tell():
            self._file.write(b"\0" * (offset - self._file.tell()))
        digest = hashlib.sha256()
        length = 0
        for blob in blobs:
            self._file.write(blob)
            digest.update(blob)
            length += len(blob)
        extent = TensorExtent(
            key=key,
            offset=offset,
            length=length,
            checksum=digest.hexdigest(),
            tensors=tuple(tensor_names or ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")),
        )
        self._extents.append(extent)
        return extent

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
        if exc_type is None:
            manifest = {
                "version": MANIFEST_VERSION,
                "alignment": self.alignment,
                "data_file": self.path.name,
                "extents": [
                    {
                        **asdict(extent),
                        "key": {"layer": extent.key.layer, "expert": extent.key.expert},
                    }
                    for extent in self._extents
                ],
            }
            self.path.with_suffix(self.path.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")


class TensorStore:
    def __init__(self, path: str | Path, use_mmap: bool = False, direct: bool = False):
        self.path = Path(path)
        manifest_path = self.path.with_suffix(self.path.suffix + ".json")
        raw = json.loads(manifest_path.read_text())
        if raw.get("version") != MANIFEST_VERSION:
            raise ValueError(f"unsupported tensor store version: {raw.get('version')}")
        self.alignment = int(raw["alignment"])
        self.extents = {
            ExpertKey(int(item["key"]["layer"]), int(item["key"]["expert"])): TensorExtent(
                key=ExpertKey(int(item["key"]["layer"]), int(item["key"]["expert"])),
                offset=int(item["offset"]),
                length=int(item["length"]),
                checksum=str(item.get("checksum", "")),
                tensors=tuple(item.get("tensors", ())),
            )
            for item in raw["extents"]
        }
        flags = os.O_RDONLY
        self.direct_requested = direct
        self.direct_active = False
        if direct and hasattr(os, "O_DIRECT"):
            try:
                self.fd = os.open(self.path, flags | os.O_DIRECT)
                self.direct_active = True
            except OSError:
                self.fd = os.open(self.path, flags)
        else:
            self.fd = os.open(self.path, flags)
        self._mmap = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ) if use_mmap and not self.direct_active else None
        self._fallback_fd: int | None = None
        self._fd_lock = threading.Lock()

    def read(self, key: ExpertKey, verify: bool = False) -> bytes:
        extent = self.extents[key]
        if self._mmap is not None:
            data = self._mmap[extent.offset : extent.offset + extent.length]
        else:
            fd = self._fallback_fd if self._fallback_fd is not None else self.fd
            try:
                data = os.pread(fd, extent.length, extent.offset)
            except OSError:
                if not self.direct_active:
                    raise
                with self._fd_lock:
                    if self._fallback_fd is None:
                        self._fallback_fd = os.open(self.path, os.O_RDONLY)
                    fd = self._fallback_fd
                data = os.pread(fd, extent.length, extent.offset)
        if len(data) != extent.length:
            raise IOError(f"short read for {key}: {len(data)} != {extent.length}")
        if verify and extent.checksum and hashlib.sha256(data).hexdigest() != extent.checksum:
            raise IOError(f"checksum mismatch for {key}")
        return data

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
        os.close(self.fd)
        if self._fallback_fd is not None:
            os.close(self._fallback_fd)

    def __enter__(self) -> "TensorStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
