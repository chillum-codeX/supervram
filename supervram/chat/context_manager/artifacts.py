from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    path: str
    byte_count: int
    sha256: str
    created_at: float
    source: dict[str, Any]
    redacted: bool = False


class ArtifactStore:
    def __init__(self, root: str | Path, secret_patterns: list[str] | None = None):
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.meta = self.root / "meta"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.meta.mkdir(parents=True, exist_ok=True)
        self.secret_patterns = [re.compile(pattern) for pattern in (secret_patterns or [])]

    def sanitize_text(self, content: str) -> tuple[str, bool]:
        redacted = False
        sanitized = content
        for pattern in self.secret_patterns:
            sanitized, count = pattern.subn("[REDACTED_SECRET]", sanitized)
            redacted = redacted or count > 0
        return sanitized, redacted

    def put(self, content: str | bytes, kind: str, source: dict[str, Any] | None = None, redact: bool = True) -> ArtifactRef:
        if isinstance(content, bytes):
            data = content
            redacted = False
        else:
            sanitized, redacted = self.sanitize_text(content) if redact else (content, False)
            data = sanitized.encode("utf-8", errors="replace")
        digest = hashlib.sha256(data).hexdigest()
        object_path = self.objects / digest[:2] / digest[2:]
        object_path.parent.mkdir(parents=True, exist_ok=True)
        if not object_path.exists():
            temporary = object_path.with_name(object_path.name + f".tmp-{os.getpid()}")
            temporary.write_bytes(data)
            os.replace(temporary, object_path)
        artifact = ArtifactRef(
            artifact_id=digest,
            kind=kind,
            path=str(object_path),
            byte_count=len(data),
            sha256=digest,
            created_at=time.time(),
            source=source or {},
            redacted=redacted,
        )
        metadata_path = self.meta / f"{digest}.json"
        if not metadata_path.exists():
            metadata_path.write_text(json.dumps(asdict(artifact), indent=2) + "\n")
        return artifact

    def get(self, artifact_id: str) -> bytes:
        if not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            raise ValueError("invalid artifact id")
        path = self.objects / artifact_id[:2] / artifact_id[2:]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != artifact_id:
            raise IOError("artifact checksum mismatch")
        return data

    def get_text(self, artifact_id: str) -> str:
        return self.get(artifact_id).decode("utf-8", errors="replace")

    def inspect(self, artifact_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            raise ValueError("invalid artifact id")
        metadata = self.meta / f"{artifact_id}.json"
        result = json.loads(metadata.read_text())
        result["available"] = (self.objects / artifact_id[:2] / artifact_id[2:]).exists()
        return result

    def reference_text(self, artifact: ArtifactRef, summary: str = "") -> str:
        source = artifact.source
        provenance = source.get("path") or source.get("command") or source.get("tool") or "tool output"
        label = f"artifact:{artifact.artifact_id}"
        detail = f"{artifact.kind}, {artifact.byte_count} bytes, source={provenance}"
        if summary:
            return f"[{label} | {detail}]\n{summary}"
        return f"[{label} | {detail}]"
