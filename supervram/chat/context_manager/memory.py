from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import time
from typing import Any


@dataclass
class MemoryState:
    requirements: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    checkpoint: int = 0
    updated_at: float = 0.0

    def add_unique(self, field_name: str, value: str, limit: int = 80) -> None:
        value = " ".join(value.split())[:1000]
        if not value:
            return
        target = getattr(self, field_name)
        if value not in target:
            target.append(value)
        if len(target) > limit:
            del target[: len(target) - limit]


class StructuredMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def _load(self) -> MemoryState:
        if not self.path.exists():
            return MemoryState()
        try:
            return MemoryState(**json.loads(self.path.read_text()))
        except Exception:
            return MemoryState()

    def save(self) -> None:
        self.state.updated_at = time.time()
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self.state), indent=2) + "\n")
        temporary.replace(self.path)

    def ingest(self, item: dict[str, Any]) -> None:
        role = item.get("role", "")
        content = str(item.get("content", ""))
        metadata = item.get("metadata", {}) or {}
        if role in {"system", "developer", "user"}:
            for sentence in self._sentences(content):
                lowered = sentence.lower()
                if role in {"system", "developer"} or any(word in lowered for word in ("must", "do not", "never", "require", "constraint")):
                    self.state.add_unique("constraints", sentence)
                elif role == "user":
                    self.state.add_unique("requirements", sentence)
        if metadata.get("decision"):
            self.state.add_unique("decisions", str(metadata["decision"]))
        if metadata.get("task"):
            self.state.add_unique("tasks", str(metadata["task"]))
        if metadata.get("command"):
            command = str(metadata["command"])
            outcome = str(metadata.get("outcome", ""))
            self.state.add_unique("commands", f"{command} -> {outcome}" if outcome else command)
        if metadata.get("error") or item.get("is_error"):
            self.state.add_unique("errors", content[:1000])
        for path in metadata.get("modified_files", []):
            self.state.add_unique("modified_files", str(path))
        for symbol in metadata.get("symbols", []):
            self.state.add_unique("symbols", str(symbol))
        if metadata.get("unresolved"):
            self.state.add_unique("unresolved", str(metadata["unresolved"]))
        artifact_id = metadata.get("artifact_id")
        if artifact_id:
            self.state.add_unique("artifacts", str(artifact_id), limit=200)
        self.state.checkpoint += 1
        self.save()

    def render(self, max_chars: int = 12000) -> str:
        sections = []
        labels = [
            ("Requirements", "requirements"), ("Constraints", "constraints"),
            ("Decisions", "decisions"), ("Current tasks", "tasks"),
            ("Commands and outcomes", "commands"), ("Errors", "errors"),
            ("Modified files", "modified_files"), ("Important symbols", "symbols"),
            ("Unresolved", "unresolved"), ("Artifact references", "artifacts"),
        ]
        for label, field_name in labels:
            values = getattr(self.state, field_name)
            if values:
                sections.append(f"## {label}\n" + "\n".join(f"- {value}" for value in values))
        rendered = "# Structured session memory\n" + "\n\n".join(sections)
        return rendered[:max_chars]

    @staticmethod
    def _sentences(content: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", content) if len(part.strip()) >= 8]
