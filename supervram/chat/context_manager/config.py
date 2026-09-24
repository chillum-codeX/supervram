from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class ModelProfile:
    name: str = "default-local"
    max_context_tokens: int = 32768
    output_reserve_tokens: int = 4096
    tool_reserve_tokens: int = 1024
    system_budget_tokens: int = 4096
    recent_dialogue_budget_tokens: int = 8192
    retrieved_code_budget_tokens: int = 8192
    memory_summary_budget_tokens: int = 4096
    active_tool_budget_tokens: int = 4096
    compaction_trigger_ratio: float = 0.80
    chars_per_token_estimate: float = 3.5
    tokenizer_endpoint: str | None = None
    kv_type_k: str = "f16"
    kv_type_v: str = "f16"
    verified: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_limit_tokens(self) -> int:
        return max(1, self.max_context_tokens - self.output_reserve_tokens - self.tool_reserve_tokens)


@dataclass
class ContextConfig:
    enabled: bool = True
    model: ModelProfile = field(default_factory=ModelProfile)
    artifact_dir: str = "context_data/artifacts"
    index_db: str = "context_data/index.sqlite3"
    session_dir: str = "context_data/sessions"
    debug_dir: str = "context_data/debug"
    max_tool_output_chars: int = 120_000
    inline_tool_output_chars: int = 8_000
    active_turns: int = 8
    stale_after_turns: int = 4
    archival_after_turns: int = 12
    retrieval_chunks: int = 12
    chunk_lines: int = 80
    chunk_overlap_lines: int = 12
    summary_item_chars: int = 800
    index_excludes: list[str] = field(default_factory=lambda: [
        ".git/**", "**/__pycache__/**", "**/.pytest_cache/**", "**/node_modules/**",
        "**/build/**", "**/build-*/**", "**/*.gguf", "**/*.bin", "**/*.so", "**/*.a",
        "**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.pdf", "**/.env", "**/.env.*",
        "**/*secret*", "**/*credential*", "context_data/**",
    ])
    secret_patterns: list[str] = field(default_factory=lambda: [
        r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*['\"]?[^\s'\"]+",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    ])

    @classmethod
    def load(cls, path: str | Path | None) -> "ContextConfig":
        if path is None:
            return cls()
        raw = json.loads(Path(path).read_text())
        model_raw = raw.pop("model", {})
        return cls(model=ModelProfile(**model_raw), **raw)

    def dump(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n")
