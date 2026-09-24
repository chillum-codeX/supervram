from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
import threading
from typing import Any

from .config import ContextConfig
from .manager import ContextManager


class ContextService:
    def __init__(self, repo_root: str | Path, data_root: str | Path, config_path: str | Path | None = None):
        self.repo_root = Path(repo_root)
        self.data_root = Path(data_root)
        self.config = ContextConfig.load(config_path)
        self._managers: dict[str, ContextManager] = {}
        self._lock = threading.RLock()

    def manager(self, session_id: str) -> ContextManager:
        session_id = re.sub(r"[^A-Za-z0-9_-]", "_", session_id)[:128] or "default"
        with self._lock:
            if session_id not in self._managers:
                self._managers[session_id] = ContextManager(self.repo_root, self.data_root, self.config, session_id)
            return self._managers[session_id]

    def assemble(self, session_id: str, messages: list[dict], query: str | None = None) -> dict[str, Any]:
        manager = self.manager(session_id)
        with manager._lock:
            result = manager.assemble(messages, query)
        return {
            "messages": result.messages,
            "estimated_tokens": result.estimated_tokens,
            "prompt_limit_tokens": result.prompt_limit_tokens,
            "trace": [asdict(item) for item in result.trace],
            "retrieved": result.retrieved,
        }

    def archive_tool(self, session_id: str, content: str, tool: str, metadata: dict | None = None) -> dict:
        manager = self.manager(session_id)
        with manager._lock:
            return manager.archive_tool_output(content, tool, metadata)

    def command(self, session_id: str, command: str, argument: str | None = None) -> dict:
        manager = self.manager(session_id)
        with manager._lock:
            return self._command_locked(manager, command, argument)

    @staticmethod
    def _command_locked(manager: ContextManager, command: str, argument: str | None = None) -> dict:
        if command == "pin":
            if not argument: raise ValueError("pin requires an identity")
            manager.pin(argument)
        elif command == "unpin":
            if not argument: raise ValueError("unpin requires an identity")
            manager.unpin(argument)
        elif command == "reset":
            manager.reset()
        elif command == "index":
            return manager.index_repository()
        elif command == "inspect":
            return manager.inspect(argument)
        elif command == "retrieve":
            if not argument: raise ValueError("retrieve requires an artifact id")
            return {"artifact_id": argument, "content": manager.retrieve_artifact(argument)}
        else:
            raise ValueError(f"unknown command: {command}")
        return manager.inspect()

    def close(self) -> None:
        with self._lock:
            for manager in self._managers.values():
                manager.close()
            self._managers.clear()
