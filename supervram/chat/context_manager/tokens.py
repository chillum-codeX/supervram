from __future__ import annotations

import json
import math
import urllib.request
from typing import Iterable

from .config import ModelProfile


class TokenCounter:
    def __init__(self, profile: ModelProfile):
        self.profile = profile

    def text(self, value: str) -> int:
        if not value:
            return 0
        if self.profile.tokenizer_endpoint:
            try:
                body = json.dumps({"content": value}).encode()
                request = urllib.request.Request(
                    self.profile.tokenizer_endpoint.rstrip("/") + "/tokenize",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    result = json.loads(response.read())
                tokens = result.get("tokens")
                if isinstance(tokens, list):
                    return len(tokens)
                if isinstance(result.get("count"), int):
                    return result["count"]
            except Exception:
                pass
        return max(1, math.ceil(len(value) / max(0.5, self.profile.chars_per_token_estimate)))

    def message(self, message: dict) -> int:
        content = message.get("content", "")
        if isinstance(content, list):
            rendered = json.dumps(content, separators=(",", ":"))
        else:
            rendered = str(content)
        protocol = {key: message[key] for key in ("tool_calls", "tool_call_id", "name", "reasoning_content") if message.get(key) is not None}
        return 4 + self.text(rendered) + self.text(json.dumps(protocol, separators=(",", ":"), ensure_ascii=False) if protocol else "")

    def messages(self, messages: Iterable[dict]) -> int:
        return 2 + sum(self.message(message) for message in messages)
